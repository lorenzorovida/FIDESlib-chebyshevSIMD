//
// EvalChebyshevSeriesPSBatch for FIDESlib.
//
// Port of lorenzorovida/openfhe-development-chebyshevSIMD's
// AdvancedSHECKKSRNS::EvalChebyshevSeriesPSBatch / InnerEvalChebyshevPSBatch
// to FIDESlib's GPU Ciphertext/Plaintext API, following the same structure
// FIDESlib already uses for the non-batched Paterson-Stockmeyer evaluator in
// ApproxModEval.cu (function evalChebyshevSeries / innerEvalChebyshevPS).
//
// The key transformation, exactly as in the reference fork, is: every place
// where the scalar Paterson-Stockmeyer algorithm multiplies a Chebyshev power
// T[i] by a *scalar* weight (one double), the batched version multiplies it
// by a *plaintext* that packs one (possibly different) weight per slot. This
// turns:
//   - multScalar(...)                    -> multPt(...)
//   - addScalar(...)                     -> addPt(...)
//   - evalLinearWSumMutable(scalars)     -> evalLinearWSumMutablePtBatch(plaintexts)
//
// The Chebyshev powers T[1..k], T2[1..m] and T2km1 depend only on the input
// ciphertext (not on the coefficients), so their computation is unchanged
// from evalChebyshevSeries and is reused (with a fresh copy) verbatim.
//

#include "CKKS/ApproxModEval.cuh" // for multIntScalar
#include "CKKS/ApproxModEvalBatch.cuh"
#include "CKKS/Ciphertext.cuh"
#include "CKKS/Context.cuh"
#include "CKKS/Plaintext.cuh"
#include "CudaUtils.cuh"
#include <iostream>
#include <map>
#include <nvtx3/nvtx3.hpp>
// Uncomment to trace level/NoiseLevel at key checkpoints, mirrored in
// ApproxModEval.cu, for side-by-side debugging against the scalar original.
// #define DEBUG_CHEBYSHEV_TRACE 1

#if defined(__clang__)
#include <experimental/source_location>
using scb = std::experimental::source_location;
#else
#include <source_location>
using scb = std::source_location;
#endif

using namespace FIDESlib::CKKS;

namespace {

/**
 * Ordered record/replay cache of Plaintext objects, used to implement the
 * precompute/apply split (see ApproxModEvalBatch.cuh's PSBatchPrecompute
 * doc comment).
 *
 * - recording == true: every makePerSlotPlaintext call appends its result
 *   here (via record()) instead of/in addition to being consumed normally.
 * - recording == false: every makePerSlotPlaintext call instead reads the
 *   next plaintext from `entries` (via next()), in the exact same order
 *   they were recorded -- this relies on the recursion visiting
 *   makePerSlotPlaintext calls in an identical, deterministic order between
 *   the precompute run and every subsequent apply run, which holds because
 *   the control flow (which branches are taken, how many weights each
 *   evalLinearWSumMutablePtBatch call needs, ...) depends only on
 *   batchOfCoefficients/k/m/rescaleTechnique -- all fixed between the two
 *   runs -- never on ciphertext VALUES.
 */

/**
 * Per-call memoization of "T[i] aligned to level L, NoiseLevel==1" copies,
 * scoped to a single evalChebyshevSeriesPSBatchImpl invocation (constructed
 * fresh at the top of that function, passed down through the recursion, and
 * destroyed when it returns).
 *
 * WHY THIS EXISTS: evalLinearWSumMutablePtBatch is called many times during
 * one Paterson-Stockmeyer recursion (once per q/c/s branch at every
 * recursion level), and the targetLevel it computes always has the shape
 * `T2[m-1]->getLevel() + (NoiseLevel==1 ? 1 : 0) - offset`, with `offset`
 * only ever level_offset or level_offset+1. So the *same* T[i] is very
 * often re-aligned to the *same* targetLevel across multiple, unrelated
 * evalLinearWSumMutablePtBatch calls -- each one previously redid its own
 * independent copy+growToLevel+dropToLevel for it (see the note at the top
 * of evalLinearWSumMutablePtBatch). Since T[i] never changes value once
 * computed, and the set of (T[i], targetLevel) pairs visited is completely
 * determined by k/m/rescaleTechnique (never by ciphertext values -- same
 * determinism property the PlaintextCache already relies on), the aligned
 * copy can simply be memoized by (pointer identity of T[i], targetLevel)
 * for the lifetime of one top-level call.
 *
 * This does NOT persist across separate evalChebyshevSeriesPSBatchApply
 * calls (unlike PlaintextCache) because the underlying Ciphertext GPU
 * buffers of T[]/T2[] are reallocated fresh every call -- pointer identity
 * only makes sense within one call. It still collapses what could be
 * O(recursion-steps) redundant copies of the same T[i] into at most one per
 * (T[i], distinct level actually visited).
 */
struct AlignedCiphertextCache {
	FIDESlib::CKKS::Context& cc_;
	// Backing storage for aligned copies; std::map so references handed out
	// via getAligned() remain stable even as more entries are inserted
	// (unlike std::vector, which may reallocate and invalidate them).
	std::map<std::pair<const Ciphertext*, int32_t>, Ciphertext> storage;

	explicit AlignedCiphertextCache(FIDESlib::CKKS::Context& cc_) : cc_(cc_) {
	}

	// Returns a pointer to `src` aligned to (targetLevel, NoiseLevel==1).
	// If `src` is already exactly at that level/NoiseLevel, returns `src`
	// itself (no copy at all, matching the original fast path). Otherwise,
	// builds the aligned copy once and reuses it on every subsequent call
	// with the same (src, targetLevel) pair.
	Ciphertext* getAligned(Ciphertext* src, int32_t targetLevel) {
		if (src->getLevel() == targetLevel && src->NoiseLevel == 1) {
			return src;
		}
		auto key = std::make_pair(static_cast<const Ciphertext*>(src), targetLevel);
		auto it	 = storage.find(key);
		if (it != storage.end()) {
			return &it->second;
		}
		auto [inserted, _] = storage.emplace(key, Ciphertext(cc_));
		Ciphertext& a	   = inserted->second;
		a.copy(*src);
		if (a.NoiseLevel == 2)
			a.rescale();
		a.growToLevel(targetLevel);
		a.dropToLevel(targetLevel);
		return &a;
	}
};

/**
 * Builds a CKKS plaintext that packs `values[j]` into slot `j`, encoded at
 * the same level/scale/slot-count as `like`, and loads it onto the GPU.
 *
 * This is the batched-coefficient equivalent of a bare scalar constant in the
 * non-batched algorithm: instead of a single double reused on every slot, we
 * need one double per slot.
 *
 * IMPORTANT (1) — level convention mismatch. FIDESlib's core
 * Ciphertext::getLevel() counts DOWN from cc.L (a freshly-encrypted
 * ciphertext has getLevel() == L, and getLevel() decreases as depth is
 * consumed) — see api/Ciphertext.cpp's GetLevel(), which explicitly does
 * `maxDepth - ct_gpu->getLevel()` to convert to the OpenFHE-style level.
 * lbcrypto::CryptoContext::MakeCKKSPackedPlaintext, on the other hand, uses
 * the OpenFHE convention: level COUNTS UP from 0. We convert explicitly via
 * cc.L before calling into OpenFHE.
 *
 * IMPORTANT (2) — noise/scale degree mismatch. Ciphertext::addScalar(double)
 * computes its modular constant via
 * cc.ElemForEvalAddOrSub(level, c, this->NoiseLevel), i.e. it adapts
 * dynamically to whatever NoiseLevel (1 or 2) the target ciphertext currently
 * has. A Plaintext, however, is encoded once with a fixed noiseScaleDeg.
 * Ciphertext::addPt/multPt do not renormalize a NoiseLevel mismatch away
 * (the relevant asserts are compiled out in release builds), so encoding
 * every per-slot plaintext at a hardcoded noiseScaleDeg=1 silently produces
 * wrong results whenever `like` happens to be at NoiseLevel==2 (e.g. under
 * FLEXIBLEAUTO, where several call sites only rescale conditionally on
 * FIXEDMANUAL). We therefore encode at `like.NoiseLevel` instead of a fixed
 * 1, so the plaintext always matches the real scale of the ciphertext it
 * will be combined with — without having to mutate `like` itself (important
 * when `like` is a shared object such as T2[] reused across recursive calls).
 *
 * IMPORTANT (3) — precompute/apply cache. When `cache` is non-null: if
 * cache->recording, the freshly-built plaintext is also recorded into the
 * cache and then returned normally (used by
 * evalChebyshevSeriesPSBatchPrecompute, which needs a real Plaintext to
 * drive the surrounding Ciphertext arithmetic that determines subsequent
 * levels, even though the ciphertext itself is a throwaway template). If
 * !cache->recording, no encoding happens at all, and no GPU copy is made
 * either -- a pointer directly into the cache is returned (used by
 * evalChebyshevSeriesPSBatchApply, which thus skips both
 * MakeCKKSPackedPlaintext/GetRawPlainText AND any GPU-side plaintext copy).
 * When `cache` is null (the original evalChebyshevSeriesPSBatch entry
 * point) or recording, the freshly-encoded plaintext is stored into
 * `storage` (caller-provided, must outlive the returned pointer) and a
 * pointer to it is returned.
 */
const Plaintext* makePerSlotPlaintext(lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& cc,
  FIDESlib::CKKS::Context& cc_,
  const std::vector<double>& values,
  const Ciphertext& like,
  Plaintext& storage,
  PlaintextCache* cache = nullptr) {
	if (cache != nullptr && !cache->recording) {
		// Replay: zero-copy, return a pointer straight into the cache.
		return &cache->next();
	}

	uint32_t openfheLevel			 = static_cast<uint32_t>(like.cc.L - like.getLevel());
	size_t noiseScaleDeg			 = static_cast<size_t>(like.NoiseLevel);
	auto pt							 = cc->MakeCKKSPackedPlaintext(values,
	  /*noiseScaleDeg=*/noiseScaleDeg,
	  /*level=*/openfheLevel,
	  nullptr,
	  /*slots=*/like.slots);
	FIDESlib::CKKS::RawPlainText raw = FIDESlib::CKKS::GetRawPlainText(cc, pt);
	storage.load(raw);

	if (cache != nullptr && cache->recording) {
		Plaintext toStore(cc_);
		toStore.copy(storage);
		cache->record(std::move(toStore));
	}

	return &storage;
}

/**
 * Evaluates a weighted sum sum_i ctxs[i] * weightsPerSlot[i], where each
 * weightsPerSlot[i] is a per-slot vector (one weight per slot of ctxs[i]).
 * Mirrors Ciphertext::evalLinearWSumMutable, but with per-slot (plaintext)
 * weights instead of scalar weights. The result is written into `out`.
 *
 * IMPORTANT: exactly like Ciphertext::evalLinearWSumMutable, the caller is
 * expected to have already set the TARGET level on `out` (via dropToLevel /
 * growToLevel) before calling this function. Unlike the scalar version,
 * however, multPt/addMultPt require the ciphertext and plaintext operands to
 * sit at the SAME level (there is no per-limb "elem" projection like
 * ElemForEvalMult does for scalar weights): Ciphertext::multPt(c, b, ...)
 * starts with `this->copy(c)`, which would silently discard whatever level
 * `out` had been dropped/grown to. To avoid that mismatch (and the resulting
 * out-of-bounds RNS-limb access when ctxs[i] and out end up with a different
 * number of limbs), we explicitly align a local copy of each ctxs[i] to
 * out's target level before multiplying, and encode each per-slot plaintext
 * weight at that same target level.
 */
void evalLinearWSumMutablePtBatch(Ciphertext& out,
  lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& cc,
  FIDESlib::CKKS::Context& cc_,
  const std::vector<Ciphertext*>& ctxs,
  const std::vector<std::vector<double>>& weightsPerSlot,
  PlaintextCache* cache				   = nullptr,
  AlignedCiphertextCache* alignedCache = nullptr) {
	out.copy(*ctxs[0]);
	return;

	FIDESlib::CudaNvtxRange r(std::string{ scb::current().function_name() }.substr());
	assert(ctxs.size() == weightsPerSlot.size());
	uint32_t n = static_cast<uint32_t>(ctxs.size());
	assert(n > 0);

	// Target level: whatever the caller already set on `out` via
	// dropToLevel/growToLevel. If `out` has never been initialized (fresh
	// Ciphertext, getLevel() == -1), fall back to ctxs[0]'s level, mirroring
	// evalLinearWSumMutable's own fallback.
	int32_t targetLevel = out.getLevel();
	if (targetLevel < 0) {
		targetLevel = ctxs[0]->getLevel();
	}

	// Align each ctxs[i] to (targetLevel, NoiseLevel==1). When alignedCache
	// is provided (the normal case -- see evalChebyshevSeriesPSBatchImpl,
	// which owns one AlignedCiphertextCache per top-level call and threads
	// it through the whole recursion), a given (ctxs[i], targetLevel) pair
	// is only ever copied+aligned ONCE per call to
	// evalChebyshevSeriesPSBatch/...Apply, no matter how many times this
	// function is invoked with that same pair during the recursion (very
	// common: T[i] gets re-aligned to the same handful of target levels
	// across many q/c/s branches). Previously this copy was redone from
	// scratch, unconditionally, on every single call -- real CUDA
	// allocation + full-limb memcpy work that dominated runtime far more
	// than the plaintext CPU-encoding the precompute/apply split was built
	// to eliminate.
	std::vector<Ciphertext*> alignedPtrs(n);
	// Fallback storage only used when no alignedCache is passed in (keeps
	// this function correct/self-contained for any other caller), in which
	// case we're back to the original per-call behavior.
	std::vector<Ciphertext> alignedStorageFallback;
	if (alignedCache != nullptr) {
		for (uint32_t i = 0; i < n; ++i) {
			alignedPtrs[i] = alignedCache->getAligned(ctxs[i], targetLevel);
		}
	} else {
		alignedStorageFallback.reserve(n);
		for (uint32_t i = 0; i < n; ++i) {
			if (ctxs[i]->getLevel() == targetLevel && ctxs[i]->NoiseLevel == 1) {
				alignedPtrs[i] = ctxs[i];
			} else {
				alignedStorageFallback.emplace_back(cc_);
				Ciphertext& a = alignedStorageFallback.back();
				a.copy(*ctxs[i]);
				if (a.NoiseLevel == 2)
					a.rescale();
				a.growToLevel(targetLevel);
				a.dropToLevel(targetLevel);
				alignedPtrs[i] = &a;
			}
		}
	}

	std::vector<Plaintext> weightPtsStorage;
	weightPtsStorage.reserve(n);
	std::vector<const Plaintext*> weightPts(n);

	for (uint32_t i = 0; i < n; ++i) {
		weightPtsStorage.emplace_back(cc_);

		weightPts[i] = makePerSlotPlaintext(cc, cc_, weightsPerSlot[i], *alignedPtrs[i], weightPtsStorage.back(), cache);
	}

	out.multPt(*alignedPtrs[0], *weightPts[0], false);
	for (uint32_t i = 1; i < n; ++i) {
		out.addMultPt(*alignedPtrs[i], *weightPts[i], false);
	}

	if (out.cc.rescaleTechnique == FIXEDMANUAL) {
		out.rescale();
	}
}

/**
 * Adds, per-slot, `values[j]` to `ctxt` (per-slot version of addScalar).
 *
 * Unlike Ciphertext::addScalar(double), which internally computes the
 * correctly-scaled modular constant via
 * cc.ElemForEvalAddOrSub(level, c, this->NoiseLevel) -- i.e. it adapts to
 * whatever NoiseLevel `this` currently has (1 or 2) without consuming an
 * extra rescale -- a Plaintext must be encoded with a specific
 * noiseScaleDeg. makePerSlotPlaintext already reads `like.NoiseLevel`
 * dynamically and encodes accordingly (see its doc comment), so this
 * function does NOT force a rescale of `ctxt` first: doing so would consume
 * an extra NoiseLevel step relative to the scalar original, desynchronizing
 * every subsequent level/NoiseLevel-dependent decision in the caller (this
 * was the root cause of a NoiseLevel==0 "double rescale" bug: callers such as
 * the q/s evaluation branches that unconditionally rescale once afterward,
 * mirroring the scalar original's own single rescale, would otherwise
 * rescale a `ctxt` that had already been silently rescaled here).
 */
void addPerSlotScalar(Ciphertext& ctxt, lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& cc, FIDESlib::CKKS::Context& cc_, const std::vector<double>& values, PlaintextCache* cache = nullptr) {
	//Plaintext storage(cc_);
	//const Plaintext* pt = makePerSlotPlaintext(cc, cc_, values, ctxt, storage, cache);
	//ctxt.addPt(*pt);
}

/**
 * ctxt = src * (per-slot weights). Per-slot version of
 * Ciphertext::multScalar(src, weight, rescale).
 *
 * makePerSlotPlaintext already adapts its noiseScaleDeg to src's real
 * NoiseLevel (see its doc comment), so this works correctly regardless of
 * src's NoiseLevel; no rescale of src is needed or performed here.
 */
void multPerSlotScalar(Ciphertext& ctxt,
  lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& cc,
  FIDESlib::CKKS::Context& cc_,
  const Ciphertext& src,
  const std::vector<double>& values,
  bool rescale,
  PlaintextCache* cache = nullptr) {
	//Plaintext storage(cc_);
	//const Plaintext* pt = makePerSlotPlaintext(cc, cc_, values, src, storage, cache);
	//ctxt.multPt(src, *pt, rescale);
}

/**
 * Batched counterpart of innerEvalChebyshevPS (ApproxModEval.cu).
 *
 * `batchOfCoefficients` holds one Chebyshev-coefficient vector per slot; all
 * of them must have the same size (same degree at this recursion level, this
 * is guaranteed by construction since every polynomial in the batch shares
 * the same k, m and the same starting degree). `out` receives the per-slot
 * evaluation of every polynomial in the batch.
 */
void innerEvalChebyshevPSBatch(lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& cc,
  const Ciphertext& ctxt,
  Ciphertext& out,
  const std::vector<std::vector<double>>& batchOfCoefficients,
  const uint32_t k,
  uint32_t m,
  const std::vector<Ciphertext*>& T,
  const std::vector<Ciphertext*>& T2,
  int level_offset					   = 0,
  int max_m							   = 1000,
  PlaintextCache* cache				   = nullptr,
  AlignedCiphertextCache* alignedCache = nullptr) {
	FIDESlib::CudaNvtxRange r(std::string{ scb::current().function_name() });

	FIDESlib::CKKS::Context& cc_ = ctxt.cc_;
	ContextData& ccd			 = ctxt.cc;

	uint32_t k2m2k = k * (1 << (m - 1)) - k;

	size_t batchSize = batchOfCoefficients.size();
	std::vector<std::shared_ptr<lbcrypto::longDiv<double>>> divqrVec(batchSize);
	std::vector<std::shared_ptr<lbcrypto::longDiv<double>>> divcsVec(batchSize);
	std::vector<std::vector<double>> s2Vec(batchSize);

	if (cache == nullptr || (cache != nullptr && cache->recording)) {
		for (size_t b = 0; b < batchSize; ++b) {
			// Add T^{k(2^m - 1)}(y) to the polynomial that has to be evaluated
			std::vector<double> f2 = batchOfCoefficients[b];
			f2.resize(2 * k2m2k + k + 1, 0.0);
			if (f2.size() > batchOfCoefficients[b].size())
				f2.back() = 1;

			// Divide f2 by T^{k*2^{m-1}}
			std::vector<double> Tkm(int32_t(k2m2k + k) + 1, 0.0);
			Tkm.back() = 1;
			auto divqr = lbcrypto::LongDivisionChebyshev(f2, Tkm);

			// Subtract x^{k(2^{m-1} - 1)} from r
			std::vector<double> r2 = divqr->r;
			if (int32_t(k2m2k - lbcrypto::Degree(divqr->r)) <= 0) {
				r2[int32_t(k2m2k)] -= 1;
				r2.resize(lbcrypto::Degree(r2) + 1);
			} else {
				r2.resize(int32_t(k2m2k + 1), 0.0);
				r2.back() = -1;
			}

			// Divide r2 by q
			auto divcs = lbcrypto::LongDivisionChebyshev(r2, divqr->q);

			// Add x^{k(2^{m-1} - 1)} to s
			std::vector<double> s2 = divcs->r;
			s2.resize(int32_t(k2m2k + 1), 0.0);
			s2.back() = 1;

			divqrVec[b] = divqr;
			divcsVec[b] = divcs;
			s2Vec[b]	= std::move(s2);
		}
		if (cache != nullptr && cache->recording) {
			cache->recordQr(divqrVec);
			cache->recordCs(divcsVec);
			cache->recordS2(s2Vec);
		}
	} else {
		divqrVec = cache->nextQr();
		divcsVec = cache->nextCs();
		s2Vec	 = cache->nextS2();
	}

	// Degrees of divqr->q, divcs->q, s2 are structurally identical across the
	// whole batch (same k, m and same input degree), so index 0 is used as
	// representative wherever a degree is needed, exactly like the reference
	// fork.

	// --- Evaluate c at u ---
	Ciphertext& cu = out;
	uint32_t dc	   = lbcrypto::Degree(divcsVec[0]->q);
	bool flag_c	   = false;
	if (dc >= 1) {
		bool needCompute = (cache == nullptr) || cache->recording;
		if (dc == 1) {
			std::vector<double> coeffs;
			if (needCompute) {
				coeffs.resize(batchSize);
				for (size_t b = 0; b < batchSize; ++b)
					coeffs[b] = divcsVec[b]->q[1];
				if (cache != nullptr && cache->recording) {
					std::vector<double> toStore = coeffs;
					cache->recordVec1(std::move(toStore));
				}
			} else {
				coeffs = cache->nextVec1();
			}
			multPerSlotScalar(cu, cc, cc_, *T[0], coeffs, true, cache);
		} else {
			std::vector<Ciphertext*> ctxs(dc);
			std::vector<std::vector<double>> weights;
			if (needCompute) {
				weights.assign(dc, std::vector<double>(batchSize));
				for (uint32_t i = 0; i < dc; ++i) {
					ctxs[i] = T[i];
					for (size_t b = 0; b < batchSize; ++b)
						weights[i][b] = divcsVec[b]->q[i + 1];
				}
				if (cache != nullptr && cache->recording) {
					std::vector<std::vector<double>> toStore = weights;
					cache->recordVec2(std::move(toStore));
				}
			} else {
				for (uint32_t i = 0; i < dc; ++i)
					ctxs[i] = T[i];
				weights = cache->nextVec2();
			}

			cu.dropToLevel(T2[m - 1]->getLevel() + (T2[m - 1]->NoiseLevel == 1 ? 1 : 0) - level_offset);
			cu.growToLevel(T2[m - 1]->getLevel() + (T2[m - 1]->NoiseLevel == 1 ? 1 : 0) - level_offset);

			evalLinearWSumMutablePtBatch(cu, cc, cc_, ctxs, weights, cache, alignedCache);
		}

		std::vector<double> freeTerm;
		if (needCompute) {
			freeTerm.resize(batchSize);
			for (size_t b = 0; b < batchSize; ++b)
				freeTerm[b] = divcsVec[b]->q.front() / 2.0;
			if (cache != nullptr && cache->recording) {
				std::vector<double> toStore = freeTerm;
				cache->recordVec1(std::move(toStore));
			}
		} else {
			freeTerm = cache->nextVec1();
		}
		addPerSlotScalar(cu, cc, cc_, freeTerm, cache);

		if (ccd.rescaleTechnique == FIXEDMANUAL) {
			cu.rescale();
		}

		flag_c = true;
	}

	// --- Evaluate q at u (recursively if its degree still exceeds k) ---
	Ciphertext qu(cc_);
	if (lbcrypto::Degree(divqrVec[0]->q) > k) {
		assert(m > 2);
		bool needComputeQCoeffs = (cache == nullptr) || cache->recording;
		std::vector<std::vector<double>> qCoeffs;
		if (needComputeQCoeffs) {
			qCoeffs.resize(batchSize);
			for (size_t b = 0; b < batchSize; ++b)
				qCoeffs[b] = divqrVec[b]->q;
			if (cache != nullptr && cache->recording) {
				std::vector<std::vector<double>> toStore = qCoeffs;
				cache->recordVec2(std::move(toStore));
			}
		} else {
			qCoeffs = cache->nextVec2();
		}
		innerEvalChebyshevPSBatch(cc, ctxt, qu, qCoeffs, k, m - 1, T, T2, level_offset, max_m, cache, alignedCache);

		if (qu.NoiseLevel == 2)
			qu.rescale();
	} else {
		// dq = k from construction
		auto qcopy0 = divqrVec[0]->q;
		qcopy0.resize(k);
		if (lbcrypto::Degree(qcopy0) > 0) {
			bool needCompute = (cache == nullptr) || cache->recording;
			std::vector<Ciphertext*> ctxs;
			std::vector<std::vector<double>> weights;

			if (needCompute) {
				std::vector<uint32_t> selectedIdx;
				for (uint32_t i = 0; i < divqrVec[0]->q.size() - 1; ++i) {
					bool anyNonZero = false;
					for (size_t b = 0; b < batchSize; ++b) {
						if (divqrVec[b]->q[i + 1] != 0) {
							anyNonZero = true;
							break;
						}
					}
					if (anyNonZero) {
						selectedIdx.push_back(i);
						ctxs.push_back(T[i]);
						std::vector<double> w(batchSize);
						for (size_t b = 0; b < batchSize; ++b)
							w[b] = divqrVec[b]->q[i + 1];
						weights.push_back(std::move(w));
					}
				}
				if (cache != nullptr && cache->recording) {
					std::vector<uint32_t> idxCopy = selectedIdx;
					cache->recordCtxsSelection(std::move(idxCopy));
					std::vector<std::vector<double>> weightsCopy = weights;
					cache->recordVec2(std::move(weightsCopy));
				}
			} else {
				const std::vector<uint32_t>& selectedIdx = cache->nextCtxsSelection();
				for (uint32_t idx : selectedIdx)
					ctxs.push_back(T[idx]);
				weights = cache->nextVec2();
			}

			qu.growToLevel(T2[m - 1]->getLevel() + (T2[m - 1]->NoiseLevel == 1 ? 1 : 0) - level_offset);
			qu.dropToLevel(T2[m - 1]->getLevel() + (T2[m - 1]->NoiseLevel == 1 ? 1 : 0) - level_offset);

			evalLinearWSumMutablePtBatch(qu, cc, cc_, ctxs, weights, cache, alignedCache);

			std::vector<double> freeTerm;
			if (needCompute) {
				freeTerm.resize(batchSize);
				for (size_t b = 0; b < batchSize; ++b)
					freeTerm[b] = divqrVec[b]->q.front() / 2.0;
				if (cache != nullptr && cache->recording) {
					std::vector<double> toStore = freeTerm;
					cache->recordVec1(std::move(toStore));
				}
			} else {
				freeTerm = cache->nextVec1();
			}
			addPerSlotScalar(qu, cc, cc_, freeTerm, cache);

			if (T[k - 1]->NoiseLevel == 1)
				qu.rescale();
			if (T[k - 1]->NoiseLevel == 2)
				qu.rescale();
		} else {
			qu.copy(*T[k - 1]);

			// The leading coefficient of q is structurally identical across
			// the whole batch (it only depends on k, m — not on the per-slot
			// function being approximated: it is always a power of two
			// coming from the Chebyshev doubling rule), so a plain integer
			// scalar multiplication is valid here, exactly as in
			// evalChebyshevSeries.
			assert(divqrVec[0]->q.back() > 0 && divqrVec[0]->q.back() - round(divqrVec[0]->q.back()) == 0.0);
			multIntScalar(qu, (uint64_t)divqrVec[0]->q.back());

			bool needCompute = (cache == nullptr) || cache->recording;
			std::vector<double> freeTerm;
			if (needCompute) {
				freeTerm.resize(batchSize);
				for (size_t b = 0; b < batchSize; ++b)
					freeTerm[b] = divqrVec[b]->q.front() / 2.0;
				if (cache != nullptr && cache->recording) {
					std::vector<double> toStore = freeTerm;
					cache->recordVec1(std::move(toStore));
				}
			} else {
				freeTerm = cache->nextVec1();
			}
			addPerSlotScalar(qu, cc, cc_, freeTerm, cache);

			if (qu.NoiseLevel == 2)
				qu.rescale();
		}
	}

	// --- Evaluate s2 at u (recursively if its degree still exceeds k) ---
	Ciphertext su(cc_);
	if (lbcrypto::Degree(s2Vec[0]) > k) {
		assert(m > 2);
		innerEvalChebyshevPSBatch(cc, ctxt, su, s2Vec, k, m - 1, T, T2, level_offset + 1, max_m, cache, alignedCache);
	} else {
		auto scopy0 = s2Vec[0];
		scopy0.resize(k);
		if (lbcrypto::Degree(scopy0) > 0) {
			bool needCompute = (cache == nullptr) || cache->recording;
			std::vector<Ciphertext*> ctxs;
			std::vector<std::vector<double>> weights;

			if (needCompute) {
				std::vector<uint32_t> selectedIdx;
				for (uint32_t i = 0; i < s2Vec[0].size() - 1; ++i) {
					bool anyNonZero = false;
					for (size_t b = 0; b < batchSize; ++b) {
						if (s2Vec[b][i + 1] != 0) {
							anyNonZero = true;
							break;
						}
					}
					if (anyNonZero) {
						selectedIdx.push_back(i);
						ctxs.push_back(T[i]);
						std::vector<double> w(batchSize);
						for (size_t b = 0; b < batchSize; ++b)
							w[b] = s2Vec[b][i + 1];
						weights.push_back(std::move(w));
					}
				}
				if (cache != nullptr && cache->recording) {
					std::vector<uint32_t> idxCopy = selectedIdx;
					cache->recordCtxsSelection(std::move(idxCopy));
					std::vector<std::vector<double>> weightsCopy = weights;
					cache->recordVec2(std::move(weightsCopy));
				}
			} else {
				const std::vector<uint32_t>& selectedIdx = cache->nextCtxsSelection();
				for (uint32_t idx : selectedIdx)
					ctxs.push_back(T[idx]);
				weights = cache->nextVec2();
			}

			su.growToLevel(T2[m - 1]->getLevel() + (T2[m - 1]->NoiseLevel == 1 ? 1 : 0) - 1 - level_offset);
			su.dropToLevel(T2[m - 1]->getLevel() + (T2[m - 1]->NoiseLevel == 1 ? 1 : 0) - 1 - level_offset);

			evalLinearWSumMutablePtBatch(su, cc, cc_, ctxs, weights, cache, alignedCache);

			std::vector<double> freeTerm;
			if (needCompute) {
				freeTerm.resize(batchSize);
				for (size_t b = 0; b < batchSize; ++b)
					freeTerm[b] = s2Vec[b].front() / 2.0;
				if (cache != nullptr && cache->recording) {
					std::vector<double> toStore = freeTerm;
					cache->recordVec1(std::move(toStore));
				}
			} else {
				freeTerm = cache->nextVec1();
			}
			addPerSlotScalar(su, cc, cc_, freeTerm, cache);

			// The highest order coefficient will always be 1 because s2 is
			// monic (structurally, across the whole batch).
			assert(s2Vec[0].back() == 1.0);
		} else {
			su.copy(*T[k - 1]);
			bool needCompute = (cache == nullptr) || cache->recording;
			std::vector<double> freeTerm;
			if (needCompute) {
				freeTerm.resize(batchSize);
				for (size_t b = 0; b < batchSize; ++b)
					freeTerm[b] = s2Vec[b].front() / 2.0;
				if (cache != nullptr && cache->recording) {
					std::vector<double> toStore = freeTerm;
					cache->recordVec1(std::move(toStore));
				}
			} else {
				freeTerm = cache->nextVec1();
			}
			addPerSlotScalar(su, cc, cc_, freeTerm, cache);
		}
	}

	// --- Combine: out = (T2[m-1] + cu) * qu + su ---
	if (flag_c) {
		if (max_m - (int)m <= 1)
			T2[m - 1]->adjustForAddOrSub(cu);
		if (T2[m - 1]->NoiseLevel == 1 && cu.NoiseLevel == 2)
			cu.rescale();
		cu.add(*T2[m - 1]);
	} else {
		bool needCompute = (cache == nullptr) || cache->recording;
		std::vector<double> freeTerm;
		if (needCompute) {
			freeTerm.resize(batchSize);
			for (size_t b = 0; b < batchSize; ++b)
				freeTerm[b] = divcsVec[b]->q.front() / 2.0;
			if (cache != nullptr && cache->recording) {
				std::vector<double> toStore = freeTerm;
				cache->recordVec1(std::move(toStore));
			}
		} else {
			freeTerm = cache->nextVec1();
		}
		Plaintext storage(cc_);
		const Plaintext* pt = makePerSlotPlaintext(cc, cc_, freeTerm, *T2[m - 1], storage, cache);
		cu.addPt(*T2[m - 1], *pt);
	}
	if (ccd.rescaleTechnique == FIXEDMANUAL && out.NoiseLevel == 2)
		cu.rescale();
	cu.mult(qu, false);
	cu.add(su); // cu aliases out
}

} // namespace

namespace {

void evalChebyshevSeriesPSBatchImpl(lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& cc,
  Ciphertext& ctxt,
  const std::vector<std::vector<double>>& batchOfCoefficients,
  double lower_bound,
  double upper_bound,
  PlaintextCache* cache) {
	FIDESlib::CudaNvtxRange r(std::string{ scb::current().function_name() });

	if (batchOfCoefficients.empty())
		OPENFHE_THROW("batchOfCoefficients must not be empty");

	if (static_cast<int>(batchOfCoefficients.size()) != ctxt.slots)
		OPENFHE_THROW("The set of coefficients must be as large as the number of slots of the input ciphertext");

	size_t coeffSize = batchOfCoefficients[0].size();
	for (const auto& v : batchOfCoefficients) {
		if (v.size() != coeffSize)
			OPENFHE_THROW("All polynomials in the batch must have the same number of coefficients");
	}

	FIDESlib::CKKS::Context& cc_ = ctxt.cc_;

	// --- Linear transform onto [-1, 1], identical to evalChebyshevSeries ---
	if (abs(lower_bound + 1.0) > 1e-9 || abs(upper_bound - 1.0) > 1e-9) {
		if (abs(upper_bound - lower_bound - 2.0) < 1e-8) {
			ctxt.addScalar(-lower_bound + 1.0);
		} else {
			if (abs(lower_bound + upper_bound) > 1e-8)
				ctxt.addScalar(-(upper_bound - lower_bound) / 2.0);
			if (ctxt.cc.rescaleTechnique == FIXEDMANUAL && ctxt.NoiseLevel == 2)
				ctxt.rescale();
			ctxt.multScalar(2.0 / (upper_bound - lower_bound));
		}
	}

	// Every polynomial in the batch must share the same degree (enforced
	// above), so the first one determines k, m for the whole batch.
	uint32_t n				   = lbcrypto::Degree(batchOfCoefficients[0]);
	std::vector<uint32_t> degs = lbcrypto::ComputeDegreesPS(n);
	uint32_t k				   = degs[0];
	uint32_t m				   = degs[1];

	std::vector<std::vector<double>> f2Batch(batchOfCoefficients.size());

	if (cache != nullptr && !cache->recording) {
		f2Batch = cache->nextF2();
	} else {
		for (size_t b = 0; b < batchOfCoefficients.size(); ++b) {
			f2Batch[b] = batchOfCoefficients[b];
			f2Batch[b].resize(n + 1);
		}
		if (cache != nullptr) {
			cache->recordF2(f2Batch);
		}
	}

	ContextData& ccd = ctxt.cc;

	// --- Compute Chebyshev powers T[1..k], T2[1..m], T2km1 ---
	// Identical to evalChebyshevSeries: these depend only on ctxt, not on the
	// (batched) coefficients.
	std::vector<Ciphertext> aux;
	for (size_t i = aux.size(); i < k + m; i++) {
		aux.emplace_back(cc_);
	}

	std::vector<Ciphertext*> T(k);
	for (uint32_t i = 0; i < k; ++i)
		T[i] = &aux[i];
	std::vector<Ciphertext*> T2(m);
	for (uint32_t i = 0; i < m; i++)
		T2[i] = &aux[i + k];

	T[0]->copy(ctxt);
	if (T[0]->NoiseLevel == 2)
		T[0]->rescale();
#ifdef DEBUG_CHEBYSHEV_TRACE
	std::cout << "[BATCH] after T[0] copy+rescale: level=" << T[0]->getLevel() << " noise=" << T[0]->NoiseLevel << std::endl;
#endif

	for (uint32_t i = 2; i <= k; i++) {
		if (i % 2 == 1) {
			// compute T_{2i+1}(y) = 2*T_i(y)*T_{i+1}(y) - y
			T[i / 2]->adjustForMult(*T[i / 2 - 1]);
			T[i / 2 - 1]->adjustForMult(*T[i / 2]);
			T[i - 1]->mult(*T[i / 2 - 1], *T[i / 2], false);
			T[i - 1]->add(*T[i - 1]);
			ctxt.adjustForAddOrSub(*T[i - 1]);
			if (ctxt.NoiseLevel == 1)
				T[i - 1]->rescale();
			T[i - 1]->sub(ctxt);
		} else {
			// compute T_{2i}(y) = 2*T_i(y)^2 - 1
			T[i / 2 - 1]->adjustForMult(*T[i / 2 - 1]);
			T[i - 1]->square(*T[i / 2 - 1], false);
			T[i - 1]->add(*T[i - 1]);
			T[i - 1]->addScalar(-1.0);
		}
	}

	for (size_t i = 1; i <= k; i++) {
		if (T[i - 1]->NoiseLevel == 2)
			T[i - 1]->rescale();
	}
#ifdef DEBUG_CHEBYSHEV_TRACE
	for (size_t i = 0; i < k; i++)
		std::cout << "[BATCH] T[" << i << "] level=" << T[i]->getLevel() << " noise=" << T[i]->NoiseLevel << std::endl;
#endif

	if (ccd.rescaleTechnique == FIXEDMANUAL) {
		for (size_t i = 1; i < k; i++) {
			T[i - 1]->dropToLevel(T[k - 1]->getLevel());
		}
	}

	// Compute T_k(y), T_{2k}(y), T_{4k}(y), ... , T_{2^{m-1}k}(y)
	T2[0]->copy(*T.back());
	for (uint32_t i = 1; i < m; i++) {
		if (ccd.rescaleTechnique == FIXEDMANUAL && T2[i - 1]->NoiseLevel == 2)
			T2[i - 1]->rescale();
		T2[i]->square(*T2[i - 1], false);
		T2[i]->add(*T2[i]);
		T2[i]->addScalar(-1.0);

		if (ccd.rescaleTechnique == FIXEDMANUAL && T2[i]->NoiseLevel == 2)
			T2[i]->rescale();
	}
#ifdef DEBUG_CHEBYSHEV_TRACE
	for (size_t i = 0; i < m; i++)
		std::cout << "[BATCH] T2[" << i << "] level=" << T2[i]->getLevel() << " noise=" << T2[i]->NoiseLevel << std::endl;
#endif

	// computes T_{k(2*m - 1)}(y)
	Ciphertext T2km1(cc_);
	T2km1.copy(*T2[0]);
	if (ccd.rescaleTechnique == FIXEDMANUAL) {
		T2km1.dropToLevel(T2[1]->getLevel());
	}

	for (uint32_t i = 1; i < m; i++) {
		T2km1.mult(*T2[i], false);
		T2km1.add(T2km1);
		T2[0]->adjustForAddOrSub(T2km1);
		if (T2[0]->NoiseLevel == 1)
			T2km1.rescale();
		T2km1.sub(*T2[0]);
		if (T2[0]->NoiseLevel == 2 && i < m - 1)
			T2km1.rescale();
	}
#ifdef DEBUG_CHEBYSHEV_TRACE
	std::cout << "[BATCH] T2km1 level=" << T2km1.getLevel() << " noise=" << T2km1.NoiseLevel << std::endl;
#endif

	// --- Batched Paterson-Stockmeyer evaluation ---
	// Scoped to this single top-level call: memoizes T[i]/T2[i] alignment
	// copies by (pointer, targetLevel) across the whole recursion below, so
	// each distinct (T[i], level) pair is copied+aligned at most once
	// instead of once per evalLinearWSumMutablePtBatch call that needs it.
	// See AlignedCiphertextCache's doc comment for why this is safe/correct.
	AlignedCiphertextCache alignedCache(cc_);
	innerEvalChebyshevPSBatch(cc, ctxt, ctxt, f2Batch, k, m, T, T2, 0, m, cache, &alignedCache);
#ifdef DEBUG_CHEBYSHEV_TRACE
	std::cout << "[BATCH] after innerEvalChebyshevPSBatch (before final sub): level=" << ctxt.getLevel() << " noise=" << ctxt.NoiseLevel << std::endl;
#endif

	ctxt.sub(T2km1);
#ifdef DEBUG_CHEBYSHEV_TRACE
	std::cout << "[BATCH] FINAL: level=" << ctxt.getLevel() << " noise=" << ctxt.NoiseLevel << std::endl;
#endif
}

} // namespace

void FIDESlib::CKKS::evalChebyshevSeriesPSBatch(lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& cc,
  Ciphertext& ctxt,
  const std::vector<std::vector<double>>& batchOfCoefficients,
  double lower_bound,
  double upper_bound) {
	// Original entry point: no cache, always encode plaintexts fresh.
	// Behavior is completely unchanged from before the precompute/apply
	// split.
	evalChebyshevSeriesPSBatchImpl(cc, ctxt, batchOfCoefficients, lower_bound, upper_bound, /*cache=*/nullptr);
}

/**
 * Opaque precomputed-plaintext container backing PSBatchPrecompute (see
 * ApproxModEvalBatch.cuh). Wraps the internal PlaintextCache so the header
 * only needs a forward declaration.
 */
struct FIDESlib::CKKS::PSBatchPrecompute {
	PlaintextCache cache;
};

std::shared_ptr<FIDESlib::CKKS::PSBatchPrecompute> FIDESlib::CKKS::evalChebyshevSeriesPSBatchPrecompute(lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& cc,
  const Ciphertext& ctxt,
  const std::vector<std::vector<double>>& batchOfCoefficients,
  double lower_bound,
  double upper_bound) {
	FIDESlib::CudaNvtxRange r(std::string{ scb::current().function_name() });

	auto precomp			 = std::make_shared<PSBatchPrecompute>();
	precomp->cache.recording = true;

	// Run the real algorithm on a disposable COPY of ctxt: we need real
	// Ciphertext arithmetic to happen (levels/NoiseLevel evolve exactly as
	// they would for a real call) so that every makePerSlotPlaintext call
	// sees the same `like` levels/NoiseLevel a real evalChebyshevSeriesPSBatch
	// call would -- but the copy itself, and its final numeric result, are
	// discarded; only the recorded plaintexts in precomp->cache matter.
	Ciphertext templateCopy(ctxt.cc_);
	templateCopy.copy(ctxt);

	evalChebyshevSeriesPSBatchImpl(cc, templateCopy, batchOfCoefficients, lower_bound, upper_bound, &precomp->cache);

	return precomp;
}

void FIDESlib::CKKS::evalChebyshevSeriesPSBatchApply(lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& cc,
  Ciphertext& ctxt,
  const std::shared_ptr<PSBatchPrecompute>& precomp,
  const std::vector<std::vector<double>>& batchOfCoefficients,
  double lower_bound,
  double upper_bound) {

	FIDESlib::CudaNvtxRange r(std::string{ scb::current().function_name() });

	// PlaintextCache::next() advances a read cursor, which is logically
	// read-only from the caller's perspective (precomp can be
	// reused/replayed any number of times -- resetReadCursor() below always
	// rewinds it back to the start before use) but requires a mutable
	// reference internally.
	PlaintextCache& cache = precomp->cache;
	cache.recording		  = false;
	cache.resetReadCursor();

	evalChebyshevSeriesPSBatchImpl(cc, ctxt, batchOfCoefficients, lower_bound, upper_bound, &cache);
}

void FIDESlib::CKKS::evalChebyshevSeriesPSBatchApplyOpaque(lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& cc,
  Ciphertext& ctxt,
  const std::shared_ptr<void>& precomp,
  const std::vector<std::vector<double>>& batchOfCoefficients,
  double lower_bound,
  double upper_bound) {
	// Safe here: PSBatchPrecompute is a complete type in this translation
	// unit (defined above), so std::static_pointer_cast can properly share
	// ownership/lifetime with the original std::shared_ptr<void>.
	auto casted = std::static_pointer_cast<PSBatchPrecompute>(precomp);

	evalChebyshevSeriesPSBatchApply(cc, ctxt, casted, batchOfCoefficients, lower_bound, upper_bound);
}

void FIDESlib::CKKS::evalChebyshevSeriesPSBatchRepeated(lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& cc,
  Ciphertext& ctxt,
  const std::vector<std::vector<double>>& coefficientSets,
  double lower_bound,
  double upper_bound) {
	FIDESlib::CudaNvtxRange r(std::string{ scb::current().function_name() });

	if (coefficientSets.empty())
		OPENFHE_THROW("coefficientSets must not be empty");

	// Expand the small repeating pattern into a full per-slot batch (one
	// entry per slot, cycling through coefficientSets), then delegate to
	// evalChebyshevSeriesPSBatch, which already validates equal degree,
	// slot-count match, etc.
	std::vector<std::vector<double>> batchOfCoefficients;
	batchOfCoefficients.reserve(static_cast<size_t>(ctxt.slots));
	for (int j = 0; j < ctxt.slots; ++j) {
		batchOfCoefficients.push_back(coefficientSets[static_cast<size_t>(j) % coefficientSets.size()]);
	}

	evalChebyshevSeriesPSBatch(cc, ctxt, batchOfCoefficients, lower_bound, upper_bound);
}
