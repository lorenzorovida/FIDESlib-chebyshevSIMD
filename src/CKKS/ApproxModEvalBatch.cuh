//
// EvalChebyshevSeriesPSBatch for FIDESlib.
//
// Port of the batched (SIMD) Paterson-Stockmeyer Chebyshev series evaluator
// found in lorenzorovida/openfhe-development-chebyshevSIMD
// (AdvancedSHECKKSRNS::EvalChebyshevSeriesPSBatch / InnerEvalChebyshevPSBatch),
// adapted to FIDESlib's GPU Ciphertext/Plaintext API.
//
// Difference with FIDESlib's evalChebyshevSeries (ApproxModEval.cu):
//   evalChebyshevSeries evaluates ONE polynomial (one set of coefficients)
//   on every slot of the ciphertext.
//   evalChebyshevSeriesPSBatch evaluates a DIFFERENT polynomial PER SLOT:
//   batchOfCoefficients[j] holds the Chebyshev coefficients to be applied on
//   slot j, and batchOfCoefficients.size() must equal the number of slots of
//   the input ciphertext. All polynomials in the batch must share the same
//   degree (i.e. every inner vector must have the same size) and are
//   evaluated on the same [lower_bound, upper_bound] range.
//
// Because the "constants" that would be plain scalars in the non-batched
// Paterson-Stockmeyer algorithm become one-value-per-slot vectors here, every
// scalar multiplication (ciphertext * weight) turns into a ciphertext-times-
// plaintext multiplication (ciphertext * per-slot-packed-plaintext). This is
// the same transformation the reference fork performs going from
// EvalLinearWSumMutable (scalars) to EvalLinearWSumMutableBatch (per-slot
// plaintexts).
//
// Unlike evalChebyshevSeries (which only ever needs scalar constants and
// therefore never touches OpenFHE), the batched variant must CKKS-encode a
// fresh per-slot plaintext vector at (essentially) every step of the
// Paterson-Stockmeyer recursion, because those per-slot weights are only
// known once the Chebyshev long-divisions have been carried out at runtime
// -- they cannot be precomputed ahead of time the way e.g. bootstrapping
// linear-transform plaintexts are. FIDESlib has no on-GPU CKKS encoder, so
// -- exactly like api/CryptoContext.cpp's LoadPlaintext does -- encoding is
// delegated to the OpenFHE CPU CryptoContext, and the resulting plaintext is
// then uploaded to the GPU via GetRawPlainText(). This is the one place
// where this core (FIDESlib::CKKS) function needs a reference to the OpenFHE
// CPU CryptoContext.
//

#ifndef GPUCKKS_APPROXMODEVALBATCH_CUH
#define GPUCKKS_APPROXMODEVALBATCH_CUH

#include "CKKS/Plaintext.cuh"
#include "CKKS/forwardDefs.cuh"
#include "openfhe-interface/RawCiphertext.cuh"
#include <cinttypes>
#include <vector>

namespace FIDESlib::CKKS {

class PlaintextCache {
  public:
	bool recording = true;

	void record(Plaintext&& pt) {
		entries.push_back(std::move(pt));
	}

	const Plaintext& next() {
		assert(readIdx < entries.size() &&
		  "PSBatchPrecompute exhausted: batchOfCoefficients/lower_bound/upper_bound "
		  "mismatch with the precompute, or ciphertext structurally different "
		  "(level/NoiseLevel/slots) from the one used to build the precompute.");
		return entries[readIdx++];
	}

	void resetReadCursor() {
		readIdx = 0;
	}

	size_t size() const {
		return entries.size();
	}

	std::vector<Plaintext> entries;

  private:
	size_t readIdx = 0;
};

class PSCache {
  public:
	bool recording = true;

	void recordQr(std::shared_ptr<lbcrypto::longDiv<double>>&& qr) {
		entriesdivqrVec.push_back(std::move(qr));
	}

  void recordCs(std::shared_ptr<lbcrypto::longDiv<double>>&& cs) {
		entriesdivcsVec.push_back(std::move(cs));
	}

  void recordS2(std::vector<double>&& s2) {
		s2vec.push_back(std::move(s2));
	}

	const std::shared_ptr<lbcrypto::longDiv<double>>& nextQr() {
		assert(readIdxQr < entriesdivqrVec.size() &&
		  "PSBatchPrecompute exhausted: batchOfCoefficients/lower_bound/upper_bound "
		  "mismatch with the precompute, or ciphertext structurally different "
		  "(level/NoiseLevel/slots) from the one used to build the precompute.");
		return entriesdivqrVec[readIdxQr++];
	}

  const std::shared_ptr<lbcrypto::longDiv<double>>& nextCs() {
		assert(readIdxCs < entriesdivcsVec.size() &&
		  "PSBatchPrecompute exhausted: batchOfCoefficients/lower_bound/upper_bound "
		  "mismatch with the precompute, or ciphertext structurally different "
		  "(level/NoiseLevel/slots) from the one used to build the precompute.");
		return entriesdivcsVec[readIdxCs++];
	}

  const std::vector<double>& nextS2() {
		assert(readIdxS2 < s2vec.size() &&
		  "PSBatchPrecompute exhausted: batchOfCoefficients/lower_bound/upper_bound "
		  "mismatch with the precompute, or ciphertext structurally different "
		  "(level/NoiseLevel/slots) from the one used to build the precompute.");
		return s2vec[readIdxS2++];
	}

	void resetReadCursor() {
		readIdxQr = 0;
    readIdxCs = 0;
    readIdxS2 = 0;
	}

	std::vector<std::shared_ptr<lbcrypto::longDiv<double>>> entriesdivqrVec;
  std::vector<std::shared_ptr<lbcrypto::longDiv<double>>> entriesdivcsVec;
  std::vector<std::vector<double>> s2vec;

  private:
	size_t readIdxQr = 0;
  size_t readIdxCs = 0;
  size_t readIdxS2 = 0;
};

/**
 * @brief Evaluates a batch of Chebyshev series, one polynomial per slot, using
 *        the Paterson-Stockmeyer algorithm.
 *
 * `batchOfCoefficients[j]` contains the Chebyshev coefficients (as produced
 * e.g. by `get_chebyshev_coefficients`) of the polynomial to be applied to
 * slot `j` of `ctxt`. Every inner vector must have the same size (i.e. all
 * polynomials in the batch must share the same degree) and
 * `batchOfCoefficients.size()` must equal `ctxt.slots`.
 *
 * The ciphertext is modified in place, mirroring `evalChebyshevSeries`.
 *
 * @param cc   OpenFHE CPU CryptoContext matching `ctxt`. Used exclusively to
 *             CKKS-encode the per-slot plaintext weight vectors generated
 *             while walking the Paterson-Stockmeyer recursion (see note
 *             above): FIDESlib has no on-GPU CKKS encoder, so encoding is
 *             delegated to OpenFHE and the result is uploaded to the GPU.
 * @param ctxt Ciphertext to transform in place. `ctxt.slots` must equal
 *             `batchOfCoefficients.size()`.
 * @param batchOfCoefficients Per-slot Chebyshev coefficient vectors. All
 *             inner vectors must have equal size.
 * @param lower_bound Lower bound `a` of the approximation interval.
 * @param upper_bound Upper bound `b` of the approximation interval.
 */
void evalChebyshevSeriesPSBatch(lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& cc,
  Ciphertext& ctxt,
  const std::vector<std::vector<double>>& batchOfCoefficients,
  double lower_bound = -1.0,
  double upper_bound = 1.0);

/**
 * @brief Like evalChebyshevSeriesPSBatch, but takes a SMALL set of `x`
 *        Chebyshev-coefficient sets and repeats them cyclically to fill all
 *        `ctxt.slots` slots, instead of requiring one full set per slot.
 *
 * Slot `j` is evaluated with `coefficientSets[j % coefficientSets.size()]`.
 * This is convenient when the same small handful of polynomials needs to be
 * applied repeatedly across a ciphertext (e.g. `n / x` copies of `x`
 * distinct functions), without the caller having to manually replicate the
 * coefficient vector `n` times.
 *
 * `coefficientSets.size()` need not divide `ctxt.slots` evenly; slots are
 * filled by cycling through `coefficientSets` in order, wrapping around as
 * needed, until all `ctxt.slots` slots are covered.
 *
 * @param cc   OpenFHE CPU CryptoContext matching `ctxt` (see
 *             evalChebyshevSeriesPSBatch for why this is needed).
 * @param ctxt Ciphertext to transform in place.
 * @param coefficientSets The small set of `x` Chebyshev coefficient vectors
 *             to cycle through. Must be non-empty, and every inner vector
 *             must have the same size (same degree).
 * @param lower_bound Lower bound `a` of the approximation interval.
 * @param upper_bound Upper bound `b` of the approximation interval.
 */
void evalChebyshevSeriesPSBatchRepeated(lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& cc,
  Ciphertext& ctxt,
  const std::vector<std::vector<double>>& coefficientSets,
  double lower_bound = -1.0,
  double upper_bound = 1.0);

/**
 * @brief Precomputed per-slot plaintext weights for evalChebyshevSeriesPSBatch,
 *        allowing the same batch of Chebyshev coefficients to be evaluated
 *        against many ciphertexts without re-encoding the per-slot weight
 *        plaintexts every time.
 *
 * evalChebyshevSeriesPSBatch CKKS-encodes a fresh per-slot plaintext vector
 * at (essentially) every step of the Paterson-Stockmeyer recursion, because
 * -- unlike scalar constants -- these can't be folded into the ciphertext
 * arithmetic directly (see ApproxModEvalBatch.cuh's top-of-file note).
 * Each such encode is a CPU-side CKKS encoding (IFFT + scaling) followed by
 * a host-to-device upload -- expensive, and, critically, IDENTICAL every
 * time the same batchOfCoefficients/lower_bound/upper_bound is evaluated on
 * a ciphertext with the same level/rescale-technique-implied structure
 * (the plaintext VALUES depend only on the coefficients, which are fixed;
 * the levels they're encoded at depend only on k, m and rescaleTechnique,
 * which are also fixed for a given batchOfCoefficients/degree).
 *
 * This struct is an opaque, ordered recording of every plaintext generated
 * by one real run of evalChebyshevSeriesPSBatch (produced by
 * evalChebyshevSeriesPSBatchPrecompute). evalChebyshevSeriesPSBatchApply
 * replays that exact same sequence of plaintexts on a NEW ciphertext,
 * skipping every MakeCKKSPackedPlaintext/GetRawPlainText call.
 *
 * IMPORTANT: a given PSBatchPrecompute is only valid for ciphertexts that
 * are structurally identical to the one it was generated from: same
 * cc.L / rescaleTechnique / initial level and NoiseLevel, same slots count.
 * Reusing it on a ciphertext with a different starting level (e.g. because
 * the caller consumed a different number of levels before calling) will
 * silently replay plaintexts encoded at the wrong level -- exactly the
 * level-mismatch bug class this codebase has already hit once (see
 * makePerSlotPlaintext's doc comment in ApproxModEvalBatch.cu). Only reuse
 * a PSBatchPrecompute across calls where every input ciphertext starts at
 * the same level and NoiseLevel as the one used to build it.
 */
struct PSBatchPrecompute;
struct PSBatchPrecomputeInner;

/**
 * @brief Runs the same recursion evalChebyshevSeriesPSBatch would, on a
 *        (disposable) COPY of `ctxt`, recording every per-slot plaintext
 *        weight it generates into the returned PSBatchPrecompute -- without
 *        actually consuming or mutating `ctxt` itself.
 *
 * `ctxt` is used only as a template: to determine slots, starting level and
 * NoiseLevel, and cc.L/rescaleTechnique. Its value/contents don't matter
 * (a copy is made internally and discarded).
 *
 * @param cc   OpenFHE CPU CryptoContext matching `ctxt`.
 * @param ctxt Template ciphertext (level/NoiseLevel/slots), not modified.
 * @param batchOfCoefficients Same as evalChebyshevSeriesPSBatch.
 * @param lower_bound Same as evalChebyshevSeriesPSBatch.
 * @param upper_bound Same as evalChebyshevSeriesPSBatch.
 * @return Opaque precomputed plaintext sequence, to be passed to
 *         evalChebyshevSeriesPSBatchApply.
 */
std::shared_ptr<PSBatchPrecompute> evalChebyshevSeriesPSBatchPrecompute(lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& cc,
  const Ciphertext& ctxt,
  const std::vector<std::vector<double>>& batchOfCoefficients,
  double lower_bound = -1.0,
  double upper_bound = 1.0);

std::shared_ptr<PSBatchPrecompute> evalChebyshevSeriesPSBatchPrecompute2(lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& cc,
  const Ciphertext& ctxt,
  const std::vector<std::vector<double>>& batchOfCoefficients,
  double lower_bound = -1.0,
  double upper_bound = 1.0, PSBatchPrecompute precomp = nullptr_t);

/**
 * @brief Same computation as evalChebyshevSeriesPSBatch, but reads every
 *        per-slot plaintext weight from `precomp` (as produced by
 *        evalChebyshevSeriesPSBatchPrecompute) instead of CKKS-encoding it
 *        on the fly -- skipping all CPU-side MakeCKKSPackedPlaintext /
 *        GetRawPlainText calls that otherwise dominate runtime.
 *
 * `precomp` must have been built from a ciphertext structurally identical
 * to `ctxt` (see PSBatchPrecompute's doc comment) and with the SAME
 * batchOfCoefficients/lower_bound/upper_bound used here -- this is not
 * re-validated at runtime beyond a slot-count and plaintext-count check;
 * passing a mismatched precompute silently produces wrong results.
 *
 * `precomp` is read-only and can be reused across multiple calls (e.g. to
 * evaluate the same batch of functions on many different ciphertexts).
 * NOT thread-safe: concurrent evalChebyshevSeriesPSBatchApply calls sharing
 * the same PSBatchPrecompute will race on its internal read cursor. Use a
 * separate PSBatchPrecompute per thread, or serialize calls.
 *
 * Takes `precomp` by shared_ptr (rather than by reference) deliberately:
 * PSBatchPrecompute is an incomplete type at this header's scope (only
 * forward-declared, defined in ApproxModEvalBatch.cu), so callers outside
 * that translation unit (e.g. api/CryptoContext.cpp) can only pass the
 * shared_ptr through opaquely -- they cannot dereference it themselves.
 *
 * @param cc      OpenFHE CPU CryptoContext matching `ctxt` (still required:
 *                only the per-slot weight plaintexts are cached, not the
 *                CryptoContext itself).
 * @param ctxt    Ciphertext to transform in place.
 * @param precomp Precomputed plaintexts from evalChebyshevSeriesPSBatchPrecompute.
 * @param batchOfCoefficients Same batch used to build `precomp` (needed to
 *                drive the Paterson-Stockmeyer control flow itself -- only
 *                the coefficient VALUES are skipped via the cache, not the
 *                long-division structure, which is cheap CPU-only work).
 * @param lower_bound Same as used to build `precomp`.
 * @param upper_bound Same as used to build `precomp`.
 */
void evalChebyshevSeriesPSBatchApply(lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& cc,
  Ciphertext& ctxt,
  const std::shared_ptr<PSBatchPrecompute>& precomp,
  const std::shared_ptr<PSBatchPrecomputeInner& precompInner,
  const std::vector<std::vector<double>>& batchOfCoefficients,
  double lower_bound = -1.0,
  double upper_bound = 1.0);

/**
 * @brief Type-erasing wrapper around evalChebyshevSeriesPSBatchApply, for
 *        callers (e.g. api/CryptoContext.cpp) that only hold a
 *        std::shared_ptr<void> (PSBatchPrecompute is an incomplete type at
 *        their scope, so std::static_pointer_cast<PSBatchPrecompute> on it
 *        is unsafe/non-portable there). Defined in ApproxModEvalBatch.cu,
 *        where PSBatchPrecompute is complete, so the cast happens safely.
 */
void evalChebyshevSeriesPSBatchApplyOpaque(lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& cc,
  Ciphertext& ctxt,
  const std::shared_ptr<void>& precomp,
  const std::vector<std::vector<double>>& batchOfCoefficients,
  double lower_bound = -1.0,
  double upper_bound = 1.0);

} // namespace FIDESlib::CKKS

#endif // GPUCKKS_APPROXMODEVALBATCH_CUH
