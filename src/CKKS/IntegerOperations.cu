#include "CKKS/Ciphertext.cuh"
#include "CKKS/IntegerOperations.cuh"

#include <cstdint>
#include <functional>
#include <memory>
#include <stdexcept>
#include <unordered_map>

namespace FIDESlib::CKKS {

ProcessArrayPrecomputation precomp8;
ProcessArrayPrecomputation precomp16;
ProcessArrayPrecomputation precomp32;
ProcessArrayPrecomputation precomp64;
ProcessArrayPrecomputation precomp128;

ProcessArrayPrecomputation precomp8b;
ProcessArrayPrecomputation precomp16b;
ProcessArrayPrecomputation precomp32b;
ProcessArrayPrecomputation precomp64b;
ProcessArrayPrecomputation precomp128b;

std::shared_ptr<PSBatchPrecompute> cacheChebyshev4BitsMultiplier;
std::vector<std::vector<double>> coeffs4BitsMultiplier;
DivIntegerLUTs lutsDiv;
SquareRootIntegerLUTs lutsSquareRoot;

// ============================================================
// evalIntegerMult recombination masks (mask1 / mask2)
//
// CPU reference (CKKSController::mul_integer) rebuilds these two
// plain std::vector<double> masks with simple loops on *every* call.
// That's cheap on CPU because building the vector doesn't encode
// anything -- encoding only happens lazily inside mult(Ctxt,
// vector<double>) -> encode(...).
//
// On GPU the expensive part is exactly that encoding step:
// makePerSlotPlaintext() runs MakeCKKSPackedPlaintext(...) +
// GetRawPlainText(...) on every single call. The *shape* of mask1/
// mask2 depends only on (bits, repetitions, slots) -- never on the
// ciphertext contents. BUT the encoded Plaintext also bakes in the
// level and noiseScaleDeg of the ciphertext it was built against
// (makePerSlotPlaintext reads both off `result`), and `result`'s
// level is NOT constant across calls: evalIntegerMult bootstraps
// (binboot / BootstrapStCFirstBits) at the end of every single call,
// including every recursive call at every bits level, so the level
// seen here varies depending on how deep/which path the recursion
// took. A previous version of this cache fixed `level` to a constant
// (12) ahead of time to mirror ProcessArrayPrecomputations, which
// silently produced wrong results whenever `result`'s real level
// didn't match -- multiplying a ciphertext by a plaintext encoded at
// the wrong level/rescale point corrupts the CKKS scale.
//
// So the cache key here is (bits, repetitions, level, noiseScaleDeg)
// rather than just `bits`: this still avoids re-encoding on repeat
// calls that land at the same recursion level with the same noise
// state (the common case, since the same bits_original always drives
// the same sequence of levels/noise), while never handing back a
// Plaintext encoded for the wrong level. Plaintext has no default
// constructor (see ProcessArrayPrecomputation::Entry, which is only
// ever move-constructed), so entries are held as unique_ptr instead
// of by value to keep this a plain global map.
// ============================================================
struct IntegerMultMaskPrecomputation {
	std::unique_ptr<Plaintext> mask1;
	std::unique_ptr<Plaintext> mask2;
};

struct IntegerMultMaskKey {
	int bits;
	int repetitions;
	int slots;
	uint32_t openfheLevel;
	size_t noiseScaleDeg;

	bool operator==(const IntegerMultMaskKey& other) const {
		return bits == other.bits && repetitions == other.repetitions && slots == other.slots && openfheLevel == other.openfheLevel && noiseScaleDeg == other.noiseScaleDeg;
	}
};

struct IntegerMultMaskKeyHash {
	size_t operator()(const IntegerMultMaskKey& k) const {
		size_t h = std::hash<int>()(k.bits);
		h		 = h * 31 + std::hash<int>()(k.repetitions);
		h		 = h * 31 + std::hash<int>()(k.slots);
		h		 = h * 31 + std::hash<uint32_t>()(k.openfheLevel);
		h		 = h * 31 + std::hash<size_t>()(k.noiseScaleDeg);
		return h;
	}
};

std::unordered_map<IntegerMultMaskKey, IntegerMultMaskPrecomputation, IntegerMultMaskKeyHash> integerMultMaskCache;

Plaintext makePerSlotPlaintext(lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& cc, FIDESlib::CKKS::Context& cc_, const std::vector<double>& values, const Ciphertext& like) {
	uint32_t openfheLevel			 = static_cast<uint32_t>(like.cc.L - like.getLevel());
	size_t noiseScaleDeg			 = static_cast<size_t>(like.NoiseLevel);
	auto pt							 = cc->MakeCKKSPackedPlaintext(values,
	  /*noiseScaleDeg=*/noiseScaleDeg,
	  /*level=*/openfheLevel,
	  nullptr,
	  /*slots=*/like.slots);
	FIDESlib::CKKS::RawPlainText raw = FIDESlib::CKKS::GetRawPlainText(cc, pt);
	return Plaintext(cc_, raw);
}

// ============================================================
// binboot
//
// CPU reference (CKKSController::binboot):
//
//   Ctxt CKKSController::binboot(const Ctxt &c) {
//       return context->EvalBootstrapStCFirstBits(c);
//   }
//
// FIDESlib doesn't expose EvalBootstrapStCFirstBits directly; every call
// site in this file instead drops to the level the "first bits" bootstrap
// circuit expects (5 if NoiseLevel==2, else 4) and then calls
// BootstrapStCFirstBits(...). This wraps that exact pattern so div_integer
// (which calls binboot(...) a dozen times) doesn't have to repeat it.
// ============================================================
void binboot(Ciphertext& out, const Ciphertext& c) {
	if (&out != &c) {
		out.copy(c);
	}

	if (out.NoiseLevel == 2) {
		out.dropToLevel(5, false);
	} else {
		out.dropToLevel(4, false);
	}

	BootstrapStCFirstBits(out, out.slots, false);
}

// ============================================================
// binary_or
//
// CPU reference (CKKSController::binary_or):
//
//   Ctxt CKKSController::binary_or(const Ctxt &a, const Ctxt &b) {
//       // a+b - a*b
//       return sub(add(a, b), mult(a, b));
//   }
// ============================================================
void binaryOr(Ciphertext& out, const Ciphertext& a, const Ciphertext& b) {
	Ciphertext prod(a.cc_);
	prod.mult(a, b, true);

	out.add(a, b);
	out.sub(prod);
}

// ============================================================
// Repeated-Chebyshev-LUT (mirrors OpenFHE's
// EvalChebyshevSeriesPSBatchRepeated(ctxt, coeffs, a, b, repeat)). We reuse
// FIDESlib's existing PS-batch machinery (evalChebyshevSeriesPSBatchPrecompute
// / evalChebyshevSeriesPSBatchApply, as used by preprocessChebyshevMultiplication
// / multiplier4bits above), but unlike OpenFHE's PSBatch, FIDESlib's
// evalChebyshevSeriesPSBatchImpl requires the coefficient vector to already
// be as large as the ciphertext's slot count -- it does NOT tile a shorter
// "base" set internally. So the tiling that OpenFHE's `repeat` parameter
// does implicitly, we have to do explicitly here: replicate the caller's
// base column set `repeat = c.slots / coeffs.size()` times before handing
// it to PSBatch. Callers (e.g. preprocessDivIntegerLUTs) keep passing just
// the base column set, same as the CPU's un-tiled `coeffs` vector.
// ============================================================
void preprocessChebyshevRepeated(ChebyshevRepeatedLUT& lut, lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& cc, Ciphertext& c, std::vector<std::vector<double>> coeffs, int a, int b) {

	if (coeffs.empty()) {
		throw std::invalid_argument("preprocessChebyshevRepeated: coeffs must not be empty");
	}
	if (c.slots % coeffs.size() != 0) {
		throw std::invalid_argument("preprocessChebyshevRepeated: c.slots must be a multiple of coeffs.size()");
	}

	const size_t repeat = c.slots / coeffs.size();

	std::vector<std::vector<double>> tiled;
	tiled.reserve(coeffs.size() * repeat);
	for (size_t r = 0; r < repeat; ++r) {
		tiled.insert(tiled.end(), coeffs.begin(), coeffs.end());
	}

	lut.coeffs	= coeffs; // keep the caller's original (un-tiled) base set for reference
	lut.repeat	= static_cast<int>(repeat);
	lut.a		= a;
	lut.b		= b;
	lut.modelLevel		= c.getLevel();
	lut.modelNoiseLevel = c.NoiseLevel;
	lut.precomp			= evalChebyshevSeriesPSBatchPrecompute(cc, c, tiled, a, b);
}

void evalChebyshevRepeatedApply(lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& cc, Ciphertext& c, const ChebyshevRepeatedLUT& lut) {

	if (!lut.precomp) {
		throw std::invalid_argument("evalChebyshevRepeatedApply: LUT not precomputed, call preprocessChebyshevRepeated first");
	}

	std::vector<std::vector<double>> tiled;
	tiled.reserve(lut.coeffs.size() * lut.repeat);
	for (int r = 0; r < lut.repeat; ++r) {
		tiled.insert(tiled.end(), lut.coeffs.begin(), lut.coeffs.end());
	}

	// Use the (a, b) this LUT was actually precomputed with -- previously
	// hardcoded to (-1, 1) here regardless of the LUT, which silently gave
	// the reciprocal-hint LUT (built with a=0, b=256) the wrong input-range
	// rescaling at apply time.
	evalChebyshevSeriesPSBatchApply(cc, c, lut.precomp, tiled, lut.a, lut.b);
}

// ============================================================
// inverse_bit_length
//
// CPU reference (CKKSController::inverse_bit_length):
//
//   int step = 1;
//   Ctxt result = a->Clone();
//   while (step < bits) {
//       result = binary_or(result, rot(result, step));
//       step *= 2;
//   }
//   result = binboot(result);
//   for (int i = 0; i < log2(bits); i++)
//       result = add(result, rot(result, -pow(2, i)));
//   // result now holds bit_length(a) in the last (partial) slot of each group
//   vector<double> mask(result->GetSlots());
//   int stride = bits * bits / 2;
//   for (int i = 0; i < zslots; i++)
//       mask[(bits - 1) + i * stride] = -1.0 / (bits / 2.0);
//   result = mult(result, mask);
//   for (int i = 0; i < zslots; i++)
//       mask[(bits - 1) + i * stride] = 1.0;
//   result = add(result, encode(mask, result->GetLevel()));
//   Ctxt resultclone = result->Clone();
//   result = add(result, rot(result, 1));
//   result = add(result, rot(result, 2));
//   result = add(result, rot(result, 4));
//   result = sub(result, rot(resultclone, 7));
//   result = rot(result, bits - 7);
//   return result;
// ============================================================
void inverseBitLength(Ciphertext& out, const Ciphertext& a, int bits, int zslots, lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& cc) {

	FIDESlib::CKKS::Context& cc_ = a.cc_;

	// --------------------------------------------------------
	// result = OR-reduce a with all rot(a, step), step = 1,2,4,...
	// so every bit slot becomes 1 iff any bit at or below it is set
	// (i.e. a "sticky OR" prefix scan up to `bits`).
	// --------------------------------------------------------
	Ciphertext result(a.cc_);
	result.copy(a);

	for (int step = 1; step < bits; step *= 2) {
		Ciphertext rotated(a.cc_);
		rotated.rotate(result, step);

		Ciphertext orred(a.cc_);
		binaryOr(orred, result, rotated);

		result.copy(orred);
	}

	binboot(result, result);

	// --------------------------------------------------------
	// Sum all the ones -> popcount of the OR-mask == bit_length(a),
	// landing in the last (partial) slot of each `stride`-sized group.
	// --------------------------------------------------------
	const int rounds = static_cast<int>(std::log2(bits));

	for (int i = 0; i < rounds; ++i) {
		const int rotation = -(1 << i);

		Ciphertext rotated(a.cc_);
		rotated.rotate(result, rotation);

		result.add(rotated);
	}

	// --------------------------------------------------------
	// result[(bits-1) + i*stride] holds bit_length(a) for group i.
	// Rescale it into [-1, 1] as: -bit_length(a) / (bits/2) + 1
	// --------------------------------------------------------
	const int stride = bits * bits / 2;

	std::vector<double> maskNeg(a.slots, 0.0);
	for (int i = 0; i < zslots; ++i) {
		maskNeg[(bits - 1) + i * stride] = -1.0 / (bits / 2.0);
	}

	result.multPt(makePerSlotPlaintext(cc, cc_, maskNeg, result));

	std::vector<double> maskOne(a.slots, 0.0);
	for (int i = 0; i < zslots; ++i) {
		maskOne[(bits - 1) + i * stride] = 1.0;
	}

	result.addPt(makePerSlotPlaintext(cc, cc_, maskOne, result));

	// --------------------------------------------------------
	// Broadcast that single normalized value to slots (bits-1)+{0,1,2,4}
	// via: result += rot(result,1) + rot(result,2) + rot(result,4)
	//               - rot(resultclone, 7)
	// then shift the whole group left by (bits - 7) so the LUT input
	// lands where the caller expects it (slot 0 onward per group).
	// --------------------------------------------------------
	Ciphertext resultClone(a.cc_);
	resultClone.copy(result);

	Ciphertext r1(a.cc_);
	r1.rotate(result, 1);
	result.add(r1);

	Ciphertext r2(a.cc_);
	r2.rotate(result, 2);
	result.add(r2);

	Ciphertext r4(a.cc_);
	r4.rotate(result, 4);
	result.add(r4);

	Ciphertext r7(a.cc_);
	r7.rotate(resultClone, 7);
	result.sub(r7);

	{
		Ciphertext rotated(a.cc_);
		rotated.rotate(result, bits - 7);
		result.copy(rotated);
	}

	out.copy(result);
}

// ============================================================
// blind_rotation
//
// Takes a `bits`-wide integer `a` and a 7-bit (LSB-to-the-right) binary
// index, and homomorphically rotates `a` right by that many positions,
// via 7 conditional (mux) rotations by powers of two.
//
// CPU reference (CKKSController::blind_rotation):
//
//   Ctxt result = a->Clone();
//   if (stride == 0) stride = bits * bits / 2;
//   for (int i = 0; i < 7; i++) {
//       vector<double> mask(a->GetSlots());
//       for (int j = 0; j < zslots; j++) mask[i + j * stride] = 1;
//       Ctxt current_index = mult(index, encode(mask, index->GetLevel()));
//       current_index = rot(current_index, i);
//       for (int j = 0; j < log2(bits); j++)
//           current_index = add(current_index, rot(current_index, -pow(2, j)));
//       // If condition (mux): result = result*(1-idx) + rot(result,-2^i)*idx
//       result = add(mult(result, sub(1, current_index)), mult(rot(result, -pow(2, i)), current_index));
//   }
//   return result;
// ============================================================
void blindRotation(Ciphertext& out, const Ciphertext& a, const Ciphertext& index, int bits, int zslots, int stride, lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& cc) {

	FIDESlib::CKKS::Context& cc_ = a.cc_;

	if (stride == 0) {
		stride = bits * bits / 2;
	}

	const int rounds = static_cast<int>(std::log2(bits));

	Ciphertext result(a.cc_);
	result.copy(a);

	for (int i = 0; i < 7; ++i) {

		// --------------------------------------------------------
		// Extract bit `i` of `index` (per group) and broadcast it
		// across the group so it can be used as a per-slot select mask.
		// --------------------------------------------------------
		std::vector<double> mask(a.slots, 0.0);
		for (int j = 0; j < zslots; ++j) {
			mask[i + j * stride] = 1.0;
		}

		Ciphertext currentIndex(a.cc_);
		currentIndex.multPt(index, makePerSlotPlaintext(cc, cc_, mask, index));
		{
			Ciphertext rotated(a.cc_);
			rotated.rotate(currentIndex, i);
			currentIndex.copy(rotated);
		}

		for (int j = 0; j < rounds; ++j) {
			Ciphertext rotated(a.cc_);
			rotated.rotate(currentIndex, -(1 << j));
			currentIndex.add(rotated);
		}

		// --------------------------------------------------------
		// Mux: result = result * (1 - idx) + rot(result, -2^i) * idx
		// --------------------------------------------------------
		Ciphertext oneMinusIdx(a.cc_);
		std::vector<double> ones(a.slots, 1.0);
		oneMinusIdx.multScalar(currentIndex, -1.0, true);
		oneMinusIdx.addPt(makePerSlotPlaintext(cc, cc_, ones, oneMinusIdx));

		Ciphertext keep(a.cc_);
		keep.mult(result, oneMinusIdx, true);

		Ciphertext shifted(a.cc_);
		shifted.rotate(result, -(1 << i));

		Ciphertext take(a.cc_);
		take.mult(shifted, currentIndex, true);

		result.add(keep, take);
	}

	out.copy(result);
}

void evalIntegerAdd(Ciphertext& ctxtA, Ciphertext& ctxtB, int bits) {
	if (bits <= 0) {
		throw std::invalid_argument("evalIntegerAdd: bits must be > 0");
	}

	/*
	 * Original CPU:
	 *
	 * p = square(sub(a, b));
	 */

	Ciphertext p(ctxtA.cc_);
	p.copy(ctxtA);
	p.sub(ctxtB);
	p.square();

	/*
	 * absum = p->Clone()
	 */

	Ciphertext absum(ctxtA.cc_);
	absum.copy(p);

	/*
	 * g = mult(a, b);
	 */

	Ciphertext g(ctxtA.cc_);
	g.copy(ctxtA);
	g.mult(ctxtB);

	/*
	 * for (int i = 1; i < bits; i *= 2)
	 */
	for (int i = 1; i < bits; i *= 2) {

		/*
		 * p_shift = rot(p, -i)
		 */
		Ciphertext p_shift(ctxtA.cc_);
		p_shift.rotate(p, -i);

		/*
		 * g_shift = rot(g, -i)
		 */
		Ciphertext g_shift(ctxtA.cc_);
		g_shift.rotate(g, -i);

		/*
		 * pg = mult(p, g_shift)
		 */
		Ciphertext pg(ctxtA.cc_);
		pg.mult(p, g_shift);

		/*
		 * mult(p, g)
		 */
		Ciphertext p_g(ctxtA.cc_);
		p_g.mult(p, g);

		/*
		 * g = sub(add(g, pg), p_g)
		 */
		g.add(pg);
		g.sub(p_g);

		/*
		 * p = mult(p, p_shift)
		 */
		if (i < bits - 1) {
			p.mult(p_shift);
		}
	}

	/*
	 * g = rot(g, -1)
	 */
	g.rotate(-1);

	/*
	 * s = square(sub(absum, g))
	 *
	 * Since ctxtA is our output buffer, overwrite it.
	 */
	ctxtA.copy(absum);
	ctxtA.sub(g);
	ctxtA.square();
}

// ============================================================
// sub_integer
//
// CPU reference (CKKSController::sub_integer):
//
//   vector<double> ones(...);  // {1 at slots [0, bits], 0 elsewhere} per group
//   Ctxt inverted = context->EvalSub(encrypt(encode(ones, b->GetLevel())), b);
//   inverted = add_integer(a, inverted, bits, clean_first);
//   // mask off the low `bits` bits to drop garbage above the carry-out slot
//   return mult(inverted, mask_low_bits);
//
// i.e. a - b == a + (ones_mask - b), where ones_mask sets the low `bits+1`
// slots of each group to 1 (one extra slot beyond `bits` itself -- matching
// the CPU's `for (int j = 0; j < bits + 1; j++) ones.push_back(1)` exactly),
// then add_integer's usual two's-complement machinery does the rest; finally
// re-mask down to just the low `bits` bits to strip the carry-out garbage
// add_integer leaves behind.
// ============================================================
void evalIntegerSub(Ciphertext& out, const Ciphertext& a, const Ciphertext& b, int bits, int zslots, lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& cc, bool cleanFirst) {
	if (cleanFirst) {
		throw std::invalid_argument("evalIntegerSub: cleanFirst=true is not implemented (no caller needs it yet; "
									"evalIntegerAdd only implements the clean_first=false path)");
	}

	FIDESlib::CKKS::Context& cc_ = a.cc_;
	const int stride			 = bits * bits / 2;

	// --------------------------------------------------------
	// ones := {1 at slots [0, bits] of each group, 0 elsewhere}
	// inverted := ones - b
	// --------------------------------------------------------
	std::vector<double> ones(a.slots, 0.0);
	for (int j = 0; j < zslots; ++j) {
		for (int i = 0; i < bits + 1; ++i) {
			ones[i + j * stride] = 1.0;
		}
	}

	Ciphertext inverted(a.cc_);
	inverted.multScalar(b, -1.0, true);
	inverted.addPt(makePerSlotPlaintext(cc, cc_, ones, inverted));

	// --------------------------------------------------------
	// inverted := add_integer(a, inverted, bits)
	// --------------------------------------------------------
	Ciphertext aCopy(a.cc_);
	aCopy.copy(a);
	evalIntegerAdd(aCopy, inverted, bits);

	// --------------------------------------------------------
	// Mask down to the low `bits` bits of each group (drop add_integer's
	// carry-out garbage above the result).
	// --------------------------------------------------------
	std::vector<double> mask(a.slots, 0.0);
	for (int j = 0; j < zslots; ++j) {
		for (int i = 0; i < bits; ++i) {
			mask[i + j * stride] = 1.0;
		}
	}

	out.multPt(aCopy, makePerSlotPlaintext(cc, cc_, mask, aCopy), true);
}

void evalIntegerEqual(Ciphertext& a, Ciphertext& b, int bits, int zslots, std::vector<double> coeffsSinc, lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& cc, int depth) {
	Ciphertext sum(a.cc_);

	sum.copy(a);
	sum.sub(b);
	sum.square();

	// ------------------------------------------------------------
	// sum = sum + rotations of sum
	// ------------------------------------------------------------

	const int rounds = static_cast<int>(std::log2(bits));

	for (int i = 0; i < rounds; ++i) {

		const int rotation = 1 << i;

		Ciphertext rotated(a.cc_);
		rotated.rotate(sum, rotation);

		sum.add(rotated);
	}

	evalChebyshevSeries(sum, coeffsSinc, 0, 256);

	// ------------------------------------------------------------
	// correction mask
	// ------------------------------------------------------------

	std::vector<double> correction(a.slots, 0.0);

	for (int i = 0; i < zslots; ++i) {

		correction[i * (bits * bits) / 2] = 1.0;
	}

	// ------------------------------------------------------------
	// Convert correction to FIDESlib plaintext
	// ------------------------------------------------------------

	size_t noise = static_cast<size_t>(sum.NoiseLevel);

	auto pt = cc->MakeCKKSPackedPlaintext(correction, noise, sum.getLevel(), nullptr, sum.slots);

	FIDESlib::CKKS::RawPlainText raw = FIDESlib::CKKS::GetRawPlainText(cc, pt);

	Plaintext correctionPt(sum.cc_, raw);

	a.multPt(sum, correctionPt, false);

	if (a.NoiseLevel == 2) {
		a.dropToLevel(5, false);
	} else {
		a.dropToLevel(4, false);
	}

	BootstrapStCFirstBits(a, a.slots, false);
}

void evalIntegerMult(Ciphertext& out,
  const Ciphertext& a,
  const Ciphertext& b,
  int bits,
  int bits_original,
  int repetitions,
  int repetitions_original,
  bool overflow,
  lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& cc) {
	const int rep_size = bits * bits / 2;

	// Size of basic multiplier.
	const int base_mult = 8;

	FIDESlib::CKKS::Context& cc_ = a.cc_;

	Ciphertext result(a.cc_);

	// ============================================================
	// BASE CASE: 8-bit multiplier
	// ============================================================

	if (bits == 8) {

		// --------------------------------------------------------
		// A
		// --------------------------------------------------------

		const int mask_size = bits_original * (bits_original / 2);

		std::vector<double> masklow(a.slots, 0);

		for (int j = 0; j < repetitions_original; ++j) {
			masklow[0 + j * mask_size] = 1;
			masklow[1 + j * mask_size] = 1;
			masklow[2 + j * mask_size] = 1;
			masklow[3 + j * mask_size] = 1;
		}

		Ciphertext a_low(a.cc_);
		a_low.copy(a);

		a_low.multPt(makePerSlotPlaintext(cc, cc_, masklow, a_low));

		// --------------------------------------------------------
		// maskhigh
		// --------------------------------------------------------

		std::vector<double> maskhigh(a.slots, 0);

		for (int j = 0; j < repetitions_original; ++j) {
			maskhigh[4 + j * mask_size] = 1;
			maskhigh[5 + j * mask_size] = 1;
			maskhigh[6 + j * mask_size] = 1;
			maskhigh[7 + j * mask_size] = 1;
		}

		const int highShift = -(16 - 4);

		Ciphertext a_rot(a.cc_);
		a_rot.rotate(a, highShift);

		std::vector<double> maskhigh_rot = rotateMask(maskhigh, highShift);

		Ciphertext a_high(a.cc_);

		a_high.copy(a_rot);
		a_high.multPt(makePerSlotPlaintext(cc, cc_, maskhigh_rot, a_high));

		a_low.add(a_high);

		Ciphertext a_processed(a.cc_);
		a_processed.copy(a_low);

		// --------------------------------------------------------
		// process_array(...)
		// --------------------------------------------------------

		if (bits_original > 8) {
			processArray(a_processed, a, precomp8);
		}

		if (bits_original > 16) {
			processArray(a_processed, a, precomp16);
		}

		if (bits_original > 32) {
			processArray(a_processed, a, precomp32);
		}

		if (bits_original > 64) {
			processArray(a_processed, a, precomp64);
		}

		if (bits_original > 128) {
			processArray(a_processed, a, precomp128);
		}

		// --------------------------------------------------------
		// Combine A
		// --------------------------------------------------------

		if (bits_original > 4) {

			Ciphertext tmp(a.cc_);
			tmp.rotate(a_processed, -8);
			a_processed.add(tmp);
		}

		if (bits_original > 8) {

			Ciphertext tmp(a.cc_);
			tmp.rotate(a_processed, -32);
			a_processed.add(tmp);
		}

		if (bits_original > 16) {

			Ciphertext tmp(a.cc_);
			tmp.rotate(a_processed, -128);
			a_processed.add(tmp);
		}

		if (bits_original > 32) {

			Ciphertext tmp(a.cc_);
			tmp.rotate(a_processed, -512);
			a_processed.add(tmp);
		}

		if (bits_original > 64) {

			Ciphertext tmp(a.cc_);
			tmp.rotate(a_processed, -2048);
			a_processed.add(tmp);
		}

		if (bits_original > 128) {

			Ciphertext tmp(a.cc_);
			tmp.rotate(a_processed, -8192);
			a_processed.add(tmp);
		}

		// out.copy(a_processed);
		// return;
		// A_PROCESSED SEMBRA GIUSTO (INT)

		// ========================================================
		// B
		// ========================================================

		Ciphertext b_processed(a.cc_);
		b_processed.copy(b);

		b_processed.multPt(makePerSlotPlaintext(cc, cc_, masklow, b_processed));

		Ciphertext b_rot(a.cc_);
		b_rot.rotate(b, -4);

		std::vector<double> maskhigh_b = rotateMask(maskhigh, -4);

		Ciphertext b_high(a.cc_);
		b_high.copy(b_rot);

		b_high.multPt(makePerSlotPlaintext(cc, cc_, maskhigh_b, b_high));

		b_processed.add(b_high);

		// --------------------------------------------------------
		// process B
		// --------------------------------------------------------

		if (bits_original > 8) {
			processArray(b_processed, b, precomp8b);
		}

		if (bits_original > 16) {
			processArray(b_processed, b, precomp16b);
		}

		if (bits_original > 32) {
			processArray(b_processed, b, precomp32b);
		}

		if (bits_original > 64) {
			processArray(b_processed, b, precomp64b);
		}

		if (bits_original > 128) {
			processArray(b_processed, b, precomp128b);
		}

		// --------------------------------------------------------
		// Combine B
		// --------------------------------------------------------

		if (bits_original > 4) {

			Ciphertext tmp(a.cc_);
			tmp.rotate(b_processed, -16);
			b_processed.add(tmp);
		}

		if (bits_original > 8) {

			Ciphertext tmp(a.cc_);
			tmp.rotate(b_processed, -64);
			b_processed.add(tmp);
		}

		if (bits_original > 16) {

			Ciphertext tmp(a.cc_);
			tmp.rotate(b_processed, -256);
			b_processed.add(tmp);
		}

		if (bits_original > 32) {

			Ciphertext tmp(a.cc_);
			tmp.rotate(b_processed, -1024);
			b_processed.add(tmp);
		}

		if (bits_original > 64) {

			Ciphertext tmp(a.cc_);
			tmp.rotate(b_processed, -4096);
			b_processed.add(tmp);
		}

		if (bits_original > 128) {

			Ciphertext tmp(a.cc_);
			tmp.rotate(b_processed, -16384);
			b_processed.add(tmp);
		}

		// out.copy(b_processed);
		// return;
		// b_processed è giusto (INT)

		// --------------------------------------------------------
		// Convert binary to decimal
		// --------------------------------------------------------

		Ciphertext a_decimal(a.cc_);
		bintodec(cc, a_decimal, a_processed, repetitions * 4);
		// a_decimal seems right (INT and 30 bool)

		Ciphertext b_decimal(a.cc_);
		bintodec(cc, b_decimal, b_processed, repetitions * 4);
		// seems right

		multiplier4bits(result, a_decimal, b_decimal, repetitions * 4, cc);

	} else {
		evalIntegerMult(result, a, b, bits / 2, bits_original, 4 * repetitions, repetitions_original, overflow, cc);
	}

	// ============================================================
	// Recombine multiplication result
	// ============================================================

	// mask1/mask2's *shape* depends only on (bits, repetitions, slots),
	// but the encoded Plaintext also bakes in result's level/noise,
	// which change across recursion depth (every call bootstraps at
	// the end). So we cache by the full (bits, repetitions, slots,
	// level, noise) key: on the first call at a given recursion depth
	// we pay the encode cost once, and every later call that lands at
	// the same depth with the same noise state reuses it -- while
	// never reusing a Plaintext encoded for the wrong level.
	uint32_t openfheLevel = static_cast<uint32_t>(result.cc.L - result.getLevel());
	size_t noiseScaleDeg  = static_cast<size_t>(result.NoiseLevel);

	IntegerMultMaskKey maskKey{ bits, repetitions, result.slots, openfheLevel, noiseScaleDeg };

	auto maskIt = integerMultMaskCache.find(maskKey);

	if (maskIt == integerMultMaskCache.end()) {
		const int dunn = (bits * bits / base_mult) * 2;

		std::vector<double> mask1(a.slots, 0.0);

		for (int j = 0; j < repetitions; ++j) {
			for (int i = 0; i < bits; ++i) {
				mask1[(j * rep_size) + i]		 = 1.0;
				mask1[(j * rep_size) + i + dunn] = 1.0;
			}
		}

		std::vector<double> mask2(a.slots, 0.0);

		for (int j = 0; j < repetitions; ++j) {
			for (int i = 0; i < bits; ++i) {
				mask2[(j * rep_size) + rep_size / 4 + i]		= 1.0;
				mask2[(j * rep_size) + rep_size / 4 + i + dunn] = 1.0;
			}
		}

		IntegerMultMaskPrecomputation entry;
		entry.mask1 = std::make_unique<Plaintext>(makePerSlotPlaintext(cc, cc_, mask1, result));
		entry.mask2 = std::make_unique<Plaintext>(makePerSlotPlaintext(cc, cc_, mask2, result));

		maskIt = integerMultMaskCache.emplace(maskKey, std::move(entry)).first;
	}

	const Plaintext* mask1Pt = maskIt->second.mask1.get();
	const Plaintext* mask2Pt = maskIt->second.mask2.get();

	// ------------------------------------------------------------
	// p1 = result * mask1
	// ------------------------------------------------------------

	Ciphertext p1(a.cc_);

	/*
	multMask(
		p1,
		result,
		mask1,
		cc);
	*/

	p1.copy(result);
	p1.multPt(*mask1Pt, true);

	// ------------------------------------------------------------
	// p2 = rot(p1, -(-rep_size/2 + bits/2))
	// ------------------------------------------------------------

	Ciphertext p2(a.cc_);

	p2.rotate(p1, -(-rep_size / 2 + bits / 2));

	// ------------------------------------------------------------
	// p3
	// ------------------------------------------------------------

	Ciphertext masked2(a.cc_);

	/*
	multMask(
		masked2,
		result,
		mask2,
		cc);
	*/

	masked2.copy(result);
	masked2.multPt(*mask2Pt, true);

	Ciphertext p3(a.cc_);

	if (bits == 8) {

		p3.rotate(masked2, 16);

	} else {

		p3.rotate(masked2, -(-rep_size / 4 + bits / 2));
	}

	// ------------------------------------------------------------
	// p4
	// ------------------------------------------------------------

	Ciphertext p4(a.cc_);

	if (bits == 8) {

		p4.rotate(p3, -12);

	} else {

		p4.rotate(masked2, ((bits - 2) * (3 * bits - 2)) / 8);
	}

	// ============================================================
	// Final CSA + bootstrap
	// ============================================================

	// p1 correct
	// out.copy(p1);
	// return;

	// p2 correct
	// out.copy(p2);
	// return;

	// p3 all zeros, but also in clear
	// out.copy(p3);
	// return;

	if (!overflow && bits == bits_original && bits != 8) {

		Ciphertext S(a.cc_);
		Ciphertext C(a.cc_);

		csa3(S, C, p1, p2, p3);

		Ciphertext rotatedC(a.cc_);

		rotatedC.rotate(C, -1);

		evalIntegerAdd(S, rotatedC, bits);

		if (S.NoiseLevel == 2) {
			S.dropToLevel(5, false);
		} else {
			S.dropToLevel(4, false);
		}

		BootstrapStCFirstBits(S, S.slots, false);

		result.copy(S);
	} else {

		csa4(result, p1, p2, p3, p4, bits);

		if (result.NoiseLevel == 2) {
			result.dropToLevel(5, false);
		} else {
			result.dropToLevel(4, false);
		}

		BootstrapStCFirstBits(result, result.slots, false);
	}

	out.copy(result);
}

// ============================================================
// preprocessDivIntegerLUTs
//
// Bundles the two repeated-Chebyshev LUT precomputations div_integer
// needs:
//   1) bitLengthDecompose: decomposes the normalized bit-length hint
//      (output of inverse_bit_length) into 7 binary "digit" slots per
//      group, so it can drive blind_rotation. CPU side loads
//      p1..p7-norm-247-LUT-DIVISION.txt plus (bits-7) garbage copies of
//      p1 to fill out a full `bits`-wide repeat period.
//   2) reciprocalHint: the actual bits+2-wide Newton-Raphson seed LUT
//      (LUT-DIVISION-<bits>-bits-<i>.txt for i in [0, bits+2)) padded
//      with copies of column 0 out to a full `bits*bits/2`-wide repeat
//      period.
//
// Callers supply the raw coefficient columns themselves (e.g. read once
// from the same files the CPU uses via read_vector_file, or generated in
// Python/offline) instead of this function touching the filesystem --
// this only handles building/caching the PSBatch precomputation, mirroring
// preprocessChebyshevMultiplication/multiplier4bits above.
// ============================================================
void preprocessDivIntegerLUTs(DivIntegerLUTs& luts,
  int bits,
  int zslots,
  lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& cc,
  Ciphertext& like,
  const std::vector<std::vector<double>>& bitLengthCoeffs,
  const std::vector<std::vector<double>>& reciprocalCoeffs) {

	if (static_cast<int>(bitLengthCoeffs.size()) != bits) {
		throw std::invalid_argument("preprocessDivIntegerLUTs: bitLengthCoeffs must have exactly `bits` columns "
									"(7 real + (bits-7) garbage, matching the CPU's p1..p7 + padding layout)");
	}

	if (static_cast<int>(reciprocalCoeffs.size()) != bits * bits / 2) {
		throw std::invalid_argument("preprocessDivIntegerLUTs: reciprocalCoeffs must have exactly bits*bits/2 columns "
									"((bits+2) real + garbage padding, matching the CPU's repeat period)");
	}

	preprocessChebyshevRepeated(luts.bitLengthDecompose, cc, like, bitLengthCoeffs, -1, 1);
	preprocessChebyshevRepeated(luts.reciprocalHint, cc, like, reciprocalCoeffs, 0, 256);
}

void evalIntegerDivision(Ciphertext& out, const Ciphertext& num, const Ciphertext& den, int bits, int zslots, DivIntegerLUTs& luts, const Ciphertext& one,
  const std::vector<std::vector<double>>& bitLengthCoeffs, const std::vector<std::vector<double>>& reciprocalCoeffs, lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& cc) {

	const int LUT_BITS			 = 8;
	FIDESlib::CKKS::Context& cc_ = num.cc_;
	const int stride			 = bits * bits / 2;

	// --------------------------------------------------------
	// b = inverse_bit_length(den, bits, zslots)  -- normalized \hat{x} in [-1,1]
	// --------------------------------------------------------
	Ciphertext b(num.cc_);
	inverseBitLength(b, den, bits, zslots, cc);

	// --------------------------------------------------------
	// s = EvalChebyshevSeriesPSBatchRepeated(b, coeffs, -1, 1, repeat)
	// s = binboot(s)
	//
	// This decomposes bits - bit_length(den) into the 7-bit binary index
	// blind_rotation expects.
	// --------------------------------------------------------
	Ciphertext s(num.cc_);
	s.copy(b);

	//TODO: inverseButLength è giusto

	// Lazy precompute, on first use (or if a previous call cached this LUT
	// against a different level/NoiseLevel than `s` actually has right now)
	// using `s` ITSELF as the model -- this guarantees the plaintext cache
	// recorded inside evalChebyshevSeriesPSBatchPrecompute is replayed
	// against a ciphertext at the exact level/NoiseLevel it was recorded at
	// (see the long comment on evalIntegerDivision in the header for why
	// this matters -- a mismatch here previously caused a GPU illegal
	// memory access, not a clean error). Cached in `luts` afterwards;
	// subsequent calls with matching level/NoiseLevel skip straight to
	// evalChebyshevRepeatedApply.
	if (!luts.bitLengthDecompose.precomp || luts.bitLengthDecompose.modelLevel != s.getLevel() || luts.bitLengthDecompose.modelNoiseLevel != s.NoiseLevel) {
		std::cout << "[evalIntegerDivision] (re)building bitLengthDecompose PSBatch precompute "
					 "(cached level="
				  << luts.bitLengthDecompose.modelLevel << ", cached noise=" << luts.bitLengthDecompose.modelNoiseLevel << "; s level=" << s.getLevel()
				  << ", s noise=" << s.NoiseLevel << ")" << std::endl;
		preprocessChebyshevRepeated(luts.bitLengthDecompose, cc, s, bitLengthCoeffs, -1, 1);
		std::cout << "[evalIntegerDivision] done " << std::endl;
	} else {
		std::cout << "[evalIntegerDivision] reusing cached bitLengthDecompose PSBatch precompute" << std::endl;
	}
	evalChebyshevRepeatedApply(cc, s, luts.bitLengthDecompose);
	binboot(s, s);

	

	// --------------------------------------------------------
	// den_norm = blind_rotation(den, s, bits, zslots)   // den << (bits - bitlen(den))
	// den_norm = binboot(den_norm)
	// --------------------------------------------------------
	Ciphertext denNorm(num.cc_);
	blindRotation(denNorm, den, s, bits, zslots, /*stride=*/0, cc);

	

	binboot(denNorm, denNorm);

	

	// --------------------------------------------------------
	// den_norm_rot = rot(den_norm, bits - 1 - LUT_BITS)
	//
	// Bring the top LUT_BITS bits of the normalized denominator down to
	// slot 0 of each group so they can be packed into a decimal index.
	// --------------------------------------------------------
	Ciphertext denNormRot(num.cc_);
	denNormRot.rotate(denNorm, bits - 1 - LUT_BITS);

	// --------------------------------------------------------
	// idx = den_norm_rot * {2^0..2^(LUT_BITS-1), 0, 0, ...} (per group)
	// idx = sum of log2(LUT_BITS) rotations of idx  -> idx in [0, 256)
	// --------------------------------------------------------
	std::vector<double> mask(num.slots, 0.0);
	for (int j = 0; j < zslots; ++j) {
		for (int i = 0; i < LUT_BITS; ++i) {
			mask[stride * j + i] = std::pow(2.0, i);
		}
	}

	Ciphertext idx(num.cc_);
	idx.multPt(denNormRot, makePerSlotPlaintext(cc, cc_, mask, denNormRot));

	{
		const int idxRounds = static_cast<int>(std::log2(LUT_BITS));
		for (int i = 0; i < idxRounds; ++i) {
			Ciphertext rotated(num.cc_);
			rotated.rotate(idx, 1 << i);
			idx.add(rotated);
		}
	}

	// --------------------------------------------------------
	// idx = idx * {1 at slot 0 of each group, else 0}   (keep only the packed value)
	// idx_masked_clone = idx.Clone()
	// idx = sum of log2(bits) rotations of idx by -2^j  (broadcast within group)
	// idx += rot(idx_masked_clone, -bits) + rot(idx_masked_clone, -bits-1)
	// --------------------------------------------------------
	std::fill(mask.begin(), mask.end(), 0.0);
	for (int j = 0; j < zslots; ++j) {
		mask[j * stride] = 1.0;
	}

	idx.multPt(makePerSlotPlaintext(cc, cc_, mask, idx));

	Ciphertext idxMaskedClone(num.cc_);
	idxMaskedClone.copy(idx);

	{
		const int idxRounds = static_cast<int>(std::log2(bits));
		for (int i = 0; i < idxRounds; ++i) {
			Ciphertext rotated(num.cc_);
			rotated.rotate(idx, -(1 << i));
			idx.add(rotated);
		}
	}

	{
		Ciphertext rotated(num.cc_);
		rotated.rotate(idxMaskedClone, -bits);
		idx.add(rotated);
	}
	{
		Ciphertext rotated(num.cc_);
		rotated.rotate(idxMaskedClone, -bits - 1);
		idx.add(rotated);
	}

	// --------------------------------------------------------
	// x = EvalChebyshevSeriesPSBatchRepeated(idx, coeffs, 0, 256, repeat)
	// x = binboot(x)
	//
	// x is the Newton-Raphson reciprocal seed ("first hint").
	// --------------------------------------------------------
	Ciphertext x(num.cc_);
	x.copy(idx);

	// Same lazy-precompute-with-validation pattern as bitLengthDecompose
	// above, but here `x` is used as the model BEFORE its own binboot
	// (unlike `s`, which was already freshly bootstrapped via
	// inverseBitLength) -- whatever level/NoiseLevel `x` happens to be at
	// right here, from blindRotation/evalIntegerMult upstream, is what gets
	// baked into the precompute, and is exactly what it needs to match on
	// every call.
	if (!luts.reciprocalHint.precomp || luts.reciprocalHint.modelLevel != x.getLevel() || luts.reciprocalHint.modelNoiseLevel != x.NoiseLevel) {
		std::cout << "[evalIntegerDivision] (re)building reciprocalHint PSBatch precompute "
					 "(cached level="
				  << luts.reciprocalHint.modelLevel << ", cached noise=" << luts.reciprocalHint.modelNoiseLevel << "; x level=" << x.getLevel()
				  << ", x noise=" << x.NoiseLevel << ")" << std::endl;
		preprocessChebyshevRepeated(luts.reciprocalHint, cc, x, reciprocalCoeffs, 0, 256);
		std::cout << "[evalIntegerDivision] done " << std::endl;
	} else {
		std::cout << "[evalIntegerDivision] reusing cached reciprocalHint PSBatch precompute" << std::endl;
	}
	evalChebyshevRepeatedApply(cc, x, luts.reciprocalHint);
	binboot(x, x);

	
	// --------------------------------------------------------
	// Newton-Raphson refinement loop:
	//   for iter in [0, ceil(log2(bits/LUT_BITS))):
	//       term = mul_integer(x, den_norm, bits, bits, zslots, zslots, true)
	//       term += rot(den_norm, -bits)        * bit `bits`   of x   (broadcast)
	//       term += rot(den_norm, -bits-1)      * bit `bits+1` of x   (broadcast)
	//       term = ~term (bitwise complement over the low bits*2+1 bits)
	//       term = binboot(add_integer(term, {1 at slot 0}, bits*2, false))  // two's complement negation, i.e. -term
	//       term = rot(term, bits)
	//       x = mul_integer(rot(x,2), rot(term,2), bits, bits, zslots, zslots, true)
	//       x = rot(rot(rot(rot(rot(x, bits), -1), -1), -1), -1)
	// --------------------------------------------------------
	const int newtonIters = static_cast<int>(std::ceil(std::log2(static_cast<double>(bits) / LUT_BITS)));

	//TODO prima del loop è giusto, poi si rompe tutto (non è neanche tutto binario)
	for (int iter = 0; iter < newtonIters - 1; ++iter) {

		Ciphertext term(num.cc_);

		x.dropToLevel(x.getLevel() - 1);
		denNorm.dropToLevel(denNorm.getLevel() - 1);

		evalIntegerMult(term, x, denNorm, bits, bits, zslots, zslots, true, cc);

		// term += broadcast(bit `bits` of x) * rot(den_norm, -bits)
		std::fill(mask.begin(), mask.end(), 0.0);
		for (int j = 0; j < zslots; ++j) {
			mask[bits + j * stride] = 1.0;
		}

		{
			Ciphertext lastBit(num.cc_);
			lastBit.multPt(x, makePerSlotPlaintext(cc, cc_, mask, x));

			const int bcRounds = static_cast<int>(std::log2(bits));
			for (int j = 0; j < bcRounds; ++j) {
				Ciphertext rotated(num.cc_);
				rotated.rotate(lastBit, -(1 << j));
				lastBit.add(rotated);
			}

			Ciphertext denShift(num.cc_);
			denShift.rotate(denNorm, -bits);
			lastBit.mult(denShift, true);

			evalIntegerAdd(term, lastBit, bits);
			binboot(term, term);
		}

		// term += broadcast(bit `bits+1` of x) * rot(den_norm, -bits-1)
		std::fill(mask.begin(), mask.end(), 0.0);
		for (int j = 0; j < zslots; ++j) {
			mask[(bits + 1) + j * stride] = 1.0;
		}

		{
			Ciphertext lastBit(num.cc_);
			lastBit.multPt(x, makePerSlotPlaintext(cc, cc_, mask, x));

			const int bcRounds = static_cast<int>(std::log2(bits));
			for (int j = 0; j < bcRounds; ++j) {
				Ciphertext rotated(num.cc_);
				rotated.rotate(lastBit, -(1 << j));
				lastBit.add(rotated);
			}

			Ciphertext denShift(num.cc_);
			denShift.rotate(denNorm, -bits);
			{
				Ciphertext rotated(num.cc_);
				rotated.rotate(denShift, -1);
				denShift.copy(rotated);
			}
			lastBit.mult(denShift, true);

			evalIntegerAdd(term, lastBit, bits);
			binboot(term, term);
		}

		// term = complement(term) over the low (bits*2+1) bits of each group
		std::fill(mask.begin(), mask.end(), 0.0);
		for (int j = 0; j < zslots; ++j) {
			for (int i = 0; i < bits * 2 + 1; ++i) {
				mask[i + j * stride] = 1.0;
			}
		}

		{
			Ciphertext ones(num.cc_);
			ones.copy(term);
			Ciphertext complemented(num.cc_);
			complemented.multScalar(term, -1.0, true);
			complemented.addPt(makePerSlotPlaintext(cc, cc_, mask, complemented));
			term.copy(complemented);
		}

		// term = binboot(add_integer(term, {1 at slot 0 of each group}, bits*2, false))
		// (the "+1" that finishes two's-complement negation: -term == ~term + 1)
		//
		// The CPU calls add_integer(term, encrypt(mask, ...), ...) here, i.e.
		// it needs the "+1" constant as a genuinely-encrypted ciphertext (not
		// just a plaintext mask), because add_integer's carry-lookahead logic
		// takes two ciphertexts. `one` is exactly that: a real encryption of
		// {1 at slot 0 of every bits*bits/2-sized group, 0 elsewhere}, built
		// once by the caller (see evalIntegerDivision's doc comment in the
		// header) and passed in here, the same way `luts` is.
		{
			// `one` must have been encrypted at a level at least as fresh as
			// `term`'s here (dropToLevel only lowers, never raises). If the
			// caller built `one` at a shallower level, fail loudly instead of
			// letting a bad level slip through to the GPU kernel below (an
			// invalid drop target has previously shown up as a CUDA "illegal
			// memory access" rather than a clean error).
			if (one.getLevel() < term.getLevel()) {
				throw std::invalid_argument(
				  "evalIntegerDivision: `one` was encrypted at a lower level than term's post-bootstrap "
				  "level; DivIntegerPrecomputations must encrypt it at the top of the modulus chain (level 0)");
			}

			Ciphertext oneAtLevel(num.cc_);
			oneAtLevel.copy(one);
			oneAtLevel.dropToLevel(term.getLevel(), false);

			evalIntegerAdd(term, oneAtLevel, bits * 2);
			binboot(term, term);
		}

		{
			Ciphertext rotated(num.cc_);
			rotated.rotate(term, bits); // drop the low `bits` garbage bits
			term.copy(rotated);
		}

		// x = mul_integer(rot(x,2), rot(term,2), bits, bits, zslots, zslots, true)
		{
			Ciphertext x2(num.cc_);
			x2.rotate(x, 2);
			Ciphertext term2(num.cc_);
			term2.rotate(term, 2);

			term2.dropToLevel(term2.getLevel() - 1);

			Ciphertext newX(num.cc_);
			evalIntegerMult(newX, x2, term2, bits, bits, zslots, zslots, true, cc);
			x.copy(newX);
		}

		// CPU: x = rot(x, bits); x = rot(x, -4);  (the intermediate 4x
		// rot(x,-1) is explicitly commented out in div_integer in favor of a
		// single rot(x,-4) -- done as one rotation here too, both to match
		// exactly and because `x.rotate(x, n)` (same object as src and dst)
		// is not a pattern used safely anywhere else in this file; every
		// other rotate() call in this port writes into a distinct
		// destination ciphertext.
		{
			Ciphertext xRotated(num.cc_);
			xRotated.rotate(x, bits);
			x.rotate(xRotated, -4);
		}

	}


	// --------------------------------------------------------
	// result = mul_integer(rot(num, 2), rot(x, 2), bits, bits, zslots, zslots, true)
	// result = rot(rot(rot(rot(result, -1), -1), -1), -1)
	// result = rot(result, bits)
	// result = result * {1 at low bits*2 bits of each group}
	// --------------------------------------------------------
	Ciphertext result(num.cc_);
	{
		Ciphertext num2(num.cc_);
		num2.rotate(num, 2);
		Ciphertext x2(num.cc_);
		x2.rotate(x, 2);

		x2.dropToLevel(x2.getLevel() - 1);
		evalIntegerMult(result, num2, x2, bits, bits, zslots, zslots, true, cc);
	}

	// CPU does 4 separate rot(result, -1) here (not collapsed to rot(-4)
	// like the `x` case above), so we keep 4 separate steps too -- but via a
	// distinct destination buffer each time rather than result.rotate(result, ...)
	// in-place, for the same reason as the `x` fix above.
	for (int r = 0; r < 4; ++r) {
		Ciphertext rotated(num.cc_);
		rotated.rotate(result, -1);
		result.copy(rotated);
	}

	{
		Ciphertext rotated(num.cc_);
		rotated.rotate(result, bits);
		result.copy(rotated);
	}

	std::fill(mask.begin(), mask.end(), 0.0);
	for (int j = 0; j < zslots; ++j) {
		for (int i = 0; i < bits * 2; ++i) {
			mask[i + j * stride] = 1.0;
		}
	}
	result.multPt(makePerSlotPlaintext(cc, cc_, mask, result));

	// --------------------------------------------------------
	// result = blind_rotation(result, s, bits*2, zslots, bits*bits/2)
	// result = rot(result, bits)
	// result = result * {1 at low bits bits of each group}
	// --------------------------------------------------------
	Ciphertext rotatedResult(num.cc_);
	blindRotation(rotatedResult, result, s, bits * 2, zslots, stride, cc);
	result.copy(rotatedResult);

	{
		Ciphertext rotated(num.cc_);
		rotated.rotate(result, bits);
		result.copy(rotated);
	}

	std::fill(mask.begin(), mask.end(), 0.0);
	for (int j = 0; j < zslots; ++j) {
		for (int i = 0; i < bits; ++i) {
			mask[i + j * stride] = 1.0;
		}
	}
	result.multPt(makePerSlotPlaintext(cc, cc_, mask, result));

	out.copy(result);
}

// ============================================================
// preprocessSquareRootLUTs
//
// Bundles the two repeated-Chebyshev LUT precomputations
// square_root_integer needs:
//   1) bitLengthDecompose: same role as DivIntegerLUTs::bitLengthDecompose
//      (decomposes the normalized bit-length hint into 7 binary "digit"
//      slots per group to drive blind_rotation). CPU side loads
//      p1..p7-norm-247-LUT-DIVISION.txt plus (bits-7) garbage copies of p1
//      to fill out a full `bits`-wide repeat period -- see
//      square_root_integer's first `coeffs.push_back(...)` block, which
//      reloads these files independently of div_integer.
//   2) newtonSeed: the Newton-Raphson seed for 1/sqrt(x), loaded from
//      LUT-SQUARE-ROOT-<bits>-BITS-<i>.txt. The exact column count and
//      garbage-padding file depend on `bits` (see the bits==64 / bits==128 /
//      default branches in CKKSController::square_root_integer); so does the
//      subsequent in-place column reassignment CPU does for bits==64/128
//      (`coeffs[0] = coeffs[12]`, etc.) -- callers must apply that patching
//      to `newtonSeedCoeffs` themselves before calling this function (or
//      evalIntegerSquareRoot), since it depends only on `bits`, not on any
//      ciphertext.
// ============================================================
void preprocessSquareRootLUTs(SquareRootIntegerLUTs& luts,
  int bits,
  int zslots,
  lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& cc,
  Ciphertext& like,
  const std::vector<std::vector<double>>& bitLengthCoeffs,
  const std::vector<std::vector<double>>& newtonSeedCoeffs) {

	if (static_cast<int>(bitLengthCoeffs.size()) != bits) {
		throw std::invalid_argument("preprocessSquareRootLUTs: bitLengthCoeffs must have exactly `bits` columns "
									"(7 real + (bits-7) garbage, matching the CPU's p1..p7 + padding layout)");
	}

	if (static_cast<int>(newtonSeedCoeffs.size()) != bits * bits / 2) {
		throw std::invalid_argument("preprocessSquareRootLUTs: newtonSeedCoeffs must have exactly bits*bits/2 columns "
									"(real LUT-SQUARE-ROOT-<bits>-BITS-<i> columns + garbage padding, matching the "
									"CPU's repeat period)");
	}

	preprocessChebyshevRepeated(luts.bitLengthDecompose, cc, like, bitLengthCoeffs, -1, 1);
	preprocessChebyshevRepeated(luts.newtonSeed, cc, like, newtonSeedCoeffs, 0, 256);
}

// ============================================================
// square_root_integer
//
// Direct, per-op translation of CKKSController::square_root_integer(const
// Ctxt&, int, int). See the header comment on evalIntegerSquareRoot for the
// meaning/level requirements of `one`/`sqrt2`/`bitsPlusOne`, and the comment
// on preprocessSquareRootLUTs for `bitLengthCoeffs`/`newtonSeedCoeffs`.
//
// Just like evalIntegerDivision, the two PSBatch precomputes in `luts` are
// (re)built LAZILY, on first use -- directly against the internal `s`/`idx`
// ciphertexts themselves, right before they're consumed -- rather than ahead
// of time against a hand-picked model, for the exact same reason (see the
// long comment on evalIntegerDivision in the header).
// ============================================================
void evalIntegerSquareRoot(Ciphertext& out,
  const Ciphertext& c,
  int bits,
  int zslots,
  SquareRootIntegerLUTs& luts,
  const Ciphertext& one,
  const Ciphertext& sqrt2,
  const Ciphertext& bitsPlusOne,
  const std::vector<std::vector<double>>& bitLengthCoeffs,
  const std::vector<std::vector<double>>& newtonSeedCoeffs,
  lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& cc) {

	const int LUT_BITS			 = 8;
	FIDESlib::CKKS::Context& cc_ = c.cc_;
	const int stride			 = bits * bits / 2;

	// --------------------------------------------------------
	// hatx = inverse_bit_length(c, bits, zslots)  -- normalized \hat{x} in [-1,1]
	// --------------------------------------------------------
	Ciphertext hatx(c.cc_);
	inverseBitLength(hatx, c, bits, zslots, cc);

	// --------------------------------------------------------
	// s = EvalChebyshevSeriesPSBatchRepeated(hatx, coeffs, -1, 1, repeat)
	// s = binboot(s)
	//
	// This decomposes bits - bit_length(c) into the 7-bit binary index
	// blind_rotation expects.
	// --------------------------------------------------------
	Ciphertext s(c.cc_);
	s.copy(hatx);

	if (!luts.bitLengthDecompose.precomp || luts.bitLengthDecompose.modelLevel != s.getLevel() || luts.bitLengthDecompose.modelNoiseLevel != s.NoiseLevel) {
		std::cout << "[evalIntegerSquareRoot] (re)building bitLengthDecompose PSBatch precompute "
					 "(cached level="
				  << luts.bitLengthDecompose.modelLevel << ", cached noise=" << luts.bitLengthDecompose.modelNoiseLevel << "; s level=" << s.getLevel()
				  << ", s noise=" << s.NoiseLevel << ")" << std::endl;
		preprocessChebyshevRepeated(luts.bitLengthDecompose, cc, s, bitLengthCoeffs, -1, 1);
		std::cout << "[evalIntegerSquareRoot] done " << std::endl;
	} else {
		std::cout << "[evalIntegerSquareRoot] reusing cached bitLengthDecompose PSBatch precompute" << std::endl;
	}
	evalChebyshevRepeatedApply(cc, s, luts.bitLengthDecompose);
	binboot(s, s);

	// --------------------------------------------------------
	// hatx = blind_rotation(c, s, bits, zslots)   // c << (bits - bitlen(c))
	// hatx = binboot(hatx)
	// hatx_clone = hatx->Clone()
	// hatx = rot(hatx, bits - 1 - LUT_BITS)
	// --------------------------------------------------------
	Ciphertext hatxRotated(c.cc_);
	blindRotation(hatxRotated, c, s, bits, zslots, /*stride=*/0, cc);
	binboot(hatxRotated, hatxRotated);

	Ciphertext hatxClone(c.cc_);
	hatxClone.copy(hatxRotated);

	hatx.rotate(hatxRotated, bits - 1 - LUT_BITS);

	// --------------------------------------------------------
	// idx = hatx * {2^0..2^(LUT_BITS-1), 0, 0, ...} (per group)
	// idx = sum of log2(LUT_BITS) rotations of idx  -> idx in [0, 256)
	// idx = idx * {1 at slot 0 of each group}
	// idx_masked_clone = idx->Clone()
	// idx = sum of log2(bits) rotations of idx (broadcast index across group)
	// idx = idx + rot(idx_masked_clone, -bits) + rot(idx_masked_clone, -bits-1)
	// --------------------------------------------------------
	std::vector<double> mask(c.slots, 0.0);
	for (int j = 0; j < zslots; ++j) {
		for (int i = 0; i < LUT_BITS; ++i) {
			mask[stride * j + i] = std::pow(2.0, i);
		}
	}

	Ciphertext idx(c.cc_);
	idx.multPt(hatx, makePerSlotPlaintext(cc, cc_, mask, hatx));

	{
		const int rounds = static_cast<int>(std::log2(LUT_BITS));
		for (int i = 0; i < rounds; ++i) {
			Ciphertext rotated(c.cc_);
			rotated.rotate(idx, 1 << i);
			idx.add(rotated);
		}
	}

	std::fill(mask.begin(), mask.end(), 0.0);
	for (int j = 0; j < zslots; ++j) {
		mask[j * stride] = 1.0;
	}
	idx.multPt(makePerSlotPlaintext(cc, cc_, mask, idx));

	Ciphertext idxMaskedClone(c.cc_);
	idxMaskedClone.copy(idx);

	{
		const int rounds = static_cast<int>(std::log2(bits));
		for (int i = 0; i < rounds; ++i) {
			Ciphertext rotated(c.cc_);
			rotated.rotate(idx, -(1 << i));
			idx.add(rotated);
		}
	}

	{
		Ciphertext rotated(c.cc_);
		rotated.rotate(idxMaskedClone, -bits);
		idx.add(rotated);
	}
	{
		Ciphertext rotated(c.cc_);
		rotated.rotate(idxMaskedClone, -bits - 1);
		idx.add(rotated);
	}

	// --------------------------------------------------------
	// x = EvalChebyshevSeriesPSBatchRepeated(idx, newtonSeedCoeffs, 0, 256, repeat)
	//
	// (Any bits==16/32/64/128-specific dead-slot masking/patching the CPU
	// applies to `x` right after this must already be baked into
	// newtonSeedCoeffs by the caller for the coefficient-level patches, and
	// is applied explicitly below for the slot-masking patches -- see the
	// per-bits blocks immediately following.)
	// --------------------------------------------------------
	Ciphertext x(c.cc_);
	x.copy(idx);

	if (!luts.newtonSeed.precomp || luts.newtonSeed.modelLevel != x.getLevel() || luts.newtonSeed.modelNoiseLevel != x.NoiseLevel) {
		std::cout << "[evalIntegerSquareRoot] (re)building newtonSeed PSBatch precompute "
					 "(cached level="
				  << luts.newtonSeed.modelLevel << ", cached noise=" << luts.newtonSeed.modelNoiseLevel << "; x level=" << x.getLevel()
				  << ", x noise=" << x.NoiseLevel << ")" << std::endl;
		preprocessChebyshevRepeated(luts.newtonSeed, cc, x, newtonSeedCoeffs, 0, 256);
		std::cout << "[evalIntegerSquareRoot] done " << std::endl;
	} else {
		std::cout << "[evalIntegerSquareRoot] reusing cached newtonSeed PSBatch precompute" << std::endl;
	}
	evalChebyshevRepeatedApply(cc, x, luts.newtonSeed);

	// --------------------------------------------------------
	// Dead-slot masking, verbatim per bits (see square_root_integer's
	// bits==16/32/64/128 blocks right after the PSBatch call). Slots at
	// specific offsets have null polynomials there, so PS returns garbage in
	// them; zero those out (and, for bits==128, additionally force slot 127
	// to 1, matching the CPU's `x = add(x, encode(mask, ...))`).
	// --------------------------------------------------------
	if (bits == 16) {
		std::fill(mask.begin(), mask.end(), 1.0);
		for (int j = 0; j < zslots; ++j) {
			mask[stride * j + 14] = 0.0;
			mask[stride * j + 16] = 0.0;
			mask[stride * j + 17] = 0.0;
		}
		x.multPt(makePerSlotPlaintext(cc, cc_, mask, x));
	} else if (bits == 32) {
		std::fill(mask.begin(), mask.end(), 1.0);
		for (int j = 0; j < zslots; ++j) {
			mask[stride * j + 30] = 0.0;
			mask[stride * j + 32] = 0.0;
			mask[stride * j + 33] = 0.0;
		}
		x.multPt(makePerSlotPlaintext(cc, cc_, mask, x));
	} else if (bits == 64) {
		std::fill(mask.begin(), mask.end(), 1.0);
		for (int j = 0; j < zslots; ++j) {
			mask[stride * j + 62] = 0.0;
			mask[stride * j + 64] = 0.0;
			mask[stride * j + 65] = 0.0;
			for (int i = 0; i <= 10; ++i) {
				mask[stride * j + i] = 0.0;
			}
		}
		x.multPt(makePerSlotPlaintext(cc, cc_, mask, x));
	} else if (bits == 128) {
		std::fill(mask.begin(), mask.end(), 1.0);
		for (int j = 0; j < zslots; ++j) {
			for (int i = 0; i < 75; ++i) {
				mask[stride * j + i] = 0.0;
			}
			mask[stride * j + 126] = 0.0;
			mask[stride * j + 127] = 0.0;
			mask[stride * j + 128] = 0.0;
			mask[stride * j + 129] = 0.0;
		}
		x.multPt(makePerSlotPlaintext(cc, cc_, mask, x));

		std::fill(mask.begin(), mask.end(), 0.0);
		for (int j = 0; j < zslots; ++j) {
			mask[stride * j + 127] = 1.0;
		}
		x.addPt(makePerSlotPlaintext(cc, cc_, mask, x));
	}

	binboot(x, x);

	// --------------------------------------------------------
	// Newton-Raphson refinement loop for 1/sqrt(x):
	//   for iter in [0, ceil(log2(bits/LUT_BITS))):
	//       x2   = (x * x) >> bits, then >> 1        // mul_integer(x,x,...) >> bits, then rot(-1)
	//       mx2  = (hatx_clone * x2) >> bits          // mul_integer(hatx_clone, x2, ...) >> bits
	//       term1 = (3 << F) - mx2, masked to bits*2 bits, then binboot(add_integer(...))
	//       x, term1 masked to low (bits+2) bits of each group
	//       x = (x * term1) >> (F+1), split into low/high halves of term1
	//         term1_lo * x -> mul_integer, term1_hi (broadcast) * x -> plain mult
	//       x = binboot(xpartial)
	// --------------------------------------------------------
	const int newtonIters = static_cast<int>(std::ceil(std::log2(static_cast<double>(bits) / LUT_BITS)));

	for (int iter = 0; iter < newtonIters; ++iter) {

		// x2 = mul_integer(x, x, bits, bits, zslots, zslots, true); x2 = rot(rot(x2, bits), -1)
		Ciphertext x2(c.cc_);
		{
			Ciphertext xLo(c.cc_);
			xLo.copy(x);
			xLo.dropToLevel(xLo.getLevel() - 1);
			Ciphertext xHi(c.cc_);
			xHi.copy(xLo);
			evalIntegerMult(x2, xLo, xHi, bits, bits, zslots, zslots, true, cc);
		}
		{
			Ciphertext rotated(c.cc_);
			rotated.rotate(x2, bits);
			Ciphertext rotated2(c.cc_);
			rotated2.rotate(rotated, -1);
			x2.copy(rotated2);
		}

		// mx2 = mul_integer(hatx_clone, x2, bits, bits, zslots, zslots, true); mx2 = rot(mx2, bits)
		Ciphertext mx2(c.cc_);
		{
			Ciphertext hcCopy(c.cc_);
			hcCopy.copy(hatxClone);
			hcCopy.dropToLevel(std::min(hcCopy.getLevel(), x2.getLevel() - 1));
			Ciphertext x2Copy(c.cc_);
			x2Copy.copy(x2);
			x2Copy.dropToLevel(hcCopy.getLevel());
			evalIntegerMult(mx2, hcCopy, x2Copy, bits, bits, zslots, zslots, true, cc);
		}
		{
			Ciphertext rotated(c.cc_);
			rotated.rotate(mx2, bits);
			mx2.copy(rotated);
		}

		// const3f = encrypt_multi_int({3,...}, bits, mx2->GetLevel())
		// const3f = rot(rot(const3f, -bits), 1)
		// const3f = const3f + {1 at slot 0 of each group}
		//
		// `3` is a small, non-secret plaintext constant, so unlike `one` /
		// `sqrt2` / `bitsPlusOne` above (whose bit patterns depend on
		// caller-chosen values baked in ahead of time), we can build it
		// in-line from a plaintext mask rather than requiring another
		// pre-encrypted ciphertext input -- mirroring how `ones`/`mask`
		// plaintext constants are built inline throughout this file.
		// 3 = 0b11, i.e. bits 0 and 1 set (LSB-first packing).
		std::vector<double> three(c.slots, 0.0);
		for (int j = 0; j < zslots; ++j) {
			three[0 + j * stride] = 1.0;
			three[1 + j * stride] = 1.0;
		}

		Ciphertext const3f(c.cc_);
		{
			// Encode `3` directly as a plaintext bit-pattern and lift it to a
			// trivial ciphertext-shaped value by zeroing a clone of mx2 (to
			// match its level/NoiseLevel) and adding the plaintext in-place.
			const3f.multScalar(mx2, 0.0, true);
			const3f.addPt(makePerSlotPlaintext(cc, cc_, three, const3f));
		}
		{
			Ciphertext rotated(c.cc_);
			rotated.rotate(const3f, -bits);
			Ciphertext rotated2(c.cc_);
			rotated2.rotate(rotated, 1);
			const3f.copy(rotated2);
		}

		std::fill(mask.begin(), mask.end(), 0.0);
		for (int j = 0; j < zslots; ++j) {
			mask[stride * j] = 1.0;
		}
		const3f.addPt(makePerSlotPlaintext(cc, cc_, mask, const3f));

		// inverted = {1 at low bits*2 bits of each group} - mx2
		// term1 = binboot(add_integer(const3f, inverted, bits*2, false))
		std::fill(mask.begin(), mask.end(), 0.0);
		for (int j = 0; j < zslots; ++j) {
			for (int z = 0; z < bits * 2; ++z) {
				mask[z + j * stride] = 1.0;
			}
		}

		Ciphertext inverted(c.cc_);
		inverted.multScalar(mx2, -1.0, true);
		inverted.addPt(makePerSlotPlaintext(cc, cc_, mask, inverted));

		Ciphertext term1(c.cc_);
		{
			Ciphertext const3fCopy(c.cc_);
			const3fCopy.copy(const3f);
			const3fCopy.dropToLevel(std::min(const3fCopy.getLevel(), inverted.getLevel()));
			Ciphertext invertedCopy(c.cc_);
			invertedCopy.copy(inverted);
			invertedCopy.dropToLevel(const3fCopy.getLevel());
			evalIntegerAdd(const3fCopy, invertedCopy, bits * 2);
			term1.copy(const3fCopy);
		}
		binboot(term1, term1);

		// x, term1 := mask to low (bits+2) bits of each group
		std::fill(mask.begin(), mask.end(), 0.0);
		for (int j = 0; j < zslots; ++j) {
			for (int z = 0; z < bits + 2; ++z) {
				mask[z + stride * j] = 1.0;
			}
		}
		x.multPt(makePerSlotPlaintext(cc, cc_, mask, x));
		term1.multPt(makePerSlotPlaintext(cc, cc_, mask, term1));

		// term1_lo = term1 * {1 at low bits bits of each group}
		Ciphertext term1Lo(c.cc_);
		{
			std::vector<double> maskLo(c.slots, 0.0);
			for (int z = 0; z < zslots; ++z) {
				for (int j = 0; j < bits; ++j) {
					maskLo[j + z * stride] = 1.0;
				}
			}
			term1Lo.multPt(term1, makePerSlotPlaintext(cc, cc_, maskLo, term1), true);
		}

		// term1_hi = rot(term1 * {1 at slot `bits` of each group}, bits)
		// term1_hi = sum of log2(bits) rotations of term1_hi (broadcast)
		// term1_hi = term1_hi * x
		Ciphertext term1Hi(c.cc_);
		{
			std::vector<double> maskHi(c.slots, 0.0);
			for (int j = 0; j < zslots; ++j) {
				maskHi[bits + j * stride] = 1.0;
			}
			Ciphertext masked(c.cc_);
			masked.multPt(term1, makePerSlotPlaintext(cc, cc_, maskHi, term1), true);
			term1Hi.rotate(masked, bits);

			const int rounds = static_cast<int>(std::log2(bits));
			for (int j = 0; j < rounds; ++j) {
				Ciphertext rotated(c.cc_);
				rotated.rotate(term1Hi, -(1 << j));
				term1Hi.add(rotated);
			}

			Ciphertext xCopy(c.cc_);
			xCopy.copy(x);
			xCopy.dropToLevel(std::min(xCopy.getLevel(), term1Hi.getLevel()));
			term1Hi.dropToLevel(xCopy.getLevel());
			term1Hi.mult(xCopy, true);
		}

		// xpartial = mul_integer(x, term1_lo, bits, bits, zslots, zslots, true); xpartial = rot(xpartial, bits)
		// xpartial = xpartial * {1 at low bits bits of each group}
		// xpartial = add_integer(xpartial, term1_hi, bits, false)
		Ciphertext xpartial(c.cc_);
		{
			Ciphertext xCopy(c.cc_);
			xCopy.copy(x);
			xCopy.dropToLevel(std::min(xCopy.getLevel(), term1Lo.getLevel()) - 1);
			Ciphertext term1LoCopy(c.cc_);
			term1LoCopy.copy(term1Lo);
			term1LoCopy.dropToLevel(xCopy.getLevel());
			evalIntegerMult(xpartial, xCopy, term1LoCopy, bits, bits, zslots, zslots, true, cc);
		}
		{
			Ciphertext rotated(c.cc_);
			rotated.rotate(xpartial, bits);
			xpartial.copy(rotated);
		}

		std::fill(mask.begin(), mask.end(), 0.0);
		for (int j = 0; j < zslots; ++j) {
			for (int z = 0; z < bits; ++z) {
				mask[z + j * stride] = 1.0;
			}
		}
		xpartial.multPt(makePerSlotPlaintext(cc, cc_, mask, xpartial));

		{
			Ciphertext term1HiCopy(c.cc_);
			term1HiCopy.copy(term1Hi);
			term1HiCopy.dropToLevel(std::min(term1HiCopy.getLevel(), xpartial.getLevel()));
			xpartial.dropToLevel(term1HiCopy.getLevel());
			evalIntegerAdd(xpartial, term1HiCopy, bits);
		}

		x.copy(xpartial);
		binboot(x, x);
	}

	// --------------------------------------------------------
	// y = mul_integer(x, hatx_clone, bits, bits, zslots, zslots, true); y = rot(y, bits)
	// --------------------------------------------------------
	Ciphertext y(c.cc_);
	{
		Ciphertext xCopy(c.cc_);
		xCopy.copy(x);
		Ciphertext hcCopy(c.cc_);
		hcCopy.copy(hatxClone);
		hcCopy.dropToLevel(std::min(hcCopy.getLevel(), xCopy.getLevel()) - 1);
		xCopy.dropToLevel(hcCopy.getLevel());
		evalIntegerMult(y, xCopy, hcCopy, bits, bits, zslots, zslots, true, cc);
	}
	{
		Ciphertext rotated(c.cc_);
		rotated.rotate(y, bits);
		y.copy(rotated);
	}

	// --------------------------------------------------------
	// parity = s * {1 at slot 0 of each group}; parity = broadcast(parity)
	// parityinverse = {1 at low bits bits of each group} - parity
	// correction = SQRT2_FP_CTXT * parity + ONE_FP_CTXT * parityinverse
	// correction = binboot(correction)
	// --------------------------------------------------------
	Ciphertext parity(c.cc_);
	{
		std::fill(mask.begin(), mask.end(), 0.0);
		for (int j = 0; j < zslots; ++j) {
			mask[stride * j] = 1.0;
		}
		parity.multPt(s, makePerSlotPlaintext(cc, cc_, mask, s), true);

		const int rounds = static_cast<int>(std::log2(bits));
		for (int j = 0; j < rounds; ++j) {
			Ciphertext rotated(c.cc_);
			rotated.rotate(parity, -(1 << j));
			parity.add(rotated);
		}
	}

	Ciphertext parityInverse(c.cc_);
	{
		std::fill(mask.begin(), mask.end(), 0.0);
		for (int z = 0; z < zslots; ++z) {
			for (int j = 0; j < bits; ++j) {
				mask[j + z * stride] = 1.0;
			}
		}
		parityInverse.multScalar(parity, -1.0, true);
		parityInverse.addPt(makePerSlotPlaintext(cc, cc_, mask, parityInverse));
	}

	// `one`/`sqrt2` are supplied pre-encrypted (see the header comment on
	// evalIntegerSquareRoot for why -- they encode caller-chosen constants
	// ONE_FP/SQRT2_FP that this function has no key material to encrypt
	// itself). Drop them to parity's level before use, same pattern as
	// evalIntegerDivision's `one` handling.
	if (one.getLevel() < parity.getLevel()) {
		throw std::invalid_argument("evalIntegerSquareRoot: `one` was encrypted at a lower level than parity's; "
									"SquareRootPrecomputations must encrypt it at the top of the modulus chain (level 0)");
	}
	if (sqrt2.getLevel() < parity.getLevel()) {
		throw std::invalid_argument("evalIntegerSquareRoot: `sqrt2` was encrypted at a lower level than parity's; "
									"SquareRootPrecomputations must encrypt it at the top of the modulus chain (level 0)");
	}

	Ciphertext oneAtLevel(c.cc_);
	oneAtLevel.copy(one);
	oneAtLevel.dropToLevel(parity.getLevel(), false);

	Ciphertext sqrt2AtLevel(c.cc_);
	sqrt2AtLevel.copy(sqrt2);
	sqrt2AtLevel.dropToLevel(parity.getLevel(), false);

	Ciphertext correction(c.cc_);
	{
		Ciphertext sqrt2Term(c.cc_);
		sqrt2Term.mult(sqrt2AtLevel, parity, true);
		Ciphertext oneTerm(c.cc_);
		oneTerm.mult(oneAtLevel, parityInverse, true);
		correction.add(sqrt2Term, oneTerm);
	}
	binboot(correction, correction);

	// --------------------------------------------------------
	// y = mul_integer(y, correction, bits, bits, zslots, zslots, true)
	// --------------------------------------------------------
	{
		Ciphertext yCopy(c.cc_);
		yCopy.copy(y);
		Ciphertext correctionCopy(c.cc_);
		correctionCopy.copy(correction);
		correctionCopy.dropToLevel(std::min(correctionCopy.getLevel(), yCopy.getLevel()) - 1);
		yCopy.dropToLevel(correctionCopy.getLevel());
		evalIntegerMult(y, yCopy, correctionCopy, bits, bits, zslots, zslots, true, cc);
	}

	// --------------------------------------------------------
	// finalshift = encrypt_multi_int({bits+1,...}, bits, y->GetLevel())
	// finalshift = binboot(sub_integer(finalshift, s, bits))
	// finalshift = rot(finalshift, 1)
	// --------------------------------------------------------
	if (bitsPlusOne.getLevel() < s.getLevel()) {
		throw std::invalid_argument("evalIntegerSquareRoot: `bitsPlusOne` was encrypted at a lower level than s's; "
									"SquareRootPrecomputations must encrypt it at the top of the modulus chain (level 0)");
	}

	Ciphertext finalshift(c.cc_);
	{
		Ciphertext bitsPlusOneAtLevel(c.cc_);
		bitsPlusOneAtLevel.copy(bitsPlusOne);
		bitsPlusOneAtLevel.dropToLevel(s.getLevel(), false);

		evalIntegerSub(finalshift, bitsPlusOneAtLevel, s, bits, zslots, cc);
	}
	binboot(finalshift, finalshift);
	{
		Ciphertext rotated(c.cc_);
		rotated.rotate(finalshift, 1);
		finalshift.copy(rotated);
	}

	// --------------------------------------------------------
	// yrot = blind_rotation(y, finalshift, bits*4, zslots, bits*bits/2)
	// yrot = rot(rot(rot(rot(yrot, bits), bits), -1), -1)
	// --------------------------------------------------------
	Ciphertext yrot(c.cc_);
	blindRotation(yrot, y, finalshift, bits * 4, zslots, stride, cc);

	{
		Ciphertext r1(c.cc_);
		r1.rotate(yrot, bits);
		Ciphertext r2(c.cc_);
		r2.rotate(r1, bits);
		Ciphertext r3(c.cc_);
		r3.rotate(r2, -1);
		Ciphertext r4(c.cc_);
		r4.rotate(r3, -1);
		yrot.copy(r4);
	}

	// --------------------------------------------------------
	// Clean garbage in other slots: yrot = binboot(yrot * {1 at low bits bits of each group})
	// --------------------------------------------------------
	std::fill(mask.begin(), mask.end(), 0.0);
	for (int z = 0; z < zslots; ++z) {
		for (int j = 0; j < bits; ++j) {
			mask[j + z * stride] = 1.0;
		}
	}
	yrot.multPt(makePerSlotPlaintext(cc, cc_, mask, yrot));
	binboot(yrot, yrot);

	out.copy(yrot);
}

/*
void processArray(Ciphertext& c_processed,
  const Ciphertext& c,
  const std::vector<std::pair<int, int>>& mask_roll_pairs,
  int mask_size,
  int rep,
  lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& cc) {
	FIDESlib::CKKS::Context cc_ = c.cc_;

	if (mask_size <= 0) {
		throw std::invalid_argument("processArray: mask_size must be > 0");
	}

	if (rep <= 0) {
		throw std::invalid_argument("processArray: rep must be > 0");
	}

	Ciphertext result(c_processed.cc_);
	result.copy(c_processed);

	const int total_size = mask_size * rep;

	for (const auto& [start, roll_base] : mask_roll_pairs) {

		std::vector<double> mask(total_size, 0);

		for (int i = 0; i < rep; ++i) {
			for (int j = 0; j < 4; ++j) {
				mask[i * mask_size + start + j] = 1.0;
			}
		}

		const int shift = roll_base - start;


		Ciphertext rolled_ctxt(c.cc_);
		rolled_ctxt.rotate(c, -shift);

		std::vector<double> rolled_mask(total_size, 0);

		for (int k = 0; k < total_size; ++k) {

			int src = (k + shift) % total_size;

			if (src < 0) {
				src += total_size;
			}

			rolled_mask[k] = mask[src];
		}



		Ciphertext masked(c.cc_);



		size_t noise					 = static_cast<size_t>(rolled_ctxt.NoiseLevel);
		auto pt							 = cc->MakeCKKSPackedPlaintext(rolled_mask, noise, rolled_ctxt.getLevel(), nullptr, rolled_ctxt.slots);
		FIDESlib::CKKS::RawPlainText raw = FIDESlib::CKKS::GetRawPlainText(cc, pt);
		Plaintext maskP					 = Plaintext(cc_, raw);

		masked.multPt(rolled_ctxt, maskP, true);

		result.add(masked);

		if (result.cc.rescaleTechnique == FIDESlib::CKKS::FIXEDMANUAL) {
			result.rescale();
		}
	}

	c_processed.copy(result);
}
*/
void preprocessProcessArray(int bits,
  int bitsOriginal,
  const std::vector<std::pair<int, int>>& mask_roll_pairs,
  int slots,
  int level,
  size_t noise,
  lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& cc,
  FIDESlib::CKKS::Context& cc_,
  bool forB) {

	int mask_size = bitsOriginal * bitsOriginal / 2;

	// Assuming full reps?
	int rep = (slots * 2) / (bitsOriginal * bitsOriginal);

	if (mask_size <= 0) {
		throw std::invalid_argument("preprocessProcessArray: mask_size must be > 0");
	}

	if (rep <= 0) {
		throw std::invalid_argument("preprocessProcessArray: rep must be > 0");
	}

	if (slots <= 0) {
		throw std::invalid_argument("preprocessProcessArray: slots must be > 0");
	}

	ProcessArrayPrecomputation* precomp = nullptr;

	if (forB) {
		if (bits >= 128)
			precomp = &precomp128b;
		else if (bits >= 64)
			precomp = &precomp64b;
		else if (bits >= 32)
			precomp = &precomp32b;
		else if (bits >= 16)
			precomp = &precomp16b;
		else if (bits >= 8)
			precomp = &precomp8b;
	} else {
		if (bits >= 128)
			precomp = &precomp128;
		else if (bits >= 64)
			precomp = &precomp64;
		else if (bits >= 32)
			precomp = &precomp32;
		else if (bits >= 16)
			precomp = &precomp16;
		else if (bits >= 8)
			precomp = &precomp8;
	}

	if (precomp == nullptr) {
		throw std::invalid_argument("preprocessProcessArray: bits must be > 8");
	}

	const int total_size = mask_size * rep;

	precomp->entries.clear();
	precomp->entries.reserve(mask_roll_pairs.size());

	for (const auto& [start, roll_base] : mask_roll_pairs) {

		if (start < 0 || start + 3 >= mask_size) {
			throw std::invalid_argument("preprocessProcessArray: invalid start position");
		}

		// --------------------------------------------------------
		// Construct the original mask
		//
		// mask[i * mask_size + start + j] = 1
		// for j = 0..3
		// --------------------------------------------------------

		std::vector<double> mask(total_size, 0.0);

		for (int i = 0; i < rep; ++i) {

			for (int j = 0; j < 4; ++j) {

				const int index = i * mask_size + start + j;

				mask[index] = 1.0;
			}
		}

		// --------------------------------------------------------
		// Compute rotation
		// --------------------------------------------------------

		const int shift = roll_base - start;

		// --------------------------------------------------------
		// rolled_mask = rot(mask, -shift)
		//
		// Original code:
		//
		// vector<int> rolled_mask = rot(mask, -shift);
		// --------------------------------------------------------

		std::vector<double> rolled_mask(mask);

		rolled_mask = rotateMask(rolled_mask, -shift);

		// --------------------------------------------------------
		// Make OpenFHE plaintext
		// --------------------------------------------------------

		auto pt = cc->MakeCKKSPackedPlaintext(rolled_mask, noise, level, nullptr, slots);

		// --------------------------------------------------------
		// Convert OpenFHE plaintext -> FIDESlib plaintext
		// --------------------------------------------------------

		FIDESlib::CKKS::RawPlainText raw = FIDESlib::CKKS::GetRawPlainText(cc, pt);

		Plaintext maskP(cc_, raw);

		// --------------------------------------------------------
		// Store precomputed entry
		// --------------------------------------------------------

		ProcessArrayPrecomputation::Entry entry{ shift, std::move(maskP) };

		precomp->entries.push_back(std::move(entry));
	}
}

void preprocessChebyshevMultiplication(std::vector<std::vector<double>> coeffs, lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& cc, Ciphertext& c) {
	cacheChebyshev4BitsMultiplier = evalChebyshevSeriesPSBatchPrecompute(cc, c, coeffs, -1, 1);

	coeffs4BitsMultiplier = coeffs;
}

void processArray(Ciphertext& out, const Ciphertext& c, const ProcessArrayPrecomputation& precomp) {
	if (precomp.entries.size() == 0) {
		std::cerr << "No precomputations found for multiplications! Call ProcessArrayPrecomputations" << std::endl;
	}

	for (const auto& entry : precomp.entries) {

		// rolled_ctxt = rot(c, -shift)
		Ciphertext rolled_ctxt(c.cc_);

		rolled_ctxt.rotate(c, -entry.shift);

		// masked = rolled_ctxt * precomputed_mask
		Ciphertext masked(c.cc_);

		masked.multPt(rolled_ctxt, entry.mask, true);

		// out += masked
		out.add(masked);
	}
}

void multiplier4bits(Ciphertext& result, Ciphertext& ctxtA, Ciphertext& ctxtB, int repetitions, lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& cc) {
	FIDESlib::CKKS::Context cc_ = ctxtA.cc_;

	result.mult(ctxtA, ctxtB, true);

	// result = result - 1
	std::vector<double> minusOne(result.slots, -1.0);

	size_t noise = static_cast<size_t>(result.NoiseLevel);

	auto pt = cc->MakeCKKSPackedPlaintext(minusOne, noise, result.getLevel(), nullptr, result.slots);

	FIDESlib::CKKS::RawPlainText raw = FIDESlib::CKKS::GetRawPlainText(cc, pt);

	Plaintext minusOnePt(result.cc_, raw);

	result.addPt(minusOnePt);

	// QUA RESULT è GIUSTO

	evalChebyshevSeriesPSBatchApply(cc, result, cacheChebyshev4BitsMultiplier, coeffs4BitsMultiplier, -1, 1);

	// evalChebyshevSeriesPSBatchApply(cc, result, precomp4bits, coeffs, -1, 1);
	// evalChebyshevSeriesPSBatchRepeated(cc, result, coeffs, -1, 1);
	// evalChebyshevSeries(result, coeffs4BitsMultiplier[0], -1, 1);

	if (result.NoiseLevel == 2) {
		result.dropToLevel(5, false);
	} else {
		result.dropToLevel(4, false);
	}

	BootstrapStCFirstBits(result, result.slots, false);
}

void cleanAndReduce(Ciphertext& out, const Ciphertext& c) {
	FIDESlib::CKKS::Context& cc_ = c.cc_;

	// sqC := c^2
	Ciphertext sqC(cc_);
	sqC.square(c, false);
	if (sqC.NoiseLevel == 2)
		sqC.rescale();

	// shifted := (c - 2)^2
	Ciphertext shifted(cc_);
	shifted.copy(c);
	shifted.addScalar(-2.0);
	shifted.square(false);
	if (shifted.NoiseLevel == 2)
		shifted.rescale();

	// out := sqC * shifted
	out.mult(sqC, shifted, false);
}

void clean(Ciphertext& out, const Ciphertext& c) {
	FIDESlib::CKKS::Context& cc_ = c.cc_;

	// sq := c^2
	Ciphertext sq(cc_);
	sq.square(c, false);
	if (sq.NoiseLevel == 2)
		sq.rescale();

	// t1 := c * (-2)
	Ciphertext t1(cc_);
	t1.multScalar(c, -2.0, false);
	if (t1.NoiseLevel == 2)
		t1.rescale();

	// termA := sq * t1
	Ciphertext termA(cc_);
	termA.mult(sq, t1, false);

	// termB := sq * 3
	Ciphertext termB(cc_);
	termB.multScalar(sq, 3.0, false);

	// out := termA + termB
	out.add(termA, termB);
}

void mod2Shallow(Ciphertext& out, const Ciphertext& c) {
	FIDESlib::CKKS::Context& cc_ = c.cc_;

	// doubled := c * 2
	Ciphertext doubled(cc_);
	doubled.multScalar(c, 2.0, false);

	// sq := c^2
	Ciphertext sq(cc_);
	sq.square(c, false);
	if (sq.NoiseLevel == 2)
		sq.rescale();

	// out := doubled - sq
	out.sub(doubled, sq);
}

void majorityBit(Ciphertext& out, const Ciphertext& a, const Ciphertext& b, const Ciphertext& c) {
	FIDESlib::CKKS::Context& cc_ = a.cc_;

	// total := a + b + c
	// a is right
	// b seems right

	Ciphertext total(cc_);
	total.add(a, b);
	total.add(c);

	// total is right

	// sq := total^2
	Ciphertext sq(cc_);
	sq.square(total, true);
	if (sq.NoiseLevel == 2)
		sq.rescale();

	// t1 := total * (-1/3)
	Ciphertext t1(cc_);
	t1.multScalar(total, -1.0 / 3.0, true);
	if (t1.NoiseLevel == 2)
		t1.rescale();

	// termA := t1 * sq
	Ciphertext termA(cc_);
	termA.mult(t1, sq, false);

	// termB := sq * 3/2
	Ciphertext termB(cc_);
	termB.multScalar(sq, 3.0 / 2.0, true);

	// termC := total * (-7/6)
	Ciphertext termC(cc_);
	termC.multScalar(total, -7.0 / 6.0, true);

	// out := termA + termB + termC
	Ciphertext ab(cc_);
	ab.add(termA, termB);
	out.add(ab, termC);
}

void csa3(Ciphertext& S, Ciphertext& C, const Ciphertext& a, const Ciphertext& b, const Ciphertext& c) {
	FIDESlib::CKKS::Context& cc_ = a.cc_;

	// S := mod2Shallow(a + b)
	Ciphertext ab(cc_);
	ab.add(a, b);
	mod2Shallow(S, ab);

	// S := mod2Shallow(S + c)
	Ciphertext sPlusC(cc_);
	sPlusC.add(S, c);
	mod2Shallow(S, sPlusC);

	majorityBit(C, a, b, c);
}

void csa4(Ciphertext& out, const Ciphertext& a, const Ciphertext& b, const Ciphertext& c, const Ciphertext& d, int bits) {
	// ------------------------------------------------------------
	// First CSA:
	//
	// (s1, c1) = csa3(a, b, c)
	// ------------------------------------------------------------

	Ciphertext s1(a.cc_);
	Ciphertext c1(a.cc_);

	csa3(s1, c1, a, b, c);

	// s1 è giusto
	// c1 è giusto

	// c1 = rot(c1, -1)
	Ciphertext c1_rot(a.cc_);
	c1_rot.rotate(c1, -1);

	// c1 rot is broken it seems

	// ------------------------------------------------------------
	// Second CSA:
	//
	// (s2, c2) = csa3(s1, c1_rot, d)
	// ------------------------------------------------------------

	Ciphertext s2(a.cc_);
	Ciphertext c2(a.cc_);

	// all zeros????
	// out.copy(d);
	// return;

	csa3(s2, c2, s1, c1_rot, d);

	// s2 è sbagliato

	// c2 = rot(c2, -1)
	Ciphertext c2_rot(a.cc_);

	c2_rot.rotate(c2, -1);

	// ------------------------------------------------------------
	// result = add_integer(s2, c2_rot, bits, false)
	// ------------------------------------------------------------

	evalIntegerAdd(s2, c2_rot, bits);

	out.copy(s2);
}

void bintodec(lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& cc, Ciphertext& out, const Ciphertext& c, int repetitions) {
	FIDESlib::CKKS::Context& cc_ = c.cc_;

	// Build the {1,2,4,8,0,0,0,0}-repeated mask, encoded at c's level, and
	// the second {sqrt(1/(225/2)), 0,0,0,0,0,0,0}-repeated rescaling mask,
	// mirroring OpenFHE's bintodec exactly (including the folding rotation
	// pattern +1,+2 then -1,-2,-4).
	std::vector<double> mask1;
	mask1.reserve(static_cast<size_t>(repetitions) * 8);
	for (int i = 0; i < repetitions; i++) {
		mask1.insert(mask1.end(), { 1.0, 2.0, 4.0, 8.0, 0.0, 0.0, 0.0, 0.0 });
	}

	std::vector<double> mask2;
	mask2.reserve(static_cast<size_t>(repetitions) * 8);
	double rescaleFactor = std::sqrt(1.0 / (225.0 / 2.0));
	for (int i = 0; i < repetitions; i++) {
		mask2.insert(mask2.end(), { rescaleFactor, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 });
	}

	// Same level-convention caveat as ApproxModEvalBatch.cu's
	// makePerSlotPlaintext: FIDESlib's core getLevel() counts DOWN from
	// cc.L, while MakeCKKSPackedPlaintext's `level` counts UP from 0, and
	// noiseScaleDeg must match the target ciphertext's real NoiseLevel.
	auto makeMaskPt = [&](const std::vector<double>& values, const Ciphertext& like) -> Plaintext {
		uint32_t openfheLevel			 = static_cast<uint32_t>(like.cc.L - like.getLevel());
		size_t noiseScaleDeg			 = static_cast<size_t>(like.NoiseLevel);
		auto pt							 = cc->MakeCKKSPackedPlaintext(values,
		  /*noiseScaleDeg=*/noiseScaleDeg,
		  /*level=*/openfheLevel,
		  nullptr,
		  /*slots=*/like.slots);
		FIDESlib::CKKS::RawPlainText raw = FIDESlib::CKKS::GetRawPlainText(cc, pt);
		return Plaintext(cc_, raw);
	};

	// res := c * mask1  (weights 1,2,4,8 at slots 0..3 of each group)
	Ciphertext res(cc_);
	Plaintext maskPt1 = makeMaskPt(mask1, c);
	res.multPt(c, maskPt1, true);

	// res := res + rot(res, 1) + rot(res, 2)  -- sums the 4 weighted bits
	// into slot 0 of each group.
	Ciphertext rot1(cc_);
	rot1.rotate(res, 1);
	res.add(rot1);
	Ciphertext rot2(cc_);
	rot2.rotate(res, 2);
	res.add(rot2);

	// res := res * mask2  (keeps only slot 0 of each group, rescaled)
	Plaintext maskPt2 = makeMaskPt(mask2, res);
	res.multPt(maskPt2, true);

	// res := res + rot(res,-1) + rot(res,-2) + rot(res,-4)  -- broadcasts
	// the decoded value back across the low 4 slots of each group.
	Ciphertext rotm1(cc_);
	rotm1.rotate(res, -1);
	res.add(rotm1);
	Ciphertext rotm2(cc_);
	rotm2.rotate(res, -2);
	res.add(rotm2);
	Ciphertext rotm4(cc_);
	rotm4.rotate(res, -4);
	res.add(rotm4);

	out.copy(res);
}

std::vector<double> rotateMask(const std::vector<double>& mask, int shift) {
	int n = mask.size();
	shift = shift % n;
	if (shift == 0)
		return mask;

	std::vector<double> result = mask;

	if (shift > 0) {
		// Positive shift → left rotation
		std::rotate(result.begin(), result.begin() + shift, result.end());
	} else {
		// Negative shift → right rotation
		shift = -shift;
		std::rotate(result.rbegin(), result.rbegin() + shift, result.rend());
	}

	return result;
}

} // namespace FIDESlib::CKKS