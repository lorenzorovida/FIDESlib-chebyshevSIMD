#ifndef GPUCKKS_INTEGEROPS_CUH
#define GPUCKKS_INTEGEROPS_CUH

#include "CKKS/ApproxModEval.cuh"
#include "CKKS/ApproxModEvalBatch.cuh"
#include "CKKS/Bootstrap.cuh"
#include "CKKS/Context.cuh"
#include "CKKS/Plaintext.cuh"
#include "CKKS/forwardDefs.cuh"

#include <utility>
#include <vector>

namespace FIDESlib::CKKS {

struct ProcessArrayPrecomputation {
	struct Entry {
		int shift;
		Plaintext mask;
	};

	std::vector<Entry> entries;
};

extern ProcessArrayPrecomputation precomp8;
extern ProcessArrayPrecomputation precomp16;
extern ProcessArrayPrecomputation precomp32;
extern ProcessArrayPrecomputation precomp64;
extern ProcessArrayPrecomputation precomp128;

extern ProcessArrayPrecomputation precomp8b;
extern ProcessArrayPrecomputation precomp16b;
extern ProcessArrayPrecomputation precomp32b;
extern ProcessArrayPrecomputation precomp64b;
extern ProcessArrayPrecomputation precomp128b;

extern std::shared_ptr<PSBatchPrecompute> cacheChebyshev4BitsMultiplier;
extern std::vector<std::vector<double>> coeffs4BitsMultiplier;
// Level/NoiseLevel the ciphertext used to build cacheChebyshev4BitsMultiplier
// had at the time (see preprocessChebyshevMultiplication / multiplier4bits).
// Tracked so multiplier4bits can detect a mismatch and rebuild instead of
// silently replaying a plaintext cache recorded for a different level/noise
// (see the long comment on evalIntegerDivision for why that mismatch is
// unsafe -- it applies here identically).
extern int cacheChebyshev4BitsMultiplierModelLevel;
extern int cacheChebyshev4BitsMultiplierModelNoiseLevel;

struct ChebyshevRepeatedLUT {
	std::shared_ptr<PSBatchPrecompute> precomp;
	std::vector<std::vector<double>> coeffs;
	int repeat = 0;
	int a = -1; // input-range lower bound this LUT was built with (see preprocessChebyshevRepeated)
	int b = 1;	// input-range upper bound this LUT was built with
	int modelLevel	   = -1; // level of the ciphertext this LUT's plaintext cache was recorded against
	int modelNoiseLevel = -1; // NoiseLevel of that same ciphertext -- both must match on every replay
};

struct DivIntegerLUTs {
	ChebyshevRepeatedLUT bitLengthDecompose; // p1..p7-norm-247-LUT-DIVISION (+ garbage)
	ChebyshevRepeatedLUT reciprocalHint;	 // LUT-DIVISION-<bits>-bits-<i> (+ garbage)
};

extern DivIntegerLUTs lutsDiv;

// ----------------------------------------------------------------------
// Repeated-Chebyshev-LUT precomputation, mirroring OpenFHE's
// EvalChebyshevSeriesPSBatchRepeated(ctxt, coeffs, a, b, repeat).
//
// The CPU code loads one polynomial (a set of Chebyshev coefficients) per
// LUT "column" from disk every time div_integer/square_root_integer run.
// On the GPU side we precompute the PSBatchPrecompute (baby/giant-step
// power basis) once for a given ciphertext shape + coeff set and cache it,
// exactly like cacheChebyshev4BitsMultiplier does for the 4-bit multiplier
// LUT. Callers build the coeffs (bits+something columns, padded with
// garbage columns up to a full repeat-period) then call
// preprocessChebyshevRepeated(...) once, and evalChebyshevRepeatedApply(...)
// on every ciphertext that shares that shape.
// ----------------------------------------------------------------------


void preprocessChebyshevRepeated(ChebyshevRepeatedLUT& lut, lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& cc, Ciphertext& c, std::vector<std::vector<double>> coeffs, int a, int b);
void preprocessIntegerMult(int bits, int repetitions, int slots, int level, size_t noise, lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& cc, FIDESlib::CKKS::Context& cc_);

void evalChebyshevRepeatedApply(lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& cc, Ciphertext& c, const ChebyshevRepeatedLUT& lut);

// binboot: FIDESlib's BootstrapStCFirstBits, wrapped as a standalone
// function (mirrors CKKSController::binboot on the CPU side, which just
// forwards to EvalBootstrapStCFirstBits). Drops to the level expected by
// the bootstrap circuit first, exactly like the inline call sites already
// in this file (see evalIntegerEqual / multiplier4bits / evalIntegerMult).
void binboot(Ciphertext& out, const Ciphertext& c);

// inverse_bit_length / blind_rotation: building blocks for div_integer and
// square_root_integer. See CKKSController::inverse_bit_length /
// CKKSController::blind_rotation on the CPU side for the reference
// algorithm; the GPU versions below are a direct, per-op translation.
void inverseBitLength(Ciphertext& out, const Ciphertext& a, int bits, int zslots, lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& cc);

void blindRotation(Ciphertext& out, const Ciphertext& a, const Ciphertext& index, int bits, int zslots, int stride, lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& cc);

void binaryOr(Ciphertext& out, const Ciphertext& a, const Ciphertext& b);

// div_integer: ciphertext / ciphertext division, translated from
// CKKSController::div_integer(const Ctxt&, const Ctxt&, int, int).
//
// preprocessDivIntegerLUTs is an internal helper -- evalIntegerDivision
// calls it lazily, on `s`/`x` themselves, at first use (see the comment
// above evalIntegerDivision for why). It's still declared here in case a
// caller wants to warm the cache ahead of time with a ciphertext they are
// certain will match the real level/NoiseLevel s/x end up at -- but for
// most callers, just calling evalIntegerDivision directly and letting it
// self-warm on first use is the safe default.
void preprocessDivIntegerLUTs(DivIntegerLUTs& luts,
  int bits,
  int zslots,
  lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& cc,
  Ciphertext& like,
  const std::vector<std::vector<double>>& bitLengthCoeffs,
  const std::vector<std::vector<double>>& reciprocalCoeffs);

// `one`: a genuine (non-trivial) encryption of the constant mask
// {1 at slot 0 of every bits*bits/2-sized group, 0 elsewhere}, at a level
// deep enough to survive the two's-complement "+1" step inside the Newton-
// Raphson loop (see the CPU's `encrypt(mask, term->GetLevel())` call in
// div_integer). FIDESlib::CKKS::Ciphertext has no key material in scope to
// encrypt this itself, so the caller builds it once -- via
// CryptoContextImpl<DCRTPoly>::Encrypt(pt, pk) (see
// CryptoContextImpl<DCRTPoly>::DivIntegerPrecomputations).
//
// `luts` is precomputed LAZILY, on first use, INSIDE this function -- not
// ahead of time by the caller. Why: evalChebyshevSeriesPSBatchPrecompute
// records a plaintext cache by running the real Chebyshev-PS algorithm on a
// copy of whatever "model" ciphertext it's given, and that recording is only
// valid to replay (via evalChebyshevSeriesPSBatchApply) against a ciphertext
// that has the EXACT SAME level and NoiseLevel the model had -- otherwise
// the cached plaintexts don't match the real ciphertext's RNS/limb layout
// and the GPU kernel that consumes them reads out of bounds (this is not
// hypothetical: it's what was crashing before this was made lazy). There is
// no way to guarantee a hand-picked model ciphertext, built ahead of time by
// the caller, will coincidentally match the level/NoiseLevel that the
// internal `s`/`x` ciphertexts happen to have at the point they're used
// below -- those depend on `num`/`den`'s own level and everything
// inverseBitLength/blindRotation/evalIntegerMult do to derive them. So we
// precompute the LUT on first call using `s`/`x` THEMSELVES as the model,
// right before they'd otherwise be consumed by evalChebyshevRepeatedApply,
// guaranteeing an exact level/NoiseLevel match. `luts` is cached afterwards
// (see the DivIntegerLUTs global) and reused on subsequent calls with the
// same (bits, zslots) -- it is only rebuilt if empty.
//
// `bitLengthCoeffs`/`reciprocalCoeffs` are the raw coefficient columns
// (same layout the CPU reads from p1..p7-norm-247-LUT-DIVISION.txt /
// LUT-DIVISION-<bits>-bits-<i>.txt, see preprocessDivIntegerLUTs) needed to
// perform that first-call precompute; on later calls (luts already
// populated) they're unused and may be passed empty.
void evalIntegerDivision(Ciphertext& out, const Ciphertext& num, const Ciphertext& den, int bits, int zslots, DivIntegerLUTs& luts, const Ciphertext& one,
  const std::vector<std::vector<double>>& bitLengthCoeffs, const std::vector<std::vector<double>>& reciprocalCoeffs, lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& cc);
void evalIntegerAdd(Ciphertext& ctxtA, Ciphertext& ctxtB, int bits);
void evalIntegerEqual(Ciphertext& ctxtA, Ciphertext& ctxtB, int bits, int zslots, std::vector<double> coeffsSinc, lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& cc, int depth);
void evalIntegerMult(Ciphertext& out,
  const Ciphertext& a,
  const Ciphertext& b,
  int bits,
  int bits_original,
  int repetitions,
  int repetitions_original,
  bool overflow,
  lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& cc);

void cleanAndReduce(Ciphertext& out, const Ciphertext& c);
void clean(Ciphertext& out, const Ciphertext& c);
void mod2Shallow(Ciphertext& out, const Ciphertext& c);
void majorityBit(Ciphertext& out, const Ciphertext& a, const Ciphertext& b, const Ciphertext& c);
void csa3(Ciphertext& S, Ciphertext& C, const Ciphertext& a, const Ciphertext& b, const Ciphertext& c);
void csa4(Ciphertext& out, const Ciphertext& a, const Ciphertext& b, const Ciphertext& c, const Ciphertext& d, int bits);

void bintodec(lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& cc, Ciphertext& out, const Ciphertext& c, int repetitions);
void multiplier4bits(Ciphertext& result, Ciphertext& ctxtA, Ciphertext& ctxtB, int repetitions, lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& cc);

void preprocessProcessArray(int bits,
  int bitsOriginal,
  const std::vector<std::pair<int, int>>& mask_roll_pairs,
  int slots,
  int level,
  size_t noise,
  lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& cc,
  FIDESlib::CKKS::Context& cc_,
  bool forB = false);

void preprocessChebyshevMultiplication(std::vector<std::vector<double>> coeffs, lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& cc, Ciphertext& c);
void preprocessChebyshevMultiplication2(lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& cc, Ciphertext& c, PSBatchPrecompute precomp);

void processArray(Ciphertext& out, const Ciphertext& c, const ProcessArrayPrecomputation& precomp);

std::vector<double> rotateMask(const std::vector<double>& mask, int shift);

} // namespace FIDESlib::CKKS

#endif // GPUCKKS_INTEGEROPS_CUH