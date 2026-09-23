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

// Livello OpenFHE a cui evalIntegerMult si aspetta i suoi operandi: e' il
// `level = 12` di ProcessArrayPrecomputations (maschere di processArray) e
// ProcessMultiplications registra la PSBatch del moltiplicatore a 4 bit a
// 12 + 3 = 15. Se cambi uno dei due, cambia anche questa costante.
constexpr int kIntegerOpsOpenFHELevel = 12;

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

// SquareRootIntegerLUTs: the two repeated-Chebyshev LUTs square_root_integer
// needs. bitLengthDecompose plays the SAME role as DivIntegerLUTs'
// bitLengthDecompose (decomposes a normalized bit-length hint into a 7-bit
// binary index for blind_rotation), but is kept as its OWN cache here --
// evalIntegerSquareRoot precomputes/replays it against its own `s`
// ciphertext, which will generally sit at a different level/NoiseLevel than
// div_integer's `s`, and the lazy-rebuild check (see evalIntegerDivision's
// doc comment) keys off exactly that model level/NoiseLevel match. newtonSeed
// is the LUT-SQUARE-ROOT-<bits>-BITS-<i> Newton-Raphson seed, analogous to
// DivIntegerLUTs::reciprocalHint but with the bits-specific column
// count/patching square root uses (see preprocessSquareRootLUTs).
struct SquareRootIntegerLUTs {
	ChebyshevRepeatedLUT bitLengthDecompose;
	ChebyshevRepeatedLUT newtonSeed;
};

extern SquareRootIntegerLUTs lutsSquareRoot;

// Cache LUT separata per la divisione ct/ct di evalUniswapV3 (vedi il
// commento sulla definizione in IntegerOperations.cu).
extern DivIntegerLUTs lutsDivUniswap;

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

// Copia `src` in `dst` e la porta al livello OpenFHE kIntegerOpsOpenFHELevel
// (NoiseLevel 1), quello che evalIntegerMult si aspetta. Va bene sia per
// input freschi (cifrati a livello OpenFHE <= 12) sia per uscite di binboot.
// Lancia std::invalid_argument se `src` e' gia' piu' profondo.
void prepareIntegerOperand(Ciphertext& dst, const Ciphertext& src);

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

// Come evalIntegerAdd ma con carry in ingresso = 1 (CPU: add_integer(...,
// carry_in = true)). `layoutBits` e' la dimensione degli interi nel layout
// dei dati (stride = layoutBits^2 / 2), che puo' differire da `bits` quando si
// somma su una finestra piu' corta/lunga del layout.
void evalIntegerAddCarryIn(Ciphertext& ctxtA, Ciphertext& ctxtB, int bits, int layoutBits, lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& cc);

// evalIntegerSub: ciphertext - ciphertext subtraction, translated from
// CKKSController::sub_integer(const Ctxt&, const Ctxt&, int, bool). Computes
// a + ~b (plaintext one's-complement of b, masked to the low `bits+1` bits
// of each bits*bits/2-sized group, matching the CPU's `ones` mask exactly)
// via evalIntegerAdd, then re-masks the low `bits` bits of the result to
// drop the carry-out slot. `cleanFirst` mirrors the CPU's `clean_first` flag
// forwarded to add_integer (see evalIntegerAdd's `clean_and_reduce(add(a,b))`
// vs `square(sub(a,b))` branch on the CPU -- NOTE: evalIntegerAdd above only
// ever implements the `clean_first=false` path; `cleanFirst=true` is
// accepted for interface parity with the CPU signature but currently
// asserts, since no caller in this port needs it yet).
//
// `carryIn`: false -> a + ~b = a - b - 1 (comportamento storico, usato da
// evalIntegerSquareRoot come il CPU `sub_integer(..., false, false)`);
// true -> a - b esatto (CPU sub_integer con carryin = true).
void evalIntegerSub(Ciphertext& out,
  const Ciphertext& a,
  const Ciphertext& b,
  int bits,
  int zslots,
  lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& cc,
  bool cleanFirst = false,
  bool carryIn	  = false);

// ----------------------------------------------------------------------
// Divisione intera con divisore in chiaro (CPU:
// CKKSController::div_integer(const Ctxt&, uint128_t, int, int)).
// ----------------------------------------------------------------------

int bitWidthU128(__uint128_t v);

// Bit (LSB-first) della parte bassa R_low del reciproco
// R = ceil(2^(bits + bit_width(den)) / den), replicati sui primi `zslots`
// gruppi di stride bits*bits/2, in un vettore lungo `slots`. Il bit alto di R
// (posizione `bits`, sempre 1) non e' incluso: evalIntegerDivisionByPlain lo
// aggiunge come num << bits, esattamente come il CPU. Restituisce anche
// bit_width(den) in `denBitLength`. Lancia per den potenza di 2.
std::vector<double> integerReciprocalMask(__uint128_t den, int bits, int zslots, int slots, int& denBitLength);

// `reciprocal`: cifratura di integerReciprocalMask(den, ...) fatta al livello
// 0 (vedi CryptoContextImpl::PlainDivisionPrecomputations). `num` puo' essere
// fresco o uscire da un binboot. L'uscita e' al livello OpenFHE 12 con
// NoiseLevel 1, quindi riutilizzabile direttamente in evalIntegerMult.
// Rotation keys necessarie: -bits e bits + denBitLength.
void evalIntegerDivisionByPlain(Ciphertext& out,
  const Ciphertext& num,
  const Ciphertext& reciprocal,
  int denBitLength,
  int bits,
  int zslots,
  lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& cc);

// ----------------------------------------------------------------------
// Esempio Uniswap v3 (main.cpp: experiment_uniswap_v3) interamente su GPU.
// ----------------------------------------------------------------------

struct PlainDivisorGPU {
	const Ciphertext* reciprocal = nullptr;
	int bitLength				 = 0;
};

struct UniswapV3GPUInputs {
	const Ciphertext* g_num_inv_L_fx = nullptr;
	const Ciphertext* user_amount	 = nullptr;
	const Ciphertext* inv_sqrt_P0_fx = nullptr;
	const Ciphertext* numerator		 = nullptr;
	const Ciphertext* m_fx			 = nullptr;
	const Ciphertext* g_den			 = nullptr;
};

struct UniswapV3GPUConstants {
	int bits	  = 128;
	int zslots	  = 1;
	int shiftDown = 21; // 1e21 = 2^21 * 5^21: la parte 2^21 e' un >> 21
	int shiftUp	  = 32; // X_post_fx << 32
	int addBits	  = 64; // u_fx = add_integer(term2_fx, inv_sqrt_P0_fx, 64)

	PlainDivisorGPU divPrec1; // 5^10 = 9765625
	PlainDivisorGPU divPrec2; // 5^11 = 48828125
	PlainDivisorGPU divFee;	  // g_num = 997

	// Divisione ct/ct a `bits` bit (DivIntegerPrecomputations)
	const Ciphertext* divOne										= nullptr;
	const std::vector<std::vector<double>>* divBitLengthCoeffs		= nullptr;
	const std::vector<std::vector<double>>* divReciprocalCoeffs	= nullptr;
};

// Copie facoltative dei risultati intermedi (restano sul device; si possono
// decifrare DOPO la chiamata per confrontarli con le stampe del CPU).
struct UniswapV3GPUTrace {
	Ciphertext* term2_fx  = nullptr;
	Ciphertext* u_fx	  = nullptr;
	Ciphertext* X_post_fx = nullptr;
	Ciphertext* diff_fx	  = nullptr;
};

// `out` = amount_fx. `luts`: passare lutsDivUniswap.
void evalUniswapV3(Ciphertext& out,
  const UniswapV3GPUInputs& in,
  const UniswapV3GPUConstants& k,
  DivIntegerLUTs& luts,
  lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& cc,
  UniswapV3GPUTrace* trace = nullptr);

// square_root_integer: ciphertext integer square root, translated from
// CKKSController::square_root_integer(const Ctxt&, int, int).
//
// preprocessSquareRootLUTs is the sqrt analogue of preprocessDivIntegerLUTs:
// it validates and stashes the raw coefficient columns needed for the two
// repeated-Chebyshev LUTs (bitLengthDecompose, newtonSeed) into
// `SquareRootIntegerLUTs`. Just like div_integer, evalIntegerSquareRoot does
// NOT call this ahead of time against a hand-picked model ciphertext --
// the actual PSBatch precompute happens lazily inside evalIntegerSquareRoot,
// on first use (or whenever the cached LUT's level/NoiseLevel no longer
// matches), directly against the internal `s`/`idx` ciphertexts themselves.
// This mirrors evalIntegerDivision's lazy-precompute strategy exactly, and
// for the exact same reason: a plaintext cache recorded by
// evalChebyshevSeriesPSBatchPrecompute is only safe to replay against a
// ciphertext with the SAME level/NoiseLevel it was recorded against, and
// there's no way to guarantee a hand-picked model matches what `s`/`idx`
// happen to be at the point they're consumed -- that depends on `c`'s own
// level and everything inverseBitLength/blindRotation/evalIntegerMult do
// upstream. `preprocessSquareRootLUTs` is still exposed here in case a
// caller wants to warm the cache ahead of time with a ciphertext they are
// certain will match, but for most callers just calling
// evalIntegerSquareRoot directly and letting it self-warm on first use is
// the safe default -- same guidance as evalIntegerDivision.
//
// `bitLengthCoeffs` are the p1..p7-norm-247-LUT-DIVISION.txt columns (+
// (bits-7) garbage copies of p1) -- the SAME raw file layout div_integer
// uses for its own bitLengthDecompose LUT (the CPU's square_root_integer
// reloads these files itself rather than sharing div_integer's cache, and
// this port preserves that -- the LUT cache objects are kept distinct, see
// SquareRootIntegerLUTs). `newtonSeedCoeffs` are the raw, UN-patched
// LUT-SQUARE-ROOT-<bits>-BITS-<i>.txt columns plus garbage padding, laid out
// exactly as the CPU's square_root_integer reads them (bits-dependent column
// count and padding file -- see the `bits == 64` / `bits == 128` / default
// branches in the CPU reference); the bits==64/128 in-place column
// reassignment the CPU does right after loading (`coeffs[0] = coeffs[12]`
// etc.) must ALREADY be applied by the caller before calling this function
// or evalIntegerSquareRoot, since that patching only depends on `bits`, not
// on any ciphertext.
void preprocessSquareRootLUTs(SquareRootIntegerLUTs& luts,
  int bits,
  int zslots,
  lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& cc,
  Ciphertext& like,
  const std::vector<std::vector<double>>& bitLengthCoeffs,
  const std::vector<std::vector<double>>& newtonSeedCoeffs);

// `one`/`sqrt2`/`bitsPlusOne`: genuine (non-trivial) encryptions of the
// per-group constants square_root_integer's tail needs -- mirroring how
// `one` is supplied to evalIntegerDivision, since FIDESlib::CKKS::Ciphertext
// has no key material in scope to encrypt these itself:
//   - `one`:  ONE_FP = 1 << (bits-1), bit-packed LSB-first per group (the
//     CPU's `encrypt_multi_int({ONE_FP,...}, bits, parity->GetLevel())`).
//   - `sqrt2`: SQRT2_FP = round(sqrt(2) * (1 << (bits-1))), same packing
//     (`encrypt_multi_int({SQRT2_FP,...}, bits, parity->GetLevel())`). The
//     128-bit case uses the CPU's literal hex constant
//     0xb504f333f9de6484597d89b3754abe9f instead of the floating-point
//     round-trip -- callers targeting bits==128 must encrypt that exact
//     128-bit value.
//   - `bitsPlusOne`: the constant `bits + 1`, bit-packed the same way (the
//     CPU's `encrypt_multi_int({bits+1,...}, bits, y->GetLevel())`).
// All three must be encrypted at a level at least as fresh as the point
// they're consumed at (evalIntegerSquareRoot only ever drops them down, via
// dropToLevel, and throws if the supplied level is too shallow) -- the
// caller (see CryptoContextImpl<DCRTPoly>::SquareRootPrecomputations)
// encrypts them at the top of the modulus chain (level 0) for exactly this
// reason, same as DivIntegerPrecomputations does for its `one`.
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
  lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& cc);

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