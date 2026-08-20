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
	std::vector<int> shifts;
	std::vector<Plaintext> masks;
};

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
  std::vector<std::vector<double>>& coeffsFor4Bits,
  lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& cc);

void cleanAndReduce(Ciphertext& out, const Ciphertext& c);
void clean(Ciphertext& out, const Ciphertext& c);
void mod2Shallow(Ciphertext& out, const Ciphertext& c);
void majorityBit(Ciphertext& out, const Ciphertext& a, const Ciphertext& b, const Ciphertext& c);
void csa3(Ciphertext& S, Ciphertext& C, const Ciphertext& a, const Ciphertext& b, const Ciphertext& c);
void csa4(Ciphertext& out, const Ciphertext& a, const Ciphertext& b, const Ciphertext& c, const Ciphertext& d, int bits);

void bintodec(lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& cc, Ciphertext& out, const Ciphertext& c, int repetitions);
// Can be heavily optimized by precomputing masks
// void processArray(Ciphertext& c_processed, const Ciphertext& c, const std::vector<std::pair<int, int>>& mask_roll_pairs, int mask_size, int rep, lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& cc);
void multiplier4bits(Ciphertext& result, Ciphertext& ctxtA, Ciphertext& ctxtB, int repetitions, std::vector<std::vector<double>> coeffs, lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& cc);

void preprocessProcessArray(int bits,
  const std::vector<std::pair<int, int>>& mask_roll_pairs,
  int mask_size,
  int rep,
  int slots,
  int level,
  size_t noise,
  lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& cc,
  FIDESlib::CKKS::Context& cc_,
  bool forB = false);

void processArray(Ciphertext& out, const Ciphertext& c, const ProcessArrayPrecomputation& precomp);

std::vector<double> rotateMask(const std::vector<double>& mask, int shift);

} // namespace FIDESlib::CKKS

#endif // GPUCKKS_INTEGEROPS_CUH