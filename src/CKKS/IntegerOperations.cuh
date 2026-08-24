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

extern PSBatchPrecompute cacheChebyshev4BitsMultiplier;
std::vector<double> coeffs4BitsMultiplier;

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
void multiplier4bits(Ciphertext& result, Ciphertext& ctxtA, Ciphertext& ctxtB, int repetitions, std::vector<std::vector<double>> coeffs, std::shared_ptr<void> precomp4bits, lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& cc);

void preprocessProcessArray(int bits,
  const std::vector<std::pair<int, int>>& mask_roll_pairs,
  int slots,
  int level,
  size_t noise,
  lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& cc,
  FIDESlib::CKKS::Context& cc_,
  bool forB = false);

void preprocessChebyshevMultiplication(std::vector<std::vector<double>> coeffs, lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& cc);

void processArray(Ciphertext& out, const Ciphertext& c, const ProcessArrayPrecomputation& precomp);

std::vector<double> rotateMask(const std::vector<double>& mask, int shift);

} // namespace FIDESlib::CKKS

#endif // GPUCKKS_INTEGEROPS_CUH