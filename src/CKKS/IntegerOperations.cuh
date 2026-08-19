#ifndef GPUCKKS_INTEGEROPS_CUH
#define GPUCKKS_INTEGEROPS_CUH

#include "CKKS/Context.cuh"
#include "CKKS/Plaintext.cuh"
#include "CKKS/forwardDefs.cuh"
#include "CKKS/ApproxModEvalBatch.cuh"
#include "CKKS/Bootstrap.cuh"

#include <utility>
#include <vector>

namespace FIDESlib::CKKS {

void evalIntegerAdd(Ciphertext& ctxtA, Ciphertext& ctxtB, int bits);

// Can be heavily optimized by precomputing masks
void processArray(Ciphertext& c_processed,
  const Ciphertext& c,
  const std::vector<std::pair<int, int>>& mask_roll_pairs,
  int mask_size,
  int rep,
  lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& cc,
  FIDESlib::CKKS::Context& cc_);

void multiplier4bits(Ciphertext& ctxtA, Ciphertext& ctxtB, int repetitions, std::vector<std::vector<double>> coeffs, lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& cc, FIDESlib::CKKS::Context& cc_);

} // namespace FIDESlib::CKKS

#endif // GPUCKKS_INTEGEROPS_CUH