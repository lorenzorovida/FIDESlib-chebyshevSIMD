#ifndef GPUCKKS_INTEGEROPS_CUH
#define GPUCKKS_INTEGEROPS_CUH

#include "CKKS/forwardDefs.cuh"
#include "CKKS/Plaintext.cuh"
#include "CKKS/Context.cuh"

#include <utility>
#include <vector>

namespace FIDESlib::CKKS {

void evalIntegerAdd(
    Ciphertext& ctxtA,
    Ciphertext& ctxtB,
    int bits);

void processArray(
    Ciphertext& c_processed,
    const Ciphertext& c,
    const std::vector<std::pair<int, int>>& mask_roll_pairs,
    int mask_size,
    int rep,
    lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& cc,
    IDESlib::CKKS::Context& cc_,);
} // namespace FIDESlib::CKKS

#endif // GPUCKKS_INTEGEROPS_CUH