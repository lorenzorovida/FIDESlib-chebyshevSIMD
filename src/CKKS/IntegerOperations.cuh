#ifndef GPUCKKS_INTEGEROPS_CUH
#define GPUCKKS_INTEGEROPS_CUH

#include "CKKS/forwardDefs.cuh"

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
    int rep);
    
} // namespace FIDESlib::CKKS

#endif // GPUCKKS_INTEGEROPS_CUH