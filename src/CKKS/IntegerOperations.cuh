#ifndef GPUCKKS_INTEGEROPS_CUH
#define GPUCKKS_INTEGEROPS_CUH

#include "CKKS/forwardDefs.cuh"

namespace FIDESlib::CKKS {

void evalIntegerAdd(
    Ciphertext& ctxtA,
    Ciphertext& ctxtB,
    int bits);

} // namespace FIDESlib::CKKS

#endif // GPUCKKS_INTEGEROPS_CUH