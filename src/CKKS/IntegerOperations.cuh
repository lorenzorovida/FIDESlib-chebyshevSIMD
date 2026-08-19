#ifndef GPUCKKS_INTEGEROPS_CUH
#define GPUCKKS_INTEGEROPS_CUH

#include "CKKS/forwardDefs.cuh"
#include "openfhe-interface/RawCiphertext.cuh"
#include <cinttypes>
#include <vector>

namespace FIDESlib::CKKS {


void evalIntegerAdd(Ciphertext& ctxtA, Ciphertext& ctxtB, int bits);

} // namespace FIDESlib::CKKS

#endif // GPUCKKS_INTEGEROPS_CUH
