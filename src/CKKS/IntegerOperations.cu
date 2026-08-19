//
// Created by lollo on 19/08/26.
//

#include "CKKS/ApproxModEval.cuh"
#include "CKKS/Ciphertext.cuh"
#include "CKKS/IntegerOperations.cuh"
#include "CKKS/Context.cuh"
#include "CudaUtils.cuh"
#include <iostream>
// Uncomment to trace level/NoiseLevel at key checkpoints, mirrored in
// ApproxModEvalBatch.cu, for side-by-side debugging against the batch port.
#if defined(__clang__)
#include <experimental/source_location>
using sc = std::experimental::source_location;
#else
#include <source_location>
using sc = std::source_location;
#endif

constexpr bool PRINT = false;

using namespace FIDESlib::CKKS;

void evalIntegerAdd(Ciphertext& ctxtA, Ciphertext& ctxtB, int bits) {
    
}
