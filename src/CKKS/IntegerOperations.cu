//
// Created by lollo on 19/08/26.
//

#include "CKKS/ApproxModEval.cuh"
#include "CKKS/Ciphertext.cuh"
#include "CKKS/Context.cuh"
#include "CKKS/IntegerOperations.cuh"
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

void FIDESlib::CKKS::evalIntegerAdd(Ciphertext& ctxtA, Ciphertext& ctxtB, int bits) {

    if (bits <= 0) {
        throw std::invalid_argument(
            "evalIntegerAdd: bits must be > 0");
    }

    // p = square(sub(a, b))
    Ciphertext p = ctxtA.copy();
    p.sub(ctxtB);
    p.square();
    
	// absum = p
    Ciphertext absum = p.copy();

    // g = mult(a, b)
    Ciphertext g = ctxtA.copy();
    g.mult(ctxtB);

    for (int i = 1; i < bits; i *= 2) {

        // p_shift = rot(p, -i)
        Ciphertext p_shift = p.copy();
        p_shift.rotate(-i);

        // g_shift = rot(g, -i)
        Ciphertext g_shift = g.copy();
        g_shift.rotate(-i);

        // pg = mult(p, g_shift)
        Ciphertext pg = p.copy();
        pg.mult(g_shift);

        // p_g = mult(p, g)
        Ciphertext p_g = p.copy();
        p_g.mult(g);

        // g = add(g, pg)
        g.add(pg);

        // g = sub(g, p_g)
        g.sub(p_g);

        // p = mult(p, p_shift)
        if (i < bits - 1) {
            p.mult(p_shift);
        }
    }

    // g = rot(g, -1)
    g.rotate(-1);

    // result = square(sub(absum, g))
    ctxtA.copy(absum);
    ctxtA.sub(g);
    ctxtA.square();
}