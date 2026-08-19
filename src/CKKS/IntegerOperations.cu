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

	Ciphertext p = result.copy();

	if (clean_first) {
		p.add(other);
		// clean_and_reduce equivalent here
	} else {
		p.sub(other);
		p.square();
	}

	Ciphertext absum = p.copy();

	Ciphertext g = result.copy();
	g.mult(other);

	for (int i = 1; i < bits; i *= 2) {

		Ciphertext p_shift = p.copy();
		p_shift.rotate(-i);

		Ciphertext g_shift = g.copy();
		g_shift.rotate(-i);

		Ciphertext pg = p.copy();
		pg.mult(g_shift);

		Ciphertext p_g = p.copy();
		p_g.mult(g);

		g.add(pg);
		g.sub(p_g);

		if (i < bits - 1) {
			p.mult(p_shift);
		}
	}

	g.rotate(-1);

	result.copy(absum);
	result.sub(g);
	result.square();
}
