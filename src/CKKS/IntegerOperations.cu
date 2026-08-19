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

using namespace FIDESlib::CKKS {

	void evalIntegerAdd(Ciphertext & ctxtA, Ciphertext & ctxtB, int bits, Ciphertext& result) {

		if (bits <= 0) {
			throw std::invalid_argument("evalIntegerAdd: bits must be > 0");
		}

		Ciphertext p(ctxtA.cc_);

		p.sub(ctxtA, ctxtB);
		p.square();

		// absum = p
		Ciphertext absum(ctxtA.cc_);
		absum.copy(p);

		// g = a * b
		Ciphertext g(ctxtA.cc_);
		g.mult(ctxtA, ctxtB);

		for (int i = 1; i < bits; i *= 2) {

			// p_shift = rot(p, -i)
			Ciphertext p_shift(ctxtA.cc_);
			p_shift.rotate(p, -i);

			// g_shift = rot(g, -i)
			Ciphertext g_shift(ctxtA.cc_);
			g_shift.rotate(g, -i);

			/*
			 * pg = p * g_shift
			 */
			Ciphertext pg(ctxtA.cc_);
			pg.mult(p, g_shift);

			Ciphertext p_g(ctxtA.cc_);
			p_g.mult(p, g);

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

} // namespace FIDESlib::CKKS
