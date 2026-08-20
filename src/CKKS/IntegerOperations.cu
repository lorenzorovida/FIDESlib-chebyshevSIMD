#include "CKKS/Ciphertext.cuh"
#include "CKKS/IntegerOperations.cuh"

#include <stdexcept>

namespace FIDESlib::CKKS {

void evalIntegerAdd(Ciphertext& ctxtA, Ciphertext& ctxtB, int bits) {
	if (bits <= 0) {
		throw std::invalid_argument("evalIntegerAdd: bits must be > 0");
	}

	/*
	 * Original CPU:
	 *
	 * p = square(sub(a, b));
	 */

	Ciphertext p(ctxtA.cc_);
	p.copy(ctxtA);
	p.sub(ctxtB);
	p.square();

	/*
	 * absum = p->Clone()
	 */

	Ciphertext absum(ctxtA.cc_);
	absum.copy(p);

	/*
	 * g = mult(a, b);
	 */

	Ciphertext g(ctxtA.cc_);
	g.copy(ctxtA);
	g.mult(ctxtB);

	/*
	 * for (int i = 1; i < bits; i *= 2)
	 */
	for (int i = 1; i < bits; i *= 2) {

		/*
		 * p_shift = rot(p, -i)
		 */
		Ciphertext p_shift(ctxtA.cc_);
		p_shift.rotate(p, -i);

		/*
		 * g_shift = rot(g, -i)
		 */
		Ciphertext g_shift(ctxtA.cc_);
		g_shift.rotate(g, -i);

		/*
		 * pg = mult(p, g_shift)
		 */
		Ciphertext pg(ctxtA.cc_);
		pg.mult(p, g_shift);

		/*
		 * mult(p, g)
		 */
		Ciphertext p_g(ctxtA.cc_);
		p_g.mult(p, g);

		/*
		 * g = sub(add(g, pg), p_g)
		 */
		g.add(pg);
		g.sub(p_g);

		/*
		 * p = mult(p, p_shift)
		 */
		if (i < bits - 1) {
			p.mult(p_shift);
		}
	}

	/*
	 * g = rot(g, -1)
	 */
	g.rotate(-1);

	/*
	 * s = square(sub(absum, g))
	 *
	 * Since ctxtA is our output buffer, overwrite it.
	 */
	ctxtA.copy(absum);
	ctxtA.sub(g);
	ctxtA.square();
}

void evalIntegerEqual(Ciphertext& a, Ciphertext& b, int bits, int zslots, std::vector<double> coeffsSinc, lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& cc) {
	Ciphertext sum(a.cc_);

	sum.copy(a);
	sum.sub(b);
	sum.square();

	// ------------------------------------------------------------
	// sum = sum + rotations of sum
	// ------------------------------------------------------------

	const int rounds = static_cast<int>(std::log2(bits));

	for (int i = 0; i < rounds; ++i) {

		const int rotation = 1 << i;

		Ciphertext rotated(a.cc_);
		rotated.rotate(sum, rotation);

		sum.add(rotated);
	}

	//evalChebyshevSeries(sum, coeffsSinc, 0, 256);

	// ------------------------------------------------------------
	// correction mask
	// ------------------------------------------------------------

	std::vector<double> correction(a.slots, 0.0);

	for (int i = 0; i < zslots; ++i) {

		correction[i * (bits * bits) / 2] = 1.0;
	}

	// ------------------------------------------------------------
	// Convert correction to FIDESlib plaintext
	// ------------------------------------------------------------

	size_t noise = static_cast<size_t>(sum.NoiseLevel);

	auto pt = cc->MakeCKKSPackedPlaintext(correction, noise, sum.getLevel(), nullptr, sum.slots);

	FIDESlib::CKKS::RawPlainText raw = FIDESlib::CKKS::GetRawPlainText(cc, pt);

	Plaintext correctionPt(sum.cc_, raw);

	Ciphertext corrected(sum.cc_);

	corrected.multPt(sum, correctionPt, false);

	BootstrapStCFirstBits(corrected, corrected.slots, false);
}

void evalIntegerMult(Ciphertext& out,
  const Ciphertext& a,
  const Ciphertext& b,
  int bits,
  int bits_original,
  int repetitions,
  int repetitions_original,
  bool overflow,
  std::vector<std::vector<double>>& coeffs,
  lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& cc) {
	const int rep_size = bits * bits / 2;

	// Size of basic multiplier.
	const int base_mult = 8;

	FIDESlib::CKKS::Context& cc_ = a.cc_;

	Ciphertext result(a.cc_);

	// ============================================================
	// BASE CASE: 8-bit multiplier
	// ============================================================

	if (bits == 8) {

		// --------------------------------------------------------
		// A
		// --------------------------------------------------------

		const int mask_size = bits_original * (bits_original / 2);

		std::vector<double> masklow(a.slots, 0);

		for (int j = 0; j < repetitions_original; ++j) {
			masklow[0 + j * mask_size] = 1;
			masklow[1 + j * mask_size] = 1;
			masklow[2 + j * mask_size] = 1;
			masklow[3 + j * mask_size] = 1;
		}

		Ciphertext a_low(a.cc_);

		size_t noise = a.NoiseFactor;

		auto pt = cc->MakeCKKSPackedPlaintext(masklow, noise, a.getLevel(), nullptr, a.slots);

		FIDESlib::CKKS::RawPlainText raw = FIDESlib::CKKS::GetRawPlainText(cc, pt);
		Plaintext maskP(cc_, raw);

		a_low.multPt(a, maskP, false);

		// --------------------------------------------------------
		// maskhigh
		// --------------------------------------------------------

		std::vector<double> maskhigh(a.slots, 0);

		for (int j = 0; j < repetitions_original; ++j) {
			maskhigh[4 + j * mask_size] = 1;
			maskhigh[5 + j * mask_size] = 1;
			maskhigh[6 + j * mask_size] = 1;
			maskhigh[7 + j * mask_size] = 1;
		}

		const int highShift = -(16 - 4);

		Ciphertext a_rot(a.cc_);
		a_rot.rotate(a, highShift);

		std::vector<double> maskhigh_rot = rotateMask(maskhigh, -highShift);

		Ciphertext a_high(a.cc_);

		noise = a_rot.NoiseFactor;

		pt = cc->MakeCKKSPackedPlaintext(maskhigh_rot, noise, a_rot.getLevel(), nullptr, a_rot.slots);

		raw = FIDESlib::CKKS::GetRawPlainText(cc, pt);
		Plaintext maskP2(cc_, raw);

		a_high.multPt(a_rot, maskP2, false);

		a_low.add(a_high);

		Ciphertext a_processed(a.cc_);
		a_processed.copy(a_low);

		// --------------------------------------------------------
		// process_array(...)
		// --------------------------------------------------------

		if (bits_original > 8) {

			processArray(a_processed, a, { { 8, 64 }, { 12, 80 } }, mask_size, repetitions_original, cc, cc_);
		}

		if (bits_original > 16) {

			processArray(a_processed, a, { { 16, 256 }, { 20, 272 }, { 24, 320 }, { 28, 336 } }, mask_size, repetitions_original, cc, cc_);
		}

		if (bits_original > 32) {

			processArray(
			  a_processed, a, { { 32, 1024 }, { 36, 1040 }, { 40, 1088 }, { 44, 1104 }, { 48, 1280 }, { 52, 1296 }, { 56, 1344 }, { 60, 1360 } }, mask_size, repetitions_original, cc, cc_);
		}

		if (bits_original > 64) {

			processArray(a_processed,
			  a,
			  { { 64, 4096 },
				{ 68, 4112 },
				{ 72, 4160 },
				{ 76, 4176 },
				{ 80, 4352 },
				{ 84, 4368 },
				{ 88, 4416 },
				{ 92, 4432 },
				{ 96, 5120 },
				{ 100, 5136 },
				{ 104, 5184 },
				{ 108, 5200 },
				{ 112, 5376 },
				{ 116, 5392 },
				{ 120, 5440 },
				{ 124, 5456 } },
			  mask_size,
			  repetitions_original,
			  cc,
			  cc_);
		}

		if (bits_original > 128) {

			processArray(a_processed,
			  a,
			  { { 128, 16384 },
				{ 132, 16400 },
				{ 136, 16448 },
				{ 140, 16464 },
				{ 144, 16640 },
				{ 148, 16656 },
				{ 152, 16704 },
				{ 156, 16720 },
				{ 160, 17408 },
				{ 164, 17424 },
				{ 168, 17472 },
				{ 172, 17488 },
				{ 176, 17664 },
				{ 180, 17680 },
				{ 184, 17728 },
				{ 188, 17744 },
				{ 192, 20480 },
				{ 196, 20496 },
				{ 200, 20544 },
				{ 204, 20560 },
				{ 208, 20736 },
				{ 212, 20752 },
				{ 216, 20800 },
				{ 220, 20816 },
				{ 224, 21504 },
				{ 228, 21520 },
				{ 232, 21568 },
				{ 236, 21584 },
				{ 240, 21760 },
				{ 244, 21776 },
				{ 248, 21824 },
				{ 252, 21840 } },
			  mask_size,
			  repetitions_original,
			  cc,
			  cc_);
		}

		// --------------------------------------------------------
		// Combine A
		// --------------------------------------------------------

		if (bits_original > 4) {

			Ciphertext tmp(a.cc_);
			tmp.rotate(a_processed, -8);
			a_processed.add(tmp);
		}

		if (bits_original > 8) {

			Ciphertext tmp(a.cc_);
			tmp.rotate(a_processed, -32);
			a_processed.add(tmp);
		}

		if (bits_original > 16) {

			Ciphertext tmp(a.cc_);
			tmp.rotate(a_processed, -128);
			a_processed.add(tmp);
		}

		if (bits_original > 32) {

			Ciphertext tmp(a.cc_);
			tmp.rotate(a_processed, -512);
			a_processed.add(tmp);
		}

		if (bits_original > 64) {

			Ciphertext tmp(a.cc_);
			tmp.rotate(a_processed, -2048);
			a_processed.add(tmp);
		}

		if (bits_original > 128) {

			Ciphertext tmp(a.cc_);
			tmp.rotate(a_processed, -8192);
			a_processed.add(tmp);
		}

		// ========================================================
		// B
		// ========================================================

		Ciphertext b_processed(a.cc_);

		noise = b.NoiseFactor;

		pt = cc->MakeCKKSPackedPlaintext(masklow, noise, b.getLevel(), nullptr, b.slots);

		raw = FIDESlib::CKKS::GetRawPlainText(cc, pt);
		Plaintext maskP3(cc_, raw);
		b_processed.multPt(b, maskP3);

		Ciphertext b_rot(a.cc_);
		b_rot.rotate(b, -4);

		std::vector<double> maskhigh_b = rotateMask(maskhigh, 4);

		Ciphertext b_high(a.cc_);

		noise = b_rot.NoiseFactor;

		pt = cc->MakeCKKSPackedPlaintext(maskhigh_b, noise, b_rot.getLevel(), nullptr, b_rot.slots);

		raw = FIDESlib::CKKS::GetRawPlainText(cc, pt);
		Plaintext maskP4(cc_, raw);
		b_high.multPt(b_rot, maskP4);

		b_processed.add(b_high);

		// --------------------------------------------------------
		// process B
		// --------------------------------------------------------

		if (bits_original > 8) {

			processArray(b_processed, b, { { 8, 32 }, { 12, 40 } }, mask_size, repetitions_original, cc, cc_);
		}

		if (bits_original > 16) {

			processArray(b_processed, b, { { 16, 128 }, { 20, 136 }, { 24, 160 }, { 28, 168 } }, mask_size, repetitions_original, cc, cc_);
		}

		if (bits_original > 32) {

			processArray(
			  b_processed, b, { { 32, 512 }, { 36, 520 }, { 40, 544 }, { 44, 552 }, { 48, 640 }, { 52, 648 }, { 56, 672 }, { 60, 680 } }, mask_size, repetitions_original, cc, cc_);
		}

		if (bits_original > 64) {

			processArray(b_processed,
			  b,
			  { { 64, 2048 },
				{ 68, 2056 },
				{ 72, 2080 },
				{ 76, 2088 },
				{ 80, 2176 },
				{ 84, 2184 },
				{ 88, 2208 },
				{ 92, 2216 },
				{ 96, 2560 },
				{ 100, 2568 },
				{ 104, 2592 },
				{ 108, 2600 },
				{ 112, 2688 },
				{ 116, 2696 },
				{ 120, 2720 },
				{ 124, 2728 } },
			  mask_size,
			  repetitions_original,
			  cc,
			  cc_);
		}

		if (bits_original > 128) {

			processArray(b_processed,
			  b,
			  { { 128, 8192 },
				{ 132, 8200 },
				{ 136, 8224 },
				{ 140, 8232 },
				{ 144, 8320 },
				{ 148, 8328 },
				{ 152, 8352 },
				{ 156, 8360 },
				{ 160, 8704 },
				{ 164, 8712 },
				{ 168, 8736 },
				{ 172, 8744 },
				{ 176, 8832 },
				{ 180, 8840 },
				{ 184, 8864 },
				{ 188, 8872 },
				{ 192, 10240 },
				{ 196, 10248 },
				{ 200, 10272 },
				{ 204, 10280 },
				{ 208, 10368 },
				{ 212, 10376 },
				{ 216, 10400 },
				{ 220, 10408 },
				{ 224, 10752 },
				{ 228, 10760 },
				{ 232, 10784 },
				{ 236, 10792 },
				{ 240, 10880 },
				{ 244, 10888 },
				{ 248, 10912 },
				{ 252, 10920 } },
			  mask_size,
			  repetitions_original,
			  cc,
			  cc_);
		}

		// --------------------------------------------------------
		// Combine B
		// --------------------------------------------------------

		if (bits_original > 4) {

			Ciphertext tmp(a.cc_);
			tmp.rotate(b_processed, -16);
			b_processed.add(tmp);
		}

		if (bits_original > 8) {

			Ciphertext tmp(a.cc_);
			tmp.rotate(b_processed, -64);
			b_processed.add(tmp);
		}

		if (bits_original > 16) {

			Ciphertext tmp(a.cc_);
			tmp.rotate(b_processed, -256);
			b_processed.add(tmp);
		}

		if (bits_original > 32) {

			Ciphertext tmp(a.cc_);
			tmp.rotate(b_processed, -1024);
			b_processed.add(tmp);
		}

		if (bits_original > 64) {

			Ciphertext tmp(a.cc_);
			tmp.rotate(b_processed, -4096);
			b_processed.add(tmp);
		}

		if (bits_original > 128) {

			Ciphertext tmp(a.cc_);
			tmp.rotate(b_processed, -16384);
			b_processed.add(tmp);
		}

		// --------------------------------------------------------
		// Convert binary to decimal
		// --------------------------------------------------------

		Ciphertext a_decimal(a.cc_);
		bintodec(cc, a_decimal, a_processed, repetitions * 4);

		Ciphertext b_decimal(a.cc_);
		bintodec(cc, b_decimal, b_processed, repetitions * 4);

		// --------------------------------------------------------
		// 4-bit multiplier
		// --------------------------------------------------------

		multiplier4bits(result, a_decimal, b_decimal, repetitions * 4, coeffs, cc);
	} else {

		// Recursive case:
		//
		// result =
		// mul_integer(
		//     a,
		//     b,
		//     bits / 2,
		//     bits_original,
		//     4 * repetitions,
		//     repetitions_original,
		//     overflow);

		evalIntegerMult(result, a, b, bits / 2, bits_original, 4 * repetitions, repetitions_original, overflow, coeffs, cc);
	}

	// ============================================================
	// Recombine multiplication result
	// ============================================================

	const int dunn = (bits * bits / base_mult) * 2;

	std::vector<double> mask1(a.slots, 0.0);

	for (int j = 0; j < repetitions; ++j) {

		for (int i = 0; i < bits; ++i) {

			mask1[(j * rep_size) + i] = 1.0;

			mask1[(j * rep_size) + i + dunn] = 1.0;
		}
	}

	std::vector<double> mask2(a.slots, 0.0);

	for (int j = 0; j < repetitions; ++j) {

		for (int i = 0; i < bits; ++i) {

			mask2[(j * rep_size) + rep_size / 4 + i] = 1.0;

			mask2[(j * rep_size) + rep_size / 4 + i + dunn] = 1.0;
		}
	}

	// ------------------------------------------------------------
	// p1 = result * mask1
	// ------------------------------------------------------------

	Ciphertext p1(a.cc_);

	/*
	multMask(
		p1,
		result,
		mask1,
		cc);
	*/

	size_t noise = result.NoiseFactor;

	auto pt = cc->MakeCKKSPackedPlaintext(mask1, noise, result.getLevel(), nullptr, result.slots);

	FIDESlib::CKKS::RawPlainText raw = FIDESlib::CKKS::GetRawPlainText(cc, pt);
	Plaintext maskP5(cc_, raw);
	p1.multPt(result, maskP5);

	// ------------------------------------------------------------
	// p2 = rot(p1, -(-rep_size/2 + bits/2))
	// ------------------------------------------------------------

	Ciphertext p2(a.cc_);

	p2.rotate(p1, -(-rep_size / 2 + bits / 2));

	// ------------------------------------------------------------
	// p3
	// ------------------------------------------------------------

	Ciphertext masked2(a.cc_);

	/*
	multMask(
		masked2,
		result,
		mask2,
		cc);
	*/
	noise = result.NoiseFactor;

	pt = cc->MakeCKKSPackedPlaintext(mask2, noise, result.getLevel(), nullptr, result.slots);

	raw = FIDESlib::CKKS::GetRawPlainText(cc, pt);
	Plaintext maskP6(cc_, raw);
	masked2.multPt(result, maskP6);

	Ciphertext p3(a.cc_);

	if (bits == 8) {

		p3.rotate(masked2, 16);

	} else {

		p3.rotate(masked2, -(-rep_size / 4 + bits / 2));
	}

	// ------------------------------------------------------------
	// p4
	// ------------------------------------------------------------

	Ciphertext p4(a.cc_);

	if (bits == 8) {

		p4.rotate(p3, -12);

	} else {

		p4.rotate(masked2, ((bits - 2) * (3 * bits - 2)) / 8);
	}

	// ============================================================
	// Final CSA + bootstrap
	// ============================================================

	if (!overflow && bits == bits_original) {

		Ciphertext S(a.cc_);
		Ciphertext C(a.cc_);

		csa3(S, C, p1, p2, p3);

		Ciphertext rotatedC(a.cc_);

		rotatedC.rotate(C, -1);

		// add_integer(S, rot(C, -1), bits)
		evalIntegerAdd(S, rotatedC, bits);

		// binboot(...)
		//
		// Replace with your FIDESlib binboot implementation.
		//
		// result = binboot(S);

		result.copy(S);
	} else {

		csa4(result, p1, p2, p3, p4, bits);
		BootstrapStCFirstBits(result, result.slots, false);
	}

	out.copy(result);
}

void processArray(Ciphertext& c_processed,
  const Ciphertext& c,
  const std::vector<std::pair<int, int>>& mask_roll_pairs,
  int mask_size,
  int rep,
  lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& cc,
  FIDESlib::CKKS::Context& cc_) {
	if (mask_size <= 0) {
		throw std::invalid_argument("processArray: mask_size must be > 0");
	}

	if (rep <= 0) {
		throw std::invalid_argument("processArray: rep must be > 0");
	}

	/*
	 * Original:
	 *
	 * Ctxt c_processed_clone = c_processed->Clone();
	 *
	 * We use c_processed itself as the output buffer.
	 */
	Ciphertext result(c_processed.cc_);
	result.copy(c_processed);

	const int total_size = mask_size * rep;

	for (const auto& [start, roll_base] : mask_roll_pairs) {

		/*
		 * mask = [0, ..., 0]
		 *
		 * with four consecutive 1s at:
		 *
		 * i * mask_size + start + j
		 *
		 * for every i.
		 */
		std::vector<double> mask(total_size, 0);

		for (int i = 0; i < rep; ++i) {
			for (int j = 0; j < 4; ++j) {
				mask[i * mask_size + start + j] = 1.0;
			}
		}

		/*
		 * shift = roll_base - start
		 */
		const int shift = roll_base - start;

		/*
		 * rolled_ctxt = rot_fast(c, -shift)
		 *
		 * For now use the normal FIDESlib GPU rotation.
		 */
		Ciphertext rolled_ctxt(c.cc_);
		rolled_ctxt.rotate(c, -shift);

		/*
		 * rolled_mask = rot(mask, -shift)
		 *
		 * This is a plaintext/vector rotation.
		 *
		 * We do not actually need to construct rolled_mask
		 * explicitly if the plaintext multiplication API can
		 * apply the rotated mask directly.
		 */
		std::vector<double> rolled_mask(total_size, 0);

		for (int k = 0; k < total_size; ++k) {

			/*
			 * Match the semantics of the original rot(mask, -shift).
			 *
			 * Positive modulo is required for negative shifts.
			 */
			int src = (k + shift) % total_size;

			if (src < 0) {
				src += total_size;
			}

			rolled_mask[k] = mask[src];
		}

		/*
		 * rolled_ctxt * rolled_mask
		 *
		 * Then:
		 *
		 * c_processed_clone =
		 *     add(c_processed_clone,
		 *         mult(rolled_ctxt, rolled_mask));
		 *
		 * The exact FIDESlib plaintext-multiplication API depends
		 * on how your branch represents plaintext vectors.
		 */

		Ciphertext masked(c.cc_);

		/*
		 * TODO:
		 *
		 * Replace this with the FIDESlib plaintext multiplication
		 * function available in your checkout.
		 */
		// masked.mult(rolled_ctxt, rolled_mask);

		size_t noise					 = static_cast<size_t>(rolled_ctxt.NoiseLevel);
		auto pt							 = cc->MakeCKKSPackedPlaintext(rolled_mask, noise, rolled_ctxt.getLevel(), nullptr, rolled_ctxt.slots);
		FIDESlib::CKKS::RawPlainText raw = FIDESlib::CKKS::GetRawPlainText(cc, pt);
		Plaintext maskP					 = Plaintext(cc_, raw);

		masked.multPt(rolled_ctxt, maskP, false);

		result.add(masked);

		if (result.cc.rescaleTechnique == FIDESlib::CKKS::FIXEDMANUAL) {
			result.rescale();
		}
	}

	c_processed.copy(result);
}

void multiplier4bits(Ciphertext& result, Ciphertext& ctxtA, Ciphertext& ctxtB, int repetitions, std::vector<std::vector<double>> coeffs, lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& cc) {
	FIDESlib::CKKS::Context cc_ = ctxtA.cc_;

	result.mult(ctxtA, ctxtB);

	// result = result - 1
	std::vector<double> minusOne(result.slots, -1.0);

	size_t noise = static_cast<size_t>(result.NoiseLevel);

	auto pt = cc->MakeCKKSPackedPlaintext(minusOne, noise, result.getLevel(), nullptr, result.slots);

	FIDESlib::CKKS::RawPlainText raw = FIDESlib::CKKS::GetRawPlainText(cc, pt);

	Plaintext minusOnePt(result.cc_, raw);

	result.addPt(minusOnePt);

	FIDESlib::CKKS::evalChebyshevSeriesPSBatchRepeated(cc, result, coeffs, -1, 1);

	FIDESlib::CKKS::BootstrapStCFirstBits(result, result.slots, false);
}

void cleanAndReduce(Ciphertext& out, const Ciphertext& c) {
	FIDESlib::CKKS::Context& cc_ = c.cc_;

	// sqC := c^2
	Ciphertext sqC(cc_);
	sqC.square(c, false);
	if (sqC.NoiseLevel == 2)
		sqC.rescale();

	// shifted := (c - 2)^2
	Ciphertext shifted(cc_);
	shifted.copy(c);
	shifted.addScalar(-2.0);
	shifted.square(false);
	if (shifted.NoiseLevel == 2)
		shifted.rescale();

	// out := sqC * shifted
	out.mult(sqC, shifted, false);
}

void clean(Ciphertext& out, const Ciphertext& c) {
	FIDESlib::CKKS::Context& cc_ = c.cc_;

	// sq := c^2
	Ciphertext sq(cc_);
	sq.square(c, false);
	if (sq.NoiseLevel == 2)
		sq.rescale();

	// t1 := c * (-2)
	Ciphertext t1(cc_);
	t1.multScalar(c, -2.0, false);
	if (t1.NoiseLevel == 2)
		t1.rescale();

	// termA := sq * t1
	Ciphertext termA(cc_);
	termA.mult(sq, t1, false);

	// termB := sq * 3
	Ciphertext termB(cc_);
	termB.multScalar(sq, 3.0, false);

	// out := termA + termB
	out.add(termA, termB);
}

void mod2Shallow(Ciphertext& out, const Ciphertext& c) {
	FIDESlib::CKKS::Context& cc_ = c.cc_;

	// doubled := c * 2
	Ciphertext doubled(cc_);
	doubled.multScalar(c, 2.0, false);

	// sq := c^2
	Ciphertext sq(cc_);
	sq.square(c, false);
	if (sq.NoiseLevel == 2)
		sq.rescale();

	// out := doubled - sq
	out.sub(doubled, sq);
}

void majorityBit(Ciphertext& out, const Ciphertext& a, const Ciphertext& b, const Ciphertext& c) {
	FIDESlib::CKKS::Context& cc_ = a.cc_;

	// total := a + b + c
	Ciphertext total(cc_);
	total.add(a, b);
	total.add(c);

	// sq := total^2
	Ciphertext sq(cc_);
	sq.square(total, false);
	if (sq.NoiseLevel == 2)
		sq.rescale();

	// t1 := total * (-1/3)
	Ciphertext t1(cc_);
	t1.multScalar(total, -1.0 / 3.0, false);
	if (t1.NoiseLevel == 2)
		t1.rescale();

	// termA := t1 * sq
	Ciphertext termA(cc_);
	termA.mult(t1, sq, false);

	// termB := sq * 3/2
	Ciphertext termB(cc_);
	termB.multScalar(sq, 3.0 / 2.0, false);

	// termC := total * (-7/6)
	Ciphertext termC(cc_);
	termC.multScalar(total, -7.0 / 6.0, false);

	// out := termA + termB + termC
	Ciphertext ab(cc_);
	ab.add(termA, termB);
	out.add(ab, termC);
}

void csa3(Ciphertext& S, Ciphertext& C, const Ciphertext& a, const Ciphertext& b, const Ciphertext& c) {
	FIDESlib::CKKS::Context& cc_ = a.cc_;

	// S := mod2Shallow(a + b)
	Ciphertext ab(cc_);
	ab.add(a, b);
	mod2Shallow(S, ab);

	// S := mod2Shallow(S + c)
	Ciphertext sPlusC(cc_);
	sPlusC.add(S, c);
	mod2Shallow(S, sPlusC);

	majorityBit(C, a, b, c);
}

void csa4(Ciphertext& out, const Ciphertext& a, const Ciphertext& b, const Ciphertext& c, const Ciphertext& d, int bits) {
	// ------------------------------------------------------------
	// First CSA:
	//
	// (s1, c1) = csa3(a, b, c)
	// ------------------------------------------------------------

	Ciphertext s1(a.cc_);
	Ciphertext c1(a.cc_);

	csa3(s1, c1, a, b, c);

	// c1 = rot(c1, -1)
	Ciphertext c1_rot(a.cc_);

	c1_rot.rotate(c1, -1);

	// ------------------------------------------------------------
	// Second CSA:
	//
	// (s2, c2) = csa3(s1, c1_rot, d)
	// ------------------------------------------------------------

	Ciphertext s2(a.cc_);
	Ciphertext c2(a.cc_);

	csa3(s2, c2, s1, c1_rot, d);

	// c2 = rot(c2, -1)
	Ciphertext c2_rot(a.cc_);

	c2_rot.rotate(c2, -1);

	// ------------------------------------------------------------
	// result = add_integer(s2, c2_rot, bits, false)
	// ------------------------------------------------------------

	out.copy(s2);

	evalIntegerAdd(out, c2_rot, bits);
}

void bintodec(lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& cc, Ciphertext& out, const Ciphertext& c, int repetitions) {
	FIDESlib::CKKS::Context& cc_ = c.cc_;

	// Build the {1,2,4,8,0,0,0,0}-repeated mask, encoded at c's level, and
	// the second {sqrt(1/(225/2)), 0,0,0,0,0,0,0}-repeated rescaling mask,
	// mirroring OpenFHE's bintodec exactly (including the folding rotation
	// pattern +1,+2 then -1,-2,-4).
	std::vector<double> mask1;
	mask1.reserve(static_cast<size_t>(repetitions) * 8);
	for (int i = 0; i < repetitions; i++) {
		mask1.insert(mask1.end(), { 1.0, 2.0, 4.0, 8.0, 0.0, 0.0, 0.0, 0.0 });
	}

	std::vector<double> mask2;
	mask2.reserve(static_cast<size_t>(repetitions) * 8);
	double rescaleFactor = std::sqrt(1.0 / (225.0 / 2.0));
	for (int i = 0; i < repetitions; i++) {
		mask2.insert(mask2.end(), { rescaleFactor, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 });
	}

	// Same level-convention caveat as ApproxModEvalBatch.cu's
	// makePerSlotPlaintext: FIDESlib's core getLevel() counts DOWN from
	// cc.L, while MakeCKKSPackedPlaintext's `level` counts UP from 0, and
	// noiseScaleDeg must match the target ciphertext's real NoiseLevel.
	auto makeMaskPt = [&](const std::vector<double>& values, const Ciphertext& like) -> Plaintext {
		uint32_t openfheLevel			 = static_cast<uint32_t>(like.cc.L - like.getLevel());
		size_t noiseScaleDeg			 = static_cast<size_t>(like.NoiseLevel);
		auto pt							 = cc->MakeCKKSPackedPlaintext(values,
		  /*noiseScaleDeg=*/noiseScaleDeg,
		  /*level=*/openfheLevel,
		  nullptr,
		  /*slots=*/like.slots);
		FIDESlib::CKKS::RawPlainText raw = FIDESlib::CKKS::GetRawPlainText(cc, pt);
		return Plaintext(cc_, raw);
	};

	// res := c * mask1  (weights 1,2,4,8 at slots 0..3 of each group)
	Ciphertext res(cc_);
	Plaintext maskPt1 = makeMaskPt(mask1, c);
	res.multPt(c, maskPt1, true);

	// res := res + rot(res, 1) + rot(res, 2)  -- sums the 4 weighted bits
	// into slot 0 of each group.
	Ciphertext rot1(cc_);
	rot1.rotate(res, 1);
	res.add(rot1);
	Ciphertext rot2(cc_);
	rot2.rotate(res, 2);
	res.add(rot2);

	// res := res * mask2  (keeps only slot 0 of each group, rescaled)
	Plaintext maskPt2 = makeMaskPt(mask2, res);
	res.multPt(maskPt2, true);

	// res := res + rot(res,-1) + rot(res,-2) + rot(res,-4)  -- broadcasts
	// the decoded value back across the low 4 slots of each group.
	Ciphertext rotm1(cc_);
	rotm1.rotate(res, -1);
	res.add(rotm1);
	Ciphertext rotm2(cc_);
	rotm2.rotate(res, -2);
	res.add(rotm2);
	Ciphertext rotm4(cc_);
	rotm4.rotate(res, -4);
	res.add(rotm4);

	out.copy(res);
}

std::vector<double> rotateMask(const std::vector<double>& mask, int shift) {
	const int n = static_cast<int>(mask.size());

	std::vector<double> result(n);

	for (int i = 0; i < n; ++i) {
		int src = (i + shift) % n;

		if (src < 0)
			src += n;

		result[i] = mask[src];
	}

	return result;
}

} // namespace FIDESlib::CKKS