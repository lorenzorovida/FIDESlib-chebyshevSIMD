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

void evalIntegerEqual(Ciphertext& ctxtA, Ciphertext& ctxtB, int bits, int zslots, std::vector<double> coeffsSinc, lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& cc) {
    Ciphertext sum(a.cc_);

    sum.copy(a);
    sum.sub(b);
    sum.square();


    // ------------------------------------------------------------
    // sum = sum + rotations of sum
    // ------------------------------------------------------------

    const int rounds =
        static_cast<int>(std::log2(bits));

    for (int i = 0; i < rounds; ++i) {

        const int rotation = 1 << i;

        Ciphertext rotated(a.cc_);
        rotated.rotate(sum, rotation);

        sum.add(rotated);
    }


    evalChebyshevSeries(sum, coeffsSinc, 0, 256);

    // ------------------------------------------------------------
    // correction mask
    // ------------------------------------------------------------

    std::vector<double> correction(
        a.slots,
        0.0);

    for (uint32_t i = 0; i < zslots; ++i) {

        correction[
            i * (bits * bits) / 2
        ] = 1.0;
    }

    // ------------------------------------------------------------
    // Convert correction to FIDESlib plaintext
    // ------------------------------------------------------------

    size_t noise =
        static_cast<size_t>(sum.NoiseLevel);

    auto pt = cc->MakeCKKSPackedPlaintext(
        correction,
        noise,
        sum.getLevel(),
        nullptr,
        sum.slots);

    FIDESlib::CKKS::RawPlainText raw =
        FIDESlib::CKKS::GetRawPlainText(
            cc,
            pt);

    Plaintext correctionPt(
        sum.cc_,
        raw);

    Ciphertext corrected(
        sum.cc_);

    corrected.multPt(
        sum,
        correctionPt,
        false);

    BootstrapStCFirstBits(corrected, corrected.slots, false);


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

void multiplier4bits(Ciphertext& ctxtA, Ciphertext& ctxtB, int repetitions, std::vector<std::vector<double>> coeffs, lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& cc) {
    FIDESlib::CKKS::Context cc_ = ctxtA.cc_;
    
    Ciphertext result(cc_);

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


} // namespace FIDESlib::CKKS