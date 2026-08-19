#include "CKKS/IntegerOperations.cuh"
#include "CKKS/Ciphertext.cuh"

#include <stdexcept>
#include <utility>
#include <vector>

namespace FIDESlib::CKKS {

void evalIntegerAdd(
    Ciphertext& ctxtA,
    Ciphertext& ctxtB,
    int bits)
{
    if (bits <= 0) {
        throw std::invalid_argument(
            "evalIntegerAdd: bits must be > 0");
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


void processArray(
    Ciphertext& c_processed,
    const Ciphertext& c,
    const std::vector<std::pair<int, int>>& mask_roll_pairs,
    int mask_size,
    int rep)
{
    if (mask_size <= 0) {
        throw std::invalid_argument(
            "processArray: mask_size must be > 0");
    }

    if (rep <= 0) {
        throw std::invalid_argument(
            "processArray: rep must be > 0");
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
        std::vector<int> mask(total_size, 0);

        for (int i = 0; i < rep; ++i) {
            for (int j = 0; j < 4; ++j) {
                mask[i * mask_size + start + j] = 1;
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
        std::vector<int> rolled_mask(total_size, 0);

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
        masked.mult(rolled_ctxt, rolled_mask);

        result.add(masked);
    }


    c_processed.copy(result);
}


} // namespace FIDESlib::CKKS