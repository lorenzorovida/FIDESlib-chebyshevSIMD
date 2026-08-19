#include "CKKS/IntegerOperations.cuh"
#include "CKKS/Ciphertext.cuh"

#include <stdexcept>

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

} // namespace FIDESlib::CKKS