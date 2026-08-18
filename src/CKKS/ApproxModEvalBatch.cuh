//
// EvalChebyshevSeriesPSBatch for FIDESlib.
//
// Port of the batched (SIMD) Paterson-Stockmeyer Chebyshev series evaluator
// found in lorenzorovida/openfhe-development-chebyshevSIMD
// (AdvancedSHECKKSRNS::EvalChebyshevSeriesPSBatch / InnerEvalChebyshevPSBatch),
// adapted to FIDESlib's GPU Ciphertext/Plaintext API.
//
// Difference with FIDESlib's evalChebyshevSeries (ApproxModEval.cu):
//   evalChebyshevSeries evaluates ONE polynomial (one set of coefficients)
//   on every slot of the ciphertext.
//   evalChebyshevSeriesPSBatch evaluates a DIFFERENT polynomial PER SLOT:
//   batchOfCoefficients[j] holds the Chebyshev coefficients to be applied on
//   slot j, and batchOfCoefficients.size() must equal the number of slots of
//   the input ciphertext. All polynomials in the batch must share the same
//   degree (i.e. every inner vector must have the same size) and are
//   evaluated on the same [lower_bound, upper_bound] range.
//
// Because the "constants" that would be plain scalars in the non-batched
// Paterson-Stockmeyer algorithm become one-value-per-slot vectors here, every
// scalar multiplication (ciphertext * weight) turns into a ciphertext-times-
// plaintext multiplication (ciphertext * per-slot-packed-plaintext). This is
// the same transformation the reference fork performs going from
// EvalLinearWSumMutable (scalars) to EvalLinearWSumMutableBatch (per-slot
// plaintexts).
//
// Unlike evalChebyshevSeries (which only ever needs scalar constants and
// therefore never touches OpenFHE), the batched variant must CKKS-encode a
// fresh per-slot plaintext vector at (essentially) every step of the
// Paterson-Stockmeyer recursion, because those per-slot weights are only
// known once the Chebyshev long-divisions have been carried out at runtime
// -- they cannot be precomputed ahead of time the way e.g. bootstrapping
// linear-transform plaintexts are. FIDESlib has no on-GPU CKKS encoder, so
// -- exactly like api/CryptoContext.cpp's LoadPlaintext does -- encoding is
// delegated to the OpenFHE CPU CryptoContext, and the resulting plaintext is
// then uploaded to the GPU via GetRawPlainText(). This is the one place
// where this core (FIDESlib::CKKS) function needs a reference to the OpenFHE
// CPU CryptoContext.
//

#ifndef GPUCKKS_APPROXMODEVALBATCH_CUH
#define GPUCKKS_APPROXMODEVALBATCH_CUH

#include "CKKS/forwardDefs.cuh"
#include "openfhe-interface/RawCiphertext.cuh"
#include <cinttypes>
#include <vector>

namespace FIDESlib::CKKS {

/**
 * @brief Evaluates a batch of Chebyshev series, one polynomial per slot, using
 *        the Paterson-Stockmeyer algorithm.
 *
 * `batchOfCoefficients[j]` contains the Chebyshev coefficients (as produced
 * e.g. by `get_chebyshev_coefficients`) of the polynomial to be applied to
 * slot `j` of `ctxt`. Every inner vector must have the same size (i.e. all
 * polynomials in the batch must share the same degree) and
 * `batchOfCoefficients.size()` must equal `ctxt.slots`.
 *
 * The ciphertext is modified in place, mirroring `evalChebyshevSeries`.
 *
 * @param cc   OpenFHE CPU CryptoContext matching `ctxt`. Used exclusively to
 *             CKKS-encode the per-slot plaintext weight vectors generated
 *             while walking the Paterson-Stockmeyer recursion (see note
 *             above): FIDESlib has no on-GPU CKKS encoder, so encoding is
 *             delegated to OpenFHE and the result is uploaded to the GPU.
 * @param ctxt Ciphertext to transform in place. `ctxt.slots` must equal
 *             `batchOfCoefficients.size()`.
 * @param batchOfCoefficients Per-slot Chebyshev coefficient vectors. All
 *             inner vectors must have equal size.
 * @param lower_bound Lower bound `a` of the approximation interval.
 * @param upper_bound Upper bound `b` of the approximation interval.
 */
void evalChebyshevSeriesPSBatch(lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& cc,
                                 Ciphertext& ctxt,
                                 const std::vector<std::vector<double>>& batchOfCoefficients,
                                 double lower_bound = -1.0,
                                 double upper_bound = 1.0);

/**
 * @brief Like evalChebyshevSeriesPSBatch, but takes a SMALL set of `x`
 *        Chebyshev-coefficient sets and repeats them cyclically to fill all
 *        `ctxt.slots` slots, instead of requiring one full set per slot.
 *
 * Slot `j` is evaluated with `coefficientSets[j % coefficientSets.size()]`.
 * This is convenient when the same small handful of polynomials needs to be
 * applied repeatedly across a ciphertext (e.g. `n / x` copies of `x`
 * distinct functions), without the caller having to manually replicate the
 * coefficient vector `n` times.
 *
 * `coefficientSets.size()` need not divide `ctxt.slots` evenly; slots are
 * filled by cycling through `coefficientSets` in order, wrapping around as
 * needed, until all `ctxt.slots` slots are covered.
 *
 * @param cc   OpenFHE CPU CryptoContext matching `ctxt` (see
 *             evalChebyshevSeriesPSBatch for why this is needed).
 * @param ctxt Ciphertext to transform in place.
 * @param coefficientSets The small set of `x` Chebyshev coefficient vectors
 *             to cycle through. Must be non-empty, and every inner vector
 *             must have the same size (same degree).
 * @param lower_bound Lower bound `a` of the approximation interval.
 * @param upper_bound Upper bound `b` of the approximation interval.
 */
void evalChebyshevSeriesPSBatchRepeated(lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& cc,
                                         Ciphertext& ctxt,
                                         const std::vector<std::vector<double>>& coefficientSets,
                                         double lower_bound = -1.0,
                                         double upper_bound = 1.0);

} // namespace FIDESlib::CKKS

#endif // GPUCKKS_APPROXMODEVALBATCH_CUH
