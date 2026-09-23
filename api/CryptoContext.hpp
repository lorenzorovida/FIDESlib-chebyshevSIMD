#ifndef API_CRYPTOCONTEXT_HPP
#define API_CRYPTOCONTEXT_HPP

#include <any>
#include <array>
#include <complex>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <shared_mutex>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "CCParams.hpp"
#include "Ciphertext.hpp"
#include "Definitions.hpp"
#include "KeyPair.hpp"
#include "Plaintext.hpp"
#include "PublicKey.hpp"
#include "Serialize.hpp"


namespace fideslib {

/// @brief Input cifrati dell'esempio Uniswap v3 (main.cpp: experiment_uniswap_v3).
/// Interi a 128 bit, bit-packed LSB-first come encrypt_multi_int(..., 128, lvl),
/// su tutti gli slot del ciphertext. g_den_Y_prec e g_num non compaiono perche'
/// il CPU non li usa nel calcolo (997 e' usato come divisore in chiaro).
/// Livello di cifratura consigliato: OpenFHE <= 12 (es. 10 come nel CPU).
struct UniswapV3Inputs {
	Ciphertext<DCRTPoly> g_num_inv_L_fx;
	Ciphertext<DCRTPoly> user_amount;
	Ciphertext<DCRTPoly> inv_sqrt_P0_fx;
	Ciphertext<DCRTPoly> numerator;
	Ciphertext<DCRTPoly> m_fx;
	Ciphertext<DCRTPoly> g_den;
};

/// @brief Intermedi opzionali di EvalUniswapV3Example (stessi punti delle stampe
/// del CPU). Vengono riempiti sul device durante la chiamata e si decifrano dopo.
struct UniswapV3Trace {
	Ciphertext<DCRTPoly> term2_fx;	// stampalo come intero a 64 bit, come il CPU
	Ciphertext<DCRTPoly> u_fx;		// 64 bit
	Ciphertext<DCRTPoly> X_post_fx; // 128 bit, gia' << 32
	Ciphertext<DCRTPoly> diff_fx;	// 128 bit
};

/// @brief Specialization of CryptoContext for the DCRTPoly representation.
template <> class CryptoContextImpl<DCRTPoly> {

  public:
	CryptoContextImpl() = default;
	~CryptoContextImpl();

	// ---- Copy ----

	CryptoContextImpl(const CryptoContextImpl&)			   = delete;
	CryptoContextImpl& operator=(const CryptoContextImpl&) = delete;

	// ---- Move ----

	CryptoContextImpl(CryptoContextImpl&&)			  = default;
	CryptoContextImpl& operator=(CryptoContextImpl&&) = delete;

	// ---- Context Setup ----

	/// @brief Enable a particular feature in the context.
	void Enable(PKESchemeFeature feature);
	void Enable(uint32_t featureMask);

	// ---- Getters ----

	uint32_t GetCyclotomicOrder() const;
	uint32_t GetRingDimension() const;
	double GetPreScaleFactor(uint32_t slots);

	// ---- Setters ----
	void SetAutoLoadPlaintexts(bool autoload);
	void SetAutoLoadCiphertexts(bool autoload);
	void SetDevices(const std::vector<int>& devices);

	// ---- Load to devices ----

	/// @brief Load the context to the devices.
	void LoadContext(const PublicKey<DCRTPoly>& publicKey);
	/// @brief Load a plaintext to the devices.
	/// @param pt Plaintext to load.
	void LoadPlaintext(Plaintext& pt);
	/// @brief Load a ciphertext to the devices.
	/// @param ct Ciphertext to load.
	void LoadCiphertext(Ciphertext<DCRTPoly>& ct);

	// ---- Key Generation ----

	/// @brief Generate a public/private key pair.
	KeyPair<DCRTPoly> KeyGen();
	/// @brief Generate the evaluation multiplication keys.
	void EvalMultKeyGen(const PrivateKey<DCRTPoly>& sk);
	/// @brief Generate the evaluation rotation keys for the given steps.
	void EvalRotateKeyGen(const PrivateKey<DCRTPoly>& sk, const std::vector<int32_t>& steps);

	// ---- Bootstrapping ----

	/// @brief Generate bootstrap precomputation data.
	void EvalBootstrapSetup(const std::vector<uint32_t>& levelBudget = { 5, 4 },
	  std::vector<uint32_t> dim1									 = { 0, 0 },
	  uint32_t slots												 = 0,
	  uint32_t correctionFactor										 = 0,
	  bool precompute												 = true,
	  bool btsfirstboot												 = false);
	/// @brief Generate the evaluation bootstrap keys.
	void EvalBootstrapKeyGen(const PrivateKey<DCRTPoly>& secretKey, uint32_t slots);

	// ---- Serialization ----
	static bool SerializeEvalMultKey(std::ostream& ser, const SerType& sertype, const std::string& keyTag = "");
	static bool SerializeEvalAutomorphismKey(std::ostream& ser, const SerType& sertype, const std::string& keyTag = "");

	// ---- Deserialization ----
	bool DeserializeEvalMultKey(std::istream& ser, const SerType& sertype) const;
	bool DeserializeEvalAutomorphismKey(std::istream& ser, const SerType& sertype) const;

	// ---- Encoding ----

	Plaintext
	MakeCKKSPackedPlaintext(const std::vector<std::complex<double>>& value, size_t noiseScaleDeg = 1, uint32_t level = 0, std::shared_ptr<void> params = nullptr, uint32_t slots = 0);
	Plaintext
	MakeCKKSPackedPlaintext(const std::vector<double>& value, size_t noiseScaleDeg = 1, uint32_t level = 0, std::shared_ptr<void> params = nullptr, uint32_t slots = 0);

	// ---- Encryption ----

	Ciphertext<DCRTPoly> Encrypt(Plaintext& pt, const PublicKey<DCRTPoly>& pk);
	Ciphertext<DCRTPoly> Encrypt(const PublicKey<DCRTPoly>& pk, Plaintext& pt);
	Ciphertext<DCRTPoly> Encrypt(Plaintext& pt, const PrivateKey<DCRTPoly>& sk);
	Ciphertext<DCRTPoly> Encrypt(const PrivateKey<DCRTPoly>& sk, Plaintext& pt);
	DecryptResult Decrypt(Ciphertext<DCRTPoly>& ct, const PrivateKey<DCRTPoly>& sk, Plaintext* pt);
	DecryptResult Decrypt(const PrivateKey<DCRTPoly>& sk, Ciphertext<DCRTPoly>& ct, Plaintext* pt);

	// ---- Operations ----

	Ciphertext<DCRTPoly> EvalNegate(const Ciphertext<DCRTPoly>& ct);
	void EvalNegateInPlace(Ciphertext<DCRTPoly>& ct);

	Ciphertext<DCRTPoly> EvalAdd(const Ciphertext<DCRTPoly>& ct1, const Ciphertext<DCRTPoly>& ct2);
	Ciphertext<DCRTPoly> EvalAdd(const Ciphertext<DCRTPoly>& ct, Plaintext& pt);
	Ciphertext<DCRTPoly> EvalAdd(Plaintext& pt, const Ciphertext<DCRTPoly>& ct);
	Ciphertext<DCRTPoly> EvalAdd(const Ciphertext<DCRTPoly>& ct, double scalar);
	Ciphertext<DCRTPoly> EvalAdd(double scalar, const Ciphertext<DCRTPoly>& ct);
	void EvalAddInPlace(Ciphertext<DCRTPoly>& ct1, const Ciphertext<DCRTPoly>& ct2);
	void EvalAddInPlace(Ciphertext<DCRTPoly>& ct1, Plaintext& pt);
	void EvalAddInPlace(Plaintext& pt, Ciphertext<DCRTPoly>& ct1);
	void EvalAddInPlace(Ciphertext<DCRTPoly>& ct1, double scalar);
	void EvalAddInPlace(double scalar, Ciphertext<DCRTPoly>& ct1);
	Ciphertext<DCRTPoly> EvalAddMutable(Ciphertext<DCRTPoly>& ct1, Ciphertext<DCRTPoly>& ct2);
	Ciphertext<DCRTPoly> EvalAddMutable(Ciphertext<DCRTPoly>& ct, Plaintext& pt);
	Ciphertext<DCRTPoly> EvalAddMutable(Plaintext& pt, Ciphertext<DCRTPoly>& ct);
	void EvalAddMutableInPlace(Ciphertext<DCRTPoly>& ct1, Ciphertext<DCRTPoly>& ct2);

	Ciphertext<DCRTPoly> EvalAddMany(const std::vector<Ciphertext<DCRTPoly>>& ciphertexts);
	void EvalAddManyInPlace(std::vector<Ciphertext<DCRTPoly>>& ciphertexts);

	Ciphertext<DCRTPoly> EvalSub(const Ciphertext<DCRTPoly>& ct1, const Ciphertext<DCRTPoly>& ct2);
	Ciphertext<DCRTPoly> EvalSub(const Ciphertext<DCRTPoly>& ct, Plaintext& pt);
	Ciphertext<DCRTPoly> EvalSub(Plaintext& pt, const Ciphertext<DCRTPoly>& ct);
	Ciphertext<DCRTPoly> EvalSub(const Ciphertext<DCRTPoly>& ct, double scalar);
	Ciphertext<DCRTPoly> EvalSub(double scalar, const Ciphertext<DCRTPoly>& ct);
	void EvalSubInPlace(Ciphertext<DCRTPoly>& ct1, const Ciphertext<DCRTPoly>& ct2);
	void EvalSubInPlace(Ciphertext<DCRTPoly>& ct1, double scalar);
	void EvalSubInPlace(double scalar, Ciphertext<DCRTPoly>& ct1);
	Ciphertext<DCRTPoly> EvalSubMutable(Ciphertext<DCRTPoly>& ct1, Ciphertext<DCRTPoly>& ct2);
	Ciphertext<DCRTPoly> EvalSubMutable(Ciphertext<DCRTPoly>& ct, Plaintext& pt);
	Ciphertext<DCRTPoly> EvalSubMutable(Plaintext& pt, Ciphertext<DCRTPoly>& ct);
	void EvalSubMutableInPlace(Ciphertext<DCRTPoly>& ct1, Ciphertext<DCRTPoly>& ct2);

	Ciphertext<DCRTPoly> EvalMult(const Ciphertext<DCRTPoly>& ct1, const Ciphertext<DCRTPoly>& ct2);
	Ciphertext<DCRTPoly> EvalMult(const Ciphertext<DCRTPoly>& ct1, Plaintext& pt);
	Ciphertext<DCRTPoly> EvalMult(Plaintext& pt, const Ciphertext<DCRTPoly>& ct1);
	Ciphertext<DCRTPoly> EvalMult(const Ciphertext<DCRTPoly>& ct1, double scalar);
	Ciphertext<DCRTPoly> EvalMult(double scalar, const Ciphertext<DCRTPoly>& ct1);
	void EvalMultInPlace(Ciphertext<DCRTPoly>& ct1, Plaintext& pt);
	void EvalMultInPlace(Ciphertext<DCRTPoly>& ct1, double scalar);
	void EvalMultInPlace(double scalar, Ciphertext<DCRTPoly>& ct1);
	Ciphertext<DCRTPoly> EvalMultMutable(Ciphertext<DCRTPoly>& ct1, Ciphertext<DCRTPoly>& ct2);
	Ciphertext<DCRTPoly> EvalMultMutable(Ciphertext<DCRTPoly>& ct1, Plaintext& pt);
	Ciphertext<DCRTPoly> EvalMultMutable(Plaintext& pt, Ciphertext<DCRTPoly>& ct1);
	void EvalMultMutableInPlace(Ciphertext<DCRTPoly>& ct1, Ciphertext<DCRTPoly>& ct2);

	Ciphertext<DCRTPoly> EvalSquare(const Ciphertext<DCRTPoly>& ct);
	void EvalSquareInPlace(Ciphertext<DCRTPoly>& ct);
	Ciphertext<DCRTPoly> EvalSquareMutable(Ciphertext<DCRTPoly>& ct);

	Ciphertext<DCRTPoly> EvalRotate(const Ciphertext<DCRTPoly>& ciphertext, int32_t index);
	void EvalRotateInPlace(Ciphertext<DCRTPoly>& ciphertext, int32_t index);

	std::shared_ptr<void> EvalFastRotationPrecompute(const Ciphertext<DCRTPoly>& ct);
	Ciphertext<DCRTPoly> EvalFastRotation(const Ciphertext<DCRTPoly>& ct, int32_t index, uint32_t m, const std::shared_ptr<void>& precomp);
	Ciphertext<DCRTPoly> EvalFastRotationExt(const Ciphertext<DCRTPoly>& ct, int32_t index, const std::shared_ptr<void>& digits, bool addFirst);
	std::vector<Ciphertext<DCRTPoly>> EvalFastRotation(const Ciphertext<DCRTPoly>& ct, const std::vector<int32_t>& indices, uint32_t m, const std::shared_ptr<void>& precomp);
	std::vector<Ciphertext<DCRTPoly>>
	EvalFastRotationExt(const Ciphertext<DCRTPoly>& ct, const std::vector<int32_t>& indices, const std::shared_ptr<void>& digits, bool addFirst);

	Ciphertext<DCRTPoly> EvalChebyshevSeries(const Ciphertext<DCRTPoly>& ct, std::vector<double>& coeffs, double a, double b);
	void EvalChebyshevSeriesInPlace(Ciphertext<DCRTPoly>& ct, std::vector<double>& coeffs, double a, double b);
	static std::vector<double> GetChebyshevCoefficients(std::function<double(double)>& func, double a, double b, size_t degree);
 
	// Batched (SIMD) Chebyshev series evaluation: evaluates a DIFFERENT
	// polynomial per slot. batchOfCoeffs[j] holds the Chebyshev coefficients
	// applied to slot j; batchOfCoeffs.size() must equal ct's slot count, and
	// every inner vector must have the same size (same degree).
	Ciphertext<DCRTPoly> EvalChebyshevSeriesPSBatch(const Ciphertext<DCRTPoly>& ct,
	                                                 const std::vector<std::vector<double>>& batchOfCoeffs,
	                                                 double a, double b);
	void EvalChebyshevSeriesPSBatchInPlace(Ciphertext<DCRTPoly>& ct,
	                                        const std::vector<std::vector<double>>& batchOfCoeffs,
	                                        double a, double b);
 
	// Like EvalChebyshevSeriesPSBatch, but takes a small set of coefficient
	// sets and repeats them cyclically to fill all slots (slot j uses
	// coefficientSets[j % coefficientSets.size()]).
	Ciphertext<DCRTPoly> EvalChebyshevSeriesPSBatchRepeated(const Ciphertext<DCRTPoly>& ct,
	                                                         const std::vector<std::vector<double>>& coefficientSets,
	                                                         double a, double b);
	void EvalChebyshevSeriesPSBatchRepeatedInPlace(Ciphertext<DCRTPoly>& ct,
	                                                const std::vector<std::vector<double>>& coefficientSets,
	                                                double a, double b);
	std::shared_ptr<void> EvalChebyshevSeriesPSBatchPrecompute(const Ciphertext<DCRTPoly>& ct,
	                                                            const std::vector<std::vector<double>>& batchOfCoeffs,
	                                                            double a, double b);
	Ciphertext<DCRTPoly> EvalChebyshevSeriesPSBatchApply(const Ciphertext<DCRTPoly>& ct, const std::shared_ptr<void>& precomp,
	                                                      const std::vector<std::vector<double>>& batchOfCoeffs, double a, double b);
	void EvalChebyshevSeriesPSBatchApplyInPlace(Ciphertext<DCRTPoly>& ct, const std::shared_ptr<void>& precomp,
	                                             const std::vector<std::vector<double>>& batchOfCoeffs, double a, double b);

 

	// Binary operations
	Ciphertext<DCRTPoly> EvalBinaryOR(const Ciphertext<DCRTPoly>& ct1, const Ciphertext<DCRTPoly>& ct2);

	// Integer operations
	Ciphertext<DCRTPoly> EvalAddInteger(const Ciphertext<DCRTPoly>& ct1, const Ciphertext<DCRTPoly>& ct2, int bits);
	Ciphertext<DCRTPoly> EvalEqualInteger(const Ciphertext<DCRTPoly>& ct1, const Ciphertext<DCRTPoly>& ct2, int bits, int zslots, std::vector<double> coeffsSinc, int depth);
	Ciphertext<DCRTPoly> EvalMultInteger(const Ciphertext<DCRTPoly>& ct1, const Ciphertext<DCRTPoly>& ct2, int bits, int zslots, bool overflow);
	// `pk` is needed because the Newton-Raphson refinement in evalIntegerDivision
	// needs a genuinely-encrypted constant "1" ciphertext (not just a plaintext
	// mask) to finish a two's-complement negation -- see DivIntegerPrecomputations.
	// Call DivIntegerPrecomputations(ct1, bits, zslots, pk, noise) once before the
	// first EvalMultDivision call with a given (bits, zslots) combination.
	Ciphertext<DCRTPoly> EvalMultDivision(const Ciphertext<DCRTPoly>& ct1, const Ciphertext<DCRTPoly>& ct2, int bits, int zslots, const PublicKey<DCRTPoly>& pk);

	
	// Support
	void ProcessMultiplications(std::vector<std::vector<double>> coeffs, const Ciphertext<DCRTPoly>& c);
	Ciphertext<DCRTPoly> CsaSum(const Ciphertext<DCRTPoly>& a, const Ciphertext<DCRTPoly>& b, const Ciphertext<DCRTPoly>& c);
	Ciphertext<DCRTPoly> CsaCarry(const Ciphertext<DCRTPoly>& a, const Ciphertext<DCRTPoly>& b, const Ciphertext<DCRTPoly>& c);
	Ciphertext<DCRTPoly> MajorityBit(const Ciphertext<DCRTPoly>& a, const Ciphertext<DCRTPoly>& b, const Ciphertext<DCRTPoly>& c);
	Ciphertext<DCRTPoly> BinToDec(const Ciphertext<DCRTPoly>& ct, int repetitions);
	void IntegerMultPrecomputations(const Ciphertext<DCRTPoly>& c, int bits, int repetitions, int slots, int noise);
    void ProcessArrayPrecomputations(const Ciphertext<DCRTPoly>& c, int bits, int slots, int noise );
	// One-time setup for EvalMultDivision at a given (bits, zslots): encrypts
	// the constant-1 ciphertext EvalMultDivision needs, and stashes the raw
	// LUT coefficient columns so EvalMultDivision can forward them into
	// evalIntegerDivision's lazy, self-validating PSBatch precompute (see
	// the long comment on evalIntegerDivision in IntegerOperations.cuh for
	// why that precompute can't safely be done ahead of time against a
	// hand-picked model ciphertext). `c` is only used as a template
	// ciphertext for the constant-1 encryption (its .cc_/slots), the same
	// way ProcessArrayPrecomputations uses its `c` argument -- its data is
	// not read.
	void DivIntegerPrecomputations(const Ciphertext<DCRTPoly>& c, int bits, int zslots, const PublicKey<DCRTPoly>& pk, int noise,
	  const std::vector<std::vector<double>>& bitLengthCoeffs, const std::vector<std::vector<double>>& reciprocalCoeffs);

	// EvalSquareRootInteger: ciphertext integer square root. `pk` is needed
	// for the same reason EvalMultDivision needs it -- the tail of
	// square_root_integer needs three genuinely-encrypted constants (ONE_FP,
	// SQRT2_FP, bits+1), not just plaintext masks, to feed into
	// evalIntegerSub/plain multiplications the way the CPU's
	// encrypt_multi_int(...) calls do. Call
	// SquareRootPrecomputations(..., bits, zslots, ...) once before the first
	// EvalSquareRootInteger call with a given (bits, zslots) combination.
	Ciphertext<DCRTPoly> EvalSquareRootInteger(const Ciphertext<DCRTPoly>& ct, int bits, int zslots, const PublicKey<DCRTPoly>& pk);

	// One-time setup for EvalSquareRootInteger at a given (bits, zslots):
	// encrypts the three constant ciphertexts (ONE_FP, SQRT2_FP, bits+1)
	// EvalSquareRootInteger needs, and stashes the raw LUT coefficient
	// columns so EvalSquareRootInteger can forward them into
	// evalIntegerSquareRoot's lazy, self-validating PSBatch precompute (see
	// the long comment on evalIntegerSquareRoot in IntegerOperations.cuh for
	// why that precompute can't safely be done ahead of time against a
	// hand-picked model ciphertext). `c` is only used as a template
	// ciphertext for the constant encryptions (its .cc_/slots), the same way
	// DivIntegerPrecomputations uses its `c` argument -- its data is not
	// read. `bitLengthCoeffs`/`newtonSeedCoeffs` are the raw coefficient
	// columns square_root_integer reads from
	// "../coeffs/LUTs/<bits> bits/lut/p{1..7}-norm-247-LUT-DIVISION.txt" (+
	// garbage padding to `bits` columns) and
	// "../coeffs/LUTs/<bits> bits/square-root/LUT-SQUARE-ROOT-<bits>-BITS-<i>.txt"
	// (+ garbage padding to bits*bits/2 columns, WITH the bits==64/bits==128
	// in-place column patches from square_root_integer already applied --
	// see preprocessSquareRootLUTs's doc comment). File I/O and the
	// bits==64/128 patch are left to the caller, exactly like
	// DivIntegerPrecomputations leaves file I/O to its caller.
	void SquareRootPrecomputations(const Ciphertext<DCRTPoly>& c, int bits, int zslots, const PublicKey<DCRTPoly>& pk, int noise,
	  const std::vector<std::vector<double>>& bitLengthCoeffs, const std::vector<std::vector<double>>& newtonSeedCoeffs);

	// ---- Divisione per costante in chiaro (CPU: div_integer(Ctxt, uint128_t, bits, zslots)) ----

	/// @brief Cifra una volta (livello 0, con `pk`) il reciproco di ciascun divisore e lo
	/// mette in cache per (bits, zslots, den). `c` serve solo come modello (slots).
	/// Rotation keys necessarie per ogni den: -bits e bits + bit_width(den).
	void PlainDivisionPrecomputations(const Ciphertext<DCRTPoly>& c, int bits, int zslots, const PublicKey<DCRTPoly>& pk, int noise, const std::vector<__uint128_t>& divisors);
	/// @brief floor(ct / den) su interi a `bits` bit. Richiede PlainDivisionPrecomputations.
	Ciphertext<DCRTPoly> EvalDivIntegerPlain(const Ciphertext<DCRTPoly>& ct, __uint128_t den, int bits, int zslots);

	/// @brief ct1 - ct2 su interi a `bits` bit (risultato mascherato ai `bits` bit bassi).
	/// carryIn = true -> sottrazione esatta; false -> ct1 - ct2 - 1 (come square_root_integer).
	Ciphertext<DCRTPoly> EvalSubInteger(const Ciphertext<DCRTPoly>& ct1, const Ciphertext<DCRTPoly>& ct2, int bits, int zslots, bool carryIn = true);

	// ---- Esempio Uniswap v3 interamente su GPU ----

	/// @brief Rotazioni specifiche dell'esempio (21, -32, -128 e 128 + bit_width dei tre
	/// divisori in chiaro). Vanno passate a EvalRotateKeyGen PRIMA di LoadContext, insieme
	/// a quelle che usi gia' per EvalMultInteger / EvalMultDivision a 128 bit.
	static std::vector<int32_t> GetUniswapV3RotationIndices();

	/// @brief Setup una tantum dell'esempio: cifra i reciproci di 5^10, 5^11 e 997.
	/// Prerequisiti (come per i test a 128 bit): ProcessArrayPrecomputations(c, 128, ...)
	/// come ULTIMA chiamata di quel tipo, ProcessMultiplications(...) e
	/// DivIntegerPrecomputations(c, 128, zslots, pk, noise, ...).
	void UniswapV3Precomputations(const Ciphertext<DCRTPoly>& c, const PublicKey<DCRTPoly>& pk, int noise, int zslots = 1);

	/// @brief Tutto experiment_uniswap_v3() in una sola chiamata: gli input vengono caricati
	/// sul device (no-op se lo sono gia'), poi l'intera catena di operazioni gira sulla GPU
	/// senza passare dall'host. Restituisce amount_fx (128 bit). Se `trace` != nullptr ci
	/// mette anche gli intermedi, da decifrare dopo per il confronto con il CPU.
	Ciphertext<DCRTPoly> EvalUniswapV3Example(const UniswapV3Inputs& inputs, UniswapV3Trace* trace = nullptr);

	Ciphertext<DCRTPoly> Multiplier4bits(const Ciphertext<DCRTPoly>& ctxtA, const Ciphertext<DCRTPoly>& ctxtB, int repetitions, std::vector<std::vector<double>> coeffs);


	Ciphertext<DCRTPoly> Rescale(const Ciphertext<DCRTPoly>& ciphertext);
	void RescaleInPlace(Ciphertext<DCRTPoly>& ciphertext);

	static void SetLevel(Ciphertext<DCRTPoly>& ct, size_t level);

	Ciphertext<DCRTPoly> EvalBootstrap(const Ciphertext<DCRTPoly>& ciphertext, uint32_t numIterations = 1, uint32_t precision = 0, bool prescaled = false);
	void EvalBootstrapInPlace(Ciphertext<DCRTPoly>& ciphertext, uint32_t numIterations = 1, uint32_t precision = 0, bool prescaled = false);
    Ciphertext<DCRTPoly> EvalBootstrapStCFirst(const Ciphertext<DCRTPoly>& ciphertext, uint32_t numIterations = 1, uint32_t precision = 0, bool prescaled = false);
	void EvalBootstrapStCFirstInPlace(Ciphertext<DCRTPoly>& ciphertext, uint32_t numIterations = 1, uint32_t precision = 0, bool prescaled = false);
	Ciphertext<DCRTPoly> EvalBootstrapStCFirstBits(const Ciphertext<DCRTPoly>& ciphertext, uint32_t numIterations = 1, uint32_t precision = 0, bool prescaled = false);
	void EvalBootstrapStCFirstBitsInPlace(Ciphertext<DCRTPoly>& ciphertext, uint32_t numIterations = 1, uint32_t precision = 0, bool prescaled = false);


	Ciphertext<DCRTPoly> AccumulateSum(const Ciphertext<DCRTPoly>& ct, int slots, int stride = 1);
	void AccumulateSumInPlace(Ciphertext<DCRTPoly>& ct, int slots, int stride = 1);
	void AccumulateSumInPlace(Ciphertext<DCRTPoly>& ct, int slots, int stride, int start);

	void ConvolutionTransformInPlace(Ciphertext<DCRTPoly>& ct, int gStep, int bStep, const std::vector<Plaintext>& pts, const std::vector<int>& indexes, int stride = 1, int rowSize = 0);

	void SpecialConvolutionTransformInPlace(Ciphertext<DCRTPoly>& ct,
	  int gStep,
	  int bStep,
	  const std::vector<Plaintext>& pts,
	  Plaintext& mask,
	  const std::vector<int>& indexes,
	  int stride			 = 1,
	  int maskRotationStride = 1,
	  int rowSize			 = 0);

  public:
	// ---- Internal State ----

	std::any cpu;
	std::any gpu;
	/// @brief Whether the context has been loaded to the devices.
	bool loaded = false;
	/// @brief List of devices the context is loaded on.
	std::vector<int> devices = { 0 };
	/// @brief Whether plaintexts should be automatically loaded to the device upon encryption.
	bool auto_load_plaintexts = false;
	/// @brief Whether ciphertexts should be automatically loaded to the device upon creation.
	bool auto_load_ciphertexts = true;
	/// @brief Self reference to enable shared_from_this-like behavior.
	std::weak_ptr<CryptoContextImpl<DCRTPoly>> self_reference;
	/// @brief Multiplicative depth of the context.
	uint32_t multiplicative_depth = 0;
	/// @brief Rotation indexes for which rotation keys are available.
	std::vector<int32_t> rotation_indexes;
	/// @brief Bootstrap slots available.
	std::vector<uint32_t> slots_bootstrap;
	/// @brief Secret key distribution.
	SecretKeyDist keyDist = UNIFORM_TERNARY;

	/// @brief Cache of the constant-1 ciphertext used by EvalMultDivision's
	/// Newton-Raphson correction step, keyed by (bits << 32 | zslots) so a
	/// distinct constant is kept per (bits, zslots) combination. Populated by
	/// DivIntegerPrecomputations; consumed by EvalMultDivision.
	std::unordered_map<uint64_t, Ciphertext<DCRTPoly>> div_integer_one_cache;

	/// @brief Cache of the raw LUT coefficient columns (bitLength, reciprocal)
	/// needed to (re)run evalIntegerDivision's lazy PSBatch precompute, keyed
	/// the same way as div_integer_one_cache. Populated by
	/// DivIntegerPrecomputations; forwarded by EvalMultDivision into
	/// evalIntegerDivision on every call (the precompute itself is only
	/// actually redone when the cached LUT's level/NoiseLevel no longer
	/// matches -- see evalIntegerDivision).
	std::unordered_map<uint64_t, std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>>> div_integer_coeffs_cache;

	/// @brief Cache of the three constant ciphertexts (ONE_FP, SQRT2_FP,
	/// bits+1, in that order) used by EvalSquareRootInteger's Newton-Raphson
	/// tail, keyed the same way as div_integer_one_cache. Populated by
	/// SquareRootPrecomputations; consumed by EvalSquareRootInteger.
	std::unordered_map<uint64_t, std::array<Ciphertext<DCRTPoly>, 3>> square_root_constants_cache;

	/// @brief Cache of the raw LUT coefficient columns (bitLength,
	/// newtonSeed) needed to (re)run evalIntegerSquareRoot's lazy PSBatch
	/// precompute, keyed the same way as div_integer_coeffs_cache. Populated
	/// by SquareRootPrecomputations; forwarded by EvalSquareRootInteger into
	/// evalIntegerSquareRoot on every call (the precompute itself is only
	/// actually redone when the cached LUT's level/NoiseLevel no longer
	/// matches -- see evalIntegerSquareRoot).
	std::unordered_map<uint64_t, std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>>> square_root_coeffs_cache;

	/// @brief Reciproco cifrato di un divisore in chiaro + bit_width(den).
	struct PlainDivisorEntry {
		Ciphertext<DCRTPoly> reciprocal;
		int bitLength = 0;
	};

	/// @brief Cache di PlainDivisionPrecomputations, chiave (bits, zslots, den_hi, den_lo).
	std::map<std::tuple<int, int, uint64_t, uint64_t>, PlainDivisorEntry> plain_division_cache;

	/// @brief zslots con cui e' stato chiamato UniswapV3Precomputations (0 = non ancora).
	int uniswap_v3_zslots = 0;

	// ---- Copy helpers ----

	uint32_t CopyDeviceCiphertext(const CiphertextImpl<DCRTPoly>& ct);

	// --- Map Handling ----

	/// @brief  Registry of plaintexts stored on the GPU (opaque types).
	std::unordered_map<uint32_t, std::shared_ptr<void>> device_plaintexts;
	/// @brief  Registry of ciphertexts stored on the GPU (opaque types).
	std::unordered_map<uint32_t, std::shared_ptr<void>> device_ciphertexts;
	/// @brief Next available handle for GPU objects. Zero is reserved as a null handle.
	uint32_t next_gpu_handle = 1;

	uint32_t RegisterDevicePlaintext(std::shared_ptr<void>&& p);
	uint32_t RegisterDeviceCiphertext(std::shared_ptr<void>&& c);
	std::shared_ptr<void>& GetDevicePlaintext(uint32_t handle);
	std::shared_ptr<void>& GetDeviceCiphertext(uint32_t handle);
	bool EvictDevicePlaintext(uint32_t handle);
	bool EvictDeviceCiphertext(uint32_t handle);

	void Synchronize() const;

	static std::vector<int> GetConvolutionTransformRotationIndices(int rowSize, int bStep, int stride, uint32_t gStep);
};

} // namespace fideslib

#endif // API_CRYPTOCONTEXT_HPP