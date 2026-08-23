// 12_chebyshev_batch_precompute_apply.cpp
//
// Verifica che EvalChebyshevSeriesPSBatch (diretto) e
// EvalChebyshevSeriesPSBatchPrecompute + EvalChebyshevSeriesPSBatchApply
// diano risultati numericamente identici sullo stesso batch di funzioni,
// sia sullo STESSO ciphertext (il caso banale) sia su un SECONDO
// ciphertext diverso ma allo stesso livello/NoiseLevel (il caso reale
// d'uso: riusare un precompute su input diversi).
//
// Stessa base di 06_chebyshev_batch.cpp: 4 funzioni (sigmoid, tanh, x²/4,
// gauss) cicliche sui 16 slot.

#include <fideslib.hpp>

#include <cmath>
#include <functional>
#include <iomanip>
#include <iostream>
#include <vector>

using namespace fideslib;

double f_sigmoid(double x) {
	return 1.0 / (1.0 + std::exp(-x));
}
double f_tanh(double x) {
	return std::tanh(x);
}
double f_square(double x) {
	return 0.25 * x * x;
}
double f_gauss(double x) {
	return std::exp(-x * x);
}

int main() {
	uint32_t multDepth			 = 8;
	uint32_t batchSize			 = 16;
	uint32_t ring_dim			 = 1 << 12;
	ScalingTechnique rescaleTech = FLEXIBLEAUTO;

	uint32_t dcrtBits = 59;
	uint32_t firstMod = 60;
	uint32_t dnum	  = 3;

	CCParams<CryptoContextCKKSRNS> parameters;
	parameters.SetSecurityLevel(SecurityLevel::HEStd_NotSet);
	parameters.SetRingDim(ring_dim);
	parameters.SetMultiplicativeDepth(multDepth);
	parameters.SetScalingModSize(dcrtBits);
	parameters.SetScalingTechnique(rescaleTech);
	parameters.SetFirstModSize(firstMod);
	parameters.SetKeySwitchTechnique(HYBRID);
	parameters.SetNumLargeDigits(dnum);
	parameters.SetBatchSize(batchSize);
	parameters.SetDevices({ 0 });
	parameters.SetPlaintextAutoload(false);
	parameters.SetCiphertextAutoload(true);

	CryptoContext<DCRTPoly> cc = GenCryptoContext(parameters);

	cc->Enable(PKE);
	cc->Enable(KEYSWITCH);
	cc->Enable(LEVELEDSHE);
	cc->Enable(ADVANCEDSHE);
	cc->Enable(FHE);

	std::cout << "CKKS scheme using ring dimension: " << cc->GetRingDimension() << std::endl;

	auto keys = cc->KeyGen();
	cc->EvalMultKeyGen(keys.secretKey);
	cc->LoadContext(keys.publicKey);

	double lowerBound = -2.0;
	double upperBound = 2.0;

	// ---- Batch di 4 funzioni, cicliche sui 16 slot ----
	std::vector<std::function<double(double)>> functions = { f_sigmoid, f_tanh, f_square, f_gauss };
	std::vector<std::string> functionNames				= { "sigmoid", "tanh", "square/4", "gauss" };
	size_t chebyDegree										= 9;

	std::vector<std::vector<double>> batchOfCoeffs(batchSize);
	std::vector<size_t> slotFunctionIdx(batchSize);
	for (uint32_t i = 0; i < batchSize; ++i) {
		size_t fidx			   = i % functions.size();
		slotFunctionIdx[i]	   = fidx;
		auto fcopy			   = functions[fidx];
		batchOfCoeffs[i]	   = cc->GetChebyshevCoefficients(fcopy, lowerBound, upperBound, chebyDegree);
	}

	// ---- Due input DIVERSI, entrambi cifrati allo stesso livello (appena
	//      cifrati) -- il caso reale d'uso: un precompute costruito da un
	//      ciphertext "modello" viene riusato su un ciphertext diverso. ----
	std::vector<double> xValuesA(batchSize), xValuesB(batchSize);
	std::vector<double> expectedA(batchSize), expectedB(batchSize);
	for (uint32_t i = 0; i < batchSize; ++i) {
		xValuesA[i]  = lowerBound + (upperBound - lowerBound) * static_cast<double>(i) / (batchSize - 1);
		xValuesB[i]  = upperBound - (upperBound - lowerBound) * static_cast<double>(i) / (batchSize - 1); // reversed
		expectedA[i] = functions[slotFunctionIdx[i]](xValuesA[i]);
		expectedB[i] = functions[slotFunctionIdx[i]](xValuesB[i]);
	}

	auto ctA = cc->Encrypt(keys.publicKey, cc->MakeCKKSPackedPlaintext(xValuesA));
	auto ctB = cc->Encrypt(keys.publicKey, cc->MakeCKKSPackedPlaintext(xValuesB));

	// ---- Percorso 1: EvalChebyshevSeriesPSBatch diretto, su A e su B ----
	auto ctDirectA = cc->EvalChebyshevSeriesPSBatch(ctA, batchOfCoeffs, lowerBound, upperBound);
	auto ctDirectB = cc->EvalChebyshevSeriesPSBatch(ctB, batchOfCoeffs, lowerBound, upperBound);

	// ---- Percorso 2: Precompute (usando A come "modello") + Apply su A e su B ----
	auto precomp   = cc->EvalChebyshevSeriesPSBatchPrecompute(ctA, batchOfCoeffs, lowerBound, upperBound);
	auto ctApplyA  = cc->EvalChebyshevSeriesPSBatchApply(ctA, precomp, batchOfCoeffs, lowerBound, upperBound);
	auto ctApplyB  = cc->EvalChebyshevSeriesPSBatchApply(ctB, precomp, batchOfCoeffs, lowerBound, upperBound);

	auto decryptAndExtract = [&](const Ciphertext<DCRTPoly>& ct) {
		Plaintext pt;
		cc->Decrypt(keys.secretKey, ct, &pt);
		pt->SetLength(batchSize);
		return pt->GetRealPackedValue();
	};

	auto directA = decryptAndExtract(ctDirectA);
	auto directB = decryptAndExtract(ctDirectB);
	auto applyA  = decryptAndExtract(ctApplyA);
	auto applyB  = decryptAndExtract(ctApplyB);

	std::cout << std::endl << "==== Ciphertext A (same input Precompute was built from) ====" << std::endl;
	std::cout << std::setw(8) << "slot" << std::setw(10) << "func" << std::setw(10) << "x" << std::setw(12) << "expected" << std::setw(12)
			  << "direct" << std::setw(12) << "apply" << std::setw(14) << "|direct-exp|" << std::setw(14) << "|apply-exp|" << std::setw(16)
			  << "|apply-direct|" << std::endl;

	double maxErrDirectA = 0.0, maxErrApplyA = 0.0, maxDiffA = 0.0;
	for (uint32_t i = 0; i < batchSize; ++i) {
		double eDirect = std::abs(directA[i] - expectedA[i]);
		double eApply  = std::abs(applyA[i] - expectedA[i]);
		double diff	   = std::abs(applyA[i] - directA[i]);
		maxErrDirectA  = std::max(maxErrDirectA, eDirect);
		maxErrApplyA   = std::max(maxErrApplyA, eApply);
		maxDiffA	   = std::max(maxDiffA, diff);
		std::cout << std::fixed << std::setprecision(6) << std::setw(8) << i << std::setw(10) << functionNames[slotFunctionIdx[i]] << std::setw(10)
				  << xValuesA[i] << std::setw(12) << expectedA[i] << std::setw(12) << directA[i] << std::setw(12) << applyA[i] << std::setw(14)
				  << eDirect << std::setw(14) << eApply << std::setw(16) << diff << std::endl;
	}

	std::cout << std::endl << "==== Ciphertext B (different input, reusing the same Precompute) ====" << std::endl;
	std::cout << std::setw(8) << "slot" << std::setw(10) << "func" << std::setw(10) << "x" << std::setw(12) << "expected" << std::setw(12)
			  << "direct" << std::setw(12) << "apply" << std::setw(14) << "|direct-exp|" << std::setw(14) << "|apply-exp|" << std::setw(16)
			  << "|apply-direct|" << std::endl;

	double maxErrDirectB = 0.0, maxErrApplyB = 0.0, maxDiffB = 0.0;
	for (uint32_t i = 0; i < batchSize; ++i) {
		double eDirect = std::abs(directB[i] - expectedB[i]);
		double eApply  = std::abs(applyB[i] - expectedB[i]);
		double diff	   = std::abs(applyB[i] - directB[i]);
		maxErrDirectB  = std::max(maxErrDirectB, eDirect);
		maxErrApplyB   = std::max(maxErrApplyB, eApply);
		maxDiffB	   = std::max(maxDiffB, diff);
		std::cout << std::fixed << std::setprecision(6) << std::setw(8) << i << std::setw(10) << functionNames[slotFunctionIdx[i]] << std::setw(10)
				  << xValuesB[i] << std::setw(12) << expectedB[i] << std::setw(12) << directB[i] << std::setw(12) << applyB[i] << std::setw(14)
				  << eDirect << std::setw(14) << eApply << std::setw(16) << diff << std::endl;
	}

	std::cout << std::endl << "==== Summary ====" << std::endl;
	std::cout << "A: max|direct-expected|=" << maxErrDirectA << "  max|apply-expected|=" << maxErrApplyA << "  max|apply-direct|=" << maxDiffA
			  << std::endl;
	std::cout << "B: max|direct-expected|=" << maxErrDirectB << "  max|apply-expected|=" << maxErrApplyB << "  max|apply-direct|=" << maxDiffB
			  << std::endl;

	// Tolerance for direct-vs-apply agreement is much tighter than the
	// approximation-error tolerance: both paths run the EXACT same
	// arithmetic, just with plaintexts sourced differently (fresh-encode vs
	// cached-copy), so they should match to numerical noise, not to
	// Chebyshev-approximation-error levels.
	constexpr double agreementTolerance   = 1e-6;
	constexpr double approximationTolerance = 1e-2;

	bool ok = (maxErrDirectA < approximationTolerance) && (maxErrApplyA < approximationTolerance) && (maxDiffA < agreementTolerance) &&
	          (maxErrDirectB < approximationTolerance) && (maxErrApplyB < approximationTolerance) && (maxDiffB < agreementTolerance);

	std::cout << std::endl
	          << (ok ? "PASS" : "FAIL") << ": direct and Precompute+Apply "
	          << (ok ? "agree with each other and with the expected values, on both ciphertexts."
	                 : "DO NOT agree -- check the precompute/apply cache logic.")
	          << std::endl;

	return ok ? 0 : 1;
}
