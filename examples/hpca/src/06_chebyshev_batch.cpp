// 06_chebyshev_batch.cpp
//
// Verifica di FIDESlib::CKKS::evalChebyshevSeriesPSBatch /
// cc->EvalChebyshevSeriesPSBatch(...).
//
// Idea: ogni slot del ciphertext riceve un polinomio di Chebyshev DIVERSO
// (una funzione diversa), tutti con lo stesso grado. Il risultato cifrato
// viene decifrato e confrontato con la valutazione in chiaro (double) delle
// stesse funzioni sugli stessi input, per verificare che
// EvalChebyshevSeriesPSBatch calcoli davvero "un polinomio per slot" e non,
// ad esempio, applichi lo stesso polinomio a tutti gli slot.
//
// Richiede le patch descritte in CryptoContext.hpp.patch.txt /
// CryptoContext.cpp.patch.txt applicate all'API FIDESlib, e i file
// ApproxModEvalBatch.cuh / ApproxModEvalBatch.cu copiati in src/CKKS/.

#include <fideslib.hpp>

#include <cmath>
#include <functional>
#include <iomanip>
#include <iostream>
#include <vector>

using namespace fideslib;

// ---- Funzioni target, una per "famiglia" di slot -------------------------

double f_sigmoid(double x) {
	return 1.0 / (1.0 + std::exp(-x));
}
double f_tanh(double x) {
	return std::tanh(x);
}
double f_square(double x) {
	return 0.25 * x * x; // scalata per restare in range su [-2, 2]
}
double f_gauss(double x) {
	return std::exp(-x * x);
}

int main() {
	// =====================================================
	// Step 1: Parametri CKKS (stessi di 02_polynomials.cpp).
	// =====================================================

	uint32_t multDepth			 = 8;
	uint32_t batchSize			 = 16; // numero di slot; deve combaciare con ctxt.slots lato core
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

	// =====================================================
	// Step 2: CryptoContext + chiavi + caricamento su GPU.
	// =====================================================

	CryptoContext<DCRTPoly> cc = GenCryptoContext(parameters);

	cc->Enable(PKE);
	cc->Enable(KEYSWITCH);
	cc->Enable(LEVELEDSHE);
	cc->Enable(ADVANCEDSHE);
	cc->Enable(FHE);

	std::cout << "CKKS scheme using ring dimension: " << cc->GetRingDimension() << std::endl << std::endl;

	auto keys = cc->KeyGen();
	cc->EvalMultKeyGen(keys.secretKey);

	cc->LoadContext(keys.publicKey);

	// =====================================================
	// Step 3: Input — un valore x diverso per ogni slot.
	// =====================================================

	double lowerBound = -2.0;
	double upperBound = 2.0;

	std::vector<double> xValues(batchSize);
	for (uint32_t i = 0; i < batchSize; ++i) {
		xValues[i] = lowerBound + (upperBound - lowerBound) * static_cast<double>(i) / (batchSize - 1);
	}

	// =====================================================
	// Step 4: Assegna una funzione diversa ad ogni slot
	//         (ciclando su 4 famiglie) e calcola i coefficienti
	//         di Chebyshev di CIASCUNA, tutti con lo stesso grado.
	// =====================================================

	std::vector<std::function<double(double)>> functions = { f_sigmoid, f_tanh, f_square, f_gauss };
	std::vector<std::string> functionNames				= { "sigmoid", "tanh", "square/4", "gauss" };

	size_t chebyDegree = 9; // stesso grado per tutte le funzioni del batch

	std::vector<std::vector<double>> batchOfCoeffs(batchSize);
	std::vector<size_t> slotFunctionIdx(batchSize);

	for (uint32_t i = 0; i < batchSize; ++i) {
		size_t fidx			 = i % functions.size();
		slotFunctionIdx[i]	 = fidx;
		std::function<double(double)> fcopy = functions[fidx];
		batchOfCoeffs[i]	 = cc->GetChebyshevCoefficients(fcopy, lowerBound, upperBound, chebyDegree);
	}

	// Valore atteso in chiaro: f_{slot}(x_{slot}).
	std::vector<double> expectedY(batchSize);
	for (uint32_t i = 0; i < batchSize; ++i) {
		expectedY[i] = functions[slotFunctionIdx[i]](xValues[i]);
	}

	// =====================================================
	// Step 5: Cifra l'input.
	// =====================================================

	Plaintext ptxtX = cc->MakeCKKSPackedPlaintext(xValues);
	auto ctX		= cc->Encrypt(keys.publicKey, ptxtX);

	std::cout << "Input level: " << ctX->GetLevel() << std::endl;

	// =====================================================
	// Step 6: Valutazione batch (un polinomio Chebyshev per slot).
	// =====================================================

	auto ctBatchResult = cc->EvalChebyshevSeriesPSBatch(ctX, batchOfCoeffs, lowerBound, upperBound);

	std::cout << "EvalChebyshevSeriesPSBatch output level: " << ctBatchResult->GetLevel() << std::endl;

	Plaintext ptxtBatchResult;
	cc->Decrypt(keys.secretKey, ctBatchResult, &ptxtBatchResult);
	ptxtBatchResult->SetLength(batchSize);
	auto batchResults = ptxtBatchResult->GetRealPackedValue();

	// =====================================================
	// Step 7: Valutazione di riferimento — un ciphertext per
	//         funzione, usando EvalChebyshevSeries "classico"
	//         (un solo polinomio su tutti gli slot), poi si
	//         estrae dal risultato solo lo slot che interessa.
	//         Serve a validare in modo indipendente lo stesso
	//         cammino (linear transform + Paterson-Stockmeyer)
	//         già testato per il caso non-batch.
	// =====================================================

	std::vector<double> referenceResults(batchSize);
	for (size_t fidx = 0; fidx < functions.size(); ++fidx) {
		std::function<double(double)> fcopy = functions[fidx];
		auto coeffsSingle					 = cc->GetChebyshevCoefficients(fcopy, lowerBound, upperBound, chebyDegree);

		auto ctSingleResult = cc->EvalChebyshevSeries(ctX, coeffsSingle, lowerBound, upperBound);

		Plaintext ptxtSingleResult;
		cc->Decrypt(keys.secretKey, ctSingleResult, &ptxtSingleResult);
		ptxtSingleResult->SetLength(batchSize);
		auto singleResults = ptxtSingleResult->GetRealPackedValue();

		for (uint32_t i = 0; i < batchSize; ++i) {
			if (slotFunctionIdx[i] == fidx) {
				referenceResults[i] = singleResults[i];
			}
		}
	}

	// =====================================================
	// Step 8: Confronto dei risultati.
	// =====================================================

	std::cout << std::endl << "==== Results Comparison ====" << std::endl << std::endl;
	std::cout << std::setw(8) << "slot" << std::setw(10) << "func" << std::setw(10) << "x" << std::setw(12) << "expected(f64)" << std::setw(14)
			  << "batch(GPU)" << std::setw(14) << "single(GPU)" << std::setw(12) << "errBatch" << std::setw(12) << "errSingle" << std::endl;
	std::cout << std::string(102, '-') << std::endl;

	double maxErrBatch = 0.0, maxErrCrossCheck = 0.0;
	for (uint32_t i = 0; i < batchSize; ++i) {
		double errBatch		 = std::abs(batchResults[i] - expectedY[i]);
		double errCrossCheck = std::abs(batchResults[i] - referenceResults[i]);
		maxErrBatch			 = std::max(maxErrBatch, errBatch);
		maxErrCrossCheck	 = std::max(maxErrCrossCheck, errCrossCheck);

		std::cout << std::fixed << std::setprecision(4) << std::setw(8) << i << std::setw(10) << functionNames[slotFunctionIdx[i]] << std::setw(10)
				  << xValues[i] << std::setw(12) << expectedY[i] << std::setw(14) << batchResults[i] << std::setw(14) << referenceResults[i]
				  << std::setw(12) << errBatch << std::setw(12) << errCrossCheck << std::endl;
	}

	std::cout << std::endl;
	std::cout << "Max |batch - expected(f64)|        : " << maxErrBatch << std::endl;
	std::cout << "Max |batch - single-function(GPU)|  : " << maxErrCrossCheck << std::endl;

	constexpr double tolerance = 1e-2; // il grado 9 su [-2,2] non è altissima precisione, tolleranza larga
	bool ok						= (maxErrBatch < tolerance) && (maxErrCrossCheck < tolerance);

	std::cout << std::endl << (ok ? "PASS" : "FAIL") << ": EvalChebyshevSeriesPSBatch "
			  << (ok ? "matches the per-function reference within tolerance." : "DOES NOT match the reference — check the port.") << std::endl;

	return ok ? 0 : 1;
}
