// bootstrap_stc_first.cpp
//
// Test minimale per FIDESlib::CKKS::BootstrapStCFirst / cc->EvalBootstrapStCFirst.
//
// Rispecchia SimpleBootstrapExample() di bootstrap.cpp (stesso stile di
// parametri: full packing, UNIFORM_TERNARY, FLEXIBLEAUTO), con due differenze:
//   1. EvalBootstrapSetup(..., /*btsfirstboot=*/true) invece di false.
//   2. EvalBootstrapStCFirstInPlace(...) invece di EvalBootstrapInPlace(...).
//
// Confronta il risultato decrittato con i valori di input attesi, per
// verificare la correttezza numerica (non solo l'assenza di crash).
//
// NOTA: StC-first consuma un blocco SlotsToCoeffs PRIMA del ModRaise
// (invece che come ultimo passo, come nel bootstrap standard), quindi
// richiede piu' livelli disponibili all'ingresso. Se vedi un errore tipo
// "Not enough levels" o un assert su ModRaise, aumenta `depth` e/o
// `levelBudget`.

#include <cstdint>
#include <iomanip>
#include <iostream>

#include <fideslib.hpp>
#include <vector>

using namespace fideslib;

std::vector<int> devices = { 0 };

void BootstrapStCFirstExample() {
	CCParams<CryptoContextCKKSRNS> parameters;

	SecretKeyDist secretKeyDist = UNIFORM_TERNARY;

	parameters.SetSecretKeyDist(secretKeyDist);
	parameters.SetSecurityLevel(HEStd_NotSet);
	parameters.SetRingDim(1 << 12);

#if NATIVEINT == 128
	ScalingTechnique rescaleTech = FIXEDAUTO;
	uint32_t dcrtBits			 = 78;
	uint32_t firstMod			 = 89;
#else
	ScalingTechnique rescaleTech = FLEXIBLEAUTO;
	uint32_t dcrtBits			 = 59;
	uint32_t firstMod			 = 60;
#endif

	parameters.SetScalingModSize(dcrtBits);
	parameters.SetScalingTechnique(rescaleTech);
	parameters.SetFirstModSize(firstMod);
	parameters.SetKeySwitchTechnique(HYBRID);
	parameters.SetDevices(std::vector(devices));

	std::vector<uint32_t> levelBudget = { 3, 3 };

	// StC-first spends an extra SlotsToCoeffs pass before ModRaise, so we
	// give a few more levels of headroom than SimpleBootstrapExample's 25.
	uint32_t depth = 30;
	parameters.SetMultiplicativeDepth(depth);

	CryptoContext<DCRTPoly> cryptoContext = GenCryptoContext(parameters);

	cryptoContext->Enable(PKE);
	cryptoContext->Enable(KEYSWITCH);
	cryptoContext->Enable(LEVELEDSHE);
	cryptoContext->Enable(ADVANCEDSHE);
	cryptoContext->Enable(FHE);

	uint32_t ringDim  = cryptoContext->GetRingDimension();
	uint32_t numSlots = ringDim >> 1;
	std::cout << "CKKS scheme is using ring dimension " << ringDim << std::endl;

	auto keyPair = cryptoContext->KeyGen();
	cryptoContext->EvalMultKeyGen(keyPair.secretKey);

	cryptoContext->LoadContext(keyPair.publicKey);

	// The key difference vs. the standard bootstrap example: btsfirstboot=true.
	cryptoContext->EvalBootstrapSetup(levelBudget, { 0, 0 }, numSlots, 0, /*precompute=*/true, /*btsfirstboot=*/true);
	cryptoContext->EvalBootstrapKeyGen(keyPair, numSlots);

	std::vector<double> x = { 0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0 };
	size_t encodedLength  = x.size();

	Plaintext ptxt = cryptoContext->MakeCKKSPackedPlaintext(x, 1, depth - 1);
	ptxt->SetLength(encodedLength);

	Ciphertext<DCRTPoly> ciph = cryptoContext->Encrypt(keyPair.publicKey, ptxt);

	std::cout << "Initial number of levels remaining: " << depth - ciph->GetLevel() << std::endl;

	cryptoContext->EvalBootstrapStCFirstInPlace(ciph);

	std::cout << "Number of levels remaining after StC-first bootstrapping: " << depth - ciph->GetLevel() - (ciph->GetNoiseScaleDeg() - 1)
			  << std::endl;

	Plaintext result;
	cryptoContext->Decrypt(keyPair.secretKey, ciph, &result);
	result->SetLength(encodedLength);
	auto decrypted = result->GetRealPackedValue();

	std::cout << std::endl << "==== StC-first bootstrap: input vs output ====" << std::endl;
	std::cout << std::setw(6) << "slot" << std::setw(12) << "input" << std::setw(12) << "output" << std::setw(12) << "abs err" << std::endl;

	double maxErr = 0.0;
	for (size_t i = 0; i < encodedLength; ++i) {
		double err = std::abs(decrypted[i] - x[i]);
		maxErr	   = std::max(maxErr, err);
		std::cout << std::fixed << std::setprecision(6) << std::setw(6) << i << std::setw(12) << x[i] << std::setw(12) << decrypted[i]
				  << std::setw(12) << err << std::endl;
	}

	std::cout << std::endl << "Max abs error: " << maxErr << std::endl;

	// Bootstrap precision is inherently approximate; this tolerance is
	// deliberately loose (just checking the pipeline is not badly broken,
	// not measuring precision bits).
	constexpr double tolerance = 1e-1;
	bool ok						= maxErr < tolerance;
	std::cout << (ok ? "PASS" : "FAIL") << ": StC-first bootstrap "
			  << (ok ? "output matches the original input within tolerance." : "output does NOT match — check BootstrapStCFirst.") << std::endl;
}

int main(int argc, char* argv[]) {
	BootstrapStCFirstExample();
	return 0;
}
