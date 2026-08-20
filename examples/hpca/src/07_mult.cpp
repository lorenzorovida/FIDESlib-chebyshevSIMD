// 10_integer_arith.cpp
//
// Verifica di CsaSum/CsaCarry (csa3) e MajorityBit, i building block
// GPU-nativi per l'aritmetica intera bit-packed portati da
// lorenzorovida/flexible-integer-arithmetic-ckks.
//
// Testa tutte le 8 combinazioni di bit {0,1}^3, impacchettate su slot
// diversi dello stesso batch, e confronta con la logica booleana esatta di
// un carry-save-adder: S = (a+b+c) mod 2, C = 1 se (a+b+c) >= 2.

#include <fideslib.hpp>

#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

using namespace fideslib;

int main() {
	uint32_t multDepth			 = 8;
	uint32_t batchSize			 = 8; // una combinazione {0,1}^3 per slot
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

	// Le 8 combinazioni di {0,1}^3, una per slot.
	std::vector<double> aVals = { 0, 0, 0, 0, 1, 1, 1, 1 };
	std::vector<double> bVals = { 0, 0, 1, 1, 0, 0, 1, 1 };
	std::vector<double> cVals = { 0, 1, 0, 1, 0, 1, 0, 1 };

	std::vector<double> expectedSum(batchSize), expectedCarry(batchSize);
	for (uint32_t i = 0; i < batchSize; ++i) {
		int s			  = static_cast<int>(aVals[i] + bVals[i] + cVals[i]);
		expectedSum[i]	  = s % 2;
		expectedCarry[i]  = (s >= 2) ? 1.0 : 0.0;
	}

	auto ctA = cc->Encrypt(keys.publicKey, cc->MakeCKKSPackedPlaintext(aVals));
	auto ctB = cc->Encrypt(keys.publicKey, cc->MakeCKKSPackedPlaintext(bVals));
	auto ctC = cc->Encrypt(keys.publicKey, cc->MakeCKKSPackedPlaintext(cVals));

	std::cout << std::endl << "==== csa3 (cleanVals=true) ====" << std::endl;

	auto sumTrue   = cc->CsaSum(ctA, ctB, ctC);
	auto carryTrue = cc->CsaCarry(ctA, ctB, ctC);

	Plaintext ptxtSumTrue, ptxtCarryTrue;
	cc->Decrypt(keys.secretKey, sumTrue, &ptxtSumTrue);
	cc->Decrypt(keys.secretKey, carryTrue, &ptxtCarryTrue);
	ptxtSumTrue->SetLength(batchSize);
	ptxtCarryTrue->SetLength(batchSize);
	auto sumTrueVals   = ptxtSumTrue->GetRealPackedValue();
	auto carryTrueVals = ptxtCarryTrue->GetRealPackedValue();

	std::cout << std::endl << "==== csa3 (cleanVals=false) ====" << std::endl;

	auto sumFalse   = cc->CsaSum(ctA, ctB, ctC);
	auto carryFalse = cc->CsaCarry(ctA, ctB, ctC);

	Plaintext ptxtSumFalse, ptxtCarryFalse;
	cc->Decrypt(keys.secretKey, sumFalse, &ptxtSumFalse);
	cc->Decrypt(keys.secretKey, carryFalse, &ptxtCarryFalse);
	ptxtSumFalse->SetLength(batchSize);
	ptxtCarryFalse->SetLength(batchSize);
	auto sumFalseVals   = ptxtSumFalse->GetRealPackedValue();
	auto carryFalseVals = ptxtCarryFalse->GetRealPackedValue();

	std::cout << std::endl << "==== majorityBit ====" << std::endl;

	auto maj = cc->MajorityBit(ctA, ctB, ctC);

	Plaintext ptxtMaj;
	cc->Decrypt(keys.secretKey, maj, &ptxtMaj);
	ptxtMaj->SetLength(batchSize);
	auto majVals = ptxtMaj->GetRealPackedValue();

	std::cout << std::endl << "==== Results ====" << std::endl << std::endl;
	std::cout << std::setw(4) << "a" << std::setw(4) << "b" << std::setw(4) << "c" << std::setw(10) << "expS" << std::setw(10) << "expC"
			  << std::setw(12) << "S(clean=T)" << std::setw(12) << "C(clean=T)" << std::setw(12) << "S(clean=F)" << std::setw(12)
			  << "C(clean=F)" << std::setw(10) << "majority" << std::endl;

	double maxErr = 0.0;
	for (uint32_t i = 0; i < batchSize; ++i) {
		double errs[] = { std::abs(sumTrueVals[i] - expectedSum[i]),   std::abs(carryTrueVals[i] - expectedCarry[i]),
			              std::abs(sumFalseVals[i] - expectedSum[i]),  std::abs(carryFalseVals[i] - expectedCarry[i]),
			              std::abs(majVals[i] - expectedCarry[i]) };
		for (double e : errs)
			maxErr = std::max(maxErr, e);

		std::cout << std::fixed << std::setprecision(4) << std::setw(4) << aVals[i] << std::setw(4) << bVals[i] << std::setw(4) << cVals[i]
				  << std::setw(10) << expectedSum[i] << std::setw(10) << expectedCarry[i] << std::setw(12) << sumTrueVals[i] << std::setw(12)
				  << carryTrueVals[i] << std::setw(12) << sumFalseVals[i] << std::setw(12) << carryFalseVals[i] << std::setw(10) << majVals[i]
				  << std::endl;
	}

	std::cout << std::endl << "Max abs error across all columns: " << maxErr << std::endl;

	constexpr double tolerance = 5e-2;
	bool ok						= maxErr < tolerance;
	std::cout << (ok ? "PASS" : "FAIL") << ": csa3 / majorityBit "
			  << (ok ? "match exact boolean carry-save-adder logic within tolerance."
			         : "DO NOT match — check IntegerArith.cu.")
			  << std::endl;

	return ok ? 0 : 1;
}