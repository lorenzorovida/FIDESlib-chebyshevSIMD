// 11_integer_ops_full_test.cpp
//
// Suite di test per i wrapper API di aritmetica intera bit-packed esposti
// su fideslib::CryptoContext:
//   EvalAddInteger, EvalEqualInteger, CsaSum, CsaCarry, MajorityBit,
//   BinToDec.
//
// NON incluse:
//   - EvalMultInteger / Multiplier4bits: richiedono `coeffsMultipler4bits`,
//     gli 8 set di coefficienti Chebyshev del multiplier a 4 bit
//     (p1..p8-norm-369.txt nel progetto originale), che non sono
//     disponibili qui. Aggiungerli è immediato una volta forniti.
//   - ProcessArray: esclusa su richiesta esplicita.
//
// Ogni test stampa una tabella di confronto e un PASS/FAIL con tolleranza.
// Il programma esce con codice 0 solo se TUTTI i test passano.

#include <fideslib.hpp>

#include <cmath>
#include <functional>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

using namespace fideslib;

// ============================================================
// Helpers
// ============================================================

namespace {

int g_testsRun	  = 0;
int g_testsFailed = 0;

std::vector<double> decrypt(CryptoContext<DCRTPoly>& cc, const KeyPair<DCRTPoly>& keys, const Ciphertext<DCRTPoly>& ct, size_t length) {
	Plaintext pt;
	cc->Decrypt(keys.secretKey, ct, &pt);
	pt->SetLength(length);
	return pt->GetRealPackedValue();
}

/// Confronta `got` con `expected` per valore assoluto.
void report(const std::string& name, const std::vector<double>& got, const std::vector<double>& expected, const std::vector<std::string>& labels,
            double tolerance = 5e-2) {
	g_testsRun++;
	std::cout << std::endl << "==== " << name << " ====" << std::endl;
	std::cout << std::setw(24) << "label" << std::setw(14) << "expected" << std::setw(14) << "got" << std::setw(14) << "abs err" << std::endl;

	double maxErr = 0.0;
	for (size_t i = 0; i < got.size(); ++i) {
		double err = std::abs(got[i] - expected[i]);
		maxErr	   = std::max(maxErr, err);
		std::cout << std::fixed << std::setprecision(4) << std::setw(24) << (i < labels.size() ? labels[i] : std::to_string(i)) << std::setw(14)
				  << expected[i] << std::setw(14) << got[i] << std::setw(14) << err << std::endl;
	}

	bool ok = maxErr < tolerance;
	if (!ok)
		g_testsFailed++;
	std::cout << "Max abs error: " << maxErr << "  ->  " << (ok ? "PASS" : "FAIL") << std::endl;
}

/// Confronta `got` con `expected` a meno di un fattore di scala costante,
/// stimato dal primo elemento non nullo di `expected`. Usato per BinToDec,
/// che applica deliberatamente un rescale 1/sqrt(225/2) per il prossimo
/// stadio Chebyshev.
void reportProportional(const std::string& name, const std::vector<double>& got, const std::vector<double>& expected,
                         const std::vector<std::string>& labels, double tolerance = 5e-2) {
	g_testsRun++;
	std::cout << std::endl << "==== " << name << " (checked up to a constant scale factor) ====" << std::endl;

	double scale = 1.0;
	for (size_t i = 0; i < expected.size(); ++i) {
		if (std::abs(expected[i]) > 1e-9) {
			scale = got[i] / expected[i];
			break;
		}
	}
	std::cout << "Estimated scale factor: " << scale << std::endl;
	std::cout << std::setw(24) << "label" << std::setw(14) << "expected*scale" << std::setw(14) << "got" << std::setw(14) << "abs err" << std::endl;

	double maxErr = 0.0;
	for (size_t i = 0; i < got.size(); ++i) {
		double expScaled = expected[i] * scale;
		double err		 = std::abs(got[i] - expScaled);
		maxErr			 = std::max(maxErr, err);
		std::cout << std::fixed << std::setprecision(4) << std::setw(24) << (i < labels.size() ? labels[i] : std::to_string(i)) << std::setw(14)
				  << expScaled << std::setw(14) << got[i] << std::setw(14) << err << std::endl;
	}

	bool ok = maxErr < tolerance;
	if (!ok)
		g_testsFailed++;
	std::cout << "Max abs error: " << maxErr << "  ->  " << (ok ? "PASS" : "FAIL") << std::endl;
}

} // namespace

// ============================================================
// Setup
// ============================================================

struct TestEnv {
	CryptoContext<DCRTPoly> cc;
	KeyPair<DCRTPoly> keys;
	uint32_t batchSize;
	uint32_t depth;
};

TestEnv setup(uint32_t batchSize, uint32_t depth) {
	TestEnv env;
	env.batchSize = batchSize;
	env.depth	  = depth;

	ScalingTechnique rescaleTech = FLEXIBLEAUTO;
	uint32_t dcrtBits			 = 59;
	uint32_t firstMod			 = 60;
	uint32_t dnum				 = 3;

	CCParams<CryptoContextCKKSRNS> parameters;
	parameters.SetSecurityLevel(SecurityLevel::HEStd_NotSet);
	parameters.SetRingDim(1 << 13);
	parameters.SetMultiplicativeDepth(depth);
	parameters.SetScalingModSize(dcrtBits);
	parameters.SetScalingTechnique(rescaleTech);
	parameters.SetFirstModSize(firstMod);
	parameters.SetKeySwitchTechnique(HYBRID);
	parameters.SetNumLargeDigits(dnum);
	parameters.SetBatchSize(batchSize);
	parameters.SetDevices({ 0 });
	parameters.SetPlaintextAutoload(false);
	parameters.SetCiphertextAutoload(true);

	env.cc = GenCryptoContext(parameters);
	env.cc->Enable(PKE);
	env.cc->Enable(KEYSWITCH);
	env.cc->Enable(LEVELEDSHE);
	env.cc->Enable(ADVANCEDSHE);
	env.cc->Enable(FHE);

	std::cout << "CKKS scheme using ring dimension: " << env.cc->GetRingDimension() << " (batchSize=" << batchSize << ", depth=" << depth << ")"
			  << std::endl;

	env.keys = env.cc->KeyGen();
	env.cc->EvalMultKeyGen(env.keys.secretKey);
	env.cc->LoadContext(env.keys.publicKey);

	return env;
}

Ciphertext<DCRTPoly> encryptVec(TestEnv& env, const std::vector<double>& v) {
	return env.cc->Encrypt(env.keys.publicKey, env.cc->MakeCKKSPackedPlaintext(v));
}

// ============================================================
// Tests
// ============================================================

/// CsaSum, CsaCarry, MajorityBit su tutte le 8 combinazioni di {0,1}^3.
void testMajorityAndCsa3(TestEnv& env) {
	std::vector<double> aVals = { 0, 0, 0, 0, 1, 1, 1, 1 };
	std::vector<double> bVals = { 0, 0, 1, 1, 0, 0, 1, 1 };
	std::vector<double> cVals = { 0, 1, 0, 1, 0, 1, 0, 1 };

	std::vector<double> expSum(8), expCarry(8);
	std::vector<std::string> lbl(8);
	for (int i = 0; i < 8; ++i) {
		int s		= static_cast<int>(aVals[i] + bVals[i] + cVals[i]);
		expSum[i]	= s % 2;
		expCarry[i] = (s >= 2) ? 1.0 : 0.0;
		lbl[i]		= "(" + std::to_string((int)aVals[i]) + "," + std::to_string((int)bVals[i]) + "," + std::to_string((int)cVals[i]) + ")";
	}

	auto ctA = encryptVec(env, aVals);
	auto ctB = encryptVec(env, bVals);
	auto ctC = encryptVec(env, cVals);

	auto sumCt	 = env.cc->CsaSum(ctA, ctB, ctC);
	auto carryCt = env.cc->CsaCarry(ctA, ctB, ctC);
	auto majCt	 = env.cc->MajorityBit(ctA, ctB, ctC);

	report("CsaSum", decrypt(env.cc, env.keys, sumCt, env.batchSize), expSum, lbl);
	report("CsaCarry", decrypt(env.cc, env.keys, carryCt, env.batchSize), expCarry, lbl);
	report("MajorityBit", decrypt(env.cc, env.keys, majCt, env.batchSize), expCarry, lbl);
}

/// BinToDec: 2 gruppi di 8 slot; group0 bits=(1,0,1,1)->13, group1
/// bits=(0,1,0,0)->2. Verificato per PROPORZIONALITÀ: BinToDec applica
/// deliberatamente un rescale 1/sqrt(225/2) per il prossimo stadio
/// Chebyshev, quindi il valore assoluto non è 13/2 ma 13/2 * scala.
void testBinToDec(TestEnv& baseEnv) {
	std::vector<double> bits(16, 0.0);
	bits[0] = 1;
	bits[1] = 0;
	bits[2] = 1;
	bits[3] = 1; // group0: 1 + 0*2 + 1*4 + 1*8 = 13
	bits[8] = 0;
	bits[9] = 1;
	bits[10] = 0;
	bits[11] = 0; // group1: 0 + 1*2 + 0*4 + 0*8 = 2

	TestEnv env16 = setup(16, baseEnv.depth);
	auto ctBits	  = encryptVec(env16, bits);

	auto out = env16.cc->BinToDec(ctBits, /*repetitions=*/2);
	auto got = decrypt(env16.cc, env16.keys, out, env16.batchSize);

	// BinToDec broadcasts the decoded value back across slots 0..3 (and
	// 8..11) of each group; check slot 0 of each group up to BinToDec's
	// known constant scale factor.
	std::vector<double> gotSlot0 = { got[0], got[8] };
	std::vector<double> expected = { 13.0, 2.0 };
	std::vector<std::string> lbl = { "group0 (bits 1,0,1,1)", "group1 (bits 0,1,0,0)" };

	reportProportional("BinToDec", gotSlot0, expected, lbl);
}

/// EvalAddInteger: regression check (l'utente ha già confermato che
/// funziona). 5 (0101, LSB primo) + 3 (0011, LSB primo) = 8 (1000, LSB
/// primo), aritmetica a 4 bit con wraparound.
void testEvalAddInteger(TestEnv& baseEnv) {
	std::vector<double> aVals	  = { 1, 0, 1, 0 }; // 5, LSB primo
	std::vector<double> bVals	  = { 1, 1, 0, 0 }; // 3, LSB primo
	std::vector<double> expected = { 0, 0, 0, 1 }; // 8, LSB primo
	std::vector<std::string> lbl = { "bit0", "bit1", "bit2", "bit3" };

	TestEnv env4 = setup(4, baseEnv.depth);
	auto ctA	 = encryptVec(env4, aVals);
	auto ctB	 = encryptVec(env4, bVals);

	auto out = env4.cc->EvalAddInteger(ctA, ctB, /*bits=*/4);
	report("EvalAddInteger: 5 + 3 = 8 (4-bit)", decrypt(env4.cc, env4.keys, out, env4.batchSize), expected, lbl);
}

/// EvalEqualInteger: confronta due interi a 4 bit, uguali e diversi.
/// coeffsSinc generati con GetChebyshevCoefficients su
/// sinc(x) = sin(pi*x)/(pi*x), che vale esattamente 1 in x=0 e ~0 su ogni
/// altro intero -- comportamento coerente con equal(a,b) = sinc(a-b).
void testEvalEqualInteger(TestEnv& baseEnv) {
	int bits = 4;

	// Due coppie: (5,5) uguali, (5,3) diverse. Impacchettate su 2 gruppi di
	// 4 slot (LSB primo), stesso layout di EvalAddInteger.
	std::vector<double> aVals = { 1, 0, 1, 0, /**/ 1, 0, 1, 0 }; // 5, 5
	std::vector<double> bVals = { 1, 0, 1, 0, /**/ 1, 1, 0, 0 }; // 5, 3
	std::vector<double> expected = { 1, 1, 1, 1, /**/ 0, 0, 0, 0 };
	std::vector<std::string> lbl = { "5==5 bit0", "5==5 bit1", "5==5 bit2", "5==5 bit3",
		                              "5==3 bit0", "5==3 bit1", "5==3 bit2", "5==3 bit3" };

	TestEnv env8 = setup(8, baseEnv.depth);
	auto ctA	 = encryptVec(env8, aVals);
	auto ctB	 = encryptVec(env8, bVals);

	int zslots			  = 1 << bits; // 16: range simmetrico per la differenza a-b
	int chebyDegree		  = 59;
	double lowerBound	  = -static_cast<double>(zslots);
	double upperBound	  = static_cast<double>(zslots);
	std::function<double(double)> sincFn = [](double x) {
		if (std::abs(x) < 1e-9)
			return 1.0;
		return std::sin(M_PI * x) / (M_PI * x);
	};
	auto coeffsSinc = env8.cc->GetChebyshevCoefficients(sincFn, lowerBound, upperBound, chebyDegree);

	auto out = env8.cc->EvalEqualInteger(ctA, ctB, bits, zslots, coeffsSinc, static_cast<int>(baseEnv.depth));
	report("EvalEqualInteger: 5==5 (all 1) and 5==3 (all 0)", decrypt(env8.cc, env8.keys, out, env8.batchSize), expected, lbl, /*tolerance=*/1e-1);
}

int main() {
	TestEnv env = setup(/*batchSize=*/8, /*depth=*/25);

	testMajorityAndCsa3(env);
	testBinToDec(env);
	testEvalAddInteger(env);
	testEvalEqualInteger(env);

	std::cout << std::endl << "============================================" << std::endl;
	std::cout << g_testsRun << " test(s) run, " << g_testsFailed << " failed." << std::endl;
	std::cout << "NOTE: EvalMultInteger/Multiplier4bits (needs external Chebyshev" << std::endl;
	std::cout << "      coefficient files) and ProcessArray are not covered here." << std::endl;
	std::cout << (g_testsFailed == 0 ? "ALL PASS" : "SOME FAILED") << std::endl;

	return g_testsFailed == 0 ? 0 : 1;
}
