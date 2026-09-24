# Uniswap v3 on GPU: fixes and validation checklist

Branch `jpp_gpu_uniswap_fix`, based on `uniswapv3` (2c046ce). It is paired with the host-program branch
`jpp_gpu_uniswap_fix` of [advanced-arithmetic-fhe-cuda](https://github.com/lorenzorovida/advanced-arithmetic-fhe-cuda).
The CPU reference is [advanced-arithmetic-fhe](https://github.com/lorenzorovida/advanced-arithmetic-fhe),
`experiment_uniswap_v3` in `src/main.cpp`.

**Status: not compiled or run.** The fixes were written and checked by reading only, against the CPU code, on a machine
without nvcc. Treat each one as a hypothesis until the checklist below passes on a GPU.

## What was wrong, and what changed

All changes are in `src/CKKS/IntegerOperations.cu` (line numbers refer to `uniswapv3`), apart from the host program.

| # | Where | Bug | Fix | Confidence |
|---|---|---|---|---|
| 1 | `evalIntegerDivision` :1544 | Leftover debug `out.copy(x); return;`: returns the Newton reciprocal hint, not `num/den`. X_post, diff and amount are then all wrong. | Removed. | high |
| 2 | `evalIntegerDivision` :1393-1394, :1522, :1561 | 2c046ce commented out the `dropToLevel` calls before each mult. binboot outputs sit at level 11, and `evalIntegerMult`'s processArray masks are encoded at 12. Under FLEXIBLEAUTO, `multPt` with a plaintext deeper than the ciphertext hits `assert(false)` and returns unmultiplied, which in Release gives silent garbage. | `prepareIntegerOperand` on x, denNorm and term2, and on num2 and x2 before the final mult. | medium-high |
| 3 | `evalIntegerDivision` :1389 | The Newton loop ran `newtonIters - 1` = 3 times at 128 bits. The CPU runs `ceil(log2(bits/8))` = 4, and 3 iterations from an 8-bit seed give about a 64-bit-accurate reciprocal. | Loop runs `newtonIters` times. | medium: this was toggled twice on 23 Sep, so revisit it first if 4 misbehaves |
| 4 | `evalIntegerDivisionByPlain` :2942 | Shifted the raw `num`, which can be NoiseLevel 2 (e.g. `term2` right after masking), instead of the prepared `numOp`. | Shift `numOp`. | medium that it mattered; low risk |
| 5 | `evalUniswapV3` | Every mask is sized from `ct.slots`. The CPU driver never pads `m_fx` to N/2 slots, so a port that copies it misaligns silently. | Throws on mismatched slot counts. | high that this is a trap |
| 6 | host `src/main.cpp` | No driver called `EvalUniswapV3Example`. It also needs the (bits=128, zslots=1) division precomputation, `UniswapV3Precomputations`, and the rotation keys generated **before** `LoadContext`. | New `--uniswapv3` flag that does all three, pads every input, forces 128 bits, requires ring >= 14, and prints each value beside its expected value. | high |

Left alone, since they don't change the output on these inputs (checked in Python):
- The plain-divisor reciprocal rounds up on the GPU and down on the CPU.
- `evalIntegerSub` uses bits+1 ones where the CPU uses bits; the result is masked to the low bits anyway.
- The GPU re-masks after `>>21`, which is stricter than the CPU.

The CPU repo had the same bug as #1 (`return x;` in ct/ct `div_integer`, added in fa364f7). It is fixed on
`advanced-arithmetic-fhe` branch `jpp_improvements`. Use that branch, or 25fe37e, as the CPU reference.

## Validate on a CUDA machine

```sh
# 1. FIDESlib, Debug first: asserts stay on, so any remaining skipped multPt (bug #2) aborts loudly
git clone -b jpp_gpu_uniswap_fix https://github.com/lorenzorovida/FIDESlib-chebyshevSIMD && cd FIDESlib-chebyshevSIMD
cmake -B build -DCMAKE_BUILD_TYPE=Debug -DFIDESLIB_INSTALL_OPENFHE=ON -DFIDESLIB_COMPILE_TESTS=OFF -DFIDESLIB_COMPILE_BENCHMARKS=OFF
cmake --build build --target install -j

# 2. Host program. Run it from build/, since coefficients are loaded from ../coeffs
git clone -b jpp_gpu_uniswap_fix https://github.com/lorenzorovida/advanced-arithmetic-fhe-cuda && cd advanced-arithmetic-fhe-cuda
mkdir -p build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release && make -j
./AdvancedFHEGPU --ring 16 --uniswapv3

# 3. Optional CPU reference (advanced-arithmetic-fhe, branch jpp_improvements)
./scripts/install.sh && cd build && ./AdvancedFHE --ring 16 --bits 128 --uniswapv3
```

Expected values (plain uint128 arithmetic of the CPU pipeline; see the Python below):

| Value | Width | Expected |
|---|---|---|
| term2_fx | 64 | 162434547609510 |
| u_fx | 64 | 515720430530054940 |
| X_post_fx | 128 | 280151436534142656453804032 |
| diff_fx | 128 | 25887770211597044045332 |
| Amount_fx | 128 | 25965667213236754308256 |

## Checklist: confirm, or report back which line fails

- [ ] Both repos compile (nvcc has never seen these edits).
- [ ] The Debug run finishes with no assert. An assert inside `adjustPlaintextToCiphertext` or `multPt` means a level mismatch is left: bug #2 is incomplete.
- [ ] `term2_fx` and `u_fx` match. That covers the multiply, the two plaintext divisions and the add. If they're off, look at #4 or the `>>21` step.
- [ ] `X_post_fx` matches. That covers the ct/ct division, bugs #1 to #3. If it's lower by exactly 2^32, the quotient is one short: the Newton iteration count (#3) or the final correction step.
- [ ] `diff_fx` and `Amount_fx` match.
- [ ] A Release rebuild gives the same values. Then record the time printed on the "Uniswap v3 on GPU took" line, with the GPU model.
- [ ] Optional: the CPU `--ring 16 --bits 128 --uniswapv3` run on `jpp_improvements` prints the same numbers.

### Expected-value script

```python
M = (1 << 128) - 1
def pdiv(num, den, bits=128):              # CPU plaintext division: floor reciprocal
    k = bits + den.bit_length()
    return ((num * ((1 << k) // den)) >> k) & M
g, user, inv = 43315879362536242, 3750000000000000000, 515557995982445430
numer = (1823591698684026 << 64) | 9043734471353827328
m = (15188443 << 64) | 3405228930222675476
t = pdiv(pdiv(((user * g) & M) >> 21, 5**10), 5**11)   # term2_fx
u = (t + inv) & ((1 << 64) - 1)                         # u_fx
X = ((numer // u) << 32) & M                            # X_post_fx
d = (m - X) & M                                         # diff_fx
print(t, u, X, d, pdiv((d * 1000) & M, 997))            # ..., Amount_fx
```
