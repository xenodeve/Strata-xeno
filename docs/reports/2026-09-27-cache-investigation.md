# Cache correctness investigation

Scope: Strata-xeno #1 / Flash-Next #10. Baseline `f679806`, isolated worktree `.worktrees/afk`, branch `xeno/afk-phase1`.

## Evidence ledger

- Source preflight: the legacy warning in `generate.cpp` is present; it is a claim to reproduce, not a demonstrated current root cause.
- Actual entry paths: `session_capture_token` records all layers; serving with MTP uses `Verifier::run`. The per-layer loop is an alternative path.
- CMake already accepts CUDA architectures >=80; the handoff's rejection-of-sm_89 claim is stale. A real dual-architecture build remains required.
- GPU preflight: 5060 Ti idle, 0 MiB used; 4070 SUPER displays use ~1863 MiB. No unrelated processes were stopped.
- Paired issues for all nine phases exist and are native sub-issues of Flash-Next #1.

## Reproduction

`tests/xeno/cache_tokens.py` fixes prompt IDs, greedy sampling, context, threads and placement, and disables adaptive swaps and PCIe miss sharing. The native Q2_0 pack requires spec 4 and a live cache, so both arms use the same profile-filled slot allocation and prefill path. `--cache-cpu-only` stages an all-miss residency map after prefill for verification; the other arm uses the GPU cache map. The harness records raw stdout/stderr, commands, binary hashes and token IDs. Startup time is excluded from the separately reported decode rate.

| Run | Change | Observation | Implication |
|---|---|---|---|
| `compare1` | 2,048 slots, CPU-only versus GPU hits, 96 output IDs | Equal. CPU-only decode 24.92 tok/s, GPU-hit decode 26.66 tok/s, rough. | A sparse hit arm does not reproduce the stated divergence on this fixed prompt. |
| `compare5000` | 5,000 slots, same binary and fixed prompt | First difference at output index 32. CPU-only decode 24.96 tok/s, GPU-hit 36.84 tok/s, rough. | Cache placement affects output. The CPU-only results from both runs are equal, which narrows the cause to the hit path or its interaction with verification. |
| `native_q2_pool_hit_parity` | Real layer-0 Q2_0 expert 7, one deterministic activation, CPU pool versus native grouped GPU expert | 2,560/2,560 outputs differ, relative L1 0.00188108147; input int8 codes differ 0/2,560. | Expert arithmetic differs before any router or verifier combine. |
| Gate differential | Compare the CPU Q2_0 gate rows against GPU `iq_mmvq`, then round CPU input scales to GPU's fp16 scale | Relative L1 falls from 0.00023176282 to 0.000000313244765. | Input scale precision explains nearly all first-projection error; the remaining difference and second-stage precision still need testing. |
| `scaled5000` | Preserve fp32 scales in the native Q2_0 GPU hit path at both activation boundaries, retain the same CPU-only diagnostic arm | Expert relative L1 falls to 0.000000366099886; 2,327/2,560 floats still differ. All 96 greedy output IDs match the CPU-only arm on the prompt that diverged at index 32 before the change. | The identified scale mismatch caused this prompt's token divergence. Per-expert bit exactness remains unproven and currently fails. |
| `scaled-thai`, `scaled-code`, `scaled-long` | Same code and 5,000 slots; three further fixed prompts, 96 generated IDs | Thai and code match 96/96; long technical prompt diverges at index 29. | FP32 scales are necessary but insufficient for user-visible correctness. Do not mark Phase 1 complete. |
| `exact-*` | Match the CPU row's eight FP32 accumulators and correction order, and SwiGLU's double-precision `exp` result | Four real Q2_0 experts from layers 0/1/20/47 match 2,560 outputs bit for bit; a three-entry, two-group kernel check also matches. Sky still diverged at token 77 and long at 29; Thai and code matched 96/96. | Normal-distributed inputs do not cover quantization boundaries. Token parity remains red. |
| Boundary input `seed=-2` | Each 32-value chunk contains values near half-integer Q8 rounding boundaries | With the old GPU `roundf(x/d)`, 352 of 7,680 codes differ from the CPU pool; correcting GPU Q2_0 to multiply by the FP32 inverse and round half-away makes codes, scales, gate/up, hidden and all 7,680 output floats bit-exact. | This is an independent cause of expert mismatch on valid inputs. Recheck model token streams after the change. |
| Real verifier input at position 36, layer 26, token 2 | Capture the published activation and router IDs; test all ten routed experts individually | Expert 361 had one hidden value different out of 640, with matching Q8 codes, scales, gate and up. On that value, CPU `std::exp(float)` differed from GPU `float(exp(double))`; changing only native-Q2_0 CPU SwiGLU to the latter made the expert's 7,680 tested outputs bit-exact. | The gate at layer 26 exposed a separate library-rounding difference. The standalone checker also passed 64 seeded multi-token/two-expert cases. |
| `canonical-long` | New CPU Q2_0 SwiGLU and GPU Q2_0 path, 5,000 slots, same fixed prompt, 96 IDs | CPU-only and GPU-hit token IDs matched 96/96. | The first known long-prompt divergence at index 29 is removed in this controlled run; longer multi-prompt gate remains pending. |
| `final256-{sky,thai,code,long}` | Final build without trace instrumentation, 5,000 profile-ranked slots, fixed six P-core workers, spec 4, PCIe miss share 0, 256 greedy IDs each | All four CPU-only and GPU-hit pairs matched 256/256 IDs. Rough decode on/off: sky 25.30/34.56, Thai 22.15/28.10, code 31.89/37.34, long 28.16/35.50 tok/s. | Known fixed-prompt output divergence is removed, with GPU hits still providing a rough speed benefit. Numbers are not precise ABBA measurements and include background CPU noise. |

The first Q8 trace attempt read device quantization before that stage had finished and reported stale rows. Its 31,939 mismatches are invalid evidence. The trace code was removed; graph-ordered layer-output tracing identified the first real difference instead and was also removed before the final build.
