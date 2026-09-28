# Compact secondary D2H rows — isolated ablation

Base: packed metadata checkpoint `d1431ea`. This change leaves the native Q2 arithmetic, routing decision and output-clear size intact. For each secondary-owned router entry, the device destination is now a dense index `[0, S)`. CUDA D2H requests only `S × H × sizeof(float)` bytes; the host scatters those dense rows back to their original router positions. The previous code copied all `T × K × H` rows even when most belonged to primary or CPU. This is a separate change from the pinned metadata upload.

## Correctness and transfer size

The direct runner test covers full, partial and shrinking two-group-to-one-group calls and checks CPU/GPU float parity. The timed test independently checks the requested-byte counter on those inputs. Sky, Thai, code and long each matched the accepted Q2_0 greedy baseline **256/256 raw IDs** with six workers, MTP `--spec 4`, 2048-token prefill capacity, 6,653 primary expert slots and the 8.5 GiB secondary EXL3-only profile. Full CUDA build, CTest `xeno_|^gpu_` **18/18**, Python `tests/xeno` **28/28**, and `git diff --check` passed. No user launcher/profile, PR or main merge changed.

The code prompt's profiled transfer counter reported **1,336.91 MiB** for the old full-row copy versus **345.87 MiB actually requested** by the compact copy, a 74.1% reduction. The CUDA event D2H interval in separate process runs was **2.066 → 1.133 ms/round**. These are API-requested bytes and stream event intervals, not a measured PCIe bandwidth counter. The `SecondaryRunner::finish()` wait was **5.798 → 5.312 ms/round**, while its host copy-out rose **0.332 → 0.698 ms/round**. Host enqueue and CPU phase timings also varied, so the measured decode rate **47.78 → 41.24 tok/s** from those two profiled runs cannot be attributed to D2H compaction. No end-to-end speedup is claimed.

Unprofiled compact runs completed with parity: sky 40.80, Thai 35.05 and long 37.66 tok/s. Code parity was verified in the profiled compact run; a matched unprofiled code A/B remains open if this change is considered for production. The current branch is experimental and remains opt-in.

Artifacts: `%TEMP%\strata-codex-compact-profile-code256` and `%TEMP%\strata-codex-compact-parity-{sky,thai,long}256` contain command JSON, executable SHA-256, raw IDs and logs. The prior pinned-metadata baseline is documented in `docs/reports/2026-09-28-packed-secondary-metadata.md`.

Next: split `dispatch.ms_run` into CPU pool self-time and complete secondary `finish()` time before deciding whether the remaining wait belongs to worker synchronization, 4070 kernels, or the host rendezvous. The separate primary GPU stage probe `--gpu-stages` remains open.
