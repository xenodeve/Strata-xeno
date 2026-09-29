# Decode dispatch run: CPU pool versus secondary finish

The user-provided `strata-second-gpu-decode-source-audit-2026-09-28.md` found that the old `dispatch run` counter enclosed both `run_split_multi_native()` and `SecondaryRunner::finish()`. This change adds a host timestamp around each call and reports separate milliseconds per verifier round. It does not change expert arithmetic, routing, GPU data movement or the default profile.

## Fixed-placement measurements

Both runs used `xeno/codex-secondary-timing` with packed metadata and compact D2H, Q2_0, EXL3-only profile, 6,653 primary slots, 8.5 GiB secondary tier, six workers, MTP `--spec 4`, `--prefill 2048`, `--max-context 8192`, and `CUDA_VISIBLE_DEVICES=1,0`. Profiling CUDA events was **off**. Each produced the accepted baseline's 256/256 raw token IDs. Artifacts: `%TEMP%\strata-codex-dispatch-detail-{code,thai}256` (commands, binary hash, raw IDs and logs).

| Counter, ms/round | Code | Thai |
|---|---:|---:|
| `dispatch run` inclusive | 31.430 | 22.630 |
| CPU pool self (`run_split_multi_native`) | 25.394 | 19.177 |
| Secondary `finish()` total | 6.030 | 3.448 |
| Sum of the two leaf calls | 31.424 | 22.625 |
| CPU gate/up + intermediate quantize + down | 25.374 | 19.168 |
| Secondary event wait, nested in `finish()` | 5.400 | 3.064 |
| Decode rate, rough one run | 42.53 tok/s | 29.06 tok/s |

The two measured leaf calls account for essentially all of `dispatch run` in both prompts. The earlier roughly 9 ms difference between “CPU pool call” and CPU phase counters **was not evidence of unmeasured worker overhead**: the old callback/inclusive timer included secondary completion. CPU phase timers themselves include pool coordination, so their close sum with CPU pool self-time does **not** prove worker synchronization is free. To isolate coordination from row arithmetic, timers inside `run_phase` and a workload replay would still be needed. The upstream late-wakeup batch-claim race fix identified in the source audit remains a separate correctness review; no acceleration is assumed from it.

The secondary finish tail is real in these runs, about 3–6 ms/round, but much of the primary verifier's wall time is host waiting for the primary GPU doorbell and CPU expert work. The secondary transport changes already cut H2D API calls and D2H requested bytes; neither has yet shown a robust full-window speedup. Next use the existing `--gpu-stages` probe to split primary mixer, FFN/router and post work before choosing another GPU-side optimization.
