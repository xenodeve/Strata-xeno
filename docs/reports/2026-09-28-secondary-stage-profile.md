# Secondary GPU decode stage profile — 2026-09-28

Scope: opt-in instrumentation on `xeno/codex-secondary-timing`, child of `xeno/claude-prefill-mtp@c54c948`. The user-provided source audit `strata-second-gpu-decode-source-audit-2026-09-28.md` identified incorrect counter labels before this measurement. No default launcher/profile change, PR or merge.

## Corrected timer boundaries

In `Verifier::run`, `ms_wait` is **host waiting for the primary GPU doorbell** before the expert callback. `ms_pool` times that whole callback. Inside it, `dispatch.ms_run` contains the CPU pool call **and** `SecondaryRunner::finish()`, which includes the secondary completion wait and selected-row scatter. `drive.cpu_ms` also encloses the whole dispatch callback despite its historical name. The latency parser and generated labels now describe these inclusive boundaries. Their old labels did not prove either “GPU waits CPU for 45%” or “~9 ms CPU-pool coordination overhead.”

The new `--secondary-profile-timing` flag creates CUDA timing events only in profiling mode. Events bracket six H2D calls, output clear, activation quantization, grouped expert kernels and full D2H. Host counters separate plan, device select/restore, enqueue, event-query overhead and selected-row copy-out. CUDA event intervals are **stream elapsed time**, and can include gaps while host API calls enqueue later work. Host and device counters overlap and must not be added as a wall-time pie. [NVIDIA's CUDA event API](https://docs.nvidia.com/cuda/cuda-runtime-api/cuda_runtime_api/group__CUDART__EVENT.html) documents this caveat; [the CUDA programming guide](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html) notes that transfers from pageable host memory can behave synchronously.

## Reproduction and proof

Both prompts used Q2_0, `CUDA_VISIBLE_DEVICES=1,0`, six workers, MTP `--spec 4 --spec-min-p 0.5`, `--prefill 2048`, `--max-context 8192`, 8.5 GiB secondary tier with a 640 MiB measured free floor, exclusive primary tier, and `%TEMP%\strata-ranked-exl3only-profile.bin`. Primary cache filled 6,653 slots (8.57 GiB). The captured commands and binary hashes are in each run's `on.command.json` / `on.result.json`:

- `%TEMP%\strata-codex-secondary-profile-code256`
- `%TEMP%\strata-codex-secondary-profile-thai256`
- `%TEMP%\strata-codex-secondary-unprofiled-code256`

Profiled code and Thai each matched Claude's same-placement 256-token baseline **256/256 raw IDs**. The unprofiled code arm matched the profiled arm 256/256. The direct runner test preserves CPU/GPU float parity with timing off and on; a separate CTest exercises the timing mode.

| Metric, ms/round unless noted | Code, profiled | Thai, profiled |
|---|---:|---:|
| Verifier rounds | 75 | 173 |
| Host waits primary doorbell | 32.27 | 17.63 |
| Host expert callback | 36.12 | 25.82 |
| Secondary `launch()` inclusive host | 7.76 | 5.90 |
| Secondary host plan | 0.19 | 0.09 |
| Secondary device select/restore | 0.38 | 0.21 |
| Secondary host enqueue | 7.36 | 5.75 |
| Secondary timing-query overhead | 0.07 | 0.06 |
| Secondary host copy-out | 0.34 | 0.18 |
| Secondary event H2D interval | 3.50 | 3.08 |
| Secondary event clear interval | 0.71 | 0.48 |
| Secondary event quantize interval | 0.68 | 0.51 |
| Secondary event grouped expert interval | 15.46 | 9.71 |
| Secondary event D2H interval | 2.17 | 1.70 |
| Secondary `finish()` event wait | 5.26 | 3.31 |
| Decode | 44.55 tok/s | 30.23 tok/s |

The model run's own lower CUDA/NVML reserve monitor held above its floor; 0.5 s `nvidia-smi` samples saw minimum free 1,083 MiB on the 4070 for code and 921 MiB for Thai.

The same binary with profiling **off** measured code `launch` 5.80 ms/round and 45.58 tok/s, versus 7.76 ms/round and 44.55 tok/s with profiling on. Each is one process, so the difference is a measurement-overhead warning rather than a calibrated correction. Profiling did not change raw output IDs. Five control H2D inputs are ordinary host allocations; only activation/output use portable pinned buffers. The 4070 stream's grouped expert interval is substantial but overlaps host CPU work; the nonzero post-pool wait is the observed secondary tail. This profile alone does not prove a packed metadata upload or compact D2H will improve full-window time.

Verification gate after the measurement code: full CUDA build passed; CTest `xeno_|^gpu_` **18/18** including both normal and timed runner paths; Python `tests/xeno` **28/28**; profiled code/Thai and unprofiled code raw-token parity **256/256** each against same-placement artifacts; `git diff --check` passed.

## Next controlled change

The smallest candidate is one packed pinned metadata upload replacing five pageable H2D API calls, following the transport pattern in upstream PR #16 while preserving native Q2 arithmetic. First record H2D API count/bytes, then change only metadata transport. Keep the current profile as the fixed reference, verify direct float parity and four fixed greedy prompts, then compare same-boot alternating decode runs with profiler **off**. Compact D2H rows is a separate second ablation if D2H/tail remains material. The upstream CPU-pool late-wakeup fix is a separate correctness task; no pool-overhead speed claim follows from the old counters.
