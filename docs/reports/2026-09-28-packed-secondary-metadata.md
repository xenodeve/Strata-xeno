# Packed pinned secondary metadata — isolated ablation

Base: `xeno/codex-secondary-timing@234ab7c`. Change: keep the device metadata layout and Q2 kernels unchanged, but build the five control arrays (group pointers, starts, count, destinations, token indices) in one reusable portable pinned host buffer with the same offsets. Each active secondary layer now submits **one** packed metadata H2D and the existing pinned activation H2D instead of five pageable metadata H2D calls plus activation. The pinned buffer remains allocated for the runner lifetime. Output clear, grouped kernels, full D2H and host scatter are unchanged.

## Correctness and safety

The direct runner test now exercises full routing, partial routing, then one group after two groups, on both the default and profiling paths. CPU/GPU float parity and the display reserve tests passed. The packed binary produced exactly the accepted **256/256 raw IDs** for sky, Thai, code and long with MTP `--spec 4`, `--prefill 2048`, six workers and `CUDA_VISIBLE_DEVICES=1,0`. It used the same EXL3-only ranked profile, 6,653 primary slots / 8.57 GiB and 6,602 secondary slots / 8.5 GiB as the timing baseline. No default launcher/profile, PR or merge changed. Full build, CTest `xeno_|^gpu_` **18/18**, Python `tests/xeno` **28/28**, and `git diff --check` passed.

## Measured effect and limit

| Code prompt, profile on | Before | Packed |
|---|---:|---:|
| Host enqueue, ms/round | 7.356 | 4.558 |
| CUDA event H2D interval, ms/round | 3.496 | 0.744 |
| CUDA event clear interval, ms/round | 0.713 | 2.441 |
| CUDA event expert interval, ms/round | 15.455 | 15.243 |
| CUDA event D2H interval, ms/round | 2.173 | 2.066 |
| Secondary event wait after CPU pool, ms/round | 5.263 | 5.798 |

The event intervals can include gaps while the host submits work. The clear interval growing while H2D shrank is a warning against reading each event interval as isolated hardware work. The nonzero secondary completion tail remains.

The same-machine, fixed-configuration, profiler-**off** code sequence (baseline binary, packed binary, packed binary, baseline binary) gave:

| Arm | Decode | Secondary `launch` | CPU gate/up |
|---|---:|---:|---:|
| Baseline 1 | 45.58 tok/s | 5.795 ms/round | 13.191 ms/round |
| Packed 1 | 37.30 tok/s | 7.333 ms/round | 22.176 ms/round |
| Packed 2 | 48.25 tok/s | 3.935 ms/round | 11.556 ms/round |
| Baseline 2 | 47.97 tok/s | 5.080 ms/round | 11.534 ms/round |

All four arms produced identical 256 raw IDs. The two packed arms differ by 29% in tok/s while their CPU gate/up work differs almost 2×; the change did not touch that CPU kernel. Comparing the two low-CPU-time arms gives 48.25 vs 47.97 tok/s, a 0.6% gap. **This proves fewer H2D API calls and a lower profiled enqueue/H2D interval, not an end-to-end speedup.** Keep the code as an opt-in branch checkpoint; further throughput claims need controlled repeats and a larger effect than the observed machine noise.

Artifacts: `%TEMP%\strata-codex-secondary-profile-code256`, `%TEMP%\strata-codex-packed-profile-code256`, `%TEMP%\strata-codex-secondary-unprofiled-code256`, `%TEMP%\strata-codex-packed-abba-{packed-1,packed-2,baseline-2}`, `%TEMP%\strata-codex-packed-parity-{sky,thai,long}256`. Each has exact command, executable SHA-256 and raw output IDs. The display card's sampled free VRAM stayed above the configured 640 MiB experimental floor in the profiled code arm (minimum 1,031 MiB); production shared-headroom policy remains open.

Next ablation: return only secondary-owned result rows from 4070, while retaining the same grouped Q2 arithmetic and original router-row map on host. It targets the still-present full-row D2H and completion tail, and must have its own parity/performance gate.
