# Native Q2 decode GPU trace — Nsight Systems

The existing `--gpu-stages` diagnostic cannot run on the native Q2 pack: `generate.cpp` deliberately skips `session_capture` for native experts, while the diagnostic requires `SessionGraphs::split_captured`. A probe exited with `session_replay_stages: not captured with split` after loading; it produced **no usable stage numbers**. Forcing a different single-token graph would profile a different execution path.

An opt-in `--profile-decode-range` now calls `cudaProfilerStart/Stop` around the actual speculative verifier decode loop. Nsight Systems 2026.1.3 collected CUDA graph **node** activity for a 256-token code run of the compact-D2H branch, with CPU sampling disabled because administrator privileges were unavailable. The profiler reported a successful process exit and wrote `%TEMP%\strata-codex-primary-nsys-code256\decode.nsys-rep` (plus SQLite export and exact command JSON). The same binary and placement passed 256/256 code parity in the earlier unprofiled run; Nsight's child stdout was not captured as a raw-token artifact, so this trace is **timing evidence, not an additional parity check**.

The run used the EXL3-only ranked profile, 6,653 primary cache slots, 8.5 GiB secondary tier, six CPU workers, MTP `--spec 4`, `--prefill 2048`, `--max-context 8192`, and `CUDA_VISIBLE_DEVICES=1,0` (logical CUDA 0 = 5060 Ti, logical 1 = 4070 SUPER). The captured primary `native_gu_kernel` count was 7,200 = 75 verifier rounds × 48 layers × two groups, consistent with decode-only capture.

## Per-device activity in the trace

| Logical device | Kernel + copy activity union | First-to-last activity span | Idle gaps in that span | Share active |
|---|---:|---:|---:|---:|
| CUDA 0 / 5060 Ti | 4,839.47 ms | 5,366.69 ms | 527.22 ms | 90.2% |
| CUDA 1 / 4070 SUPER | 1,121.98 ms | 5,361.33 ms | 4,239.36 ms | 20.9% |

These are unions of recorded CUDA kernel/copy intervals, **not GPU utilization samples, end-to-end decode wall time or causal critical-path savings**. Concurrent streams can overlap; an individual kernel-duration sum must not be added to another device's wall time. Trace instrumentation may change scheduling.

On the 5060 Ti, `native_gu_kernel<Q2_0>` took 1,594.1 ms over 7,200 calls and `native_down_kernel<Q2_0>` 883.4 ms over 7,200 calls. Together they account for **2,477.5 ms, about 51.2% of the primary's 4,836.8 ms summed kernel durations**. Other notable primary intervals were `wait_flag_ge_kernel` 370.7 ms (includes GPU-side waits), Q3K projection kernel 250.1 ms, `copy_from_mapped_kernel` 232.3 ms, GR down 220.0 ms and GR up 152.3 ms. The primary Q2 expert kernels are the largest measured GPU work class in this trace.

On the 4070 SUPER, Q2 gate/up took 706.5 ms and down 362.3 ms, totaling 1,068.8 ms of its roughly 1,084.6 ms kernel duration sum. Nsight's memcpy records for that card showed H2D **142.3 MiB / 16.75 ms** and D2H **345.6 MiB / 20.66 ms** for this capture. The D2H byte count agrees with the compact runner's 345.87 MiB API-requested counter in a separate 256-token code run. It is much smaller than the earlier CUDA-event “H2D/D2H intervals,” validating the caveat that event bracketing across host enqueue calls includes idle gaps. [NVIDIA's event API](https://docs.nvidia.com/cuda/cuda-runtime-api/cuda_runtime_api/group__CUDART__EVENT.html) documents asynchronous event-record timing limits.

## Static load-balance probe

At a fixed 8.5 GiB secondary cap and identical EXL3-only rank order, reducing the primary cache from 6,653 to 5,000 slots changed code tier hits from **56.6/25.6/17.7% primary/secondary/CPU** to **48.2/28.6/23.3%**. Thai changed from **43.7/29.4/26.9%** to **33.9/33.3/32.8%**. The secondary tier gained some of the moved experts but displaced other residents to CPU. Both 5,000-slot prompts still matched accepted greedy IDs 256/256. Their one-run tok/s values (code 49.08, Thai 30.51) are not a speed verdict against the earlier 6,653-slot runs because CPU gate/up speed varied substantially.

Two alternating Nsight decode captures at each placement give stronger mechanism evidence while still carrying profiler/background-load limits:

| Capture order | Primary slots | Primary Q2 gate/up + down | Primary `wait_flag_ge` | Primary activity span | 4070 activity union |
|---|---:|---:|---:|---:|---:|
| baseline 1 | 6,653 | 2,477.5 ms | 370.7 ms | 5,366.69 ms | 1,121.98 ms |
| moved 1 | 5,000 | 2,130.3 ms | 1,692.3 ms | 6,360.65 ms | 1,281.80 ms |
| baseline 2 | 6,653 | 2,477.3 ms | 846.1 ms | 5,953.44 ms | 1,151.11 ms |
| moved 2 | 5,000 | 2,131.0 ms | 1,355.4 ms | 6,057.65 ms | 1,298.46 ms |

Primary Q2 kernel time fell by about **347 ms**, but GPU-side wait-flag time rose by about **915 ms on the two-run means**. Primary activity span rose about **549 ms** on those means. The exact kernel counts stayed 7,200 primary Q2 gate/up and 7,200 down in each capture, so this is less work *inside* the kernels rather than fewer graph node launches. This supports the hypothesis that reducing primary cache under a fixed secondary cap increases expert-result rendezvous time, consistent with more CPU-routed entries. The wait flag also varied **371 → 846 ms** between the two *unchanged-baseline* captures, so a production placement decision still needs controlled, profiler-off alternating decode runs. Increasing secondary hits alone did not remove the tail in this probe.

Additional artifacts: `%TEMP%\strata-codex-balance-p5000-{code,thai}256`, `%TEMP%\strata-codex-primary-nsys-p5000-code256`, `%TEMP%\strata-codex-primary-nsys-p6653-repeat`, and `%TEMP%\strata-codex-primary-nsys-p5000-repeat`. The `.nsys-rep` and SQLite exports retain per-device kernel and memcpy timestamps. The existing 4070 shared-headroom floor was preserved.

These graph traces do not split primary expert gate/up versus down **by layer** or isolate pure GPU arithmetic from waits inside every graph node. They show that optimizing only metadata H2D and D2H transfer volume has a modest ceiling relative to primary expert compute and rendezvous time on this workload.
