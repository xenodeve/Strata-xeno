# Phase 3 secondary Q2_0 compute checkpoint

Scope: engine #3 / tracker #12, branch `xeno/afk-phase3-compute`. This is an opt-in correctness and rough-speed checkpoint, **not a production default or final peak-performance result**.

`SecondaryRunner` stages activation in portable pinned host memory, sends it to device 1, uses the native Q2_0 FP32-scale quantizer and grouped multi-token kernel with verified static expert slots, returns partial rows through portable pinned host memory, and waits on a device-1 event before copying only claimed router-index rows into the verifier's host output. In `expert_pool_dispatch_multi`, primary-GPU resident and PCIe groups are decided first. Remaining entries with a secondary slot become kind 2 and are omitted from CPU jobs. Other entries remain CPU misses. The runner launches before CPU work and finishes after it, so GPU1 and CPU can overlap. `--pcie-frac 0` is required for the initial combined-routing contract. `--secondary-stage-only` and `--cache-cpu-only` remain same-binary A/B arms.

Direct real-model tests with both contexts and two secondary Q2_0 blobs passed for layer 0 (half-integer Q8 boundary input) and layer 47. Each `gpu_dual_q2_runner_*` CTest compared three entries in two groups to the CPU pool: 0/7,680 differing floats. A second call selected only rows 0 and 2; row 1 stayed untouched. The separate `gpu_4070_transfer` test covers portable pinned H2D, kernel, event and D2H. These are the component proofs for the later model run.

## Model token parity and rough speed

All runs used `CUDA_VISIBLE_DEVICES=1,0`, Q2_0 native pack, spec 4, 4K context, six P-core workers, PCIe miss share 0, and a fixed sky prompt. The published accepted Q2_0 baseline was retained:

| Primary cache | Secondary cap | Result | Secondary work | Rough decode tok/s | Lower 4070 free at staging |
|---:|---:|---|---:|---:|---:|
| 1,000 slots | 64 MiB | forced CPU-miss vs secondary compute 96/96 equal | 284 entries / 205 groups | 18.16 off / 22.49 on | >9 GiB |
| 5,000 slots | 256 MiB | forced CPU-miss vs secondary compute 96/96 equal | 598 entries / 465 groups | 20.47 off / 25.16 on | 9.24–9.25 GiB |
| 5,000 slots | 1,024 MiB | accepted single-GPU output vs secondary compute 96/96 equal | 2,422 entries / 1,891 groups | 22.43 on (unpaired) | 8.49 GiB |

The 1,000-slot trial initially queried NVML every layer and ran slower (13.39 on vs 21.12 off tok/s) despite parity. It spent ~84.7 ms/round in dispatch planning. Throttling the reserve query to at most once per 250 ms reduced that term to ~9.2 ms/round; the repeated trial passed parity and showed the rough gain above. The 5,000-slot 256 MiB trial spent ~8.7 ms/round in dispatch, while the 1 GiB trial spent ~13.3 ms/round and served more groups but decoded slower. This suggests transfer/launch/wait overhead can outweigh additional CPU work saved. Startup times varied widely with host arena load rate (e.g. 31.64 GiB at 1.21 vs 0.61 GiB/s in adjacent runs); do not infer a startup penalty from those unpaired totals. Artifacts are under `%TEMP%\strata-phase3-compute-*`.

## Safety and remaining gates

The 4070 allocation is touched and checked against the lower CUDA/NVML reading before use and after each expert fill. A dedicated monitor now samples the display card every 100 ms throughout the tier lifetime, including idle periods and layers without secondary hits. A sampled value below 2560 MiB, or a failed query, logs and exits the owning Strata process with code 3 so Windows reclaims its GPU allocations. This is an experimental fail-safe, not graceful fallback: an external desktop allocation can still breach the floor between samples, and the serving launcher/watchdog must recover from an exit before this tier becomes a default. The 96-token trial after this change passed 96/96 parity, with 75 reserve samples in the forced-CPU arm and 55 in the compute arm. A direct test injects a below-floor reading into the monitor callback and verifies it signals the breach without actually exhausting display VRAM.

Independent review found that the first runner version copied group metadata on the default stream while its kernel ran on a nonblocking stream. All metadata copies are now queued on the runner stream, and their host sources live as runner members through `finish()`. The device-1 parity test passed 20 repeated runs after this correction.

Additional work: four 256-token prompt parity gate, tests for secondary launch/error cleanup during an active verifier graph, accurate per-tier timing/bytes, paired ABBA speed runs, larger primary-cache configurations, and a live Claude Code session. The 1 GiB tier was slower than 256 MiB in one sky trial, so do not maximize the display-card fill without measurements. The large host expert arena is pageable in opt-in dual-card mode because CUDA registration of ~29 GiB repeatedly prevented initializing the second CUDA context on this 48 GB RAM machine. Phase 4 exclusive residency may relieve that pressure; this remains to be measured.

## Four-prompt 256-token gate and monitor correction

With primary cache 5,000 slots and the 256 MiB secondary cap, the same-binary forced CPU-miss and secondary-compute arms matched **256/256 raw output IDs on each of sky, Thai, code and long**. Every secondary-compute arm also matched the accepted Phase 1 `final256-*` token file 256/256. Profile rank, model and prompt were fixed; the run order was off then on, so the following single-pair speeds remain rough and may reflect changing host load.

| Prompt | Decode off / on tok/s | Secondary entries / groups | Lower 4070 free at staging |
|---|---:|---:|---:|
| sky | 20.97 / 25.49 | 2,083 / 1,564 | 9.12–9.18 GiB |
| Thai | 22.80 / 27.06 | 1,891 / 1,583 | 9.26–9.27 GiB |
| code | 30.57 / 26.28 | 1,021 / 730 | 9.27–9.28 GiB |
| long | 28.82 / 36.55 | 1,679 / 1,226 | 9.28–9.30 GiB |

The code prompt was slower with secondary compute in this pair. A later single sky 256-token compute run with the same binary/token output measured 35.76 tok/s instead of 25.49, underscoring that background load/cache state changes the rough rates. Do not choose an automatic per-request policy from these values. Artifacts: `%TEMP%\strata-phase3-compute-256-{sky,thai,code,long}` and `%TEMP%\strata-phase3-monitor-min-sky256`.

The independent review found an actual stream-ordering risk: metadata H2D copies were on the default stream while the kernel used a nonblocking stream. All metadata copies now queue on the runner stream, and their host sources are members that live through `finish()`; the layer-0 runner parity test passed 20 repeats after the correction. The monitor now polls CUDA/NVML from a dedicated thread every 100 ms throughout the tier lifetime, including idle time, so polling does not occupy every verify layer. A synthetic below-floor sample triggered the breach callback in a test; a separate child-process test verified the production fail-fast path exits the owning process with code 3 and the expected log line. No unrelated process is touched. A sampled breach releases the tier through process exit; graceful same-session CPU fallback is still open. External allocations can occur between polls, so this is a sampled safety guarantee.

After adding a minimum-free telemetry counter, a full sky 256-token compute run matched the Phase 1 baseline and reported **82 monitor samples with minimum lower free 9.18 GiB**. The other three 256-token artifacts predate the minimum counter but include staging samples and monitor counts. The 2560 MiB floor was not approached in these tests.
