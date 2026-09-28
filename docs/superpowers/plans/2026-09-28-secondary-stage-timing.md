# Secondary GPU Stage Timing Implementation Plan

> **For agentic workers:** Implement one checked slice at a time. This plan covers an opt-in measurement tool, then a data-led optimization; it does not authorize making the dual GPU configuration a default.

**Goal:** Explain the 4070 SUPER's 7–14 ms/round launch-plus-wait cost without changing default decode results or consuming the display card's shared 2.5 GB process headroom.

**Architecture:** Keep the current `SecondaryRunner::launch`/`finish` order. In an opt-in profiling mode, record CUDA events on its device-1 stream around H2D, clear, quantize, grouped expert computation and D2H. Measure host plan, device switch, enqueue, timing-query overhead and copy-out separately; print cumulative values divided by verifier rounds. CUDA stream elapsed time may include gaps while the host enqueues commands, so it is diagnostic rather than pure kernel time.

**Spec:** Engine issue `xenodeve/Strata-xeno#3`, measurement tracker `xenodeve/Qwen3.8-Flash-Next-Tuning#12`, the user-provided `%TEMP%\handoff-strata-gpu-bottleneck-2026-09-28.md`, and `Downloads/strata-second-gpu-decode-source-audit-2026-09-28.md`.

## Constraints

- Wait for `%TEMP%\strata-claude-batch4.log` to say `DONE` before compiling or starting another model.
- Use branch `xeno/codex-secondary-timing`, child of `xeno/claude-prefill-mtp`; no PR, merge or default launcher/profile change.
- Keep `CUDA_VISIBLE_DEVICES=1,0`, the accepted greedy Q2_0 baseline, MTP `n=4` and the 4070 shared 2.5 GB headroom in all comparisons.
- Check bit-exact raw IDs after engine changes. Treat one-run speed differences below 13.6% as noise until alternating repeat measurements support them.
- The source audit found an upstream CPU-pool late-wakeup race fix absent from this fork. Keep it on the correctness backlog before long high-hit-rate stress tests; do not misattribute the current measured latency gap to that race without a reproducer.

## Task 1: Opt-in secondary stage timings

**Files:** `include/strata/core/secondary_runner.hpp`, `src/core/secondary_runner.cpp`, `src/program/generate.cpp`, `tests/xeno/native_q2_pool_hit_parity.cpp`, `CMakeLists.txt`, `tests/xeno/cache_tokens.py`, `tests/xeno/latency_breakdown.py`, `tests/xeno/test_latency_breakdown.py`.

- [x] Add a failing parser test for the two new summary lines, then make it pass.
- [x] Correct `wait for rings` to host waiting for the primary GPU doorbell; include verifier `pool` as a host callback interval and stop labeling the unmeasured remainder as GPU compute. The source audit identified the nesting, and `verify.cpp`/`expert_source.cpp` confirmed it locally.
- [x] Add a direct GPU test that profiles two launches and preserves the existing default runner test.
- [x] Add optional CUDA events and host counters; retain unchanged normal-path allocations and event count.
- [x] Build and run all `xeno_`, `gpu_` and serving gates after Claude's batch finishes (18/18 CTest, 28/28 Python).

## Task 2: Locate the first bottleneck

- [x] Run one profiled 256-token code prompt on the best EXL3 profile at the same 8.5 GiB secondary tier as Claude's baseline; compare raw token IDs and report host/device milliseconds per round.
- [x] Run a representative Thai prompt to test whether the same stage dominates. Capture per-run command, binary hash, VRAM floor, and timing output.
- [x] Form ranked bottleneck hypotheses and falsifying measurements below. Candidate transport reference: upstream PR #16's packed pinned metadata and compact D2H rows. Treat its speed impact as a hypothesis until the local timing shows where the 4070 tail comes from.

### Hypotheses after the first code/Thai profile

1. Five pageable metadata H2D calls cause material host enqueue delay (5.75–7.36 ms/round while timed, normal code `launch` 5.80). **Disproof:** one packed pinned metadata copy cuts API count but host enqueue and full-window time do not fall in alternating unprofiled runs.
2. Full-row D2H adds secondary completion tail (1.70–2.17 ms/round event interval, nonzero post-pool wait). **Disproof:** compact output cuts measured D2H bytes and stage interval but the post-pool wait and complete window remain unchanged.
3. Grouped expert kernels dominate device stream time (9.71–15.46 ms/round event interval), though much overlaps CPU. **Disproof:** when a future change reduces kernel interval at fixed routing, the post-pool wait and window wall time do not fall.
4. Repeated device select/restore dominates host path. **Already weakened:** measured only 0.21–0.38 ms/round in timed runs; retain as low priority.

The CUDA event intervals may include host enqueue gaps, and six event markers per active layer raised measured code `launch` by about 2 ms/round in one profiled/unprofiled pair. Use profiler data to locate a stage, then measure throughput with profiling off.

## Task 3: One optimization at a time

- [ ] Write the smallest useful red test at the public runner/dispatch seam, implement one change (candidate: packed pinned metadata H2D or compact device output), and get the direct parity test green.
- [ ] Recheck four fixed prompts for raw-token parity and the required CTest/Python gates.
- [ ] Use alternating same-boot benchmarks against the unoptimized child-branch commit before claiming a speed gain; update engine/tracker issues bilingually with evidence.
