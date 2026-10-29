# Native Qwenputer P4 checkpoint: autonomous CUDA execution

Status: standalone functional gates verified; WHILE instrumentation coverage blocked. This document separates functional evidence from the
remaining instrumentation and integration gates. P5 language-model integration
has not started. The main-based isolated branch and real hub baseline are unchanged.

## Completed evidence

The frozen development bundle ran autonomously from a text prefix or one trusted
GO seed: 79 cases, 74,624 actual steps, all trace, output, final-state and lag checks
passed. Eight instruction-fault cases terminate at their fault token rather than
executing the reference diagnostic tail. The 29 arithmetic/error cases include
21 solved expressions and eight explicit firmware errors/refusals; they are not
29 successfully solved expressions.

The first native result was `17*23 -> 391`: 23 supplied prefix tokens, 1,543 total
steps, 13 bounded launches. The GPU selected the remaining inputs. The production
prefix-only CLI has no expected-answer or reference-trace argument. This is
compiled FFN/GDN tensor execution on the primary GPU, without a host calculator.
It is not yet a Gyro-M integration result or an end-to-end speed claim.

Thirteen control cases passed: chunk sizes 1/2/5, small output FIFOs, zero work,
relaunch, invalid-prefix state preservation, resume from a consumed TCC, cumulative
step/output caps and cancellation. A regression test exposed and fixed a fuel
counter that originally excluded prefix steps. The later held-out, deadline, device
ownership and repeated-state checks below also pass.

## Instrumentation investigation

The P3 single-step path passed memcheck, initcheck, synccheck and racecheck with
the installed sanitizer's named-pipes connection. This does not cover P4 WHILE.

P4 memcheck reaches the target but returns CUDA error 999 at event synchronization.
It does not report a specific invalid read/write. Reducing the chunk from 128 to
one step does not resolve it. The existing tiny, manually constructed WHILE probe
passes memcheck. Checking only `begin_chunk` passes; checking only the captured
`advance` kernel reproduces error 999. A manual-versus-captured control-node
experiment is prepared. No driver, TDR, HAGS, firewall or security setting changed.
The System event log was unavailable due to access permissions, so no TDR cause
is claimed. There is no debugger on the inspected PATH.

The frozen 64-case held-out corpus ran through the prefix-only CLI. An
offline checker initially rejected the pinned firmware's empty `<think>` suffix.
The response numbers were unaffected. An exact-framing regression test failed
before the checker correction and passed afterwards. Original raw results and the
original report remain intact; the independent recheck was saved as a
separate report. No circuit, corpus or expected answer was tuned from these cases.

## Gate coverage

| Gate | Result |
|---|---|
| Autonomous development | PASS:79 cases,74,624 steps, full trace/state/lag comparison |
| Frozen held-out | PASS:64 exact cases,183,052 steps, repeated with final binary; output/state/counters identical |
| Bounds and lifecycle | PASS:16 controls including cumulative step/output/deadline, FIFO, cancellation and retained-state repeat |
| Artifact/device/stream ownership | PASS: negative controls refuse before allocation, valid primary run remains exact |
| WHILE initcheck | PASS:0 errors |
| WHILE memcheck/synccheck/racecheck | BLOCKED: minimized installed-toolchain instrumentation failure; not passed |
| Resource/timeline | Measured below; no model integration or ABBA speed claim |
| Gyro-M integration and daily-use readiness | NOT RUN: later P5-P8 gates |
| HIP/SYCL | NOT RUN: unavailable non-target toolchains; no cross-backend coverage claim |

The local functional checkpoint is ready. The blocked WHILE instrumentation checks
must remain open while independent integration work proceeds.

GitHub PRD #276 was freshly read and still contains R4 (updated
2026-10-28T20:02:21Z). The local PRD is R9. The earlier protected Submit operation
has not been bypassed; no fresh write was issued. Its UI approval metadata is not
available in this task.

## Additional findings

The installed WHILE path passes initcheck. Memcheck returns error 999,
synccheck reports barrier faults, and racecheck terminates unsuccessfully. A
minimal first-party kernel with unconditional block barriers reproduces the same
failure inside a six-iteration WHILE graph, while its ordinary launches pass
under the tools and the uninstrumented WHILE run completes. Source instrumentation
(`-fdevice-sanitize=memcheck`) also fails inside the sanitizer runtime. The
captured-versus-manual control-node hypothesis was falsified. These checks remain
validation debt, not successful instrumentation. The lower-level driver/tool
fault has not been identified; no production workaround or system change was made.

Cumulative deadline controls now stop at zero steps for a zero deadline and at
one step after the deadline expires between chunk launches. The timer is never
renewed by relaunch. A retained-state test correctly revealed that a fresh-boot
reference cannot simply be reused for a second call: the output is still 391 but
four trace tokens differ. A persistent upstream/IR reference independently reproduces
exactly those four changes (indices550/551/576/601), while the final state remains
identical. The harness now accepts the second-run reference on the host only and
requires its supplied prefix to stay identical. No reset, teacher forcing or relaxed
numerical assertion has been used to hide the result. Final native verification is pending.


## Measured resource and scheduling observations

A profiler-off standalone CLI run owns 31,804,988 device-buffer bytes. Its Windows
process private-commit peak is 316,338,176 bytes and working-set peak 175,882,240
bytes, including startup, artifact loading, execution and final-state hashing.
Sampled global device free memory falls by 35,651,584 bytes after Executor setup;
that is not an exclusive process VRAM peak. Minimum sampled system commit headroom
is 60,566,884,352 bytes and physical availability 27,639,128,064 bytes. No process
memory cap is used. These observations do not include the language model or the
later base/shadow state copies.

A separate timeline-enabled `17*23` run records 13 chunks, 1,543 steps and
5,067.383 ms of GPU work; maximum chunk duration is 422.091 ms. Host enqueue and
collect total 2.201 ms and overlap the GPU wait, so they are not added to GPU time.
A cancellation trace stops after four steps; request start to completed collection
is 3.4404 ms. These are instrumentation observations, not ABBA speed claims or
proof of daily-use readiness. GPU execution still needs substantial optimization.

Actual two-device testing refuses a wrong current device and a foreign stream
before runner allocation, then completes the valid primary-device computation
with zero trace/state mismatches. No kernel runs on the display GPU.

A final API review found that Runtime did not bind its Artifact argument to the
Executor's uploaded artifact. A negative test reproduced `mismatched_artifact_accepted`.
The fix stores the validated digest in Executor and checks it before any Runtime
allocation; its green test refuses the mismatched artifact and foreign device/stream, then
runs the valid computation with zero mismatches. The final binary repeat passes all64 cases with identical outputs, full state hashes and counters.


## Final standalone repeat

The final prefix-only CLI (`5eb883a1e886adbba239eadd8123665f50a3327a82083c6e688ddfbb0e78c133`)
passes all 64 frozen held-out cases, 183,052 steps. A separate comparison confirms
identical result token IDs, full final-state hashes, terminal reasons and all
step counters against the first run. Artifact identity remains
`edcd063736a3b3575cf81b5bac524943e0694e23c12af0cd261b44b150e704b0`;
corpus identity remains `d1ddd54c5c86d93eab55e3bee98873be16fa777b175583e76b5ce77e34da32f3`.
The selected regression suite passed111 tests and110 subtests with4 stated skips;
the later digest-pairing fix has its own native negative/positive control and this
final 64-case repeat. Whole-model integration and speedup remain untested.

P4 is not declared fully complete: WHILE memcheck/synccheck/racecheck coverage is
blocked by the minimized installed-toolchain failure. Initcheck passes. This debt
must stay visible through integration and cannot be used as a release/default gate
pass. Independent P5 ownership tests may proceed; they do not waive the debt.


The source-instrumentation control uses NVIDIA's documented
[`-fdevice-sanitize=memcheck` option](https://docs.nvidia.com/cuda/archive/13.3.0/pdf/CUDA_Compiler_Driver_NVCC.pdf).
The minimal source is [conditional_barrier_probe.cu](../../tests/qwenputer/conditional_barrier_probe.cu).
Changing runtime linkage and adding a manual control node both failed to resolve
this instrumentation fault, so neither experiment changed the production path.
