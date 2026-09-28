# AGENTS.md — rules for coding agents working on Strata-xeno

## Measure latency, do not guess where the time went

Every claim about **why** decode or prefill got faster, slower or stayed flat must
come from the per-stage latency counters of **the same runs** being compared. Do not
estimate it from reasoning, from another run, or by combining numbers from different
runs or tools.

- **Read the counters `strata generate` already prints** for every run:
  `verify window` (`wait for rings`, `pool`, `host`, `commit`), `pool multi`
  (gate/up, quantize, down), `dispatch`, `dispatch detail` (CPU pool, secondary
  finish), `secondary timing` (launch, wait), `mtp`, and `tier hits`. Keep each
  run's stdout/stderr, and parse them with `tests/xeno/latency_breakdown.py`
  rather than by eye.
- **Compare A and B stage by stage** from paired, alternating (ABBA),
  profiler-off runs. Report the stage whose milliseconds moved, not a story.
  Example: if a kernel change lowers `wait for rings` but tok/s stays flat, the
  table shows which stage absorbed the time.
- **If a stage is not instrumented, add a timer** (off the default path, or cheap
  enough to leave on) and measure it. Do not infer it from the residual.
- **Know what each counter means before quoting it.** `wait for rings` is the
  host waiting for the **primary GPU** doorbell. It is not "the GPU waiting for
  CPU experts". That mislabel was published once and had to be retracted
  (Strata-xeno #3).
- **Nsight Systems** is for splitting a stage further, for example the kernels
  inside `wait for rings`. It is not a replacement for the paired counters: the
  profiler changes scheduling, and a trace from one run cannot be subtracted
  from another run's counters.
- **Anything not measured this way is a hypothesis** and must be labelled as
  one. The 2026-09-28 estimate that a faster Q2 kernel would "gain only
  single-digit %", built from an Nsight trace plus a different run's dispatch
  counters, is the example not to repeat.

## Run every speed measurement at high CPU priority

Other programs on this machine (browser, Discord, Wallpaper Engine, Defender scans) take CPU time from the pool's
pinned workers. A pinned worker cannot migrate, so one preempted worker stalls the whole layer. That is the
most likely source of the random slow runs (CPU rows 13–14 GB/s against 19–24 GB/s on identical work,
2026-09-28 sweeps).

- **Every A/B, ABBA or sweep passes `--pool-priority 2`** (THREAD_PRIORITY_HIGHEST for the pool workers and the
  host thread) **in every arm**, so both sides get the same condition. Record it in the report.
- Report whether a run was disturbed (for example, another CPU-heavy job was running) rather than silently
  keeping it. Do not run builds or other CPU-heavy work while a measurement is in flight.
- Measured numbers at high priority describe the engine, not a normal-priority server. Whether the serving
  launcher also raises priority is a separate decision for the developer, backed by the same measurements.

## Other standing rules

- **The correctness gate is greedy raw-token parity against a same-session
  baseline.** Run the unchanged build alongside the candidate in the same ABBA
  session. Token IDs recorded in an earlier session are not a gate: on
  2026-09-28 the same exe with a byte-identical command produced different
  tokens hours later, diverging already in prefill
  (`docs/reports/2026-09-28-ggml-cpu-build-mode.md`). `native_q2_pool_hit_parity`
  must stay bit-exact (ADR `docs/adr/0001-q2-exp-baseline.md`).
- **Record the exe sha256 with every run** and check it before reusing a build
  dir: build dirs get rebuilt in place.
- **ggml-cpu SIMD mode does not matter here.** Generic, AVX2 and AVX2+VNNI gave
  identical tokens and speed within 4 %. The native Q2_0 CPU kernel is Strata's
  own AVX-VNNI code. A configure shell with an empty `CMAKE_SYSTEM_PROCESSOR`
  silently yields `GGML_CPU_GENERIC`, which is harmless but should not be
  mistaken for a real difference between builds.
- **Launching `strata.exe` from a script** needs both CUDA `bin` and `bin/x64`
  on `PATH`. Without them the exe exits with `0xC0000135` before loading
  anything.
- **Effects under the 13.6 % noise gate are unproved** unless paired ABBA runs
  show them consistently.
- **GPU roles.** Primary is the RTX 5060 Ti (`CUDA_VISIBLE_DEVICES=1,0`, logical
  0). Secondary is the RTX 4070 SUPER display card, which keeps **2.5 GB total**
  headroom for other processes, **including** the desktop's current use.
