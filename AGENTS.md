# AGENTS.md — rules for coding agents working on Strata-xeno

## Where did the time go? Record a timeline, do not hunt stage by stage

The developer's rule (2026-09-29): **do not hunt for the slow part one point at a time.** The engine records the
whole pipeline in one run. Start every "why is this slow / where does the time go" question here, **before** adding
ad-hoc timers, hand-built host traces or an Nsight capture:

```sh
STRATA_TIMELINE=run.json strata.exe <the same args>      # generate or --serve; no other flag needed
python tests/xeno/perf/timeline.py run.json               # the budget; --perfetto closed.json for ui.perfetto.dev
```

What the report already answers:

- **Prompt path:**
  - GPU time by phase and by layer, and the host thread's budget.
  - Why the copy engine idled: *not issued yet (host)* (and what the issuing thread was doing), *in the issue call*,
    or *issued, not started (device)*.
  - Copy durations by the compute kernel running at the same time, and the largest gaps by layer.
- **Decode:** ms per round of every stage, with the unaccounted rest; the slowest layers; the 4070's busy time; the
  pool workers' wake-up and stragglers.
- **Helper threads** (stagers, issuer, adapt), **lane utilisation**, and **server requests** from HTTP to first token.

The tool's lanes, usage and cost are in `tests/xeno/perf/README.md`, and its first findings in
`docs/reports/2026-09-29-pipeline-timeline.md`. #31 needed about 12 hand-built experiments before this tool existed;
the first timeline run answered it.

- **A stage the timeline does not show yet gets a span in the timeline**, not a one-off timer or a printf.
  - Host code: `strata::timeline::Span` / `complete()` (`include/strata/timeline.hpp`).
  - A GPU stream: `timeline::GpuClock` (`include/strata/timeline_gpu.hpp`).
  - The server: `serve/timeline.py`.
  - Names are string literals; numbers go in the two integer arguments.
  - Add the analyzer's test first (`tests/xeno/test_timeline.py`).
- **The timeline splits a stage; it does not replace the paired counters.** It costs about 6 % of an 8K prefill.
  - A/B speed claims still come from profiler-off ABBA runs and their counters (next section).
  - Use the timeline to explain which stage moved.

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
- **The pipeline timeline** (`STRATA_TIMELINE=<file>`, then `tests/xeno/perf/timeline.py`; #33) is the first
  tool for **where** a stage's time goes. It records every thread and GPU lane on one clock: the prompt
  path's phases per layer, each expert copy and why the copy engine idled, each decode round's stages, the
  pool workers, the 4070, and the server's requests. It costs about 6 % of an 8K prefill, so it splits a
  stage and does not replace the paired counters for A/B numbers.
- **Nsight Systems** is for splitting a stage further, for example the kernels
  inside `wait for rings`. It is not a replacement for the paired counters: the
  profiler changes scheduling, and a trace from one run cannot be subtracted
  from another run's counters.
- **Anything not measured this way is a hypothesis** and must be labelled as
  one. The 2026-09-28 estimate that a faster Q2 kernel would "gain only
  single-digit %", built from an Nsight trace plus a different run's dispatch
  counters, is the example not to repeat.

## Do not guess a fix: measure the mechanism, then design the fix from it

The developer's rule (2026-09-30): **never write, propose or queue a fix from a guess about where the time goes.**
A guessed fix wastes the build, the A/B and the write-up when the guess is wrong, and here it usually is. First
measure the exact mechanism the fix would change, from the run itself. Then design the fix from that number.

- **Before proposing a change, name the measurement that shows the time it removes.** Give the span, the counter or
  the file. If no tool shows it, extend the tools first: a timeline span, `decode_paths.py`, `decode_gpu_layers.py`,
  or `prefetch_sim.py`, with its test. Then measure.
- **Measure the ceiling before the build.** Take the time the fix can remove at most from the measured spans. If
  that ceiling is below what an ABBA here can resolve, do not build it.
- **Averages hide the mechanism.** Split a wait by its cause before reading it.
  - Example: the 4070 event sync is 0.9 ms/round on average.
  - Split by whether the GPU had finished (`decode_paths.py`), it was 0.08 ms of sync cost and 0.77 ms of real
    waiting for a 4070 still computing.
- **The three guesses of 2026-09-30 that measurement overturned (#44),** each once written down as a plan:
  1. "The 4070 already returns after the CPU pool every round." In fact the CPU pool is the tail: the 4070 finishes
     first in 72 % of layers.
  2. "A prefetch must cover every CPU expert of a layer." In fact the pool's time is per expert, so a partial cover
     also shortens it.
  3. "The 4070 sync mostly waits on a GPU that is already done; a spin or query-first wait would remove it." In fact
     it saves at most 0.08-0.15 ms/round.

  Two of them led to proposed work, the prefetch plan and D0, that the next measurement cancelled.

## Run every speed measurement at high CPU priority

Other programs on this machine (browser, Discord, Wallpaper Engine, Defender scans) take CPU time from the pool's
pinned workers. A pinned worker cannot migrate, so one preempted worker stalls the whole layer. That is the
most likely source of the random slow runs (CPU rows 13–14 GB/s against 19–24 GB/s on identical work,
2026-09-28 sweeps).

- **Every A/B, ABBA or sweep passes `--pool-priority 2`** (THREAD_PRIORITY_HIGHEST for the pool workers and the
  host thread) **in every arm**, so both sides get the same condition. Record it in the report.
- **Every arm also runs in the HIGH process priority class** (`--process-priority 2`, or start the process with
  HIGH_PRIORITY_CLASS). Same-session A/B `strata-claude-hiclass` (2026-09-29): code 76.76 -> 78.73 tok/s, thai
  48.72 -> 49.33, CPU pool 12.41 -> 11.37 ms/round, and a tighter spread; desktop programs had been taking CPU
  time from the pool. It is a measurement condition, not the serving default: at HIGH class the pool can starve
  the desktop, so the serving default stays normal until the developer decides.
- Wait for the measurement's `DONE` line before a build, a test run or another measurement; a wait loop that
  times out is not completion.
- Report whether a run was disturbed (for example, another CPU-heavy job was running) rather than silently
  keeping it. Do not run builds or other CPU-heavy work while a measurement is in flight.
- Measured numbers at high priority describe the engine, not a normal-priority server. Whether the serving
  launcher also raises priority is a separate decision for the developer, backed by the same measurements.

## Defaults: no trade-off becomes the default, a trade-off stays an option

The developer's standing rule (2026-09-28, restated 2026-09-29) decides every knob:

- **A measured win with no trade-off becomes the default at once.** It needs a same-session ABBA with identical
  outputs, and the per-stage counters must explain the change. Leave a flag to turn it off for A/B.
- **Anything with a trade-off stays an option, off by default.** Examples: it costs prefill, takes cores or VRAM
  the desktop needs, needs a privilege, or changes numerics or outputs. Record the trade-off beside the flag and in
  the issue, with the numbers on both sides, so the developer can choose.
- **Removing the trade-off is the preferred fix.** If a feature wins but carries a cost, try to remove the cost,
  and then it becomes a default under the first rule.
- The launcher and serving profile (`strata-xeno.json`) are still the developer's. Engine defaults change on the
  branch; the profile does not.

## Write it to the tracker as you go, not at the end

Context gets compacted and sessions end. Anything that exists only in the conversation is lost, so the tracker has to
hold the knowledge and the history (the developer, 2026-09-29):

- **When one piece of work is finished, update its issue straight away.** Post a comment with what was done, the
  commit, the measured numbers and the log path, and what comes next. Do this per item, not in a batch at session
  end.
- **When you find a problem, open an issue for it at once.** This covers a perf bottleneck, a bug, an instrument
  fault, or a claim that turned out wrong. Give the evidence (counter lines, trace files, commands) even if you will
  not fix it now. A finding that lives only in the chat is a finding that will be lost.
- A negative result is worth writing down too. Record a lever that was measured and did nothing, with its numbers,
  so the next session does not test it again.
- Issue bodies and comments are bilingual (English plus a full Thai mirror). `gh` is at
  `"C:\Program Files\GitHub CLI\gh.exe"`, and every command passes `--repo xenodeve/Strata-xeno` explicitly.
  Without it, `gh` resolves the upstream fork parent.

## Never put a window on the developer's display

The developer works on the display card while runs happen. Every process an agent starts runs in the background,
with no console window:

- Use the Bash tool's `run_in_background`, or a detached launcher. `tests/xeno/perf/launch_ab.py` uses
  `DETACHED_PROCESS | CREATE_NO_WINDOW`.
- Any script that starts `strata.exe` or another console program passes `CREATE_NO_WINDOW` (0x08000000).
  `ab.py` and `mem_trace.py` do. A detached parent has no console, so without the flag Windows opens a new
  window for every child.
- Do not use `start`, `cmd /c start`, a visible PowerShell `Start-Process` or anything else that opens a window.

## Keep `docs/BLUEPRINT.md` current

`docs/BLUEPRINT.md` is the system blueprint: the processes, the life of a request, the memory hierarchy, the
checkpoints, every flag / config key / environment variable, and the diagnostics (the developer, 2026-09-30). A
blueprint that describes old code sends the next session down the wrong path, so it is updated **in the same commit**
as the change, not at the end of the work:

- **Read it first** before a structural change, and check that its baseline commit is the code you are changing.
- **Update it when a change** adds or removes a process, a per-request or per-round thread, or a file with its own
  responsibility; changes the request path, the engine protocol (`GEN` keys, reply lines), prefix reuse, the prompt
  parts or the decode round; changes expert placement, tiers, profiles, swaps or the CPU pool; changes the KV cache or
  the checkpoints; adds, removes or changes the default of a flag, a JSON config key or an environment variable; adds
  a diagnostic; or merges upstream / moves the served engine to another branch.
- **Work on a branch that is not the blueprint's baseline** goes into its "Not yet in the baseline" table first, and
  moves into the sections when the baseline includes it.
- **Moving the baseline** (a merge, a new served engine) means re-checking the `file:line` references the moved code
  touches and adding a row to the revision log.
- **A commit-time check enforces the floor of this rule.** `tools/hooks/commit-msg` runs `tools/blueprint_check.py`
  on the staged diff: a new or removed engine flag, GEN key, `STRATA_*` environment variable or run-config key, or a
  new source file, with `docs/BLUEPRINT.md` not in the commit, stops the commit. Update the blueprint, or add the
  trailer `Blueprint: n/a - <why>` (a bare `n/a` is refused). Install it once per clone:
  `git config core.hooksPath tools/hooks`. It cannot see a changed request path or a moved thread; those still need
  judgment.

The full procedure is the file's own "Keeping this file current" section.

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
