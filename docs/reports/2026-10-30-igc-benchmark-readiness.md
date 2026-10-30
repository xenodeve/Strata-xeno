# Low-memory benchmark-readiness work

Issue #276. This follows the coherent audit/scorecard checkpoints and addresses
existing measurement infrastructure needed by I5. No Gyro-M launch, fresh
observation, GPU work, training, acceptance access, admission polling or serving
configuration change is part of this work.

## Existing dependencies and remaining work

| Dependency | Existing evidence / implementation | What is still needed |
| --- | --- | --- |
| Coherent archive integrity and row-aligned metrics | `coherent_audit.py` / `coherent_evaluate.py`, checkpoints `8ef24d5b` / `d5cf3ae8` | Actual model observations and native replay-ledger output; synthetic scorecards do not close I2/I3 |
| Memory accounting for benchmark resource gates | `tests/xeno/perf/ab.py`, also used by `mem_trace.py` | Correct the discovered system-commit semantics and unavailable-sample reporting, covered below |
| Paired per-stage engine measurements | Existing ABBA runner, raw-token parser and `tests/xeno/latency_breakdown.py` | Live eligible arms on the selected Hub workload; do not create a second generic runner or reuse old Swift reference prompts as the new baseline |
| Transaction/cancellation | `igc_pipeline.hpp` trial APIs, `pipeline_trial_test.cpp` abort/replay/commit and reset tests; learned wrapper cancellation/context/ownership support | Real Verifier/accepted-prefix/cache/head behavior, then repeated committed triggers; more CPU-only fake callbacks would not establish those gates |
| End-to-end I5 and capabilities | Primary PRD I2/I3 then I4/I5, followed by C1-C4/A1 | Learned use, repeated calls and quality/latency require the real native/model paths; do not advance a new sandbox/interpreter ahead of those dependencies |

## Reproduced instrumentation defect

The old `ab.system_memory()` returns physical in-use, physical available and an
alleged system commitment. Its third value is `ullTotalPageFile - ullAvailPageFile`
from `GlobalMemoryStatusEx`. Microsoft documents those values as potentially
process-limited and directs callers to `GetPerformanceInfo` for system commitment.
`CommitTotal` is in pages and `PageSize` supplies the conversion. See
[MEMORYSTATUSEX](https://learn.microsoft.com/en-us/windows/win32/api/sysinfoapi/ns-sysinfoapi-memorystatusex)
and [PERFORMANCE_INFORMATION](https://learn.microsoft.com/en-us/windows/win32/api/psapi/ns-psapi-performance_information).

This is the same semantic class corrected earlier in the P0 task harness, but
the shared legacy benchmark helper still contained it. The caller survey finds
both `ab.py` and `mem_trace.py`; no other caller appears in the repository search.
It affects a diagnostic field, not model inference, arithmetic or the engine's
per-stage latency counters. Preserve historical raw logs. Their label alone is
not authoritative system-commit evidence, even when a value happens to coincide.

### Debug-mantra ledger

1. A fake-API test reproduces the defect without querying the running machine.
   Set system commitment to 50 GiB, physical total/available to 48/20 GiB and
   process allowance total/available to 42/2 GiB. Baseline reports `(28,20,40)`;
   the correct system tuple is `(28,20,50)`.
2. A `bdb` line trace of the pinned `d5cf3ae8` helper follows the GlobalMemoryStatusEx
   path and records its two allowance fields before subtraction. No real Win32
   memory query runs. Source tracing finds no flag controlling this computation.
3. The disproof control changes allowance values to 82/32 GiB while keeping the
   same system commitment. The old number then coincides at 50 GiB, showing why
   numerical agreement in a single run cannot validate the old semantics. Unit
   conversion and process-private confusion do not explain the differential.
4. Red01, the pinned debugger trace and green01 preserve that chain. Further
   failure tests show sampling errors must not become a valid all-zero peak.
   One test-harness failure dropped thread callback arguments; it was fixed in
   the fake Thread. Another integration test exposed a summary log still open
   after `DONE`; the runner now closes it explicitly.

The correction reuses the existing checked `tools.qwenputer.memory.system_snapshot`
implementation. The three-value GiB contract, successful peak text and trace TSV
columns remain unchanged. Failed/empty sampling prints `memory unavailable`, with
no numerical peak for parsers to mistake as valid zero. The shared sampler also
distinguishes an exited process from unavailable memory on a still-running one.
`mem_trace` keeps stdout, stderr events and any prior samples, writes an error
receipt, then fails instead of presenting incomplete memory data as successful.

No engine command/flag, priority, sampling interval, CUDA mapping, ABBA order or
serving default changes. Tests mock launchers; no benchmark engine is executed.

## Verification

The final selected suite passes **9 tests plus 2 subcases**: seven new memory/
observer/wiring cases and the two existing real-run stage-parser fixtures. The
fake Win32 boundary works without the Windows-host guard; the non-Windows branch
is simulated, not an actual Linux/macOS runtime validation. Independent review
found this portability issue before commit; it is fixed without skipping tests.

The test process takes 0.797 seconds including pytest initialization, with lifetime
peaks of **29,945,856 bytes private commit / 39,976,960 bytes working set**. These are
owned-process measurements, not system admission checks or serving performance.
There is no hard memory cap. No NumPy/model/training stack is needed.

Evidence basenames: `igc-perf-memory-red01.log`,
`igc-perf-memory-debug-trace-01.json`, `igc-perf-memory-red02.log`,
`igc-perf-memory-red03.log`, `igc-perf-memory-integration05.log`,
`igc-perf-memory-final08.log` and `igc-perf-memory-final08.json`.
All engine/backend code is unchanged; no CUDA/HIP/SYCL build is required.

## Session completeness and parity evidence

A second small fixture reproduces a false reporting gate: the old summary says
`outputs identical across arms: True` when only one recorded arm exists. It
compares the set of available token strings without knowing which runs were
scheduled or whether they exited successfully. This is not an inference bug, but
cannot serve as the same-session cross-arm correctness gate required by AGENTS.

The existing runner now validates arm/prompt names and schedule before filesystem
side effects, requires a fresh output directory and freezes full executable
hashes, prompts and order in `session.json`. Per-run receipts bind prompt, ordinal,
arm, executable, exit status, stdout/stderr hashes and memory completeness. The
summary checks the entire multi-prompt schedule, not just one prompt in isolation,
and verifies snapshots and raw-log identity. Missing/extra/failed/mismatched runs,
replaced logs, empty token output and unsupported receipts cannot qualify parity.
At least two planned arms and identical raw token output are required. A valid
single-arm measurement may be complete but cannot qualify cross-arm parity.

Recorded output agreement, whole-session completeness and memory completeness
are separate. Memory may be unavailable while token parity is established; that
does not satisfy the resource gates. Divergent successful runs remain complete
but fail parity. Legacy directories without session receipts remain readable as
unverified completeness; their raw counters and outputs are preserved. Fresh
measurement mode refuses nonempty directories instead of mixing or overwriting
old evidence. A summary is not a substitute for independent task quality, an
appropriate baseline, complete profiler-off ABBA counters or actual Hub traffic.

Independent review reproduced three additional gaps before commit: arm-name
validation after copying could escape the output directory; run status did not
bind log content; and checking one prompt could qualify while another scheduled
prompt was missing/failed. New red fixtures reproduce all three; the final fixes
validate before copies, bind both logs and require whole-schedule completeness.
Reviews have no remaining material findings. The executable snapshot/run receipt
is evidence from this trusted local harness, not cryptographic proof against a
party able to rewrite all local evidence and manifests.

The final CPU suite passes **17 tests and 10 subcases**: eight session tests,
seven memory tests and two existing stage-parser fixtures. One process takes
1.047 seconds including pytest initialization, with lifetime peaks of
**32,362,496 bytes private commitment / 41,353,216 bytes working set**. Fake APIs,
fake launchers and small files are used; no model, GPU, training, admission check
or actual benchmark runs. Evidence: `igc-perf-session-red01.log`,
`igc-perf-session-red02.log`, `igc-perf-session-review-red04.log`,
`igc-perf-session-review-green04.log`, `igc-perf-session-final05.log`/`.json`.

Engine commands/priorities, sampling interval, CUDA placement and default arm
order remain unchanged. Snapshot/log verification adds diagnostic file I/O after
the engine finishes. The legacy script's `wall` field includes postprocessing;
do not compare that field to historical wall values as a claim about engine
latency. Engine stage counters remain the paired source for those claims; Hub
end-to-end latency needs its own request measurements. No speedup is claimed.

## Critical-path backlog after these corrections

The inspected CPU-only measurement gaps are now addressed. The repo already has
ABBA orchestration, per-stage parsing and the pipeline timeline, so a second
generic benchmark runner would duplicate them. I5 still needs validated I3/I4
arms, a frozen supported workload/rubric and actual Hub sessions. Preparing a
benchmark invocation around the current unqualified learned route would not
establish usability or meet the user's requested sequence.

The next decisive I2/I3 checks need coherent observations from the real Verifier
and native head/cache behavior, then actual native readout/controller and output
control replay. I4 needs accepted-prefix/MTP/repeated-trigger behavior on the real
decoder. Existing host metadata and cancellation/trial interfaces are present;
their unresolved claims concern those GPU/model states, which new CPU fake
callbacks cannot validate. Further optimization needs the timeline and actual
quality/per-stage evidence. The remaining capability milestones depend on that
foundation, so this work does not invent another interpreter, change the frozen
corpora or claim PRD completion while live execution is unavailable.
