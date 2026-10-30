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
