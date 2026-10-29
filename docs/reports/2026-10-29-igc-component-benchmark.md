# Typed executor versus compiled Qwenputer: first component measurements

The checked software CUDA executor is much cheaper on these three small integer
tasks than our current compiled-circuit implementation. This is a standalone
component result. No learned mapper, language model, speculative controller or
result injection participates, so it does not establish a Gyro-M speedup or IGC
end-to-end readiness. Both implementations return the exact expected answers.

## What was measured

2026-10-29, Windows WDDM, RTX 5060 Ti 16 GB primary, CUDA 13.3 / driver 621.01,
i5-13500. The optional primary-only target was built for SM120 in Release.
The same executable contains both arms. Each process requests HIGH priority;
there is no CPU expert pool in this component, so pool priority is inapplicable.
No timeline/profiler, model load, private-commit cap, serving change or concurrent
task-owned build/test was present during the measurement.

The caller supplies prepared native inputs. The Qwenputer arm receives a bound
token prefix containing the calc expression; the software arm receives typed
sign/digit operands and an operation. Tokenization, learned operand extraction,
model-generated syntax and output injection are excluded from both intervals.
The representations and execution mechanisms differ intentionally. This measures
the component cost after its inputs are available, not the unimplemented cost of
obtaining correct operands from a model.

The interval includes Qwenputer reset/prefix replay, host-to-device input transfer,
execution, chunk supervision/waits, result copies and collection. Exact output
conversion and Qwenputer token-byte decoding occur after the timer. Python
arithmetic expectations are computed before launch and checked after the native
process returns. The native harness accepts no oracle or reference trace.

For warm results, both modules are resident in one context. Each case has one
recorded warmup per arm, then two ABBA blocks (Qwenputer, exact, exact, Qwenputer),
four measured calls per arm. Every Qwenputer call resets its machine. Timings
include repeated per-call transfers and waits; they are not batched kernel rates.

| Task / exact answer | Exact CUDA ms: min / median / p90=max | Qwenputer ms: min / median / p90=max | Circuit evaluations each call |
|---|---:|---:|---:|
| `17*23 = 391` | 0.0469 / 0.06635 / 0.0886 | 5078.6697 / 5079.0054 / 5081.9643 | 1543 |
| `-123+456 = 333` | 0.0726 / 0.09265 / 0.1502 | 6016.7944 / 6018.5617 / 6020.2356 | 1828 |
| `1000-1 = 999` | 0.0475 / 0.08490 / 0.1234 | 4946.2897 / 4946.5520 / 4946.9741 | 1502 |

These are four observations per cell; p90 is the nearest-rank sample maximum,
not a production tail-latency estimate. The software arm executes a single bounded
exact kernel per call, not zero computation. All cold, warmup and measured outputs
pass independent exact checks. Previously completed held-out correctness suites
remain separate; this three-task latency corpus does not replace them.

## Same-call stage counters

For `17*23`, Qwenputer medians are prepare 0.06825 ms, issue 0.80150 ms and
collect/wait 5078.08635 ms, across 13 chunks per invocation. The software medians
are prepare/H2D 0.00715 ms, issue 0.05615 ms and final stream wait 0.00130 ms.
The issue bucket includes the pageable result-copy call, which may itself wait;
it is **not** a measurement of pure launch overhead or GPU arithmetic time.
Qwenputer's collect bucket includes its event wait, control/output copies and
collection. These are same-call host counters, not profiler subtraction. There
are no per-kernel GPU-duration claims here.

## Fresh-process calls and memory

Fresh-process ABBA on `17*23` (two observations per arm) gives:

| Measurement | Exact software | Qwenputer |
|---|---:|---:|
| Full process launch-to-exit ms | 118.240–122.575 | 5466.006–5492.919 |
| Setup to ready, excluding process launch/exit, ms | 78.581–81.311 | 355.178–380.046 |
| First call after setup, ms | 0.8285–0.8685 | 5068.7938–5069.0310 |
| Process private-commit lifetime peak, bytes | 247,271,424–247,836,672 | 311,590,912–312,528,896 |
| Explicit component device buffers, bytes | 248 | 31,804,988 |
| Context-ready to module-ready CUDA free decrease, bytes | 2,097,152 | 35,651,584 |

Fresh process means a new context, not a flushed disk/JIT/driver cache. Process
launch-to-exit includes cleanup and diagnostic output. Device buffer counts omit
driver/context/code and allocator granularity; CUDA free changes are sampled
global observations, not an attribution or continuous transient-peak measurement.
The shared warm process peaked at 310,677,504 bytes of private commit. No full
model RAM/VRAM, expert-cache impact or integrated resource-fit conclusion follows.

## Reproduction and identities

Build the optional `STRATA_BUILD_COMPONENT_BENCHMARK=ON` target
`igc_component_benchmark` under `tests/qwenputer/native`. This isolated target is
SM120; full D2x engine builds still require `89;120`.

Prepare with `python -B -m tools.igc.component_benchmark prepare --output <new-dir>
--tokenizer <existing-tokenizer>`. Launch `tools.igc.run_component_benchmark`
through the existing bounded `tools/qwenputer/headless.py` wrapper with the
primary UUID, executable, bound artifact and binding. It serializes fresh-process
ABBA then warm paired ABBA, freezes inputs/executable and checks them again after
completion. The outer job only owns these test children and has no memory cap.

| Input / output | SHA256 |
|---|---|
| Executable | `f1b7a837e265e920f6dc411663acb0c9b99cccaca27867f995d4d65ae929cf88` |
| Circuit artifact | `bc0db822ca05ee50a92b72adaeaf4a8c61d4a44b78846a091a2045c2b7b75152` |
| Binding | `291b2bb5e51e8d91e3d9826d414231d26e0eb536dbdfe8ca3820aeb16c13c6eb` |
| Corpus binary | `5e010a855a9d6f30c2dbd5d28fbf1555bb8392950d38e85b06cdc9b27ae70bda` |
| Corpus/oracle manifest | `e2374eb0ac468afdb75b94a9b62a2ceb3abe7e0915d3d55a27ee486052f8e107` |
| Warm raw JSONL | `f19196b5db4c7ab5157bf0f79145b131d4a7b17cee22478a1c75d500e0d326f2` |

Local evidence is in `igc-component-bench-02/`: `identities.json`, `preflight.json`,
`runs.json`, each process's JSONL/stderr, `summary.json`, and `complete.json`.
The bounded session log ends `DONE exit=0`. All frozen identities stayed unchanged.
Five analyzer tests pass, including wrong-answer/status, missing whole ABBA block,
duplicate/order, invalid timing and envelope negative controls.

The first attempt stopped in the **offline checker**: the firmware's valid
`<tool_response>391</tool_response><think></think>` postlude was stricter than its
initial parser expected. Native execution succeeded. The raw attempt is retained
in `igc-component-bench-01/`; a RED test reproduces it, the parser now permits only
the complete empty-thinking envelope and rejects extra text/duplicate answers.
The completed session was rerun with unchanged native executable/artifact/corpus.

## Next gates

I2 needs a frozen real-Gyro activation dataset, training-only fit and held-out
readiness/operator/operand checks. I3 needs a conditional output mapper and real
result-use validation. I0 currently covers a late final-head STE seam only:
dequantized head transpose backward, not gradients through earlier frozen Gyro
layers. These requirements determine the end-to-end benchmark schedule; this
component measurement supplies no credible completion date for them. Ordinary
text parity, repeated committed triggers, MTP/ownership, actual Hub workload,
cold/warm/tail/stability and backend coverage remain integration/rollout gates.
No default is enabled by this result. WHILE-path sanitizer debt is unchanged.
