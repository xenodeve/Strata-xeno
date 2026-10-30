# Joint native observation infrastructure, live qualification still open

Issue #276. The default-off diagnostic now pairs genuine early features with the
same request's unadapted final state. This implements the next collection
dependency from the [cached gate diagnosis](2026-10-30-igc-controller-gating-diagnosis.md).
It does not improve learned accuracy or close I2/I3. No full language model,
new real-model activations, training, sealed acceptance or served configuration
change is part of this checkpoint.

## Implemented path

`--igc-observation-plan` and `--igc-observation-root` exist only in the existing
CUDA diagnostic build. They require the bound learned bundle, native single
serve path and exactly one generated token per diagnostic request. The normal
build option remains off. Existing model/tokenizer content binding runs before
model allocation; the plan must match those bindings. Controlled oracle fixtures
and observation mode cannot be combined.

The observer reuses `IgcLearnedDiagnostic`, its `ContextWindow` and the existing
two-lane ordered prefill snapshot/accepted-input callbacks. It does not add a
second competing history or ask the engine to force cold/serial prefill. The
observer head records a stable D2D prehook snapshot during graph capture and
prompt-head activity, preserving the model input. No learned pipeline, request
or result injection is initialized. Total observer storage is exactly
**14,113,680 device bytes**, including context, snapshot and private head-replay
scratch. The synthetic lifecycle test checks this number, not whole-model memory.

Before the first decoder window, `stage()` gathers the complete owned pre-anchor
window. Those output planes survive the accepted anchor overwriting its ring
slot. Missing cached features fail the diagnostic, rather than fabricating rows
from KV/final state or forcing a different workload. At request end, the engine
waits for commit and export synchronizes the context stream. Only a completed,
uncancelled one-token request may publish. Actual `Verifier::copy_head_input(0)`
and `copy_logits(0)` supply unadapted final data. Mixed bytes must match the fenced
prehook snapshot. On the first four actual Verifier records, a separate Q8
projection using the already loaded immutable head weights must reproduce every
logit bitwise. It uses private activation scratch without duplicating head weights
or modifying decoder buffers.

The v2 producer plan contains only model/tokenizer digests, bounded consumed
prefixes and anchor/output base/step identity. It contains no current target or
oracle operands. Original anchors precede teacher-forced output requests. The
output prefix contains only the already consumed completion tokens. The archive
keeps capture-request owner and original anchor owner separate.

## Archive and bounded collector

The native writer stages one self-digested **v3** record below 8 MiB. Geometry, token
prefix, plan/model/tokenizer identity, canonical mask, finite tensors and
monotonic request ownership are checked. Reused/accepted input counts, previous
owner, cache decision, producer-stream mask, completed logical fence,
completion/snapshot flags and actual-Verifier versus synthetic origin are
persisted and validated with each compact row. Tap/layer/F32 geometry is fixed by
v3. A `.pending` file becomes `.bin` only after writing succeeds.
Existing/partial output directories are refused; **v3
does not implement resume across processes**. Cancellation, failed commit and
missing history cannot create a successful row. Old evidence is never overwritten.

The Python compactor streams one row at a time. It stores attention tails and
mixed vectors, deduplicates native embedding rows by token only after exact byte
checks, retains four full reference-logit vectors and records compact head/target
statistics elsewhere. Targets enter this offline step after request completion,
not the producer plan or learned feature input. Each committed row has byte/hash
receipts; `COMPLETE.json` is written last. The reader rejects partial archives,
wrong roles/bindings, altered receipts/arrays, invalid indices and nonfinite data.
`load_input_observation` refuses output roles and removes target/head receipts.
The old v1 observation reader remains strict and unchanged.

`coherent_collect.py` uses the actual Hub helper and selected profile, preserving
KV/cache, D2x/MTP/Vision and priorities. It changes only task-owned executable,
log and expert-profile output locations plus observation flags. It imports the
compaction runtime before admission, and checks again after helper/tokenizer
initialization. There is no private/job commit cap. External time/system-safety
limits remain disclosed diagnostics, outside any serving benchmark claim.

The actual launcher selects server hash
`10be781197be5ffe7e41e9307c53d7444631f697cc1c81d456a2b75ee950ccbc`;
the other checkout's similarly named server is different. The collector refuses
a different server or selected profile hash
`06ef847e063bb9643c7bf3f486e2ffdadc29737493268238c21f59188d5f6c5b`.
It preserves pool/process priorities 2/2. `coherent_sources.py` pins current
auxiliary files, pack/MTP/Vision inputs and candidate DLLs; the native producer
separately binds the actual GGUF sources. Historical fallback files cannot stand
in for current live inputs. These receipts cover project inputs, not every OS or
CUDA driver library.

An explicit checkpoint validates all four actual native/head-fidelity receipts
and reference files before requesting anchor five. Synthetic receipts cannot
qualify. The first four fixed anchors and first four fixed output cases price the
remaining work against the unchanged 1800-second capture/2100-second outer
bounds, including staging reads, compaction, hashing, fsync, journal and cleanup.
Telemetry exceptions, unexpected monitor exit or failure to drain cancel the run
and prevent a successful report. Reports follow owned cleanup and final job
telemetry. An incomplete or over-budget run stops. Only just-consumed task-owned
staging files are removed after their committed compact receipt; source evidence
digests remain. The 400 original anchor receipts freeze before output collection.

## Manifest preparation completed, no activations collected

The seven inspected archives are hash allowlisted and read directly. No corpus
generator is called and no sealed acceptance module is imported. The actual tokenizer
and historical canonical wrapper produce:

- 400 anchors: 192 eligible, 208 no-call/ordinary/outside-current-mapper.
- 48 output samples with **570** additional output positions: **970 requests**.
- Maximum consumed prefix 85 tokens; 44 structural pair/fact clusters.
- 512,364 bytes of local plan/manifest/wrapper metadata.
- Plan SHA256 `37678b6c11475de76b2d451ac9cbee9e3d36b5897026dba0d2aed0a018389ff9`.
- Manifest SHA256 `0d5e3079b4453fcc6f4aad753482fa1522973f03080345cc8b0a27c262c4a586`.

This is a diagnostic arithmetic cohort, not representative long-prefix Hub
traffic, unseen scalar values, a fit or acceptance. Sealed overlap is unknown.
The prior protocol's 1488 extra-output upper bound and 5.5-GiB storage requirement
remain conservative; no admission reserve is reduced from these counts.

## Verification and limits

The full isolated diagnostic engine builds with **sm89;sm120**. Synthetic host
and CUDA checks cover plan digest/order, graph reuse created before arming,
pre-gather followed by anchor ring overwrite at positions 79/80/81/137, output
ownership, missing history, cancellation, incomplete commit, existing/partial
archive refusal and release/reprepare. Existing context and two-stream prefill
snapshot regressions pass. Normal and memcheck raw captures are byte-identical
on all five synthetic rows; compact roundtrip reproduces their values and owners.
Final v3 memcheck reports zero errors. A synthetic Q8 head with exactly
representable scale 1 / input 127 verifies projection replay and rejects an incorrect
reference. Normal/memcheck outputs match bitwise. This covers existing projection
kernels and copy/context/archive behavior, not full-model head fidelity or global
race determinism.

Python tests exercise target exclusion, ordinary-negative archive parsing,
resource classification, telemetry-failure/early-exit cancellation,
malformed/nonfinite/binding refusal, partial completion
and compact ownership. The final test/readback receipts name exact counts and
source hashes. CLI admission checks do not load a model; the two optional
model-metadata admission cases remain skipped when no model input is supplied.
Final actual-Hub-Python verification is **19/19** unit tests, **8/8** serialized
host/native/memcheck checks, and CLI **3 passed /2 skipped**. Both review axes
close with no remaining material finding. The actual Hub Python reproduces the
prepared plan/manifest byte-for-byte against the GPU-Python preparation.

Final engine SHA256 is
`9f055a2954bebf021560c4aa6d03f97aa1a51a99c4323c570cf08cccb1e8e240`;
synthetic lifecycle executable SHA256 is
`dd02d7539785274b69171009d0e709db62806aa312fe7294618b38efecc11017`.
Auxiliary source pinning completes for 39 files / 10,314,895,495 bytes using bounded
streaming, without inference. Exact source paths remain private.

Development failures are retained: missing-module/header red tests; malformed
hash whitespace accepted by `bytes.fromhex`, now rejected by strict hex syntax;
64 legitimate ordinary negatives without operand fields, now contributing prompt
exclusions without invented operands; the first CLI red attempt lacking CUDA DLL
paths; a sandbox temporary-directory test failure; and pre-run source-map/cold
runtime-import problems caught in review. The GPU-training Python lacks psutil;
final runtime checks use the actual Hub Python 3.11 and its existing NumPy/psutil,
without an installation. None of the failed attempts is relabeled as passing.

**Not verified:** actual `finish_observation` readback from full-model Verifier
buffers, live cached-prefix/checkpoint behavior, whole-model disabled greedy
parity, full-model resource peaks, learned controller quality/output use or Hub
latency. The CUDA test uses a synthetic final-copy callback, not the real Verifier.
Missing checkpoint/session feature history still refuses observation. HIP/SYCL
builds were not run; the diagnostic target explicitly requires CUDA. No change
is promoted or enabled in the served Hub.

## Ready bounded run, pending fresh admission and review

Use the [frozen protocol](../plans/2026-10-30-igc-coherent-development-protocol.md).
Prepare auxiliary source receipts with `python -m tools.igc.coherent_sources`,
then launch `python -m tools.igc.coherent_collect` through the owned headless
wrapper with timeout 2100 seconds, actual profile/server, candidate engine, separate
replay bundle, prepared metadata, pack and those receipts. Exact local command
paths and hashes stay in the private execution proposal/receipt.

Admission remains actual Hub physical 24 GiB, disclosed commit-headroom estimate
51,128,263,976 bytes, workspace free 5.5 GiB and no active engine. Preserve
`CUDA_VISIBLE_DEVICES=1,0` with `CUDA_DEVICE_ORDER` unset. No model run occurs
while those requirements fail. Do not change pagefile, Hub settings, weights or
thresholds to pass this gate. After live producer fidelity passes, replay frozen
heads and strict per-role/selection gates before any later training or I4/I5 work.

At 2026-10-30 02:55:37 UTC, after importing the actual compaction runtime, physical
availability is 17.5316 GiB and system commit headroom is 9.6448 GiB. Shortfalls are
6.4684 GiB and 37.9721 GiB respectively. Workspace free space is 9.8290 GiB, which satisfies storage.
The owned job has flags 8192 (kill-on-close), no private/job memory cap, and no
active engine is found. No collection/model launch is attempted. Evidence:
`igc-coherent-final-python-03.log`, `igc-coherent-native-checks-04.json`,
`igc-coherent-final-cli-02.log`, `igc-coherent-preparation-hubpy-02.json`,
`igc-coherent-current-sources-01.json`, and `igc-coherent-resource-final-01.json`.
