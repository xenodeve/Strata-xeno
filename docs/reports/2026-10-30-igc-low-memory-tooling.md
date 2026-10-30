# Low-memory coherent evidence tools

Issue #276. Live Gyro-M observation remains resource-blocked while another task
uses the machine. This implements offline dependencies from the
[coherent development protocol](../plans/2026-10-30-igc-coherent-development-protocol.md).
It does not collect activations, train, execute GPU work, open sealed acceptance,
change learned policy or alter Hub configuration. The full IGC goals remain open.

## Whole-archive audit

`tools/igc/coherent_audit.py` checks complete v3 archives one row at a time using
the existing strict reader. It additionally hashes the entire embedding table
with bounded streaming, checks every dictionary row (including unreferenced
entries), validates row version/source-capture identity, compact head statistics
and the four retained full-logit references, and refuses partial markers or
unexpected files. Receipt metadata binds the report to plan, model, tokenizer,
sources, completion hash, row owner and role. It retains no collection of
activation tensors.

The normal command requires actual-Verifier receipts and the existing first-four
head-fidelity gate. Synthetic audit is an explicit diagnostic option and remains
marked `native_qualified: false`. Archive consistency does not establish that a
model used a result, learned generalization, hard-off parity or serving latency.

```powershell
python -B -m tools.igc.coherent_audit --archive <complete-compact-archive> --output <new-audit.json>
```

Use `--allow-synthetic` only for explicitly identified component fixtures.
The tool never resumes, repairs or rewrites its input. Existing output reports
are refused. No inference or training code is imported.

## Verification

The test seam is the offline archive-consumer interface already required by the
approved protocol. Red/green evidence includes an unreferenced dictionary row
whose outer hashes were recomputed while its individual digest was wrong. The
first implementation accepted that row; the added full-table row check rejects
it. Independent review identified unchecked row version/source-capture fields;
resealed-negative tests reproduced both before the audit fix. Further tests
reseal invalid owners/head statistics, add a partial marker, alter the global
table digest and confirm synthetic receipts cannot qualify as actual native
observations.

The final targeted CPU suite passes **24/24** tests, including the review regression.
That process runs serially for 2.235 seconds including startup, with measured
lifetime peaks of **694,083,584 bytes private commitment** and **64,819,200 bytes working
set**. Counts/resources are recorded in the checkpoint receipt. These are
owned diagnostic observations, not a serving benchmark. No hard memory cap was
imposed. NumPy imports account for much of the process commitment.

The auditor also reads the five-row compact archive saved by the earlier CUDA
synthetic lifecycle test: four anchors and one output row. It correctly reports
`native_qualified: false`. This CPU-only read takes 0.391 seconds including module
initialization, with peaks of 674,385,920 bytes private commitment and 44,851,200
bytes working set. It launches no GPU process or language model.

Local evidence basenames are `igc-coherent-audit-red01.log`,
`igc-coherent-audit-red02.log`, `igc-coherent-audit-red04.log`,
`igc-coherent-audit-final05.log` and its resource JSON, plus
`igc-coherent-existing-archive-audit-01.json` and
`igc-coherent-existing-archive-audit-memory-01.json`.

No backend build is required for these Python-only tools. Existing CUDA/HIP/SYCL
code and diagnostic producer are unchanged. Actual full-model archive integrity,
live cache lineage, disabled parity, memory peaks and learned quality/latency
remain unverified. The prepared collection command and its pinned producer
sources remain unchanged.

## Row-bound development scorecards

`tools/igc/coherent_evaluate.py` consumes a complete audited archive, the frozen
development manifest and a streamed replay ledger. The ledger header binds the
plan/completion, model/tokenizer, canonical manifest digest and controller digest;
the last must equal the bundle identity in archive provenance. Every original
anchor row must match its ordinal, owner and receipt hash. Existing native ABI
checks verify typed request, result, status and output finiteness against the
selected operands. Missing, extra, reordered, stale or teacher-forced input rows
fail. Files are bounded and rechecked before an exclusive report is written
outside the archive. Inputs remain unchanged.

The report separates readiness, raw per-head tuples, agreement and encoding
validity from guarded selection and native-result correctness. Each readout
reports sign/length/digit/operand exactness and padding-only, active-only or joint
errors, including digit confusion by slot, true/predicted class, width and padding.
Readiness false positives, negative operator no-call counts and actual negative
calls stay distinct.
Valid arithmetic on wrong operands is still a task error.

Language, intent, operand width/sign and operation cells include matched quoted
and label negatives. Ordinary/outside-current-mapper cases remain explicitly
unclassified on expression axes. Complete-condition rates resample structural
clusters with fixed seed 20261101 and 512 bootstrap draws, rather than treating
tokens/digits/prompts as independent. The interval is descriptive development
evidence; shared values can correlate errors across those clusters.

Predicted-length padding composition is a separately named offline representation
counterfactual. It keeps readiness and all active predictions unchanged. It never
repairs raw scores, changes historical results or deploys a different gate.
Teacher-forced hard-off output target accuracy/NLL is reported separately from
input/readiness results. Learned and oracle-owned output controls are explicitly
`not supplied`; free generation, repeated calls and final-answer quality remain
unverified. Even a passing observed input score cannot qualify live/acceptance
or close the full protocol.

```powershell
python -B -m tools.igc.coherent_evaluate --archive <complete-compact-archive> --manifest <frozen-manifest.json> --replay <native-replay.jsonl> --output <new-scorecard.json>
```

The bounded JSONL replay format is version 1, kind `coherent-native-replay`.
Its header contains `plan_sha256`, `complete_sha256`, `model_sha256`,
`tokenizer_sha256`, `controller_sha256` and `manifest_sha256` (the canonical
`_json_bytes` representation of the frozen manifest). Each following row contains
`ordinal`, `owner`, `receipt_sha256`, both nine-field categorical predictions as
`primary`/`secondary`, two `readiness_logits`, and the complete existing native
pipeline record in lowercase `record_hex`. Only original anchors enter the input
scorecard. A future producer of this ledger must export actual native replay
outputs; the consumer does not fabricate or run them. Actual coherent collection
and native replay-ledger export are still prerequisites for real new-cohort scores.

Independent review reproduced and fixed three metric/provenance faults before
commit: an unchecked controller binding, readiness masking raw disagreements,
and missing expression cells for matched negatives. Negative tests fail on the
previous implementation and pass after the fixes. Synthetic examples verify
that 17*23 gives the literal native receipt 391, whereas valid execution of wrong
arguments 18*23 gives 414 and is scored wrong. These are test fixtures, not model
or GPU results. A padding-only disagreement is recoverable in the counterfactual;
an active digit error remains wrong. No model prediction has been repaired.

Final CPU verification passes **38/38** tests: fourteen scorecard tests, five audit
tests and nineteen existing observation/qualification regressions. It runs one
process for 7.031 seconds including startup, with lifetime peaks of **694,595,584
bytes private commitment** and **65,355,776 bytes working set**, no hard memory
cap. Evidence: `igc-coherent-evaluate-red01.log`,
`igc-coherent-evaluate-review-red05.log`, `igc-coherent-evaluate-review-green05.log`
and `igc-coherent-low-memory-final02.log`/`.json`. These measurements concern
the CPU diagnostic tests only. No GPU, fit, collection, model, acceptance, system
or serving change is part of either checkpoint.
