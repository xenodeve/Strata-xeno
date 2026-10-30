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
