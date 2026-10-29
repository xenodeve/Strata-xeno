# I2 one proposed warm-start diagnostic

Issue #276. **Not executed; awaiting parent review.** This proposal follows the
[operand-generalization diagnosis](../reports/2026-10-29-igc-generalization-diagnosis.md).
It is one fixed experiment, not a hyperparameter search or permission to retrain.
The retained live candidate and failed fresh artifact remain immutable.

## Question and single changed factor

Can adapting a COPY of the retained broad-corpus head learn matched quote-copy
intent without destroying its operand extraction? Relative to the failed fresh
fit, change only initial learned weights; use a new AdamW optimizer, no old
moments, with the same architecture, data, seed/order, loss and complete recipe.

This tests value of the existing learned prior, not whether data size alone
caused the failed fit. Historical exposure is explicit: the retained head has
seen 19/32 matched development signed pairs in its old training corpus. Report
those pairs separately from the other 13. No passing result here closes I2 or
qualifies as untouched acceptance.

## Inputs and required implementation before execution

- Initialization is the exact retained contextual first-attention `InputReadout`,
  96,964 parameters, SHA256
  `519fbc196993c94676f4031f7bd45ad9abf661f8b7b4fd3b9d6358204b5294e2`.
- Reuse qualified matched cache states
  `6757cde83254fff23010d26d720bc579880e2854750a4ee163b5224b8e12d82a`
  and mask `d96bed0c391bb9bdfd544c23a9615823229920ecc714903087d32add325f74a0`.
  Train only the same 768 train rows; no new activations or data generation.
- Before a reviewed run, add an explicit offline initialization option with
  tests for strict tensor names/shapes/F32/finiteness and expected source digest.
  Bind initialization to pricing/fit provenance. Default fresh initialization
  must remain unchanged; warm-start must not reuse the fresh pricing receipt.
- Extend source/schema/protocol identity receipts for that implementation,
  then freeze them before pricing or fitting. No source mutation between price
  and fit, no on-the-fly tolerance or identity changes.

No implementation of that new option is included in this diagnosis checkpoint.

## Fixed future recipe and bounds

If reviewed: seed 20261031, batch64, 70 epochs, 840 AdamW updates, lr0.001,
weight_decay0.01, cosine to0.00001. Use F32, TF32 off, deterministic algorithms,
math attention, four CPU threads, original qualified CUDA Python and primary GPU.
The exact shuffle must reproduce schedule digest
`fc2d71800fca6d5b62d1cc3c1d2869cad64cc1138ecf10dc22d4741e64085f25`.
Loss and finite-update guards stay unchanged. No clipping, threshold adjustment,
best-development checkpoint or retry; evaluate/save final checkpoint only.

Price exactly one training-only backward batch without an optimizer step. Then
restart seed/model/optimizer in a separate fit process, verifying loaded initial
weights and receipt. Fit bound1800s / owned outer bound1920s. This is a single
new future experiment; an unfavorable result stops it, not a reason to sweep.

Fresh resource checks: physical8GiB, system commit headroom12GiB, primary GPU2GiB,
output allowance32MiB plus4GiB disk reserve. These are disclosed offline-task
admission checks, not Hub behavior or a process memory cap. The earlier fresh
fit observed CUDA allocated peak729MB and process lifetime private peak4.27GB;
they are observations, not guaranteed maxima. Use bounded cache I/O and one heavy
job. Keep timeout/owned cleanup; no system, pagefile, served-profile or unrelated
process changes.

## Frozen evaluation and gates

1. Re-evaluate retained and candidate using the same F32/eval/no-grad protocol,
   batch64 and exact ordered matched rows, after the final update. Verify reload
   categorical parity and finite tensors; preserve both raw arrays.
2. Matched development gate is unchanged: **384/384 raw exact compute tuples and
   0/384 quote-copy false-call candidates**. Train/dev/per-field, sign, length,
   language/template and signed-pair cluster metrics remain separate. Report
   development's old-training-exposed19 versus unexposed13 pairs separately.
3. Add the already identified **343 old-development rows** whose signed pair is
   absent from ALL96 matched train/development pairs. This is inspected legacy
   regression, not new acceptance: 259 compute /84 negative rows, drawn from
   the existing old cache. Bind its ordered IDs and source digest before run;
   no old train rows are included. Recompute retained baseline in the same
   evaluation session. Require **no previously correct raw compute tuple becomes
   wrong and no new false call on a previously correct negative**. Aggregate
   gains do not hide per-row regression. Record existing failures too.
4. The predicted-length padding variant may be reported under its existing
   separate name, but never replaces raw scoring or supplies oracle lengths.
5. No acceptance64 read, new full model launch, deployment, live gating changes,
   production-artifact replacement or benchmark claim. Passing development only
   permits consideration of the next independent readiness gate.

The 343 regression IDs are frozen in local
`igc-generalization-auxiliary-01.json`; bind that file and ID-list digest in the
future reviewed receipt. Its current file SHA256 is
`87189b7633a05d1503748fba8aec2baede2980882224db50c5cfe79f053ae8b1`;
the ordered compact-JSON ID list is
`5e63f56e919c677f4dd88a56ceec1df80df90772f1e0e35ae386bfafaa8a064b`.
Old cache identity is the qualified expanded-v2 cache,
not an arbitrary path chosen at run time. Budget approximately18MiB for extra
small predictions/reports and weights, covered by the32MiB output allowance;
no new multi-gigabyte cache is needed.

## Decision boundary

Review this one proposal before implementation/pricing/training. If the run
later fails either gate, retain the final failure, report it and return for the
next evidence-based decision. Do not automatically enlarge data, switch heads,
rerun seeds, relax padding or tune on inspected development.
