# I2 approved single warm-start experiment

Issue #276. Parent reviewed local commit `48d0c085` and authorized the
[single proposal](2026-10-29-igc-warm-start-diagnostic-proposal.md).
This execution record supersedes that proposal's pending-review status, while
retaining its frozen recipe and gates. One fit only; no automatic promotion,
extra seeds, sweeps, checkpoint selection or retries on poor quality.

## Initialization and controls

Train a COPY of the retained 96,964-parameter contextual first-attention head,
SHA256 `519fbc196993c94676f4031f7bd45ad9abf661f8b7b4fd3b9d6358204b5294e2`.
The offline fitter's explicit `--initial-state` and `--initial-state-sha256`
must be paired. `initial_state.py` accepts only a bounded regular safetensors
file with the exact names, shapes, F32 dtype and finite values expected by the
target head. Validate every tensor before copying; never assign source storage
or deserialize optimizer state. Recheck the source artifact digest at the end.
The optimizer is always a newly constructed AdamW with zero initial state entries.
Initialization identity is bound into the pricing/reset recipe and final report.
The default fresh seeded model remains a no-op at this initialization boundary.

Retain the old weights, failed fresh final weights and every original cache.
No production head, route, threshold, Hub profile, model or system setting changes.
Real cached-data evaluation is F32/frozen/eval/no-grad, with validated finite
logits before categorical argmax. No expected operand or token ID becomes a model
feature. Existing synthetic guard unit tests are separate from real training.

## Fixed recipe

The only intentional learning change from the failed fresh fit is starting
weights. Same matched cache, 768 TRAIN rows only, seed20261031, batch64,
70 epochs/840 updates, AdamW lr0.001/weight_decay0.01, cosine to0.00001,
foreach=False/fused=False, no clipping. Source/schedule checks must reproduce
`fc2d71800fca6d5b62d1cc3c1d2869cad64cc1138ecf10dc22d4741e64085f25`.
Use original CUDA Python/PyTorch2.11.0+cu130, four CPU threads, TF32 off,
deterministic algorithms, math attention and original primary GPU mapping.

First record all 343 regression predictions before fitting. Then one backward-only
training batch for pricing, no optimizer constructed or step; restart model/seed
in a separate fit process and bind the new pricing receipt. Evaluate/save only
the final checkpoint. Re-evaluate serialized final weights and retained weights
afterward with identical batch64 settings, including all matched rows and all
343 old-development rows. No dev-based checkpoint selection.

Fit phase bound1800s, owned outer bound1920s. Standalone evaluation phase
bound300s, owned outer bound360s. Pricing keeps its60s phase and180s outer bound.
Every process is headless and drains before the next heavy job. Fresh offline-task
admission: physical8GiB, system commitment headroom12GiB, primary GPU2GiB,
output32MiB plus4GiB disk reserve. These are not Hub limits or a process cap.
No cloud job, new activation capture or full model load is part of this experiment.

## Frozen data and regression selection

| Input | SHA256 |
| --- | --- |
| Matched cache states | `6757cde83254fff23010d26d720bc579880e2854750a4ee163b5224b8e12d82a` |
| Matched mask | `d96bed0c391bb9bdfd544c23a9615823229920ecc714903087d32add325f74a0` |
| Matched cache protocol | `2de4d752b3df50a3623624f9acbc95e874b80d0acd02fdc3ee458a679cf75e0c` |
| Old cache states | `3d1e27038d0c7dd7cc440dd876266237869595fbd20a701750670601db735806` |
| Old mask | `d8c8f95e1891c204d4d67006eee86b8ee0cbcdb93123aceb5784c27dde51f2cc` |
| Old cache protocol | `0a078202ccba84943916ec2965a32b68e76ab86c3060621c6e2335f36876a4b8` |
| Regression selection receipt | `87189b7633a05d1503748fba8aec2baede2980882224db50c5cfe79f053ae8b1` |
| Ordered343-ID compact JSON | `5e63f56e919c677f4dd88a56ceec1df80df90772f1e0e35ae386bfafaa8a064b` |

Read existing manifests only; never call acceptance generators. Regression is
259 compute/84 negative rows from OLD DEVELOPMENT, excluding every one of the
96 signed pairs in matched train/development. Ordered IDs and hashes are bound
before the first evaluation. Verify complete selection, uniqueness, split,
pair disjointness and expected targets. All data is inspected development.
Record the 19 matched development pairs exposed to old training separately from
the other13; passing them is diagnostic adaptation, not fresh generalization proof.

## Unchanged gates and reporting

- Matched development:384/384 RAW exact compute tuples and0/384 quote-copy
  false-call candidates. Keep train/per-field/group/template/language tables.
- Regression: no previously correct raw tuple becomes wrong, and no previously
  correct negative gains a false-call candidate. Retain per-row before/after
  vectors, regressions, improvements and existing errors; aggregate gains do
  not compensate for a per-row regression. Retained before/after predictions
  must themselves match exactly.
- Predicted-length deterministic padding is a separate named counterfactual.
  Preserve raw vectors and never use oracle lengths or adjust gates to it.
- Verify serialized reload against all saved final categorical predictions.
  Preserve source/artifact hashes, command receipts, resource observations,
  initialization/optimizer evidence and actual terminal status.

Any numerical/resource failure stops the affected run for diagnosis, preserving
its evidence; no silent retry. An unfavorable quality result is still recorded
and ends this experiment. Even a pass does not deploy the head, open acceptance64,
close I2, establish live result use or prove Hub performance. Return the terminal
result and one evidence-grounded next step before another fit.
