# I2 retained mapper on matched development data

Issue #276. No new fit, weights, threshold, live routing or acceptance data.
This evaluates the retained 96,964-parameter contextual first-attention mapper
on the independently captured matched corpus, before any matched-corpus fit.
The labels `train` and `development` describe the NEW corpus design; they do
not mean this retained mapper was trained on the new training subset.

## Result: I2 remains open

| Design split | Compute raw exact tuple | Quote-copy false-call candidates | Active digits | Padding slots |
| --- | ---: | ---: | ---: | ---: |
| Train | 252/384 (65.625%) | 98/384 | 1068/1152 | 260/384 |
| Development | 264/384 (68.75%) | 112/384 | 1056/1152 | 254/384 |

On both splits, positive operator selection is 384/384, signs 768/768 and
lengths 768/768. No positive predicts no-call. False calls above are raw
nonzero operator predictions on negatives, regardless of canonical operand
encoding. They are not claims about the separate readiness/dual-head live gate.
Negative operand fields are not scored as executable targets.

The development gate requires 384/384 exact positive tuples and zero false calls
among 384 negatives. This retained mapper fails both conditions. It is not a
usable integration, a fresh-acceptance result or a whole-model quality result.

## Uncertainty and matched comparisons

The independent unit is the signed operand pair: 64 training-design and 32
development pairs. The three operations, both languages and paired conditions
within a pair are dependent. Equal-weight pair means use 4,096 fixed-seed
bootstrap resamples, percentile 95% intervals:

| Design split | Exact tuple rate and interval | False-call rate and interval |
| --- | --- | --- |
| Train, 64 pairs | 65.625% [54.525%, 76.823%] | 25.521% [21.354%, 29.427%] |
| Development, 32 pairs | 68.75% [53.125%, 84.375%] | 29.167% [22.656%, 36.198%] |

These are descriptive intervals for inspected development, not independent
token trials or a release guarantee. Perfectly uniform observed contrasts can
produce a degenerate bootstrap interval; that does not prove population
equivalence or zero uncertainty on future prompts.

For the SAME development expression and language, seen/new templates have
132/192 both-correct compute pairs and 60/192 both-wrong, with no success-status
transitions. This is more informative than comparing unmatched cohorts, but
does not say the erroneous categorical predictions are identical.

Quote-copy false calls increase from 45/192 (seen) to 67/192 (new). Paired no-call
transitions: 122 both correct, 25 seen-only correct, 3 new-only correct, 42 both
wrong. New-minus-seen no-call difference is -11.458 percentage points, signed-pair
bootstrap interval [-15.104, -7.8125]. Positive accuracy did not improve in this
matched comparison, and the negative-intent behavior worsened.

Compute/quote-copy joint success within the same expression, template and
language is 185/384 in train-design and 191/384 in development. Pair-cluster
intervals are [39.323%, 57.194%] and [36.458%, 63.021%]. Full paired transitions,
41 marginal/intersection partitions and 288 expression-group tables are retained
in the JSON report. Group intervals with fewer than two pairs are absent.

## Field failures and composition counterfactual

All 96 development active-digit errors are on single absolute value 3:
94 predict 4 and 2 predict 1. Training-design has 84 such errors: 70 predict 4,
11 predict 1 and 3 predict 7. Signs and lengths are correct. Absolute 8 has no
active-digit error in this run; its wrong operand encodings involve padding.
The observed pattern is consistent with the previously inspected diagnostic,
but does not establish an attention-copy, centroid or quantization cause.

| Design split | Raw exact by true length pair 1/1,1/2,2/1,2/2 | Operand errors: padding-only / active-only / both |
| --- | --- | --- |
| Train | 48/96,60/96,48/96,96/96 | 48 / 8 / 76 |
| Development | 72/96,48/96,48/96,96/96 | 36 / 2 / 94 |

Language/template details distinguish arithmetic from intent. Development
positive exactness is 132/192 in each language, while quote-copy false calls are
23/192 English and 89/192 Thai. Train-design is 123/192 and 129/192 positive exact,
with 0/192 and 98/192 false calls. These marginals do not isolate a language cause;
the full matched conditions remain the evidence.

The separately predeclared PREDICTED-length padding composition changes 426 raw
rows, recovers 48 train-design and 24 development positive tuples, and regresses
zero exact tuples. Resulting exactness is 300/384 and 288/384. It leaves active
digits, signs, lengths and operators unchanged; false-call counts remain 98/112.
It does not fix the active 3 errors. It is an offline representation variant,
not learned padding accuracy, a replacement raw score or a deployed policy.

## Frozen identity, execution and resources

- Retained weights SHA256 `519fbc196993c94676f4031f7bd45ad9abf661f8b7b4fd3b9d6358204b5294e2`;
  strict contextual/learned-query/no-final-query architecture and 96,964 parameters.
- State cache `6757cde83254fff23010d26d720bc579880e2854750a4ee163b5224b8e12d82a`;
  mask identity is bound through pinned completed report
  `7a3e5a5b9178df601a14a3bc7e4af4592ca45ccd3f3a5cfb82bdcc2f3b85fecd`.
- Evaluation protocol `8e732a66675f68d7ef7fee0b270a783f4be6a28032fd3067203930f0377cd366`;
  cache protocol `2de4d752b3df50a3623624f9acbc95e874b80d0acd02fdc3ee458a679cf75e0c`.
- Original CUDA Python executable SHA256
  `060435b7c202f88ad5404eadbea481c06c0bb9cee00ebac9084840dcbf03ff92`;
  PyTorch 2.11.0+cu130, CUDA 13.0, RTX 5060 Ti, TF32 off/math attention, four CPU threads.
  GPU visibility 1,0 and unset device order were enforced before CUDA access.
- `eval()`, `requires_grad_(False)` and `no_grad()`; no optimizer. Every parameter
  remained bit-identical and every gradient remained absent. Source hashes and
  complete cache state/mask digests matched again after inference.

The first attempt used the default Python's PyTorch 2.14.0+cpu and refused before
inference; no output directory was created. Its original log is retained. The
successful second attempt uses the original training environment above. No
package was installed and no GPU/system setting changed.

The successful run evaluated 24 bounded batches, at most 64 rows at a time. Inference
including per-batch observation took 4.691 s; total validation/inference/report
preparation 14.727 s. The inference interval runs only the head on cached activations, without Gyro
layer execution. These are instrumented offline costs, not a Hub speed claim.
CUDA allocator peak 666,169,344 B, reserved 687,865,856 B. Process lifetime private
peak 3,672,035,328 B; working-set peak 1,740,439,552 B. The 24 post-batch system samples
had minimum physical available 16,816,238,592 B and commit headroom 22,279,364,608 B;
they do not cover every instant of the process. No private-memory cap was used.
Whole-device VRAM peak and full-Hub inference were not measured.

All raw predictions, derived tables and source identities are in
`igc-retained-matched-evaluation-02/{report,predictions}.json`; independent literal
field-label/scoring checks are in `igc-retained-matched-qualification-01.json`.
The command/environment receipts and both attempt logs are retained locally.
Independent scoring confirms all 1,536 IDs/targets, raw and composed exact counts,
false calls, cluster counts, raw predictions checksum and unchanged sources.

## Verification and limits

Test-first checks covered raw error decomposition, duplicate/class rejection,
matched pairing, equal pair weighting, no-grad/unchanged parameters, final-batch
deadlines, frozen mask receipt, GPU environment and CLI refusal. Independent
review found three gaps before prediction; all were reproduced and closed:
mask identity from an unpinned report, unvalidated GPU mapping, and a late last
batch that could report success. No model result was used to tune these fixes.
Regression: 23/23 passed in 6.433 s on the default CPU test environment; original CUDA Python
runtime regression initially hit sandbox temporary-directory permissions in five
file fixtures. The preserved native-context rerun with a task-owned Temp root
passed 23/23 in 4.614 s; no test or source change was needed for that rerun.

No acceptance64 read, second tap, new training, live-policy/model replacement,
Hub default, pagefile change, push or backup upload. Centroid exploration was
not run and did not delay the core evaluation. Full-Hub admission remains a
separate gate. See the fixed training proposal below; do not treat this failed
retained baseline as a reason for an automatic tuning loop.

## Fixed next-training proposal, not executed

Use the already frozen matched fit protocol without changing architecture,
thresholds, loss, target/padding convention, dataset or split. Initialize a NEW
contextual learned-query head from seed 20261031, not a continuation of these
retained weights. Fit only 768 train-design rows (384 compute / 384 quote-copy);
keep 32 development signed pairs out of fitting. The new training set explicitly
contains all signed 3/8 slot cases and paired intent controls, which address the
observed coverage gaps without changing the numeric domain.

Recipe: 70 epochs, batch 64, AdamW lr 0.001 / weight_decay 0.01, cosine to 0.00001,
F32/TF32 off, math attention, four CPU threads. Exactly 840 update steps and 24
final evaluation batches; use only the final checkpoint. Stop at the existing
1,800 s fit bound (owned outer bound 1,920 s), or any nonfinite loss/gradient/state.
No checkpoint selection, threshold tuning or automatic retry after failure.
Require 384/384 development positive exact tuples and 0/384 false calls under RAW
strict scoring. A pass would remain development evidence, not full-I2 completion.

Before fit, test the fitter's finite-gradient/parameter guards: current
`embedding_readout.py` checks finite loss, but does not explicitly check every
gradient before `optimizer.step()`. Then price ONE bounded training-only batch
with no optimizer step and fresh initialization; discard that pricing instance
and reset seed/model/optimizer for the actual fit. This is a proposed prerequisite,
not backward computation performed by the present evaluation.

Use the same cache through NpyRows; do not copy it or collect another tap.
Existing admission is 8 GiB physical / 12 GiB commitment and 2 GiB free GPU, with the
retained cache plus at least 16 MiB output and 4 GiB free-disk reserve. Measure again
before launch; no process cap or system/profile changes. Previous same-geometry
fit observed 728,959,488 B CUDA allocated / 920,649,728 B reserved, while this no-grad
run observed 666,169,344/687,865,856 B. These observations do not guarantee a new
training peak. The old fit had 11,200 updates versus 840 proposed here; that work
count is not an end-to-end runtime or quality prediction.

The proposed command uses the verified original CUDA Python, existing
`tools.igc.embedding_readout` CLI, `--contextual --device cuda --epochs 70
--seed 20261031 --schedule cosine --activation-cache context --fit-timeout 1800`,
the matched cache, frozen matched protocol and a NEW output directory. Exact
arguments/resource prerequisites are saved separately in the local proposal.
Nothing in this report starts that fit or opens acceptance64.
