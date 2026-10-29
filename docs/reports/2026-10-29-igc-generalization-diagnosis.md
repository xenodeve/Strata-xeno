# I2 operand-generalization diagnosis

Issue #276. The single fresh matched fit failed: train 384/384 exact compute
tuples, development 24/384. This investigation found no data-path or mode defect
in the checks below. It identifies a generalization failure, with several changed
training conditions still confounded. It does not establish a unique attention,
digit-classification, data-size or quantization cause.

No new fit, cached-data backward pass, activation collection, full model launch or
acceptance64 access occurred. Both saved heads, old datasets and live policy remain unchanged.
The failed final checkpoint is retained; no earlier checkpoint was substituted.
All scores here describe input readout fields, not executed arithmetic or model
answer quality. I2 remains open.

## Exactly matched comparison

The retained and fresh predictions are joined by all 1,536 IDs and canonical
targets on the SAME matched cache. Each split has 384 compute and 384 quote-copy
rows. Operands are within +/-99. The nine fields are operator, lhs/rhs signs,
lhs/rhs length-minus-one, lhs ones/tens, rhs ones/tens. Tens padding is supervised
on positive one-digit operands and must remain zero.

| Development field | Retained | Fresh fixed fit |
| --- | ---: | ---: |
| Raw exact compute tuple | 264/384 | 24/384 |
| Correct operator on compute | 384/384 | 384/384 |
| Signs | 768/768 | 762/768 |
| Lengths | 768/768 | 543/768 |
| Active decimal digits | 1056/1152 | 405/1152 |
| Unused tens slots | 254/384 | 280/384 |
| Quote-copy false-call candidates | 112/384 | 0/384 |

All 360 incorrect fresh compute tuples still select the correct nonzero operator.
A learned no-call cannot hide these operand errors. Conversely, operand fields
on quote-copy rows have no training target and are not scored as extraction
accuracy. Matching intent across these prompts is useful, but does not compensate
for incorrect compute operands.

| True lhs/rhs decimal lengths | Retained exact | Fresh exact |
| --- | ---: | ---: |
| 1/1 | 72/96 | 12/96 |
| 1/2 | 48/96 | 12/96 |
| 2/1 | 48/96 | 0/96 |
| 2/2 | 96/96 | 0/96 |

Each operation has 128 compute rows: retained is 88/128 for each of +, -, *;
fresh is 8/128 for each. By true lhs/rhs sign, retained is ++ 60/96, +- 72/96,
-+ 60/96, -- 72/96; fresh is zero in the first three groups and 24/96 in --.
These are repeated conditions from 32 signed pairs, not 384 independent pairs.

Fresh per-slot development scores are lhs sign 378/384, length 288/384, ones
171/384, active tens 24/192; rhs sign 384/384, length 255/384, ones 179/384,
active tens 31/192. Full true-value results are in the appendix below. The
diagnostic JSON also retains true-to-predicted digit confusion by role, true
length, sign and operation, including padding.

## Coverage and historical exposure

| Training coverage | Matched fresh fit | Original expanded-v2 retained fit |
| --- | ---: | ---: |
| Rows | 768 | 10,240 |
| Positive operand-supervised rows | 384 | 7,680 |
| Negative operator-only rows | 384 (50%) | 2,560 (25%) |
| Independent signed pairs across all rows | 64 | 5,870 |
| Signed pairs with positive operand targets | 64 | 4,742 |
| Unique complete prompts | 768 | 10,152 |
| Template conditions | 4 | 30 |
| Lhs signed / absolute values on positives | 44 / 34 | 183 / 92 |
| Rhs signed / absolute values on positives | 42 / 37 | 183 / 92 |
| Optimizer updates | 840 | 11,200 |

Matched training repeats each signed pair over three operations, two languages
and two intents: 12 rows per pair, of which six supervise operands. Development
has 32 disjoint signed pairs and 24 conditions per pair, including both template
families. There are no duplicate complete matched prompts. Original expanded
training contains 88 duplicate prompt rows. Row count is not independent operand
coverage, and these runs differ in more than row count.

The new train set includes single 3/8 in both operand roles, unlike the old
value-exclusion experiment. It still omits single magnitudes 4/9 on lhs and 7
on rhs. Of the 747 wrong fresh active-digit slots, 711 have a target class already
seen in the same training role, true width and decimal slot. Missing classes
alone therefore cannot account for the failure.

The retained model has prior signed-pair exposure to 36/64 matched train pairs
and **19/32 matched development pairs** through its old training corpus. On
the latter 19 pairs: retained 228/228 compute tuples, fresh 12/228. On the other
13: retained 36/156, fresh 12/156. Thus 264/384 is a useful fixed historical
baseline, not an all-unseen-pair comparison or an isolated data-size experiment.

## What kind of operand errors?

Across 768 fresh development operands, 185 have a correct canonical value
(lhs 105, rhs 80). Of 583 wrong operands, 79 have invalid sign/length/padding
encoding, 60 encode the OTHER signed operand, and 444 encode another wrong value.
There are no complete valid operand swaps or canonical digit reversals in this
run. Ignoring sign, 105 wrong valid operands match the other operand's magnitude;
this overlaps the signed-value category and must not be added to it.

329/747 wrong digit predictions match some active digit elsewhere in the input.
That descriptive coincidence does not demonstrate that attention selected that
digit. The remaining predictions need not come from a literal input digit at all.
For example, `(92) - (-57)` expects operands 92 and -57; fresh produces canonical
12 and -13 while retained is exact on the same row. `(-3) * (8)` expects -3 and
8; fresh produces -3 and 3. These are readout failures before the exact executor.

Token positions were recovered from saved native prefix bytes, not character
offsets. All 12,160 old/new saved prefixes and masks align with their metadata,
both operands are inside the 80-position window, and subtraction order is
preserved. Within the matched data, 216/1152 development active-digit slots use
a token position absent from that training role/width/slot; 140/747 errors are
in those slots. The other **607 errors occur at already covered positions**.
For joint target-digit/position coverage, 273/1152 slots are novel and contain
194/747 errors. All development positions are covered by old expanded training.
Position novelty may contribute but is not a complete explanation.

## Data, labels, losses and optimizer audit

Independent source review plus targeted verification checked the following:

- `context_cache.py:111` selects state rows and masks with the same indices.
  A new sentinel test selects [2,0,2] from real bounded NPY storage, with distinct
  rows, masks, positions, HC streams and channels; order and duplicates survive.
- `readout.py:16` encodes signed magnitudes into the canonical little-endian
  decimal slots. A separate literal scorer verifies all saved IDs and targets.
  Positive padding is included in the loss, not masked out.
- `attention_readout.py:14` uses per-state LayerNorm over 2560 channels, four
  projected HC streams, explicit 80-position embeddings, and a zero-dropout
  contextual encoder. Key masks and final pooling masks exclude padding. There
  are no batch-dependent normalization statistics.
- `training_guards.py:68` averages operator CE over every row and each operand
  CE over positive rows only. Negatives have no operand CE; they still affect
  the shared encoder/key/value/norm through operator CE. The operand term is
  not simply halved when the negative proportion rises.
- `embedding_readout.py:165` selects train indices only. Replaying the exact
  seeded CPU shuffle gives 70 epochs, each train row once per epoch, 12 batches
  per epoch, 840 updates, and 19-43 positives in every batch. The first batch
  matches the backward-only pricing receipt. Actual fit reports 840 finite
  updates with the declared cosine schedule and no clipping. Full per-update
  optimizer snapshots were not stored, so this is source/schedule replay plus
  the recorded update count, not independent replay of every gradient.
- Reloading the final artifact already reproduced every saved categorical
  prediction. The new frozen-forward check additionally tests train FLAG versus
  eval on the same 256 rows: logits bitwise equal for both heads, max difference
  zero. This excludes that tested mode switch as the train/development gap.

No production code bug was reproduced, so no speculative fitter/cache fix or
automatic retraining was made. The new code is diagnostic instrumentation and
stronger tests only.

## Bounded frozen-head controls

A predeclared metadata-only selection (seed 20261110) uses 256 rows: every
matched train pair in one compute/quote condition, every development pair in two
family conditions, operations/languages selected without reading predictions.
Each split contributes 64 compute and 64 quote-copy rows. Both heads see exactly
the same rows. Frozen F32/no-grad, batch32, TF32 off/math attention, original
PyTorch 2.11.0+cu130 and primary GPU mapping were retained.

`readout_probe.py` exposes the real encoder, learned attention, values and queries.
Its decomposed logits are bit-identical to the native head, as are saved/replayed
predictions. Reconstructing 256 original archive SHA256 values from persisted
states, masks and prefix headers also matches. No parameter or gradient changed.

For exploratory operand controls only, replace each field's pooling distribution
with a normalized true-operand span, wrong other-operand span, or all valid
positions; preserve the original operator logits bitwise. A query-only prior
removes the pooled value. All control logits are finite F32 with validated shape
before argmax. True-span metadata is an offline intervention, never a model
feature or production fallback.

| Development control (64 compute rows) | Retained exact / active digits | Fresh exact / active digits |
| --- | ---: | ---: |
| Original learned pooling | 44/64; 176/192 | 4/64; 66/192 |
| Uniform TRUE operand span | 0/64; 37/192 | 2/64; 33/192 |
| Uniform OTHER operand span | 0/64; 26/192 | 2/64; 31/192 |
| Uniform all valid positions | 0/64; 84/192 | 0/64; 67/192 |
| Learned query only | 0/64; 6/192 | 0/64; 22/192 |

Quote-copy false-call counts remain retained 17/64 and fresh 0/64 under every
operand intervention, since operator output is unchanged. Fresh TRAIN active
digits stay 192/192 with uniform global pooling, but only 17/64 whole tuples
survive because signs change (66/128 signs correct, lengths 128/128); original
train is 64/64.

Literal attention-top-one location is not a reliable selector label here:
174/176 correctly predicted retained development active digits, and 65/66 fresh
ones, attend most strongly outside either literal operand span. The encoder and
captured states are contextual. Wrong-span pooling does not isolate information,
and hard uniform span pooling is outside the trained distribution. It also
destroys the better retained head. Therefore these controls do **not** support
a hard-span fix, or a causal partition of every error into selector versus
digit-classifier failures. The observed failure is in the coupled contextual
readout; that finer causal partition remains unresolved.

The bounded probe phase took 7.793 s, excluding final identity work and launcher
overhead; it is not Hub latency. CUDA allocated/reserved peaks were
352,233,984 / 356,515,840 bytes. Largest observed process lifetime private peak
was 3,559,378,944 bytes; minimum sampled system commit headroom 21,639,700,480
bytes and physical available 20,399,157,248 bytes. Samples are not continuous
peak instrumentation. All owned commands reached DONE0 and drained; no process
cap, system/profile change or unrelated cleanup was introduced.

## Ranked hypotheses and next discriminator

1. **Insufficient independent coverage for this fresh learning recipe:** strongest
   working hypothesis, given perfect fit on only 64 pairs and severe development
   loss. Diversity, number of updates, templates, positional support and negative
   proportion all changed together; this is not a causal data-size finding.
2. **A contextual/global readout overfits pair/template structure:** supported as
   a mechanism to investigate by the pooling observations, but literal top-one
   attention does not identify the error source. Hard token-span pooling is not
   justified by these results.
3. **Shared intent/operand representation interactions:** plausible, because
   quote negatives change shared features while leaving operand fields
   unsupervised. Current evidence does not isolate this effect. No-call masking
   is ruled out for these positive errors, and operand-loss averaging is correct.
4. **Indexing, ABI labels, masks, mode or serialization defect:** reduced support
   after exact identity/row/mode checks; no such defect found. A changed
   quantization is also unsupported: both heads consume the same frozen states.

The one proposed next experiment is [warm-start adaptation under the same fixed
matched recipe](../plans/2026-10-29-igc-warm-start-diagnostic-proposal.md).
It changes only initialization relative to the failed fit, tests reuse of the
retained broad-corpus readout and adds a disjoint old-development regression
check. It is a proposal awaiting parent review, not a completed or started fit.
It does not isolate data size or prove full deployment generalization.

## Verification and reproducible evidence

The relevant suite passed 44/44 with default CPU Python and 44/44 with the
original CUDA Python. New decomposition tests were first red (two failures),
then green. Independent source/probe review completed before executing controls.
Existing guard unit tests include tiny synthetic gradient/update fixtures;
these do not fit either saved head or consume real cached examples.
The full engine, HIP/SYCL, live learned integration and fresh acceptance were not
run by this diagnostic-only change.

The identical suite in each Python environment was:

```
python -B -X utf8 -m unittest tools.test_igc_readout_probe tools.test_igc_context_cache tools.test_igc_training_guards tools.test_igc_embedding_readout tools.test_igc_matched_evaluate tools.test_igc_input_manifest tools.test_igc_development_audit
```

The cached control command was the qualified CUDA Python executable with
`-B -X utf8 igc_generalization_probe_01.py`, through the owned headless launcher
(240s outer, 180s phase bound). Metadata audits used the same launcher with
default CPU Python; no native model executable was launched.

Reproduction receipts retain command lines, executable/source identities,
selection IDs, timings and original unmodified evidence. Basenames:

- `igc_generalization_audit_01.py` / `igc-generalization-audit-01.json`:
  all-row field/value/confusion and coverage audit.
- `igc_generalization_alignment_01.py` /
  `igc-generalization-alignment-01.json`: all old/new prefixes and frozen selection.
- `igc_generalization_probe_01.py`, `igc-generalization-probe-plan-01.json`,
  `igc-generalization-probe-01/report.json`: bounded cache controls and schedule.
- `igc-generalization-alignment-full-02.json`,
  `igc-generalization-position-coverage-01.json`,
  `igc-generalization-position-errors-01.json`: digit-specific position audit.
- `igc-generalization-auxiliary-01.json`: historical exposure, example fields and
  the frozen 343-row disjoint regression IDs.
- `igc-generalization-tests-{cpu,runtime}-01.log`: exact 44-test commands/results.

These local evidence files are not published training data. No public absolute
machine paths, model files or large caches are added to Git.

| Frozen artifact | SHA256 |
| --- | --- |
| Retained weights | `519fbc196993c94676f4031f7bd45ad9abf661f8b7b4fd3b9d6358204b5294e2` |
| Failed fresh final weights | `29abae54c429f881d128ff049edfb6d8ae40de19e363583553f813c3564ac1eb` |
| Matched states | `6757cde83254fff23010d26d720bc579880e2854750a4ee163b5224b8e12d82a` |
| Matched mask | `d96bed0c391bb9bdfd544c23a9615823229920ecc714903087d32add325f74a0` |
| Retained matched predictions | `019ae84e7a3f1c8a21e4beed1999a85bd0ed3ae1e3f23d221d88be4920099103` |
| Failed fresh predictions | `b5707138056dd61c00e53d2da993c63d03da78caefaed48f58945d06e733cd92` |
| Frozen probe selection/alignment | `15de774501a235562072da4b6667c299725c473cdf01f77c5dec5997fe8babff` |
| Reconstructed training schedule | `fc2d71800fca6d5b62d1cc3c1d2869cad64cc1138ecf10dc22d4741e64085f25` |
| All-row audit result | `fcf5a3905108bbc70bc3334aab6a8b3e989512c739d59867bc62965044750710` |
| Frozen bounded probe plan | `3ed5507dd5741c99c7ad44c788561d3a98811ceb07362f46c763c5f386f49abf` |
| Bounded probe result | `ab3c09696ec015606bc51cb5f00e90bd97a4dcc67d5b1fd7486b716e74d31e3a` |
| Digit-position error result | `e65e0a64ad9b0bcc484542742946b82e747fc0d7ba977e4898a01bb1ad8e62b9` |

## Appendix: exact operand fields by true signed value

Counts below use compute rows only and all four fields for that operand,
including canonical padding. They are descriptive repeated-condition counts,
not independent task estimates. The other operand and operator need not be
correct for an operand to be counted here.

### LHS

| True signed value | Retained exact | Fresh exact |
| --- | ---: | ---: |
| -97 | 12/12 | 0/12 |
| -79 | 12/12 | 0/12 |
| -74 | 12/12 | 0/12 |
| -64 | 12/12 | 0/12 |
| -55 | 12/12 | 0/12 |
| -47 | 12/12 | 0/12 |
| -38 | 12/12 | 0/12 |
| -37 | 12/12 | 0/12 |
| -9 | 12/12 | 0/12 |
| -8 | 12/12 | 12/12 |
| -3 | 0/36 | 21/36 |
| -2 | 24/24 | 12/24 |
| -1 | 12/12 | 0/12 |
| 0 | 24/24 | 24/24 |
| 3 | 0/36 | 12/36 |
| 7 | 12/12 | 0/12 |
| 8 | 12/12 | 12/12 |
| 9 | 12/12 | 0/12 |
| 17 | 12/12 | 0/12 |
| 20 | 12/12 | 12/12 |
| 27 | 12/12 | 0/12 |
| 44 | 12/12 | 0/12 |
| 62 | 12/12 | 0/12 |
| 67 | 12/12 | 0/12 |
| 92 | 12/12 | 0/12 |
| 94 | 12/12 | 0/12 |

### RHS

| True signed value | Retained exact | Fresh exact |
| --- | ---: | ---: |
| -93 | 12/12 | 0/12 |
| -89 | 12/12 | 3/12 |
| -69 | 12/12 | 0/12 |
| -66 | 12/12 | 0/12 |
| -65 | 12/12 | 0/12 |
| -61 | 12/12 | 12/12 |
| -57 | 12/12 | 0/12 |
| -31 | 12/12 | 0/12 |
| -8 | 0/12 | 12/12 |
| -7 | 12/12 | 0/12 |
| -5 | 24/24 | 0/24 |
| -4 | 24/24 | 12/24 |
| -3 | 0/12 | 0/12 |
| -1 | 12/12 | 0/12 |
| 1 | 24/24 | 12/24 |
| 3 | 0/12 | 0/12 |
| 4 | 12/12 | 0/12 |
| 5 | 24/24 | 5/24 |
| 8 | 0/24 | 12/24 |
| 19 | 12/12 | 0/12 |
| 30 | 12/12 | 0/12 |
| 54 | 12/12 | 0/12 |
| 67 | 12/12 | 0/12 |
| 85 | 12/12 | 12/12 |
| 88 | 12/12 | 0/12 |
| 92 | 12/12 | 0/12 |
| 98 | 12/12 | 0/12 |
