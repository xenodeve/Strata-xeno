# I2 single fixed-recipe matched fit

Issue #276. The one approved fit completed numerically, but failed the operand
extraction quality gate. It is not promoted and does not close I2. No second fit,
checkpoint search, threshold change or acceptance64 access occurred.

## Final result

| Design split | Raw exact compute tuple | Quote-copy false-call candidates | Active digits | Padding slots |
| --- | ---: | ---: | ---: | ---: |
| Train | 384/384 | 0/384 | 1152/1152 | 384/384 |
| Development | 24/384 (6.25%) | 0/384 | 405/1152 | 280/384 |

Development operator selection is 384/384, signs 762/768 and lengths 543/768.
The strict gate requires 384/384 raw exact positives and zero false-call
candidates among 384 quote-copy negatives. The intent condition passes on this
corpus; operand extraction fails. These are readout tuples, not arithmetic-core
execution results or final model answers. The retained serving/controller weights
remain unchanged.

There are 64 train-design signed pairs and 32 development pairs. Only 2/32
development pairs have every compute condition correct. The descriptive signed-pair
bootstrap exactness interval is [0%, 15.625%], using the previously fixed 4,096
resamples/seed 20261109. Zero observed false calls produce a degenerate empirical
bootstrap interval; this is not a guarantee for future negatives.

The separately declared predicted-length padding composition gives the SAME
384/384 and 24/384 exact tuples, with no tuple recovered. It cannot rescue this
failure and was not adopted. Length errors mean a predicted-unused slot can be
truly active; raw and variant field metrics remain separate.

## Paired comparisons

Against the retained mapper on the SAME cache, development raw exactness falls
264/384 to 24/384: -62.5 percentage points, signed-pair bootstrap interval
[-81.25, -43.75] points. Paired outcomes: 12 both correct, 252 retained-only
correct, 12 new-only correct, 108 both wrong. Quote-copy false-call candidates
fall 112/384 to 0/384, a -29.167-point difference, interval [-36.198, -22.656].
Improved intent does not offset the numeric extraction regression.

Within the same development expression/language, the two template families have
12/192 both-correct compute pairs and 180/192 both-wrong, with no success-status
transitions. All 192 quote-copy family pairs have no-call correct in both.
Compute/quote-copy joint success is 24/384. The JSON artifact retains 41 partition tables: 24 full
split/family/language/length intersections, 17 marginal summaries, and 288
expression-group tables.

This is inspected paired development, not untouched acceptance. The new training
set includes the single values 3/8 in both roles; it is not the former value-exclusion
experiment. The retained model had a different, much larger training corpus and
11,200 updates versus 840 here. This comparison does not isolate data size,
optimization budget or a representation cause.

## One backward-only pricing batch

Before fitting, exactly one shuffled TRAIN-only batch of 64 rows ran backward.
No optimizer was constructed, optimizer steps=0, no weight file was saved, and
fresh model fingerprint remained
`7f363077fe63297aeabefd9d38b9436332fbb3e1805f1ec849f9d0fa665d6547`.
Its 64 IDs, recipe, source, protocol and complete state/mask/report/manifest
identity were bound in the receipt. All IDs are from the 768-row train subset.

Initial loss 13.3227253 was finite. Observed CUDA allocator peak 699,594,752 B,
reserved 918,552,576 B; process private lifetime peak 4,002,820,096 B. The reported
2.253 s covers the backward phase before final cache rehash, not total job latency.
The owned launcher ended DONE0 and drained before the separate fit.

## Exactly one fresh fit

Guard implementation: local commit `3c84f75f2e58e3bd3ec259701f8cfa22dff8740d`.
The separate process verified the initial fingerprint and first-batch order
against the pricing receipt before updates. Seed 20261031, contextual learned-query
head 96,964 parameters, 70 epochs, batch 64, exactly 840 AdamW updates,
lr 0.001 / wd 0.01, cosine to 0.00001. Only 768 train rows were used; no development
checkpoint selection. Only the final checkpoint was evaluated/saved.

Final evaluation used eval/frozen/no-grad, 24 batches of at most 64 rows. It was
then independently RELOADED and evaluated again with the same batch/mathematical
settings: every categorical prediction matched the saved final prediction array.
All parameters were finite, and no optimizer/backward ran in that reload check.
This rules out serialization as the explanation for the reported train/dev gap.

The fit plus final in-memory evaluation interval was 132.191 s; epoch 70 ended
at 129.632 s. This is not whole-job/Hub latency. Initial epoch mean loss 12.5522744
fell to 0.0205263. Every update passed finite loss/gradient/parameter/optimizer
state checks without clipping; numerical failure count is zero. Fit bound was
1,800 s and owned outer bound 1,920 s; every owned launcher reached DONE0/drained.

CUDA allocator peak 728,945,664 B, reserved 918,552,576 B. Across the 14 five-epoch
resource snapshots, largest observed process lifetime private peak 4,274,421,760 B;
minimum system commit headroom 21,151,617,024 B and physical available
19,759,194,112 B. These snapshots do not measure every instant. No private-commit
cap, pagefile/profile change or unrelated process termination occurred. PyTorch
2.11.0+cu130, original CUDA Python executable, RTX 5060 Ti, TF32 off/math attention,
four CPU threads and logical GPU visibility 1,0 were retained.

## Identity and independent verification

| Artifact | SHA256 |
| --- | --- |
| Final experimental weights | `29abae54c429f881d128ff049edfb6d8ae40de19e363583553f813c3564ac1eb` |
| Final raw predictions JSON | `b5707138056dd61c00e53d2da993c63d03da78caefaed48f58945d06e733cd92` |
| Consumed pricing receipt | `77fdeeb5a77f4cb30eaa578ad33b8b2061d12ee6e67ec689117d1c8c157e54f2` |
| Retained weights, unchanged | `519fbc196993c94676f4031f7bd45ad9abf661f8b7b4fd3b9d6358204b5294e2` |
| Matched states, unchanged | `6757cde83254fff23010d26d720bc579880e2854750a4ee163b5224b8e12d82a` |
| Frozen matched protocol | `2de4d752b3df50a3623624f9acbc95e874b80d0acd02fdc3ee458a679cf75e0c` |

The independent literal scorer verified all 1,536 IDs/targets, raw/composed exact
counts, false calls, initialization/receipt identity and source hashes. The final
artifact directory occupies 1,745,733 B; large weights/caches are not committed.
Evidence basenames: `igc-matched-training-price-01/price.json`,
`igc-matched-context-fit-01/{report,predictions}.json`,
`igc-matched-final-reload-01/report.json`, `igc-matched-fixed-fit-qualification-01.json`,
plus command receipts and owned-launcher logs. No private absolute paths are
needed in this public-facing report.

Guard tests passed 40/40 on both default CPU Python and original CUDA Python;
independent code review and source/provenance checks completed before pricing.
No source changed between price and fit. All new weights remain separate and
experimental; `eligible_for_fresh_evaluation` is false. No acceptance, second tap,
live model/controller replacement, deployment/default, push or backup upload.

## Interpretation and stop point

Observed: the fixed small-corpus recipe fits every training tuple and learns the
quote-copy distinction, while generalizing poorly to held-out signed pairs.
Saved label/ID audits match the canonical numeric encoding, and reloaded artifact
predictions match. The failure is a valid negative quality result, not a
numerical crash or a reason to select an earlier checkpoint.

A read-only exposure audit also finds errors on development operand values that
occurred in the same training role, so absence of individual values is not a
complete explanation. It does not establish a unique memorization/attention/
quantization cause. The next justified work is analysis of pair/value coverage
and extraction behavior under a separately specified experiment. No larger fit,
fine-tuning of the retained state, hyperparameter search or automatic retry was
performed. Full-Hub admission, model result use and end-to-end benchmarking remain
separate open gates. I2 stays open; this experiment stops here as requested.
