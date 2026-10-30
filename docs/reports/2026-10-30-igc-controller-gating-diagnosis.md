# Why cached controller coverage remains 33/48

Issue #276. Cached diagnosis is complete. No model weights, thresholds, gates or
production settings change, and no new activation collection or fit runs. The
[one proposed coherent development protocol](../plans/2026-10-30-igc-coherent-development-protocol.md)
targets the measured readiness/embedding failures rather than another executor
benchmark.

## Exact mechanism on the reused 64 prompts

Both original and replay attention heads already predict all nine fields
correctly on all 48 positive prompts. The embedding head is exact on 36/48;
readiness permits 37/48. Their intersection is 33. Changing attention weights
therefore does not change any positive tuple, selected request or native result
on this particular capture.

| Exclusive positive-case partition | Original | Replay |
| --- | ---: | ---: |
| Selected and correct | 33 | 33 |
| Readiness wrong, embedding correct | 3 | 3 |
| Embedding wrong, readiness permits | 4 | 4 |
| Both readiness and embedding wrong | 8 | 8 |
| Attention wrong | 0 | 0 |

The eleven readiness misses include the eight joint errors; they are not added
again to the four disagreements. Conditional positive-only oracle substitutions
give ceilings of **36/48 for readiness alone, 37/48 for embedding alone and
48/48 for both**, with the other fields fixed. These are explanatory offline
bounds, not trained results, a threshold proposal or a runtime bypass.

| Actual prompt family | Positive cases | Selected correct | Readiness bypass | Disagreement after readiness |
| --- | ---: | ---: | ---: | ---: |
| `en-exact-result`: What is the exact result ...? | 12 | 0 | 8 | 4 |
| `en-calculate` | 12 | 11 | 1 | 0 |
| `en-evaluate` | 12 | 11 | 1 | 0 |
| `th-calculate` | 12 | 11 | 1 | 0 |

English positives select 22/36, Thai positives 11/12; all sixteen negatives remain
unselected. The readiness training positives used four templates (24 examples
each): `en-calculate`, `en-evaluate`, `th-calculate`, `th-solve`. The failed query
family is not one of those fit templates. This is a measured transfer pattern,
not proof of the neural cause; the next corpus holds operands constant while
varying the frame and intent.

The unchanged native readiness head was replayed on the same 64 saved final
mixed vectors, in normal and graph modes. Raw logits and decisions agree between
those modes; every decision matches both frozen native controller audits.
Compute-minus-no-call margins for the eleven misses range from -14.14627 to
-1.91886 (median -6.82615). They are not near ties at the established 0.001
forward tolerance. No threshold is changed. Softmax values in the JSON are
uncalibrated display statistics and never selection inputs.

These continuous scores are **new native batch64 replays**, not historically
recorded raw logits from the earlier capacity1 pipeline. The readiness artifact
and final states are shared by original/replay, so the row table shows a common
score with separate recorded decisions and audits. Six negative rows change
unused attention fields; their operator/readiness remain no-call, and none
changes selection or native result.

The capture's historical split is 48 TRAIN/16 HELDOUT. It has been inspected
repeatedly and is distinct from sealed acceptance64. This diagnosis does not
convert it into fresh evaluation or model-use evidence.

## Broader legacy evidence: a conditional gain, still not actual calls

The matching embedding-head artifact's saved predictions are available for all
343 legacy rows and bind to the unchanged native embedding state. Joining those
fields with qualified attention predictions gives:

| Legacy, 259 positive /84 negative rows | Original attention | Replay attention |
| --- | ---: | ---: |
| Raw attention positive tuples exact | 206 | 259 |
| Raw embedding tuples exact, unchanged | 167 | 167 |
| Correct consensus if readiness permits every positive | 127 | 167 |
| Nonzero agreed/canonical negative requests if readiness is 1 | 0/84 | 0/84 |
| Raw attention nonzero predictions on negatives | 55/84 | 42/84 |

Thus replay can recover forty additional consensus-correct legacy rows, but the
actual call rate is unknown without their paired readiness states. Even perfect
readiness cannot exceed 167/259 with these fixed embedding fields. No readiness
tensor is fabricated for this comparison. The 42 raw attention errors are not
42 demonstrated whole-controller calls: none has a nonzero agreed request in
this cached pairing. That does not establish live safety or fresh generalization.

## Precisely what is missing from the larger caches

The previous qualification report's statement that both larger cohorts had only
attention data was too broad. The bounded index/prefix audit corrects it:

| Cohort | Attention available | Existing embedding lookup available | Paired final/output observations |
| --- | ---: | ---: | --- |
| Legacy | 343/343 | 343/343 | Missing for all 343 |
| Matched development | 768/768 | 384/768 | Missing for all 768 |

The old embedding/attention manifests have identical 10,624 rows and masks. All
old lookup indices were checked against content-hashed stored token prefixes;
the existing 217-row native embedding table and its array hashes are intact.
Matched compute prompts have all required dictionary tokens, while all 384
quote-copy prompts need at least one of eleven absent token IDs, equally split
across English/Thai. Filling those eleven rows would be new embedding extraction,
which is not performed here.

None of these 1,111 prompt hashes appears in the readiness 320, readiness
augmentation 416 or output 30 captures. Those final vectors cannot be joined by
similar numbers, geometry or a different prompt. Missing roles are the same
request's final-HC anchor/readiness state, prefix/invocation lineage and subsequent
pre-injection output-position states/targets. The prefix extractor is a genuine
`[0,1)` early stage, not a full-Hub substitute. Joint observation on actual
batched/wave and accepted decode paths remains a producer prerequisite.

## Next experiment and resource limits

The proposed protocol has 48 expression groups, each crossed with both languages
and imperative/question/quoted-copy/label conditions, plus sixteen ordinary or
out-of-module negatives: **400 original anchors**. A balanced predeclared subset
of 48 positives also supplies at most 1,488 additional teacher-forced output
states. The metadata witness uses 44 connected pair/fact clusters; it is not 400
independent trials. It is disjoint from the seven named accessible archives on
declared keys, with no claim about unopened acceptance64. No dataset is collected
or fitted. Zero-operand freshness and daily-use representativeness remain outside
this diagnostic's claim.

First implement the joint full-Hub observation/schema/writer and its ownership,
missing-cache and future-label tests, then collect this one cohort when admitted.
Anchor capture must precede teacher-forced output capture. Compare unchanged
original/replay gating and separately labeled output oracle controls; keep
actual generated-result use and I4/I5 tests separate. This is the next useful I2/I3
dependency; another exact-executor microbenchmark cannot supply it.

At 2026-10-30 01:36:25 UTC, physical availability is **18.1936 GiB**, below the
actual selected Hub's 24-GiB gate. Commit headroom is **15.0507 GiB**, versus the
new protocol's disclosed 47.6169-GiB task estimate (observed same-profile peak,
fixed components, bounded host streaming and 2-GiB reserve). This is an admission
estimate, not a process cap or guaranteed upper bound. No full-model launch is
attempted. Planned disk demand including reserve is 5.4066 GiB, rounded to a
5.5-GiB preparation requirement. The temporary volume has only 4,563,030,016 B
free, but the workspace volume has 10,877,382,656 B and can hold the proposed
ignored `build-igc-coherent-dev-01/data` archive without deleting old evidence.

Configuration audit also corrects the earlier proposed "normal priority" wording:
the actual selected profile and both historical captures already specify
`--pool-priority 2` /`--process-priority 2`. Preserve those actual settings and
label future measurements accordingly; do not edit the served profile to match
an assumption. The selected profile and server content still match their pinned
source hashes.

## Verification and deliverables

`controller_diagnosis.py` is an offline join, not an inference component. Tests
first fail on the absent module, then **9/9** focused tests pass; the final
qualified-Python suite passes **15/15**, including audit and provenance regressions. They cover
exclusive error accounting, oracle bounds, dormant negative-field drift, row
order/identity, readiness nonfinite/mismatch, native owner/byte-count faults and
the prior typed-record oracle. The native readiness probe uses existing weights;
no learning or policy selection is performed. The result report, full 64 CSV,
machine JSON, source hashes and underlying score/audit binaries remain local.

The first broader test attempt could not create provenance fixtures in the
sandbox's default temporary directory. A process-local scratch directory under
the authorized task root resolves that environment fault. The initial wrapper
also needed its project import path; both failed logs are retained and the final
unchanged tests pass. No permission, filesystem security or assertion is weakened.
The final named-process inventory finds no task-owned leftovers or active engine;
it does not stop any process. At 01:42:11 UTC, physical/commit availability remains
18.4222/14.7844 GiB, still below the live/collection requirements.

Evidence basenames: `igc-controller-diagnosis-01/{report.json,all64.csv,all64.md}`,
`igc-controller-capture-gap-02.json`, `igc-controller-final-capture-overlap-01.json`,
`igc-legacy-consensus-ceiling-02.json`,
`igc-coherent-development-feasibility-01.json`,
`igc-coherent-development-design-witness-02.json`, and
`igc-coherent-development-resource-estimate-01.json`. Large arrays, private paths
and sealed-test contents are not published.

## All 64 aligned rows

Fields are `[op, lhs_sign, rhs_sign, lhs_two_digit, rhs_two_digit, lhs_ones,
lhs_tens, rhs_ones, rhs_tens]`. O/R means original/replay. Consensus compares all
nine fields, even though unused negative fields do not matter after readiness
rejects a call. Selection 0 is selected, 1 is no-call and 4 is disagreement. `none`
is a valueless audit, not numeric zero. The CSV/JSON retain full F32 scores and
the complete synthetic prompt; the table rounds logits for readability.

| # / id | Ready L0 / L1 (shared) | O/R ready | Embedding tuple | Original attention | Replay attention | O/R consensus | O/R selection | O/R native result |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 / op2-12 | -3.826501 / 2.53478 | 1/1 | `2,1,1,0,1,7,0,4,6` | `2,1,1,1,0,6,7,4,0` | `2,1,1,1,0,6,7,4,0` | 0/0 | 4/4 | none/none |
| 1 / op3-03 | -8.490267 / 5.777466 | 1/1 | `3,0,1,1,1,1,3,4,2` | `3,0,1,1,1,1,3,4,2` | `3,0,1,1,1,1,3,4,2` | 1/1 | 0/0 | -744/-744 |
| 2 / op1-03 | 3.476769 / -3.66112 | 0/0 | `1,0,0,1,1,4,3,0,3` | `1,0,0,1,1,4,3,0,3` | `1,0,0,1,1,4,3,0,3` | 1/1 | 1/1 | none/none |
| 3 / op0-08 | 6.361367 / -5.897553 | 0/0 | `0,0,1,1,0,3,2,5,0` | `0,0,1,0,1,3,3,2,6` | `0,0,1,0,1,3,3,2,6` | 0/0 | 1/1 | none/none |
| 4 / op2-05 | -1.953834 / 1.684261 | 1/1 | `2,0,1,1,1,0,2,5,1` | `2,0,1,1,1,0,2,5,1` | `2,0,1,1,1,0,2,5,1` | 1/1 | 0/0 | 35/35 |
| 5 / op1-00 | -7.216865 / 4.557928 | 1/1 | `1,1,1,1,1,8,2,9,4` | `1,1,1,1,1,8,2,9,4` | `1,1,1,1,1,8,2,9,4` | 1/1 | 0/0 | -77/-77 |
| 6 / op1-12 | 5.350087 / -5.90147 | 0/0 | `1,0,0,0,0,0,0,2,0` | `1,0,0,1,1,0,6,2,2` | `1,0,0,1,1,0,6,2,2` | 0/0 | 1/1 | none/none |
| 7 / op0-14 | 7.348021 / -7.163519 | 0/0 | `0,0,1,1,0,8,3,5,0` | `0,0,1,0,1,7,3,8,7` | `0,0,1,0,1,7,3,8,7` | 0/0 | 1/1 | none/none |
| 8 / op0-13 | 6.341626 / -5.833664 | 0/0 | `0,0,0,1,0,8,0,5,0` | `0,0,0,0,1,3,7,8,9` | `0,0,0,0,1,3,7,8,9` | 0/0 | 1/1 | none/none |
| 9 / op1-07 | -4.141241 / 2.040788 | 1/1 | `1,0,0,1,1,8,6,5,2` | `1,0,0,1,1,8,6,5,2` | `1,0,0,1,1,8,6,5,2` | 1/1 | 0/0 | 93/93 |
| 10 / op3-09 | -10.1431 / 6.858655 | 1/1 | `3,1,1,1,1,1,6,0,3` | `3,1,1,1,1,1,6,0,3` | `3,1,1,1,1,1,6,0,3` | 1/1 | 0/0 | 1830/1830 |
| 11 / op2-09 | -8.221696 / 5.350421 | 1/1 | `2,1,0,1,1,6,5,3,6` | `2,1,0,1,1,6,5,3,6` | `2,1,0,1,1,6,5,3,6` | 1/1 | 0/0 | -119/-119 |
| 12 / op3-04 | -8.74173 / 5.82558 | 1/1 | `3,1,1,1,1,6,5,7,9` | `3,1,1,1,1,6,5,7,9` | `3,1,1,1,1,6,5,7,9` | 1/1 | 0/0 | 5432/5432 |
| 13 / op3-06 | -9.277167 / 6.510259 | 1/1 | `3,1,0,1,1,3,4,6,5` | `3,1,0,1,1,3,4,6,5` | `3,1,0,1,1,3,4,6,5` | 1/1 | 0/0 | -2408/-2408 |
| 14 / op0-03 | 7.438479 / -7.749834 | 0/0 | `0,1,1,1,1,7,0,3,5` | `0,1,1,0,1,7,2,3,5` | `0,1,1,1,1,2,2,3,5` | 0/0 | 1/1 | none/none |
| 15 / op3-14 | -4.899231 / 3.33714 | 1/1 | `3,1,1,0,1,6,0,6,6` | `3,1,1,1,1,6,6,8,3` | `3,1,1,1,1,6,6,8,3` | 0/0 | 4/4 | none/none |
| 16 / op0-10 | 6.819091 / -9.742047 | 0/0 | `0,0,0,1,0,9,4,4,0` | `0,0,0,1,1,0,4,9,2` | `0,0,0,1,1,0,4,9,2` | 0/0 | 1/1 | none/none |
| 17 / op0-02 | 6.522698 / -5.652934 | 0/0 | `0,0,0,1,0,1,1,5,0` | `0,0,0,1,1,0,1,8,1` | `0,0,0,1,1,0,1,8,1` | 0/0 | 1/1 | none/none |
| 18 / op0-01 | 6.420133 / -9.575922 | 0/0 | `0,0,0,1,0,4,2,4,0` | `0,0,0,1,1,2,4,4,6` | `0,0,0,1,1,2,4,4,6` | 0/0 | 1/1 | none/none |
| 19 / op2-03 | -7.207268 / 5.538573 | 1/1 | `2,0,1,1,1,2,6,6,2` | `2,0,1,1,1,2,6,6,2` | `2,0,1,1,1,2,6,6,2` | 1/1 | 0/0 | 88/88 |
| 20 / op2-01 | -6.929545 / 4.316808 | 1/1 | `2,1,0,1,1,9,8,2,3` | `2,1,0,1,1,9,8,2,3` | `2,1,0,1,1,9,8,2,3` | 1/1 | 0/0 | -121/-121 |
| 21 / op1-15 | 0.6452693 / -1.273591 | 0/0 | `1,1,0,0,0,6,0,6,0` | `1,1,0,1,1,6,9,6,2` | `1,1,0,1,1,6,9,6,2` | 0/0 | 1/1 | none/none |
| 22 / op3-13 | 1.005005 / -1.066213 | 0/0 | `3,0,0,0,0,4,0,9,8` | `3,0,0,1,1,4,4,9,8` | `3,0,0,1,1,4,4,9,8` | 0/0 | 1/1 | none/none |
| 23 / op1-10 | -1.476739 / -1.021066 | 1/1 | `1,0,0,1,0,9,3,9,0` | `1,0,0,1,0,9,3,9,0` | `1,0,0,1,0,9,3,9,0` | 1/1 | 0/0 | 48/48 |
| 24 / op1-04 | -7.416594 / 4.978025 | 1/1 | `1,1,1,1,1,8,6,4,7` | `1,1,1,1,1,8,6,4,7` | `1,1,1,1,1,8,6,4,7` | 1/1 | 0/0 | -142/-142 |
| 25 / op1-01 | 0.211953 / -2.580093 | 0/0 | `1,0,0,1,0,3,1,2,0` | `1,0,0,1,0,3,1,2,0` | `1,0,0,1,0,3,1,2,0` | 1/1 | 1/1 | none/none |
| 26 / op2-00 | -5.743542 / 4.240571 | 1/1 | `2,0,0,1,1,3,5,4,9` | `2,0,0,1,1,3,5,4,9` | `2,0,0,1,1,3,5,4,9` | 1/1 | 0/0 | -41/-41 |
| 27 / op0-15 | 7.592103 / -7.348145 | 0/0 | `0,0,0,1,0,4,0,5,0` | `0,0,0,1,1,9,2,4,6` | `0,0,0,1,1,9,1,4,6` | 0/0 | 1/1 | none/none |
| 28 / op1-14 | 4.708774 / -5.187381 | 0/0 | `1,0,0,0,0,4,0,9,0` | `1,0,0,1,1,4,9,9,9` | `1,0,0,1,1,4,9,9,9` | 0/0 | 1/1 | none/none |
| 29 / op2-11 | -5.013523 / 3.331103 | 1/1 | `2,0,0,1,1,8,3,8,9` | `2,0,0,1,1,8,3,8,9` | `2,0,0,1,1,8,3,8,9` | 1/1 | 0/0 | -60/-60 |
| 30 / op3-02 | -3.925686 / 3.150455 | 1/1 | `3,0,0,1,1,0,8,4,1` | `3,0,0,1,1,0,8,4,1` | `3,0,0,1,1,0,8,4,1` | 1/1 | 0/0 | 1120/1120 |
| 31 / op3-11 | -7.633712 / 5.614641 | 1/1 | `3,1,0,1,1,9,3,5,3` | `3,1,0,1,1,9,3,5,3` | `3,1,0,1,1,9,3,5,3` | 1/1 | 0/0 | -1365/-1365 |
| 32 / op0-07 | 7.350234 / -10.25764 | 0/0 | `0,0,0,1,0,9,3,4,0` | `0,0,0,1,1,0,3,9,7` | `0,0,0,1,1,0,3,9,7` | 0/0 | 1/1 | none/none |
| 33 / op3-10 | -5.741617 / 3.114858 | 1/1 | `3,0,0,1,1,0,3,4,2` | `3,0,0,1,1,0,3,4,2` | `3,0,0,1,1,0,3,4,2` | 1/1 | 0/0 | 720/720 |
| 34 / op2-15 | 2.993893 / -3.631373 | 0/0 | `2,0,0,0,0,2,0,9,0` | `2,0,0,1,1,2,1,9,9` | `2,0,0,1,1,2,1,9,9` | 0/0 | 1/1 | none/none |
| 35 / op1-09 | -9.052313 / 5.710165 | 1/1 | `1,1,1,1,1,6,6,5,6` | `1,1,1,1,1,6,6,5,6` | `1,1,1,1,1,6,6,5,6` | 1/1 | 0/0 | -131/-131 |
| 36 / op0-09 | 8.680503 / -8.976891 | 0/0 | `0,0,0,0,1,9,0,9,6` | `0,0,0,0,1,9,0,9,6` | `0,0,0,0,1,9,0,9,6` | 1/1 | 1/1 | none/none |
| 37 / op3-07 | -6.805949 / 4.196833 | 1/1 | `3,0,1,1,1,5,2,2,3` | `3,0,1,1,1,5,2,2,3` | `3,0,1,1,1,5,2,2,3` | 1/1 | 0/0 | -800/-800 |
| 38 / op1-02 | -7.739315 / 5.781358 | 1/1 | `1,1,1,1,1,8,4,9,4` | `1,1,1,1,1,8,4,9,4` | `1,1,1,1,1,8,4,9,4` | 1/1 | 0/0 | -97/-97 |
| 39 / op2-13 | -3.211973 / 2.035927 | 1/1 | `2,1,1,0,1,0,0,3,8` | `2,1,1,1,1,8,1,3,3` | `2,1,1,1,1,8,1,3,3` | 0/0 | 4/4 | none/none |
| 40 / op1-13 | 6.933918 / -7.21235 | 0/0 | `1,0,0,0,0,7,0,9,0` | `1,0,0,0,0,4,0,9,0` | `1,0,0,0,0,4,0,9,0` | 0/0 | 1/1 | none/none |
| 41 / op1-11 | 1.144563 / -1.016289 | 0/0 | `1,0,0,1,1,5,5,5,1` | `1,0,0,1,1,5,5,5,1` | `1,0,0,1,1,5,5,5,1` | 1/1 | 1/1 | none/none |
| 42 / op2-02 | -6.359311 / 4.705392 | 1/1 | `2,0,0,1,1,3,5,3,6` | `2,0,0,1,1,3,5,3,6` | `2,0,0,1,1,3,5,3,6` | 1/1 | 0/0 | -10/-10 |
| 43 / op1-05 | -6.511885 / 4.825421 | 1/1 | `1,1,1,1,1,2,5,1,2` | `1,1,1,1,1,2,5,1,2` | `1,1,1,1,1,2,5,1,2` | 1/1 | 0/0 | -73/-73 |
| 44 / op0-05 | 7.573726 / -6.869217 | 0/0 | `0,0,1,1,0,1,7,5,0` | `0,0,0,1,1,8,0,8,7` | `0,0,0,1,1,8,1,8,7` | 0/0 | 1/1 | none/none |
| 45 / op3-12 | 3.279174 / -3.546979 | 0/0 | `3,0,0,0,1,9,0,5,2` | `3,0,0,1,0,2,9,5,0` | `3,0,0,1,0,2,9,5,0` | 0/0 | 1/1 | none/none |
| 46 / op2-07 | -3.930087 / 1.903374 | 1/1 | `2,0,0,1,1,3,1,5,6` | `2,0,0,1,1,3,1,5,6` | `2,0,0,1,1,3,1,5,6` | 1/1 | 0/0 | -52/-52 |
| 47 / op2-06 | -7.425426 / 4.323642 | 1/1 | `2,1,0,1,1,7,6,3,1` | `2,1,0,1,1,7,6,3,1` | `2,1,0,1,1,7,6,3,1` | 1/1 | 0/0 | -80/-80 |
| 48 / op3-01 | -8.275208 / 5.640718 | 1/1 | `3,1,0,1,1,2,3,4,6` | `3,1,0,1,1,2,3,4,6` | `3,1,0,1,1,2,3,4,6` | 1/1 | 0/0 | -2048/-2048 |
| 49 / op0-12 | 6.689307 / -6.523777 | 0/0 | `0,1,0,1,0,1,0,5,0` | `0,0,0,0,1,7,0,0,5` | `0,0,0,0,1,7,0,0,5` | 0/0 | 1/1 | none/none |
| 50 / op0-00 | 8.751261 / -9.292187 | 0/0 | `0,0,0,1,1,7,3,4,1` | `0,0,0,0,1,7,3,4,1` | `0,0,0,0,1,2,3,4,1` | 0/0 | 1/1 | none/none |
| 51 / op0-06 | 8.070323 / -8.341307 | 0/0 | `0,0,1,0,1,9,0,7,7` | `0,0,1,0,1,9,0,7,7` | `0,0,1,0,1,9,0,7,7` | 1/1 | 1/1 | none/none |
| 52 / op0-11 | 6.483773 / -5.727235 | 0/0 | `0,0,1,1,0,6,3,5,0` | `0,0,1,0,1,5,6,4,3` | `0,0,1,1,1,5,6,4,3` | 0/0 | 1/1 | none/none |
| 53 / op3-05 | -5.200031 / 4.245622 | 1/1 | `3,0,0,1,1,5,7,1,1` | `3,0,0,1,1,5,7,1,1` | `3,0,0,1,1,5,7,1,1` | 1/1 | 0/0 | 825/825 |
| 54 / op2-08 | -7.206684 / 4.987571 | 1/1 | `2,1,0,1,1,1,2,9,8` | `2,1,0,1,1,1,2,9,8` | `2,1,0,1,1,1,2,9,8` | 1/1 | 0/0 | -110/-110 |
| 55 / op3-00 | -4.762361 / 3.794937 | 1/1 | `3,0,0,1,1,1,5,2,1` | `3,0,0,1,1,1,5,2,1` | `3,0,0,1,1,1,5,2,1` | 1/1 | 0/0 | 612/612 |
| 56 / op3-15 | -2.904338 / 1.67075 | 1/1 | `3,0,1,0,0,7,0,6,0` | `3,0,1,1,1,7,2,6,1` | `3,0,1,1,1,7,2,6,1` | 0/0 | 4/4 | none/none |
| 57 / op2-14 | 4.202468 / -4.944421 | 0/0 | `2,0,0,0,0,5,0,1,0` | `2,0,0,1,1,5,1,1,2` | `2,0,0,1,1,5,1,1,2` | 0/0 | 1/1 | none/none |
| 58 / op2-10 | -5.511277 / 3.351737 | 1/1 | `2,1,0,0,1,5,0,0,8` | `2,1,0,0,1,5,0,0,8` | `2,1,0,0,1,5,0,0,8` | 1/1 | 0/0 | -85/-85 |
| 59 / op1-06 | -8.249532 / 5.253006 | 1/1 | `1,1,1,1,1,3,7,5,4` | `1,1,1,1,1,3,7,5,4` | `1,1,1,1,1,3,7,5,4` | 1/1 | 0/0 | -118/-118 |
| 60 / op3-08 | -6.694559 / 4.625519 | 1/1 | `3,1,1,1,1,2,1,3,1` | `3,1,1,1,1,2,1,3,1` | `3,1,1,1,1,2,1,3,1` | 1/1 | 0/0 | 156/156 |
| 61 / op1-08 | -2.928 / 1.756818 | 1/1 | `1,1,0,1,0,5,2,7,0` | `1,1,0,1,0,5,2,7,0` | `1,1,0,1,0,5,2,7,0` | 1/1 | 0/0 | -18/-18 |
| 62 / op2-04 | -3.787549 / 1.482291 | 1/1 | `2,0,0,1,0,3,6,9,0` | `2,0,0,1,0,3,6,9,0` | `2,0,0,1,0,3,6,9,0` | 1/1 | 0/0 | 54/54 |
| 63 / op0-04 | 6.509298 / -9.219766 | 0/0 | `0,0,0,1,0,6,8,4,0` | `0,0,0,0,1,5,8,6,7` | `0,0,0,1,1,5,8,6,7` | 0/0 | 1/1 | none/none |
