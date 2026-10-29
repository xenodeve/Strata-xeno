# I2 single warm-start result: matched pass, retention fail

Issue #276. The one approved warm-start fit completed all 840 finite updates.
It passes the matched development gate but **fails the disjoint legacy
regression gate: 22 previously correct compute rows become wrong**. The new
artifact is retained separately and is not promoted. I2 stays open.

This is input-mapper evaluation on real saved native first-attention activations,
not live model generation, arithmetic-core execution or a Hub benchmark. No second
fit, earlier-checkpoint choice, threshold change or acceptance64 access occurred.

## Matched result and prior exposure

| Head on the identical matched development rows | Raw exact compute | Quote-copy false-call candidates |
| --- | ---: | ---: |
| Retained original | 264/384 | 112/384 |
| Earlier fresh fixed fit | 24/384 | 0/384 |
| This copied-head warm fit | **384/384** | **0/384** |

Warm train is also 384/384 compute and 0/384 false-call candidates. On development,
all operators, signs, lengths, active digits and canonical padding are correct:
384/384, 768/768, 768/768, 1152/1152, 384/384 respectively. The separately named
predicted-length padding counterfactual is identical here and rescues no tuple.

| Matched development signed-pair exposure in OLD training | Retained exact | Warm exact | Retained / warm quote-copy false calls |
| --- | ---: | ---: | ---: |
| 19 previously exposed pairs | 228/228 | 228/228 | 64/228;0/228 |
| 13 pairs not in old training | 36/156 | 156/156 | 48/156;0/156 |

The 13 pairs are disjoint from both relevant train pair sets, but the whole
matched development collection has already been inspected during diagnosis.
These are diagnostic adaptation results, not untouched generalization or
deployment evidence. The fitter's `eligible_for_fresh_evaluation=True` flag
only summarizes its matched train/development checks; it does not include the
separate retention gate and does not authorize opening acceptance64.

## All 343 disjoint legacy development rows

Before the fit, recompute all ordered 343 rows from the frozen old cache: 259
compute and 84 negative rows, excluding ALL 96 matched train/development signed
pairs. After fitting, recompute both retained and candidate with the same
F32/eval/no-grad batch 64 settings. Retained before/after vectors match exactly;
the before vectors also match the historical saved predictions.

| Legacy metric | Retained before and after | Warm final |
| --- | ---: | ---: |
| Raw exact compute tuple | 206/259 | 234/259 |
| Negative false-call candidates | 55/84 | 34/84 |
| Operator on compute | 259/259 | 259/259 |
| Signs | 518/518 | 510/518 |
| Lengths | 518/518 | 518/518 |
| Active digits | 850/886 | 869/886 |
| Canonical unused tens | 105/150 | 149/150 |
| Separately composed-padding exact tuple | 223/259 | 235/259 |

Raw paired compute outcomes: 184 both correct, 50 newly correct, 22 newly wrong,
3 both wrong. Negative outcomes: 29 both no-call correct, 21 false calls fixed,
34 still false calls, **zero new false-call candidates**. Aggregate improvements
do not compensate for the 22 regressions under the frozen gate. The negative
non-regression condition passes; operand retention fails.

The 22 regressions contain 23 changed fields: lhs sign 5, lhs tens 7 (one is unused
padding), rhs ones 6 and rhs tens 5. Operators and lengths remain correct on all
259 compute rows. Example: `(-1) + (97)` originally gives the exact operand
fields; warm predicts rhs tens 0 instead of 9 while still predicting length 2.
The executor must reject that inconsistent tuple, not publish an arithmetic
result from it. The appendix records every regressed row and field change.

These are training-induced changes on the same frozen inputs with the retained
control stable. They establish a retention failure in this adaptation. They
do not by themselves identify which shared submodule caused it. No padding
repair or gate relaxation is adopted.

## Exactly one copied initialization and fit

Implementation commit `fc03fa7b3a8fbbcbd96041eaf77ea94dd08030da` adds the
explicit offline initialization guards and freezes the
[approved execution contract](../plans/2026-10-29-igc-warm-start-experiment.md).
The source artifact is bounded, SHA-bound safetensors with exact F32 tensor
names/shapes/finiteness. All tensors validate before copying, with no source
storage assignment. Source bytes are checked again before success.

The default fresh seeded fingerprint remains
`7f363077fe63297aeabefd9d38b9436332fbb3e1805f1ec849f9d0fa665d6547`.
Loaded warm initial fingerprint is
`c30c061edcd438bf8fbf15c18b9f6dcf6cd18e2210a37c0da65f43ffcbb8f943`.
Pricing and the separate fit agree on this fingerprint, all first 64 training IDs,
full source/cache/protocol identity and complete recipe. No old optimizer state
is loaded; AdamW begins with zero state entries. Both original head artifacts
remain unchanged.

Relative to the failed fresh fit, only initial learned weights change. Seed
20261031, 768 TRAIN rows, 70 epochs, batch 64, 840 updates, 96,964 parameters,
AdamW lr 0.001/wd 0.01/foreach=False/fused=False, cosine to 0.00001, F32, four CPU
threads, deterministic/math attention and TF 32 off stay fixed. No development
checkpoint selection; final checkpoint only. Original CUDA Python is
PyTorch 2.11.0+cu 130 on the primary RTX 5060 Ti. The loaded cache is native
first-attention output, not token-ID substitution or final-head activations.

Pricing runs exactly one training-only backward batch with no optimizer
constructed/steps=0. Initial loss 3.88738656 is finite. The recorded backward
phase is 2.185 s, excluding final cache rehash; CUDA allocated peak 699,594,752 B,
process lifetime private peak 4,001,808,384 B. A new process resets initial weights
and order before the separate fit.

Fit plus final in-memory evaluation takes 132.998 s; epoch mean loss 0.99146376
falls to 0.00014989. All 840 loss/gradient/parameter/optimizer checks pass without
clipping. CUDA allocated/reserved peaks 728,945,664/918,552,576 B. Across 14 resource
snapshots, largest observed process lifetime private peak 4,274,294,784 B;
minimum system commit headroom 20,892,639,232 B and physical available 20,621,381,632 B.
Samples do not cover every instant. These timings are offline phase observations,
not profiler-off paired Hub speed claims.

Fit phase bound 1800 s/outer 1920 s; no timeout or numerical failure. The before/after
evaluation phases take 53.835 s/65.721 s, including large-cache identity checks;
both finish within 300 s phase/360 s outer bounds. They are not serving latency.
After loading serialized final weights, every one of 1536 matched categorical
predictions matches the saved final array. No optimizer/backward runs in these
evaluations. Before/after all 343 rows and both 1536-row matched heads use the same
math/batch configuration; raw and padding-composed vectors are retained separately.

## Safeguards, reviews and evidence

Tests pass 50/50 on default CPU Python and 50/50 on the qualified CUDA Python.
The new initialization tests reject wrong digest, extra/missing names including
optimizer payload, wrong shape/dtype, nonfinite tensors and oversized files;
they prove exact copy, source immutability, fresh moments, default no-op and
initialization-bound pricing refusal. Existing guard tests use tiny synthetic
gradient fixtures, separate from the single real fit.

The first expected-red run lacked the new module. The first green attempt hit
a native safetensors fixture-writer I/O permission error; writing the same safe
serialized fixture bytes through the permitted file API fixes the fixture without
changing production loading or system permissions. Initial preflight aborted at
a missing optional psutil import, before model/data work; a read-only built-in
Windows inventory replaced that dependency, without installing anything. Both
failed receipts remain. No training retry occurred.

Independent review before execution closed two evaluator provenance gaps:
full expected recipe/source/cache/reset equality, and entry/exit binding of the
actual parsed plan plus phase-generated inputs. The final plan binds 16 sources
and 17 small inputs; full state bytes are checked before/after use. Original
cache data, retained and failed-fresh artifacts remain unchanged. No heavy job
overlap, cloud spending, process memory cap, pagefile/profile change, unrelated
process termination, push or backup upload occurred. Every owned launcher reached
DONE 0 and drained. No full engine/HIP/SYCL/live model validation was run for this
offline Python change.

| Artifact | SHA256 |
| --- | --- |
| Immutable retained source weights | `519fbc196993c94676f4031f7bd45ad9abf661f8b7b4fd3b9d6358204b5294e2` |
| Prior failed fresh weights, preserved | `29abae54c429f881d128ff049edfb6d8ae40de19e363583553f813c3564ac1eb` |
| New warm final weights, not promoted | `ed889ddedf9d7886969c374e443bf372fb1fcfaec6c85108d38a0ddd9c2f09ad` |
| New pricing receipt | `c55aff92fb2502884eca950cfd27cff2d6179cbfdc9dc0e943a966e2e25961ad` |
| New fit report | `f969280b74334f478a4afbd4a70179a3b565abac58eb0d98edd8eefa1a7d489a` |
| Fit raw categorical predictions | `be1c250281f29c0ddcbb056df026223072c4f8c9a00ce5b5fd125f4bfc710ce8` |
| Frozen execution plan 02 | `299a4ede3570e4bfedbaa72bf007e2a34185b344055e2a9f9c1d77a3b14f85c7` |
| Before-regression report | `895899dad5bf19177535c0e0373493cb4eca12c5c4759148328d81a08eba48b0` |
| After/reload report | `6d86d43ff867ac3fcb464ebf509b35c5957c1275cac79c595c550c972b8a7345` |
| After raw/padding predictions | `3fbb907f915d0e1f0032bab6dc19d9097a512a2df615fc15299d61a0fd84147d` |

Local evidence basenames: `igc-warm-execution-plan-02.json`,
`igc-warm-preflight-02.json`, `igc-warm-{price,fit}-command-01.json`,
`igc-warm-training-price-01/price.json`,
`igc-matched-warm-fit-01/{report,predictions,input.safetensors}`,
`igc-warm-evaluation-{before,after}-01/{report,predictions}.json`, and the
matching headless/test logs. Large caches, weight files and private absolute
paths are not published or added to Git.

## Terminal decision and one next step

Do not promote the warm head: matched passes, retention fails. Preserve this
complete final result rather than choosing another epoch or rerunning the seed.
The original controller/head remains the candidate in use, and acceptance64 is
still sealed. I2, live result use and actual Hub end-to-end benchmarking remain open.

One evidence-grounded next step is to prepare a **single retention-aware replay
protocol** using old TRAIN activations alongside matched TRAIN, while keeping
all 343 legacy development rows and matched development evaluation-only. This
requires excluding all 32 matched development signed pairs from old-TRAIN
replay too: 19 were historically present there. Validate pair-level exclusion,
not only row IDs, before selecting replay examples.
This
targets the observed training-induced regressions without changing architecture,
thresholds or numerical contracts. Freeze the replay selection, objective,
update budget and resource cost before another reviewed fit; do not train on
the 22 regressed development rows or start that fit automatically.

## Appendix: all 22 previously correct rows that regressed

Field differences are between the frozen target and warm final prediction;
the retained prediction equals the complete target for each row. Length is the
encoded digit-count-minus-one. Row IDs refer to existing inspected old development.

| Row ID | Expression | Wrong field: expected -> actual |
| --- | --- | --- |
| `new-values-new-templates-op1-0022` | `(-1) + (97)` | rhs.tens: 9 -> 0 |
| `seen-values-new-templates-op3-0025` | `(54) * (36)` | lhs.tens: 5 -> 0 |
| `new-values-new-templates-op3-0014` | `(-61) * (97)` | rhs.tens: 9 -> 0 |
| `new-values-seen-templates-op1-0025` | `(-2) + (83)` | lhs.sign: 1 -> 0 |
| `seen-values-new-templates-op1-0021` | `(-74) + (-56)` | lhs.tens: 7 -> 0 |
| `new-values-new-templates-op2-0020` | `(-61) - (-83)` | lhs.sign: 1 -> 0 |
| `seen-values-new-templates-op2-0003` | `(54) - (73)` | lhs.tens: 5 -> 0 |
| `new-values-new-templates-op1-0007` | `(-82) + (17)` | rhs.ones: 7 -> 1 |
| `new-values-seen-templates-op3-0006` | `(-43) * (-7)` | rhs.ones: 7 -> 3 |
| `new-values-seen-templates-op1-0020` | `(-83) + (-43)` | lhs.sign: 1 -> 0 |
| `seen-values-new-templates-op1-0026` | `(-54) + (4)` | lhs.tens: 5 -> 0 |
| `new-values-seen-templates-op3-0000` | `(-97) * (0)` | lhs.sign: 1 -> 0 |
| `new-values-new-templates-op1-0001` | `(54) + (-29)` | lhs.tens: 5 -> 0 |
| `new-values-new-templates-op3-0030` | `(-17) * (37)` | rhs.ones: 7 -> 3; rhs.tens: 3 -> 0 |
| `new-values-new-templates-op3-0020` | `(29) * (17)` | rhs.tens: 1 -> 0 |
| `seen-values-new-templates-op3-0012` | `(4) * (-26)` | lhs.tens: 0 -> 4 |
| `new-values-new-templates-op2-0031` | `(54) - (-83)` | lhs.tens: 5 -> 0 |
| `new-values-new-templates-op1-0030` | `(-83) + (1)` | lhs.sign: 1 -> 0 |
| `new-values-new-templates-op2-0002` | `(97) - (-97)` | rhs.tens: 9 -> 0 |
| `new-values-new-templates-op3-0023` | `(-17) * (17)` | rhs.ones: 7 -> 1 |
| `seen-values-new-templates-op2-0002` | `(62) - (27)` | rhs.ones: 7 -> 2 |
| `new-values-new-templates-op1-0031` | `(-2) + (-17)` | rhs.ones: 7 -> 1 |
