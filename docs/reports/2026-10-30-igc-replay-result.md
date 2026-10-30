# I2 mixed replay: both frozen development gates pass

Issue #276. The single authorized replay fit completed at source commit
`78e44b7bc3fca2cdef89f22db4ca7e4010325538` (local, unpushed). It passes the
predeclared matched-development and legacy-retention gates. **No artifact is
promoted.** This is an input mapper evaluated on saved first-attention states;
it is not live Gyro inference, arithmetic execution, an end-to-end IGC result,
or completion of I2.

## Results and the remaining trade-off

All scores below use raw categorical predictions, including canonical padding.
An exact positive tuple requires the correct operator, both signs, both lengths
and all four digit slots. Negative false calls count a nonzero predicted
operator on a no-call prompt; they are candidates at this head, not observed
controller invocations.

| Inspected development cohort | Original retained | Previous unpromoted warm fit | Replay final |
| --- | ---: | ---: | ---: |
| Matched positive exact tuples | 264/384 | 384/384 | **384/384** |
| Matched negative false-call candidates | 112/384 | 0/384 | **0/384** |
| Legacy positive exact tuples | 206/259 | 234/259 | **259/259** |
| Legacy negative false-call candidates | 55/84 | 34/84 | **42/84** |
| Legacy previously correct tuples lost, versus original retained | — | 22 | **0** |
| Legacy new false-call candidates, versus original retained | — | 0 | **0** |

The replay fixes 53 previously wrong legacy compute rows and 13 negative false
calls relative to the original retained model. The 206 previously correct
compute rows remain correct; the 29 previously correct no-call rows remain
no-call. There are no remaining wrong compute tuples on these 259 rows.

The gate's reference is the original retained artifact. Passing that gate does
not mean the replay dominates every experimental alternative: compared with the
unpromoted warm fit, it fixes 25 compute rows but introduces **8 additional
false-call candidates**, with none of that fit's false calls fixed. All 42
remaining negative errors are label-only prompts: 19 English `dev-en-label`
and 23 Thai `dev-th-label`. This unresolved intent behavior prevents a general
readiness claim and is preserved in the per-row evidence.

The full original matched TRAIN also scores 384/384 positives and 0/384 negative
false calls. It contains 768 rows, of which only 672 entered this replay; this
training score is not an independent evaluation.

### Representative input-mapper outputs

For `Work out 93 * -3 and give its value.`, the final predicted tuple is
`[3, 0, 1, 1, 0, 3, 9, 3, 0]`, exactly the expected multiplication operator,
signs, one/two-digit length flags and little-endian digits for operands 93 and
-3. This verifies extraction of the request, **not** computation of the product.

For `Treat 0+-84 as a label. Only acknowledge it.`, the expected operator is
no-call (0), but the final head predicts add (1). This is one of the 42 retained
intent failures. Correct operand digits do not make this a valid invocation.

### Field and exposure checks

On matched-development compute rows, the candidate gets operators 384/384, signs 768/768,
lengths 768/768, active digits 1,152/1,152 and padding 384/384. On legacy compute,
the corresponding counts are 259/259, 518/518, 518/518, 886/886 and 150/150.
The separately declared predicted-length padding-composition counterfactual
does not rescue any candidate row: its scores equal the raw scores. It is not
enabled as a runtime repair. For the retained model, that variant raises legacy
compute from 206/259 to 223/259; historical raw results remain unchanged.

Inherited exposure from original retained initialization is not erased by
filtering this replay. Matched development has 32 independent signed pairs,
with the following distinct views (the rows and categories are not additive):

| Exposure to OLD TRAIN before replay | Candidate positive exact | Candidate negative false calls |
| --- | ---: | ---: |
| 19 exact ordered pairs seen | 228/228 | 0/228 |
| 13 exact ordered pairs unseen | 156/156 | 0/156 |
| 20 pairs with any equivalent arithmetic fact seen | 240/240 | 0/240 |
| 12 pairs with no equivalent arithmetic fact seen | 144/144 | 0/144 |
| Individual rows whose fact was seen | 164/164 | 0/164 |
| Individual rows whose fact was unseen | 220/220 | 0/220 |

The legacy evaluation also inherits 43 unordered-pair and 40 fact overlaps with
OLD TRAIN, despite zero exact ordered-pair overlap. Both development sets have
been inspected during diagnosis. These are descriptive counts, not fresh
acceptance or independent token-level statistical evidence. Acceptance64 stayed
sealed; no thresholds, weights or policies were selected from its contents.

## Fixed recipe and batch-bounded input path

The [prepared replay](2026-10-30-igc-replay-preparation.md) and
[execution protocol](../plans/2026-10-30-igc-replay-execution.md) were frozen before
measurement. The head has 96,964 parameters, uses first-attention context,
positions and masks, and has no final-query shortcut. It was copied from the
**original retained** artifact, not the unpromoted warm artifact. Seed 20261031,
fresh AdamW (learning rate 0.001, weight decay 0.01), cosine to 0.00001, 840
updates, batch 64 and final checkpoint only were kept unchanged.

The 10,516 selected member references comprise 9,844 OLD TRAIN rows and 672
matched TRAIN rows. Every batch has 32 matched rows and 8 old rows from each of
no-call/add/subtract/multiply. This gives 53,760 presentations, 26,880 from each
source. Matched rows appear 40 times each; old rows appear 2 or 3 times. This is
a data-mixture/exposure change, not an initialization-only comparison.

Selected training has zero intersection with all 768 matched-development and
343 legacy rows on the declared ordered pair, unordered pair, parsed expression,
canonical arithmetic fact and normalized-prompt keys. No regression label is
added to training. Equal scalar answers to different problems are not treated
as equivalent expressions.

`MixedContextInputs` gathers into a caller-owned batch buffer. Selected unique
row payload totals about 34.46 GB; the two complete existing caches contain
12,160 rows, about 39.85 GB of row payload plus headers. Neither is preloaded
into RAM or uploaded.
The actual run checks exact finite F32 values, masks and positions for 134
mixed/duplicate-selected rows against direct source-array reads, and verifies
the zero query. This uses value equality, not a signed-zero bit comparison.
Source/index metadata only locates rows and never becomes a neural input.

## Pricing, admission and the one fit

The pricing process started 2026-10-30 00:32:31 UTC. It gathered the full 840-batch
schedule and ran eight fixed TRAIN-only backward batches on a disposable copy.
It constructed no optimizer and performed zero optimizer steps. The copied
parameter fingerprint stayed unchanged. Pricing is not an extra training run.

| Scope | Measured seconds |
| --- | ---: |
| Pricing initial validation | 35.692 |
| Pricing schedule loop | 77.459 |
| Pricing loop plus final checks | 111.211, within 120 |
| Pricing CPU gather total | 72.670 |
| Initial conservative fit / owned-outer estimates | 839.358 / 890.050 |
| Refreshed fit / owned-outer estimates at fit launch | 843.100 / 895.663 |
| Fit initial validation | 37.563 |
| Actual 840-update fit loop | 161.366 |
| Fit through final matched predictions | 164.573 |
| Fit phase including final rechecks | 197.186, within 1,800 |
| Retained before evaluation | 64.038 |
| Serialized after evaluation | 65.844 |

The fit started 00:35:49 UTC and completed under its unchanged 1,920-second owned
outer timeout. All four owned wrappers ended `DONE exit=0`. The before/after
evaluations include cache validation; none of these figures is a Hub latency or
optimization benchmark.

Admission conservatively budgets twice measured full-schedule gather time,
the larger of four times the historical same-shape fit and twice projected
cold/warm backward time, twice current validation and a 60-second save allowance.
The outer estimate adds validation and 15 seconds; it also checks actual elapsed
outer prefix. This is a planning estimate, not a guaranteed duration. Current
physical/commit/GPU availability was checked before starting. No artificial
private-commit cap, system setting change or full-model load was introduced.

Training completed 840 finite updates with zero initial optimizer-state entries
and no transferred moments. Loss started at 1.58381665 and ended at 0.000183785;
the first/last 42-update means were 0.28973805 / 0.000181519. Loss is a diagnostic,
not the acceptance criterion. There was no clipping, best-checkpoint selection,
additional seed, retry or second fit.

### Measured resource scope

On the primary RTX 5060 Ti, PyTorch 2.11.0+cu130 with math attention and TF32 off:

| Quantity | Pricing | Fit |
| --- | ---: | ---: |
| CUDA allocated peak | 910,011,392 B | 910,801,920 B |
| CUDA reserved peak | 952,107,008 B | 954,204,160 B |
| Process private-commit lifetime peak | 4.028 GiB | 4.214 GiB |
| Minimum sampled physical availability | 17.950 GiB | 16.869 GiB |
| Minimum sampled system-commit headroom | 19.011 GiB | 18.663 GiB |

Process peaks come from `GetProcessMemoryInfo`; system commitment is from
`GetPerformanceInfo`. System minima above are samples at 42-update intervals,
not continuous worst-case measurements. They do not establish that a full Gyro
Hub launch fits the current machine state. No source arrays, model weights or
unrelated processes were removed to admit this experiment.

## Verification and immutable evidence

The implementation suite passed **53/53** on CPU Python and **53/53** on the
qualified original CUDA Python before execution. It covers row/mask ordering,
duplicates, batch-buffer and index validation, binding/recipe tampering,
bounded receipt reads, fresh initialization/optimizer guards and admission
refusal. Final source and input hashes matched the frozen invocation.

After saving, both heads were evaluated without gradients on all 1,536 matched
rows and all 343 legacy rows. All 1,536 reloaded candidate categorical vectors
equal the saved final in-memory vectors. All retained vectors before and after
are identical. The original retained, failed-fresh and previous-warm artifacts
remain byte-identical. No stage loads or updates Gyro's learned weights.

A separate stdlib-only checker recomputed canonical tuple labels directly from
the saved manifests, raw scores, all per-row retention/false-call transitions,
training exclusion keys, presentation totals, original hashes and serialized
predictions. It used neither the training scorer nor cache arrays. Independent
read-only review found no discrepancy in the evidence or execution safeguards.
The checker's first assertion mistakenly applied the fit-reset flag to pricing;
inspection confirmed that flag is fit-only. That audit assertion was corrected,
with no change to run files, source, labels, predictions or score thresholds.

Local evidence basenames (under the task evidence root, not published):

- `igc-replay-execution-plan-02.json`, `igc-replay-price-command-01.json`,
  `igc-replay-fit-command-01.json` and their inventory receipts.
- `igc-replay-price-01/price.json`, `igc-replay-fit-01/admission.json`,
  `igc-replay-fit-01/{report,predictions}.json` and `input.safetensors`.
- `igc-replay-evaluation-before-01/{report,predictions}.json` and
  `igc-replay-evaluation-after-01/{report,predictions}.json`.
- `igc-replay-outcome-decomposition-01.json`,
  `igc_replay_outcome_check_01.py`, `igc-replay-outcome-qualification-01.json`.
- The four corresponding `igc-replay-*-01.log` terminal logs and
  `igc-replay-run-final-cpu-01.log`, `igc-replay-run-runtime-01.log`.

| Identity | SHA256 |
| --- | --- |
| Replay preparation bytes | `b89622b1b4e0e363ca0eb7dfdcc6c41b75f8758e50087e7b92183606b6354096` |
| Fixed batch schedule | `f0c3850b5a3653056354380cd732e8908833671952479e3cde812aaf74e27ec4` |
| Execution plan 02 | `59027ac64b4ddc842d08ed8e4410c6f9ddb8c25527df73a4bb5b8fd556f70d4a` |
| Original retained weights | `519fbc196993c94676f4031f7bd45ad9abf661f8b7b4fd3b9d6358204b5294e2` |
| Replay final weights, 391,040 B | `3cb5a645c3a9d74778d807672d5fe0dd4d4d025d7992e729cc002d816828043d` |
| Pricing receipt | `c1b46a29f44a6b3c5f065738d6268eee81efa10f50ab8916668cca2f24c5c690` |
| Fit report | `0ff00e0973a446fcff16df00c55aa3142210230b3735163ff6a52b6dc6a50f1a` |
| Final after-evaluation report | `5b1d3b39ee5f887b7740a92914d8dea347098459cfc2f706645d4a7af0e8f1b6` |
| Final after-evaluation predictions | `aa356b88a4d1e78872dd16d9529503bc094779bb30f7270471218e60523d665e` |
| Independent checker source | `a177db5ce153691e3619001f3221411194891a6e9f48c95c94a610bf77e3da8b` |
| Independent qualification receipt | `60d4c883d515536c160eb8bc52fa72d7a0a966558ab48e04382548372ad82b9a` |

Recorded commands with the qualified Python via the headless wrapper:

```text
igc_replay_launch_01.py price
igc_replay_launch_01.py fit
igc_replay_evaluate_01.py after 59027ac64b4ddc842d08ed8e4410c6f9ddb8c25527df73a4bb5b8fd556f70d4a
python -B -X utf8 igc_replay_outcome_check_01.py
```

These are evidence commands, not instructions to repeat the once-only fit. The
launcher binds its working tree and input hashes, refuses existing output or
command receipts, and uses `CUDA_VISIBLE_DEVICES=1,0`, unset
`CUDA_DEVICE_ORDER`, `CUBLAS_WORKSPACE_CONFIG=:4096:8` and a hidden owned child.

## Next dependency and limits

This closes the single replay experiment, not the whole capability roadmap.
The next useful dependency is to qualify this separate experimental head's
native export and replay with the existing readiness/consensus rules unchanged.
That must precede live request/result ownership tests and broader held-out
evidence. The 42 negative false-call candidates require explicit treatment in
that evaluation; passing a no-new-errors gate is not permission to loosen
readiness. No further fit, runtime replacement, sealed acceptance evaluation,
Hub default change, full-model launch, cloud use, push or deployment occurs at
this checkpoint.
