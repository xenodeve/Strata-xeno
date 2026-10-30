# I2 one retention-aware replay protocol

Issue #276. **Preparation only; no fit authorized by this document.** Parent
requested this one protocol after the [warm-start retention
failure](../reports/2026-10-29-igc-matched-warm-start.md). Return it for review
before implementing the mixed-cache fitter, pricing or training. All previous
artifacts remain immutable; acceptance64 stays sealed.

## Question and fixed starting point

Can old-TRAIN rehearsal retain the original readout's broader competence while
learning the paired matched intent/operand task? The warm fit solved matched
development but regressed 22 previously correct legacy rows. This protocol tests
a changed training distribution, not initialization alone and not a clean holdout.

Start from a COPY of the ORIGINAL retained contextual first-attention head:
SHA256 `519fbc196993c94676f4031f7bd45ad9abf661f8b7b4fd3b9d6358204b5294e2`,
fingerprint `c30c061edcd438bf8fbf15c18b9f6dcf6cd18e2210a37c0da65f43ffcbb8f943`.
Do not initialize from the failed fresh or warm final artifact. New AdamW with
empty moments; preserve strict copied-initialization/provenance guards. No
parameter freezing, architecture change, loss weighting, new gate or padding repair.

## Evaluation exclusion contract

Use only the two existing identity-pinned cached manifests. No new prompt,
activation, full model, acceptance generator or regression-label training.
Evaluation for exclusion is all 768 matched DEVELOPMENT rows (32 signed pairs)
plus the frozen 343 old DEVELOPMENT rows. The 343 IDs remain exactly unchanged.
Only metadata is used for exclusion; the warm model's errors do not select
training examples. No evaluation row enters a batch.

Every training candidate from BOTH sources is rejected on any of:

1. Same ordered signed operand pair, regardless of operation/intent/template.
2. Same unordered signed pair, conservatively excluding swapped subtraction too.
3. Same fully parsed binary expression after whitespace/parentheses/sign/decimal
   normalization. The parser accepts only bounded +/-99 operands and +,-,*;
   parsed expression must match metadata, including no-call rows.
4. Same canonical computational fact: addition sorts its signed terms;
   subtraction becomes addition of the negated rhs; multiplication sorts absolute
   factors and normalizes sign parity (zero has no sign). This handles reordered
   addition/multiplication and equivalent sign/subtraction forms.
5. Same NFKC/casefold/whitespace-normalized complete prompt.

These finite canonicalizations are the declared semantic-duplicate contract.
Equal scalar answers to different problems are not duplicates: 1+2 and 1*3 remain
different computational facts. This is not a general algebraic-equivalence
checker. Intent is deliberately ignored for expression/fact exclusion, so a
quote of an evaluation computation cannot become a training negative.

## Fixed candidate selection

| Source | Original TRAIN | Eligible after all exclusions | Selected |
| --- | ---: | ---: | ---: |
| OLD expanded-v 2 | 10,240 | 9,844 | 9,844 |
| Matched | 768 | 680 | 672 |

The old pool contains classes 0/1/2/3 with 2469/2462/2453/2460 rows respectively.
It has 5765 signed pairs,4651 positive-supervised pairs and 9763 distinct complete
prompts. Repeated prompts retain their original distinct row identities; row
count is not independent example count.

Matched eligibility leaves 57/56/57 complete expression groups for +,-,*.
With selection seed 20261113, shuffle sorted group keys and choose 56 groups per
operation. Keep all four conditions of each: English/Thai x compute/quote-copy.
That is 168 groups/672 rows,112 positives per operation and 336 negatives,59 signed
pairs. All required signed 3/8 values remain in both operand roles. The extra
8 eligible rows are excluded by this predeclared balance step. No prediction
score enters selection.

The old pool loses 396 rows; matched loses 88 on evaluation collision and another
8 for group balance. Exclusion reasons may overlap and are not additive. Both
selected sources have zero overlap against evaluation on every key above.

## Inherited exposure is not erased

Filtering future replay cannot remove knowledge already encoded in initialization.
Old adapter TRAIN has exact-pair exposure to 19/32 matched development pairs
(456 of 768 rows), canonical-fact exposure to 20 pairs (328 rows), and 2 identical
prompts. For legacy 343, exact ordered pair/expression/prompt exposure is zero,
but 43 rows have unordered-pair overlap and 40 have canonical-fact overlap with
old TRAIN. These categories overlap and must not be summed.

Report these inherited adapter-corpus exposures explicitly. Keep matched's
historical 19/13 exact-pair breakdown and add 20-fact-pair exposure where relevant;
never relabel this inspected development as untouched acceptance.

## Fixed future mixing and optimizer budget

Prepared file stores cache row locators and 840 exact 64-row batches. Each batch:
32 matched rows, then 8 old rows of each ready/operator class 0,1,2,3. Locators are
not neural features; a future consumer must gather only actual states/masks and
canonical targets from qualified TRAIN rows. Never feed source/index/ID/selection
metadata into the readout.

- Matched stream: seed 20261114; shuffle 672 selected rows once per cycle; exhaust
  each cycle before reshuffling.840 batches traverse 40 complete cycles, so each
  matched row appears 40 times.
- Old streams: separate seeds 20261115+class. Shuffle each eligible class pool,
  exhaust it before reshuffling; take 8 each batch. Every class contributes 6720
  presentations.2652 old rows appear twice and 7192 appear three times.
- Total 53,760 presentations: 26,880 matched and 26,880 old;20,160 negatives and
  11,200 positives per operation. These counts are fixed, not adaptively reweighted.

Keep model seed 20261031, batch 64,840 updates,96,964 parameters, AdamW lr 0.001,
wd 0.01, foreach=False/fused=False, cosine to 0.00001, F32, four CPU threads,
deterministic/math attention and TF 32 off in the qualified CUDA Python. Same
loss: operator CE on all rows, each operand CE on positive rows including padding.
No change to the canonical ABI.

The update and total-presentation budgets equal the previous fit, but matched
exposure changes from 70 presentations for each of 768 rows to 40 for each of 672.
There are 40 matched cycles, not 70 epochs. Data filtering, mixing and effective
objective distribution change together; do not describe this as initialization-only.

## Required future implementation and bounded execution

Current preparation does not make the single-cache fitter consume two caches.
Before any reviewed run, implement/test a bounded mixed-source gather and
recipe/reset receipt binding to this exact file, schedule, member IDs, both
source headers/masks/states, initializer and protocol. Resolve each member's
cache index back to its pinned ID and TRAIN metadata. Verify all exclusion and
schedule guards again; do not trust editable locators alone.

Gather one preallocated 64-row batch, avoiding whole-cache loading and unnecessary
concatenation copies. Preserve canonical normalization, masks, positions, model
math and finite-update guards. Add any missing I/O/progress spans before making
performance claims. No automatic fit follows this preparation.

If later reviewed: one TRAIN-only backward pricing batch with zero optimizer
steps, then a separate process resets initializer/optimizer/order for ONE fit.
Final checkpoint only, no intermediate development selection, sweeps, retries or
extra seeds. Proposed pricing 120 s phase/240 s outer; fit 1800 s phase/2100 s outer.
The larger outer allowance versus 1920 s budgets validation of the additional old
cache; it is a safety bound, not a predicted run time. Record the actual bound
semantics before execution. Timeout/nonfinite/resource emergency stops and retains
the affected run; no partial artifact is eligible.

Fresh offline admission remains physical 8 GiB, system commit headroom 12 GiB,
primary GPU 2 GiB, output 32 MiB plus 4 GiB disk reserve. Keep runtime emergency floors
physical 1 GiB/commit 2 GiB. These are disclosed offline checks, never Hub limits or
a private-commit cap. No pagefile/security/profile changes or unrelated cleanup.

## Unchanged acceptance and reporting

- All matched DEVELOPMENT 384 compute rows raw exact and 0/384 quote-copy false
  calls. Preserve train/per-field/language/template/exposure and pair-level results.
- All 343 legacy rows evaluated before and after. No previously correct RAW
  compute row becomes wrong, and no previously correct negative gains a false
  call; retain per-row vectors/regressions/fixes and remaining false calls.
  Net gains cannot compensate for regressions. Retained before/after is exact.
- Evaluate serialized final reload against every saved final prediction; ensure
  no stale state, wrong source row, mode or serialization difference.
- Raw tuples and deterministic predicted-length padding composition remain
  separate. Neither gate uses oracle lengths or repaired outputs.
- No promotion, acceptance64 opening, full model/live Hub benchmark, production
  changes or subsequent fit. Return terminal evidence for the next gate even
  when these diagnostic criteria pass.

## Artifact identity and resource estimate

Prepared `igc-replay-preparation-02/replay.json` occupies 1,633,202 bytes;
SHA256 `b89622b1b4e0e363ca0eb7dfdcc6c41b75f8758e50087e7b92183606b6354096`.
Schedule SHA256 `f0c3850b5a3653056354380cd732e8908833671952479e3cde812aaf74e27ec4`.
The current preparer SHA256 is
`1dc4e413b71761580f39197991c7d4cbc9b38cdfbb7ca63b552100973c958bd8`.
File SHA plus deterministic reconstruction bind source manifests, exclusion,
groups, complete schedule and expected activation identities. Expected state
hashes in this artifact are declarations; actual states/masks must be revalidated
by the future consumer. Preparation reads metadata, not activation arrays.

No new cache is needed. 64 F32 rows occupy 209,715,200 bytes before model/scratch;
two file handles replace one. The previous same-shape fit observed CUDA allocated
peak 728,945,664 B and private lifetime peak 4,274,294,784 B; they are estimates for
planning, not measurements or guaranteed upper bounds of the mixed loader.
Keep 32 MiB output allowance for weights/predictions/reports beyond the 1.64 MB
prepared metadata.

Nominal training row traffic is 176,160,768,000 bytes across both sources before
OS cache effects, the same total row presentations as before. Unique selected
row data is 34,458,828,800 bytes, much larger than the former matched-only pool;
random old-cache reads may add disk latency. No fit-speed or memory-saving claim
has been measured. Future pricing must validate memory, and bounded execution
must record actual I/O/time rather than assuming the earlier 133 s fit repeats.
