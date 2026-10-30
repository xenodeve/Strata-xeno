# I2 replay execution gate

Issue #276. Parent reviewed `ab4d6d41` and authorized mixed-cache implementation,
tests, independent review and bounded pricing. **Exactly one fit is conditional
on complete measured pricing and resource admission.** This record supplements
the [frozen replay protocol](2026-10-30-igc-retention-replay.md).
The latest instruction keeps **1800 s fit /1920 s owned outer**; it supersedes the
older proposed 2100 s outer allowance. Do not increase either bound or alter recipe.

## Fixed learning contract

Keep replay file SHA256
`b89622b1b4e0e363ca0eb7dfdcc6c41b75f8758e50087e7b92183606b6354096`
and its complete 840-batch schedule unchanged:32 matched plus 8 OLD per ready/op
class, B 64, original copied retained initializer `519fbc19...`, fresh AdamW,
840 updates, model seed 20261031, cosine 0.001 to 0.00001, wd 0.01. Final checkpoint
only; no cloud, new capture/cache, full model, acceptance64, sweeps, extra seed,
policy change or promotion. Preserve original/fresh/warm artifacts.

## Mixed gather and qualification

Load and hash both actual source state/mask/header/manifests using the declared
expected cache identities. Rebuild the prepared payload against the pinned
manifests and frozen 343-ID selection. Resolve each source/index back to the
expected ID and TRAIN metadata; evaluation targets never train the readout.

`NpyRows.read_into` fills a caller-owned bounded F32 buffer; `MixedContextInputs`
gathers contiguous source runs without sorting away their order. Duplicate indices,
HC/channel layout, masks, positions and zero query stay exact. One 64-row state
buffer is reused during pricing/fit; no whole-cache preload. Finite selected
states are checked. Before real pricing/fit, compare both first/last planned
batches and a mixed duplicate selector against direct source rows/masks (134
comparisons). Synthetic file-backed/proxy tests separately reject wrong IDs,
evaluation rows, invalid geometry and whole-cache materialization.

## Pricing is not training

Pricing uses a disposable copied head and **constructs no optimizer**. Read all
840 scheduled CPU batches through the actual gather path, rather than estimating
I/O from a small RAM-resident subset. On fixed batch indices
0,119,239,359,479,599,719,839, run forward/loss/backward with cleared gradients,
but no parameter update. These eight TRAIN-only probes measure first-use and
warm computation. Assert the parameter fingerprint remains unchanged; write no
weight artifact. Reset model, seed and optimizer in a separate fit process.

Pricing phase 120 s / owned outer 240 s. It must finish all 840 gathers/eight backwards,
source/input rechecks and numerical/resource guards. Incomplete or over-budget
pricing cannot admit a fit; retain its failure/progress and stop that action.
No partial-schedule extrapolation, silent repricing or recipe/budget change.

## Conservative admission calculation

Record actual full-schedule CPU gather time, eight synchronized backward+transfer
times, validation time, peaks and resource snapshots. Compute allowances:

- I/O:2x measured full-schedule gather time.
- Core computation: the greater of 4x the recorded same-shape full warm-fit time
  (132.9981655 s, includes optimizer/guards absent from backward-only pricing), or
  `2 * (first backward + 839 * slowest warm backward)` from current pricing.
- End validation:2x measured initial validation; save/report allowance 60 s.
- Fit estimate is the sum; outer estimate adds initial validation plus 15 s setup.

Only admit when fit estimate<=1800 s AND outer estimate<=1920 s. This is a
conservative planning estimate, not a guaranteed upper bound or a speed claim.
Actual runtime/resource guards still stop a run. Preserve the raw components;
do not reinterpret estimates as a profiler-off paired Hub measurement.

The receipt binds complete recipe, initializer/fingerprint, first 64 source IDs,
prepared file/schedule/member digests, source code, execution protocol and BOTH
actual cache identities. Before fitting, recompute the estimate from receipt
numbers and require exact binding. A mere `admitted=true` flag is insufficient.
At fit entry, read only a bounded regular receipt (256KiB), hashing the exact
bytes parsed. Recompute admission with the slower of priced/current validation
and the actual elapsed outer prefix, before constructing the optimizer. Record
and refuse a changed admission instead of silently relying on an earlier estimate.

## Conditional single fit and evaluation

Fresh offline admission:8 GiB physical available,12 GiB system commit headroom,
2 GiB primary VRAM,32 MiB output plus 4 GiB disk reserve. Keep runtime emergency
floors 1 GiB physical/2 GiB commit; no private-process cap or OS/Hub changes.
Run one heavy job at a time, headless, with owned cleanup.

After the valid price and reset, run exactly 840 updates from the stored schedule.
All loss/gradient/parameter/optimizer states must remain finite, without clipping.
Stop on error/timeout; do not retry or choose another checkpoint. After the final
update only, evaluate all 1536 matched rows in frozen/eval/no-grad and preserve
categorical predictions for serialized reload comparison. Recheck inputs before
saving separate final weights. Do not auto-promote.

Before/after independent evaluation covers all 1536 matched rows and all 343
legacy rows, with a retained unchanged control and identical F32/batch 64 math.
Require raw matched 384/384 and 0/384 false calls, no legacy correct-to-wrong row
and no new negative false call. Preserve per-row outcomes, inherited 19/13 exact
pair groups, semantic exposure disclosure and separate padding counterfactual.
Reload must reproduce every saved final categorical prediction. A component
pass remains distinct from fresh acceptance, live result use and Hub readiness.
