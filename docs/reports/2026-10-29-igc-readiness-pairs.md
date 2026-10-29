# Actual final-state readiness: strong diagnostic, not a passed I2 gate

## Capture and frozen tests

The new explicit readiness-anchor mode reuses the qualified output collector's
actual Hub/Vision/GPU lifecycle. It captures one completed fixed-answer-start
head-mixed-after-final-HC2560-vector for each of320 prompts. There are144 matched
calculation/copy-or-quotation pairs and32 ordinary/initially unsupported negatives.
208 rows train and112 are template/pair-disjoint development. No answers,
teacher-forced completion or arithmetic executor is used. A single sampled token
only completes each owned request; it is not a training target or feature.

An initial Temp-script import failed before model or manifest creation. Explicit
worktree import resolution fixed that harness error, not any model behavior.
The next capture finishes320/320 in234.031s. Selected unchanged Gyro-M262K/five
workers, Vision, MTP suffix, GPU mapping and D2x tiers are retained; no memory cap
or serving changes. Sampled max process private46,661,042,176B, min available
physical10,420,535,296B, min system commitment headroom13,686,964,224B. No timeout
or emergency floor occurs; owned tree drains. Collection time is not latency.

Seven new/existing accounting and corpus checks plus three collector checks pass.
They cover matched-pair ownership/splits, no answer targets, exact input anchor,
invalid classes, pair-level false-call accounting and train-only normalization.
The loader revalidates every completed request/tensor/prefix and source/corpus
digest. The original input acceptance64 remains unconsumed.

## Final checkpoints, fixed rubric

Both CPU fits use the same320 real vectors, seed20261107, training-only mean/std,
300 full-batch AdamW steps lr0.001/weight_decay0.01, deterministic F32/four threads,
final argmax and no development fitting, clipping, threshold or checkpoint search.
Only readout parameters receive gradients; no backward through deployed Gyro.

| head | parameters | fit correct | development positive | development false calls | complete development pairs |
|---|---:|---:|---:|---:|---:|
|2560→128→2 GELU|328,066|208/208|47/48|0/64|47/48|
|2560→2 linear capacity control|5,122|208/208|48/48|1/64|48/48|

Both have111/112 development classification accuracy. MLP misses `(19) - (19)`
asked with a new English template; logits `[0.6321703,-0.7321997]`. Linear instead
falsely admits the currently unsupported `Find the quotient 29 / 7.` with logits
`[-0.1534486,0.6358205]`. All paired negative quotations are correctly rejected by
both. The simpler model is not an unconditional improvement, and neither meets
the predeclared all-development-success diagnostic gate. Do not adjust the
threshold or relabel unsupported inputs to convert this result into a pass.

MLP state `0ca4f0b8cc66a0f876d86623dd95ecef828bf9927c9fd9a55e2021b3d1497264`;
linear state `48bb9f7f0fa51230d9e1da351143cbc83e1ac450c7f4b23a0b6daeef57c872cc`.
Fit/evaluation0.532s and0.109s respectively are small offline training durations,
not serving performance. These outcomes separate readiness from operands more
cleanly but do not demonstrate learned runtime use, generalization outside this
pilot, operator eligibility or full I2 completion. Next expand matched training
coverage of trivial/equal operands and unsupported operations under a new frozen
protocol; preserve these reports and the development/acceptance distinction.

## Evidence

Frozen [capture/MLP protocol](../plans/2026-10-29-igc-readiness-pairs.md) and
[linear protocol](../plans/2026-10-29-igc-readiness-linear-control.md).
Corpus `8396b19143c1e62475051fec964b24a8ffaf715806f36580135746dab5880458`;
records `e49df6f3b8feef9346c06539be0937660d3a194034931b13ffa5c4d48a33ecbe`;
input manifest `70a2711462e7285da8a0449a5d5396fc903160d3373b34781231624aa6a4a31a`.
Engine `74c104b24727202f156c0326afc1f18e586cce29b9b6bffff495d1385d089258`.
Private evidence: `igc-readiness-anchors-01/`, initial failed
`igc-readiness-anchors-01.log`, completed `igc-readiness-anchors-02.log`,
`igc-readiness-fit-01/` (archived original source), `igc-readiness-linear-fit-01/`
and each DONE0 training log. No fresh acceptance, default activation or benchmark.
