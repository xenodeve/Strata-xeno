# I2 regression audit and replay preparation

Issue #276. Preparation complete; **no new fit or promotion**. The previous warm
final artifact remains a retention failure. This checkpoint audits its 22
regressions and prepares [one fixed replay
protocol](../plans/2026-10-30-igc-retention-replay.md) for parent review.

## Regression evidence, before choosing replay

All 22 raw correct-to-wrong rows are compute tasks. They contain 23 changed fields:
lhs sign 5, lhs tens 7 (including one padding), rhs ones 6 and rhs tens 5. Operator
and predicted lengths remain correct on every legacy compute row. Every affected
expression/ID/value is retained in the [prior report's
appendix](2026-10-29-igc-matched-warm-start.md#appendix-all-22-previously-correct-rows-that-regressed)
and local `igc-replay-preparation-audit-02.json`.

| Regression partition | Counts |
| --- | --- |
| True lhs/rhs width | 1/2: 4;2/2: 14;2/1: 4;1/1: 0 |
| Operation | +:9; *:8; -:5 |
| Template | dev-en-work: 10; dev-th-help: 8; aug-th-solve: 1; aug-en-solve: 2; en-calculate: 1 |
| Development cohort | new-values/new-templates: 12; seen-values/new-templates: 6; new-values/seen-templates: 4 |
| Intent | compute 22; no-call 0 |

Lhs 54 occurs in 4 regressed rows; lhs-61,-2,-83,-17 each in 2; other listed lhs
values occur once. Rhs 17 occurs in 3;97 and-83 each in 2; other rhs values occur
once. The appendix includes exact targets/changes, so repeated value observations
are not treated as independent causal findings. Some errors zero an active tens
slot while preserving two-digit length. This is observed drift, not evidence
that any particular attention component caused it.

Keep negative behavior separate: old 55/84 false calls become 34/84,21 fixed,
**zero new false calls**,29 correct no-call rows unchanged. Raw compute outcomes
are 184 both correct,50 fixed,22 regressed,3 both wrong. Aggregate 234/259 does not
pass the no-regression gate.

## Why pair-only exclusion was insufficient

The original matched TRAIN has 768 rows and no exact ordered-pair intersection
with the 343 legacy rows. Nevertheless, conservative swapped-pair/fact checks
against all legacy and matched development remove 88 matched rows, leaving 680.
Ignoring these collisions would silently reuse equivalent evaluation problems.
The fixed balanced selection retains 672 rows, complete in language/intent.
OLD TRAIN falls from 10,240 to 9,844. Both retained pools have zero collisions
against evaluation on ordered pair, unordered pair, parsed expression, declared
canonical arithmetic fact and normalized prompt. No warm regression label or
predicted value enters training selection.

The initializer's historical exposure is retained, not erased: matched
development 19 exact pairs/20 fact pairs; legacy 43 unordered-pair and 40 fact
overlaps with old TRAIN. These overlap counts are not additive. This is inspected
diagnostic development, not clean fresh acceptance.

## Preparation implementation and checks

`tools/igc/replay_protocol.py` reads only SHA-pinned manifests and the frozen 343
ID receipt, parses bounded expressions without execution, validates metadata,
filters BOTH training sources, selects complete groups and materializes only
source/index/ID references plus 840 fixed batches. Each batch has 32 matched and 8
old rows per operator/no-call class. `verify_prepared` checks an external file
SHA and reconstructs the entire payload against pinned inputs; a self-asserted
digest is insufficient. A future fitter must requalify the actual state arrays
and must not use these locators as model features.

Tests pass 45/45 on CPU Python and 45/45 on qualified original CUDA Python.
They include parser/metadata disagreement, quote/no-call symbol handling,
subtraction/sign/commutation semantics, prompt normalization, split exclusion,
deterministic mixing, tampered schedule/index/source identities and file/source
provenance refusal. Synthetic fixtures are independent of acceptance64.

Independent review found a JSON round-trip defect before any fitting:
integer histogram keys became strings on deserialization. The actual prepared
file failed verification; the fixed implementation emits string keys and a
prepare->JSON->verify histogram test now passes. Preserve the first artifact
as failed evidence; qualified artifact 02 is separate. No silent identity update.

One test-edit NameError was corrected before green verification. The first
standalone repro missed the project import path; the corrected repro reaches
the actual round-trip failure. An initial metadata audit repeatedly rebuilt
exclusion sets inside its row loop; after verifying its exact parent invocation,
only that task-owned audit was stopped. The corrected bounded audit precomputes
sets and completes in 0.486 s without activation/model access. This is harness
bookkeeping, not a Strata speed claim. No unrelated process was stopped.

## Frozen deliverables and remaining work

Qualified replay file 1,633,202 B, SHA256
`b89622b1b4e0e363ca0eb7dfdcc6c41b75f8758e50087e7b92183606b6354096`;
schedule SHA256
`f0c3850b5a3653056354380cd732e8908833671952479e3cde812aaf74e27ec4`.
Local metadata evidence: `igc-replay-preparation-audit-02.json`,
`igc-replay-preparation-02.log`, `igc-replay-preparation-green-04.log`,
`igc-replay-preparation-runtime-02.log`, and failed round-trip receipts.
Large activation caches, weight artifacts and private paths are not published.

The proposal keeps 840 updates/53,760 presentations but changes the data mixture:
672 matched rows each 40 times and 9,844 old rows each 2 or 3 times, rather than 768
matched rows each 70 times. It is not another initialization-only comparison.
Use the original retained artifact with fresh optimizer, final checkpoint only,
strict per-row retention plus matched gates, and the protocol's bounded resource
checks. New mixed-cache gather/fit integration and new pricing remain required;
neither is implemented or executed by this preparation.

Return the protocol/resource estimate for review before any fit. No full model,
activation collection, cloud spending, acceptance64 access, production/system
change, push or upload occurred. Original retained, failed fresh and warm final
weights remain unchanged. Current preparation is not proof that replay prevents
the observed regressions or that the feature is ready for daily Hub use.
