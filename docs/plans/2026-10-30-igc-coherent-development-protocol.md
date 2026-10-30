# One coherent development collection for the measured gate failures

Issue #276. **Proposed collection; not executed. No fit or policy change.**
The [cached diagnosis](../reports/2026-10-30-igc-controller-gating-diagnosis.md)
finds readiness and embedding/consensus limits, not an arithmetic failure. The
next useful work is a joint observation producer and one controlled collection
from the actual Hub path. Further standalone-executor polishing cannot establish
the missing gate behavior.

## Question and fixed comparators

For the same arithmetic expression, do imperative/question phrasing, language,
quotation and label intent change readiness or operand extraction incorrectly?
If a correct result is available, does the existing late mapper use it at the
actual output positions? Keep original retained and replay input heads, the
embedding/readiness/output assets, every threshold and the full model frozen.
No checkpoint selection or training on this collection is part of this protocol.

This is diagnostic development, not representative daily-use traffic or fresh
acceptance. The PRD order remains I2 coherent input/readiness evidence, I3 real
result use, I4 repeated committed calls, then I5 matched practical evaluation.
The internal sandbox, programming/debugging, geometry, memory and later model
adapter goals remain open; this arithmetic experiment does not replace them.

## Minimal controlled cohort: 400 anchors

Use one expression per cell of 3 operations x 4 nonzero operand-width pairs
(1/1, 1/2, 2/1, 2/2) x 4 sign pairs: **48 expression groups**. For each, use
both English and Thai and four intent/frame conditions:

| Condition | Example frame | Expected eligibility |
| --- | --- | --- |
| Compute imperative | Work out `{expr}`; return its exact integer value. | Compute |
| Compute question | What exact integer value does `{expr}` have? | Compute |
| Quoted copy | Copy `"{expr}"` character for character without evaluation. | No call; copy is not arithmetic verification |
| Label acknowledgement | The item label is `"{expr}"`. Acknowledge the label; do not solve it. | No call |

The Thai frames carry the same four intents: `หาค่าของ {expr} แล้วตอบเป็นจำนวนเต็มที่แน่นอน`,
`ค่าจำนวนเต็มที่แน่นอนของ {expr} คือเท่าใด`,
`คัดลอก "{expr}" ให้ตรงทุกตัวอักษรโดยไม่คำนวณ`, and
`ป้ายรายการคือ "{expr}" ให้ตอบว่ารับทราบป้ายโดยไม่คำนวณ`.

These give 384 paired anchors: **192 compute and 192 no-call**. Add 16 separately
reported ordinary/unsupported negatives, balanced by language: eight ordinary
instructions and eight tasks outside the current two-digit +,-,* mapper. That
means unsupported by this mapper, not unsupported by the project's future
rational/programming capabilities. Total: **400 anchors, 192 positives and 208
negatives**. Freeze every literal prompt, label, prefix and row order before any
new activation or prediction is observed. Reject exact normalized-prompt
collisions; never edit a case after seeing its output.

For output-position diagnosis, choose **one positive frame per expression group**
by a fixed rotation, balanced across English/Thai and imperative/question: 48
output samples, not all 192 positives. Freeze canonical completions and tokenize
with the actual pack before collection; at most 32 tokens per completion and at
most 256 total prefix tokens. Use the established reasoning/final-answer boundary
format with the correct integer, not an invented tool-call sequence. Refuse an
overlong sample instead of truncating it. Anchor position zero is already in the
400-anchor pass; at most **1,488 additional output states** are needed.

## Freshness and independent units

Seed 20261101. Exclude ordered pair, unordered pair, parsed expression, canonical
add/subtract/multiply-sign fact and normalized prompt against these **named
existing manifests**, including both their training and inspected development:

- Original coherent64, balanced input 4,480 and expanded input 10,624.
- Matched input 1,536, readiness 320, readiness augmentation 416 and output 30.

Metadata audit finds 8,528 known ordered pairs, 6,586 unordered pairs and 7,729
canonical facts. Eligible counts remain nonzero in all 48 proposed cells. The
private metadata witness fixes one feasible choice per cell without invoking a
model. It has **44 connected pair/fact clusters**, because some expressions
within the new cohort share operands or an equivalent fact. Count 48 expression
groups for complete-condition success, but use those 44 connected clusters for
resampling/uncertainty, not 384 prompts or individual digit slots as independent
trials. These are structural resampling units, not a proof that all errors are
statistically independent; shared operand values remain a limitation. Revalidate
the connected components on the final prepared manifest.

Freshness here is relative to those listed archives. **Do not open acceptance64
or invoke its generator for rejection sampling.** Its overlap is unknown, so this
protocol makes no sealed-test-disjointness claim. It does not replace the sealed
set; later independent acceptance reporting must disclose any overlap rather
than retroactively renaming this development set as acceptance.

Individual operand values have already been seen; this tests new combinations,
facts and matched frame/intent transfer, not unseen scalar values. No fresh
zero-operand examples remain under these exclusion rules within +/-99. Keep
existing zero controls as historical regressions and explicitly leave fresh
zero-controller coverage open. The width/sign grid is intentional stress
coverage, not a claim about the frequency of daily-use inputs.

## Capture the actual architecture, with shared ownership

Use the unchanged selected Gyro-M262K/five-worker Hub configuration, actual
model/tokenizer/pack, D2x tiers, KV/prefix behavior, auto batched/wave prefill, MTP
and Vision. Re-read the launcher and resolved arguments; current profile has
`--pool-priority 2` and `--process-priority 2`. Preserve these actual settings;
do not describe them as normal priority or add a different measurement baseline.
Do not add a private-commit cap, force serial prefill, force cold requests or
substitute the layer0-only prefix extractor for a full-Hub observation.

At each original answer-start anchor, archive together:

1. Canonical token prefix, request/invocation identity, source/prefix lineage,
   accepted input positions, tap/layer/dtype, stream/fence and cache provenance.
2. Both genuine pre-anchor tails: embedding broadcast and first-attention after
   layer0 attention HC write. Each has 80 x 4 x 2560 F32 geometry and an explicit
   valid mask. Only consumed/accepted input rows may enter; no rejected MTP suffix.
3. The same anchor's final-HC mixed 2560-vector, before any result injection, for
   readiness and output mapping; actual native-head output/receipt for fidelity.

One generated token completes an owned anchor request. It is evidence of request
completion, not a label or input feature. Observation is a diagnostic capture of
the unadapted stream, not a profiler-off serving baseline or hard-off parity run.

Missing rows, foreign lineage or an unobserved cached prefix reject the row.
Never fill them from final R/KV, copy another prompt's states or label an isolated
early-layer pass as a full-Hub capture. Preserve the actual prefix reuse policy;
if completeness fails, stop and implement correct feature-history support before
continuing. Prove the producer on the first four predeclared cases of this same
manifest, without selecting cases or changing rules from the predictions.

Complete and hash all 400 original anchors **before** the teacher-forced output
pass. For output sample j, capture the true pre-injection final mixed state at
`base + completion[:j]`; target token j must not enter that state. Reuse the
original anchor's identity/result when replaying the late mapper. Never recompute
input/readiness labels from a prefix already containing the forced answer.

## Three separately named output controls

Replay captured states through the existing native head/mapper with:

- Hard-off mapper: unchanged mixed bytes.
- Learned gated result: actual unchanged selection/typed executor, including a
  valueless no-call/error and mandatory output bypass when selection fails.
- Oracle-owned result control on positives only: independently checked operands
  passed directly to the diagnostic native exact executor to isolate output use.
  This is not a learned call and must never feed a readout or production path.

Compare requests/results to an independent exact oracle outside native inference.
Report raw tuples, readiness, consensus and actual selected-correct/wrong calls
separately. On negatives, copied digits are not a verified arithmetic answer.
Output-position metrics are conditional target-token accuracy/NLL, sign/digit
errors, nonnumeric behavior and exact failure bypass. They do not establish free
generation or spontaneous repeated reasoning calls; the bounded live free-answer
smoke and later task/latency benchmark remain separate.

## Storage and admission

Use an ignored task-owned directory on the workspace volume, for example
`build-igc-coherent-dev-01/data` (`build*/` is already ignored). Do not copy the
model or move/delete existing evidence. Plan a streamed archive:

| Planned storage | Bytes |
| --- | ---: |
| 400 attention tails | 1,310,720,000 |
| At most 1024 native embedding rows | 41,943,040 |
| Embedding indices plus masks | 160,000 |
| 400 anchor mixed vectors | 4,096,000 |
| At most 1488 additional output vectors | 15,237,120 |
| Bounded metadata/logs | 67,108,864 |
| Four full-vocabulary reference logit vectors | 3,973,120 |
| Temporary streamed staging | 67,108,864 |
| Free-space reserve | 4,294,967,296 |
| Total before small array headers | 5,805,314,304 |

Require **5.5 GiB free** before preparing the archive; the difference covers
headers and small receipts. Embedding rows may be deduplicated only after
byte-exact token/tap/source checks. If they are not constant across the actual
producer, stop; do not silently switch to a larger layout or lower precision.
Store compact output statistics and full-logit hashes for the other positions;
the four full references and source-bound native head validate the replay seam.
The existing observation reader requires every full logits file, so a **new
versioned compact reader is required**, not a relaxed old schema check.

Memory admission is separate from Hub behavior: after preparing helper metadata,
require at least the observed same-profile summed private peak 48,869,707,776 B,
plus existing fixed components 43,963,688 B, future streamed host staging 64 MiB and
2 GiB commitment reserve: **51,128,263,976 B (47.6169 GiB) available commit**. Physical
availability must satisfy the actual Hub 24 GiB gate and the observed working-set
peak plus1 GiB reserve; the Hub gate currently dominates. These estimates are not
caps or guaranteed upper bounds. Measure actual private/commit/VRAM/transient
peaks, stage buffers and cache impact on the new run.

Proposed bounds are 1800 seconds for capture and 2100 seconds for the owned outer
job. Price the first four fixed anchors, then price the first fixed output cases
after anchor freezing; continue each remainder only if conservative measured
projections fit its remaining budget. No timing claim from a different collector
may substitute for this gate. A timeout or incomplete role set stays incomplete;
do not extend silently, discard hard cases or rerun a fit. No cloud/downloads or
OS/pagefile changes are part of this protocol.

## Implementation prerequisites and acceptance

The current code has the component seams, not this complete collector. Before
any collection, implement/test the following as one diagnostic capability:

- `igc_context` / `igc_prefill_snapshot` consumers and the learned program wrapper:
  a joint observation sink on actual ordered wave/decoder streams, separated
  from the result-injection hook; explicit accepted-prefix and cache lineage.
- `generate.cpp` and its diagnostic CLI guards: pair early tails and pre-hook
  final state at one owned anchor; keep production hard off unchanged.
- `igc_observation.hpp`, the bounded archive writer and versioned Python reader:
  role/shape/size/digest/partial-write refusal, per-row journaling, compact head
  receipts and teacher-forced output roles without future-label leakage.
- A manifest-only preparation path that reads the seven frozen manifests,
  tokenizes actual prompts, validates the 400 anchors/48 output samples and
  clustering, never invokes a corpus generator that opens acceptance64.
- Unit/native tests first: mixed-owner/tap mismatch, missing cached features,
  output label leakage, rejected suffix, cancel/reset and failed/partial resume.
  Keep Blueprint and issue evidence current with that implementation.

Preserve the PRD's strict diagnostic quality requirements: raw readiness correct
on all 192 positives and 208 negatives; both readouts exact on supported positive
tuples and no-call operators on negatives; all 192 positives selected correctly
and no negative calls, with valid native results and failure bypass. Guarded
selection must not conceal failed per-head gates. Also report raw per-role counts, complete-condition success,
language/frame/width/sign cells and cluster-level uncertainty. Any failure is
evidence for a separately designed training/data change, not permission to tune
thresholds, fit this evaluation set, promote an artifact or close I2/I3. This
protocol is the next measured gate investigation, not another executor benchmark.
