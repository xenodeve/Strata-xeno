# Replay candidate passes cached native qualification

Issue #276. The replay candidate passes native export/forward, cached controller,
component lifecycle and the instrumented controls below. **It is not promoted.**
There was no training, new activation collection, full-model inference or sealed
acceptance evaluation. All changes and commits remain local and unpushed.

This follows the [frozen qualification plan](../plans/2026-10-30-igc-replay-native-qualification.md)
and [replay result](2026-10-30-igc-replay-result.md). Model/tokenizer bindings,
embedding input head, readiness head, result mapper, selection policy, thresholds
and production defaults remain unchanged. Only the experimental first-attention
artifact is replaced in a separate diagnostic bundle.

## Export and native forward

The existing kind1 schema still consumes actual right-aligned F32
`[80,4,2560]` first-attention states, an 80-row mask, fixed positions 0..79 and a
zero unused query. It requires no new feature, token-ID input, parser, oracle
operand or result. LayerNorm, position embedding, contextual attention and
categorical field layout are unchanged.

The strict Python reader checks all 40 named tensor shapes and finite F32 data.
An independent post-run audit compares all 96,964 parameter values byte-for-byte
with the source safetensors. Native host readers accept the diagnostic artifact
and reject production use in the regression suite. The headers bind the exact
model, tokenizer, state, training protocol and unchanged kind1 schema.

| Item | Result |
| --- | --- |
| Matched development rows | 768/768 forward comparisons pass |
| Frozen legacy rows | 343/343 forward comparisons pass |
| All nine field decisions | All 1,111 equal qualified Torch and saved final predictions |
| Maximum absolute logit difference | 0.000034332275390625 |
| Frozen numerical tolerance | atol 0.001; rtol 0.0001 |
| Normal/graph/repeated/alternating input | Pass |
| Capacity 1 and 64 | Pass |
| Empty/bad mask, valid-row NaN, finite overflow | Distinct failure status; caller output preserved |
| NaN in an absent row | Ignored according to the existing mask contract |
| Hard off | Caller logits/status unchanged; mapper device allocation zero |

There were 25 native probe invocations; each also exercises five repeated calls
and input A/zero/A transitions. Normal/graph and repeated valid results are
bit-identical on these fixtures. This is a narrower claim than bit-identical
native-versus-Torch logits, which are compared at the frozen tolerance above.
All source, state, protocol, saved-reference and complete cache identities are
checked before/after. Only a batch is read into RAM; temporary owned probe inputs
are removed by exact filename. Existing arrays remain untouched. The retained
eight-row invalid-control fixture is for instrumentation, not a new data cache.

The input mapper reserves 12,197,712 device bytes at capacity 64. The qualification
process's CUDA allocated peak is 666,116,096 B and its highest sampled process
private-commit lifetime peak is 3,800,903,680 B. These include reference/staging
work and do not describe the model's total resources. Elapsed qualification time
81.328 seconds includes cache hashing, Torch reference calls, native probes and
controls; it is not a serving-latency benchmark.

## Actual gated execution on the old coherent development capture

Reuse the same immutable 64-prompt coherent capture, with genuine embedding,
first-attention and final mixed states. Its historical labels are 48 TRAIN and
16 HELDOUT; it has been inspected repeatedly and is **not** sealed acceptance64.
There are 48 compute prompts and 16 no-call prompts. The larger matched/legacy
sets lack paired final-anchor/output data, so raw head scores must not be relabeled
as measured controller call rates. A subsequent
[index audit](2026-10-30-igc-controller-gating-diagnosis.md) confirms existing
embedding lookups for all 343 legacy and 384/768 matched-development rows; the
other 384 need eleven additional embedding token rows.

| Same coherent 64 prompts | Original retained | Replay candidate | Warm fit counterfactual |
| --- | ---: | ---: | ---: |
| Embedding positive tuples exact | 36/48 | 36/48 | 36/48 |
| Attention positive tuples exact | 48/48 | 48/48 | 46/48 |
| Selected correct positive requests | 33/48 | 33/48 | 32/48 |
| Selected wrong positive requests | 0 | 0 | 0 |
| Readiness no-call on positives | 11 | 11 | 11 |
| Disagreement after readiness passes | 4 | 4 | 5 |
| Actual negative executor calls | 0/16 | 0/16 | Not executed; predicted 0/16 |

Original and replay columns are **actual native controller executions using the
same current probe binary**. Warm is a separately labeled Torch counterfactual,
not another native or training run. Original raw fields/readiness/selections
also reproduce their frozen historical reference.

In the retained native audits for 64 logical cases, 33 have exactly one executor
evaluation and 31 have none; all 33 selected requests match the intended tuples.
Normal/graph controls deliberately restage cases, so 33 is the selected-case
count, not a global count of all evaluations across the qualification process.
Example: `Calculate 31*-24. Give the exact integer result.`
produces native `-744`. Every selected request/result record, owner, anchor,
attempt count and status matches an independent offline integer oracle. The
oracle never feeds operands or answers into the native input path.

The other 15 positives are not executed: 11 fail readiness and 4 disagree. All
four disagreements have correct attention fields and wrong embedding fields;
eight other embedding errors occur on rows that readiness already blocks. These
overlapping conditions must not be counted as 12 extra calls recoverable by an
embedding-only change. No gate is relaxed here.

The 64 typed audit/output records are identical between retained and replay
(SHA256 `10a313a7183d217bd8319ddda069039c371d5212237eed1f0f7c7042171ec39b`).
Replay improves broader raw development retention but adds no gated coverage on
this older capture, where the original attention head was already exact. The
42/84 legacy negative false-call candidates from the prior report remain an open
quality concern; 0/16 calls on this capture does not establish their safety.

The native result mapper's output matches the unchanged Torch mapper to maximum
absolute difference 0.000003814697265625. Failed selection preserves mixed-state
bytes exactly. This verifies the activation transformation, not a generated
language answer's use of the result.

## Ownership, cancellation and instrumentation

The cached pipeline passes ordinary/graph/repeat execution, retained one-shot
provenance, broadcast to 64 output rows, unavailable-context reset and empty-mask
failure without a stale result. Three genuine records selected by declared
disagreement/success/no-call status exercise module trial abort, replay, commit,
new-owner reset and stable checkpoint storage at both 1 and 64 rows.

All eight instrumentation stages pass with zero reported errors/hazards **and**
bit-identical semantic outputs to the uninstrumented controls:

| Tool | Input mapper: 8 valid/invalid rows, graph mode | Pipeline: 3 coherent records |
| --- | --- | --- |
| ComputeSanitizer memcheck | Pass | Pass |
| initcheck | Pass | Pass |
| synccheck | Pass | Pass |
| racecheck | Pass | Pass |

Pipeline instrumentation includes selected/disagreement/no-call, graph/repeat,
broadcast64, missing-context and malformed-mask controls. It is not an
instrumented run of every development row. Racecheck's zero hazards does not
prove absence of every global-memory race or establish reduction determinism.

Separate native regressions pass for the context ring, prefill snapshot,
invocation and dispatch. The **actual learned wrapper** also passes result,
cancel/audit revocation, cancellation before stage, same-prefix reuse, foreign
lineage, graph, release and reprepare controls using cached activations. It
freshly validates actual model/tokenizer content, reading the existing files;
it does not allocate language-model weights or generate language tokens. That
test takes 53.454 seconds with observed process lifetime private-commit peak
259,092,480 B. This is content binding/lifecycle, not model startup latency or an
optimization measurement.

Both selected native component probes were rebuilt and `cuobjdump --list-elf`
confirms sm89 and sm120 images. Execution is on the primary 5060 Ti only. No new
native kernel/backend source is changed in this checkpoint. Secondary execution,
HIP, SYCL and Linux validation remain unrun; compilation coverage is not a
substitute for those runs.

The selected host/qualification suite passes **21/21** in the qualified original
Python. The new offline qualification oracle was test-first and uses synthetic
fields/literal results, with no corpus construction or acceptance access.

### Preserved harness failures and corrections

1. Forward attempt01 stops before artifact creation/GPU execution because
   PowerShell preserved an empty `CUDA_DEVICE_ORDER` key. The contract requires
   absence, not an empty value. Attempt02 removes the key in that command's
   process environment; no policy/tolerance/system change is made.
2. The sanitizer runner finishes all eight instrumented stages, then stops on
   its assumption that `igc_invocation_test.exe` already exists in this build.
   The missing invocation/dispatch targets are built and all four regressions
   run in a separate completion script. Raw failed log and the eight-stage
   `report.partial.json` are retained; neither is relabeled as an exit-zero run.

## Identities and evidence

| Item | SHA256 |
| --- | --- |
| Source safetensors | `3cb5a645c3a9d74778d807672d5fe0dd4d4d025d7992e729cc002d816828043d` |
| Diagnostic kind1 artifact, 389,808 B | `527b1e5f2f54c62d9352a176954e0e998293b1fc5a66512e2025311a4a115e07` |
| Separate four-network bundle manifest | `ce6f15e6c6ba79f223c11fbece2221cf62d2a6b498799520d95e060ed3c2f293` |
| Model content | `7aa36d15131d255179ab6c87fa9c9f81a0b541fc9e81e959d4c3e29eb20feb5f` |
| Tokenizer content | `889964f986b93c35102f25f5e396efeed65fcb180429c35d97536fb7f6b72106` |
| Unchanged kind1 schema | `cb73a018f1b785602b1b05e3754dbd0d1687f010efa92526f2bc92ea994a3101` |
| Immutable coherent input capture | `669becea5404dd4aff88eac2efeadab872b8fe30ae5ddfb9ee912927534721f7` |
| Input probe binary | `8190a042f00e5e25979f7d2992d96e4cc85c40eb33583f8e58cd3a6a0c3c3a35` |
| Pipeline probe binary | `505661416a061d1cb6f63eb8f399ac5f4b4a221ea81d0bf391bcb03606122495` |
| Trial binary | `4a2fc8d02d2a0191bc0744fad4f97efaf6e78c0fd8b1c278214af1c37d8e36a2` |
| Native forward report | `aab006f3a5544393b72dfdc19a099a810595382d1bd59158283bb644a28c5d11` |
| Controller report | `82646f239eae50a1e43b27e12d63fca311af682d52d010bc223aae6a08770d59` |
| Eight-stage instrumentation receipt | `c9848824949725d1f30b3f346e8a195a73fdd9f5aeb473f81777ee2ef12725a1` |
| Completion/regression report | `3f2e19e954c5b2432200e93c72be791501dd589d2f1ef4639686aa92d21bce7b` |
| Independent tensor/receipt audit | `2816884185ad3f96b7b3d7ab27ec90208dfbf18951ab45edf8c970542bcdcb61` |

Evidence remains under the task evidence root: `igc-replay-native-input-01`,
`igc-replay-native-controller-01`, `igc-replay-native-sanitizers-01`,
`igc-replay-component-regression-01`, `igc_replay_lifecycle_01`,
`igc-replay-native-literal-qualification-01.json` and corresponding terminal
logs. Original retained, failed-fresh, warm, capture and production-bundle files
remain unchanged. An independent reviewer checked the runner contracts and
saved evidence. No artifact/cache/model files or private absolute paths are
published with this report.

## Remaining gates and bounded live proposal

These component gates pass; I2/generalization, I3 real-model result use and I4
repeated learned triggers remain open. G3/G4 still need real language acceptance,
MTP/suffix, prefix/history/session and snapshot integration. G6 same-session
hard-off greedy raw-token parity, G8 actual Hub paired quality/latency/token
measurements and G9 practical-use stability/rollout are not run here. Output
vector fidelity or a selected exact number alone cannot close those gates.

Next, after separate review and fresh resource admission, use the existing
bounded live diagnostic harness on the unchanged Gyro-M262K/five-worker Hub
profile with only task-owned executable/log/profile-save paths and this separate
diagnostic bundle. Preserve model/quantization, RoPE, workers, expert tiers, MTP
suffix, cache and the actual profile's serving priorities. The subsequent config
audit confirms pool/process priorities 2/2; the earlier "normal priority" proposal
was incorrect. No profile was changed. No private-commit cap. The existing
six requests are three known compute prompts (`31*-24`, `20--15`, `-28+-49`),
two quoted/no-compute prompts and a repeat of the first prompt. Check each
actual staged feature/anchor, native audit, exact result, output use and cleanup.
Use the existing greedy 96-token request limit and retain the selected profile's
vision setup. Keep the 600-second task /120-second per-request bounds and system emergency
floor. This is a first-anchor smoke, not the subsequent ABBA benchmark.

At 2026-10-30 01:08:02 UTC, after components finish:

| Resource | Available | Unchanged admission requirement | Shortfall |
| --- | ---: | ---: | ---: |
| Physical memory | 14.8139 GiB | 24 GiB | 9.1861 GiB |
| System commit headroom | 15.1689 GiB | 47.5544 GiB | 32.3855 GiB |

The 24-GiB physical gate is the selected Hub launcher's `NEEDGB=24`. The commit
requirement is the **task's disclosed admission estimate**, not a Hub setting or
process cap: observed same-profile process-private peak 48,869,707,776 B plus
43,963,688 B fixed component/artifact reservation and a 2-GiB emergency reserve.
The historical peak is not a guaranteed upper bound; unknown new CUDA-driver
transients still require live measurement. No launch was attempted, no unrelated
process was stopped and no memory/system setting was changed. The current
qualification therefore returns a prepared live proposal with a measured
resource shortfall, rather than a full-model success claim.
