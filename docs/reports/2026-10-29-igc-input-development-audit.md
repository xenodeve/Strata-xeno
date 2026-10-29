# I2 development audit and matched-corpus preparation

Issue #276. Audit baseline `a67c4d0ef80047e29d8ad13886f8e082288cfd11`.
This reproduces old development accounting and prepares a new diagnostic. It
does not rerun learned inference/training, alter retained weights/gating or
consume the sealed acceptance64. Existing corpora and failed results are retained.

## Reproducible evidence

`tools/igc/development_audit.py` reads exactly six named OLD evidence JSON files,
verifies their full digests before parsing and refuses to overwrite its output:

```
python -B -m tools.igc.development_audit --evidence-root <existing-task-evidence> --output <new-audit.json>
```

The complete file identities are in the utility's `FILES` map. Primary inputs:

| Evidence | SHA256 |
|---|---|
| expanded-v2 corpus file | `0aa4fa0c34e45b47f213279a638d7d11d1a04f0256877ddcf5e8be6c803452d9` |
| native-context final predictions | `37c8200140c8a00461cab988f810fed5af4d33bffd5a2ce90f3e5bf94c7d386f` |
| native-context fit report | `6b47259d3b8ce9fb88319cebced1fbbb6ce1d9dd188bfc6d069c707668133870` |
| OLD64 pipeline reference | `f072e19dedad9b32136334c687a99c8006be23aa6ec08aa5f2946b7e5636bffc` |
| output-fit report | `8da2cf607098f4c4daed86d4f9216436d640ff9f1346a4bb2721f845751a2fd6` |
| readiness-fit report | `08989d4c4c07963d264a71570e6f79fa6fcc348291bd74993235adb7e22c0610` |

Native-context state is `519fbc196993c94676f4031f7bd45ad9abf661f8b7b4fd3b9d6358204b5294e2`.
The96/50/67 positive-tuple table concerns native first-attention/context;
embedding comparator is50/77/33. Source traces:
`balanced_corpus.py:50-106` (sampling/exclusion), `readout.py:13-24,35-49`
(categorical labels/strict scoring), `embedding_readout.py:146-150` (all positive
digit slots supervised, including unused tens), `attention_readout.py:14-30`
(position-aware learned-query readout), `igc_dispatch.cu:19-39` (readiness first,
then agreement/canonical checks). No attention-path causal experiment was run.

## Observations and causal distinctions

The two new-value cohorts have zero shared signed operand pairs. Operations
are32 each, but length/sign/template composition differs. Raw50/96 versus67/96
does not isolate template effects:

| lhs/rhs length | Seen templates exact/total | New templates exact/total |
|---|---:|---:|
|1/1|2/10|3/7|
|1/2|16/30|13/22|
|2/1|5/29|7/23|
|2/2|27/27|44/44|

Every wrong operand is single-digit absolute3/8; signs/lengths/operators match.
These absolute values are absent from all training operands. Digit3/8 are NOT
absent from multidigit positive labels: lhs/rhs ones3=331/336, ones8=418/422,
tens3=472/486 and tens8=397/406. Thus lack of any training exposure to the digit
symbol is falsified; single-value/slot generalization remains a real failure.

Across the two new-value splits, true ones3 predicts4 in48 errors and1 in2;
ones8 has one error predicting2. All66 padding errors predict the operand's
own true ones3/8. Only10 of51 ones errors match another operand's active digit.
This is a concentrated pattern, not dispersed random errors. Matching digit
values alone does not prove attention copying, misalignment or quantization.
Length stays1, so nonzero unused tens is invalid encoding, not a valid34/88.
The machine-readable audit includes the full sparse confusion by slot/length,
padding, per-magnitude accuracy and association counts; no prediction is repaired.

OLD64 contains48 positive requests and16 negatives, not fresh acceptance:

| Positive outcome | Two readouts | First-attention only, same readiness |
|---|---:|---:|
|selected, correct tuple/result|33|37|
|readiness no-call|11|11|
|disagreement|4|0|
|selected wrong tuple|0|0|

Embedding tuples are36/48 exact; first-attention48/48. Of11 readiness no-calls,
eight also have an embedding error and three have both tuples correct. Only
four disagreements are rescued while holding readiness fixed. Calling all12
embedding errors the sole cause of bypass would overstate causation. All16
negatives stay unselected. This descriptive counterfactual does not change policy.

The output mapper has708,481 parameters but only24 fit prompts/341 positions
(149 numeric).421 is total capture including6 development prompts/80 positions
(32 numeric). Development26/32->30/32 is four corrected numeric positions and
two still wrong in one question, under teacher forcing. These are correlated
positions, not32 independent responses. Old readiness111/112 concerns routing
on its separate dataset, not operand tuples. Neither metric establishes a usable
latent model. The small output pilot already records these limitations.

## Separately evaluated predicted-length composition

The later requested offline variant changes only unused tens selected by
PREDICTED length. Original arrays/scores stay immutable; malformed field
shape/ranges are refused, sign/length/active digits are untouched, and negative
zero or other active/canonical errors are not automatically repaired.

| Cohort | Raw exact | Composed exact | Recovered | Padding-only / active-only / both operand errors |
|---|---:|---:|---:|---:|
|seen values/new templates|96/96|96/96|0|0/0/0|
|new values/seen templates|50/96|68/96|18|18/5/23|
|new values/new templates|67/96|73/96|6|6/4/19|

Every wrong tuple in these cohorts contains one wrong operand, so the recovery
counts coincide here; generally operand and tuple denominators differ. This is
a changed composition contract, not proof the existing model learned padding.
On OLD64, normalizing both readouts changes1 primary and7 secondary raw tuples,
but agreement remains38/64 and readiness-on agreement33: no selected-request gain.
The independent first-attention-only counterfactual remains37, not the same lever.
No runtime/threshold/artifact changed and no original score was overwritten.

Metadata-only centroid feasibility found just one exact single3/control4 prompt
match against training when preserving all other expression/template details.
One match each also exists for controls1,2,9; no new-template match exists.
Sparse support cannot explain the overall50 failing3 operands. Cache token IDs
and per-position F32 states exist, but no centroid geometry was computed. The
linked protocol specifies token alignment, training-only normalized centroids,
head-specific pre-classifier representations and confounds before such a run.
LOVO research and later all-digits deployment training remain separate planning
tracks; original holdouts and the sealed acceptance set retain their identities.

## Debug ledger and preparation

Reproduction: frozen JSON, exact target recomputation and offline counterfactual;
no live process to attach a debugger to. Source tracing precedes a model fix.
Ranked hypotheses: (1) cohort composition confound is observed; (2) categorical
target/padding objective mismatch is contradicted by source and actual scorer
tests; (3) missing one-digit-value generalization is observed, mechanism open;
(4) attention copying/misalignment remains unproved by digit coincidence.
No proposed neural/padding fix is promoted to a root cause.

The [matched protocol](../plans/2026-10-29-igc-matched-input-experiment.md) freezes
pair/template/intent controls, strict metrics and future fit recipe BEFORE any
new learned result. Generator uses only a subset of the already audited pair
pool; it does not import/regenerate the sealed evaluation manifest. The source
pool's exclusion guarantee comes from its recorded original construction; it is
not a fresh independent reread of the sealed set. Historical exposure is explicit.

Observed test-first ledger: empty builder fails three behavioral assertions;
initial random critical selection fails signed-slot coverage; coverage-led
selection fixes it. CLI stub fails its unrecognized-pool refusal assertion, then
the identity gate passes. Audit stubs fail literal confusion/readiness counts;
association counts have their own red->green test. First real audit invocation
rejects a mistyped expected report digest before creating output; correcting the
constant to the recorded hash permits replay without changing source data.

Prepared corpus has1,536 rows (768 train/768 development),96 signed pairs,
288 pair-operation groups and balanced length/sign/op strata. Expressions are
matched across English/Thai, template families and compute/quote-copy intent.
Both splits contain +/-3/8 at both operand slots. Train/development pairs are
disjoint; development remains historically known. No >=3-digit claim or new ABI.
Generated rows SHA256 `bd250c5bb91ea0035caa580e9f11ca4753c8ccc38f5df8e5dcab63913d6c8681`.
Serialized corpus SHA256 `2a07d8b45ba2e4b02ddd571cdbe3fbb88b6493cb2a3aadb84e89c947ed0d7e19`.

Preparation uses no GPU/model, no pagefile/profile/backup change or network data
transfer. Raw one-tap80x4x2560F32 cache would need5,033,164,800 bytes plus masks,
excluding temporary archives and logs. Capture is NOT run; resource admission
and disk reserve must pass first. Basic live smoke remains a separate pending
wiring gate; statistically meaningful quality/Hub benchmarking follows usable
integration. No retained model, readiness threshold or dispatch policy changes.

## Final preparation verification

Thirteen lightweight unittest cases pass: eight new matched/audit controls plus
five existing diagnostic/readout accounting checks. Full pinned audit replay
passes; output `igc-development-audit-03.json` SHA256
`e2eb06345a871aa17a79a05f7d3a9c69532c13d7f5b9f2505852dbc855f94b15`.
Two separately created corpus directories have byte-identical files. Raw source
corpus/predictions and the six pinned evidence files remain unchanged. A read-only
peer review finds no important implementation issue; it does not validate future
capture/training, attention causality, model quality or deployment.

Relevant source identities at audit baseline (working-file bytes):

| Source | SHA256 |
|---|---|
| `readout.py` | `098a58794ca0469bb9afe818cc058892ffcfb19ee592afc8bc87a1bbdc8e0b4d` |
| `attention_readout.py` | `04b200586afcacbed7344886f7add14e98b03a9fcb71c18d41bdd69de0317a34` |
| `embedding_readout.py` | `6aa371c5362968d73ac43538eecd7ecb058d993dd9f7966f9f2b7e12cba3ffa2` |
| `balanced_corpus.py` | `99c01a490fb9365cc0f3339fb58dc10d7c4fd3720e656f9acce9413d1b0fce75` |
| `igc_dispatch.cu` | `24ead6d6054f179e3ad692f38e041941899ef39642b8736784f6536a9e24b3bc` |

No native runtime file changes, so CUDA/HIP/SYCL builds and full-model parity are
not rerun for this Python/documentation checkpoint. The whole model/backend test
suite is not claimed. Exact named simplify/security-review skills remain
unavailable; manual source/simplification/security review and the recorded tests
are the scope of this check. No scanner, training or centroid run is claimed.

Next execution needs a new identity-bounded matched-manifest adapter: existing
capture/fitter loaders intentionally accept the old expanded-v2 manifest only.
Do not relabel the new data to bypass that guard. This remaining adapter plus
resource admission is explicit preparation debt, not a completed collection run.
