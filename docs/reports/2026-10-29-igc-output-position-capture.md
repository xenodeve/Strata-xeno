# I3 native output-position capture and I2 field diagnosis

## What completed

Capture03 completed all421 teacher-forced positions across24 fit and6 development
questions. Each position is the real deployed Gyro-M quantized model's final head
input and full248320-entry native logits, with canonical prefix/position binding.
Thirty exact CUDA results match the independent offline integer oracle and remain
bound to their initial invocation anchors. No expected next token enters mapper
features. The dataset is for output learning; I2 input selection remains unpassed.

The actual selected Hub profile,262K context,5 workers, cache/offload/RoPE settings,
Vision initialization and launcher GPU mapping were retained. Only the isolated
main-based candidate executable, task-owned logs/profile output and opt-in
observation paths differ. Direct canonical token requests/teacher forcing are not
HTTP latency or a serving benchmark. No private-commit cap or served setting changed.

Collection completed in70.656s, including boot/requests/cleanup. This number is
dataset preparation time. Passive295-sample observations saw maximum single owned
process private commit46,578,860,032B, maximum concurrent owned process sum
48,632,705,024B, minimum available physical10,349,809,664B and minimum system commit
headroom14,746,435,584B. These are sampled values, not exact instantaneous peaks.
The parent job's peak is not the model's peak; do not substitute it for these values.

Corpus SHA256 `913fbdf842ce9cc55862200167147dee26f116f81e99ea5792e49ccb817725e3`.
Records `7a7269fd4c8f477cf3ecd1280d6a960599d80c7f45569da696e9d1689e17ba40`.
Report `85ff9560f6c0ea22d788b9c8843fe9418280597179e379003038c85082062800`.
Source manifest `b43f886c7765480d0ff7c54982dff504ce8b42df17184d7b19840253373c0fb7`.
Executable `ae6c485fb6b765ae4005183bac1712508b801dbfb1da35ee46bae50966597d7d`.
Tokenizer binding `889964f986b93c35102f25f5e396efeed65fcb180429c35d97536fb7f6b72106`.

Private evidence names: `igc-output-positions-03/{report,records,memory}.json`,
`igc-output-position-collection-03.log`, under the existing task evidence root.
The exact collector source was copied immediately after the completed run with
no intervening edits. Its hash `676a95c6e7437385e16937c210584443b9d6c31a3ea28b465bd63e482c7ecabc`
is explicitly a post-run archive, not a claimed pre-execution source pin.

Capture01 refused the conservative collector's unsupported Vision guard before
loading the model. Capture02 initialized Vision but lacked the launcher's inherited
GPU mapping, so the engine correctly refused its secondary arena. Both owned trees
drained. The corrected collector requires `CUDA_VISIBLE_DEVICES=1,0` and default
`CUDA_DEVICE_ORDER` before allocation. These were harness defects, not Gyro failures.

## I2 unused-digit check

The final contextual input checkpoint remains a failed development result, state
`519fbc196993c94676f4031f7bd45ad9abf661f8b7b4fd3b9d6358204b5294e2`.
Positive routing/operator predictions are all correct in the inspected development
splits; remaining operand errors are in single-digit examples. Signs/lengths and
active tens of two-digit operands are correct.

The canonical target for an unused tens slot is0. `targets()` writes0 and the
training loop applies cross entropy to **all positive operands in every field**,
including unused tens. It does not mask that slot or use an ignore index. Thus
the observed error is consistency/generalization, not an objective/ABI mismatch.
CPU final-checkpoint replay of training `-55 + -1` gives the right canonical
padding and per-example unused-tens CE2.3841855e-7. This is a new diagnostic forward,
not the original CUDA batch's unreduced loss capture.

| split | operand padding only | operand active only | operand both | correct operand/total | correct tuple/total |
|---|---:|---:|---:|---:|---:|
| seen values/new templates |0|0|0|192/192|96/96|
| new values/seen templates |18|5|23|146/192|50/96|
| new values/new templates |6|4|19|163/192|67/96|

For these inspected splits, each wrong tuple contains one wrong operand, so the
tuple error-category counts happen to equal the operand counts; the denominators
are different. No historical prediction was repaired and validation stays strict.
`93 * -3` independently predicts the wrong active ones4 and unused tens3 for the
right operand; their replay CE is7.591619 and2.163802. Padding alone cannot fix it.

There are665 operand/operator pairs carrying both positive and negative intent
labels across fit plus development, but no matched pair confined to development.
Therefore existing held-out data cannot isolate intent while holding numbers
constant; a new separately frozen paired diagnostic is needed. This does not
consume acceptance64. No overall quality improvement follows from positive-only
template transfer while negative false calls remain30/32 and32/32 in two splits.

## Next gate

Pure planning/reader/metric tests:7 passed. Reconstructed421 prefixes/targets and
source-bound records validate. The separate frozen I3 fit contract requires full
native-head logit equality before optimization, final-weight teacher-forced
metrics and result-content controls. Native NN injection, independent free
generation, spontaneous use, I2 readiness and practical Hub latency are still open.
