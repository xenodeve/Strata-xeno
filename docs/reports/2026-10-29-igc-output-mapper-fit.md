# I3 first late output-mapper fit: partial development result

## Result and scope

The frozen70-epoch fit completed, final checkpoint only. All421 native mixed rows
first reproduced their full248320-entry Q8 logits **bitwise** through the qualified
native head. This qualifies the captured-state/native-loss seam, not full-model
backward. Only the708481-parameter gated result mapper trained; the Gyro body and
base head stayed frozen. The backward is the explicitly declared dequantized-head
transpose STE, while forward uses Strata's actual quantized head.

| control | fit numeric symbols | development numeric symbols | development other symbols | development complete numeric reference sequence | development complete reference sequence |
|---|---:|---:|---:|---:|---:|
| learned, correct result |149/149|30/32|48/48|5/6|5/6|
| unchanged/hard-off |124/149|26/32|30/48|1/6|0/6|
| failed result |124/149|26/32|30/48|1/6|0/6|
| zero numeric content |135/149|26/32|46/48|2/6|1/6|
| swapped question's numeric content |133/149|26/32|45/48|3/6|2/6|

These are **teacher-forced agreement with a specific short reference continuation**.
In particular, word/template agreement is not independent answer correctness and
the base's0/6 complete reference sequences does not mean it cannot solve them.
Prior correct reference tokens are consumed at later positions. No autonomous
response or in-engine adapter injection was run. Oracle operands isolate I3;
I2 has not passed. The result-content controls show sensitivity/use in this pilot,
not general62-digit correctness or spontaneous capability selection.

Development examples `92 - 18 = 74`, `-96 + 35 = -61`, `37 - -43 = 80`,
`69 + 34 = 103` and `42 * -20 = -840` match every reference position. `12 * 64 = 768`
fails: at both occurrences of the answer's first digit, expected7 but predicted1.
Do not combine predictions under teacher-forced prefixes into a claimed generated
answer. One failed question is enough to leave the result-use gate open.

Hard-off and failed-result mixed activation bytes match the original; their
predictions and CE match base throughout. There is no failed numeric result
silently treated as successful zero. No source/protocol changed during fitting.
Fresh acceptance64 remains untouched.

## Reproduction and accounting

Recipe: [frozen fit contract](../plans/2026-10-29-igc-output-fit-pilot.md),
`python -B -m tools.igc.output_fit` with the frozen capture03, source manifest,
tokenizer, model, native-head library and separate capture/fit protocols. Existing
Torch2.11.0+cu130, RTX5060Ti, deterministic F32/TF32 off.70 epochs/batch16,
seed20261105, AdamW0.001/0.01, cosine to0.00001, no clipping. Final fit loss
0.0059957054; final development CE0.13499475. Fit plus final controls192.759s;
this is offline training time, not inference latency or a Hub benchmark.

State SHA256 `241ef7032aed2bb9b3e7a3bc8bddd990c963c1d182c63ea968136e1e863c0337`.
Native library `d1716cd48a92d6e9d57bfe3e1fe0f8b1d00eda21210443b899cfa798cebba381`.
Capture report `85ff9560f6c0ea22d788b9c8843fe9418280597179e379003038c85082062800`.
Corpus `913fbdf842ce9cc55862200167147dee26f116f81e99ea5792e49ccb817725e3`.
The model/source identities and exact command are in the private task evidence
`igc-output-fit-01/{report,predictions,qualification}.json` and
`igc-output-fit-01.log` (DONE0). No model download, served change or memory cap.

Torch peak allocated161515520B/reserved197132288B **excludes** the native head's
675430400B weight allocation and CUDA runtime allocations; it is not total VRAM.
The base model body is absent during this offline cached-state fit. Do not compare
these numbers with full Hub resident memory. Process/system snapshots are retained.

## Next work

The completed component is native capture plus offline result-mapper learning,
not the I3 phase. Native adapter forward/injection, free-running complete answers,
range/language coverage and production transaction/ownership remain open. Resolve
I2 routing/single-digit generalization independently, preserve the failed pilot,
and qualify any later native mapper implementation against these frozen tensors
before full model evaluation. Practical Hub benchmarking follows usable integration.
