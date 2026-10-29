# I2 nullable digit-attention supervision

Diagnosis first: the embedding control attends to the annotated active digit at
argmax305/305 on new-values/seen-templates, but has only77/96 complete tuples;
mean true-position mass is0.7501. For `93 * -3`, even the correctly located left
tens query predicts0. First-attention control has0/968 digit-query argmax matches,
while many tuples are correct, so its values can carry distributed information.
Attention location alone is not a causal explanation or a proof that the source
activation lacks digits. CPU replay agrees with all saved CUDA predictions.

One next discriminator: add **one** mean cross-entropy alignment term (weight1)
to the unchanged nine-field contextual embedding readout/main objective. No new
parameters, hidden features, final query, digit layout or executor ABI. The four
digit queries learn actual annotated ones/tens positions. An unused tens query
targets the observed `<think>` marker at the final pre-anchor row79, a bounded
learned null slot; its existing main numerical label remains0. Validate this
marker/window for every row, rather than fabricating cached activation rows.

Annotations derive only from the known synthetic expression spans and actual
bound tokenizer bytes. Reconstruct all10,624 prefix/cache index/mask identities
independently, distinguish binary subtraction from unary minus, and reject
ambiguous/missing spans. IDs/span labels enter only supervised losses and
provenance, never neural features. Apply alignment only to positive requests;
negative readiness/operator labels remain supervised by the original objective.

Use the same qualified expanded native **embedding** cache,96964-parameter network,
seed20261031/common initial values,70 epochs/batch64/AdamW0.001/0.01/global cosine
to0.00001/no clipping, deterministic F32/math SDP/TF32 off, four CPU threads.
Final weights only,600s fit/outer720s. The only changed training mechanism is the
additional alignment objective. Frozen comparator is embedding control state
`ec7273bed946ea3fa03ca375f453a602de82b34a96cdca88da880945505c427a`.
Do not compare this as a one-variable change against the first-attention arm.

Report exact tuples, operator/sign/length/active digits/padding consistency and
false calls separately per existing development axis; active/null attention
accuracy and mass are diagnostics. If localization tightens but numeric errors
remain, localization is insufficient. Positive gains do not close I2 while
negative false calls remain. Preserve the failed left/PAD view and old predictions;
do not repair them or relax canonical validation. Acceptance64 remains untouched.
This is not inference benefit, spontaneous use or model-body training.
