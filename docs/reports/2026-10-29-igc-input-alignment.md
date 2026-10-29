# I2 digit-attention diagnosis and one-objective experiment

## What the saved checkpoints show

The offline diagnostic independently reconstructs every one of the 10,624 native
prefix/cache index/mask identities and digit spans using the bound tokenizer.
It distinguishes binary subtraction from unary negative signs and rejects
ambiguous spans or missing rows. IDs and positions only annotate the supervised
targets; they never become ordinal NN features. Full embedding/context cache
hashes are checked. CPU replay matches every saved CUDA prediction on the 288
positive development cases. No model execution or training occurs in this step.

| development axis | embedding query argmax on active digit | mean mass at digit | first-attention query argmax on active digit |
|---|---:|---:|---:|
| seen values/new templates |308/338|0.5720|0/338|
| new values/seen templates |305/305|0.7501|0/305|
| new values/new templates |268/325|0.5567|0/325|

These numbers are not a causal explanation. The first-attention arm can return
correct tuples without attending at the literal numeric token, so its contextual
values carry information elsewhere. Conversely the embedding arm's correctly
located tens query for `93 * -3` still predicts 0 rather than 9. Localization alone
cannot explain every value error. This falsifies a simplistic wrong-position-only
diagnosis and motivates a narrowly controlled objective test.

Diagnostic source SHA256
`f58a05c157d1134e906de9641c652467ee5654dcdad0079f0f273817df34ae5f`.
Labels `bbce82b76c96e21b8fb65ad74ccb11266c58c4ac2c711677ff36774a66f6654e`.
Private evidence `igc-i2-alignment-diagnostic-01/{report,details}.json` and matching
wrapper log (DONE0). 28.359 seconds is offline validation/analysis time, not model
or runtime latency. Existing CPU Torch 2.14, four threads; no gradient/body change.

## Frozen nullable-attention experiment

[Protocol](../plans/2026-10-29-igc-input-alignment-experiment.md): unchanged
contextual embedding readout, 96,964 parameters, same initialization/seed,
70 epochs, batch 64, AdamW/cosine and fixed data as embedding control
`ec7273bed946ea3fa03ca375f453a602de82b34a96cdca88da880945505c427a`.
The only training-objective addition is mean cross entropy of four digit queries
against their annotated token position (weight 1). For unused tens, the target is
the actual observed `<think>` marker in the final pre-anchor row 79. The original
numeric label remains 0 and strict ABI validation remains unchanged.

No new model features or parameters, final-query shortcut, digit-layout change,
host calculation or inference-time annotation is introduced. Native-cache values
remain exact. Four tests cover annotations (including compact subtraction), absent
slots, and positive-only gradients/observed-target enforcement. The original
failed left/PAD experiment and all predictions remain intact.

| development axis | old embedding exact tuple | new exact tuple | old false calls | new false calls |
|---|---:|---:|---:|---:|
| seen values/new templates |50/96|49/96|28/32|16/32|
| new values/seen templates |77/96|94/96|0/32|0/32|
| new values/new templates |33/96|44/96|32/32|16/32|

Both fit all 7,680 positive tuples and have no false call on 2,560 fit negatives.
All positive development operators are correct in the new run. Active/null
attention accuracy is respectively 220/338 and 31/46, 301/305 and 79/79,
212/325 and 43/59. Improved tuple extraction does not imply every localization
metric improves, and this single experiment is not an overall quality win.
The two new-template axes and false calls still fail I2. No fresh acceptance is
permitted, no artifact is promoted to serving, and no output quality/speed gain
for the full model is claimed.

Final state `72a0020591a7fb59f0f74e53ab1eadaee5a31ec6ae3b45744c20e3dea99cdefd`.
Cache `2ba5242d128fcfef111b342ca4ee79443dc2c2e55de04427c18d41cdf1e76315`.
Rows `fe0e9d86210b4c0c7ef544a7b4a5982dc9c8f53c3a3eebada7a2e34b2356fa7c`.
Private evidence `igc-aligned-input-fit-01/{report,predictions}.json`,
`input.safetensors`, matching wrapper log (DONE0). Final checkpoint only,
257.961 seconds of offline fitting/evaluation. Existing Torch 2.11+cu130 on primary
RTX 5060 Ti, deterministic F32/math SDP/TF32 off; language weights frozen.

Next isolate semantic readiness using actual final Gyro activations and matched
positive/negative prompts, while improving extraction without weakening canonical
checks. The output/runtime executor connection proceeds independently. Original
acceptance 64 remains unconsumed; dynamic reasoning-time triggers and production
transactions are still later gates, not established by this fixed-anchor pilot.
