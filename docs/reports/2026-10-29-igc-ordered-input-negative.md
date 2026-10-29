# I2 left-aligned digit/PAD experiment: negative result

The separately frozen neural-view experiment completed and is **not selected**.
It changes neither the little-endian exact ABI nor historical predictions.
All common initial parameters match the legacy contextual embedding control;
only two new PAD logits add130 parameters (97094 total). Three tests cover that
control, neural/ABI distinction and strict rejection without repair.

| existing development axis | legacy embedding control exact positive | left/PAD exact positive | left/PAD false calls | left/PAD noncanonical positive |
|---|---:|---:|---:|---:|
| seen values/new templates |50/96|14/96|32/32|20/96|
| new values/seen templates |77/96|64/96|0/32|29/96|
| new values/new templates |33/96|11/96|32/32|25/96|

Both controls fit7680/7680 training positives and no false calls on2560 training
negatives. The left/PAD run is not a fix for generalization, despite the original
failure's concentration in single-digit/padding outputs. First left-operand digit
is correct96/96 across all axes, but length/second-digit/right-operand and intent
errors remain. Do not promote that isolated field gain into a quality win.

Comparison uses the same **native embedding cache**, corpus, seed,70-epoch
AdamW/cosine recipe and contextual body as `igc-cuda-cosine-fit-01`, state
`ec7273bed946ea3fa03ca375f453a602de82b34a96cdca88da880945505c427a`.
It is not a one-variable comparison against the later first-attention feature
checkpoint. The new loss necessarily includes explicit PAD classes; no claim
that the two differently defined output distributions are bitwise equivalent.

Run `igc-ordered-input-fit-01` DONE0,230.985s; final weights only, no model load.
State `5bb85bd5372d04638801960f0b877c4a4f1e22573e14436120c576208a4eedfe`.
Cache `2ba5242d128fcfef111b342ca4ee79443dc2c2e55de04427c18d41cdf1e76315`.
Rows `fe0e9d86210b4c0c7ef544a7b4a5982dc9c8f53c3a3eebada7a2e34b2356fa7c`.
Recipe [here](../plans/2026-10-29-igc-ordered-input-experiment.md), exact command,
loss/history/per-field predictions in the private task evidence's identically
named log/directory. Existing Torch2.11+cu130/RTX5060Ti, no downloads or installs.
Acceptance64 untouched, no production format loader or serving change.

Next input work must discriminate a stronger mechanism, including learned
attention's actual operand-position alignment and intent evidence, rather than
repeat this view change or relax canonical validation. Native output forward
integration can proceed independently while I2 remains open.
