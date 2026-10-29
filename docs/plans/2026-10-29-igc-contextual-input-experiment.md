# I2: contextual input-proxy comparison

The frozen balanced query-free baseline completed: 1,457/3,072 training tuples
and 20/96, 6/96, 5/96 development tuples across the three axes. Template-new
negatives produced 16/32 false calls on each of the two affected axes. No fresh
evaluation was consumed. The baseline remains a failed, preserved experiment.

Source inspection shows that its pre-attention token features are pointwise
embedding projections plus positions. Token features cannot incorporate a nearby
operator, negation or another digit before the learned-query pooling step.
The new-value axes show 23–24 left-digit reversals per 96 positive samples,
against one on the seen-value axis. This motivates testing contextual token
features; it does not prove insufficient context is the only failure mechanism.

## Frozen change

Reuse `igc-balanced-embeddings-01` exactly: row SHA256 `193f34b2…`, native cache
SHA256 `f9d24a28…`, tokenizer/source/wrapper identities and the original cache
preparation protocol SHA256
`f4a960fd95ae5bfec9f817364dada83422271732b1c27b11a0616cb7e968f340`.
The loader verifies that original preparation protocol separately from this fit
protocol; do not overwrite historical cache metadata.

Change only the query-free input architecture: after its per-stream projection
and learned position addition, apply one `TransformerEncoderLayer` with width 64,
four heads, feed-forward width 128, GELU, dropout 0, batch-first and norm-first.
Mask absent rows as keys; keep the existing cross-attention mask. Keep all common
weights' initialization identical under seed 20261030 by initializing the new
block after the baseline modules. Total parameters are 96,964, versus 63,492.
No final mixed query, token IDs, text, oracle operands or arithmetic answers
enter the network. The new block is part of the trained activation proxy, not
an extra layer in the frozen Gyro model or a host parser.

Keep the same 4,096 fit rows, three 128-row development axes, seed 20261030,
AdamW 0.001/0.01, deterministic batch 64 order, 30 epochs, four CPU threads,
600-second fit/660-second outer bound and final-epoch-only selection. Do not
change data, extend epochs, tune thresholds or select weights after viewing
predictions. The already completed baseline is the comparison; do not rerun it.

## Checks and acceptance

Before fitting, test common initialization, padding exclusion and finite
gradients for the added block. Persist architecture and both protocol identities.
After fitting, use the same exactness/readiness/operand/canonical-encoding rubric
and the same three development axes. Report reversals and false calls as well as
aggregate tuples. An improvement tests accessibility to this readout/recipe;
it does not establish absent information in later Gyro states or a quantization
cause.

Eligibility remains all 3,072 training positive tuples, all 96 positives on each
development axis, and no false calls. Otherwise retain evaluation64 untouched,
record the failed arm, and declare any next experiment separately. There is no
new output mapper, model execution, serving/default change or benefit claim in
this CPU input-head experiment. Usable integrated IGC is followed immediately
by actual Hub end-to-end benchmarking.
