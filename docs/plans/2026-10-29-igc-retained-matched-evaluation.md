# Retained first-attention mapper: matched development evaluation v1

Issue #276. Freeze before inference. This is no-grad evaluation, not a fit,
release acceptance or a change to live routing. Sealed acceptance64 stays unread.

## Fixed identities and execution

Use the completed 1,536-row matched cache from collection commit a136ce22.
States SHA256 `6757cde83254fff23010d26d720bc579880e2854750a4ee163b5224b8e12d82a`;
semantic manifest `88d4b39a3c2542c644f5eac4647e90f90780eb7bb1dcb3a9d6f5c324edbe25ff`;
cache protocol `2de4d752b3df50a3623624f9acbc95e874b80d0acd02fdc3ee458a679cf75e0c`.
The cache must pass existing content, shape, source and manifest verification.

Retain `input.safetensors` SHA256
`519fbc196993c94676f4031f7bd45ad9abf661f8b7b4fd3b9d6358204b5294e2`.
It is the existing 96,964-parameter contextual InputReadout: learned queries,
no final activation query, native first-attention input, 80 positions, 4 streams
of width 2560, nine categorical outputs. Strict tensor-name/shape/F32/finite
checks and the original fit receipt must agree. No optimizer, gradients,
normalization fit, threshold selection or weight updates are permitted.

Evaluate in CUDA F32 with deterministic algorithms, TF32 off, math attention,
four CPU threads and the existing GPU visibility `1,0`. Use at most 64 rows per
batch, 24 batches total, bounded file-row reads and no whole-cache mapping.
The predeclared inference bound is 180 seconds, outer owned-job bound 240 seconds.
Keep source, cache, protocol and weight hashes before and after. Store raw
categorical predictions separately from all derived metrics/counterfactuals.
The existing offline component preflight requires 8 GiB physical, 12 GiB system
commit headroom and 2 GiB GPU free; these are admission checks, not Hub caps or
measured peaks. Require 128 MiB disk free for bounded outputs; retain old files.

## Metrics frozen before prediction

Positive means compute intent; raw exact tuple requires all nine categorical
fields, including ready/operator, signs, lengths, active digits and canonical
zero padding. Report operand-only exactness separately. For positives count
raw no-call (operator zero) and wrong nonzero operator. For quote-copy negatives
count raw false-call candidates (any nonzero operator) and no-call. This is the
single first-attention head, not the separate readiness/dual-head live policy.

Score signs, lengths, active digit positions and unused padding slots on
positive rows with separate numerators/denominators. Classify operand errors
as padding-only, active-only, both, or other sign/length errors. Preserve
per-field and digit-position/true-length confusion counts. Invalid shape,
class range, duplicate row identity or noncanonical raw predictions do not
silently become correct. Negative operand labels are not execution targets.

Report raw and separately composed summaries by split, family, language,
length pair and the split/family/language/length intersection; also retain
pair-operation expression-group summaries. Compare seen/new families within
the same development expression, language and intent. Compare compute and
quote-copy within the same expression, family and language. Record paired
success transitions and their joint success rather than comparing unmatched
cohort percentages. No result may tune the retained mapper or gate.

For uncertainty, use the signed pair (`pair_group`) as the independent cluster,
not rows, digits or the three nested operation groups. Equal-weight pair means
with 4,096 bootstrap resamples, fixed seed 20261109, percentile 95% interval.
Report cluster counts and the estimand; with fewer than two clusters return no
interval. Apply this to positive exactness, negative false-call rate and paired
family/joint-intent contrasts. Descriptive development intervals do not establish
fresh-population acceptance or a causal attention explanation.

## Separate padding counterfactual

Apply the previously declared `predicted_padding` composition to a COPY of raw
predictions: predicted length class zero clears only the unused tens slot;
class one changes nothing. Do not use true length or alter active digits, signs,
operator or length. Report raw/composed exact transitions including regressions.
This changes the composition contract; it is not learned padding accuracy, a
replacement raw score or a deployed policy. Preserve malformed/refusal semantics.

## Follow-up, not part of this run

Return a fixed training proposal after describing these results; do not train.
The previously frozen matched fit recipe and strict development gate remain
unchanged unless the owner explicitly changes them. Keep old weights/results,
new development evidence and sealed acceptance distinct. Centroid work is
optional exploratory analysis under its existing train-only matched protocol;
it must not delay this evaluation. Full-Hub live admission and end-to-end
quality/latency evidence remain independent open gates.
