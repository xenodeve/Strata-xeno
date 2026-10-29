# I2: early-seam comparison and query-route diagnosis

The input-quality gate still fails. Changing the observation location did not
produce one exact positive diagnostic tuple with this pilot recipe. Removing
the measured final-query shortcut also did not fix it. No new evaluation data
was consumed, no output mapper was trained and no feature was enabled.

## Frozen cohort and fits

`igc-i2-early-fit-01` completed all64 requests in231.859s, ready at18.500s,
with candidate `ae6c485f…`, native/tokenizer/profile identities unchanged and
owned cleanup drained. Sampled process private peak47,012,835,328B, minimum system
commit headroom13,270,593,536B and physical available9,174,061,056B. No private
cap or served configuration change. These are diagnostic resource observations,
not serving performance figures.

The [three-seam protocol](../plans/2026-10-29-igc-early-input-experiment.md)
used the same architecture/seed/budget and final-mixed query from this cohort.
All three arms fit48/48 operator classes and36/36 positive tuples. Diagnostic
data are the already known sixteen examples, twelve positive/four negative:

| input stage | correct operator /16 | exact lhs /12 | exact rhs /12 | exact tuple /12 | false calls /4 |
|---|---:|---:|---:|---:|---:|
| final-HC |12|0|0|0|0|
| native embedding |15|0|1|0|0|
| first native attention, before layer-zero MoE combine |11|0|1|0|0|
| embedding, learned queries only (separate declared arm) |15|1|0|0|0|

First three fits took15.344/46.219/14.734s respectively; the complete serial
driver took79.813s. They are training durations, not a speed comparison. States:

- final `a22ea4098210135cc7622fd8a38756fcdf7fbe384f1a92c1d01037128837ee32`
- embedding `a46fdbfb18f2bc5f60d9cdf0e3383ec44f72218c9a81c982bf5f39a2691d8633`
- first-attention `4fd5ec9b0dc80f44583f570c1285e99708a2c4ca543e97065014b9940322245f`
- query-free `0b67e2f40721c1961e8d698f4a465d314b5f1cbaa19634e8ac8e14978ed60c51`

All first three have227396 parameters. The query-free arm has63492, removes
only the context projection/conditioning, keeps the position/order features,
fits36/36 training tuples in13.766s, and still gets0/12 diagnostic tuples.
Its [protocol](../plans/2026-10-29-igc-query-free-experiment.md) was written before
fitting. This negative result falsifies the claim that removing that route alone
solves generalization; it does not remove the observed shortcut from the prior fit.

## Interventions, ordering and coverage

Saved embedding-model intervention results, on known data only:

| input intervention | training exact tuples /36 |
|---|---:|
| observed |36|
| rotate states, keep positions |36|
| zero states, keep final query |21|
| zero final query |0|
| rotate final query across samples |0|

Rotating complete state/position pairs is an invariant control (maximum logit
difference2.862e-6). The final query is necessary for this fitted model's perfect
training performance. These interventions do not establish that every internal
feature is ignored or explain the full generalization failure.

Order features are present in the network: learned80x64 relative positions are
added before key/value projection. A structural test changes outputs when states
move but positions stay fixed; moving complete pairs preserves outputs. The
query-free test proves that changing final queries cannot change its output.
Six attention tests and two diagnostic-accounting tests pass, plus three existing
readout accounting tests (11 total). Tests cover reversed digits, swapped
subtraction operands, nonpadding digits, noncanonical leading/negative zero and
unused nonzero padding. These are offline training diagnostics, not production
prediction repair.

The pilot has only47 distinct absolute operand values. Every sign/length class
is represented in training, as are all ones digits and right tens digits. Left
tens digit9 is absent in training but present in diagnostics. This is a concrete
coverage gap, not an explanation for all twelve failed tuples. Future development
must separate seen-value/new-template, new-value/seen-template and both-new axes.

## Label and capture audit

The saved64 request prefixes contain the expected expression exactly. Every
expression token is inside the observed80-row window and before the answer
anchor; maximum distance from the anchor is23. All128 labeled operands roundtrip
to the independently packed executor signed-magnitude little-endian ABI.
Subtraction keeps lhs/rhs order. Targets, sample IDs and saved prediction order
agree. Padding is masked before projection/attention, normalization uses only
fit rows, and no missing rows are filled from KV.

The diagnostic errors are not predominantly swapped operands or reversed digit
order (zero of each). The learned-query-free arm gets left ones3/12 and left
nonpadding tens2/11; right ones3/12 and nonpadding tens1/9. It emits three invalid
operand encodings across24 predicted operands. Invalid encodings are failures,
never normalized into correct answers. Full per-field counts are retained in
`igc-early-experiment-01/audit.json`.

Evidence IDs: `igc-early-three-fits.log`, `igc-early-experiment-01/summary.json`,
`interventions.json`, `audit.json`, `igc-query-free-fit.log`, and
`igc-early-label-position-audit.log`. Protocol SHA256 values are in each fit report.
The new evaluation64 is still untouched. This48-fit pilot is not a reproduction
or refutation of IGC's10k-sample study and does not prove information is absent
in later activations or that quantization caused the failure.

Next work remains input-head development with declared data/coverage controls,
followed by output mapping and committed-state integration. A genuinely usable
integrated path is followed immediately by actual Hub end-to-end benchmarking.
