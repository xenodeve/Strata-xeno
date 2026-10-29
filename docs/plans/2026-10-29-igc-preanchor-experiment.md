# I2: controlled pre-anchor input-mapper experiment

This protocol is frozen before fitting the candidate. It compares input readouts,
not execution or whole-model benefits. PRD R11 and all wider capability gates stay
in force; neither a forced call nor a small arithmetic probe completes the project.

## Question and frozen inputs

Does bounded attention over actual pre-anchor states extract operator/operands
better than one final-head vector on this small signed-integer domain? The prior
334,516-parameter final-vector MLP fit training data but got 0/12 positive held-out
operand tuples. Those sixteen examples are now **known diagnostic development**
data. They cannot be called unseen after architecture selection.

Keep the existing 64-row corpus unchanged: 48 fit rows, 16 diagnostic rows,
seed20261029, canonical row SHA256
`fce8f8c80d3c370804fec675dd99065ef1922afb562d3990ea9b386ed4df954b`.
Collect both final mixed vectors and bounded F32 pre-anchor HC rows from the same
new cohort/candidate. Use the selected Gyro-M 262K/five-worker Hub args/env/cache/
offload settings, source/build identity and owned cleanup. No private-commit cap.
Capture is diagnostic, with timing/resource safety bounds; it is not a serving
performance baseline. Preserve missing rows and masks rather than fabricating
source states or disabling normal prefix caching.

Freeze `evaluation_manifest()` separately: seed20261102, 64 evaluation-only rows,
16 per no-call/add/subtract/multiply class; English and Thai templates, disjoint
operand pairs and prompt templates from the known 64. Signed magnitudes remain
≤99, with parenthesized signed operands. Save its manifest/hash before fitting.
Never optimize, normalize, select epochs or tune thresholds on these new rows.
Frozen canonical evaluation-row SHA256:
`0b6158b0df2ce66edca7bad158b75e32b36e93543a40230891d8d761933bc5ef`.
After candidate weights are frozen, collect/evaluate them once. A failed new gate
is retained; further changes require a fresh declared experiment/evaluation set.

## Fixed variants and learning budget

Baseline: the existing 2560→128→52 GELU MLP, retrained from its fixed seed on the
new cohort's final mixed vectors. Fit normalization only on 48 training rows.

Candidate: maximum80 observed pre-anchor rows, each four HC streams ×2560F32.
LayerNorm2560 and a shared Linear2560→16 encode each stream; concatenate the four
streams to64 features per token. Add a learned relative-to-anchor position
embedding (80×64). Nine learned64-dimensional queries, conditioned by a
Linear2560→64 projection of the train-normalized final mixed vector, attend to
learned64-dimensional keys/values. Each query has its own categorical readout.
No prompt text, token IDs, expected arithmetic answer or oracle operand is a model
input. Token IDs bind observation provenance only. Mask absent/padded rows before
projection and attention; refuse samples with no observed state for this variant.

Both variants: CPU-only Torch, four threads, seed20261029, AdamW lr0.001,
weight_decay0.01, full48-row batch, 300epochs. Candidate timeout180s/outer210s,
not an unbounded training job. Train only the input readout; do not backpropagate
through frozen Gyro or claim an output mapper. Save safe tensor weights,
normalization, architecture and source/corpus/executable identities.

## Rubric, uncertainty and non-regression

Score no-call/operator classification, exact canonical operand tuple on positive
rows, joint correctly routed tuple, missed calls and false calls independently.
A wrong sign, length or digit is a wrong invocation, even if the arithmetic
executor would compute those wrong operands exactly. Check the independent
label/metric negative controls before fitting.

The new evaluation gate for this two-digit feasibility variant is all48 positive
operator/operand tuples correct and no false call on16 negatives. Report raw
counts, denominators and confusion matrices even on failure. This finite gate is
not a deployment error bound, 62-digit coverage or general task-quality proof.
Diagnostic differences on the known sixteen rows are not new generalization
claims. Do not trade tuple exactness for a score or change the corpus/rubric after
seeing predictions. No default activation follows from this experiment.

Correct input extraction, correct native execution, faithful model result use and
improved successful-task token/time outcomes remain separate gates. I3/I4 and
actual Hub paired quality/latency/resources/cache/cancel/stability checks still
follow. Freeze benefit-experiment baselines, accounting and uncertainty criteria
before those measurements, including normal cached external-tool controls.

## Next action

Collect the frozen64-row pre-anchor cohort with candidate
`f8030aa2d26e48e8383e58a15fb12d49a468bb94bdc66495caa6b1ed46f483a0`, validate
positions/source/masks and identities, fit both variants, then lock weights before
the new evaluation cohort. The real short-read observation seam already passed
two-request smoke; other paths remain explicit coverage debt.
