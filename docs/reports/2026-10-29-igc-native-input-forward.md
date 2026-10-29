# Native input forward matches both frozen feature arms

`InputMapper` reproduces the existing kind1 query-free contextual neural readout
in CUDA: per-HC LayerNorm/projection, learned positions, pre-norm four-head
transformer/exact GELU, learned-query attention and52 field logits. Inputs are
actual right-aligned80×4×2560 activations and byte masks; no token IDs, host
arithmetic, literal parser or oracle operands. Weights/workspace load once with
capacity1..64 on one explicit stream; `run` allocates/synchronizes nothing. Hard
off returns before arguments/device work. Errors preserve caller output bytes.
This standalone diagnostic is not yet connected as a learned serving controller.

## Qualification

Full CUDA89;120 build, primary RTX5060Ti execution. Frozen Torch CPU2.14 F32/math
reference and previous saved CUDA field decisions, all384 known development rows
per feature/weight arm (positive and negative):

| native input arm | max absolute logit difference | exact nine-field decisions |
|---|---:|---:|
|embedding, nullable-objective weights|0.0005166530609130859|384/384|
|first-attention, original contextual weights|0.00007736682891845703|384/384|

Both meet the frozen atol0.001/rtol0.0001 tolerance; they are not bitwise Torch
logits. Regular graph and normal launches, capacity1/64 and repeated A/B/A
successful results are bitwise equal. Empty/noncanonical masks, nonfinite valid
states and finite overflow fail closed; masked-out NaN rows are ignored. Every
failed invocation preserves its own caller buffer, and hard off leaves all bytes
and statuses untouched with no mapper allocation. Four tests across the two arms
pass. No full-model quality, generalization or performance claim follows.

Graph/failure A/B/A runs under ComputeSanitizer memcheck, initcheck and synccheck
report0 errors; racecheck0 hazards/errors/warnings. Semantic error statuses and
preserved output sentinels are independently checked. This is ordinary graph
coverage, not a resolution of earlier Qwenputer conditional-WHILE tooling debt.
HIP/SYCL/Linux not run. Requested device weights plus workspace12,197,712B at
capacity64,1,864,144B at8 and572,448B at1; excludes caller tensors, CUDA context,
allocation granules and graph memory. No latency benchmark or default change.

## Reproduced harness defect, not a mapper fix

Initial probe `ccd0fee323d4059293e59fd907862db492be9ca920170b93061445182b99d49a`
already passes384-row numerical replay but fails invalid A/B/A with
`repeat_mismatch`. B turns A's nonfinite rows into valid zero input and writes a
new output. The next failed A correctly retains B's current caller buffer, which
the probe incorrectly compares against its first A sentinel. The same unchanged
binary passes a finite-only control. The corrected probe checks failed bytes
against each call's before-image and valid A outputs against the first A, reporting
both properties separately. It publishes the initial A observation. The native
kernel did not change. Original binary/source/failure/control are retained.

The first validation driver also omitted explicit child stdout/stderr handles on
Windows, losing unittest diagnostics. Its exit/hash record survived. The next
driver captures both streams explicitly; no missing log is counted as success.
All subsequent tests and tool logs are complete.

## Identities and next integration work

Qualified probe `fd985871d744299386abbd741f089ed180c4acde0baf0b43d56c45dd68901492`.
Embedding state `72a0020591a7fb59f0f74e53ab1eadaee5a31ec6ae3b45744c20e3dea99cdefd`;
context state `519fbc196993c94676f4031f7bd45ad9abf661f8b7b4fd3b9d6358204b5294e2`.
[Frozen contract](../plans/2026-10-29-igc-native-input-forward.md).
Private evidence: `igc-native-input-build-01/02.log`,
`igc-native-input-embedding-01/02`, `igc-native-input-context-01`, corresponding
command identity files/logs, and `igc-native-input-sanitizers-01/`, all corrected
qualification jobs DONE0/drained. Historical01 remains failed. Archived probe
binaries are content-addressed; each feature test's source is retained separately.

Next connect the separately trained readiness head and typed request assembly,
then test learned input/executor/output ownership end to end. I2 models still
have known errors; this component establishes faithful native forward only.
Acceptance64, serving settings and all later PRD goals remain unchanged/open.
