# Stateful native latent-controller connection

Issue #276. Connect already qualified components before attaching the engine
program hook: bound four-network bundle, two actual pre-anchor readouts, actual
final-state readiness, strict typed selection, retained one-shot exact execution
and the late output mapper. No token parser, prompt lookup or host arithmetic.

Change inventory: `core/igc_pipeline.hpp/.cpp` owns startup-only neural weights and
fixed scratch on one created stream; `igc_invocation.hpp` / `igc_exact.cu` retain
the actual first attempted tuple/reason, because dispatcher scratch changes on
later graph launches; native CMake and pipeline/ownership tests verify lifecycle
and actual recorded activation forward. Blueprint/report record component scope.
The later program wrapper must bind model/tokenizer receipts and accepted context.

Stage inputs once outside capture, with only device activation/mask addresses and
host invocation/anchor metadata. Missing context arms no call and clears prior
eligibility. Head execution is capture-safe: readiness from row0, selection,
one-shot execution, retained result broadcast and output mapping for1..64 rows.
Later replays cannot calculate again or rewrite execution provenance. Hard off
returns before device work/argument validation. Diagnostic audit occurs only at
request end; never put it in the production per-token decision path.

Qualify actual coherent saved Gyro early/final activations with frozen trained
assets, ordinary/graph repeats, re-stage/reset, missing/invalid mask and hard-off
controls. Compare neural fields/typed results/output tensors to independent
offline references outside execution. Retain the distinction: saved-activation
replay is not live language-model integration, fresh acceptance, whole-model
speed or completed I2 quality. First-anchor support is not dynamic reasoning
calls; those retain the later committed-shadow/rollback gates.
