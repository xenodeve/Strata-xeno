# Replay candidate: native component qualification

Issue #276. Qualify replay weights `3cb5a645...828043d` without promotion, training,
full-model inference, new activation collection or acceptance64. Read the existing
native-input, bundle, selective-dispatch, pipeline and live-controller contracts
first. Keep model/tokenizer bindings, embedding/readiness/output assets and all
selection rules unchanged. The new first-attention weights require the existing
kind1 geometry and no additional input features.

Change inventory:

- `tools/igc/native_qualification.py`: offline, bounded field/selection and exact
  typed-result oracle; it cannot be called by production. Unit tests use synthetic
  fields and literal results, including ownership/length/stale-result refusals.
- Task-owned runners: export using the existing strict reader, freeze all source,
  binary, state, protocol and reference identities; stage only bounded temporary
  probe inputs, preserving all old arrays/artifacts. Native probes execute actual
  GPU forward and selection/exact/output components. References run separately.
- Report/Blueprint and bilingual issue comment: actual pass/fail, raw versus gated
  calls, exposure/coverage limits, resource observations and next live proposal.

Forward gate: all 768 matched-development plus 343 frozen legacy rows, using
genuine right-aligned80x4x2560 states, mask and fixed positions. Compare 52 raw
logits against qualified original Torch math forward at atol0.001/rtol0.0001 and
require every field argmax to match saved final predictions. Test normal/graph,
repeat/alternating inputs, capacity1/64, invalid mask, nonfinite/overflow and off
behavior. Use no oracle operand as a neural input. Hash both complete caches
before/after, but read only a minibatch into memory; temporary binary staging is
not a new persistent cache and is removed only by its exact owned filename.

Controller gate: reuse the existing coherent64-prompt development capture and its
content identity, explicitly separate from sealed acceptance64. Compare original
retained and replay candidate using the same current executable, then qualify the
candidate's request/result bytes and mapped output against offline references.
Report each raw input head, readiness, no-call/disagreement/noncanonical reasons,
selected-correct requests, actual exact results and negative calls. A warm-fit
counterfactual may use the same saved states without promoting it. No runtime
policy or threshold changes. Ordinary/graph/repeated calls, reset, module
abort/replay/commit and new-owner isolation must remain correct. Choose lifecycle
fixture rows by their declared statuses; never make a wrong tuple an oracle input.

Run applicable ComputeSanitizer tools with semantic-output checks and bounded
owned children; report a timeout or unsupported tool honestly. Racecheck does not
prove all global-memory or floating-reduction properties. Existing model-content
binding/lifecycle tests may read hashes without loading model weights; distinguish
those tests from a real model launch and its memory requirements.

Use one heavy job, primary5060Ti, current physical/system commitment and disk
preflight, hidden children and fixed timeouts. No process private-commit cap,
Hub/system change, cloud spending, push or deployment. Verify binary hashes and
CUDA89;120 coverage before reuse; build if current source identity is not proven.
After component gates, return an evidence-backed bounded live proposal; do not
launch the full model in this qualification task.
