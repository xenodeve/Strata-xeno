# I3 controlled decoder smoke: native path works, gates remain open

## Implementation

An optional diagnostic build now inserts the native learned result mapper after
final HC mixing and before native head quantization/sampling inside `Verifier`.
It is configured before init/capture, prepares weights/workspace on the explicit
stream outside capture, and uses stable device addresses in captured graphs.
Release occurs after graph/stream drain and before stream destruction. Default
builds do not compile the seam. Diagnostic builds without the two explicit CLI
paths install no callback or mapper buffers.

Flags `--igc-output-diagnostic-artifact` and
`--igc-output-diagnostic-fixture` require single-request native serve, no batch,
layer split or pipeline windows. The selected D2x secondary expert offload and
MTP suffix stay intact. The bounded exact-prefix fixture contains earlier checked
I1a results, **not learned I2 selection or runtime arithmetic execution**. The host
matches canonical prefix hashes and validates metadata; it never parses arithmetic
or calculates an answer. Fresh `VerifiedInputs` content-binds actual GGUF/tokenizer
and retains read handles before allocation. Outer source provenance separately
covers pack/MTP/profile/build. This is still an output-path isolation experiment.

At first decode window, bind the complete prefix to the initial anchor and stage
the fixed typed result/owner arrays once on the verifier stream. Unmatched prefixes
stage an empty failure result, preserving base activations. Later windows use GPU
data only. Diagnostic status readback/fence after each completed window fails
closed on unexpected statuses before publishing that window. This extra fence
and startup hashing are disclosed diagnostic costs, not a practical latency claim.
Normal greedy MTP commit/suffix handling, positions, penalties and stop/cancel
loop remain in use; dynamic-call/snapshot ownership still needs I4 validation.

The89;120 full candidate build passes. Native host fixture test covers accepted
prefix, unmatched prefix, wrong anchor, duplicate entry, malformed integer,
truncation/trailing bytes, wrong artifact digest and invalid prefix token. Two
pure serializer/final-format tests pass. Source and fixtures reject before model
allocation. The qualified native CUDA/graph/sanitizer component is documented
separately; no HIP/SYCL/Linux integration build was run.

## Actual three-arm smoke

Actual Hub Gyro-M262K/five-worker profile, Vision, inherited GPU mapping, original
RoPE/cache/offload and process settings. No private-commit cap or served changes.
Four prompts per arm: two known development arithmetic prompts and two unmatched
ordinary prompts. Temperature0/max96 tokens. Pre-hook binary, candidate-off and
candidate-on run sequentially with identical initial task-owned expert profiles.
This is a functional test, not ABBA performance evidence or a release corpus.

| prompt | expected | pre-hook final | candidate-off final | candidate-on final |
|---|---:|---:|---:|---|
|92 - 18|74|74|74|74|
|12 * 64|768|768|768|`12 * 64 = 768` followed by `The result is 768.`|

Both enabled mathematical responses are correct by inspection against the
independently checked result. However the **frozen automated rubric** required
exactly one integer after `</think>`. It therefore records1/2 single-integer
format matches for enabled, versus2/2 for the other arms. This is not evidence
that the enabled multiplication answer is wrong. Preserve the original report
and rubric outcome; do not silently rescore it as a mathematical failure or edit
the original result into a pass. Future evaluation must separate format compliance,
final arithmetic correctness and internal evidence consistency explicitly.

No limit/fault occurs. Enabled multiplication produces32 tokens and subtraction13;
off produces68 and33. These two known-input examples do not establish reasoning
token savings generally or speed benefit; no inference timing comparison is made.

Unmatched candidate-on/off raw token IDs are identical on both ordinary prompts.
**Pre-hook versus candidate-off raw parity is only3/4**: the Thai ordinary prompt
first differs at generated index24,39 versus37 tokens. Earlier three prompts
match raw output, but MTP window/tier counters already differ (first arithmetic
22 versus21 windows). Dynamic scheduling/placement is a hypothesis, not a proven
cause. This unresolved gate prevents a disabled-parity or phase-completion claim.
Next reproduce with alternating/repeated same-binary controls and existing state
diagnostics before proposing a fix; do not hide it by changing the Hub workload.

Run136.578s including three boots/cleanup and enabled content hashing, not inference
latency. Sampled max single-process private46,813,847,552B; min available physical
10,225,741,824B; min system commit headroom14,270,443,520B. No guard changed behavior.
The original JSON's `success:true` means the harness completed, not that parity,
I3 or release acceptance passed.

## Frozen evidence

Candidate SHA256 `0f68c5f648b8324c1a168f0229eef64349754815a71ac32f70e0c32ab63fbbdd`.
Pre-hook `ae6c485fb6b765ae4005183bac1712508b801dbfb1da35ee46bae50966597d7d`.
Fixture3708B/30 prefixes
`1ca80fb6485b960c0217cd80b5b21095a72adbd8a6e2e1d6fa5f385e4d05fc30`.
Artifact `43cf5c8d341bf91966ffc6fa22aba67eb3e04be6a3dcfd694af4b21fb1c62c2c`.
Runner source `69a5f3f9e3627f93b3249e46e5500ee818a247c7739287453b931b9f8d6a35cb`
is archived beside the original report before further harness edits.

Private evidence: `igc-output-injection-build-01.log`,
`igc-output-integration-smoke-01/{inputs,report,memory}.json`, each arm's
`{records.json,engine.log,vision.log,config.json}`, original runner source and
`igc-output-integration-smoke-01.log` (DONE0, owned tree drained).
Acceptance64 untouched. No spontaneous learned use, runtime I1a call, real Hub
benchmark, default activation or full-project completion is claimed.
