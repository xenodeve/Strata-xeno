# Device-side neural fields become bounded exact requests

The new internal `SelectionInputs` API accepts two52-logit readouts/statuses,
readiness logits/status and explicit invocation/anchor device arrays. Argmax
agrees with the trained field slices and tie-to-lowest behavior. Dispatch requires
readiness, two finite successful readouts agreeing on all nine fields and canonical
signed one/two-digit integers. It rejects negative zero, leading zero and nonzero
unused tens without repairing a prediction. No token IDs, text parsing, host
calculation or host numeric readback. Stateless, explicit stream, count1..64,
no allocation/synchronization; hard off bypasses before arguments/device work.

Every enabled invocation writes a fresh160-byte request, including inactive
domain0 records on failed gates. This prevents old successful operands surviving
no-call, disagreement or neural failure. Status codes distinguish no-call,
upstream/owner error, nonfinite, disagreement and canonical failure. The exact
executor turns inactive records into valueless failures; a failed dispatch never
manufactures a numerical zero. A real successful zero retains length1 and remains
distinguishable from empty failure. Off callers skip downstream execution too.

The behavioral red is an explicitly unimplemented stub, built successfully and
failing `hard_off_contract` before CUDA setup (exit2). Its binary is preserved
as `96dc06dd2c94be7c8ad38125ddc9fd57a86a0b038a40d56ca5b692ac4c5fd591`.
The completed implementation passes64 fixture cases, normal/graph A/B/A and
hard-off byte preservation. Cases cover signed add/subtract/multiply, input/output
zero, readiness/operator ties, unsupported/no-call, upstream failures, nonfinite,
readout disagreement, malformed canonical forms and owner/anchor bounds. Typed
request bytes and actual I1a result bytes are compared with independent fixed
test expectations. Success→failure must erase previous successful request/result.
The host fixture/oracle is only in the test executable, not the dispatcher.

CUDA89;120 build and primary RTX5060Ti execution. Memcheck/initcheck/synccheck
report0 errors, racecheck0 hazards/errors/warnings; each instrumented program also
passes its typed/exact assertions. HIP/SYCL/Linux unrun. The dispatcher owns no
persistent device allocation. This is component verification, not a speed result,
learned full-model use, session exactly-once ledger or passed I2 quality gate.

The selective policy is motivated by the already inspected130/288 correct agreed
development tuples, not a guarantee on unseen tasks. The original models and
failed full-coverage gates remain unchanged; acceptance64 stays sealed. Next
connect genuine pre-anchor activations and the committed first decoder anchor,
test prefix reuse/MTP/cancel/ownership, then evaluate learned behavior. Do not
replace that work with a prefix-keyed table of precomputed learned predictions.

Evidence: [frozen contract](../plans/2026-10-29-igc-selective-dispatch.md), private
`igc-dispatch-red-build/run.log`, `igc-dispatch-green-build/run-01.log`,
`igc-dispatch-sanitizers-01/` command/source/executable hashes and tool logs;
all final jobs DONE0/drained. Source artifacts and serving settings unchanged.
