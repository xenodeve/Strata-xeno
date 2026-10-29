# Learned activation fields to bounded exact requests

The native input and readiness components now have faithful forward, while their
trained quality remains imperfect. The already inspected embedding/context heads
agree on130/288 positive development tuples, all correct there, and no negative
has an agreed nonzero tuple. Test strict two-readout agreement as an explicitly
selective diagnostic controller. This is not independent acceptance or a change
that closes I2's original quality/coverage gates.

Inputs: primary and secondary52-logit arrays, their NN statuses, the two readiness
logits/status, and explicit invocation/anchor device arrays. No token IDs, strings,
oracle operands, program pointers or host arithmetic. Argmax uses the existing
nine field slices/tie-to-lowest semantics. Require semantic readiness, two valid
finite readouts agreeing on all nine fields, supported operation and canonical
signed one/two-digit integers. Reject negative zero, leading zero and nonzero
unused tens. A failed gate must not repair or truncate the prediction.

Write a fresh160-byte ExactRequest for each invocation. Accepted requests use
ABI1/domain1; all other outcomes explicitly overwrite it with an inactive domain0
record retaining only valid ownership metadata. This prevents a stale successful
request being executed after a missed call, disagreement or fault. Separate
statuses:0 ready,1 no call,2 upstream/owner invalid,3 nonfinite,4 disagreement,
5 noncanonical. Readiness0 can skip irrelevant operand fields. Downstream exact
execution of inactive records returns a valueless failure and output mapping must
bypass; no zero answer or false successful verification. Hard off skips before
arguments/device work; the caller also omits downstream execution when disabled.

Stateless CUDA, one explicit stream, count1..64, no allocation/synchronization or
host numeric readback. Test independently constructed field fixtures, all signed
ops and zero cases, wrong/missing ownership, unsupported/no call, each malformed
canonical representation, disagreement and nonfinite. Compare typed request bytes
and the real exact executor's results against an independent offline oracle,
including successive success→failure so no stale answer survives. Graph/repeat
and all applicable sanitizer tools; build89;120, primary execution.

Only after component qualification, integrate genuine native pre-anchor embedding
and first-attention buffers with the first committed decoder anchor and readiness
activation. Exactly-once execution and accepted-prefix ownership must be tested;
do not call a prefix fixture or offline learned table a live learned mapper. No
production activation, benchmark or general guarantee is implied by this contract.
