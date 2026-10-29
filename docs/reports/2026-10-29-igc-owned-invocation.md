# Retained exact invocation on the device

Issue #276; [contract](../plans/2026-10-29-igc-owned-invocation.md).
The learned engine integration needs a device lifetime because its final-head
callback is captured into a graph and replayed for each window. The new
`igc_invocation.hpp` state binds a request ID/anchor, consumes one selection
attempt, retains its terminal result and broadcasts that result without another
arithmetic evaluation. A fresh arm clears the old answer. No host arithmetic,
per-step result readback, allocation or synchronization enters these kernels.

The built test stub first failed on `hard_off_arm` (exit2), before CUDA access;
red executable SHA256
`bf016815acbeb325cda314576a38621fece71d1d1554d4c74a2c39c8167bb07e`.
The final CUDA89;120 build and native primary-GPU execution pass:

- Counts1 and64, signed multiplication and canonical zero against a fixed
  independent test-only oracle; no-call/rejected selection, wrong owner/anchor,
  malformed operands and invalid ID all retain a valueless failure.
- Repeated ordinary/captured graph launches after replacing operands preserve
  the first terminal state, with exactly one attempt and one actual evaluation
  only for requests that reach arithmetic. No-call/owner rejection never evaluates.
- Re-arm clears successes before new execution; failure broadcast cannot expose
  stale success. A device shadow copy can be re-armed/executed without changing
  its retained base. Hard off is byte-identical; default stream/null/count bounds
  are refused when enabled.
- Memcheck, initcheck, synccheck and racecheck each exit0; the first three report
  zero errors and racecheck reports zero hazards/errors/warnings. Racecheck is
  not a proof against arbitrary global-memory races. The existing64-fixture
  selective-dispatch/exact regression also passes.
- The freshly rebuilt original exact executor and graph-broadcast suites pass
  eight tests, zero failures/errors/skips, including the frozen held-out signed
  integer corpus,62-digit bounds/overflow and stale-output rejection. Regression02
  is the valid run; regression01 could not import `tools` from a temporary script
  directory and ran no real tests. Its failed log is preserved.

Final test executable SHA256:
`76593483ecc780bb1e6cca95ebf95b7fa413e758346aaa32e10aeb30003afd7d`.
Source SHA256 `igc_exact.cu`:
`46c2e6a982e694206973c87640d299f6164ff00a06f60c92ae5d1357c822afb4`.
Raw evidence is `igc-invocation-{red,green}-*.log` and
`igc-invocation-validation-02/report.json` with each command/hash/sanitizer log.
Validation01's kernels also passed all four tools, but the harness incorrectly
required `ERROR SUMMARY` from racecheck, whose format is `RACECHECK SUMMARY`.
The format check was reproduced against saved logs with negative summary controls,
corrected, and the final expanded test rerun as validation02. Original evidence
is preserved; no production kernel fix was required for that harness fault.

This completes the device one-shot component, not a phase-wide integration gate.
It is not yet a session/checkpoint ledger or proof of accepted-prefix publication.
Live native activation ownership, prompt-cache/batched-path qualification,
model-selected calls, cancellation and I2 quality remain open. No serving default
or language state changed, and no latency claim is made. HIP/SYCL are unrun debt:
this opt-in diagnostic component is CUDA-only and no equivalent integration has
been implemented for those backends.
