# Retained device invocation for the learned engine path

Issue #276. The source survey found that the final-head transform is captured
once and replayed for each decode window. A host `first_window` conditional inside
that callback would be baked into the graph. Re-running the existing unowned exact
API would also calculate again on each replay. Introduce a device-owned one-shot
state, with explicit request-entry reset, before attaching learned readouts.

Change inventory: `core/igc_invocation.hpp` defines caller-owned lifecycle state;
`kernels/igc_exact.cu` shares the existing exact calculation without altering its
ordinary entry point; `tests/igc/invocation_test.cpp` verifies the lifecycle and
real arithmetic; `tests/igc/native/CMakeLists.txt` adds the isolated test;
`docs/BLUEPRINT.md` and the report record this non-baseline seam. No flags, serving
defaults, language state or external ABI change in this component.

An explicit arm clears prior results and binds invocation/anchor. The first
execution consumes the attempt whether it is ready, declined or invalid. Only a
ready request with matching ownership reaches exact arithmetic; its evaluation
count is separate from attempts. Later graph replays preserve the terminal
result. Broadcasting a retained result does not execute it again. Hard off
returns before pointer validation/device work and preserves bytes.

Tests must first fail on a built behavioral stub, then pass with real kernels:
64-row signed/zero calculations, no-call, invalid selection/owner/input, terminal
reason, exact count, repeated graph launches after operand mutation, re-arm with
no stale answer, device shadow-copy isolation and result broadcast. Validate
count/null/stream bounds, CUDA89;120 and applicable four sanitizer tools. The test
oracle is confined to the test executable. No performance claim.

This state is not yet a session ledger or checkpoint schema. The initial engine
integration will bind the mandatory first input anchor, then separately qualify
accepted-prefix ownership, cancellation, cached-prefix activation availability and
exactly-once publication. Later dynamic reasoning calls remain a separate gate.
