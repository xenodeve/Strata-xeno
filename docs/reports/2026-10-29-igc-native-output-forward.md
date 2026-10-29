# I3 native CUDA result-mapper qualification

## Qualified component

The learned output mapper now has a Torch-free CUDA forward implementation. It
consumes actual final head-input floats and a checked typed exact result, plus
expected invocation/anchor arrays. It does not parse text, compute a host answer,
choose a token by hand or receive a target token. The validated existing kind2
artifact is still **diagnostic only**; numerical qualification does not promote
the5/6 development pilot or establish usable integration.

Weights and bounded64-row scratch load once. Each run uses the caller's bound
explicit stream, no allocation/synchronization, and checks the canonical result,
ownership, status and activation finiteness on device. A whole row is published
only after all proposed values are finite. Bad ownership/malformed result,
upstream failure and nonfinite activation remain separate statuses; every
non-success row preserves original activation bytes, including NaN payloads.
Hard-off performs no device/module work and requires no loaded weights.

Two post-fix GPU tests pass. All421 real captured positions compare with the
frozen deterministic Torch mapper within the predeclared tolerance. Maximum
absolute mixed difference7.62939453125e-6. Through the actual native quantized
head, all421 token decisions match both Torch-reference head decisions and the
saved final-fit predictions. Enabled logits are **not bit-identical**:1,234,923
entries differ, maximum absolute0.00784111023. This is enabled F32 reduction-order
variation, not disabled-baseline parity. It must remain visible in later quality
tests and cannot justify silently changing numerical requirements.

Batch1/64, regular CUDA graph relaunch and three repeats are checked; reference
Torch scoring uses the frozen fit's16-row batches. The adversarial64-row fixture
covers upstream failure, wrong owner, digit10, NaN and finite overflow, mixed with
valid rows. All failure bytes and hard-off bytes remain exact. Explicit weights
plus max-batch scratch10,305,284B excludes caller buffers, context and graph costs;
this is not full Hub VRAM. No latency/speed claim from this functional test.

## Bug found before acceptance

The initial uninstrumented421-position/graph comparison passed, but memcheck
instrumentation produced `repeat_mismatch`. Both normal and graph launches
reproduced on the unchanged binary. Memcheck itself reported0 addressing errors;
that did not make the failed target a pass. Racecheck reported2 shared read/write
errors (404/424 hazards in the shown instances) inside `prepare`.

Ranked hypotheses were shared reduction-buffer reuse, per-row status publication,
NaN bit preservation and graph-specific replay state. Normal-launch reproduction
ruled out graph-only state. Racecheck identified the first: the reduction's final
barrier protected its writes, but a faster warp could overwrite `shared[0]` in the
next reduction before every warp consumed the previous return value. Copy the
result into each thread's register, then add a consumption barrier before return.
No arithmetic order or learned weights changed.

Old binary `592d1821912e4b19368cbf66c99a1744b880e70c790bed076f15d4c2a5f8421f`
and source hash are archived. Original failed logs/fixture remain. After the fix,
the **same input hash** runs three regular graph repeats under each tool:

| tool | target exit | findings |
|---|---:|---|
| memcheck |0|0 errors|
| initcheck |0|0 errors|
| synccheck |0|0 errors|
| racecheck |0|0 errors,0 warnings|

Failure-status/byte assertions pass under each tool. Racecheck coverage is shared
memory/synchronization instrumentation; it is not a proof of all global-memory
race freedom or floating-point determinism. Only integer atomic status reporting
is used; no floating atomic reduction. The historical Qwenputer WHILE-path
sanitizer999 limitation is separate and remains unclaimed.

## Reproduction and remaining gates

Standalone CMake `-S tests/igc/native`, Release/Ninja, architectures`89;120`, existing
CUDA13.3/MSVC14.44. `cuobjdump --list-elf` confirms sm89 andsm120 cubins. Execution
was on RTX5060Ti primary; secondary-device execution, Linux/HIP/SYCL are unrun.
This standalone target does not rebuild or change the default/served engine.

Post-fix executable SHA256
`73f9e5be58f9e5273ad5333ee2f33f7db92e0503e63feed2a8eed13f22fd71b2`.
Diagnostic artifact2834820B SHA
`43cf5c8d341bf91966ffc6fa22aba67eb3e04be6a3dcfd694af4b21fb1c62c2c`.
Its state is the existing output pilot
`241ef7032aed2bb9b3e7a3bc8bddd990c963c1d182c63ea968136e1e863c0337`.
Private evidence: `igc-output-native-{configure,build,build-02,tests-01,tests-02}.log`,
`igc-output-sanitize-{01,repro-01,02}/`, `igc-output-sanitizers-{01,02}.log`,
`igc-output-before-reduction-fix.{exe,json}`, `igc-output-cubin-coverage.txt`,
`igc-native-output-fixed-421.bin`. Tests02 DONE0 in4.940s, a test duration only.

Next: controlled native injection/free-running response tests using known checked
operands, then integration with learned readiness/operands. I2, full I3, request
transaction/MTP/snapshot gates and end-to-end practical benchmarks are still open.
No spontaneous model use, trained-body backward, serving activation or full-PRD
completion is asserted here.
