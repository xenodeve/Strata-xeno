# Native readiness preserves the trained readout, including its known miss

Diagnostic kind3 stores the original MLP's mean/scale and two linear layers.
The bounded artifact reader validates six exact tensor layouts, identities and
finite values, plus scale≥0.0001 before weight allocation. Both Python and the
strict-warning native host reader reject rehashed invalid scales. Kind1/2 schema
digests stay `cb73a018f1b785602b1b05e3754dbd0d1687f010efa92526f2bc92ea994a3101`
and `8e443014ea93de240eccc32fc7a196fa07321cdd594b2f02350853c16cde2395`;
native roundtrips verify their compatibility. Kind3 is still diagnostic-only and
cannot be enabled in production by passing numerical tests. Nine host/schema
tests pass, including explicit production refusal and correctly typed, resigned
malformed files. An earlier bytearray test stopped at the file-type guard instead
of the intended scale check; it now passes immutable bytes to reach that check.

`ReadinessMapper` CUDA standardizes actual2560-wide final activations with the
stored fit-only statistics, computes128 exact-GELU neurons and two logits. One
explicit stream, capacity1..64, startup-only allocation, no run-time sync or host
numeric readback. Hard off leaves caller bytes/status untouched without mapper
allocation; failures preserve each invocation's caller output. Both graph and
normal paths validate nonfinite/finite-overflow failure and repeated A/B/A.

All320 original real-Gyro activation vectors match the Torch reference within the
declared tolerance and **all320 saved classification decisions**. Maximum absolute
logit difference0.0000095367431640625. This includes the original incorrect missed
call: labels remain208/208 fit and111/112 known development. Faithful porting is
not a readiness-quality fix or I2 completion. No extra training or acceptance use.

CUDA89;120 build and two native tests pass on primary RTX5060Ti. Memcheck,
initcheck and synccheck report0 errors; racecheck0 hazards/errors/warnings on
ordinary graph/failure/repeat controls. Requested weights/workspace2,021,384B at
capacity64, excluding caller tensors/context/graph/allocation granules. No latency
claim, Hub setting change or default activation; HIP/SYCL/Linux remain unrun.

Qualified probe `95dfb5db7e313a1ae95d69091e4577369a1210dd419e28b2e6d354259c0a334f`.
State `0ca4f0b8cc66a0f876d86623dd95ecef828bf9927c9fd9a55e2021b3d1497264`;
kind3 schema `a8cdfe4066180e2cf412e3c2d139bc8ab94c33a3c1d13e92eca075e427b840fa`.
[Protocol](../plans/2026-10-29-igc-native-readiness.md).
Private evidence: `igc-readiness-host-build-01.log`,
`igc-readiness-host-validation-01/02.json`, `igc-native-readiness-build-01.log`,
`igc-native-readiness-01/` plus command/log/streams, and
`igc-native-readiness-sanitizers-01/`, all final jobs completed/drained.

Review scope: bounded weight/normalization format and local diagnostic APIs were
checked against the existing trust boundary. The separate full-codebase security
audit skill is not installed; no third-party installer or full audit was run.
Next is device-side typed assembly/selective agreement and actual engine ownership,
not a cloud migration. Cached-activation portability was assessed only; no upload,
runtime or Colab quota was used, and the denied browser route is not retried.
