# Diagnostic neural artifact and native host loader

Tracker [#276](https://github.com/xenodeve/Strata-xeno/issues/276).
[Binary contract](../plans/2026-10-29-igc-adapter-artifact.md).

The Python exporter/validator and native C++ host loader implement the same bounded
`STIGCM01` weight-only format. Tensor schemas are generated from the first-party
classes, explicit operator semantics and class-source identity, and checked for
drift by a generation test. Model, tokenizer, training-state/protocol and schema
digests are bound before weights are exposed. Canonical offsets/shapes/counts,
zero padding, finite F32 values, exact length and both content digests are checked.
The native finite check uses exponent bits and does not depend on fast-math
compiler assumptions. File loads reject non-regular/reparse files and cap8MiB.

Both input and output types roundtrip byte-identically through the host loader.
Native refusals include disabled diagnostic permission, wrong full-file digest,
truncation/trailing data and correctly rehashed malicious offsets/NaN weights.
No failed load publishes a payload. Version1 is diagnostic-only; there is no path
from it to serving enablement. This is parser/weight identity, not native neural
forward parity or usable model integration.

Evidence:

- Actual missing-executable RED precedes implementation.
- `igc-adapter-host-configure.log` and `igc-adapter-host-build.log` DONE0. MSVC
  host library uses `/W4 /WX`, reusing the existing tested SHA256 utility. No GPU
  context or inference backend is required by this host-only target.
- `igc-native-adapter-tests.log`:8 tests pass (six Python boundary/schema tests
  and two native contract tests covering both architectures and refusal cases).
- Existing **failed** input control `ec7273be...` is exported strictly for later
  numerical-parity diagnostics:389,808 bytes,40 tensors, full SHA256
  `2270bef80eddd7a9edf88277d8f3ecd6ae95ef2ac9ffa61ee9895ecda4988d2c`,
  schema `cb73a018f1b785602b1b05e3754dbd0d1687f010efa92526f2bc92ea994a3101`.
  Production loading is explicitly refused. Export does not turn a failed fit
  into a successful adapter.

No serving flag, base model, Hub profile or core forward path is changed. Native
CUDA neural operators/forward parity and Linux/HIP/SYCL host build coverage are
not established by this Windows host-only qualification. The existing full-engine
CUDA89;120 build remains separate evidence.
