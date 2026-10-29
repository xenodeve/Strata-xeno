# I3 preparation: reusable native-head training bridge

The I0 process-per-call STE proof is correct but does not provide a reusable
CUDA training interface. The existing CUDA Torch runtime now makes a narrow
first-party bridge possible without downloading a framework or changing Gyro.
This component prepares output-mapper training; it does not close input quality,
output result use or model integration gates by itself.

## Contract and change inventory

- Optional CMake target under the existing CUDA head-probe option: a shared
  library using `NativeHead` and the already tested `head_vjp.cu`. No Torch C++
  ABI dependency or change to production `NativeHead`/serving flags.
- `tests/igc/head_bridge.cpp`: bounded C API for open/project/VJP/close, a validated
  opaque handle registry, fixed Gyro head geometry/type, current-device checks,
  caller-owned device tensors and explicit non-default streams. Preserve frozen
  weights; allocate head/scratch only at open, not per projection.
- `tools/igc/cuda_head.py`: CUDA Torch autograd function with tensor shape/dtype/
  contiguity/device validation before calling the library. Use an owned stream,
  explicit stream dependencies and allocator lifetime records. No process per
  call, host result arithmetic or language-model body.
- `tools/test_igc_cuda_head.py`: actual native forward fidelity, independent
  sparse-gradient oracle, malformed/refused calls, repeat/batch order, lifetime
  and one bounded loss-decreasing step. Report unsupported sanitizer coverage.
- `docs/BLUEPRINT.md` and an evidence report: optional diagnostic responsibility,
  ABI/lifecycle, numerical scope and backend debt in the same commit.

## Numerical and lifecycle rules

Forward uses the exact native Q8_1 activation quantizer and Q8_0 head. Backward
uses the declared dequantized-weight straight-through estimator; it is not an
exact derivative of quantization and does not reach HC/GDN/MoE/attention layers.
The output mapper would inject after final HC in this variant, not reproduce the
paper's early-layer injection. Keep that distinction in all evidence.

Reject invalid handles, null/misaligned/non-device pointers, batches outside
1..64, wrong device and invalid/default streams before enqueueing. Model-supplied
records never carry pointers; these are trusted in-process tensor interfaces.
Serialize mutable head scratch across streams with an event. Close waits only
for work enqueued through this handle before freeing its owned weights. Refusal
returns no valid result; CUDA faults invalidate the handle. No exceptions cross
the C ABI and errors are bounded strings. The Python wrapper refuses use after
close and retains the bridge object through autograd.

## Verification order

Wait for the current input-head fit to drain before building or testing. Establish
a real RED for the absent target/API. Build CUDA `89;120`; a first-party diagnostic
may run only on the primary GPU. Then compare captured nonzero inputs and the
existing head-probe output bitwise, check independent Q8_0 sparse VJP values,
repeat/batch and invalid-input controls, and verify bounded cleanup. Run applicable
regular sanitizer paths, recording any unsupported tool/runtime combination.

Only after these checks train a result mapper with native-forward LM loss and
held-out result-use controls. Do not label a synthetic class-loss step as numeric
result understanding. No deployment, defaults, base-model training or download.
