# I3 component: reusable native-head training bridge

The native-head training seam is implemented and its focused tests pass. This
does not complete I3: no result output mapper has been trained, input quality
still fails, and there is no usable integrated IGC or end-to-end benchmark.

## Implementation

Optional `igc_head_bridge` builds only under the existing CUDA head-probe option.
It reuses `NativeHead` and the independently tested dequantized-weight VJP kernel.
It has no Torch C++ ABI dependency and does not change the serving head or flags.
The trusted in-process C API has version 1 and opaque numeric handles, fixed
2560/248320 geometry, Q8_0 head binding, at most two handles and batches 1..64.
Model-supplied records never contain these host tensor pointers.

Head weights and quantization scratch are loaded once per handle, not per
projection. Caller-owned device arrays and explicit non-default streams are
validated for alignment, device and overlap. Events serialize mutable scratch
and fence queued work before close. Refused input returns no output; CUDA faults
invalidate the handle. A fault before event recording falls back to fencing its
own stream. No exceptions cross the C ABI and error strings are bounded.

`tools/igc/cuda_head.py` pins the first-party DLL SHA256 before loading, validates
CUDA F32 tensors and their shape/contiguity/finite values, maintains stream
dependencies and allocator lifetimes, and retains the head through autograd.
Explicit close is idempotent; later forward/backward fails rather than using
freed weights. No process is launched per projection.

Forward is the actual native Q8_1 activation quantizer/Q8_0 projection. Backward
is explicitly `W_dequant.T @ delta`, a straight-through estimator across activation
quantization. It does not differentiate through final HC, GDN, attention, MoE or
earlier layers. The injection seam is after final HC, not the paper's early-layer
output mapper. Base weights remain frozen.

## Evidence

- CUDA `89;120` shared-library build passed. DLL SHA256:
  `d1716cd48a92d6e9d57bfe3e1fe0f8b1d00eda21210443b899cfa798cebba381`.
- Eight functional tests passed. A nonzero actual Gyro-M head input reproduces
  all 248,320 captured logits bitwise and repeats bitwise. Two-row sparse VJP
  matches independently decoded Q8_0 rows, including a negative coefficient.
- CPU/wrong dtype/shape/alignment/nonfinite tensors are refused. Raw invalid
  handles/default streams/oversized batches/host pointers/overlap leave output
  unchanged and subsequent valid work still succeeds. Close fences pending work
  and rejects later backward.
- One diagnostic native-forward CE step decreases 27.9106674 to 27.7806511 with
  the original activation and base weights unchanged. Target 391 is an arbitrary
  diagnostic token class, not an arithmetic answer or evidence of result use.
- Compute Sanitizer memcheck/initcheck/synccheck each reports `0 errors` on the
  selected forward/VJP cases. Racecheck reports `0 hazards (0 errors, 0 warnings)`.
  Each tool ran serially with bounded headless cleanup and a frozen DLL hash.
  These are regular paths, not the comparator's conditional-WHILE instrumentation;
  the earlier WHILE coverage debt is unchanged. Racecheck is not a proof of
  arbitrary global-memory ordering or untested concurrent callers.

Functional evidence: `igc-native-head-bridge-build.log`,
`igc-native-head-bridge-tests-final.log`; sanitizer evidence:
`igc-head-bridge-memcheck.log`, `igc-head-bridge-sanitizers-01/` and the wrapper
logs. Torch runtime is the already installed 2.11.0+cu130 on the primary RTX 5060 Ti,
with the task DLL built by CUDA 13.3/MSVC. No installation, model-body load,
unapproved external code, serving change or default activation occurred.

The head's 675,430,400 weight bytes are not a total process/VRAM peak. The standalone
training component owns its own head; future production integration must reuse
the engine's already loaded head instead of duplicating these weights. No Hub
expert-cache/capacity or performance benefit is established. HIP/SYCL were not
built for this CUDA-only diagnostic, and disabled full-model parity remains a
separate integration gate.

## Next

Use the qualified bridge for bounded result-mapper training with real numeric
results and native-forward LM loss, then test held-out faithful result use.
Continue I2 contextual/routing work independently; do not treat a trained
diagnostic class or forced calculator call as spontaneous model utilization.
Actual Hub end-to-end quality/latency/token/context/resource benchmarking follows
usable integration as requested.
