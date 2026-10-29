# P2 native CPU artifact loader checkpoint

P2 validates the exported circuit and token embeddings in portable C++ before any
GPU upload. It does not execute arithmetic on CUDA or integrate the language model.
The default engine build and request path do not link or call this loader yet.

## Deliverables and boundaries

- `include/strata/core/qwenputer_artifact.hpp` and `src/core/qwenputer_artifact.cpp`:
  bounded STQPUT01 container, SHA256, strict section/schema/shape/index validation,
  six independent identity bindings and an explicit supported arithmetic ABI.
- Transactional failure: a refused load preserves the caller's previous Artifact
  and MemoryPlan. All sparse source node IDs and shared value-head identities are
  checked; an unknown numeric ABI is refused even when the caller repeats it.
- Known token features cover tagged Trace/Text inputs without source callbacks.
  Invalid tokens leave the output unchanged.
- `STRATA_BUILD_QWENPUTER_HOST_TESTS=OFF` adds two optional test executables. The
  standalone host project requires C++20 only, with no CUDA library or allocation
  API. PE imports were inspected with MSVC dumpbin: CRT and KERNEL32 only.

The checked memory estimate includes the serialized input, typed arrays, derived
text index and conservative validation scratch. An already retained artifact is
the caller's additional allocation. Device buffer estimates include immutable
weights, three mutable state/history sets, scratch and bounded FIFO. Driver graph
resources remain explicitly unmeasured until P4 instantiation; this is not a full
VRAM-fit claim.

## Verification

Evidence directory: the task-owned `strata-qwenputer-p0-20261028` temporary directory.
The exact commands and raw outputs are retained in the named logs below.

| Check | Result | Evidence |
|---|---|---|
| Stub SHA/container/real-artifact boundary | 11 expected assertion failures | `p2-host-red.log` |
| Stub feature boundary | 8 expected assertion failures | `p2-feature-red.log` |
| Resigned semantic mutants | First valid run: two expected failures for missing node/ABI validation; remaining 11 subtests passed | `p2-mutants-red-02.log` |
| Final MSVC build, warnings as errors | PASS | `p2-host-build-04.log` |
| Real artifact, binding, budgets and 32 seeded truncation/bit-flip pairs | PASS, zero failures | `p2-loader-green-04.log` |
| Native C++ full token-feature domains | 263,254,200 values, digest identical to original compiler proof | `p2-features-green-01.log` |
| Final parser/export/format/feature suite | 19 passed, 47 subtests, 9.62 seconds | `p2-final-suite-04.log` |
| Separate temporary loader copy with the term-index check removed | Test rejects the deliberately broken loader (exit 1); checked production source untouched | `p2-negative-result.log` |

The first mutant invocation (`p2-mutants-red.log`) used an older driver without
the new command mode. It is invalid as a parser result and was replaced by the
rebuilt-driver run `p2-mutants-red-02.log`; both raw logs are preserved.

The real artifact remains 18,298,944 bytes, SHA256
`edcd063736a3b3575cf81b5bac524943e0694e23c12af0cd261b44b150e704b0`.
The exhaustive feature digest is
`190676489d4adacf6ee6a78d80f9fcbd1a2bc98eced7210c5881f567762478f6`.

Derived bytes: weights 19,227,432; each state 12,845,056; each lag set 350,160;
scratch 750,304; FIFO 18,688; device buffers 59,582,072; host peak estimate
38,342,120. These are buffer/validation estimates, not measured driver allocation.

Final executable SHA256:

- artifact test: `7120a3ee1f6b31310a2ee5b6b776ac141fe6266b06c8a8273ca33d63a07d15ca`
- feature test: `96884b06d610f4d39d33df5e0d500aeb01e200f4a5da3e44da73684f08dbcf96`

## Remaining work

P3 must prove deterministic native CUDA steps against these offline tensors. P4
must establish actual graph resource costs and device-driven execution. Model
ownership, snapshot, API/UI, parity and practical performance remain later gates.
No HIP/SYCL toolchain execution occurred; the new CPU library is isolated from
those engines, and cross-platform compilation remains validation debt. The PRD
issue still awaits its earlier Submit operation; no alternate write was attempted.
