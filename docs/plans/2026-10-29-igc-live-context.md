# Live learned activation ownership

Issue #276, continuation of the selective-dispatch and one-shot-invocation plans.
Survey: serial verifier early tensors exist but its current copy API synchronizes
and returns host rows. Batched prefill has no early-row callback; `on_chunk` sees
only final residuals. Cached prompts may skip all80 desired input positions.
Decode commits consumed input tokens, not emitted next tokens; rejected MTP rows
must never enter retained feature history. Snapshot/slot/control/image changes
cannot assume the last cached feature tensors belong to the restored session.

Change inventory and intended validation:

- `core/igc_context.hpp` / `core/igc_context.cpp`: fixed80-row native embedding and
  first-attention ring plus right-aligned input buffers/mask. Host canonical
  prefix metadata tracks only identity/accepted positions, never arithmetic.
  Stage rows on explicit producer streams; accept the retained input prefix;
  preserve compatible cached rows, clear unverified lineage, bypass missing rows.
  Cross-stream event ordering; no per-row host tensor readback. Test byte-exact
  synthetic native device rows, wrap, >80-row chunk, ownership, rewind/missing
  rows, canonical-prefix mismatch, scope reset and bounded allocation first.
- `verify.hpp` / `verify.cpp`: expose completed actual early device rows and their
  owning stream to the first-party consumer without changing kernel numerics.
  Qualify against the existing host observation on genuine model weights.
- `prefill.hpp` / `prefill.cpp`: opt-in early device-row callbacks after embedding
  broadcast and layer0 attention HC write, preserving the existing fused path.
  Validate actual batched versus qualified serial feature behavior, not assume
  floating-point equality across GEMM/GEMV implementations.
- New diagnostic learned controller and `generate.cpp`: load bound kind1/kind3/
  kind2 weights once, stage actual accepted features, use first real final anchor
  for readiness, device-select typed operands, execute once, retain result for
  output mapping. Keep v1/v2 oracle-controlled fixtures separately labeled.
  Test request/cancel/reuse/MTP boundaries and actual free generation. No parsed
  operand, prefix answer table or generated invocation serialization is allowed.
- CMake, focused tests and `BLUEPRINT.md`: link only the explicit opt-in diagnostic
  path, retain a hard bypass, record backend debt and evidence in the same commit.

This is a staged inventory, not a claim that all sites are implemented. Missing
features must bypass IGC without changing Hub cache/prefill behavior. Keep the
fresh acceptance64 sealed during development; imperfect learned routing/digits
remain an open quality gate. Do not publish a whole-phase completion from a
component test. Actual Hub E2E benchmarking follows usable model integration.
