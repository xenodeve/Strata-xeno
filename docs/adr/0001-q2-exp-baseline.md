# ADR 0001: Accept the canonical Q2_0 expert baseline

Status: Accepted, 2026-09-27. Developer decision in the Strata-xeno AFK session.

## Context

Phase 1 aligns the native Q2_0 GPU hit path with the CPU expert pool. Matching its Q8 scale, accumulation order, rounding, and SwiGLU required the CPU native-Q2_0 path to compute `float(exp(double))` where it previously used `std::exp(float)`. The cache-on and forced CPU-miss arms of the resulting build matched all 256 greedy output IDs on each of four fixed prompts (sky, Thai, code, long). This checks cache placement parity within the new build; it does not prove the answer quality is unchanged from the older build.

The new CPU-only output differs from the previous build: the first known difference is at output token index 77 for sky and 60 for Thai. A controlled run restoring the old `std::exp(float)` line restored the old sky baseline. The candidate code is commit `8c28ad0`; measurements and limitations are in `docs/reports/2026-09-27-cache-investigation.md` and the saved `final256-*` artifacts under `%TEMP%\strata-afk-runs`.

## Decision

Accept this output change as the current baseline and continue the performance work. Record the difference for diagnosis if end-to-end Claude Code quality becomes worse; it is not a blocker by itself. Do not label the changed output better or worse without quality evidence. Keep the bit-exact CPU/GPU Q2_0 contract as the correctness gate for later GPU placement work.

## Consequences and recovery

- Phase 1 can proceed to review and Phase 3 can use the new build as its greedy reference.
- Later phases compare cache placement and device tiers against this accepted baseline, using the same fixed prompts and token IDs.
- At the final quality gate, compare real Claude Code tasks and the fixed prompts against the old build. If a regression appears, isolate this `exp` change by rebuilding with the old CPU expression and testing the same tasks before changing kernels or reverting unrelated performance work.
- The four 256-token prompt pairs and rough speed figures are diagnostic evidence, not a broad quality evaluation or precise speed claim.
