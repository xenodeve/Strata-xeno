# Static Exclusive Primary Expert Tier Implementation Plan

**Goal:** Release host pages for expert blobs that have a verified, static owner on the 5060 Ti, preserving Q2_0 tokens and enabling more RAM headroom for the second CUDA context.

**Spec:** `docs/superpowers/specs/2026-09-28-exclusive-primary-tier-design.md`.

**Constraints:** Windows/i5-13500/Q2_0 only in this slice; flag defaults off; no user launcher/profile edits, PR or main merge; keep 4070 SUPER >=2560 MiB free if it is used; no dynamic ownership swaps yet.

### Task 1: Host-blob ownership and page-release primitive

**Files:** modify `include/strata/core/expert_source.hpp`, `src/core/expert_source.cpp`, add `tests/xeno/exclusive_host_pages.cpp`, update `CMakeLists.txt`.

- [x] Write a failing two-expert test: claimed pair refused, neighbor readable, aligned released-byte count, duplicate refused. A real small CUDA-registered backing also refused release; OS-locked refusal is implemented but not separately fault-injected.
- [x] Implement Windows page decommit only after backing checks; keep VA/offset layout. On error, leave ownership unpublished for that pair and return a reason.
- [ ] Run the test and build dual-arch `strata`; commit.

### Task 2: Static primary ownership after verified profile fill

**Files:** modify `src/program/generate.cpp`, `tests/xeno/cache_tokens.py`; report under `docs/reports/`.

- [x] Add strict opt-in flag and reject cache borrow, adaptive swaps, forced CPU miss, missing profile or non-Q2 native format.
- [x] After primary profile fill and slot 0 byte verification, claim only `xcache.slot_of(layer, expert) >= 0`. Log number/bytes decommitted. Keep secondary copy-kept expert tier unchanged.
- [x] Run fixed sky 96-token prompt on one and two cards; both matched accepted raw IDs 96/96.
- [x] Measure process private commit before/after: 1,000 primary slots 37.68→36.39 GiB, 5,000 primary + secondary 43.37→36.94 GiB. Commit/push and issue update follow review.

### Task 3: Production Phase 4 ownership swaps (later)

Build a spare RAM slot, H2D newcomer and D2H victim ordering, then atomically publish ownership only after both events. Add decayed routing counts, hysteresis, bounded swap budget and aligned unbuffered pack-read recovery. Require parity through swaps and the issue's measured decode-loss gate before calling Phase 4 complete.
