# Static exclusive primary expert ownership — Phase 4 first slice

Status: implementation design, not a Phase 4 completion claim. Source: engine #4, tracker #13, three-tier design tracker #2, Phase 3 compute report and Q2_0 baseline ADR 0001.

## Why this slice comes next

The 5060 Ti currently caches profile-ranked experts **as copies** of a 31.64 GiB committed host arena. The second CUDA context could not initialize while roughly 29 GiB of that arena was CUDA-registered in slices; the opt-in dual-card path made the entire host arena pageable. A static, exclusive primary set can release the host pages of verified resident blobs and create RAM headroom before full two-card performance tuning. The later Phase 4 swap protocol remains separate work.

## Contract

Only with an explicit `--exclusive-primary-experts` flag, native Q2_0, a filled profile cache, `--no-prefill-borrow`, `--adapt-swaps 0`, and no forced CPU-miss diagnostic: a `(layer, expert)` pair verified resident in the primary 5060 Ti cache may become GPU-owned. After ownership is published, `ArenaExpertSource::blob` refuses that pair. The primary hit and prefill resident paths compute from GPU; the CPU pool and host-to-GPU miss path must never read the released host range. If any required condition fails, refuse startup rather than compute from decommitted bytes. Single-GPU and non-exclusive defaults remain byte-for-byte unchanged.

On Windows, the dual-card host arena uses normal pageable anonymous pages. For each exclusively GPU-owned blob, decommit only whole OS pages strictly inside that blob. Keep the original virtual address and per-layer offsets; do not repack host blobs in this first slice. Track committed bytes released and print them. The few boundary pages left committed are a known overhead; the ownership table, not decommit alignment, determines whether `blob()` may return a pointer. An error in any decommit aborts startup. The source must not decommit CUDA-registered, OS-locked or large-page backing.

## Risks and proof gates

1. A test source with two small blobs proves that a verified resident pair becomes unreadable through `blob()`, another pair stays readable, and close releases the whole VA reservation.
2. Profile fill and byte verification precede host release. One model prompt must reach both prefill and native verify windows without accessing a GPU-owned host blob. Greedy token IDs must match the accepted baseline and the copy-kept same-binary arm.
3. On the target Windows machine, committed process memory must fall roughly by the sum of decommitted interior pages, then recover on process exit. Check that the host arena is actually pageable and that no cache borrow or adaptive swap is active.
4. Measure speed separately. Decommitting hot host experts can alter page cache and CPU-miss locality; a RAM saving alone does not prove peak performance. The user requires the display 4070 SUPER to retain at least 2560 MiB free whenever the secondary tier is on.

Full Phase 4 later adds H2D newcomer and D2H victim using a spare RAM slot, two-event ownership publication, decayed-count hysteresis, and an aligned unbuffered pack recovery path. No swap is authorized by this static-only slice.
