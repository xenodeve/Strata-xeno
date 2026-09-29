# Open work

Strata-xeno PRD: xenodeve/Qwen3.8-Flash-Next-Tuning#1. Engine implementation and measurement issues are paired. The approved 2026-09-27 handoff is the phase scope.

| Phase | Engine | Measurement | Current state |
|---|---|---|---|
| 1 GPU expert correctness | xenodeve/Strata-xeno#1 | xenodeve/Qwen3.8-Flash-Next-Tuning#10 | Developer accepted Q2_0 `double-exp` output as current baseline (ADR 0001); candidate branch xeno/afk-phase1 pushed at 9ffb846; cache parity green; integration review pending |
| 2 CPU core policy and dual architecture build | #2 | #11 | Built and reviewed on xeno/afk-phase2; PR not opened |
| 3 Static 4070 expert tier | #3 | #12 | Experimental child branch `xeno/codex-secondary-timing` corrected decode counter labels, packed pinned secondary metadata and compact D2H with four-prompt 256/256 parity; no robust throughput gain or default choice yet. Native verifier Nsight traces show primary Q2 expert work dominates measured kernel duration; the 5,000-slot primary probe had more CPU-routed entries and GPU wait than the 6,653-slot baseline in alternating traces. See `docs/reports/2026-09-28-primary-gpu-nsys.md`; cost-aware secondary work, graceful fallback and long sessions remain open. **K0** (`xeno/claude-q2-kernel@ab93dfd`): four CPU-order Q2_0 rows per warp, bit-exact, same-session ABBA +36 % code / +40 % long, Thai/sky below the gate (CPU-bound) — `docs/reports/2026-09-28-q2-rows-per-warp.md`. Next K1 `hx` precompute, K2 `dp4a`, K3 multi-token row. Open: CPU-rows swing 13–24 GB/s on identical work; prefill routing depends on boot state (`2026-09-28-ggml-cpu-build-mode.md`). |
| 4 Exclusive swaps | #4 | #13 | Static opt-in primary ownership released 6.43 GiB host pages and lowered private commit 43.37→36.94 GiB with sky 96/96 parity; newcomer/victim swaps and recovery still open |
| 5 4070 prefill | #5 | #14 | Depends on Phase 3 |
| 6 Claude Code serving | #6 | #15 | Upstream #7–#9, count_tokens, billing-header normalization, thinking signature, loop guard, stop sequences, dead-child health, and document blocks green; #10 parked for token-5 divergence; further compatibility open |
| 7 Long context | #7 | #16 | Depends on memory fit |
| 8 Speculation and sampler | #8 | #17 | Depends on correctness |
| 9 Telemetry and default decision | #9 | #18 | Final measured gate |

## Hybrid runtime and decode speed (xenodeve/Strata-xeno#12, branch `xeno/claude-q2-kernel`)

State on 2026-09-29. Every number comes from a same-session ABBA with identical greedy outputs (tool: `tests/xeno/perf/ab.py`).

| Item | State |
|---|---|
| H1: 4070 swaps `--adapt-secondary` | `87e9922`. Opt-in: the CPU pool drops, tok/s is unchanged because the round is bound by the primary GPU. |
| H2: primary-GPU Nsight | Done: `docs/reports/2026-09-28-h2-primary-timeline.md`. The trunk is 41 % of the primary's timeline, its own experts 15 %. |
| Mapped-copy merge | #26 closed (`94d347d`): thai +3.4 %. |
| Swap host cost | `86b6419` (parallel staging copy) and `5748dc5` (4070 swaps on their own thread): code +2 %, thai +2.2 %. New counters: `adaptive tier` thread/apply/join, `adapt detail`, `verify edges`. |
| 4070 launch as a CUDA graph | #25 closed (`3ec52cb`), on by default: thai +5.4 %, code +1.2 %. |
| 4070 partials clear dropped | `aec0da7`: bit-exact, no measurable change. |
| `gr_down_multi` one-wave geometry | Tried and not kept: `wait for rings` unchanged. The parity test `xeno_gr_multi_parity` was kept (`21a2ed0`). |
| HIGH process class | `0a2d70d`: `--process-priority` is opt-in; code +2.6 %, thai +1.3 %. Every measurement arm now runs at HIGH class. |
| Measurement tools in the repo | #24 closed (`99c5cdf`). |
| #16 pool workers | `ExpertPool::rest()` (`ba0cade`, default on) lets 13 workers win: code 89.2, thai 53.8 tok/s, CPU rows 42.8 GB/s. The engine default is already all cores; the serving worker count is the developer's call. |
| #27 serving decode | Root cause: the PCIe read path stalls the verify window on the x4 primary. `--pcie-frac` now defaults to 0 (`704ac58`): serving 8K decode 3.24 -> 61.92 tok/s. |
| Exclusive 5060 Ti, default | `fdaddbe`: it works together with prompt borrowing, and pageable staging runs on 4 workers. Serving 8K: -7.86 GiB host RAM, decode +5.3 %, TTFT unchanged. |
| Placement-first cold start | `1d35b34`..`77bbd38`: the arena is reserve-only, GPU tiers fill straight from the pack with unbuffered pipelined reads, and only host-owned experts are committed. No boot spike: dual peak 31.0 GB (was 46.3), serving 38.9 GB (was 46.6). Boot ~161 s -> ~18 s. |
| Exclusive 4070 (#4 A+B) | `5213b90`, `ea04a91`, default in `0a0791d`: paired 4070 swaps. The longer A/B showed no trade-off (code 84.75 vs 85.76, thai 51.64 vs 51.36); peak RAM 39.8 -> 31.3 GB. |
| #11 NVMe cold tier | N0 simulator (`c2aabee`); N1 opt-in `--ram-cache-gib` (`bf0baf2`). The output is identical, but for this model it is a poor trade: 12 GiB gives 34.3 GiB commit at -21 % decode, 8 GiB gives 30.3 GiB at -41 %. It is for models that do not fit. |
| #28 P1 host-memory modes | Opened. Supersedes #17 (closed): normal / locked / hybrid, layer-contiguous layout. |
| #33 pipeline timeline | `STRATA_TIMELINE=<file>` + `tests/xeno/perf/timeline.py`: every thread and GPU lane on one clock (prompt phases per layer, each expert copy and why the copy engine idled, decode round stages, pool workers, 4070, server). First runs answer #31 for the default path (router sync blocks the only issuing thread: 4.5 s of copy idle on 8K) and show in-flight copies stalling during `gdn` / `qsa attn` in issuer mode: `docs/reports/2026-09-29-pipeline-timeline.md`. Off-mode cost not yet A/B-measured against the pre-timeline exe. |

**Waiting for the developer:**
- Deploying the new engine to `engine-xeno/strata.exe`, which picks up the #27 fix, the exclusive default and placement-first.
- Whether the serving default uses `--process-priority 2`, given the risk of desktop starvation.
- The dual-GPU serving profile (#12/#21).
- Quantizing the BF16 hyper-connection weights, which breaks ADR 0001.
- The trunk split across GPUs (H4 on dense projections).

The source audit at Flash-Next commit `8bf2c853417c5d381c1a4d4d1c20c3f55b5ad1c0` supersedes the earlier report at `ced0445`.

## Session state before compaction (2026-09-29)

**Speed against a bandwidth ceiling.** This is an estimate, not a measurement. Per verify round, code window 4 with 3.36 accepted tokens:
- trunk ~3.2 GB at 448 GB/s = 7.1 ms;
- 5060 Ti experts ~1.1 GB = 2.4 ms;
- CPU experts ~0.27 GB at 60-108 GB/s = 2.5-4.5 ms, overlappable;
- 4070 experts ~0.55 GB = 1.1 ms, overlappable;
- output head + MTP ~2.4 ms.

The floor is therefore about 12 ms/round. Measured is ~39.5 ms/round on code (85 tok/s), about 30 % of the ceiling; thai is about 45 % (53 of ~115-120 tok/s). The largest gaps:
1. The trunk `mmvq` reads at ~36 % of bandwidth (H3 Nsight). Two fixes were tried and neither was kept: the rows-per-block variant was slower, and the upstream layout was not faster.
2. The layer-by-layer GPU<->CPU ping-pong, with ~7 ms/round of GPU idle.
3. The mapped-copy and 4070 waits, ~2-3 ms/round.

**Prefill profile, corrected 2026-09-29:** 2K prefill: span 7.79 s, GPU busy 2.84 s (37 %), memcpy 2.90 s for 20.05 GB. ~~Expert streaming over x4 bounds prefill~~ - wrong: copy and compute overlap only 0.65 s and the compute thread spends 4.88 s in CUDA API calls, so prefill is launch-bound (#29). Tensor-core GEMMs are not bit-identical across sm_89/sm_120 (`docs/reports/2026-09-29-gemm-cross-arch.txt`), so #5 is an opt-in for later. #29 item 1 (batched QSA indexer append, 767fcfc): 7.17 -> 6.75 s, identical output.

**Upstream 0.1.20 merge (#30), branch `xeno/claude-merge-0.1.20`, 2026-09-29.** Merge commit `2a9b26c`, then `c7a469e`, `ff2bdae`.
- The merge introduced one bug of its own, fixed before commit: the stager branch lost its `copied` event, so MMQ read ring slots early (NaN under exclusive ownership).
- Measured: 2K prefill −19 % single GPU and −12 % dual, with identical output. 8K with `--prefill auto`: 339 → ~700 tok/s, but the output differs from pre-merge (MMQ numerics).
- Prefill breakdown at 8K, single GPU (`STRATA_PREFILL_TIMING`): GPU0 timeline 10.8 s. MoE 5.4 s, of which the x4 copies take 4.1 s. QSA 2.1 s, GDN 1.6 s, hc/embed 1.6 s. The 4070 does no prefill compute. The CPU staging threads take 3.7 s (overlapped, pageable arena).
- **Unexplained:** at 8K the x4 link idles ~200 ms per layer. The gap ends at the next router synchronize while the copy thread sits inside `cudaMemcpyAsync`. Ruled out: stager threads, ring size, mapped router ids, stream flushes.
- **Cross-arch numerics** (#5):
  - A single floating `mma.sync` differs between sm_89 and sm_120; integer `mma.sync` is exact.
  - MMQ is identical once stream-k is off (default since `ff2bdae`, no speed cost). swiglu and q8_1 quantize are identical too.
  - A bit-exact MoE-layer offload to the 4070 is designed, **parked on the 4070 VRAM decision** (#5).
- Still to validate on the merged build: decode ABBA, `--serve`, `--pcie-frac` on x4 after upstream's 0.1.14 copy kernel.

**Open questions:**
- Short prompts show TTFT +0.12-0.19 s with placement-first (4/4 runs), while the 8K prompt is unchanged. The cause is unknown.
- `tests/xeno/short_read_probe.py` appeared untracked in this worktree. It was not written by this session and was left alone.
- `--serve` has no paired 4070 swap path yet, so under `--serve` the exclusive 4070 runs without 4070 swaps.

**Waiting for the developer:**
- Deploying the new engine to `engine-xeno/strata.exe`, which picks up #27, the exclusive defaults and placement-first.
- `--process-priority 2` and 13 workers for serving, which are desktop trade-offs.
- Quantizing the BF16 hyper-connection weights (breaks ADR 0001).
- The trunk split across GPUs (H4).
- #28 host-memory modes.
