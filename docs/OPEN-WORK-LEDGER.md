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
| #28 P1 host-memory modes | Opened. Supersedes #17 (closed): normal / locked / hybrid, layer-contiguous layout. |

**Waiting for the developer:**
- Deploying the new engine to `engine-xeno/strata.exe`, which picks up the #27 fix, the exclusive default and placement-first.
- Whether the serving default uses `--process-priority 2`, given the risk of desktop starvation.
- The dual-GPU serving profile (#12/#21).
- Quantizing the BF16 hyper-connection weights, which breaks ADR 0001.
- The trunk split across GPUs (H4 on dense projections).

The source audit at Flash-Next commit `8bf2c853417c5d381c1a4d4d1c20c3f55b5ad1c0` supersedes the earlier report at `ced0445`.
