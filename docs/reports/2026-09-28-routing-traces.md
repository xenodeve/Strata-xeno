# Routing traces: static profile ceiling and the case for adaptive swaps

**Tool:** `--route-trace PATH` (`3780ec5`) appends every verify-window layer's routed expert ids.

**Traces:**
- Ten 256-token runs, `--pool-priority 2`, exe `x2-exp`, artefacts in `%TEMP%\strata-claude-traces`.
- **Training prompts (6):** Python async refactor, bash script, JSON config review, C++ race bug, and two Thai prompts (networking, system migration). They are disjoint from the benchmark prompts.
- **Benchmark prompts (4):** code, thai, sky, long.

**Scripts** (scratchpad):
- `route_tools.py`: stats, profile writer, static tier simulator.
- `blend_profile.py`: EXL3 counts blended with the traces, evaluated held-out.
- `adapt_sim.py`: decayed-LFU swaps against the oracle.

**Simulator check:** the static simulator reproduces the measured tier split exactly (thai-net: 60.4 / 25.4 / 14.3 % primary / secondary / CPU, in both the simulator and `tier hits`).

## Static ranking is near its ceiling

The profile is placed as 6,653 primary slots and 6,602 secondary slots. The held-out results are for the four benchmark traces:

| Profile | code P / CPU | thai P / CPU | sky P / CPU | long P / CPU | mean CPU |
|---|---|---|---|---|---|
| EXL3 router counts (current) | 56.9 / 17.5 | 42.1 / 30.3 | 43.2 / 28.5 | 60.1 / 12.9 | 22.3 % |
| 6 Strata training traces only | 50.8 / 21.5 | 44.7 / 28.8 | 40.3 / 28.6 | 60.9 / 14.7 | 23.4 % |
| Blend, w = 0.5 (best) | 53.9 / 19.0 | 44.0 / 28.5 | 41.2 / 27.4 | 61.9 / 12.6 | 21.9 % |

- Six short prompts are a worse prior than the EXL3 counts collected over many sessions.
- Blending gains only 0.4 points of CPU share.
- Per-prompt working sets differ too much for any single static order.

## Adaptive swaps (simulated)

**Setup:** decayed LFU (0.97 per round), hysteresis 1.0, up to N primary swaps after each verify round, starting from the EXL3 placement, 256 tokens (~75 rounds).

| Prompt | static P / CPU | 4 swaps/round | 8 | 16 | 32 | oracle (own trace) |
|---|---|---|---|---|---|---|
| code | 56.9 / 17.5 | 61.8 / 15.8 | 64.0 / 15.2 | 66.7 / 14.6 | 70.6 / 13.9 | 85.9 / 1.0 |
| thai | 42.1 / 30.3 | 55.7 / 21.3 | 61.8 / 18.3 | 69.4 / 15.3 | 76.0 / 13.6 | 91.0 / 0.0 |
| sky | 43.2 / 28.5 | 51.5 / 23.3 | 57.7 / 19.7 | 65.6 / 15.8 | 74.5 / 12.4 | 93.2 / 0.0 |
| long | 60.1 / 12.9 | 67.6 / 10.6 | 70.6 / 9.6 | 74.6 / 8.7 | 79.5 / 8.0 | 92.8 / 0.0 |

- Session-adaptive placement nearly halves the CPU share on the CPU-bound prompts (Thai 30 → 15 %, sky 28 → 16 % at 16 swaps/round), and longer sessions move further toward the oracle.
- This is the largest lever measured so far for Thai and sky, where the CPU pool, not the primary GPU, sets the round time.
- **Caveats:** the simulation ignores the swap cost (1.38 MB H2D per swap on the 5060 Ti's x4 link, overlapped with MTP drafting in the existing implementation) and assumes a swap takes effect the next round.

**Existing implementation:** `generate.cpp`'s `adapt()` already swaps asynchronously on its own stream while the MTP drafts. It is blocked only with `--exclusive-primary-experts`, because an evicted expert must keep a host copy for the CPU. A same-session sweep of `--adapt-swaps` 8/16/32 without exclusive-primary is running (`%TEMP%\strata-claude-adapt1`). Making it work together with exclusive-primary is Phase 4's paired D2H swap.
