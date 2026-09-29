# #11 N0: four-tier replay of real routing traces

Date: 2026-09-29. Issue: Strata-xeno #11 (capacity mode, NVMe cold tier), step N0. Tool:
`tests/xeno/perf/nvme_sim.py`. Raw table: `2026-09-29-nvme-n0-simulation.txt`.

## Setup

- **Traces:** the ten `--route-trace` recordings from 2026-09-28 (`%TEMP%/strata-claude-traces`), 256 tokens each:
  - six training prompts,
  - the four benchmark prompts.
- **Replays:** each benchmark trace alone, and all ten concatenated as one session of about 2,560 tokens, which crosses several topic switches.
- **VRAM tiers:** static, as at runtime. VRAM0 holds the 6,653 and VRAM1 the 6,602 top EXL3-ranked experts.
- **RAM cache:** starts with the next-ranked experts.
- **NVMe:** everything else.
- **Miss accounting:** a miss loads once per verify window (the N5 grouping) and is admitted to the RAM cache. The cache then evicts by LRU, LFU, decayed LFU (×0.97 per window), or decayed LFU seeded with the EXL3 counts (`prior`).

## Result (session replay, `prior` policy, the best in every row)

| RAM cache | entries from NVMe | NVMe loads / verify token | NVMe MB / verify token |
|---|---|---|---|
| 4 GiB | 1.8 % | 7.8 | 10.3 |
| 8 GiB | 0.7 % | 3.2 | 4.2 |
| 12 GiB | 0.2 % | 0.7 | 1.0 |
| 16 GiB | 0 % | 0 | 0 (every non-GPU expert fits) |

Today the host needs about 15.6 GB for the experts neither GPU owns.

- **8 GiB cache:** the host returns about 7.6 GB more, and 0.7 % of routed entries come from NVMe.
- **12 GiB cache:** it returns about 3.6 GB, at 0.2 %.

The benchmark traces alone show the same order:
- code at 8 GiB: 1.3 %, 7.2 MB per token;
- sky, with the lowest primary hit rate: 1.4 % at 8 GiB.

Among policies, `prior` wins every row. LRU and decayed LFU are close behind, and plain LFU is worst at small caches because it keeps stale early favourites.

## What this does not say

- **Bandwidth is not the question; latency is.** At 4.2 MB per verify token and roughly 70 verify tokens per second, NVMe reads run at about 300 MB/s, far below the drive's rate. But a miss sits inside a layer, on the critical path, so per-read latency decides the cost. N1 must measure it.
- **The traces are short and the caches start warm.** Longer sessions and cold starts need their own replay.

## Next

N1: the demand-only runtime tier. NVMe feeds a pinned ring, then a bounded RAM cache, then the CPU. It uses the `DirectFile` reads that placement-first already uses, and the ownership state `NVME`. Acceptance is the #11 list, with parity checked against a same-session all-memory run.
