# CPU pool sweeps: priority, page lock, async 4070 launch, worker count

**Code:** branch `xeno/claude-q2-kernel`, K0–K4 plus the opt-in flags from `3780ec5` and `af4e19e`.

**Runs:**
- Runner `ab.py` (scratchpad): exe snapshotted with sha256, same-session ABBA, profiler off.
- Command identical to the benchmark runs: EXL3-only ranked profile, 6,653 primary slots, 8.5 GiB secondary, `--pool-workers 6`, MTP `--spec 4`, `CUDA_VISIBLE_DEVICES=1,0`.
- Outputs were identical across all arms in every prompt (32 runs).

**Artefacts:** `%TEMP%\strata-claude-cpu1` (exe `b871bf2`) and `%TEMP%\strata-claude-cpu2` (exe `32315ec`).

## Sweep 1: 5 workers, async launch, page lock (order B W S L L S W B)

| Prompt | Arm | tok/s (runs) | pool | CPU pool | CPU rows GB/s per run |
|---|---|---|---:|---:|---|
| thai | B base | 35.5 (36.0, 34.9) | 20.4 | 15.8 | 20.2, 19.0 |
| thai | W `--pool-workers 5` | 34.4 (36.6, 32.2) | 21.6 | 16.6 | 20.8, 16.9 |
| thai | S `--secondary-async-launch` | 38.0 (38.5, 37.5) | 17.8 | 15.1 | 20.7, 20.0 |
| thai | L `--lock-cpu-experts` | 38.3 (39.6, 36.9) | 17.7 | 13.4 | 24.6, 21.7 |
| code | B | 64.9 (65.3, 64.6) | 22.8 | 16.8 | 19.6, 20.5 |
| code | W | 59.6 (63.2, 56.1) | 27.3 | 21.0 | 18.0, 14.4 |
| code | S | 66.1 (67.6, 64.6) | 21.7 | 16.8 | 21.3, 18.9 |
| code | L | 60.4 (52.9, 68.0) | 26.9 | 19.8 | 13.8, 22.1 |

## Sweep 2: thread priority (order B P M Q Q M P B)

| Prompt | Arm | tok/s (runs) | pool | CPU pool | CPU rows GB/s per run |
|---|---|---|---:|---:|---|
| thai | B base | 26.9 (26.1, 27.7) | 30.9 | 24.2 | 11.9, 13.6 |
| thai | **P `--pool-priority 2`** | **36.7** (35.8, 37.6) | 19.5 | 14.8 | 20.3, 21.5 |
| thai | M lock + async | 35.7 (35.2, 36.2) | 20.2 | 15.4 | 19.3, 20.7 |
| thai | Q lock + async + priority | 34.6 (34.0, 35.3) | 21.2 | 15.1 | 20.9, 20.0 |
| code | B | 57.0 (59.9, 54.1) | 29.6 | 22.4 | 16.0, 14.1 |
| code | **P** | **66.8** (67.2, 66.4) | 21.5 | 15.7 | 21.3, 21.5 |
| code | M | 61.6 (57.9, 65.4) | 25.7 | 18.5 | 17.0, 19.5 |
| code | Q | 65.2 (63.6, 66.8) | 22.7 | 16.3 | 19.8, 21.4 |

The first thai B run overlapped a tokenizer load on the CPU (the author's mistake) and is marked disturbed. The second B run had no known disturbance and was equally slow.

## Findings

- **The CPU swing comes from other processes preempting pinned pool workers.**
  - Each run is either "clean" (CPU rows 19–24 GB/s) or "disturbed" (12–16 GB/s). Within a session the disturbance hits the normal-priority arms.
  - In sweep 2, every normal-priority base run was disturbed and every `--pool-priority 2` run was clean, giving **+17 % code and +36 % Thai** against the same-session base.
  - Workers are hard-pinned (`SetThreadAffinityMask`), so a preempted worker cannot migrate, and the layer waits for its task.
- **Page lock (`--lock-cpu-experts`)** locks 14,947 MiB in 11,321 ranges (the CPU-owned experts). It protects against trimming but not against preemption: one of its sweep-1 runs was disturbed (13.8 GB/s). On top of priority (Q vs P) it adds nothing measurable. It stays opt-in and is not recommended now.
- **Async 4070 launch (`--secondary-async-launch`)** removes the launch from the host path (`secondary timing launch` 3.6 → 0.1 ms/round). However, the helper thread on an E-core enqueues more slowly, and `finish()` then waits for it: secondary finish rose from 0.5–0.8 to 4–5.8 ms/round. It is net neutral or negative. It stays opt-in, is not recommended, and is recorded so it is not retried unchanged. A helper on a P-core would take a core from the pool.
- **`--pool-workers 5`** (P-cores only) was worse than 6 (5P + 1E) in both prompts. The E-core worker contributes and is not a straggler, so the next topology step is more workers (9, 13) at high priority.
- **Measurement rule adopted:** every speed measurement passes `--pool-priority 2` in every arm (`AGENTS.md`).

## Decision parked for the developer

Whether the serving launcher (Claude Code traffic) should also run Strata with `--pool-priority 2`. These sweeps say it removes the dominant source of run-to-run loss. It changes the serving default, which the developer reserves.
