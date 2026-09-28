# Q2_0 CPU-order kernel: four rows per warp (K0)

Commit `ab93dfd` on `xeno/claude-q2-kernel`, a child of `xeno/codex-secondary-timing@f653657`.

## Why

A roofline check of Codex's Nsight capture (`%TEMP%\strata-codex-primary-nsys-code256\decode.sqlite`) put the Q2_0 expert kernels at about **42 GB/s effective weight throughput per expert on the 4070 SUPER**. That figure is useful model bytes divided by kernel time, fitted over 3,523 launches whose gridY equals the real expert count. On the 5060 Ti the same kernels reached **≤ ~22 GB/s** (an estimate, because gridY there is the capacity, not the count). The primary's Q2 gate/up + down took about 33 ms of a ~72 ms round.

The cause is in the source. The bit-exact parity path `row_dot_q2_cpu_order` (`src/kernels/cuda/iq_kernels.cu`) mirrors the CPU's eight FP32 accumulators with lanes 0–7 only, and `native_gu_kernel` / `native_down_kernel` gave each warp one row. That left **24 of 32 lanes idle**.

## Change

`native_gu_q2_kernel` and `native_down_q2_kernel` give each aligned 8-lane group its own row, so each warp computes four rows and each block 32 rows. Per-row arithmetic, reduction order and the lane-0 `corr` chain are unchanged. The shuffles take the group's mask instead of `0xff`. The new kernels serve only the Q2_0 path with CPU-order scales. Other formats and the unscaled Q2_0 path keep the old kernels.

## Correctness

- `native_q2_pool_hit_parity` compares the CPU pool with the GPU hit path bit for bit, including a two-entry group. On the 5060 Ti (`CUDA_VISIBLE_DEVICES=1`) it passed before and after the change: **0/7680 differing** at layers 0, 20, 33 and 47.
- **Mutation check:** writing the row result from warp lane 0 only (so three rows in four are never written) made the same test fail **7680/7680, exit 1**. The test does exercise the new kernel.
- CTest with `CUDA_VISIBLE_DEVICES=1,0`: all pass except `ple_parity`, which cannot open its relative-path PLE shard. It fails the same way on the unchanged baseline.
- **End to end:** in every prompt below, all four ABBA runs (A, B, B, A) produced **identical 256-token outputs**.
- All 16 runs, the unchanged baseline A included, differ from Codex's accepted IDs at the same token per prompt (code 237, Thai 60, sky 5, long 29). This is not caused by K0, and not by the ggml-cpu SIMD mode. Codex's own unchanged exe, rerun with its byte-identical command, now diverges at the same token, so the drift comes from boot-time state, first visible in prefill (see `2026-09-28-ggml-cpu-build-mode.md`). Parity is therefore judged against the same-session baseline A.

## Throughput: paired ABBA, profiler off

**Setup:**
- A = `f653657` unchanged (`q2-base-build`); B = A + K0 (`q2-kernel-build`). Both builds use ggml-cpu AVX2.
- Command identical to Codex's `strata-codex-dispatch-detail-*` / `compact-parity-*` runs:
  - native Q2_0 pack, EXL3-only ranked profile
  - 6,653 primary slots, 8.5 GiB secondary tier, six pool workers
  - MTP `--spec 4`, `--prefill 2048`, `--max-context 8192`, `--vram-reserve-mib 2400`, secondary free floor 640 MiB
  - `CUDA_VISIBLE_DEVICES=1,0`
- Order A B B A per prompt.
- Artefacts: `%TEMP%\strata-claude-k0\{prompt}-{i}-{arm}.stdout/.stderr` and `summary.log`.

Means of the two runs per arm, in ms per verify round (Strata's own counters):

| Prompt | Arm | tok/s (runs) | wait for rings | pool | CPU pool | secondary finish | secondary wait | CPU rows GB/s |
|---|---|---|---:|---:|---:|---:|---:|---:|
| code | A | 45.0 (41.6, 48.4) | 32.5 | 33.8 | 23.3 | 4.6 | 4.1 | 15.7 |
| code | B | **61.1** (52.7, 69.5) | **21.9** | 26.2 | 19.7 | **1.1** | 0.6 | 18.6 |
| thai | A | 32.5 (32.1, 33.0) | 17.3 | 21.1 | 14.4 | 2.4 | 2.2 | 21.5 |
| thai | B | 35.9 (34.3, 37.6) | 14.8 | 19.6 | 15.2 | 0.5 | 0.2 | 20.4 |
| sky | A | 38.9 (38.5, 39.2) | 23.2 | 28.3 | 19.1 | 4.4 | 4.1 | 18.7 |
| sky | B | 41.6 (39.2, 44.0) | 19.0 | 28.7 | 21.7 | 1.3 | 0.5 | 16.6 |
| long | A | 42.0 (41.8, 42.2) | 30.3 | 20.8 | 9.5 | 6.7 | 6.4 | 18.9 |
| long | B | **58.7** (58.1, 59.3) | **18.6** | **16.4** | 10.9 | **1.2** | 0.9 | 16.7 |

Tier hits did not change between arms: primary / secondary / CPU of routed entries was code 56.9/25.6/17.5 %, Thai 42.1/27.5/30.3 %, sky 43.2/28.3/28.5 %, long 60.1/26.9/12.9 %.

## Reading the counters

- **What K0 changes:**
  - `wait for rings` (the host waiting for the primary GPU) fell 10.6 ms (code), 11.8 ms (long), 4.2 ms (sky) and 2.5 ms (Thai). The drop scales with the primary hit share.
  - Secondary finish fell 3–5.5 ms in every prompt, because the 4070 runs the same kernel. It now finishes almost entirely inside the CPU pool.
- **Where the time is left:**
  - Code and long gain **+36 % and +40 %**, above the 13.6 % noise gate in every pair. Both also have the lowest CPU share.
  - Thai (+10 %) and sky (+7 %) are below the gate. In those prompts `pool` was already larger than `wait for rings` before K0, and it did not shrink (Thai 21.1 → 19.6, sky 28.3 → 28.7), so the round is set by the CPU experts.
- **The CPU still swings between runs:**
  - The two B code runs differ by 17 tok/s (52.7 vs 69.5) with the same code and identical outputs.
  - The counters put the whole difference in the CPU pool: rows ran at 13.6 vs 23.6 GB/s and the pool took 25.1 vs 14.3 ms.
  - `wait for rings` stayed at 22.7 / 21.1 ms.
  - The CPU swing is an open item; K0 does not affect it.

## What this settles and what it leaves

- **K0 is kept.** It is bit-exact, it produced identical outputs in 16 of 16 runs, and it is a clear gain wherever the primary GPU was the long pole.
- **The next GPU-side steps are unchanged.** K1 moves the `hx` precompute into activation quantize. K2 is the packed `dp4a` integer dot. K3 reuses one weight load for several routed tokens. The empty PCIe-share launches remain. Each will be judged with the same per-stage table.
- **Thai and sky need less CPU work per round.** Per the counters above, that means more primary hits (ranked profile, adaptive swaps) or misses served by the 4070 over x16 DMA. It also means finding the cause of the CPU-rows swing (13–24 GB/s on identical work).
- **An earlier estimate here was wrong.** A prediction that a faster Q2 kernel would gain "single-digit %" was built by stitching an Nsight trace to another run's counters. The measured code gain is +36 %. That estimate is withdrawn, and `AGENTS.md` now requires per-stage counters from the same paired runs.
