# H2: what fills the primary GPU's verify round

Date: 2026-09-28. Issue: Strata-xeno #12 (hybrid runtime, stage H2), follow-up #26.

## Setup

- Exe `x7-sec` (sha `0af6c829390d`, commit 87e9922).
- Code prompt from `strata-codex-dispatch-detail-code256`, plus `--pool-priority 2 --adapt-swaps 8 --adapt-every 1 --adapt-secondary 8 --profile-decode-range`.
- `nsys profile -t cuda --cuda-graph-trace=node --capture-range=cudaProfilerApi`.
- Raw data: `%TEMP%\strata-claude-h2-nsys\decode.sqlite`. Per-kernel totals: `kernels.txt`.

The capture holds 57 verify rounds of 4 tokens (`route` n = 11007).

**Caveat.** Node-level graph tracing slows the host and adds per-node cost, so every absolute number below is inflated:
- The profiled round is 52 ms.
- The unprofiled round of the same arm is 47.7 ms (`strata-claude-sec1/code-4-Z.stdout`: 77 rounds, 3671.6 ms).

Read the table below as proportions.

## Primary (5060 Ti, stream 29) timeline

| class | ms/round | share |
|---|---|---|
| trunk: `native_mmvq_multi` 9.1, `gr_*` hyper-connection 5.8, `bf16_f32_mmvf` 1.8, GDN 1.7, rest 2.9 | 21.3 | 41 % |
| experts (`native_gu_q2` + `native_down_q2`) | 7.7 | 15 % |
| `wait_flag_ge` for the CPU share (after the GPU's own experts) | 8.6 | 17 % |
| `copy_from_mapped` of the CPU/4070 rows over x4 | 3.1 | 6 % |
| no kernel: one gap per round between `copy_indexed` and the next round | 9.9 | 19 % |

The other `wait_flag_ge` (the pool's plan, flagA) totals 17 ms over the whole capture, which is negligible.

The 4070 is busy for 7.6 % of the span (its experts take 3.6 ms/round).

## Trunk bytes (from the GGUF headers, `gguf_sizes.py`)

- Experts: 32,400 MiB.
- Non-expert tensors: 30,936 MiB. Of that:
  - 27,466 MiB is `per_layer_token_embd` and 260 MiB is `token_embd`. Both are lookups, not read per round.
  - The rest, about 3.2 GB, is read once per round.
- The largest item is `hc_{attn,ffn}_{up,down}` in **BF16**, 1,200 MiB, which the `gr_*` kernels read.

At 448 GB/s the trunk floor is about 7.5 ms/round. The profiled 21.3 ms is well above that floor. How much of the gap is profiler inflation is not measured yet.

The profile shows about 3,520 kernels per round.

## Conclusions

1. **H3 cannot pay by itself.** Inside a layer the primary finishes its own experts and then waits for the CPU. Less primary expert work only becomes more waiting.
2. **The mapped copy read rows the GPU never needed.** Fixed as `moe_hit_merge_mapped` (#26, commit 94d347d). It is bit-exact. Same-session ABBA `strata-claude-merge1` (x7-sec `0af6c829390d` vs x8-merge `fb77d2b89b47`, `--pool-priority 2`, paired 8 + 4070 swap 16) gave identical outputs in all 16 runs:
   - Thai: 43.43 → 44.91 tok/s (+3.4 %). Every new run was faster than every old one. `wait for rings` went from 14.22 to 13.57 ms/round.
   - Code: 69.20 → 69.87 tok/s mean. One new run was disturbed (61.95, CPU rows at 17.5 GB/s); without it the new runs are 71.65–73.68 against old runs of 66.70–70.82. `wait for rings` went from 20.54 to 18.94 ms/round.
3. **Next measurements (unprofiled or graph-level):**
   - Measure the real trunk time per round and the per-node graph overhead with `--cuda-graph-trace=graph`, which gives whole-graph durations only.
   - Measure the ~5.5 ms/round that the host counters do not cover. The unprofiled counters (rings 20.3, pool 17.3, mtp 2.4, host 1.3, commit 0.9) sum to 42.2 of 47.7 ms.
4. **Two options need the developer, so they are parked:**
   - Quantizing the BF16 hyper-connection weights. It breaks ADR 0001 bit-exactness.
   - Splitting the trunk across both GPUs (H4 on dense projections). It needs one exchange per layer over x4.
