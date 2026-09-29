# Prompt chunk size, borrowed cache slots and their RAM cost (for finding a cheaper way to the same speed)

Date: 2026-09-29. Branch `xeno/claude-merge-0.1.20`, exe built from `9c800c5`. Single-GPU serving flags as in
`strata-xeno.json` (only `--prefill` varies), the 8,023-token prompt. All runs: HIGH class, `--pool-priority 2`,
identical output (`8c6c97e1`). Logs: `%TEMP%/strata-claude-stage/chunk{A,S}*.std*`, `c4{a,b}.std*`.

## What happens today

The prompt path needs large per-chunk buffers. With an expert profile (the serving default), it does not keep
them allocated: it **borrows the expert cache's last slots** for them (`generate.cpp`, `lend` / `refill`):

1. At boot the cache is filled in profile order, so slot 0 holds the most used expert. The last slots hold the
   least-used experts that still made it into VRAM.
2. When a prompt arrives, the prompt path lays out its buffers over the last `k` slots and marks those experts as
   not resident. During the prompt they are streamed like any uncached expert.
3. When the prompt is done, `refill` copies them back into their slots before the first decode window. Decode
   always sees the whole cache.

**The hidden cost.** Under `--exclusive-primary-experts` (the default), an expert that lives in VRAM has no copy
in host RAM, which is what saves RAM. The lendable tail is the exception: those experts keep their host copies,
because `refill` reads them from there. A bigger chunk needs more slots, so a bigger tail is kept in RAM for the
whole session:

```
exclusive: cache slots 3893.. keep their host copies (the lendable tail)     # --prefill auto (8192)
exclusive: cache slots 5826.. keep their host copies (the lendable tail)     # --prefill 2048
```

## Measured

| `--prefill` | chunk buffers (borrowed VRAM) | slots in the lendable tail | private commit after load | prefill, 8K prompt | TTFT |
|---|---|---|---|---|---|
| 2048 (today's profile) | 1,112 MiB | 1,138 | 39.38 GiB | 20.48 / 20.54 s | 20.55 / 20.62 s |
| 4096 | 1,961 MiB | 1,782 | 40.17 / 40.20 GiB | 14.14 / 13.85 s | 14.25 / 13.95 s |
| auto (8192) | 3,660 MiB | 3,071 | 41.86 GiB | 10.79 / 10.64 s | 10.89 / 10.73 s |

Two runs per row, same session; the 2048 and auto rows were alternated (ABBA). The refill after the prompt costs
about 65 ms at 2048 and about 90 ms at 8192 (TTFT minus prefill). Decode speed was the same in every arm.

- **Why a bigger chunk is faster.** Every expert the chunk routes to is streamed once per chunk. Four 2K chunks
  stream the experts four times (97 GB of copies on this prompt); one 8K chunk streams them once (28 GB).
- **What it costs.**
  - VRAM is only borrowed during the prompt, and the refill is cheap.
  - The real cost is **+2.5 GiB of host RAM for the whole session** at 8192 (+0.8 GiB at 4096). That is the
    lendable tail's host copies, not the buffers.

## Where the borrowed 3,660 MiB goes (T = 8192, K = 10, H = 2560)

| buffer | size | shape |
|---|---|---|
| `Dm` (expert outputs before the combine) | 839 MB | T·K × H × f32 |
| `GU` (gate/up outputs) | 419 MB | T·K × 1280 × f32 |
| `Xq` (activations, q8_1, one row per (token, k)) | 236 MB | T·K × 2880 B |
| `H` (SwiGLU) | 210 MB | T·K × 640 × f32 |
| `Hq` | 59 MB | T·K × 720 B |
| ring slots (96 × 1.38 MB) + group slots | ~155 MB | |
| trunk, attention and QSA buffers | the rest | T × … |

Most of the size grows with **T·K**, the per-(token, k) buffers: 1.76 GB of 3.66 GB.

## Ways to keep the 8K speed at a lower cost (not implemented; each needs its own measurement)

1. **Per-group scratch for `GU`, `H` and `Hq`** (about 0.69 GB less to borrow at 8K, so about 0.5 GiB fewer tail
   host copies).
   - They only hold one MMQ group's rows at a time: 16 experts, typically about 2-3K of the 82K rows.
   - The gate/up product would write group-relative rows. That changes only buffer offsets, not arithmetic, so it
     should stay byte-identical; a parity test would confirm it.
2. **`Xq` by token instead of by (token, k)** (236 MB → 24 MB).
   - Quantize the T activation rows once, then let the product read row `src[i]`.
   - #32 lists this as "quantize once". It needs input-row indirection in MMQ, which may mean a ggml change.
   - It is exact: a row's q8_1 bytes depend only on that row.
3. **Refill the lendable tail from the pack on disk instead of from RAM.** This removes the tail's host copies
   entirely: −4.2 GB at 8192, and also −1.6 GB compared with today's 2048.
   - The price is reading the tail back from NVMe after each prompt that borrowed it: about 4.2 GB, roughly 0.7 s
     at 6 GB/s. The #11 NVMe tier already reads experts from the pack.
   - A cheaper variant refills only the slots the prompt actually overwrote.
   - Refilling in the background during decode is not exact: an expert served by the CPU instead of the GPU gives
     different floats. So the refill must finish before the first window, as today.
4. **Shrink `Dm` with a rank-streamed combine** (#32 step 5).
   - The combine is an fmaf chain over k = 0..9 per token. It can only run once all ten rows of a token exist,
     so `Dm` cannot simply be dropped.
   - Streaming it exactly means ordering the products by rank. That is a larger change, and the one #32 needs
     for the 4070 anyway.
5. **A middle chunk (4096) as a stopgap:** −31 % prompt time against 2048 (14.0 vs 20.5 s) for +0.8 GiB, against
   −48 % for +2.5 GiB at 8192.

Options 1 and 2 cut the borrowed size, and with it the tail's RAM, without changing any result. Option 3 moves
the tail's RAM cost to disk reads. Together they would bring 8K chunks to, or below, today's 2048 RAM.

## Dual-GPU note

The dual-GPU arms of the other reports use `--no-prefill-borrow`: the buffers are reserved in VRAM for the whole
session, and no tail keeps host copies. The cost there is cache slots, and so primary hit rate during decode, not
RAM. The same buffer reductions (1, 2, 4) shrink that reservation too.
