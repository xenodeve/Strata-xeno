# Dual-GPU prompt path (expert_split, #32): where the time goes, and what is left

2026-09-29.

- **Branch:** `xeno/claude-merge-0.1.20`, commits `31943cf` (S3), `7db2963` (S4a), `b9d2f57` (S4b).
- **Opt-in:** `STRATA_PREFILL_EXPERT_SPLIT=1`.

Every number below names its run. MEASURED means a run on this machine in this session. UNMEASURED means an
estimate or an idea, and each is labelled as such.

## 1. The machine and its links (MEASURED unless marked)

| part | value | source |
|---|---|---|
| primary GPU | RTX 5060 Ti 16 GB, sm_120, **PCIe x4**: 6.9 GB/s H2D; D2H ~5.4 GB/s (1.38 MB copies) | `xeno_copy_overlap_probe`; resident-copy lane below |
| secondary GPU | RTX 4070 SUPER 12 GB, sm_89, **x16**: 22.7 GB/s alone, 19.1 GB/s during a VRAM-bound kernel. It is also the display card. | `xeno_copy_overlap_probe` (#32) |
| peer-to-peer | none. `cudaMemcpyPeerAsync` is staged through the host and blocked the launching thread | #32 |
| host → pinned memcpy | ~9 GB/s per thread (1.38 MB expert in 0.13-0.15 ms) | stager spans, tl runs |
| NVMe | 4.8-5.1 GB/s in 8 MB requests; one 1.4 MB aligned read ≈ 0.9-1.0 ms | #34, stager spans |
| driver model | WDDM (Windows) | |

## 2. The configuration measured

- **Dual config:** `--prefill auto` (8,192-token chunks), `--spec 3`, `--kv int8`, `--max-context 16384`, `--pcie-frac 0`, `--expert-cache 8000`, `--vram-reserve-mib 2400`, `--secondary-free-floor-mib 640`, `--pool-priority 2 --process-priority 2`, `CUDA_VISIBLE_DEVICES=1,0`. The full command is in §9.
- **Prompt:** 8,023 tokens (`prompt8k.ids`), greedy.
- **Tiers:**
  - The 5060 caches ~6,240 experts; 2,245 of its slots are lent to the prompt path's buffers during a prompt.
  - The 4070 tier holds 6,407 experts (8,704 MiB) in today's config, or 5,148 routed per prompt at 6,912 MiB.
  - The rest is in host RAM.
- **Model:** 48 layers, each with an attention part (GDN or QSA) and an MoE part. MoE: 512 experts, top-10, n_embd 2,560, n_ff 640, Q2_0 blobs of ~1.38 MB.

## 3. Result (MEASURED, same-session ABCCBA, identical greedy output `616dae2a` in all six runs)

Binary `fin.exe` (sha256 `f02042c54be84b7…`), 128 generated tokens:

| arm | 4070 tier | prompt path | decode |
|---|---|---|---|
| A: today, split off | 8,704 MiB | 11.07 / 11.10 s | 62.8 / 61.9 tok/s |
| **B: split on** | 6,912 MiB | **9.41 / 9.54 s (−14.5 %)** | 61.1 / 60.0 tok/s |
| C: control, split off | 6,912 MiB | 11.02 / 11.15 s | 56.6 / 55.6 tok/s |

- **The gain is the split's, not the tier's:** C, with the smaller tier, is as slow as A.
- **Decode:** the differences are inside the known cross-boot spread (`CORRECTIONS` §23), so no decode verdict.
- **4070 memory:** B needs 1.7-1.8 GiB on the 4070. Here it is taken from the tier, so the display card still has
  more free VRAM than in A (2.34 vs 1.72 GiB).

For reference, the single-GPU serving config (deployed today) reads the same prompt in 10.5-10.9 s. That is a
different session, so not a paired comparison.

## 4. What each design does per MoE layer

### A: today, whole_5060

- The 5060 computes every routed expert.
- Non-resident experts are streamed to it over **x4**:
  - host pageable 16.2 GB, through 4 stager threads into pinned buffers;
  - the 4070's own experts 9.1 GB, peer-copied through the host;
  - its lent tail 3.1 GB, read from the #34 tail file.
- 5060 compute lane (MEASURED, `tlA`): `gemm gate/up` 4,676 ms (41 %), `dequant` 646, `gemm down` 407,
  `combine` 505. The MoE products wait on the x4 copies.

### B: split, the simulator's whole_4070

This policy wins every layer of the real dual 8K trace: 2.2 s lower bound, against 4.0 s today and 3.3 s for
expert_split. See `prefill_route_sim.py` on `routedual8k.bin`.

On the 5060:

- It runs the trunk (hyper-connection, GDN/QSA, router, shared expert) and quantizes the chunk's activations
  **once per token** (q8_1, 2,880 B per token, 23.6 MB at 8K).
- Over x4 D2H (the `gpu0 relay` stream) it sends down the activations first, then the shared-expert output
  (T × N f32, 84 MB) and the gates.
- Its own resident experts go down on a dedicated 5060-bound thread (`gpu0 resident copies`), 86 per layer
  (~119 MB), into pinned slots.

On the 4070:

- It takes its own experts in place.
- It receives host experts over **x16**: one stream plan per chunk, a 4070 stager (4 threads, 16 pinned
  buffers), an issuer thread and a 96-slot ring.
- It receives the 5060's experts from the pinned slots, on a second copy stream.
- It gathers each MMQ group of 16, runs `mmq::expert_rows` (gather q8 rows, gate/up MMQ, swiglu, q8, down MMQ)
  in sub-products of ≤ 4,096 rows, and runs `moe_combine`.
- The output (T × N f32, 84 MB) goes back: 4070 D2H, then 5060 H2D over **x4**.

**Exactness** (MEASURED, each has a test):

- MMQ with stream-k off is byte-identical on sm_89 and sm_120 (`xeno_mmq_cross_arch`).
- A row's value does not depend on the batch composition (#32, condition 1).
- `moe_combine` is byte-identical across the cards (`xeno_combine_cross_arch`).
- One whole MoE layer is byte-identical: 0 of 13.1M values differ (`xeno_moe_layer_cross_arch`).
- Chunk size does not change the output (2,048 vs 8,192 identical, #34).

## 5. Where B's time goes (MEASURED, `tlB`: STRATA_TIMELINE, tier 6,912, final build)

`python tests/xeno/perf/timeline.py tlB.json --layer 20`. The timeline itself costs ~6 %: 9.74 s here against
9.4-9.5 s untimed.

### Lanes over the whole prompt

| lane | phase (ms) |
|---|---|
| gpu0 compute | gather* 2,621 · combine* 1,722 · gdn 1,627 · qsa attn 1,528 · hc read 792 · embed+steps 547 · qsa proj 302 · host grouping 215 · router+shared 157 |
| gpu1 compute (split) | inputs up 748 · products 744 · wait copies 709 · gather 470 · wait inputs 298 · output down 198 · combine 123 |
| gpu1 copy engine (split) | copy staged 1,623 (host experts) · copy resident 419 (the 5060's experts) |
| gpu0 relay (split) | shared down 561 |
| gpu0 resident copies | resident down 1,539 |
| split issuer (thread) | outside any span 57.8 % (mostly its ring-slot wait loop, which has no span yet) · stager wait 33.0 % · issuing 9.2 % |
| 4 stager threads | wait buffer 86.7 % · NVMe read 7.6 % · memcpy 5.6 % |

\* In split mode the 5060's `gather` and `combine` phases are **idle**: `gather` runs from the moment the host
hands the layer to the 4070 until it returns, and `combine` is the wait for the 4070's output. **The 5060 waits
about 4.3 s and the 4070 waits for the 5060's trunk: the cards alternate instead of overlapping.**

### One layer (layer 20, a GDN layer; times in ms from the layer's start)

| step | lane | from → to | note |
|---|---|---|---|
| trunk | gpu0 hc read + gdn | 23 → 72 | ~56 ms of work |
| router + shared | gpu0 | 79 → 82 | |
| host grouping | main | 82 → 144 | includes issuing the 4070's layer; the 5060 waits |
| activations down | gpu0 relay | 89 | 23.6 MB, too short to resolve |
| **inputs up** | gpu1 | 89 → 104.5 | **15.6 ms** for ~24 MB + 0.7 MB of tables (1.3 ms at x16 speed): not explained yet, §7.2 |
| experts | gpu1 | 104.5 → 145.4 | wait copies 18.2, gather 8.7, products 14.0 |
| copies of this layer | gpu1 copy | 0.2 → 145 | busy only 47.5 ms over a 145 ms window: paced by the ring and the stager |
| combine + output down | gpu1 | 145.4 → 151.6 | 2.2 + 4.0 |
| output up | gpu0 relay → gpu0 | 151.6 → ~168 | 84 MB over x4 ≈ 12 ms, plus hand-off |

**Critical path per MoE layer after the trunk: ~85 ms.** It is made of host grouping and issuing (~7 ms before
the 4070 starts), inputs up (15.6), experts gated by copies (41), combine and output down (6.2), and output up
(~17). The 4070's MMQ work itself is ~23 ms: gather 8.7 and products 14.0.

### The same layer with a whole-layer ring (MEASURED, `tl9`: tier 7,168, ring 512, 128 pinned resident slots)

- **Copies leave the critical path.** The layer's 469 copies finished before its inputs arrived (`wait copies`
  0.1 ms), and the 4070's MoE took 25.4 ms.
- **The end-to-end time did not improve** (8.85-9.1 s over ring 96-512). The host-thread spans grew: `host
  grouping` 14.5 ms on the gpu0 lane, against 3.6 ms with ring 96.
- **Hypothesis (UNMEASURED):** the helper threads' spin-yield loops (issuer, resident thread, 4 stagers, host
  `wait_issued`) run far ahead with the bigger ring, compete for CPU with the host thread, and the host thread is
  on the critical path.

## 6. Tried, measured, not kept (all in `b9d2f57`'s message)

| lever | result | source |
|---|---|---|
| S4a: the 4070 computes only its own experts; f32 rows back over x4, combine on the 5060 | −2.6 % (10.98/11.04 → 10.72/10.72 s): at 8K the rows relayed (~80 rows × 10 KB per expert) cost about what the weight copies did | `7db2963` |
| per-layer 4070 stager (routed experts only, started after routing) | host reads on the critical path; replaced by a chunk-level plan | timeline `tl_split` |
| 4070 stager threads 4 / 8 / 12 / 16 | no difference (8.77-8.84 s, and 9.01-9.12 s later) | `st_*`, `sp_*` |
| 4070 ring 96 / 192 / 320 / 480 / 512 slots | no difference (8.85-9.14 s); see §5 | `rg_*`, `rb_*` |
| stager pinned buffers 16 → 64 | ~1 %, within noise | `kr_*` |
| forced WDDM submit (`cudaStreamQuery`) after each copy | no difference | `fl_*` |
| 5060-resident experts issued from the 4070 issuer (device switch per expert) | **1.5 ms per expert** on WDDM (6 s of issuer time); moved to a 5060-bound thread | `tl6` |
| separate 4070 stream for the 5060's experts; shared output on its own stream | no measurable change alone | `ov_*`, `cr_*` |

## 7. What is left (every item UNMEASURED unless it says otherwise)

### 7.1 Overlap the cards: wavefront over chunks (the big one)

- **Why:** the two GPUs alternate. The per-layer trunk on the 5060 (~56 ms for GDN, ~125 ms for QSA) and the MoE
  on the 4070 (~85 ms including relays) run one after the other.
- **How:** chunking is exact (2,048 vs 8,192 identical). So a two-stage pipeline over chunks keeps the output
  unchanged:
  - the 5060 runs the trunk of chunk c+1 at layer l while the 4070 runs the MoE of chunk c at layer l;
  - chunk c+1's attention at layer l needs only chunk c's layer-l K/V, which chunk c's trunk wrote before its MoE.
- **Ceiling:** ≈ max(trunk, MoE) per layer instead of their sum. Total trunk ≈ gdn 1.6 + qsa ~2.0 + hc 0.8 +
  embed 0.5 ≈ 5-5.5 s. MoE plus relays ≈ 4.3 s of the 5060's idle. Lower bound ~5.5-6 s against 9.5 s today.
- **Costs:**
  - smaller chunks mean smaller MMQ batches;
  - relays and launches per chunk;
  - a second set of 5060 trunk buffers;
  - the 4070's buffers per chunk in flight.
- #32 already lists "wavefront microbatches" as its last step, with a byte-identity probe first.

### 7.2 The 4070's input upload: 15.6 ms per layer for ~25 MB

- **Expected:** ~1.3 ms at x16.
- **Seen:** it was 15 ms even when no expert copy competed (`tl9`).
- **Candidates:**
  - the pinned `hx` buffer (`cudaHostAllocPortable`, allocated on the 5060) is being staged by the driver for the
    4070;
  - H2D engine queueing;
  - host lag between the two marks.
- **Next measurement:** a GpuClock span around just the `xtok` copy. Then try allocating `hx` while the 4070 is
  current, or `cudaHostRegister` with `Portable`.
- **Size:** up to ~0.7 s if it drops to ~1.5 ms × 48 layers.

### 7.3 Copies still gate the MoE with ring 96 (18 ms of `wait copies` per layer)

- A ring that holds a whole layer (~470 slots, ~650 MB) removes it (`tl9`), but only helps if the host thread
  stops slowing down (7.4).
- VRAM for it must come from the tier (7.6).

### 7.4 Host-thread latency on the critical path

- **Evidence:** `host grouping` 3.6-14.5 ms per layer. It covers router sync, CPU grouping of 82K (token, k) ids,
  and ~400 CUDA calls per layer for the 4070 (per group: gather, memsets, events, marks; per sub-product: 5
  launches).
- **Ideas:**
  - replace the spin-yield waits with blocking waits (condition variables) so helper threads do not compete for
    CPU with the host thread;
  - group routes on the device (#31: router sync);
  - capture the 4070's per-layer launches as a CUDA graph;
  - move the 4070's issuing to its own host thread.

### 7.5 The 5060 still lends 2.9 GB for buffers split mode does not use

> **CORRECTED 2026-09-30 (#38): this section is wrong.** `STRATA_PREFILL_BUFFERS=1` itemises the borrowed bytes
> (`bb*.stderr`, exe built from the commit that added it). At 8,192 tokens they are: trunk 1,551 MiB, shared set
> 1,126-1,153 MiB, rest 131-247 MiB. The shared set is the largest of the GDN, QSA and MoE scratch.
> - Without split, the MoE set is the largest, but only just: 1,153 against GDN's 1,126 MiB.
> - With the split layout (D6) it falls to 415 MiB and hides under GDN.
> - A wave lane (4,096 tokens): trunk 824, GDN 563 (MoE 331, hidden), rest 91 MiB.
>
> So the 2.9 GB is the trunk and the GDN scratch, which the split path needs too. MoE-only buffers left to reclaim:
> under about 60 MiB (ring, MMQ group buffers, DQ). The text below is kept as written.

- In split mode the 5060 runs no routed expert, yet `--prefill auto` still borrows 2,245 cache slots for its MoE
  buffers: `Dm` (840 MB), GU, the 96-slot ring and so on.
- **Cost of that:** those experts are read from the #34 tail file during the prompt (2,245 NVMe reads, ~2.2 s of
  stager thread time) and refilled after it (0.62-0.66 s per prompt, not inside the prompt-path time above).
- **Idea:** a split-aware layout that borrows only the trunk's buffers. The experts then stay resident and come
  down through the resident thread, and the refill goes away.
- **Size:** TTFT −0.6 s from the refill, plus fewer disk reads.

### 7.6 Where the 4070's 1.7-1.8 GiB comes from

- `Dm` (all T·K rows, f32, 839 MB at 8K)
- `shared` and `bo` (84 MB each)
- activations (24 MB)
- ring (96 × 1.38 MB = 133 MB)
- group buffers (~30 MB)
- MMQ scratch (~50 MB)
- CUDA and stager overhead

Options:

- **Today (measured):** shrink the tier to 6,912 MiB.
- **S4c:** borrow the tier's own slots during a prompt, as the 5060 lends its tail, and refill them from a 4070
  tail file. About 1.8 GB per prompt at ~5 GB/s ≈ 0.4 s, which eats part of the gain. The tier is exclusive (no
  host copies), so it needs the tail file, and 4070 swaps must not touch the borrowed range.
- **A smaller `Dm`:** it is T·K·N f32 because combine needs all of a token's rows. Tiling the chunk by tokens for
  the MoE is exact (MoE is per token), but it re-streams every host expert per tile over x16: 2 tiles ≈ +0.9 s of
  x16.

### 7.7 The output relay: 84 MB over x4 per layer (~12-17 ms, ~0.7 s per prompt)

- It is inherent to combining on the 4070: the combined output is f32.
- Wavefront (7.1) hides it behind the next chunk's trunk.
- Combining on the 5060 would need every expert row over x4: 10× more.

## 8. Constraints any change must keep

- **Byte-identical greedy output.**
  - MMQ with stream-k off (`nsm = 1`).
  - `moe_combine`'s per-token fmaf order k = 0..9.
  - Anything that changes which rows share a floating-point GEMM (trunk microbatching) needs its own byte-identity
    probe.
- **The 4070 is the display card:** 2.5 GB of headroom including desktop use. Its VRAM comes from the tier, never
  from the headroom.
- **Host RAM:** Strata runs stay near 31 GB; the dual config peaks around 31 GB and commits ~37 GiB.
- **Measurement:** every speed claim from a same-session ABBA with identical output, `--pool-priority 2` and HIGH
  class. Cross-boot differences below ~13.6 % are noise.

## 9. Reproduce

```sh
# build: cmake --build D:\Github\Strata\.worktrees\merge-build --target strata
M='C:\AI\models\ISTA-DASLab-Qwen3.8-Flash-Next-GSQ-RCO\Q2_0'
CUDA_VISIBLE_DEVICES=1,0 STRATA_PREFILL_TIMING=1 STRATA_PREFILL_EXPERT_SPLIT=1 STRATA_TIMELINE=tl.json strata.exe \
  --pack 'D:\Github\Strata\packs\q2_0' --native "$M\Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf" \
  --ple-gguf "$M\Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf" --prefill auto --spec 3 --spec-min-p 0.5 \
  --mtp 'D:\Github\Strata\mtp\rt' --kv int8 --tokens-file prompt8k.ids --max-new 128 --greedy --max-context 16384 \
  --pcie-frac 0 --expert-cache 8000 --expert-profile <full profile .bin> --vram-reserve-mib 2400 \
  --secondary-free-floor-mib 640 --secondary-expert-mib 6912 --pool-priority 2 --process-priority 2
python tests/xeno/perf/timeline.py tl.json --layer 20      # every lane by phase, one layer across the lanes
python tests/xeno/perf/prefill_route_sim.py route.bin       # policies from STRATA_PREFILL_ROUTE_TRACE=route.bin
```

Drop `STRATA_PREFILL_EXPERT_SPLIT` for arm A and C. Arm A uses `--secondary-expert-mib 8704`.
