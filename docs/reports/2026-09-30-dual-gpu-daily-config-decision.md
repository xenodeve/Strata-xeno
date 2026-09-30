# The daily server on one card or two, and the dual-GPU prompt path (split + wave): a decision brief

**Date:** 2026-09-30. **Issue:** #35 (parent #32, #12). **Status:** the developer is researching the decision; nothing
here is a default yet. **Branch:** `xeno/claude-merge-0.1.20`, up to `db17e31`.

Every number carries its source. **MEASURED** means it was measured on this machine in a same-session ABBA with
identical output, unless the line says otherwise. **UNMEASURED** marks an estimate or an open question. A number with
no tag should be treated as unsupported.

---

## 0. The question

The daily server (`D:\Github\Strata\strata-xeno.json`, engine `engine-xeno\strata.exe` from `48e44be`) runs on **one
card**: the RTX 5060 Ti. It has no expert tier on the RTX 4070 SUPER. The dual-GPU prompt path in #32/#35 (**split** and
**wave**) only exists in the **dual config**, where the 4070 holds an expert tier. So the decision has two parts:

1. Should the daily server move to the dual config?
2. If it does, should it read prompts with split + wave? That shrinks the 4070's decode expert tier from 8,704 MiB to
   6,400-6,912 MiB.

## 1. Options at a glance

| option | prompt 8K (dual) | 4070 decode tier | new code in the daily path | what is still unmeasured |
|---|---|---|---|---|
| **A. stay on one card** (today) | 10.5-10.9 s (one card, other session) | none | none | - |
| **B. dual, no split** | 11.07 / 11.10 s | 8,704 MiB | the dual config only | decode against A |
| **C. dual + split + wave** | **7.6-7.9 s** | 6,400 MiB | split, wave (opt-in env today) | decode at 6,400 against 8,704 |
| **D. dual + split + "S4c"** | ~7.6-7.9 s + ~0.4 s refill (estimate) | 8,704 MiB between prompts | S4c is not built | everything |

Sources:
- **A:** `docs/reports/2026-09-29-dual-gpu-prefill.md` §3 (a different session, so not paired).
- **B:** the same report §3, arm A.
- **C:** #35's D7 comment and this session's ABBA (§3 below).
- **D:** the same report §7.6 (UNMEASURED).

## 2. What split and wave do

**Hardware:**
- 5060 Ti: sm_120, PCIe 4.0 **x4** through the chipset. H2D is about 6.9 GB/s, and it is the same under decode (MEASURED, #22).
- 4070 SUPER: sm_89, PCIe 4.0 **x16**. H2D is about 24 GB/s. It is also the display card.
- There is no P2P between the cards.

**The model:** 48 layers of attention (GDN or QSA) plus MoE: 512 experts, top-10, Q2_0 blobs of about 1.38 MB. An 8K prompt
chunk routes nearly every expert of every layer, so the prompt path streams almost the whole expert set once per chunk.

**Today's prompt path (one card, and dual without split):** the 5060 runs everything. Each expert it does not hold
crosses its x4 link. The MoE phases are link-bound.

**Split** (`STRATA_PREFILL_EXPERT_SPLIT=1`, #32 S4 + #35 D1-D6):
- For each MoE layer of a chunk of 2,048 tokens or more, the 5060 sends only the chunk's q8 activations and the routing
  (about 25 MB) to the 4070.
- The 4070 gathers every routed expert over its own x16 link. The experts come from its tier in place, from the 5060's
  cache, or from host RAM through a pinned stager. It runs the expert products (`mmq::expert_rows`, byte-identical
  across sm_89 and sm_120) and returns the routed sum.
- The 5060 keeps the trunk and the shared expert and finishes the layer. The combine split is exact (D1).

**Wave** (`STRATA_PREFILL_WAVE=1`, D7):
- Two prompt lanes read alternate chunks at half the chunk size. Chunk c's layer l starts once chunk c-1 has handed
  that layer's MoE to the 4070.
- The 5060's trunk of one chunk therefore overlaps the 4070's MoE of the other.
- The two lanes share one expert stream, so a layer's experts cross the x16 once per pair of chunks.

**Exactness:** the output is byte-identical to the one-card path in every measured run (greedy token hashes below).
The parity tests `xeno_moe_layer_cross_arch`, `xeno_combine_split_parity` and `xeno_combine_cross_arch` report 0 differing
values.

## 3. What it gains (MEASURED)

| run | without | with | source |
|---|---|---|---|
| 8K generate, dual, ABCCBA: today vs split | 11.07 / 11.10 s | 9.41 / 9.54 s (split) | report §3 (`616dae2a` x6) |
| 8K generate, split vs split + wave | 8.61 / 8.59 s | 7.86 / 7.73 s (−9.4 %) | #35 D7 comment |
| 15K generate, split vs split + wave | 15.68 s | 14.04 s (−10.5 %) | #35 D7 comment |
| 8K served, split vs split + wave (tier 6,400) | 8.81 s | 7.90 s | #35 serve comment |
| 3,000-token served request after an 8K one | 3.80 s | 3.71 s | #35 serve comment |
| 8K generate, split + wave, 2026-09-29 ABBA (d7x / simp) | - | 7.87 / 7.86 / 7.84 / 7.59 s | `e10e27f` message (`02996115` x4) |

**Where the gain applies:**
- It applies to prompt **reading** only: chunks of 2,048 tokens or more. Shorter prompts run the one-card path
  unchanged. At about 3,000 tokens the gain is within noise.
- The primary client is Claude Code. Serve reads a prompt in parts at turn boundaries, and the prompt cache reuses
  earlier checkpoints. So the tokens actually read per request are often the new turn, not the whole prompt.
  **UNMEASURED:** the distribution of batched-read sizes in real Claude Code sessions. It decides how much of the 8K
  gain a day of use sees.

## 4. What it costs

### 4.1 The 4070's VRAM, taken from the decode tier

The 4070 is the display card. The rule is 2.5 GB of headroom including desktop use, so the split's buffers come out
of the expert tier, never out of the headroom. Split needs about 1.7-1.8 GiB (report §7.6):

| buffer | size |
|---|---|
| `Dm` (all T·K rows, f32, 8K) | 839 MB |
| `shared` and `bo` | 84 MB each |
| activations | 24 MB |
| ring (96 × 1.38 MB) | 133 MB |
| group buffers | about 30 MB |
| MMQ scratch | about 50 MB |
| CUDA and stager overhead | the rest |

Wave adds the second lane's buffers (half-size chunks) and a 512-slot shared ring (about +0.5 GB). With the wave the 4070
holds about 2.0 GB and keeps at least 2.5 GB free at tier 6,400 (#35).

| config | 4070 tier (MiB) |
|---|---|
| dual, no split (today's dual) | 8,704 |
| split | 6,912 |
| split + wave | 6,400 |

### 4.2 Decode: the unknown that decides it

The tier holds the experts that decode computes on the 4070. Shrinking it moves some experts to the 5060's cache or
to the CPU pool. What that does to decode is **not settled**:

| same session, ABCCBA (report §3) | tier | decode tok/s |
|---|---|---|
| A: split off | 8,704 | 62.8 / 61.9 |
| B: split on | 6,912 | 61.1 / 60.0 |
| C: split off | 6,912 | 56.6 / 55.6 |

- B and C have the same tier and should decode alike, yet they differ by about 8 %. The spread inside this experiment is
  as large as the effect being looked for.
- Tier 6,400 (the wave's) has **no** paired decode measurement.
- **The measurement that settles it:** a same-session ABBA of decode only (dual config, tier 8,704 vs 6,400, split +
  wave off in both so only the tier differs), on the thai and code prompts that serve as the decode benchmarks. Report
  tok/s, 4070 tier hits and CPU pool ms/round. About 20 minutes. Not run yet.

### 4.3 One card vs two for decode (option A vs B)

- **UNMEASURED as a pair.** The best measured dual decode is code 89 / thai 54 tok/s: 13 workers, pool-rest, paired 8
  + 4070 16 swaps, HIGH class (#12, #16). No same-session one-card vs dual decode comparison on the current engine
  exists.
- **RAM:** the dual config commits about 37 GiB and its working set peaks around 31 GB. The one-card daily config commits 37.9 GiB
  with the tail file (#34). Both are near the 31 GB working-set cap that this machine (48 GB) tolerates.

### 4.4 Code maturity

- Split and wave are opt-in env vars, not flags.
- They have had one code review (`9bd207f`: 7 faults fixed), `/simplify` (`e10e27f`) and `/scrutinize`. `/scrutinize`
  found a real hang, fixed in `e34a1d0`: a served prompt read in turn-split parts hung when one lane ran alone. The
  regression check is `tests/xeno/perf/serve_wave_parts.py`.
- They have not served real Claude Code sessions for any length of time.
- Known nit, not fixed: if wave lane 2 fails before lane 1 has published its stream, that run's issuer is joined only
  at the next split plan or at teardown. The output is not affected.

## 5. The options in detail

**A. Stay on one card.**
- For: no change and no new code in the daily path. The 4070 stays fully free for the desktop.
- Against: 8K prompts take about 10.5-10.9 s, and the 4070's x16 link is unused.

**B. Dual config, no split.**
- For: the 4070 holds an 8,704 MiB decode tier. This is the config the decode work (#12) was measured in.
- Against: prompts are no faster than one card (11.07 s dual vs 10.5-10.9 s one card, different sessions). Decode
  against A is unmeasured.

**C. Dual + split + wave.**
- For: the fastest prompt path, −29 % at 8K against B (7.9 vs 11.1 s). The output is identical.
- Against: the tier is 6,400 MiB, so decode is at risk by an unmeasured amount (§4.2). The code is new (§4.4).

**D. Dual + split, tier borrowed during the prompt ("S4c", not built).**
- The design: while a prompt is read, the split borrows the tier's own slots the way the 5060 already lends its cache
  tail. Afterwards it refills them from a 4070-side tail file (about 1.8 GB at about 5 GB/s ≈ 0.4 s).
- For: decode keeps the full 8,704 MiB tier.
- Against:
  - about +0.4 s per prompt (UNMEASURED);
  - 3.5 GB more disk for a second tail file;
  - 4070 swaps must avoid the borrowed range;
  - the work is not started.

## 6. Open research questions

1. **Decode vs 4070 tier size** (§4.2): the measurement is cheap and should come first. If 6,400 loses nothing
   measurable, C is the natural choice. If it loses clearly, D or a smaller split footprint (Q4) is worth building.
2. **What Claude Code actually reads per request:** the size distribution of batched prompt parts with the prompt cache on.
   8K fresh reads (a new chat with a long system prompt and tools) gain the most. Short deltas gain nothing.
3. **One card vs dual decode, paired** (§4.3). It decides between A and B/C/D on decode alone.
4. **Shrinking the split's 4070 footprint without S4c:**
   - `Dm` is T·K·N f32 (839 MB at 8K). Tiling the chunk by tokens is exact but re-streams host experts over x16 (+0.9 s
     per extra tile, report §7.6).
   - Half-size chunks (the wave already halves each lane) might allow a smaller `Dm` per lane. UNMEASURED.
5. ~~**The 5060 still lends 2.9 GB in split mode** for MoE buffers it no longer uses.~~ **CORRECTED 2026-09-30
   (#38):** the 2.9 GB is the trunk (824 MiB per wave lane) and the GDN scratch (563 MiB); the MoE set (331 MiB) hides
   under GDN. Under about 60 MiB is MoE-only. The 0.6 s refill can only shrink with the trunk or GDN scratch, e.g. GDN
   tiled over tokens (UNMEASURED).
6. **Remaining per-layer critical path with the wave** (report §5, #35 D4 comment): trunk 56 ms, host route grouping
   8 ms, activations over x4 5 ms, 4070 MoE 31 ms, routed sum back over x4 15 ms. The wave overlaps trunk and MoE; route
   grouping on the device (about 0.4 s per prompt) is the next listed lever.
7. **Upstream 0.1.21-0.1.22** also works on the prompt path:
   - tensor-core QSA attention (D-1);
   - GDN recurrence split over value columns (D-2);
   - queued-copy refill (D-4);
   - an issuer thread (D-5);
   - C-1..C-4.

   Its own multi-GPU is a layer split whose host arena did not fit this machine's RAM (parked by the developer). A merge
   would need the layer split kept off, so that it does not fight split + wave. Merge size is unknown.

## 7. Constraints any choice must keep

- **Byte-identical output** against the one-card path: MMQ with stream-k off, the k = 0..9 fmaf order in the routed
  sum, and no trunk microbatching without its own identity probe.
- **The 4070 is the display card:** 2.5 GB headroom including desktop use. Its VRAM comes from the tier.
- **Host RAM:** the working set stays near 31 GB. A config that exceeds it is not an option on this machine.
- **Measurement:** same-session ABBA, HIGH class, `--pool-priority 2`. Cross-boot differences below about 13.6 % are noise
  (#18: the same exe gives different tokens across boots; a reboot capture is pending).

## 8. Reproduce

- **Build:** `cmake --build D:\Github\Strata\.worktrees\merge-build --target strata`. It needs the MSVC environment; the
  full configure line is in `docs/reports/2026-09-29-dual-gpu-prefill.md` §9.
- **Dual 8K, split + wave:**
  ```
  CUDA_VISIBLE_DEVICES=1,0 STRATA_PREFILL_EXPERT_SPLIT=1 STRATA_PREFILL_WAVE=1 strata.exe \
    --prefill auto --spec 3 --spec-min-p 0.5 --kv int8 --max-context 16384 --pcie-frac 0 --expert-cache 8000 \
    --vram-reserve-mib 2400 --secondary-free-floor-mib 640 --secondary-expert-mib 6400 \
    --pool-priority 2 --process-priority 2 ...
  ```
  Drop both env vars and use `--secondary-expert-mib 8704` for option B.
- **Served check:** `serve_wave_check.py` (two requests) and `tests/xeno/perf/serve_wave_parts.py` (a turn-split prompt).
- **Timeline of one layer:** `STRATA_TIMELINE=<file>`, then `python tests/xeno/perf/timeline.py <file> --layer 20`.
