# Porting the fork's features to upstream's Qwen3.6 path (q36), 2026-11-04 (#298)

The developer decided to run Qwen3.6-35B-A3B NVFP4 on strata-xeno with W4A8 on the RTX 5060 Ti and the RTX 4070 SUPER (option B), and to port every engine and server feature the fork built and upstream lacks.

The inventory below comes from reading the code only, without running anything. It covers `docs/BLUEPRINT.md`, the README's "What it adds over upstream", grep of the fork at `3aed1e9b`, and `git show v0.1.45:` of `src/qwen35/q36_engine.cu`, `q36_main.cpp`, `q36_kernels.cuh` and `docs/QWEN36.md`. The q36 line numbers are at `v0.1.45`; a `~` means approximate. Every port size is an estimate.

## Facts that set the order

1. **Two cards only just hold every expert, and only at short context** (an estimate).
   - The UD-Q4_K_XL file is 22.85 GB, so the routed experts are about 20 GB. The NVIDIA NVFP4 file is 21.8 GB.
   - The 5060 Ti has about 11-12 GB free after the dense Q8_0 weights, the reserve and the KV cache. The 4070 has about 9 GB after the display's 2.5 GB.
   - At 32K context (1.5 GB of KV per session, fp32 K/V) the two cards just fit. At 262K the KV alone is about 11.8 GB (45 KB/token), so they do not. The CPU pool therefore stays: the 4070 tier shrinks the CPU share rather than removing it.
2. **q36 reuses a held conversation only when it is a strict prefix of the new prompt** (`match_len` `:3401-3410`), and it has no checkpoints. My reading, not run: a request that diverges reads the whole prompt again. Examples are a side request, a cut drafted tail after STOP, or a template that drops earlier thinking.
3. **q36's hybrid mode turns the decode graph off** (`if (E.main_hybrid) E.use_graph = false;` ~`:2887`). On upstream's 3060, a 17 ms token was 6 ms of GPU and 9 ms of waiting for the CPU pool.
4. **q36 skips what it does not know, without an error.** An unknown flag in `--serve` prints "ignoring" (~`:3779`). An unknown GEN `k=v` is dropped silently (~`:3449-3475`). This causes the silent failures listed further down.

## Inventory

A port size of S, M or L means small, medium or large. "Reject" means the feature is tied to the Flash-Next pack or shape, or q36 does not need it.

| # | Feature | In q36 v0.1.45? | Port |
|---|---|---|---|
| A1 | 4070 SUPER expert tier (secondary runner, per-layer graph, dispatch kind 2); `SecondaryRunner` takes any widths | no, one GPU plus the CPU pool only | L: a third destination in `moe_hybrid_cpu` / `k_merge_cpu`, a graph on device 1 |
| A2 | The relay between the cards (packed metadata, owned-rows D2H) | no | part of A1 |
| A3 | Display reserve, breach policy (`--display-reserve-mib`, `--secondary-breach`) | no | S after A1 |
| A4 | Dynamic experts: an owned expert has no RAM copy; placement first | no: every hybrid-layer expert also stays in host memory | M (needs a paired swap back) |
| A5 | Adaptive swaps on an adapt thread, gain threshold, paired 4070 swaps (#229: 60.1 → 65.1 tok/s) | partly: `rebalance(8)` runs synchronously every 32 positions, decay 0.5 | M, after A4 |
| A6 | Boot placement from a ranked profile | no: the first S experts of each layer | S |
| A7 | Capacity mode: RAM cache and NVMe tier | no: needs every expert in host memory | L, last (capacity mode is optional) |
| A8 | Per-tier dispatch counters | partly (`hy_gpu` / `hy_cpu` at exit) | S |
| B1 | Prompt expert split and two-lane wave | no | L: q36 has its own prompt driver; to be redesigned for q36, not copied |
| B2 | Borrowing tail file | not applicable | reject |
| B3 | KV streaming | upstream's own feature, not the fork's; q36 keeps fp32 K/V in VRAM | not a fork port (fp32 KV is what keeps q36 off 262K) |
| B4 | IQ1_M MMQ tile, bf16/q8 split sum, `cuda0_owns` | not applicable | reject (pack-specific) |
| C1-C3 | `--pool-priority`, `--process-priority`, `--pool-rest` | ignored | S |
| C4 | One-phase native pool (#147) | comes with the fork's `pool.cpp` | S (check `ExpertJobMulti`) |
| C5 | No park spin in capacity mode | default | S, with A7 |
| C6 | Q2_0 VNNI, TQ trellis kernels, Hadamard | not applicable (K-quant / NVFP4 experts) | reject |
| D1 | Thai tokens in the draft vocabulary | q36 uses the full head; upstream dropped `--draft-vocab` | reject |
| D2 | Ban mask (`--ban-ids`, GEN `ban=1`) | ignored without a word | M: mask in argmax, sampler and `spec_step` |
| D3 | MTP catch-up, batched ring K/V | not applicable | reject |
| D4 | One-token windows while MIP delivers | no | part of G1 |
| E1 | Checkpoints + `ckpt_at` + the server's `CacheSlots` family picker | no checkpoints at all | L: GDN conv/state snapshot and restore, checkpoints at root and turn, then `ckpt_at` |
| E2 | Asynchronous slot park (#145) | no | M, after E1 |
| E3 | `checkpoint_save` retries `bad_alloc` | not applicable | S, after E1 |
| F1 | Pipeline timeline (`STRATA_TIMELINE`) | no spans; the family switch runs before the arming | M: move the arming, q36 spans, analyzer names and tests |
| F2 | `STATS` line before DONE | no | S |
| F3 | Latency counters | one line per request | S-M |
| F4 | `INFO cpu_isa / gpu_arch` | no | S |
| F5 | Routing trace format 2 | no | S-M (feeds A6) |
| F6-F7 | `end_engine` deadline, crash report | before the family switch? | S |
| F8 | LAZY module loading | q36 forces EAGER on Windows | S, decided by a measurement |
| G1-G3 | MIP text level: inline value, loop close, sandbox bridge (SBX/SBXW/SBXR) | no | M-L: pull `InlineTurn` and the inline/sandbox logic out of `IgcLearnedDiagnostic` into the shared MIP module (#298 point 3); token text from the GGUF |
| G4 | Learned head / diagnostic controller | no; tied to the Flash-Next shape, weights on Gyro's activations | reject for now: needs retraining. Layer 1's probe is calibrated per model instead |
| G5 | Sidecar (numbers in words) | no | M, after G1 |
| G6 | MIP in batch slots (#284) | no | M, after G1 |
| G7 | Qwenputer circuit | not wired into inference | reject |
| G8 | MIP inject log, HIDPOS | no | S, with G1 |
| H1 | Qwen3.5-4B dense (#297) | no | reject the code; a dense FFN path in q36 is a separate feature |
| I1-I5, I8 | Server: telemetry, web app, runs, MCP hub and coding tools, sandbox tool route, loop guard, think budget | protocol-level: these work through the engine protocol | none (I2 needs `--model` in `gguf_info`: S) |
| I6-I7 | Reasoning tool, MIP adapter | need G1 + G3 | after G1 |
| I8 | Continuation after STOP | possibly a full re-read, since q36 can hold accepted drafts that were never emitted | to verify with E1 |
| I9-I12 | CacheSlots, CJK guard, degraded restart, prefix cache | see E1, D2 and A1 | — |

## Silent failures to fix first

These come from reading the code, not from a run.

1. **CJK guard.** The server passes `--ban-ids` and sends `ban=1`, and q36 drops both. The server believes the ban is on.
2. **Reasoning tool.** The server's check finds `--igc-sandbox-tool` in the `strata` binary, adds the `--igc-*` flags and sends `igc=`. q36 ignores all of it, so a `run_python` call written in the thinking is never answered.
3. **No `STATS` and no `INFO cpu_isa / gpu_arch`.** The web app shows "not measured".

## Protocol keys and lines q36 lacks

- **GEN keys:** `ban`, `ckpt_at`, `igc`, plus upstream's own `cache_slot`, `logit_bias`, `cvec`, `ckpt`, `pin`, `pcie_frac`, `spec_min_p`, `pipe` and `aux_cpus`.
- **Reply lines:** `STATS`, `INFO cpu_isa/gpu_arch`, `INFO ban=`, `SBX` / `SBXW` (reading `SBXR`), `RESUME`, `REUSED`, `INFO cache_slots`, `INFO logit_bias`, `YIELDED`.
- **Commands:** `VRAM`, `SAVE` / `RESTORE`, `QUIT`.

## Order proposed for option B

1. The silent failures and the protocol lines: `STATS`, `INFO`, `RESUME`, `ban`.
2. NVFP4 native: read the routed experts' `.scale` into the tail, NVFP4 in `native_mmvq`, and gate against llama.cpp.
3. The 4070 tier (A1 + A2 + A3), the reason for option B. Then the rough speed comparison against llama.cpp on two cards.
4. Pool and process priority, and #147 (C1-C4).
5. The timeline and counters (F1-F3).
6. Checkpoints and prefix reuse (E1), for Claude Code's request shape.
7. MIP as the shared module, text level (G1-G3 and G8), with I6-I7 behind it.
8. Dynamic host release, adaptive swaps, the profile (A4-A6).
9. NVMe and capacity (A7), last.

Rejected (Flash-Next pack or shape, or not needed by q36): B2, B4, C6, D1, D3, G4 for now, G7, and H1's code.
