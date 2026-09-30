# Dual GPU VRAM saturation experiment — 2026-09-28

Scope: experimental branch `xeno/afk-vram-saturation`, based on pushed Phase 4 static ownership. No default launcher/profile change, PR or merge. The user clarified that **2.5 GB is total headroom for other processes**, not an additional 2.5 GB of free framebuffer after counting their existing use. Tests conservatively use 2560 MiB. For each run, the harness read 4070 SUPER `memory.total - memory.free` before Strata, then set `--secondary-free-floor-mib` to the remaining part of 2560 MiB plus 128 MiB, rounded up to 64 MiB (minimum 256 MiB). This is an experimental fixed floor; changing desktop allocations can still force the monitor to abort. The normal CLI default remains a 2560 MiB **free** floor.

The original `data/expert-profile.bin` ranks only 8,000 of 48×512 pairs. A temporary benchmark profile kept those 8,000 pairs in original order and appended all other pairs in layer/expert order, with header ranked/slots set to 24,576. The added pairs have **no measured frequency ranking**. File: `%TEMP%\strata-benchmark-full-profile.bin`; the user profile was not edited. Filling unused VRAM with these pairs is a capacity test, not an optimal placement policy.

## Capacity and fail path

| Primary request/reserve | Secondary cap | Prefill chunk | Observation |
|---|---:|---:|---|
| 8,000 / 700 MiB | 6 GiB | 2048 | 7,991 primary slots / 10.29 GiB, 6 GiB secondary staged; prefill device buffers did not fit. |
| 8,000 / 700 MiB | 6 GiB | 512 | Prefill passed, but the 43.3 MiB verify arena did not fit on primary. |
| 8,000 / 1600 MiB | 8 GiB | 512 | 7,309 primary slots / 9.41 GiB; 6,213 secondary slots / 8.00 GiB; sky 96/96 parity, 30.21 tok/s. |
| 8,000 / 1600 MiB | 9 GiB | 512 | 7,309 primary, 6,990 secondary / 9.00 GiB; sky 96/96 parity, 31.62 tok/s; sampled 4070 lower-free minimum 0.60 GiB. |
| 8,000 / 1200 MiB | 9.125 GiB | 512 | 7,612 primary / 9.80 GiB, 6,981 secondary / 8.99 GiB; sky 256/256 parity, 33.16 tok/s. Sampled `nvidia-smi` free minimum: 5060 Ti 815 MiB, 4070 SUPER 630 MiB. Secondary CUDA/NVML lower-free minimum 0.74 GiB. |
| 8,000 / 1200 MiB | 9.125 GiB | 512 | Thai trial aborted during secondary fill: CUDA/NVML lower-free reading crossed its run's 576 MiB floor. The 0.5 s `nvidia-smi` sampler saw 645 MiB minimum, so sparse NVML samples did not capture the lower reading. This near-limit setting is not stable. |
| 8,000 / 1200 MiB | 8.5 GiB | 512 | 7,612 primary and 6,602 secondary slots. Thai, code and long 256-token runs completed with parity; details below. |

The primary card's memory must include later head, prefill and verify buffers. “Use full 16 GB” cannot mean allocating all framebuffer to expert cache before decode. The 7,612-slot run sampled primary maximum used at 15,236 MiB of 16,311 MiB, with 815 MiB minimum free. Secondary 9 GiB was possible once, but 8.5 GiB had more room for desktop/CUDA variance. The full profile also lengthened startup: roughly 170–180 s including loading and per-slot byte verification in these runs; decode figures below exclude startup.

## MTP comparison on saturated placement

The existing benchmark config used `--mtp --spec 4 --spec-min-p 0.5`; the user normally uses `n=3`. In Strata, `--spec N` is the maximum verify window including the current token, so `n=4` offers at most three drafts and `n=3` at most two. The `drafts accepted / offered` counter is the direct accept rate. All completed outputs matched the accepted baseline raw IDs for 256/256 tokens. The 8.5 GiB runs use the same executable, 7,612 primary slots, 6,602 secondary slots, six P-core workers, `--prefill 512`, and the same prompt IDs. Each measurement is one separate process; background load was not controlled.

| Prompt | `n=4` accepted | `n=4` rounds | `n=4` decode | `n=3` accepted | `n=3` rounds | `n=3` decode |
|---|---:|---:|---:|---:|---:|---:|
| sky | 152/238 (63.9%) | 104 | 33.16 tok/s | 136/198 (68.7%) | 120 | 34.05 tok/s |
| Thai | 83/171 (48.5%) | 173 | 19.77 tok/s | 71/138 (51.4%) | 185 | 31.10 tok/s |
| code | 181/213 (85.0%) | 75 | 37.02 tok/s | 164/182 (90.1%) | 94 | 37.75 tok/s |
| long | 159/229 (69.4%) | 97 | 37.54 tok/s | 140/188 (74.5%) | 117 | 38.60 tok/s |

**Sky is not a matched placement pair:** its `n=4` run staged about 9 GiB on 4070, while `n=3` staged 8.5 GiB. Thai `n=3` was much faster in this pair despite more verifier rounds, but its CPU pool timing also changed substantially, so this is a signal for repeated A/B trials, not proof of causal speedup from `n=3`. Code differs little. The first priority for a performance verdict is paired alternating `n=3`/`n=4` trials on the same 8.5 GiB placement, with a stable background load.

Artifacts: `%TEMP%\strata-saturate-*` contain per-run command, raw stdout/stderr, result JSON and optional 0.5 s `on.vram.csv`. The benchmark harness now accepts `--profile`, `--primary-reserve-mib`, `--prefill-chunk`, `--spec`, `--secondary-free-floor-mib`, and `--sample-vram`. The secondary free-floor override is passed through arena admission, post-touch/fill checks and the lifetime monitor; default behavior remains unchanged.

Verification after the final code change: full dual-architecture build, 9/9 `xeno_` CTests, 8/8 `gpu_` CTests, 23/23 Python serving tests, Python benchmark syntax check and `git diff --check` passed. A new budget test covers a 512 MiB free floor, and the secondary arena test exercises that floor with injected free readings. No live long-session/serving test was run at this saturation setting.

Open: repeat `n=3` vs `n=4` under paired conditions, including sky at identical 8.5 GiB placement; test practical 2048-token prefill at a smaller primary cache; profile/optimize unranked experts rather than using layer order; test long Claude Code sessions and graceful failure recovery before production use.
