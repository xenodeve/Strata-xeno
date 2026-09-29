# Phase 4 static exclusive primary pages — first slice

Scope: engine #4 / tracker #13, branch `xeno/afk-phase4`. **This is a static ownership and RAM-release experiment, not the complete Phase 4 swap protocol or a production default.**

`--exclusive-primary-experts` opts in and refuses startup unless the model is native Q2_0, spec >=2, the profile/primary cache are present, the CPU pool is enabled, and `--no-prefill-borrow --adapt-swaps 0 --pcie-frac 0` are set. Forced CPU-miss and mmap expert modes are incompatible. The large host arena is pageable in this mode. After the primary cache is profile-filled, each distinct resident pair is read back and byte-verified against its host blob before ownership is published and whole Windows pages strictly inside that blob are decommitted. Duplicate profile pairs are skipped. The arena's virtual address, layout and neighboring blobs stay in place. `ArenaExpertSource::blob()` refuses a GPU-owned pair, so a later CPU path cannot silently read decommitted memory. The flag defaults off and no launcher/profile was edited.

The new `xeno_exclusive_host_pages` test used a two-expert temporary pack. It went red before `release_host_copy` existed, then passed: the claimed pair became unavailable through `blob()`, the adjacent blob's byte remained readable, the byte count was page-aligned and below the blob size, and a second claim was refused. The release primitive refuses CUDA-registered, OS-locked and large-page backing. The test also loaded the same small pack with a real CUDA-registered host arena on the 5060 Ti and verified that exclusive release was refused while `blob()` remained readable. OS-locked rejection is code-checked but not separately fault-injected. This first slice is Windows-only.

## Real model checks

All runs used the accepted Q2_0 baseline, six P-core workers, a fixed sky prompt, spec 4, no adaptive swaps, no cache borrow and PCIe miss share 0. Process private commit was read with `GetProcessMemoryInfo` immediately before and after release.

| Placement | Released host pages | Process private commit | Raw token parity | Rough decode |
|---|---:|---:|---|---:|
| 5060 Ti primary cache 1,000, no secondary | 1.29 GiB | 37.68 → 36.39 GiB | sky 96/96 vs accepted baseline | 27.03 tok/s |
| 5060 Ti primary cache 5,000 + 4070 SUPER secondary compute 256 MiB | 6.43 GiB | 43.37 → 36.94 GiB | sky 96/96 vs accepted baseline | 31.80 tok/s |

The dual-card run staged 194 non-primary experts on the 4070, served 598 entries in 465 groups and sampled a minimum lower CUDA/NVML free value of **9.21 GiB**, above the 2560 MiB display floor. A prior copy-kept dual-card sky run at the same cache sizes measured 25.16 tok/s, but these are unpaired runs under changing CPU/desktop/file-cache load; the 31.80 versus 25.16 comparison is a hypothesis about benefit, not a measured speedup claim. Artifacts: `%TEMP%\strata-phase4-exclusive-single1`, `%TEMP%\strata-phase4-exclusive-single96`, `%TEMP%\strata-phase4-exclusive-dual96`.

## Full-slot verification run

After code review, the exclusive arm verifies every distinct resident primary slot before releasing its host pages. The build passed 9/9 `xeno_` and 8/8 `gpu_` CTests; 23 Python serving tests also passed. On the dual-card 5,000-slot / 256 MiB setup, all four 256-token outputs matched the accepted baseline exactly:

| Prompt | Raw IDs | Decode | Minimum sampled 4070 free VRAM |
|---|---:|---:|---:|
| sky | 256/256 | 26.20 tok/s | 9.11 GiB |
| Thai | 256/256 | 25.03 tok/s | 9.16 GiB |
| code | 256/256 | 32.25 tok/s | 9.09 GiB |
| long | 256/256 | 38.32 tok/s | 9.09 GiB |

These are unpaired decode measurements, not a speedup claim. Verifying about 6.44 GiB by device-to-host readback raises startup time. Artifacts: `%TEMP%\strata-phase4-verified256-{sky,thai,code,long}`.

The same binary with host copies kept also produced identical 256/256 raw IDs in all four prompts, completing the four-prompt parity gate against both the accepted baseline and same-binary copy-kept mode. One-run decode comparison was mixed: copy-kept sky 35.03 vs exclusive 26.20 tok/s; Thai 22.27 vs 25.03; code 35.56 vs 32.25; long 28.52 vs 38.32. These large, opposing differences require repeated paired measurements before any performance conclusion. Copy-kept artifacts: `%TEMP%\strata-phase4-copykept256-{sky,thai,code,long}`.

A sky 256-token ABBA order across separate processes measured exclusive 26.20, copy-kept 35.03, copy-kept 31.19, exclusive 32.71 tok/s; other prompt runs intervened between the middle two. All four outputs were byte-identical in raw IDs. The two-run means were 29.46 vs 33.11 tok/s (exclusive 11.0% lower), but each arm varied substantially, so this is a provisional speed signal rather than a stable throughput estimate. Minimum sampled 4070 free VRAM in the sequence was 8.95 GiB. Follow-up artifacts: `%TEMP%\strata-phase4-copykept256-sky-b2`, `%TEMP%\strata-phase4-verified256-sky-a2`.

## Required next gates

- Repeat paired throughput and committed-memory measurements under controlled background load. Test the default-sized primary cache and a long Claude Code session.
- Implement the actual Phase 4 spare-RAM-slot newcomer H2D/victim D2H swap, two-event ownership publication, hysteresis/budget and aligned unbuffered pack recovery before marking issue #4/#13 complete. The current flag explicitly disables swaps.
- Preserve the opt-in 4070 100 ms lifetime monitor; graceful CPU fallback and default policy remain open.
