# Phase 4 static exclusive primary pages — first slice

Scope: engine #4 / tracker #13, branch `xeno/afk-phase4`. **This is a static ownership and RAM-release experiment, not the complete Phase 4 swap protocol or a production default.**

`--exclusive-primary-experts` opts in and refuses startup unless the model is native Q2_0, the profile/primary cache are present, the CPU pool is enabled, and `--no-prefill-borrow --adapt-swaps 0 --pcie-frac 0` are set. Forced CPU-miss and mmap expert modes are incompatible. The large host arena is pageable in this mode. After the primary cache is profile-filled and slot 0 verified, each resident pair is marked GPU-owned and whole Windows pages strictly inside its old host blob are decommitted. The arena's virtual address, layout and neighboring blobs stay in place. `ArenaExpertSource::blob()` refuses a GPU-owned pair, so a later CPU path cannot silently read decommitted memory. The flag defaults off and no launcher/profile was edited.

The new `xeno_exclusive_host_pages` test used a two-expert temporary pack. It went red before `release_host_copy` existed, then passed: the claimed pair became unavailable through `blob()`, the adjacent blob's byte remained readable, the byte count was page-aligned and below the blob size, and a second claim was refused. The release primitive refuses CUDA-registered, OS-locked and large-page backing. The test also loaded the same small pack with a real CUDA-registered host arena on the 5060 Ti and verified that exclusive release was refused while `blob()` remained readable. OS-locked rejection is code-checked but not separately fault-injected. This first slice is Windows-only.

## Real model checks

All runs used the accepted Q2_0 baseline, six P-core workers, a fixed sky prompt, spec 4, no adaptive swaps, no cache borrow and PCIe miss share 0. Process private commit was read with `GetProcessMemoryInfo` immediately before and after release.

| Placement | Released host pages | Process private commit | Raw token parity | Rough decode |
|---|---:|---:|---|---:|
| 5060 Ti primary cache 1,000, no secondary | 1.29 GiB | 37.68 → 36.39 GiB | sky 96/96 vs accepted baseline | 27.03 tok/s |
| 5060 Ti primary cache 5,000 + 4070 SUPER secondary compute 256 MiB | 6.43 GiB | 43.37 → 36.94 GiB | sky 96/96 vs accepted baseline | 31.80 tok/s |

The dual-card run staged 194 non-primary experts on the 4070, served 598 entries in 465 groups and sampled a minimum lower CUDA/NVML free value of **9.21 GiB**, above the 2560 MiB display floor. A prior copy-kept dual-card sky run at the same cache sizes measured 25.16 tok/s, but these are unpaired runs under changing CPU/desktop/file-cache load; the 31.80 versus 25.16 comparison is a hypothesis about benefit, not a measured speedup claim. Artifacts: `%TEMP%\strata-phase4-exclusive-single1`, `%TEMP%\strata-phase4-exclusive-single96`, `%TEMP%\strata-phase4-exclusive-dual96`.

## Required next gates

- Compare all four fixed prompts for 256 raw tokens against the accepted baseline and same-binary copy-kept mode. A later primary resident expert may reveal a source access the sky-96 probe did not reach.
- Run paired ABBA speed and committed-memory measurements; record background load. Test the default-sized primary cache and a long Claude Code session.
- Implement the actual Phase 4 spare-RAM-slot newcomer H2D/victim D2H swap, two-event ownership publication, hysteresis/budget and aligned unbuffered pack recovery before marking issue #4/#13 complete. The current flag explicitly disables swaps.
- Preserve the opt-in 4070 100 ms lifetime monitor; graceful CPU fallback and default policy remain open.
