# Merge report: upstream v0.1.42 into the fork's main (2026-10-11, #212)

Upstream `Niko1221/Strata` `v0.1.42` (`61b3fb5d`, 2026-10-10) merged into the fork's `main` `8abbfa02` (the deployed
`run-main-8abbfa0`, which holds v0.1.38, #129, and everything up to #210) on `xeno/212-merge-0.1.42`, worktree
`C:\Strata-exp\src-212`, build dir `C:\Strata-exp\build-212` (`build-212.cmd`: CUDA 13.3, sm_89 + sm_120, tests on).
Working records: `C:\Strata-exp\merge-212-record\`: `BRIEF.md` (the rules the resolving agents followed),
`notes\{generate,prefill,core,kernels,serve,orchestrator}.md` (every hunk with its reason), `sides\` (base / ours /
theirs of each conflicted file), `hunks.txt`, `conflicts.txt`, `upstream-commits.txt`, `fork-commits.txt`,
`serve-web-commits.txt`, `blueprint-reasons.txt`, and the build, ctest and pytest logs.

**Verdict.** On the served D2x config the merge keeps the main model's and the drafter's prompt state byte-identical
to the deployed `main` (all seven `STRATA_STATE_HASH` fingerprints in four boots, `mtp=` included) and gives identical
replies in Claude Code's request shape and identical generate-mode outputs. In the same session it reads every
Claude Code part 84-149 ms faster (5-7 %), the long prefix at the same speed, and answers at 73.0 against 63.6 tok/s
in the server with the same windows and draft acceptance; generate-mode decode is equal within noise (§5a). Which
stage gives the server's decode gain is not measured yet. One regression is measured: the window's PLE rows are asked
twice (+4,096 rows and about +4,100 SSD reads per 256 tokens, 0.1-0.15 ms a token, under the ABBA's resolution; #213).
The tree builds at the first try, ctest passes but the known `ple_parity`, and the pytest suites pass. Every upstream
default that would change D2x's outputs or its prompt-path scheduling is kept at the fork's value, with upstream's
value behind a switch, with two exceptions: the Windows host-thread pin (#626) is taken with no switch back, and
upstream's PCIe probe has no switch that restores it (D2x does not reach it: `--pcie-frac 0`). The merge goes to
`main`; whether D2x moves to its exe is the developer's call.

## 1. What moved

- **Base:** the fork's `origin/main` `8abbfa02` (the deployed `run-main-8abbfa0`).
- **Upstream:** `upstream/main` `61b3fb5d` = tag `v0.1.42` (2026-10-10). Since v0.1.38: 1,159 commits, 1,083 of them not
  merges. Releases 0.1.39, 0.1.40 (.1-.4), 0.1.41 and 0.1.42. 0.1.41.1 was not released; its fixes are in 0.1.42
  (`cdbf4586`).
- **Size:** `git diff --shortstat 99f3dbd0 upstream/main`: 2,605 files, +632,453/-3,442. Most of it is the new SYCL
  backend under `sycl/` and the community results under `bench/` (2,128 files, +547,566).
- **The fork since the same base:** 407 non-merge commits; 657 files, +102,649/-1,790.
- **Conflicts:** 36 files, 305 hunks in zdiff3 style (`hunks.txt`):

| file | hunks | what collided |
|---|---|---|
| `serve/server.py` | 68 | upstream's multi-pass `_run` loop (#537 forced call, #728 recovery, #1053 close retry, #615 per-pass stats), `--batch` slots, replicas, #606 repeat stop, #1317 stall check, #1619 queue keep-alive, #1727 keep-awake, logit bias, `/config`, `/v1/vram`, `/metrics/prometheus`, against the fork's `_generate` budget legs (#123 / xeno #49 S3), #199 cycle close, #200 reasoning carry, `RequestGate` priorities, #59 floor-breach restart, loop guard, timeline spans, `/metrics/requests` |
| `src/program/generate.cpp` | 67 | upstream's `--batch` slots and new serve loop head, disk sessions (`SAVE` / `RESTORE`), tail skip default, CPU prefill share, PcieAuto, `--vram-elastic`, `--kv-grow`, `--adapt-async`, Foresight, `--remote-expert-opt`, `STRATA_ADAPT_LAG`, per-part refills, `--version`, forced EAGER loading, against the fork's 4070 tier, paired swaps, wave lanes, loan (`lend_slots` / `refill_lent`), #53 `serve_fatal`, #203 `end_engine`, `ExitTrace`, timeline spans |
| `src/prefill/prefill.cpp` | 55 | upstream's PeerPrefill (now with a no-P2P route), CPU prefill share, #583 byte-budget ring, #789 stream-ahead, pp-opt stager batch, KV prefetch, layer-split pipeline and help, #1454 `bo = emb`, against the fork's split, wave, copy issuer, expert order, #29 group gather, #31 PLE read-ahead, #122 `kv_end` sizing |
| `README.md` | 17 | the fork's own README (#166) against upstream's |
| `src/core/verify.cpp` | 12 | upstream's #783 decode series (`stage_inputs`, zero-doorbell graph, PLE overlap, one-token commit), logit bias, against the fork's Q2_0 `x_scales` path, `pcie_share_`, H2 `moe_hit_merge_mapped`, #49 ban sampling, timeline spans |
| `src/core/expert_source.cpp` | 9 | upstream's #285 deferred registration, `prefetch` / `release`, the dispatch's `gpu_zeroes_hits` / `any_cpu` / offload counters, tail skip, against the fork's placement-first load, 4070 claim, tier counters, #95 NVMe second run |
| `src/kernels/cpu/pool.cpp` | 8 | upstream's `ThreadAffinity` pin (#626), `--pool-tasks`, `--aux-cpus`, #642 hybrid sort, `q2_native_kernels`, against the fork's #147 one-phase block, `spin_us` (#64), priorities, `end_engine` stall exits |
| `src/kernels/cuda/iq_kernels.cu` | 7 | upstream's #783 PR-h sub-warp kernels, staged IQ grid, AMD layouts, #606 finite q8_1, against the fork's Q2_0 CPU-order path (`x_scales`) |
| `src/kernels/cpu/q2_avx2.cpp` | 6 | upstream's own AVX-VNNI Q2_0 rows and opt-in bit-plane kernel against the fork's VNNI `row_multi` |
| `serve/test_server.py`, `serve/frontend.py` | 5 each | test classes both sides added; the fork's billing-header strip (xeno #6) against upstream's `pin_billing_stamp`, `_check_parts`, `_detect_caps` |
| `setup.py`, `include/strata/core/expert_source.hpp` | 4 each | `DRAFT_VOCABS` (`thai` / `fr`); the `ExpertSource` virtuals and `load_experts_gguf`'s parameters |
| `src/kernels/ngram.cpp`, `src/kernels/cuda/sampler.cu`, `include/strata/prefill/prefill.hpp`, `CMakeLists.txt` | 3 each | two PLE read-aheads under different names; the fork's `ban` against upstream's `logit_bias` / Gumbel; `bytes_needed`'s fourth parameter; link lists and test blocks |
| `src/platform/direct_file.cpp`, `src/ngram/ple_reader_test.cpp`, `src/kernels/cpu/expert_layout.cpp`, `src/core/mtp.cpp`, `serve/telemetry.py`, `include/strata/prefill/kernels.hpp`, `include/strata/kernels/cpu/pool.hpp` | 2 each | `--aux-cpus` pins beside the fork's #139 issuers; both sides' test modes; two `cpu_avxvnni_ok()` definitions; #783's draft against the fork's spans and #44 D4 `on_draft`; the sampler thread's stop; wire functions against no-P2P peer kernels; the pin API and the pool constructor |
| twelve files with 1 hunk | 1 each | `tools/draft_vocab.py`, `src/prefill/kernels.cu`, `src/core/remote_experts.cpp`, `src/core/pinned.cu`, `src/core/device.cu`, `include/strata/prefill/moe_mmq.hpp`, `include/strata/kernels/sampler.hpp`, `include/strata/kernels/cpu/expert_layout.hpp`, `include/strata/kernels/cpu/expert.hpp`, `include/strata/core/pinned.hpp`, `docs/SECOND_GPU.md`, `AGENTS.md` (§3) |

**Upstream rewrote its history.** The two lines have no merge base. Upstream's root commit is `01529aaa`, the fork's
`f2a08d4b`. Every old upstream tag points at a new commit: the fetch refused `v0.1.4` .. `v0.1.39` as "would clobber".
The trees are the same, though: upstream's v0.1.38 `09817da3` and the fork's `99f3dbd0` both have tree `b430923a`
(v0.1.39 is `27b0e86f`). The base was joined by hand:

- A plain `git replace 09817da3 99f3dbd0` gave the merge bases `e4e0ceed` + `bd649acd`, the parents of 0.1.38's merge,
  because git still sees the commit id `09817da3`.
- `git replace --graft 09817da3 99f3dbd0` gave the merge base `99f3dbd0`. A dry run with that base equals the explicit
  `git merge-tree --merge-base` dry run (the same 36 files).
- The graft is local only. It is deleted after the merge commit and never pushed. The merge commit's second parent is
  upstream's real `61b3fb5d`, so the next merge has a normal base.

Merge commit, later commits and the exe sha256: §6.

## 2. Upstream's changes that matter here, and what the fork does with them

D2x (`strata-flash-next-d2x.json`): Swift 1.5 IQ2_XS, `--exclusive-primary-experts`, the 4070 tier
(`--secondary-expert-mib 4100`), paired adaptive swaps (`--adapt-swaps 8`), split + wave (`STRATA_PREFILL_EXPERT_SPLIT=1`,
`STRATA_PREFILL_WAVE=1`), `STRATA_SPLIT_BO=q8`, `--prefill auto`, `--pcie-frac 0`, `--kv int8`, `--kv-resident 65536`,
`--spec 4 --spec-min-p 0.5 --mtp`, `--ple-gguf`, `--short-read 256`, `--vision`, `--pool-priority 2
--process-priority 2`, `STRATA_POOL_SPIN_US=0`. For each item: whether it reaches that path, and whether it is on.

**Defaults kept at the fork's value on D2x, with upstream's value behind a switch:**

| item | reaches D2x | on | why / switch to upstream's value |
|---|---|---|---|
| `STRATA_ROUTE_TAIL_SKIP` default rank 7 on CUDA serve (`b4e3407b`) | yes | **off** | It changes outputs (KL ~0.003-0.006 per upstream). Upstream's predicate does not see the fork's 4070 tier (it excludes only `--peer-device`, helper GPUs, `--batch`), so D2x would have had it on. `tail_default_ok` (generate.cpp ~9727) adds `o.secondary_expert_mib <= 0`. `STRATA_ROUTE_TAIL_SKIP=7` is the arm. With it, the dispatch's skip loop now spares an expert the 4070 holds (`expert_source.cpp:3616`, `secondary_res >= 0`), as it spares a CUDA0-resident one |
| CPU prefill share, on by default since 0.1.41 (`162ce64e`) | yes | **off** | Upstream's "one GPU" test (`stages.empty() && !multi_gpu`) passes with the 4070 tier. Two guards: `share_by_default` needs `o.secondary_expert_mib <= 0` (generate.cpp:8339, serve and generate mode), and prefill.cpp's `cpu_share_env` / `PoolHold` never arm it by default in a split layer or with the 4070 tier (`m.peer_res == nullptr`). `STRATA_PREFILL_CPU_SHARE=1` / `auto` arms it. It has never run on the fork's walk (§7) |
| `STRATA_RING_BYTES` (#583 byte-budget ring + `--prefill auto` scan, `bcd3603e`) | yes (`--prefill auto`) | **off** | It moves D2x's ring, its loan and its outputs. `=1` is upstream's default. The fork's corrected buffer count (#122) applies either way (prefill.md hunks 28-31) |
| `STRATA_PREFILL_STREAM_AHEAD` (#789: deferred shared expert, `STAGE` copies in flight) | yes (staged chunks) | **off** | It reschedules D2x's staged parts. `=1` is upstream's default. It also governs the auto-merged `pending < STAGE` look-ahead |
| `STRATA_MTP_CATCHUP_ALL` (#783 PR-i, `3fd0460b`) | yes (`--spec 4 --mtp`) | unset = **the whole window** (the fork's) | Upstream's default skips the drafter's K/V catch-up of the rejected rows. Its exactness rests on an argument about which cells are read later, not on identical kernels, and it is unmeasured here (the 0.1.38 `STRATA_MTP_BATCH_RING` precedent). `STRATA_MTP_CATCHUP_ALL=0` is upstream's (mtp.cpp:63-69) |
| CUDA module loading | yes | **LAZY** (fork #30) | Upstream forces EAGER at the top of `main`. `CUDA_MODULE_LOADING=EAGER` in the environment gives upstream's |
| `--ple-inflight` | yes | **64** | Kept from 0.1.38. `--ple-inflight 256` is upstream's |
| PCIe share with no `--pcie-frac` (start-up probe + PcieAuto refinement, `aa7b0cb1`) | no (D2x passes `--pcie-frac 0`) | **0** (fork #27) | On this machine's x4 link the probe chose 0.15 and decode fell to 3.24 tok/s (#27). The probe is not taken, so `pcie_frac_auto` is never set and PcieAuto never runs; its code stays in place. **No switch restores upstream's no-flag default** (an explicit `--pcie-frac` turns PcieAuto off upstream too) |
| #606 `repeat_stop_tokens` (serve) | yes | **256 (on), since 2026-10-12** | At the merge commit it was 0: with 256 a single-token loop ended as "length" before the fork's loop guard, uncounted in `/status loops_stopped`. The developer chose to remove that cost instead (2026-10-12): the stop is now reported as the guard's (`stop_detail: "loop"`, counted in `loops_stopped`), so it is on by default. It sees what the guard cannot: a token that never becomes reply text (a tool call's arguments, a special token) and a repeated token longer than the guard's 8-character period. `"repeat_stop_tokens": 0` turns it off |
| Claude Code billing header | yes (D2x serves Claude Code) | **the fork's strip** (xeno #6, `bc1b18f2`) | Upstream pins the stamp instead (`pin_billing_stamp`). Both aim at a stable prefix; the fork's prompt bytes stay. `pin_billing_stamp` and `BILLING_HEADER` stay defined, uncalled. No switch |

**Upstream defaults taken that reach D2x.** What upstream claims for each is in its row. For the kernel and decode rows,
greedy parity (§5a) is the check, and their switches are the bisection order if it diverges:

| item | reaches D2x | on | why / switch |
|---|---|---|---|
| #783 decode series in `verify.cpp`: `STRATA_ONE_TOKEN_COMMIT` (a T=1 window advances the state itself), `STRATA_HEAD_MIX_MULTI`, `STRATA_SH_STREAM`, `STRATA_DEC_BATCH` | yes | yes (CUDA defaults) | claimed bitwise; each variable is its own switch |
| #783 zero-doorbell graph (`ar_on()`) | no | - | only when every expert of every layer has a CUDA0 slot. With `--exclusive-primary-experts` a 4070-only expert has `h_res < 0`, so `all_resident_` is false and every D2x window takes the doorbell path that calls the host dispatch and the 4070 claim (core.md, checked in `init` / `refresh_ar()`) |
| #783 PLE overlap in `Verifier::run` (`prefetch_rows` of the window's rows before the launch, `gather_batch` after layer 0 rang; verify.cpp:2014 / 2045) | yes | yes | the overlap itself; it may read rows the fork's #44 D4 read-ahead already asked for (hypothesis, §7) |
| #783 PR-h sub-warp GPU expert kernels (`row_dot_80_sub16` for gate/up at `n_embd == 2560`, `row_dot_20_sub16` for a Q2_0 down at `n_ff == 640`) and the staged IQ grid (`STRATA_IQ_STAGE_GRID`, IQ2_XS is in `kStageIqGrid`) | yes (the grouped expert kernels on D2x's IQ2_XS gate/up and Q2_0 down) | yes | claimed bitwise by upstream (`iq_multi_parity`, `native_grouped_parity`, `native_expert_parity`). The fork's Q2_0 CPU-order path (`x_scales`) bypasses them as before. Switches: `STRATA_NO_SUB16_GU=1`, `STRATA_IQ_STAGE_GRID=0`, `STRATA_OLD_IQ_MMVQ=1`. Whether D2x's layers meet the sub16 sizes exactly was not checked against the model file |
| AVX-VNNI forms of the IQ AVX2 kernels and `STRATA_IQ256_GATHER` auto (gathered grid on the P-cores of AVX-VNNI Intel CPUs) | yes (the CPU pool's IQ2_XS gate/up through `iq256_gu_rows` for groups of 2+ tokens) | yes | claimed "the same bits" (integer sums). Switches: `STRATA_NO_AVXVNNI=1` (or the fork's `STRATA_FORCE_AVX2=1`), `STRATA_IQ256_GATHER=0` |
| Upstream's own AVX-VNNI Q2_0 rows (`q2_avx2_rows.inl`, `plain::` / `vnni::`) in place of the fork's `row_multi<NT, Vnni>` | yes (the CPU pool's Q2_0 down rows) | yes | compared line by line: the same integer dot, the same two FMAs per block in the same order, the same `corr` chain and reduction (kernels.md). The opt-in bit-plane kernel (`STRATA_Q2_BITPLANE=1`) stays off, as upstream has it |
| float4 `moe_combine` (`moe_combine4_kernel`, `528c1fb1`) | yes (the non-split combine) | yes | "bitwise equal" per upstream: the same k order and expression. `STRATA_W_NO_COMB` is a build define, not a run switch. `xeno_combine_split_parity` / `xeno_combine_partial_parity` passed (§5) |
| `m.bo = m.emb` (#1454) | yes | yes | memory only: `emb` is dead after `gr_broadcast`. Checked against the split: the routed sum reaches `bo` on the relay stream only after the 4070 has read this layer's activations, and `m.cs` waits `ev_done` before the combine; the wave's lanes have their own buffers. `bytes_needed` still counts `bo` unless `STRATA_EMB_REUSE_ACCOUNT=1`, so loans do not move |
| `STRATA_ADAPT_LAG` (#764, `d69424c9`) | yes | 1 | 1 = wait for `pending` every window, #463's wait, the fork's behaviour before. D2x's paired swaps keep no `pending`, so this is `apply_pending(true)` as before. `STRATA_ADAPT_LAG=2` is the other value |
| `--ple-io direct` start-up probe (`30017336`) | yes | yes | up to ~0.6 s at start; it falls back to mmap below 15k rows/s, so a fast NVMe keeps Direct. A set `STRATA_PLE_PROBE` skips it. Boot time on D2x not measured |
| #620 the native head and the logits buffer allocated before the expert arena | yes | yes | moves D2x's allocation order. Both orders are before the expert cache is sized, so the cache's free-VRAM reading should be unchanged; to read in the boot log's free-VRAM lines (§5a) |
| Server restart retries (PR #637) | yes | yes | a failed restart that the display floor does not explain now retries up to 3 times, 15 s apart. A floor breach (#59) still starts degraded without the 4070 tier, with no retry |
| Windows host-thread pin (#626, `5a8b2924`) | yes | yes, **no switch back** | `pin_current_thread` is now a CPU Set selection (`SetThreadSelectedCpuSets`, soft) with a restore, instead of `SetThreadAffinityMask` (hard). Pool workers still hard-pin through `SetThreadGroupAffinity` (the same mask on a PC with 64 or fewer logical processors). It places session.cpp's host loop and the fork's `SecondaryRunner` thread (`secondary_runner.cpp:210`). Upstream's reason: processor groups, more than 64 logical processors. **The one D2x default moved with no switch to the fork's old value**: adding one would be new code (brief rule 2). Taken as upstream's fix; covered only by the D2x ABBA (§5a), not measured alone |

**New refusals with the 4070 tier,** each in upstream's own form for that feature ("off, said once"):

- `--adapt-async 1` with `--exclusive-primary-experts` or `--secondary-expert-mib > 0`: the async tier does not know the
  fork's paired swaps, the 4070's residency, or experts owned with no RAM copy. Upstream already turns it off for
  `--peer-device`, `--remote-expert-opt` and batch groups.
- `--kv-grow` (same condition): an exclusively placed expert has no RAM copy for the CPU to compute when the cache gives
  its slot to the K/V. D2x also fails upstream's own `kv_resident <= 0` test.
- `--vram-elastic` (same condition): a shrink would make the slots' experts CPU misses.
- `STRATA_FS_SLOTS` (Foresight swap space, serve): not created with `--secondary-expert-mib > 0`. The dispatch's
  `d.fs->take` / `note_miss` runs for every expert not in CUDA0's cache before anything knows the 4070 holds it, and
  `note_miss` would copy from `d.src->blob`, which an exclusive 4070 expert does not have.
- Kept from 0.1.38: `--peer-device` with the 4070 tier, `--exclusive-secondary-experts` or
  `STRATA_PREFILL_EXPERT_SPLIT=1` exits 2.
- **Not refused (open, §7):** upstream's helper caches (`--expert-cache-device1..3`, `--remote-expert-opt`,
  `STRATA_PREFILL_HELPER_SOURCE`) with the 4070 tier (not refused at 0.1.38 either); upstream's helper GPUs
  (`RemotePrefillSource`) with the fork's NVMe tier.

**Upstream features not taken, with the reason:**

- **PeerPrefill and everything built on it:** `--peer-device`'s share of the prompt path, including 0.1.42's no-P2P
  route (`81af1da0`, `f4bfedc5`, `550159fc`, `da0024fe`, `59ea77e5`: `STRATA_PF_PEER_HOST/SUMS/F16/GATHER/XSTREAM`), and
  the layer-split help (`4175cbcb`, `STRATA_PREFILL_HELP`, `STRATA_PREFILL_HELP_FRAC`). The 0.1.38 precedent: the fork's
  prompt path (split, wave, issuer, expert order) is restructured. `Prefill::set_peer` still declines;
  `set_stage_helper` is a stub that does nothing unless `STRATA_PREFILL_HELP=1`, then declines with a message, and
  generate.cpp prints it and goes on. The no-P2P peer kernels (`moe_combine_peer`, `peer_scatter_add`,
  `peer_gather_add`, `moe_combine_peer16`, `sums_to_f16`, `f16_to_f32_wide`) stay declared and compiled, uncalled.
- **#954's fused-buffer backstop** (`73d39830`, `STRATA_DBG_FORCE_SLOW_LAYER`): the fork's `moe_bufs` keep MMQ's full
  size, and the backstop would fire falsely on a split chunk of a fused layout.
- **`STRATA_MMQ_RESIDENT_SORT_NE`** (`c6a0d403`, opt-in): it rebuilds slot / src / off / order after the fork's
  `split_row_layout` and expert order; not a simple port.
- **Upstream's owned-buffer carve** (the ring in one cudaMalloc, per-take 2 MiB pricing): the fork owns one block (#30).
  `bytes_needed_owned` is ported as the fork's count rounded up to 2 MiB.
- **The `stager_sleep()` yield-spin arm** (#1057): the fork's stager always blocks (BlockingSync events, atomic waits,
  #35 D4). On Windows upstream's default is off anyway, so D2x is unchanged. `stager_sleep()` is defined, unused.
- **mtp.cpp's draft-time PLE prefetch** (`a36be1d`, `prefetch_ple` -> `PleTable::prefetch_rows`): the same job as the
  fork's #44 D4 (`on_draft` -> `PleAhead::push` -> `PleTable::prefetch`, on by default, `--ple-ahead`). One mechanism
  for one job, the fork's on D2x (the #29 / #372 precedent). `ple_ss_` / `set_ple_session` stay, unused here.
- **The per-part refill helpers** (`refill_issue` / `refill_wait` / `refill_one` / `refill_serial`, with #1237's
  `blob_stable` + `note_async_read`): the fork had removed them for `refill_lent` (#34/#139), which fills with
  `fill_slot_blocking`, so the source is read before the call returns. Upstream's Windows `srcp->release(...)` after the
  refill is ported into the fork's refill (a no-op on the fork's arena on Windows).
- **#372's `used_of` / `group_gather` / `flush`:** still not taken (0.1.38; the fork's #29).
- **Upstream's PCIe start-up probe** (above).
- **Upstream's #224 `_Exit` and the watchdog's `release_gpu_waits` + `abort`:** the fork's `serve_fatal` (#53) and
  `end_engine` (#203) do those jobs.

**The rest (taken as upstream has it; off unless given, or not on the D2x path):**

- **Serve:** `--batch` / `"parallel"` slots, batch groups, batch MTP, `--pipeline-windows`, replicas; disk sessions
  (`SAVE` / `RESTORE`, `conversation_file.hpp`, `session_save_reclaim.hpp`; refused upstream with a layer split,
  `--peer-device`, `--batch` or no `--prompt-cache`); per-request `logit_bias` (`ver.set_logit_bias`, beside the fork's
  `ban` bitmap; they compose); `ckpt=0` (#830) and `pin=N` (`strata_prefix`); `STRATA_CACHE_MESSAGE_BOUNDARY=1`; the
  #1317 stall check; the #1619 queue keep-alive; #1727 keep-awake; #1013 `reused` clamp; #893 `BadBody`; `/config`
  (#564), `/v1/vram` (#533), `/metrics/prometheus`; #748 fatal verifier lines; #997 / #890 `last_err`; #1705's 256 KiB
  `death_note` read; #458 effort end; #537 literals; #843 empty assistant turns not rendered (with the fork's `_carried`
  exception, §3).
- **Engine, opt-in or not D2x:** `--vram-elastic` / `--vram-segment-mib` and the `VRAM` command; `--kv-grow`;
  `--adapt-async`; Foresight (`STRATA_FS_SLOTS`); `--remote-expert-opt`, helper caches, `STRATA_DISJOINT_ADAPT`;
  `--aux-cpus` (pool workers call `note_owned_thread()`, so it never moves them); `--host-core`; `--pool-tasks`
  (`phase_tasks` = `3 * threads` by default, as before); `--lookup-chain`; `--window-hashes`; `--version` / `-V`;
  the `CUDA_LAUNCH_BLOCKING` warning (#1341); `--gpu LIST`; the HIP `GPU_PINNED_MIN_XFER_SIZE` block; `STRATA_FILL_AHEAD`
  (mapped sources only, not the arena); `STRATA_STAGE_PIN` (opt-in since `fb58e0db`: on by default it corrupted IQ3_S
  decode); `STRATA_DEFERRED_REGISTER` (#285, opt-in, Windows); `STRATA_ARENA_MMAP` (Linux); the KV prefetch
  (`STRATA_KV_PREFETCH=1`; never in a wave, see §3); K8V4 (`--kv k8v4`); `STRATA_PREFILL_EQUAL=1` (`equal_chunk`
  returns the same chunk as the fork's inline rule with it unset); the pp-opt stager batch (`STRATA_STAGER_BATCH`; 8
  only with a GGUF read in place, 1 on D2x); `STRATA_SPEC_PROB` / `STRATA_SPEC_GUMBEL`; the S23 / HIP switches
  (`STRATA_PF_PAD`, `STRATA_PF_HCDOWN`, `STRATA_HC_UPMIX`, `STRATA_CVEC_FUSE`, ...); `STRATA_DBG_*` research hooks;
  `STRATA_WATCHDOG_IO_S` (#1407).
- **Hardware and platforms the fork does not build or run:** the SYCL backend (`STRATA_ENABLE_SYCL`, default OFF), the
  AVX1 rung (`q2_avx1.cpp`, `iq_avx1.cpp`, `kq_avx1.cpp`; only where `!cpu_avx2_ok()`), gfx906, sm_70
  (`STRATA_SM70_TABLE=1`), Linux #798 hybrid detection, #771 THP, the IQ3_S one-token kernel (`STRATA_IQ3S_MT1=1`), the
  PLE format table (Q5_1, Q8_0, Q4_0, BF16 rows), `--ple-io ram` on Windows, setup and docs.

## 3. Every conflict, file by file

Six groups resolved the files in parallel under `BRIEF.md`; the orchestrator resolved the rest. Each group's notes hold
every hunk with its side and reason; this section is their summary.

### `src/program/generate.cpp` (67 hunks; `notes\generate.md`)

- **Kept from the fork:** LAZY module loading (#30; upstream's forced EAGER dropped); the PCIe default 0 with no probe
  (#27); the 4070-tier set-up block ("Phase 3" .. "lock CPU experts", 342 lines) whole; the paired swaps (stages 1-3,
  `sec_adapt`); the wave lanes (`sp2`, `sp_wave`, `prefill2`, `wave_link`, #35 D7); the loan (`lend_bytes`, `lend_slots`
  with `kv_end` / `wave`, `refill_lent`, #34/#56/#122/#139/#172); `run_prompt` and the always-on "prompt part" line
  (#37, parsed by `tests/xeno/perf/read_sizes.py`); `serve_fatal` (#53); `arm_end_engine` (#185/#203); `ExitTrace`
  (#45); the timeline spans; the `STATS` line; `ban` (#49 S4), `cache_slot` (`req_slot`), `ckpt_at` (#49 S7); the
  `--adapt-gate` EMA; `LinkProbe` (#22); `--profile-decode-range`.
- **Ported from upstream onto the fork's code:**
  - the `ExpertPool` call is `ExpertPool(workers, pin, host_works, spin_us, affinity, o.pool_tasks)`;
  - `blob_stable` in place of `blob` in the profile fill (with io prefetch `blob` may hand out a recycled stage buffer);
  - `slots_from_bytes` / `bytes_from_slots` under the fork's `lend_slots` (upstream's one-argument `lend_slots` dropped);
  - every upstream `bytes_needed(..., srcp != nullptr)` call now passes five arguments (§4);
  - the CPU prefill share's arming (default off with the 4070 tier, §2), `set_cpu_pool` on lane 1 only (the share is
    for a pool no other thread runs meanwhile, and the two lanes run at once);
  - `prefill_src` (`STRATA_PREFILL_HELPER_SOURCE` wrapper) in both lanes' `init`;
  - the drafter's prompt K/V: the fork's per-lane `serve_chunk(lane)` factory (#48) with upstream's
    `STRATA_SPLIT_MTP_BATCH`, `draft_kv::plan`, the `why` decline line and `use_mtp` guards (the drafter's prompt path is
    `multi_gpu ? stages.back()->sp : *lp`; on one GPU the plan is `try_batched`, as before);
  - upstream's `resident_stage_swaps(..., Locate)` / `SwapHome` template: every fork call of the old form is rewritten
    to it, with a CUDA0-only `SwapHome` in generate mode;
  - the candidate filters keep the fork's 4070 and lent-tail rules and add `!(remote_opt && remote_opt->owns(l, e))`
    and `!helper_holds(l, e)`;
  - the INFO line carries `ban=` (fork) and `conversation_cache_min_tokens`, `tail_role_token`, `vram_elastic` and the
    `--batch` suffix (upstream), in that order;
  - the watchdog: upstream's `STRATA_WATCHDOG_IO_S` allowance and the session files' `allow_until_ms`, with the fork's
    `arm_end_engine()` moved **after** upstream's two `continue` checks (armed first, the 20 s deadline would cut a slow
    file tier's allowance of up to 10 x the limit);
  - the serve loop: upstream's `for (;;) { try_next_line / next_line }` head with `BSTOP`, `BYIELD`, `VRAM`,
    `adapt_tick(true)`, `BGEN`, after the fork's `serve_fatal` lambda;
  - the split points: `cuts = {reread_to, root_at, shared_at, message_at, turn_at, n - 1}`, sorted only when a pin is
    added (upstream's rule); a part ending at `pin_at` / `message_at` is labelled "new turn" (the parser keeps the fork's
    five labels); checkpoints at turn, root, `shared_at`, `message_at`, `pin_at`;
  - the part reader: the fork's `run_prompt` with upstream's `pp_exact` bracketing and `read_part` (with no slots it is
    `sp.run`);
  - generate mode: upstream's local `max_new`, the context-room clamp, `--lookup-chain`, `ver.set_spec_q`,
    `aux_cpus::pin_current_thread()` in the adaptive threads, `fclose(hashes)`, the "elastic K/V" summary line beside
    the fork's 4070-tier lines;
  - `relend` (the `--vram-elastic` `VRAM` command) now resets the fork's `lend_first` / `lend_first_now` (§4).
- **Not taken:** the PCIe probe and PcieAuto's arming, the forced EAGER block, the per-part refill helpers and #1237
  inside them, upstream's generate-mode prefill loop (the fork's wave lanes), upstream's generate-mode `adapt` lambda
  (the fork's carries #463's trace and `adapt_rounds` from 0.1.38), `ple_inflight` 256, the `--dump-routing` writes the
  fork had replaced, the watchdog's `release_gpu_waits` + `abort`, the #224 `_Exit`.
- **The refusals and defaults of §2** (tail skip, CPU share, `--adapt-async`, `--kv-grow`, `--vram-elastic`,
  `STRATA_FS_SLOTS`) live here.

### `src/prefill/prefill.cpp` (55), `include/strata/prefill/prefill.hpp` (3), `kernels.hpp` (2), `kernels.cu` (1), `moe_mmq.hpp` (1) (`notes\prefill.md`)

Resolved in place from the fork's side, then audited with `git diff --no-index` against the fork's side file (every
added line is a listed port) and against upstream's file filtered to lines upstream added since the base (every
dropped line is in "not taken").

- **Kept from the fork:** `RING_MAX = 512` (#103; upstream's 1024 dropped), `split_min()` (#119), `split_routed_max()`
  (#142), the expert order (#41), `fused_ring()`'s any-layer rule (0.1.38: `220e0e8` not taken), the Stager's
  always-BlockingSync `dma_done` events (#35 D4), the unconditional `issued.notify_all()`, `SplitTier`
  (#32/#35/#133/#142/#208), `stage_cells` and `take_stage(kv_end)` (#122), the #31 PLE read-ahead, `stream_all`, the
  copy issuer, `split_row_layout`, `upload_tables` / `order_4070` / `mark_routed`, the split finish (#133 / #181 /
  #183), the sub-product-sized `STRATA_DBG_NAN` checks, the #35 D7 wave publish, the wave rule for `ple_prev`, the
  timeline spans.
- **Ported from upstream onto the fork's code:**
  - the Stager's pp-opt batch (claim runs of `batch` jobs, `copy_blobs`) wrapped around the fork's per-job walk: #142
    skip flags, the wait chain `wait(i - kRing)`, 0.1.38's first-generation wait, the spans, and the fork's three job
    kinds. Upstream's loop would have memcpy'd from null for the fork's #11 NVMe jobs. The batch is 1 on D2x;
  - `--aux-cpus` pinning in the stager, issuer and PLE threads (Linux only);
  - the ring's byte budget (`ring_bytes`, `ring_budget_slots`, `emb_reuse_account`) behind `STRATA_RING_BYTES=1` (§2);
  - one buffer count, `bytes_needed_impl(g, ss, chunk, kv_end, src)`: the fork's corrected count unconditionally
    (upstream gates its identical fix on `STRATA_RING_BYTES`; taking the gate would revert the fork to the old
    over-count with the switch off), plus upstream's `hc_pad` stride, `prompt_f16()`, `fused_layout(T, src)`, and the
    fork's #38 `STRATA_PREFILL_BUFFERS` report. Init's own block is `bytes_needed(g, ss, chunk, src != nullptr)` (the
    same bytes on D2x, which has a source);
  - `bytes_needed_owned` (the fork's one-block count rounded up to 2 MiB) and `bytes_needed_no_ring`;
  - `take_stage`: the fork's `pages` plus upstream's K8V4 hybrid branch;
  - `release()`: both sides' teardown; the carve's diagnostic messages (free / total MiB, the failing buffer);
  - `m.bo = m.emb` (#1454, §2);
  - upstream's KV prefetch (`STRATA_KV_PREFETCH=1`), **never in a wave** (`!m.wave`): the prefetch of layer l+1 would be
    queued before this lane waits for the other lane's chunk to publish layer l+1, whose host-copy appends it would miss;
  - #789's `defer_shared = !stream_all && !split_l && stream_ahead_enabled()` (off by default, §2);
  - the CPU prefill share: `cpu_maybe` with `!split_l` (in place of upstream's `!m.pp`), its RAM-short fallback, the
    auto arms' events, the D2H copy, the measured share; CPU candidates exclude 4070-tier experts (`peer_res >= 0`:
    their host pages are released); the CPU block is laid out last by `split_row_layout(m.cnt, at, on_cpu, m.off)`; the
    MMQ bounds end at `T * K - rows_cpu`; the CPU rows go into `Dm`'s tail before the fork's finish;
  - `blob_stable` for every blob the stager or a DMA reads later (the same pointer unless `STRATA_IO_PREFETCH`, Linux);
  - upstream's layer-split pipeline: `run()` = `run_impl()` + `drain_pipeline()`; `run_wave` calls `run()` on both
    lanes, unchanged;
  - `draft_kv(..., const char** why)`, `--mtp-hnorm stream`, the #579 `progress_at` texts inside the fork's "router
    sync" span, the `STRATA_DBG_FEAT` / `STRATA_DBG_ROUTE_OVERRIDE` hooks, `fused_only` (`mmq_plan().fo`), S23's
    `steered && !cv_fused`;
  - `PrefillStats`: both sides' fields. `merge_lane` (the wave) adds `experts_cpu` and keeps the larger `cpu_share`.
- **Headers:** `kernels.hpp` / `kernels.cu` hold the fork's #181/#183 wire functions and upstream's peer kernels
  (compiled, uncalled). `moe_mmq.hpp` keeps the fork's `warm()` (#30) and one comment with upstream's Q6_K / HIP
  wording plus the fork's IQ1_M tile (#169). Upstream's `moe_combine` float4 kernel is auto-merged (§2).
- **Not taken:** PeerPrefill and everything on it, #954's backstop, `STRATA_MMQ_RESIDENT_SORT_NE`, the owned ring carve,
  the `stager_sleep()` arm (§2). **Residue removed:** a duplicate `bool fused_bufs` (compile error), `helper_`,
  `single_chunk_`, `bind_stage_helper`, `HelpScope`, `next_->single_chunk_` at the hand-off.

### `src/core/*`, `src/platform/direct_file.cpp`, `src/ngram/ple_reader_test.cpp` and their headers (`notes\core.md`)

- **`verify.cpp` (12):** resolved as "upstream's file + the fork's delta"; `git diff upstream/main` on it now shows only
  the fork's ~100 lines: the Q2_0 `x_scales` path, `pcie_share_`, H2 `moe_hit_merge_mapped`, #49 ban sampling,
  `ms_launch` / `ms_tail`, the three-argument scratch size, the spans.
  - the MoE input quantize: a Q2_0 layer always runs the fork's `quantize_q8_1_rows_scaled` (it alone writes `hit_xs_`);
    otherwise upstream's `if (!qdedup && !q8_ffn)`. `STRATA_QFUSE` and `STRATA_VERIFY_QDEDUP` are opt-in;
  - the `grouped` lambda: upstream's `dst_buf` plus the fork's `x_scales` before `gy`;
  - the expert paths: upstream's structure (the zero-doorbell path into `parts_out`, the plan copy inside `post`, the
    `sh_stream` join, `STRATA_LFUSE`). In its non-ar branch: the fork's `if (pcie_share_)` block (skipped at
    `--pcie-frac 0`) and H2 `moe_hit_merge_mapped` when there is no device plan and no `remote_opt_`;
  - **the fork's spans re-homed** into upstream's new `run()`, because git had dropped the fork's `run()` prologue into
    the new `void stage_inputs`: "verify window", "verify capture" (after `refresh_ar()`), "ple gather" around both
    `gather_batch` sites, "verify stage", the GpuClock "verify graph";
  - the tail: upstream's #871 `h_plan_err_` check, the `copy_used_`-gated sync, `collect_profile()` over `[lb_, le_)`;
    the fork's `ms_tail`;
  - head sampling runs when `sampled || hist_d_ || sampling_.ban || sampling_.logit_bias`;
  - `commit`: upstream's `one_token_self_commit()` early-out with the fork's "commit graph" span.
- **`expert_source.cpp` (9) / `.hpp` (4):**
  - `load_experts_gguf(..., threads, const uint8_t* skip = nullptr, bool unbuffered = false, const std::atomic<int>* ready
    = nullptr)`: the fork's parameter first, as in 0.1.38;
  - `ArenaExpertSource::open`: upstream's `STRATA_DEFERRED_REGISTER` (opt-in) now also needs `!defer_load &&
    pin_for_cuda` (the Deferred constructor registers its slices for CUDA, which a pageable arena must not, and a
    placement-first arena is read later by `load_rest`);
  - `close`: both sides' teardown; after it both the fork's host-copy functions (paired swaps, exclusive placement) and
    upstream's `prefetch` / `release` (no-ops on Windows);
  - the dispatch: the fork's 4070 claim **first**, then upstream's `dec_batch` / `device_plan` / `gpu_zeroes_hits` /
    `any_cpu` (§4); both sides' counters (`offload_entries`, `peer_entries`, `tier_entries`); the fork's `run_jobs` /
    #95 second run / spans with upstream's `njobs > 0` guard;
  - the tail skip's loop spares 4070-held experts (§2), and the fork's NVMe miss loop tests `kind[i] != -1` (§4).
- **`mtp.cpp` (2):** upstream's #783 `draft` structure (every step captured up front, `stage_source_R`, `max_steps`
  clamped by the context's end; the `min_p <= 0` path launches the round and every step back to back; the `min_p > 0`
  path launches step by step with a spin wait). D2x passes `--spec-min-p 0.5`. Ported into it: the #33 GPU spans ("mtp
  round", "mtp step") and the #44 D4 `on_draft` callback (per draft with `pj >= min_p`; every draft on the `min_p <= 0`
  path). Not taken: `prefetch_ple` (§2). `STRATA_MTP_CATCHUP_ALL`'s default changed (§2).
- **`pinned.hpp` / `pinned.cu` (1 each):** the fork's constructor, its `= delete` guard for upstream's argument order
  (#56) and `reserve_only`, plus upstream's #285 `Deferred` constructor and `register_slices`. Port: upstream's
  `reserve(..., std::string{}, 0, ...)` call passes `true` in the fork's `allow_large_pages` slot (§4).
- **`remote_experts.cpp` (1):** the fork's three-argument `native_expert_scratch_bytes(CAP, FF, H)` plus upstream's
  `meta_bytes`. **`device.cu` (1):** the fork's `(void) allow_display_sm89;` and upstream's gfx906 block (HIP only).
- **`direct_file.cpp` (2):** the fork's `io_threads(issuers, N)` (#139) with upstream's
  `aux_cpus::pin_current_thread()` in each issuing thread; the fork's #185 per-thread buffers untouched.
- **`ple_reader_test.cpp` (2):** the fork's `--prefetch` check (#44 D4) first, then upstream's `--make-table` /
  `--ram-time`.

### Kernels: `pool.*`, `q2_avx2.cpp`, `expert*.hpp`, `expert_layout.cpp`, `iq_kernels.cu`, `sampler.*`, `ngram.cpp` (`notes\kernels.md`)

- **`q2_avx2.cpp` (6): upstream's file.** Upstream added the same AVX-VNNI Q2_0 kernel the fork had, with the same
  arithmetic (§2). Kept from the fork: `q2_0_gguf_rows_multi_avxvnni`, **re-homed** as a one-line wrapper of
  `q2_0_gguf_rows_multi_avx2_v(true, ...)`. Upstream's `q2_0_gguf_rows_multi_avx2` now picks VNNI itself. Two follow-ups
  outside the file, done by the orchestrator: `tests/xeno/q2_isa_parity.cpp`'s AVX2 arm calls
  `q2_0_gguf_rows_multi_avx2_v(false, ...)` (otherwise it compared VNNI with VNNI), and `tools/check_isa_guard.py`'s
  `WIDE_PUBLIC` names the wrapper.
- **`expert_layout.cpp` (2) / `.hpp` (1): one `cpu_avxvnni_ok()`.** Both sides defined it; two definitions would not
  link. Kept upstream's body (`STRATA_NO_AVXVNNI=1`, an older-CPU build or `STRATA_FORCE_ISA` cap -> false; needs
  `cpu_avx2_ok()`; CPUID 7.1 EAX[4]; every compiler) with the fork's `STRATA_FORCE_AVX2=1 -> false` first. The fork's
  `cpu_expert_isa()` (`INFO cpu_isa=`) and upstream's `cpu_avx512bw_ok()` both stay. `q2_rows_any` is upstream's
  dispatch; the fork's explicit VNNI branch is gone because upstream's AVX2 entry does the same with the same bits.
  `STRATA_FORCE_AVX2=1` now also turns off upstream's new VNNI forms of the IQ kernels (the same direction, further).
- **`expert.hpp` (1):** the fork's wrapper declaration and upstream's AVX1 rung (#1699).
- **`pool.hpp` (2) / `pool.cpp` (8):**
  - upstream's `ThreadAffinity pin_current_thread(int)` / `restore_thread_affinity` replace the fork's (= the base's)
    `long long` mask API (#626, §2). The fork's `set_worker_priority` / `set_current_thread_priority` stay. Every caller
    compiles against the new type (session.cpp, generate.cpp:1932, secondary_runner.cpp:210, the affinity tests);
  - the constructor: `ExpertPool(n_workers, pin, host_works, int spin_us = -1, PoolAffinity affinity = All, int tasks =
    0)`, plus an overload in upstream's order (§4);
  - the Windows topology keeps the fork's unconditional stable sort by EfficiencyClass (xeno #2). Upstream's #642 sorts
    only on a hybrid CPU; on a one-class CPU the fork's sort is a no-op, so the orders are the same. The fork's
    "query failed" fallback now enumerates processors per group as upstream's does;
  - the worker start: upstream's `note_owned_thread()` and `pin_this_thread(core, i)`, then the fork's priority;
  - `drain` modes 5/6: the fork's `native_rows` (#147), with upstream's `q2_native_kernels(type)` (a Q2_0 layer without
    AVX2 runs ggml-cpu) ported into `native_rows` and `native_quant`. On D2x's CPU the branch is the same as before;
  - `run_split_multi_native`: the fork's #147 one-phase block with upstream's `phase_tasks(...)` counts (= `3 * threads`
    by default, as the fork had).
- **`iq_kernels.cu` (7):** `q8_1_store` with upstream's #606 finite `d` / `ds` and the fork's `scales` / `hx` (with both
  null it is upstream's q8_1 exactly; the fork's CPU-order rounding now divides by the clamped `d`, identical for every
  finite block). `launch_gu` / `launch_down` are upstream's (#783 PR-h) with the fork's extra `nullptr`.
  `native_expert_grouped` is upstream's structure with the fork's Q2_0 CPU-order path inserted: gate/up
  (`q8_entry_hx_kernel`, `native_gu_q2_kernel`), the two-kernel SwiGLU when `sw_v1 || cpu_order`, the fork's
  `native_down_q2_kernel` branch first. On CUDA the AMD layouts never run (`exp_mode()` is 0 except sm_70 with
  `STRATA_SM70_TABLE=1`). Upstream's opt-in `s26_launch_l` was fixed to the fork's kernel signatures (§4).
- **`sampler.hpp` (1) / `sampler.cu` (3):** `SamplerParams` holds the fork's `ban` / `ban_vocab` (#49 S4) and upstream's
  `logit_bias` and `gumbel`. The kernels skip a banned id first, then apply the bias. The greedy launch: the fork's
  `if (p.ban) sampler_greedy_kernel<true>`, then upstream's cluster argmax only when `!p.logit_bias`. With no bias and
  no Gumbel the values are the fork's as before.
- **`ngram.cpp` (3):** both read-aheads, on the same `PleReader`: the fork's #44 D4 `PleTable::prefetch(rows, n)` and
  `cache_hits()` (Direct mode with a row cache, up to 32 tickets, drained first by `gather_batch`), and upstream's
  `a36be1db` `prefetch_rows` / `wait_prefetches` (16 one-token slots, a fast path in `gather_batch`). `close()` calls
  `drain_ahead` then `wait_prefetches`. Upstream's format-table decode (`fmt->dequant`, `ce62f505`) replaces the
  fork's unchanged-from-base `fp8` / `q5_0` decode. `Ahead::raw` is sized `n * rb`, so the new BF16 / Q8_0 / Q5_1 rows
  fit.

### `serve/server.py` (68), `serve/frontend.py` (5), `serve/test_server.py` (5), `serve/telemetry.py` (2) (`notes\serve.md`)

- **`Service.run` / `_run` (hunks 28-40): the fork's skeleton,** written by hand as one text (the auto-merge had
  interleaved upstream's multi-pass loop into the fork's single `_generate` call). Kept: `run` -> `_generate` (#123 /
  xeno #49 S3 budget legs, #199 cycle close, `_opening`, the drained tokens past a cut, `_merge_legs`), the loop guard,
  the #200 carry settle (which now also excludes `looped` / `repeated` ends), the spans, `tok_times`, the history
  store, `timing_report`, `cjk_chars`, `stats`, `last_stop_reason`, `stop_detail`, the `RequestGate` priority.
- **Ported from upstream onto it:**
  - `run()` is upstream's keep-awake wrapper (#1727) around `_run(..., force=None)`;
  - the budget read after the defaults merge, the #984 `ANSWER_RESERVE_MIN` cap;
  - `OutputParser(..., recover=...)`, `parser.finish(finish)` (#1058);
  - **stops:** upstream's `stop_strings()` + `StopMatcher` + `cut()` replace the fork's `StopSequenceFilter` (the same
    feature; upstream's is the superset: OpenAI `stop`, a 400 for a bad value). The fork's `finish = "stop"` and
    `last_stop_reason = "stop_sequence"` stay;
  - #606 repeat stop (on by default since 2026-10-12, §2), `reasoning_tokens`, the `KEEPALIVE_S` ping, `latencies.observe`, #588
    `pcie_share` (`"offloaded"` is summed by `_merge_legs`), #465 per-request status;
  - #1619 queue keep-alive: `self.fifo.acquire(priority, timeout=QUEUE_BEAT_S)` in a loop; a client that leaves while
    queued gets `finish: "cancel"`;
  - **forced tool calls (#537), #1053 `reasoning_close_retry` (opt-in) and #728 "recover" now all go through
    `_generate`** with one cut per request (`state["cut"]`): a retry, a forced opening or a recovery after a budget leg
    cannot happen (upstream's loop could chain passes). A budget or cycle cut writes `close + force`. Retry plus a
    forced call writes `REASONING_CLOSE + force` (upstream's loop would let the model write one answer token first).
    #728 recover keeps upstream's checks (coverage at a clean point every `LOOP_CHECK_EVERY` tokens, no pictures, the
    xhigh sentence) and sampling rule, continues once from `focused_recovery_prompt(...)` with `max_new - written` tokens
    and counts `reasoning_recoveries`;
  - **not ported:** upstream's per-pass `segments` / `request_stats` (#615). The fork's `engine.last` identity and
    `_merge_legs` decide the DONE figures; `request_stats` stays defined, unused.
- **Other hunks, with the behaviour each one moves:**
  - `_pump(self, proc=None, lines=None)`: the fork passes its process and queue (#344 review); upstream's test calls it
    with none;
  - `restart(self, tries=3, full=None)`: upstream's retries (PR #637) with the fork's #59 floor-breach rule inside.
    `serve/replicas.py` calls `restart(tries)` positionally. **Visible change:** a plain failed restart now retries up
    to 3 times, 15 s apart;
  - `death_note`: upstream's 256 KiB read for the `issue #29` line (#1705);
  - `_parse_done`: upstream's DONE field 16 `offloaded` (#588), then the fork's `STATS` / `prefill_points`;
  - `sampling_keys`: `aux_cpus`, the fork's `ckpt_at=`, upstream's `ckpt=0` and `pin=N`, the fork's `ban_key`, the bias
    key;
  - the quiet-wait heartbeat: upstream's #1317 stall check with the fork's `self.QUIET_S` in place of the literal 10.0
    (the auto-merged wait now polls every 0.5 s, so the fork's `wait >= QUIET_S` rule would have sent a heartbeat every
    poll);
  - `close()`: upstream's guarded `stdin.close()` and the fork's #48 rule (stdout closed only when the pump has ended);
  - `ensure_loaded`: the fork's `EngineDied` (529) with upstream's vision unload; `EngineStuck` keeps its own handler;
  - **`prepare(self, messages, tools, kwargs, max_new=None, priority=1, force=None, req=None, count_only=False)`:**
    `prepare(..., None)` now means upstream's "the rest of the context" (upstream's test expects the room check); the
    fork's count (`_count_tokens`, xeno #46) passes `count_only=True`. `_fit` takes upstream's context rule
    (`known_ctx` after a failed restart, `EngineStarting` only while starting, the message prefix). The combined
    embeddings file is written by upstream's `write_temporary`, with the fork's cleanup restored (§6a);
  - `prepare_carried(..., force=None)` passes `force=` and `req=` on; a resumed reply generates on from the cut prompt;
  - GET: both `/metrics/requests` (fork) and `/metrics/prometheus`; `/health` is the fork's (503 `engine_exited`,
    `loops_stopped`, `thinking_cycles_closed`, `degraded`) with upstream's `reported_ctx()`;
  - POST: the fork's `_do_post` (with the `/classic/` strip) and `timeline.span("http request")` around upstream's
    `BadBody`; all the fork's routes, then `/config`, `_control_body()` for `/unload` / `/load`, the fork's 403;
    `/v1/vram`; the fork's `cache_slot` validation, now on `/v1/responses` too; upstream's malformed-shape 400 then the
    fork's `(EngineDied, EngineStarting)` -> 529 on `/v1/messages`, 503 elsewhere;
  - `_openai` / `_anthropic`: the fork's `strata_skill` / `strata_agent`, `cjk`, `vision=`, `side_effort`,
    `prepare_carried`, `with_slot`, `_meta`, `_think_budget`, `_opening`, MCP coding tools, with upstream's
    `bias_api`, `_no_local_images`, `tool_choice`, `forced_call`, `stop_strings` / `check_request_sampling`, #592; both
    event builders pass `force=force`;
  - `SAMPLING_PRESETS` (#206) and upstream's `lift_reasoning_effort_arg` (#1641), `SAMPLING_ALIASES` (#1819); the preset
    first, then the alias folding;
  - `main`: upstream's lazy vision encoder, `effort_end_args`, replicas, `api_key_of` with the #569 warning, the slot save
    path, `tool_call_recovery`, `reasoning_loop_recovery`, with the fork's `--ban-ids` (cjk guard), `loading_server`,
    history / checkpoints / UI paths, `reasoning_carry`, `monitor_gpus`. With replicas the cjk `--ban-ids` is not passed
    (not D2x).
- **`frontend.py`:** both `efforts()` (fork) and `_detect_caps()` (upstream); the fork's `_tool_result_content` (xeno
  #46); **the fork's billing-header strip wins** (§2); upstream's `_check_parts` on the system and each message, with
  `"document"` added to the kinds it accepts (the fork reads documents, xeno #46; without it `_check_parts` would log
  them as not read); upstream's `tool_arguments` / `_tool_list` (auto-merged); `anthropic_to_messages(..., vision=None)`
  (was the fork's `False`): `None` keeps a tool result's image so the server's `_note_unreadable_tool_images` notes it,
  `False` (the fork's handlers pass `svc.vision is not None`) keeps the fork's note. `ChatTemplate.render`'s #843 filter
  keeps a turn marked `"_carried"` (§6a).
- **`telemetry.py`:** upstream's `_stop` event with the fork's `_storage_facts` thread and per-card series;
  `_stop.wait(self.interval())`; upstream's no-disk guard ported to the fork's per-disk reading.
- **`test_server.py`:** the fork's `ANSWER`, both sides' test classes (`RequestHistory`, `MonitorGpus`, `HipEnvGuard`,
  `EndsInsideThinkingEngine`, `ReasoningCloseRetry`, `ReasoningBudget`, `EffortLevelsTests`, `WebSecurity`,
  `TelemetryNoDisks`); `StatusHandover.SlowRelease` keeps upstream's armed `release`; the fake `acquire` takes the
  `RequestGate` signature. `RepeatStop` (2026-10-12): `test_the_default_is_on` (256), the stop reported as the loop
  guard's (`stop_detail`, `loops_stopped`), and `test_a_run_the_loop_guard_cannot_see` (a run inside a tool call's
  arguments, which only #606 ends); the "off" case uses a 300-token run and the alternating case 400 characters (upstream's
  1,000 and 800 are inside the loop guard's 512-character window). `ClaudeCodeBillingStamp`: 5 pin-specific tests
  skipped with the reason "the fork strips the Claude Code billing header instead of pinning it (xeno #6)"; the 2 that
  also hold under the strip run.

### Files resolved by the orchestrator (`notes\orchestrator.md`)

- `CMakeLists.txt` (3): the engine link keeps the fork's `strata_secondary_compute` and adds upstream's
  `exchange_storage` / `memory_guard` tests; `strata_prefill` keeps the fork's `split_plan` / `kv_stage_plan` and adds
  upstream's HIP `wmma_gemm`; the fork's xeno test block and upstream's `dequant_q5_1` / `router_dot` / `pool_tasks` /
  `iq_avx2` / `iq_avx1` / `q8k` tests.
- `setup.py` (4): `DRAFT_VOCABS` keeps the fork's `thai` default (#55 W7) and adds upstream's `fr`; the docstring,
  `ok()` text and `--draft-vocab` help name both. `tools/draft_vocab.py` (1): the fork's `thai` in `--add`'s help and
  upstream's `--corpus` / `--coverage` (#597).
- `AGENTS.md` (1): upstream's "Contributing a change or report" section in the upstream part, above the fork's rules.
- `docs/SECOND_GPU.md` (1): the fork's peer note and upstream's server paragraph for `--peer-device`.
- `README.md` (17): the README section below.

## 4. Semantic hazards checked beyond the markers

- **`Prefill::bytes_needed`'s fourth parameter:** the fork's is `int64_t kv_end = 0` (#122), upstream's `bool src =
  true` (`a26fcbcd`). A bool in the fork's slot converts silently to `kv_end = 1` and sizes every loan wrong. The final
  form is `bytes_needed(g, ss, chunk, int64_t kv_end = 0, bool src = true)` **plus** an exact-match overload
  `bytes_needed(g, ss, chunk, bool src)`, so upstream's spelling picks the overload; a fourth argument of type `int` /
  `size_t` is ambiguous (a compile error, never a conversion). Every upstream call in generate.cpp passes five
  arguments (`/*kv_end=*/0, /*src=*/srcp != nullptr`). The merged header declares `kv_end` before `src`, the order those
  calls need; a header with `(..., bool src, int64_t kv_end)` would compile every one of them silently wrong
  (generate.md and prefill.md agree on the final form).
- **`ExpertPool`'s constructor:** the combined form puts the fork's `spin_us` before upstream's `affinity` and `tasks`.
  Upstream's `pool_tasks_test.cpp` calls `ExpertPool(1, false, true, PoolAffinity::All, tasks)`, and an enum class does
  not convert to `int`, so an overload in upstream's order delegates with `spin_us = -1`. No ambiguity: its fourth
  parameter has no default and `PoolAffinity` is not constructible from an int.
- **`load_experts_gguf`:** a call in upstream's order `(..., readers, unbuffered, rp)` would not compile (bool to
  pointer), so a missed caller shows at build. Callers checked: `tests/core/expert_layout_test.cpp`,
  `ArenaExpertSource::open`.
- **`PinnedArena`'s deferred constructor:** upstream's `reserve(..., std::string{}, 0, ...)` against the fork's
  `reserve(..., bool allow_large_pages, shared_file, ...)` would not compile; it passes `true` (a deferred arena is
  CUDA-registered, the fork's `pin_for_cuda = true` default).
- **Two definitions of one function:** `cpu_avxvnni_ok()` (C++, would not link; one kept, §3). Python's later `def`
  silently replaces an earlier one: `_foreign_origin` was defined by both sides. Upstream's predicate is kept (it also
  honours `trusted_origins`, and upstream's `_own_page` / `_no_local_images` use it); the fork's one caller (`/unload`,
  `/load`) sends its 403 itself. **Behaviour:** a trusted origin may now load and unload. A duplicate `import
  contextlib` was removed. In `test_server.py` the fork's second plain `release` would have replaced upstream's armed
  one; dropped.
- **Python signatures changed on both sides:** `_pump`, `restart` (positional `tries` from `replicas.py`, keyword
  `full=` from every fork caller, checked by grep), `prepare` (`priority` 5th positionally; `force=` / `req=` by keyword,
  `serve/subagent.py` and the tests included; `None` vs `count_only`), `RequestGate.acquire(priority=1, blocking=True,
  timeout=None)` and `locked()` (upstream's `vram()` and #1619 call `fifo.acquire(timeout=...)`, a TypeError at run time
  otherwise), the 7-tuple `spawn` (`lazy`, `cpus`) unpacked as `exe, args, cwd, log, env, *rest`, `self.proc` checked for
  None after a failed start, `slot_q = getattr(self, "slot_q", [])` for engines made with `__new__` in the fork's tests.
- **The verify window's dispatch order:** upstream's `any_cpu` reads `kind[i] == -1`, so the fork's 4070 claim must turn
  the 4070's rows into kind 2 before it; otherwise activations are quantized and misses prefetched for experts the 4070
  computes. `gpu_zeroes_hits` (#783: the host no longer zeroes kind 0/1 rows) is safe with the fork's H2 path:
  `merge_mapped_kernel` (`s2_expert_grouped.cu:710`) writes `hit_out` into hit rows and never reads their mapped row;
  kind 2 rows still get the memset and `finish` overwrites them. `kind == 2` is shared by the fork's 4070 tier and
  upstream's peer; the 0.1.38 refusal still keeps them apart.
- **The tail skip's `-2` kind:** the fork's #11 NVMe miss loop used `if (kind[i] >= 0 || ...) continue;`, so a
  tail-skipped entry (-2) would have been read from NVMe and then zeroed. It now tests `kind[i] != -1` (capacity mode
  only). The other `kind` tests were checked: the CS-T prefetch and `any_cpu` use `== -1`; the row loop tests `== -2`
  before `>= 0`; the 4070 claim's `kind[i] < 0` cannot see a -2 any more.
- **`return 1` inside the serve loop:** the fork's rule is that nothing in the loop unwinds `main` (#53, #45, #203). 13
  `return 1` lines that git or upstream brought into the loop body (batch pump / `BSTOP` drain, `adapt_tick`, the parked
  conversation's K/V growth and restore, `kvg_trim`, `--pipeline-windows`, batch-MTP admission, the decode loop's K/V
  growth) plus the ones in `SAVE` / `RESTORE`, `kvg_ensure`, `logit_bias` and the checkpoint are `return serve_fatal()`.
- **`relend`** (upstream's `VRAM` command) moves `pf_parts[0].first`; the fork's serve loan reads `lend_first` /
  `lend_first_now`, set once at set-up. It now resets them, or a shrink would leave the loan laying buffers past the
  cache's end (single-GPU configs; `--vram-elastic` is off with the 4070 tier).
- **The watchdog's order** (`arm_end_engine` after upstream's allowances, §3).
- **Auto-merged code built on things the fork does not have:** PeerPrefill residue in prefill.cpp (`fused_bufs`
  duplicate, `single_chunk`, `bind_stage_helper`, `HelpScope`), found by a residue grep over 30 identifiers: only
  comments remain. Upstream's opt-in `s26_launch_l` called `swiglu_entries_kernel` / `quantize_q8_1_kernel` with
  upstream's signatures: now `(..., nh, false)` and `(h, hq, nullptr, nullptr, nh)`. Upstream's new
  `native_expert_bench.cpp:96` called the two-argument scratch size: now `(NE, FF, H)`.
- **Identifiers:** every `o.<field>`, `drive.d.<field>` and member call in generate.cpp exists in the merged headers;
  the prefill group checked the cross-file names it uses (`copy_blobs`, `cached_share`, `blob_stable`,
  `run_split_multi_native`, `ActQ`, the kernels' `ld*` parameters, `aux_cpus::*`) by grep.
- **Defaults that would move D2x:** each listed in §2, with its switch.
- **MSVC C1061:** the option parser's third `else if` chain now holds ~109 links (88 in the fork; the limit is 128
  nested blocks). It compiled.
- **Line endings:** `q2_avx2.cpp` was written from the LF side file; the others kept CRLF. Git normalises on commit.
- **SYCL:** `sycl/src/prefill/prefill.cpp` (upstream's own copy) does not match the fork's `prefill.hpp` (`carve(T,
  alloc, kv_end)`), and `sycl/` uses the old `long long` `pin_current_thread`. A SYCL build of the fork does not
  compile; it did not before this merge either (the `carve` signature). Not built here (`STRATA_ENABLE_SYCL` is OFF).

## 5. Verification

- **Build:** `build-212.cmd --target strata` (151/151), then the whole tree (319/319), CUDA 13.3, sm_89 + sm_120: no
  error at the first build (`merge-212-record\build-strata-1.log`, `build-all-1.log`). This includes the three
  orchestrator fixes outside the conflicted files (`native_expert_bench.cpp`, `check_isa_guard.py`,
  `q2_isa_parity.cpp`).
- **ctest** (serial, no `-j`, #140): **160/161** in ctest's count, which is 158 passed and 2 skipped
  (`merge-212-record\ctest-212.log`, 203 s). 0.1.38 had 107 tests.
  - **Failed:** `ple_parity`, the known one: `required PLE table is missing or incompatible:
    ../../Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf` (the Q2_0 file was deleted; it fails on `main` too).
  - **Skipped (exit 77):** `expert_multi_test` (no AVX-512; it failed in the 0.1.38 report, upstream now marks it
    `SKIP_RETURN_CODE 77`), and **`isa_guard_check`** (77 = the tools it needs are missing here). So the
    `WIDE_PUBLIC` entry added to `tools/check_isa_guard.py` was **not exercised**.
  - **Passed, covering this merge's resolutions:** `xeno_q2_isa_parity` (with the AVX2 arm now really AVX2),
    `xeno_cpu_dispatch_vnni`, `xeno_cpu_dispatch_forced_avx2`, `q2_bitplane_parity`, `pool_tasks_test` (upstream's
    constructor order through the overload), `pool_affinity_test`, `aux_cpus_test`, `iq_avx2_parity` (+ `_iq3s_mt1`),
    `iq_avx1_parity`, `router_dot_parity` (+ `_avx`), `q8k_quant_parity` (+ `_avx`), `native_grouped_parity`,
    `iq_multi_parity`, the `native_expert_parity_*` set, `sampler_parity` / `_one_block` / `_old`, `logit_bias_test`,
    `spec_verify_parity`, `spec_prob_test`, `coupled_draft_test`, `kv_stream_parity`, `kv_hybrid_parity`,
    `batch_read_test` / `batch_rows_test` / `batch_groups_test`, `draft_kv_plan_test`, `message_boundary_test`,
    `conv_cache_test`, `share_rules_test`, `host_limits_test`, `exchange_storage_test`, `memory_guard_test`,
    `ple_reader_selftest`, `decode_cluster_parity`, `gdn_rec_parity`.
  - **The fork's own passed:** `gpu_4070_iq_parity_layer{0,1,8,12,35}`, `gpu_4070_iq_graph_formats`,
    `gpu_4070_transfer`, `xeno_combine_split_parity`,
    `xeno_combine_partial_parity`, `xeno_combine_bf16_parity`, `xeno_combine_q8_parity`, `xeno_frontier_combine_parity`,
    `xeno_gather_group_parity`, `xeno_hit_merge_mapped`, `xeno_pool_fused_parity` (+ `_l12`), `xeno_split_plan`,
    `xeno_kv_stage_plan`, `xeno_pool_spin`, `xeno_pool_core_policy`, `xeno_pool_rest`, `xeno_end_engine`,
    `xeno_direct_file_issuers`, `xeno_direct_file_thread_exit`, the `xeno_nvme_*` set, `xeno_secondary_budget`,
    `xeno_secondary_profile`, `xeno_exclusive_host_pages`, `xeno_timeline_core`, `xeno_timeline_gpu`.
  - `native_q2_pool_hit_parity` is built but not registered with ctest (the Q2_0 file is gone), so ADR 0001's bit-exact
    gate was **not run**, as at 0.1.38.
- **pytest `serve` + `tests/xeno`:** **1602 passed, 19 skipped, 0 failed** (5643 subtests, 8 min 14 s; `notes\serve.md`
  "Follow-up"). The skips: 5 billing-stamp tests and 3 request-accounting tests (reasons in §3 and below), and 11 that
  skip in their own code (no pack tokenizer, symlinks or modes on Windows, the `.cmd` wrapper, `serve/ui/node_modules`).
- **pytest `tools/test_*.py`** (`STRATA_GGUF_PY=C:/Strata-exp/llamacpp-169/gguf-py`): **650 passed, 11 skipped,
  1 failed** after upstream's `Refresh::test_fr_replaces_a_shipped_subset_and_back` was adapted (the fork names
  `data/draft_vocab.bin` "thai", xeno #55 W7; the test now refreshes to `thai` where upstream used `cjk`). The failure,
  `test_setup_draft_vocab.py::SmallCardNote::test_sizes_follow_the_shipped_subsets` (`KeyError: 'cjk'`), fails the same
  way on the pre-merge `main` (the 0.1.38 report). `merge-212-record\pytest-tools-2.log` is the run after the
  adaptation; `pytest-tools.log` the one before it (649 passed, 2 failed).
- **Test files changed to match the fork's behaviour,** beyond the conflicted ones (reasons per file in
  `notes\serve.md`). **To review before the commit:**
  - `serve/test_request_accounting.py` (upstream #615): 3 tests skipped, because the fork merges a budget
    continuation's DONE figures its own way (`_merge_legs`: the prompt read is the first call's);
  - `serve/test_fatal_recovery.py`: `/health` answers 503 `engine_exited` for an ended engine (fork `e9c69582`);
  - `serve/test_frontend.py`: `document` is read by the fork (xeno #46), so it is out of upstream's "left out" list;
  - `serve/test_reasoning_loop_recovery.py`: the default `repeat_stop_tokens` is 256 (0 at the merge commit), and the
    loop guard ends the passage loop (`stop_detail: "loop"`);
  - `serve/test_lifecycle.py` (the mock pump reports not alive, xeno #48), `serve/test_logit_bias.py` (the GEN line
    always names `cache_slot=0`, xeno #49 S7), `serve/test_media.py` (one distinct picture, not one encode call),
    `tests/xeno/test_count_tokens.py` (`count_only=True`; the staging test uses a request that fits),
    `tests/xeno/test_serve_protocol.py` (the fake `proc` gains `poll`), `tests/xeno/test_anthropic_stream.py`
    (`FakeService.run(..., force=None)`).
- **Not run:** a SAVE / RESTORE round trip on the fork's state, the CPU prefill share on the fork's walk, capacity
  mode (`strata-swift-capacity.json`), the opt-in arms of §2, a HIP / Linux / SYCL build. Gates on this merge:
  `/code-review`, `/simplify` and `/scrutinize` have **not** run yet (nothing in the record says so).

## 5a. Parity and speed on the served config

Same session, 2026-10-11, on `strata-flash-next-d2x.json` as served (every arm with its `--pool-priority 2
--process-priority 2`, the vision encoder on the 4070), the D2x server not running, nothing else CPU-heavy. Arms: `main` =
`run-main-8abbfa0\strata.exe` (sha256 `0936c394b5c96a64…`, the deployed exe) with the pre-merge server
(`C:\Strata-exp\src-main`, the same `serve/` as `8abbfa0`); the merge = `run-212\strata.exe` (sha256
`910652258eaadd37…`, a copy of `build-212`'s) with the merge's server. Scripts and data:
`merge-212-record\ab212.py`, `dec212.py` (a copy of `merge-137-record\pool_ab.py`), `run-gates.sh`, `gates.out`,
`ab212-hash-results.json`, `ab212-speed-results.json`, `dec212-runs\`.

- **Main-model and drafter state** (`ab212.py hash`: `STRATA_STATE_HASH=1`, `max_tokens=1`, a 9,932-token prefix and six
  parts of 470-2,564 fresh tokens that branch from it, boots main / merge / merge / main): **all seven fingerprints are
  identical in all four boots**, every field: `gdn`, `ple`, `tail`, `pooled`, `kv`, **`mtp`**, `stale`, `dead`,
  `pooled_full`, `ple_prev`. The drafter's prompt K/V, which differed at the 0.1.38 merge (#149), is identical here.
- **Claude Code's request shape** (`ab212.py speed`, boots main / merge / merge / main / merge / main / main / merge;
  greedy, 32 tokens a part, then a 400-token answer to a short prompt). Prompt read in ms, median of four boots each:

  | part (fresh tokens) | main | merge | merge - main |
  |---|---:|---:|---:|
  | prefix (9,932) | 7,732 | 7,744 | +12 |
  | 470 | 1,351 | 1,266 | -84 |
  | 735 | 1,604 | 1,490 | -113 |
  | 939 | 1,523 | 1,435 | -88 |
  | 1,304 | 1,710 | 1,618 | -92 |
  | 1,831 | 2,020 | 1,870 | -149 |
  | 2,564 | 2,434 | 2,326 | -109 |

  - **Every part is read 84-149 ms faster (5-7 %); the long prefix is equal.** The merge's last boot (run 7) was its
    slowest in every part and still within main's range.
  - **Replies are identical** in every part, in all eight boots (one distinct reply per part across both arms).
  - **Decode in the server:** the 400-token answer runs at 73.0 tok/s (72.6, 75.9, 73.5, 69.1) against 63.6
    (67.4, 64.4, 62.8, 62.2). Both arms take the same 197 windows with the same 204 of 285 drafts accepted and the same
    tier shares (`request rates` line), so the gain is time per window (5,506 against 5,936 ms in the first pair,
    27.9 against 30.1 ms a window). **Which stage moved is not measured:** the serve request lines do not split a
    window. A timeline of each arm would. The two arms also differ in their servers here.
- **Decode in generate mode** (`dec212.py`, V F F V, 256 tokens greedy, V = the merge, F = main): **outputs identical**
  on all three prompts. tok/s:

  | prompt | merge (V) | main (F) |
  |---|---|---|
  | n0-thai-net | 64.1, 65.1 | 63.0, 64.9 |
  | n0-py-async | 86.4, 76.5 | 77.0, *13.2* |
  | n0-long8k | 69.5, 70.9 | *37.7*, 70.5 |

  Two main runs were **disturbed** (in italics): their per-round counters are normal (`wait for rings` 17.2 / 16.5 ms,
  `pool` 14.7 / 17.1 ms), but the whole decode took 19.4 s / 6.8 s against 3.3 s / 3.6 s, a stall outside the rounds.
  Without them generate-mode decode is equal within noise. The CPU pool per round is 0.4-0.7 ms higher on the merge in
  thai and long8k (`dispatch detail`), inside the run-to-run spread of py-async (8.7-12.1 ms).
- **The PLE reads** (`ple reads` line of the same runs): the merge asks for **exactly 4,096 more rows per 256 tokens**
  on every prompt, and about 4,100 more of them go to the SSD: thai 4,952 -> 9,049-9,080 SSD reads, blocked 8.0-8.6 ->
  32.4-47.5 ms; py-async 5,680-5,704 -> 9,793, blocked 11.1-13.5 -> 34.8-37.1 ms; long8k 92,6xx -> 96,7xx, blocked
  unchanged (~1 s). This confirms §7's double-read hypothesis in its row counts: upstream's `prefetch_rows` asks for
  the window's 16 rows a token again beside the fork's #44 D4 read-ahead. Its cost (0.1-0.15 ms a token) is under the
  ABBA's resolution. **#213.**
- **Not measured:** the boot log's free-VRAM lines and the start-up time (#620's allocation order, the `--ple-io direct`
  probe); a timeline of either arm; the opt-in arms of §2.

## 6. What moved: commits and the exe

- Base: `main` `8abbfa02`; upstream `upstream/main` `61b3fb5d` (tag `v0.1.42`); branch `xeno/212-merge-0.1.42`.
- The merge commit: the commit that adds this report (parents `8abbfa02` and upstream's real `61b3fb5d`, not the
  graft); its id is in #212 and in the PR. The local graft (`refs/replace/09817da3`) is deleted right after it
  (`git replace -d 09817da3`) and is never pushed.
- Engine measured in §5a: `C:\Strata-expun-212\strata.exe`, sha256
  `910652258eaadd37b9e272f68040f7851c840f3c13502217c53c814482571eb9`, a copy of `build-212\strata.exe`, built from the
  merged working tree. No C, C++ or CUDA file changed after that build (checked with `find -newer` on the exe); later
  changes are Python, tests and docs.

## 6a. Bugs the merge introduced

**Auto-merge breakage found by the groups' sweeps before the first build** (brief rule 5; the evidence for each is in
the group's notes and §4):

- code that would not compile against the fork's side: the duplicate `fused_bufs` (prefill.md), `s26_launch_l`'s kernel
  calls and `native_expert_bench.cpp:96`'s two-argument scratch size (kernels.md), the deferred arena's `reserve` call
  (core.md), the two `cpu_avxvnni_ok()` definitions (kernels.md);
- behaviour the auto-merge broke: the fork's NVMe miss loop let the new `-2` kind through (`kind[i] >= 0`, core.md); the
  `return 1` lines in the serve loop and `relend`'s stale `lend_first` (generate.md);
- a test that stopped testing: `q2_isa_parity.cpp` compared VNNI with VNNI once upstream's AVX2 entry took VNNI
  itself (kernels.md).

The `bytes_needed` conversion, the 4070 claim's place before `any_cpu` and the watchdog's order are hazards the
resolutions were designed around (§4), not bugs found in merged code.

**Found by the test runs and fixed in code:**

- **A failed image staging left a partial `req-*.sve` file.** Upstream's `write_temporary` in `prepare` has no cleanup,
  which lost the fork's guarantee. It now unlinks the file and clears `embeddings.path` on any error (fork test
  `tests/xeno/test_count_tokens.py::test_failed_image_staging_removes_partial_request_file`). Introduced by the merge.
- **#200's carried reply dropped by #843.** Upstream no longer renders an empty assistant turn, and the carried cut
  reply is exactly such a turn (reasoning, no text): `serve/test_reasoning_carry.py` failed 3 tests. `prepare_carried`
  now marks the reply `"_carried": True` and #843's filter keeps it. (Exempting every reasoning-only turn instead broke
  upstream's `EmptyAssistantHistory` tests.) Introduced by the merge.
- **`_merge_legs` counted the first DONE twice when the continuation ended without a DONE line**
  (`engine_generated` 115 instead of 20 in upstream's
  `test_request_accounting.py::test_failed_continuation_does_not_count_the_first_done_twice`). **Not introduced by the
  merge:** a pre-existing fork gap that upstream's #615 test exposed. `_generate` now passes the leg-1 DONE object, and
  `_merge_legs(first, n_close, leg1)` adds nothing when `engine.last` is still that object.

**Review smells left as they are,** to keep the next merge small:

- code kept with no caller: upstream's no-P2P peer kernels (compiled), `frontend.pin_billing_stamp` and
  `BILLING_HEADER`, `request_stats`, `stager_sleep()`, `ple_ss_` / `set_ple_session` in mtp.cpp, PcieAuto (never armed),
  `Alloc::granule` (never set);
- stubs and pairs: `set_stage_helper` (a stub), `set_peer` beside `set_peer_tier`, the two `bytes_needed` overloads,
  the two `ExpertPool` constructors, two ways to ban a token (the fork's bitmap, upstream's bias of -inf);
- `cpu_expert_isa()` says "AVX-VNNI" from `cpu_avxvnni_ok()` alone; with `STRATA_AVXVNNI=0` at build or
  `STRATA_Q2_BITPLANE=1` the Q2_0 rows are not the VNNI ones (same bits in the first case, opt-in in the second);
- `sample_tokens_spec` (opt-in `STRATA_SPEC_PROB`) honours the fork's ban but does not check `ban_vocab == n_vocab` as
  `sample_tokens` does;
- the fork's Stager uses `std::atomic::wait` unconditionally, so a GCC 10 build does not compile it (upstream guards its
  own with `STRATA_ATOMIC_WAIT`, #1488); not a D2x concern;
- upstream's own: `serve/test_server.py` defines `class VisionTempFiles` twice (the first one's tests are shadowed);
  with `STRATA_QFUSE=1` a T=1 window neither self-commits nor runs the commit graph (`commit()` vs `record_window`,
  verify.cpp:775); `STRATA_ARENA_MMAP` returns from `open` before the fork's `path_` / `exclusive_` fields are set
  (Linux only); `RemotePrefillSource` forwards none of the fork's capacity-mode virtuals;
- `DisplayFloorRecovery::test_a_failed_degraded_start_keeps_the_configured_command` now takes 30 s (upstream's restart
  retries; the fork's test does not patch `RESTART_RETRY_S`);
- the `#11 materialize` line's odd indent in `expert_hit_run` is the fork's own, not the merge's.

## 7. What is left

- **Whether D2x moves to the merge's exe:** the developer's call, on §5a.
- **Measured: the window's PLE rows are asked twice on D2x (#213).** Upstream's `Verifier::run` issues
  `prefetch_rows` for the window's rows before the launch (verify.cpp:2014), beside the fork's #44 D4 read-ahead
  (`PleAhead::push` -> `PleTable::prefetch` during the draft), and `prefetch_rows`' slots do not see the fork's tickets.
  §5a: +4,096 rows and about +4,100 SSD reads per 256 tokens, +25-39 ms blocked on short prompts; the bits are the
  same. One read-ahead for one job is the fix, chosen by an ABBA. (verify.cpp's second site, :3057, is in
  `Verifier::pl_stage`, the `--pipeline-windows` path, which D2x does not pass; mtp.cpp's draft-time copy was not
  taken.)
- **Not explained yet: the server's decode gain** (73.0 against 63.6 tok/s with the same windows, §5a). A timeline of
  each arm splits the window.
- **ABBA arms not measured** (each upstream value behind its switch, §2): `STRATA_MTP_CATCHUP_ALL=0`,
  `STRATA_RING_BYTES=1`, `STRATA_PREFILL_STREAM_AHEAD=1`, `STRATA_PREFILL_CPU_SHARE=1` / `auto`,
  `STRATA_ROUTE_TAIL_SKIP=7`, `--ple-inflight 256`, `CUDA_MODULE_LOADING=EAGER`. The host
  pin (#626) has no arm; it is measured only inside the merge-vs-main ABBA.
- **Questions for the developer:**
  - upstream's PCIe probe + PcieAuto are unreachable without `--pcie-frac`; no switch restores upstream's no-flag
    default;
  - the host-thread pin (#626) moved a D2x default with no switch back;
  - ~~#606 `repeat_stop_tokens`: 0 or 256~~ - answered 2026-10-12: 256, reported as the loop guard's stop (§2);
  - helper caches (`--expert-cache-device1..3`, `--remote-expert-opt`, `STRATA_PREFILL_HELPER_SOURCE`) with the 4070
    tier, and upstream's helper GPUs with the NVMe tier, are not refused;
  - the billing-stamp tests are skipped (the fork strips the header, xeno #6).
- **Untested combinations:**
  - the CPU prefill share has never run on the fork's walk (expert order, `split_row_layout`'s CPU pass, the MMQ
    bounds at `T * K - rows_cpu`); `STRATA_PREFILL_CPU_SHARE=auto` on a single-GPU, non-split config exercises it;
  - disk sessions (`SAVE` / `RESTORE`): whether `conversation_snapshot_sources` / `restore` capture the fork's own state
    (cache slots `req_slot`, tail checkpoints, `shared_at`);
  - with both `ckpt_at` and `STRATA_CACHE_MESSAGE_BOUNDARY=1`, a `message_at` below `shared_at` is skipped (`to <= at`);
  - `STRATA_ROUTE_TAIL_SKIP=7` on D2x (the dispatch now spares 4070 experts; not run);
  - the Stager batch path (`k > 1`) around the fork's per-job waits;
  - `RequestGate` with #1619's 10 s re-tries: among requests of the same priority the order can rotate after a 10 s
    wait (priorities are kept);
  - capacity mode as a whole.
- **Checks not run:** `isa_guard_check` (skipped, tools missing); `native_q2_pool_hit_parity` (not registered).
- **Done before the commit:** the six test files adapted outside the serve group's list were reviewed (each matches a
  deliberate fork behaviour named in its comment); `docs/BLUEPRINT.md` has the revision-log row and the "Not yet in
  the baseline" row for this merge; §9 lists every new configuration name. Push and the PR are the developer's.
- **Gates:** `/code-review` (two axes), `/scrutinize`; `/simplify` is not meant for a merge (the diff is upstream's code
  plus resolutions, not new design), as at 0.1.38.
- **Follow-ups carried:** `tools/test_setup_draft_vocab.py` `SmallCardNote` (`KeyError: 'cjk'`, also on `main`);
  `ple_parity` needs a PLE table that exists; the SYCL tree does not compile against the fork's headers.

## 8. The classic web app

Upstream changed `serve/web/` in 5 files, +165/-11 (`app.css`, `app.js`, `favicon.svg`, `index.html`, `monitor.html`;
`git diff --stat 99f3dbd0 61b3fb5d -- serve/web`). The classic app is kept untouched and served at `/classic/` (xeno
#67). The non-merge commits (`serve-web-commits.txt`), each **to check against the new app (`serve/ui/`)** for a port;
none was checked commit by commit in this merge:

| commit | change |
|---|---|
| `b0eb3965` | a favicon for the web app and the API Monitor |
| `6a7c6f82` | unlabelled sparklines were scaled by their own noise |
| `bafc8dd5` | the Monitor shows the prefill rate in recent requests |
| `c13ddd62` | an assistant turn with no answer text is sent back in the history (#1392) |
| `d7c5dd0f` | the dashboard says why a Windows AMD card shows no load or VRAM (#1380) |
| `4546bf7c`, `8cd36070` | a Model settings view for documented run-config keys (#564; the server's `/config` route is merged, §3) |
| `a9fe9236` | a Conversation cache card in the Monitor tab (#596) |
| `b591a45b` | the PCIe share beside the hit rate (#588; the server's `pcie_share` is merged) |

`34ff00bd` and `92ca0c7f` are merge commits.

## README

`README.md` is the fork's own (#166): the conflict (17 hunks) was resolved by taking the fork's file, then:

- "last merged" bumped from v0.1.38 to **v0.1.42**;
- carried by hand from upstream's user-facing changes: the experimental hardware written and tested by upstream's
  community, which the fork has not built or measured ([older GPUs](../OLDER_GPUS.md) through a second CUDA 12
  engine, [Intel Arc](../INTEL_ARC.md), [Strix Halo](../STRIX_HALO.md), CPUs without AVX2);
  `/v1/responses` for Codex CLI and other apps on the OpenAI Responses API; upstream's `"parallel": 2` batching
  ([BATCHING.md](../BATCHING.md)), noted as not measured with the fork's two-card profile; private security reports
  ([SECURITY.md](../../SECURITY.md));
- not carried: upstream's speed figures (that machine is not this one).

## 9. Configuration surfaces this merge adds (from `tools/blueprint_check.py`)

Every name `tools/blueprint_check.py` finds added in `git diff 8abbfa0` of the merge, grouped. Upstream's own docs (`docs/DETAILS.md`, `docs/BATCHING.md`, `docs/OLDER_GPUS.md`, `docs/INTEL_ARC.md`) describe its names; the ones that reach D2x and their defaults are in §2 and in `docs/BLUEPRINT.md`'s row for this merge. Nothing was removed.

**Engine flags** (32): `--adapt-async`, `--adapt-min-gain`, `--aux-cpus`, `--batch`, `--batch-groups`, `--batch-mtp`, `--conversation-cache-min-tokens`, `--gpu`, `--host-core`, `--kv-grow`, `--lookup-chain`, `--lookup-chain-min`, `--memory-limit-mib`, `--mtp-draft-vocab`, `--mtp-hnorm`, `--mtp-q4`, `--no-kv-grow`, `--pipeline-windows`, `--pool-tasks`, `--prompt-cache-tail`, `--reasoning-effort`, `--remote-expert-opt`, `--session-min-free-mib`, `--session-save-reclaim`, `--slots`, `--spec-follow`, `--tail-role-token`, `--trim-stage-weights`, `--vram-elastic`, `--vram-reserve-later-mib`, `--vram-segment-mib`, `--window-hashes`

**GEN keys** (6): `aux_cpus`, `ckpt`, `lazy_load`, `logit_bias`, `pin`, `reasoning_budget_tokens`

**Run-config keys** (21): `codex_compaction_cache`, `codex_thread_titles`, `effort_position`, `expert_profile_save`, `gpu_order`, `min_tokens`, `open_browser`, `parallel`, `prevent_sleep`, `reasoning_close_retry`, `reasoning_effort`, `reasoning_loop_recovery`, `remote_expert_opt`, `repeat_stop_tokens`, `replica_cpus`, `replica_load_tokens`, `replicas`, `slot_save_path`, `tool_call_recovery`, `vram_elastic`, `vram_segment_mib`

(`args`, `env`, `exe`, `gpu`, `lib_dirs`, `log`, `vision`, `sampling`, `backend`, `cuda` also show as "added" because new upstream files name them; they are not new keys.)

**Environment variables** (218): `STRATA_ADAPT_JOB_CPU`, `STRATA_ADAPT_LAG`, `STRATA_ARENA_LOCK`, `STRATA_ARENA_MMAP`, `STRATA_ARENA_RELEASE`, `STRATA_ATTN_LANECELL`, `STRATA_ATTN_PRE75`, `STRATA_AUX_CPUS`, `STRATA_AUX_STAGER`, `STRATA_BATCH_DECODE_SHARE`, `STRATA_BATCH_MTP`, `STRATA_BATCH_TRACE`, `STRATA_BATCH_YIELD_CHUNKS`, `STRATA_BF16_TC`, `STRATA_CACHE_MESSAGE_BOUNDARY`, `STRATA_CUDA`, `STRATA_CVEC_FUSE`, `STRATA_CVEC_FUSE_CHECK`, `STRATA_DBG_CPU_GATE`, `STRATA_DBG_FEAT`, `STRATA_DBG_FEAT_LAYERS`, `STRATA_DBG_GDN`, `STRATA_DBG_NAN`, `STRATA_DBG_ROUTE_OVERRIDE`, `STRATA_DEC_BATCH`, `STRATA_DEFERRED_REGISTER`, `STRATA_DENSE_MMQ`, `STRATA_DF_BRANCH`, `STRATA_DF_PDL`, `STRATA_DF_PLE`, `STRATA_DF_QB_Q4`, `STRATA_DISJOINT_ADAPT`, `STRATA_DMA_BATCH`, `STRATA_DUMP_FIRST_LOGITS`, `STRATA_EMB_REUSE_ACCOUNT`, `STRATA_EXCHANGE_ROTATE`, `STRATA_EXPERIMENTAL_GFX1103`, `STRATA_EXPERT_V2`, `STRATA_EXPERT_V2K`, `STRATA_EXP_MODE`, `STRATA_FILE_RELEASE`, `STRATA_FILL_AHEAD`, `STRATA_FORCE_AVX2`, `STRATA_FORCE_ISA`, `STRATA_FS_AHEAD`, `STRATA_FS_SLOTS`, `STRATA_FUSED_SWIGLU_Q81`, `STRATA_FUSE_HEAD_GR`, `STRATA_GDN_CHUNKED`, `STRATA_GDN_COMMIT_SPLIT`, `STRATA_GDN_CONVL2`, `STRATA_GDN_HEAD`, `STRATA_GDN_NOY`, `STRATA_GDN_PP`, `STRATA_GDN_REC_HEADS`, `STRATA_GDN_SPLIT`, `STRATA_GFX1151_DEFAULTS`, `STRATA_GFX12_DEFAULTS`, `STRATA_GFX906_ATTN_QUERY_SWIZZLE`, `STRATA_GFX906_ATTN_QUERY_SWIZZLE_TRACE`, `STRATA_GFX906_ATTN_REDUCE12`, `STRATA_GFX906_ATTN_REDUCE12_TRACE`, `STRATA_GR_FAST`, `STRATA_GR_SPLIT`, `STRATA_HCD_EXACT`, `STRATA_HC_Q8`, `STRATA_HC_Q8_INJECT`, `STRATA_HC_UPMIX`, `STRATA_HC_UPMIX_CHECK`, `STRATA_HEAD_MIX_MULTI`, `STRATA_HIP_ADAPT_KERNEL_COPY`, `STRATA_HIP_PROMPT_F16`, `STRATA_HIP_ROUTER_FAST`, `STRATA_HIP_WMMA`, `STRATA_HOST_CORE`, `STRATA_HTTP_BACKLOG`, `STRATA_IO_PF_AHEAD`, `STRATA_IO_PF_STAGE`, `STRATA_IO_PF_THREADS`, `STRATA_IO_PREFETCH`, `STRATA_IO_PREFETCH_DEPTH`, `STRATA_IO_STATS`, `STRATA_IQ256_GATHER`, `STRATA_IQ3S_MT1`, `STRATA_IQ_PREFETCH`, `STRATA_IQ_STAGE_GRID`, `STRATA_IQ_STAGE_GRID_MMVQ`, `STRATA_ISA_FLOOR`, `STRATA_KEEP_EMPTY_TURNS`, `STRATA_KEEP_MAPPING`, `STRATA_KV_GROW`, `STRATA_KV_GROW_FLOOR`, `STRATA_KV_GROW_HOLD`, `STRATA_KV_GROW_INIT`, `STRATA_KV_GROW_STEP`, `STRATA_KV_HOST_DMA`, `STRATA_KV_PREFETCH`, `STRATA_KV_ROT`, `STRATA_LAYER_SPLIT_GATE`, `STRATA_LAYER_SPLIT_PROBE`, `STRATA_LFUSE`, `STRATA_LFUSE_GATE`, `STRATA_LFUSE_PAIR`, `STRATA_LOGPOS`, `STRATA_LOOKAHEAD_STATS`, `STRATA_LOOKUP_CHAIN_FIXED`, `STRATA_MAX_BODY_MIB`, `STRATA_MEMORY_LIMIT_MIB`, `STRATA_MMQ_RESIDENT_SORT_NE`, `STRATA_MMVF_ROWS`, `STRATA_MMVQ_IL`, `STRATA_MMVQ_IL_ROWS`, `STRATA_MMVQ_WAVE`, `STRATA_MTP_CATCHUP_ALL`, `STRATA_MTP_FULL_HEAD`, `STRATA_MTP_PRIORITY`, `STRATA_MTP_SHARED_BRANCH`, `STRATA_MTP_TOP2`, `STRATA_MULTI_BLOCK_ARGMAX`, `STRATA_NORM_ROPE`, `STRATA_NO_ARENA_THP`, `STRATA_NO_AVXVNNI`, `STRATA_NO_BATCH_KV_STEP`, `STRATA_NO_IQ128`, `STRATA_NO_LARGEPAGES`, `STRATA_NO_MMVQ_ROWS2`, `STRATA_NO_MULTI_GR`, `STRATA_NO_NORM_ROPE`, `STRATA_NO_Q8K_AVX1`, `STRATA_NO_Q8K_AVX2`, `STRATA_NVCC`, `STRATA_ONE_TOKEN_COMMIT`, `STRATA_OWNED_PRICE`, `STRATA_PARALLEL_SOLO`, `STRATA_PA_FAST`, `STRATA_PEER_NO_P2P`, `STRATA_PF_BK64`, `STRATA_PF_FUSED`, `STRATA_PF_FUSED_KQ`, `STRATA_PF_GEMM`, `STRATA_PF_HCDOWN`, `STRATA_PF_OCC`, `STRATA_PF_OCC_W`, `STRATA_PF_PAD`, `STRATA_PF_STEP_SYNC`, `STRATA_PF_SWITCH_MIN_T`, `STRATA_PIN_GUARD`, `STRATA_PIN_RESERVE_GIB`, `STRATA_PIPELINE_ADAPT_ASYNC`, `STRATA_PIPELINE_DEBUG`, `STRATA_PLE_BATCH`, `STRATA_PLE_PROBE`, `STRATA_PREBUILT_URL`, `STRATA_PREFILL_CPU_SHARE`, `STRATA_PREFILL_CPU_SHARE_MAX`, `STRATA_PREFILL_DUMP_R_ALL`, `STRATA_PREFILL_EQUAL`, `STRATA_PREFILL_HELP`, `STRATA_PREFILL_HELPER_SOURCE`, `STRATA_PREFILL_LEND_PCT`, `STRATA_PREFILL_STREAM_AHEAD`, `STRATA_Q2_BITPLANE`, `STRATA_Q6P_GRID`, `STRATA_Q6P_SELFTEST`, `STRATA_Q6_PACKED`, `STRATA_Q8_PACKED`, `STRATA_QFUSE`, `STRATA_QSA_EARLY_EXIT`, `STRATA_READ_AHEAD`, `STRATA_REPLICA_START`, `STRATA_RING_BYTES`, `STRATA_ROUTER_BLOCK`, `STRATA_ROUTE_PER_TOKEN`, `STRATA_ROUTE_PRIOR`, `STRATA_ROUTE_PROJ_PER_TOKEN`, `STRATA_ROUTE_RESIDENT`, `STRATA_ROUTE_RESIDENT_MTP`, `STRATA_ROUTE_RESIDENT_RANKS`, `STRATA_ROUTE_TAIL_SKIP`, `STRATA_RSS_TRACE`, `STRATA_S2_SWIGLU_Q8`, `STRATA_SELECT_SIMT`, `STRATA_SESSION_BUFFERED`, `STRATA_SH_FORK_LATE`, `STRATA_SH_STREAM`, `STRATA_SKIP_SHA256`, `STRATA_SM70_TABLE`, `STRATA_SOURCE`, `STRATA_SPEC_DEPTH`, `STRATA_SPEC_MIN_TEMP`, `STRATA_SPEC_PROB`, `STRATA_SPEC_PROB_DT`, `STRATA_SPEC_PROB_GATE`, `STRATA_SPLIT_COVER_B`, `STRATA_SPLIT_MTP_BATCH`, `STRATA_SPLIT_NEAR_TIE_PCT`, `STRATA_SPLIT_OWN_BUFFERS`, `STRATA_STAGER_BATCH`, `STRATA_STAGER_SLEEP`, `STRATA_STAGER_SSD`, `STRATA_STAGE_PIN`, `STRATA_STAGE_TRIM`, `STRATA_STARTUP_UNBUFFERED`, `STRATA_SYCL_BIN`, `STRATA_TEST_PAGEABLE`, `STRATA_TOPK_STREAM`, `STRATA_TSUM`, `STRATA_UMA_HEADROOM_GIB`, `STRATA_UNBUFFERED_LOAD`, `STRATA_VCVARS`, `STRATA_VERIFY_ALL_RESIDENT`, `STRATA_VERIFY_ARENA`, `STRATA_VERIFY_COMPLEMENT`, `STRATA_VERIFY_DEVICE_PLAN`, `STRATA_VERIFY_QDEDUP`, `STRATA_WATCHDOG_IO_S`, `STRATA_WMMA_BF16`, `STRATA_WMMA_GEMM`

**New source files** (65): `include/strata/core/arch_defaults.hpp`, `include/strata/core/batch_rows.hpp`, `include/strata/core/conversation_file.hpp`, `include/strata/core/dma_batch.hpp`, `include/strata/core/exchange_storage.hpp`, `include/strata/core/foresight_swap.hpp`, `include/strata/core/logit_bias.hpp`, `include/strata/core/remote_expert_opt.hpp`, `include/strata/core/session_save_reclaim.hpp`, `include/strata/core/spec_prob.hpp`, `include/strata/core/vmm.hpp`, `include/strata/hip_compat/math_constants.h`, `include/strata/kernels/cpu/iq_avx1.hpp`, `include/strata/kernels/cpu/kq_avx1.hpp`, `include/strata/kernels/gfx_arch.hpp`, `include/strata/kernels/pdl.hpp`, `include/strata/kernels/q8_1_finite.hpp`, `include/strata/kernels/route_prior.hpp`, `include/strata/platform/aux_cpus.hpp`, `include/strata/platform/hip_compat/cublas_v2.h`, `include/strata/platform/hip_compat/cuda_fp16.h`, `include/strata/platform/hip_compat/cuda_runtime.h`, `include/strata/platform/hip_compat/math_constants.h`, `include/strata/platform/hip_compat/strata_hip.h`, `include/strata/platform/host_limits.hpp`, `include/strata/prefill/share_rules.hpp`, `include/strata/prefill/xmx_moe.hpp`, `include/strata/program/batch_groups.hpp`, `include/strata/program/batch_read.hpp`, `include/strata/program/draft_kv_plan.hpp`, `include/strata/program/message_boundary.hpp`, `include/strata/program/split_ring.hpp`, `include/strata/program/split_rules.hpp`, `include/strata/spec/draft_source.hpp`, `serve/cpu_affinity.py`, `serve/fixtures/rcall_specimens.json`, `serve/keepawake.py`, `serve/logit_bias.py`, `serve/prometheus.py`, `serve/replicas.py`, `serve/responses.py`, `serve/runconfig.py`, `serve/session_save_retry.py`, `serve/web/favicon.svg`, `src/core/arch_defaults.cpp`, `src/core/conversation_copy.hpp`, `src/core/conversation_file.cpp`, `src/core/remote_expert_opt.cu`, `src/core/vmm.cpp`, `src/kernels/cpu/iq_avx1.cpp`, `src/kernels/cpu/iq_avx2_rows.inl`, `src/kernels/cpu/kq_avx1.cpp`, `src/kernels/cpu/pool_affinity_linux.hpp`, `src/kernels/cpu/pool_affinity_win.hpp`, `src/kernels/cpu/q2_avx1.cpp`, `src/kernels/cpu/q2_avx2_rows.inl`, `src/kernels/cuda/q8_1_il.cuh`, `src/kernels/cuda/route_prior.cu`, `src/kernels/cuda/s26_tsum.cuh`, `src/kernels/fused_gr_bench.cpp`, `src/kernels/native_expert_bench.cpp`, `src/prefill/mmq_resident_sort.hpp`, `src/prefill/wmma_gemm.cu`, `src/prefill/wmma_gemm.h`, `src/spec/draft_source.cpp`
