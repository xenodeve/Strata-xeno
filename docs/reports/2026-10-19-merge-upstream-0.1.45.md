# Merge report: upstream v0.1.45, then v0.2.0-beta, into the fork's main (2026-10-19, #268)

The merge has two steps on one branch, `xeno/268-merge-0.1.45` (worktree `C:\Strata-exp\src-m145main`):

1. **v0.1.45.** Upstream `v0.1.45` (tag object `b39442e6`, commit `477ac13c`, 2026-10-16; fetched from `agentionai/Strata`,
   git remote `agention`) merged into the fork's `main` `e5737a27` (the merge of #238, 2026-10-15). Merge commit
   `1faf709a`, then `33ab01d5` (the build fix and the docs of §6a). Sections 1-4, 6a, 7-9 and the README section.
2. **v0.2.0-beta on top.** Upstream `Niko1221/Strata` `v0.2.0-beta` (tag object `b73dbce3`, commit `3d1fb480`,
   2026-10-19) merged into `33ab01d5`. Merge commit `f06366df`, then `8ed30815` (Windows build fixes) and `f20ec6ab`
   (test fixes). Section B.

§5 (verification) and §6 (commits and exes) cover the tree of both steps: it was built and tested once, at the end.

Two build dirs, both CUDA 13.3, sm_89 + sm_120, Ninja Release, tests and xeno tests on:

- `C:\Strata-exp\build-m145u` (`build-m145u.bat`): the default ggml pin, the D2x build;
- `C:\Strata-exp\build-m145g` (`build-m145g.bat`): `STRATA_GGML_DIR=C:/Strata-exp/agention-llama.cpp`, the Gyro build.

Working records: step 1 `C:\Strata-exp\m145-notes\{prefill,generate,core,kernels,serve}.md`, step 2
`C:\Strata-exp\m020-notes\{gyro-cpu,gpu,core,serve}.md` and `blueprint-check.txt` (every hunk with its side and reason,
per area); the build logs `C:\Strata-exp\build-m145u.out` and `build-m145g.out`; the gate records
`C:\Strata-exp\m145-record\` (§5).

**Why `main`.** #268's decision comment: trial merges into `main` (32 files, 190 hunks) and into `xeno/defer` (37 files,
about 220 hunks). `main` was chosen, as at 0.1.42: the merge goes to the base the other branches follow.

**Why v0.2.0-beta on top.** #268's second comment: v0.2.0-beta was published the same day (2026-10-19 13:52 UTC). A trial
merge onto `33ab01d5` conflicted in 26 files, 126 hunks (30 of them in README), fewer than v0.1.45's. Each Gyro file
conflicted once, because both sides carry the same agentionai rc2 code. Merging it before the verification means one
verification cycle instead of two. This follows the developer's "if there's no trade-off, use the latest".

**Verdict.** Verified for D2x; Gyro-M is state-identical and faster in its normal boots, with one open issue (#275).
What is known about the tree of both steps (§5):

- no conflict marker is left;
- every fork default that changes served outputs is kept (§2, §B2);
- after `8ed30815`, `build-m145u` builds every target and `build-m145g` builds the engine;
- ctest after `1bd671c7` (`build-m145u`): only the known `ple_parity` fails;
- pytest after `484eaaf0`: `serve` + `tests/xeno` 1 failed (no `psutil` on this box), 1802 passed; `tools` collects
  and its merge-only failures are fixed or explained (§5);
- **D2x:** `STRATA_STATE_HASH` identical to the merge parent `e5737a27` in 4 boots, identical replies, five of Claude
  Code's six parts read 36-128 ms faster (the 2400 part +42 ms), generate-mode decode with swaps off 8-11 % faster on
  code with identical outputs, and **the 23-turn served ABBA x1.038-1.059 per turn** (`ms_gpu_wait` -0.7 ms,
  `ms_draft` -0.6 ms, acceptance equal, prompt read equal);
- **Gyro-M, swaps off:** all 7 `STATE_HASH` fingerprints identical in 6 boots across `main` (agentionai ggml), the
  merge (agentionai ggml) and **the merge on the default ggml with the trellis overlay**; so one default-ggml build
  can serve both models (#258);
- **Gyro-M served** (the merge carried into the served Gyro branch, `xeno/defer-m145`, against the installed
  `run-gyro-tq`): state-identical in 4 boots; in its normal boots x1.007-1.087 per turn with `ms_gpu_wait` -0.75 ms
  and `ms_draft` -0.6 ms in every boot; but 3 of its 10 boots ran the CPU pool 1.5-3.2x slower for the whole boot
  (#275, cause not found; the installed exe showed it in none of 6).

Not settled yet:

- `native_q2_pool_hit_parity` (with and without `STRATA_GU_SWIGLU=1`) and the ctest of `build-m145g`;
- #275 before the Gyro-M install.

## 1. What moved

- **Base:** the fork's `main` `e5737a27` (2026-10-15).
- **Upstream:** `v0.1.45`, tag object `b39442e6` -> commit `477ac13c` (2026-10-16, tagger Niko1221), fetched from
  `agentionai/Strata` (the 0.1.42 report named `Niko1221/Strata`; the tagger is the same). Five releases since 0.1.42,
  651 commits (585 not merges):

  | release | date | commits |
  |---|---|---:|
  | v0.1.43 | 2026-10-11 | 72 |
  | v0.1.43.1 | 2026-10-12 | 33 |
  | v0.1.44 | 2026-10-13 | 271 |
  | v0.1.44.1 | 2026-10-14 | 11 |
  | v0.1.45 | 2026-10-16 | 264 |

- **Merge base:** `61b3fb5d` = `v0.1.42`, the second parent of the 0.1.42 merge. A normal base this time: no graft.
- **Upstream's size:** `git diff --shortstat 61b3fb5d 477ac13c`: 570 files, +84,111/-2,491. Most of it is new code
  the fork does not build or run:
  - `metal/`: 89 files, +20,363, the Apple Silicon engine;
  - `src/qwen35/`: 4 files, +5,212, an engine for Qwen3.6-35B-A3B (`strata_qwen35`, linked into `strata`);
  - `tools/qwen35/`: 27 files, +1,728;
  - `bench/`: 73 files, +6,548 (community results);
  - `sycl/`: 46 files, +2,841/-297.

  The rest by directory: `src/` 113 files +28,408/-1,251 (the Qwen3.6 engine included), `tools/` 86 files
  +10,639/-336, `serve/` 33 files +4,734/-153, `include/` 60 files +2,567/-154, `docs/` 26 files +2,752/-92, `tests/`
  17 files +2,663/-23. `third_party/` is unchanged: the ggml pin stays `3cf03257` (kernels.md §4).
- **The fork's side:** `git diff --shortstat 61b3fb5d e5737a27`: 694 files, +109,750/-2,704. This is the fork's whole
  delta over v0.1.42, the side the three-way merge weighs, not work since the last merge. Since the 0.1.42 merge
  landed (`27d8d5d6`, PR #214, 2026-10-12): 74 commits (66 not merges), 54 files, +5,696/-140. (`git log
  61b3fb5d..e5737a27` lists 1,145 commits because upstream's rewritten history, 0.1.42 report §1, leaves the fork's
  older commits unreachable from the base. That number is not new work.)
- **The merge into `main`:** `git diff --shortstat e5737a27 1faf709a`: 571 files, +83,682/-2,264.
- **Conflicts:** 32 files, 190 hunks (counted from `git show --remerge-diff 1faf709a`):

| file | hunks | what collided |
|---|---:|---|
| `src/prefill/prefill.cpp` | 39 | upstream's `--prefill-pipe` planning, `PinnedUpload` / `PinnedRing` (#1712, #1996), NVFP4 buffers and walk, STRATA_MMQ_DIRECT, 653f0113's grouped short-prompt gathers, `stop_at_chunk`, the layer-split hand-off pipe, against the fork's split, wave, expert order (#41), #34 sub-products, #29 group gather, #122 KV staging, APR / Gyro rotation, timeline spans |
| `src/program/generate.cpp` | 32 | upstream's head alias, per-part loan and refill, PCIe probe and balance, pin by need, adapt overlap, checkpoint pool, #879 guard, `pool_measure`, against the fork's 4070 tier, paired swaps, wave lanes, CUDA0 loan, cache slots, `serve_fatal`, timeline spans |
| `serve/server.py` | 28 | upstream's #879 multi-pass retry, model-card sampling (#2685), `answer_room`, #567 `PromptEncoder`, #1999 client watch and `cancel`, #2228 image keys, image generation, `before_load_*`, against the fork's single-pass `_generate`, `count_only`, priority gate, `/health`, `free_vram_mib(index)`, replay cache |
| `setup.py` | 12 | upstream's NVFP4 / Qwen3.6 / Unsloth families, `engine_step()`, the stale-binary check (#2054), against the fork's Gyro family, pins, build folder and size rule |
| `src/core/expert_source.cpp`, `src/core/mtp.cpp` | 9 each | pin by need (`PadMap`, `arena_off`, `blob_at`) against placement-first, the slab and the NVMe tier; #2065 / STRATA_DRAFT_TIMING in `draft` against the fork's spans and #44 D4 |
| `src/core/verify.cpp`, `src/kernels/cuda/iq_kernels.cu` | 8 each | STRATA_PCIE_BALANCE, 143-amdk fused waits / hits, split overlap, async launch, adapt gate, against the fork's `pcie_share_`, H2 `moe_hit_merge_mapped` and spans; NVFP4 (`Fmt<40>`) and Q5_K / Q6_K downs against the fork's APR trellis types 144-146 and Q2_0 CPU order |
| `CMakeLists.txt` | 6 | upstream's NVFP4 sources, `host_graph_launch_test`, the moved `STRATA_NATIVE_EXPERTS` option, MMQ tests, against the fork's `tq_*` sources, `strata_secondary_compute`, IQ1_M tile |
| `src/prefill/moe_mmq.cu` | 5 | K-quant MMQ behind `kq_ok()` / the `strata_kq` library, NVFP4, STRATA_MMQ_DIRECT, against the fork's TQ and IQ1_M cases and `expert_rows*` |
| `src/kernels/cpu/pool.cpp` | 4 | `--host-core sibling`, STRATA_POOL_UMWAIT, per-layer geometry (`n_ff` / `n_embd`), against the fork's xeno #2 core order, `rest_` early sleep and #147 one-phase block |
| `README.md` | 3 | the fork's own README (#166) against upstream's Mac and model text |
| `docs/DETAILS.md`, `include/strata/core/verify.hpp`, `include/strata/kernels/iq_kernels.hpp`, `include/strata/prefill/kernels.hpp`, `include/strata/prefill/moe_mmq.hpp`, `include/strata/prefill/prefill.hpp`, `src/platform/memory.cpp` | 2 each | sampling text and #879; `AdaptGate` / `set_pre_launch` beside the fork's GPU lane; EXPDEC API beside the fork's Hadamard; `row_sd` / `f16img`; `gather_native` tails and `kGatherGroupMax`; new private members; two `lock_resident_ranges` |
| `.gitattributes`, `include/strata/core/expert_source.hpp`, `include/strata/core/pinned.hpp`, `include/strata/kernels/cpu/native_expert.hpp`, `include/strata/platform/memory.hpp`, `include/strata/prefill/gemm.hpp`, `serve/test_server.py`, `src/core/pinned.cu`, `src/ngram/ple_reader_test.cpp`, `src/prefill/kernels.cu`, `tools/check_isa_guard.py`, `tools/iq_pack.py`, `tools/test_setup_choices.py` | 1 each | see §3 |

Merge commit, later commits and the exe sha256: §6.

## 2. Upstream's changes that matter here, and what the fork does with them

D2x as served (`strata-flash-next-d2x.json`, exe `C:\Strata-exp\run-d2x-260\strata.exe`): Swift 1.5 IQ2_XS,
`--exclusive-primary-experts`, the 4070 tier (`--secondary-expert-mib 4100`), `--adapt-swaps 8`, split + wave
(`STRATA_PREFILL_EXPERT_SPLIT=1`, `STRATA_PREFILL_WAVE=1`, `STRATA_PREFILL_SPLIT_MIN=256`, `STRATA_SPLIT_BO=q8`),
`--prefill auto`, `--pcie-frac 0`, `--kv int8`, `--kv-resident 65536`, `--spec 4 --spec-min-p 0.5 --mtp`, `--ple-gguf`,
`--short-read 256`, `--vision`, `--pool-priority 2 --process-priority 2`, `STRATA_POOL_SPIN_US=0`; config keys
`"sampling": {"preset": "reasoning"}`, `"reasoning_carry": "history"`, `"fit_max_tokens": true`. The Gyro-M profile
(`strata-gyro-m-d2x.json`, `run-gyro-tq`) has the same shape on agention's ggml. For each item: whether it reaches that
path, and whether it is on.

**Kept at the fork's value, with upstream's value behind a switch:**

| item | reaches D2x | on | why / switch to upstream's value |
|---|---|---|---|
| Model-card sampling for a request that gives none (#2685) | yes | **off** | Upstream sets `svc.model_arch = "qwen4exp"` for every strata engine, so `model_card_sampling(...)` sits under every request's defaults. On D2x the `reasoning` preset still wins for thinking requests, but its non-thinking requests (Claude Code's side requests, the classifier) would gain the card's `presence_penalty: 1.5`; a config with no `sampling` block (`strata-swift-capacity.json`) would stop being greedy, and so would parity harnesses that start a server without one. Decided in the merge commit: `"model_card_sampling": true` arms it; unset, `model_arch` stays `None` and `model_card_sampling(None, ...)` returns `{}` (server.py:8225) |
| `answer_room` (0.1.45, default on upstream) | yes | **off** | A thinking request with `max_tokens >= ANSWER_ROOM_MIN_CAP` and no budget would get a #123 budget of about 3/4 of `max_tokens`: new behaviour for every Claude Code turn. Decided in the merge commit: `"answer_room": true` arms it |
| PCIe start-up probe, PcieAuto (now with an NVFP4 base of 0.25 and `STRATA_PCIE_AUTO_CANON`) | no (`--pcie-frac 0`) | **0** (#27) | as at 0.1.42 (generate.md hunk 6). No switch restores upstream's no-flag default |
| `STRATA_ROUTE_TAIL_SKIP` default | yes | **off** | never on by default on the fork (PR #214 review). Upstream's #1884 `STRATA_TAIL_SKIP_MAX_CACHED` and its NVFP4 exemption refine a default the fork does not have: not taken (generate.md hunk 14). `STRATA_ROUTE_TAIL_SKIP=7` arms it |
| The 0.1.42 rows: LAZY module loading (#30), `--ple-inflight 64`, `STRATA_MTP_CATCHUP_ALL` (whole window), the CPU prompt share, `STRATA_RING_BYTES`, `STRATA_PREFILL_STREAM_AHEAD`, `RING_MAX = 512` | yes | fork's | unchanged by this merge: checked in the merged tree (generate.cpp:520 `ple_inflight = 64`, generate.cpp:2943, mtp.cpp:77-83; prefill.md §2) |

**Upstream defaults taken, on by default.** Upstream claims each is bitwise or output-neutral; greedy parity (§5) is
the check, and each has its own switch for bisection if parity fails:

| item | reaches D2x | on | why / switch |
|---|---|---|---|
| `STRATA_COMMIT_BATCH` (verify.cpp `capture_commit`) | yes (the commit graph) | yes | every GDN layer's conv history and state in two launches instead of two per layer, about 70 graph nodes fewer; "the same arithmetic per layer, bitwise" per upstream. `=0` gives the per-layer loop |
| `STRATA_NAN_GUARD` (#879) | yes | yes | a non-finite check of the logits recorded in the window graphs; the DONE / BDONE line carries `nonfinite`, and the server retries once (`Service._engine_once_more`, the fork's port, below). `=0` records no kernel. Its cost per window is not measured |
| `STRATA_DF_QB_Q4` (on since 0.1.45) | **no** (`--kv int8`; Q4_0 KV only) | yes | a Q4_0 KV cache takes the batched decode window too. `=0` gives the token-by-token path |
| `STRATA_MMQ_KQUANTS` and `STRATA_Q6K_EXPERTS` ON in CMake (465cd61d, "formats cost nothing unless used") | the build, not the walk | yes | Both build dirs are configured with both ON (their `CMakeCache.txt`). D2x's experts (IQ2_XS, IQ1_M, Q2_0) and Gyro's trellis types do not use these kernels; the Q6_K gate/up and Q5_K / Q6_K down instances are compiled into `iq_kernels.cu`, and `strata_kq.dll` is built and opened only with `STRATA_MMQ_KQ=1`. The fork keeps LAZY module loading, so unused kernels should stay unloaded (a hypothesis). Upstream measured "slot counts unchanged" on a P100 and an R9700 only. **Checked on D2x (§5), with step 2's trellis MMQ instances also compiled in:** the boot lines of the merge's exe are the base's: "9.38 GiB free ... cap 8.70 GiB", 6485 CUDA0 slots, 2992 secondary experts, "544 MiB of VRAM free with everything loaded" (`m145-record\ab145-speed-{base,new}-*.log`). `-DSTRATA_Q6K_EXPERTS=OFF -DSTRATA_MMQ_KQUANTS=OFF` is not needed |
| `PinnedUpload` / `core::PinnedRing` (#1712, #1996) | yes (the QSA step records and token ids; D2x's routing tables are mapped and its PLE pinned) | yes | pageable async copies go up through pinned slots; bit-identical. It adds a host `cudaEventSynchronize` on the previous chunk's upload before each upload, which should be long done after the per-layer router sync. Unmeasured |
| `core::qsa_shapes(g)` and `K = g.n_expert_used` | yes | yes | `n_rot`, `idx_block`, `idx_top_k` and K now come from the GGUF's geometry instead of compiled constants; equal for this model if its geometry check passes (prefill.md §4, generate.md hunk 9) |
| #1999: the client watch starts before `prepare` | yes | yes | a client that leaves during tokenising or a picture's encode now cancels it (`cancel` through `prepare` / `prepare_carried`, `ClientGone`). Image encodes go through the fork's priority gate (`_turn(cancel, priority)`) |
| `kv_quiesce = adapt_quiesce` (serve, after `wait_commit`) | yes | yes | a superset of the fork's `cudaDeviceSynchronize(); apply_pending(true);`: it also runs `adapt_tick(true)` |
| `progress_at("mtp draft: waiting for the GPU")` (#2065) and the draft timers | yes | yes | the watchdog's text names a stall in the draft. `STRATA_DRAFT_TIMING=1` prints the split; its "PLE prefetch" column is always 0 on the fork (the fork's read-ahead is #44 D4, not upstream's `prefetch_ple`) |
| #567 `PromptEncoder`, #2518 `pick_slot` follows the turn checkpoint | yes (serve) | yes | tokenising resumes from the last shared special token; not measured here |

**Refused or held off on the fork,** each with a message:

- **`--prefill-head-alias`: exit 2** ("not ported onto this fork's prompt loan (wave lanes, KV streaming)",
  `head_alias_ported = false`). Upstream keeps the head-alias layout fixed inside its per-part `lend`; the fork's `lend`
  relays the buffers out per part, and the per-request rebuild resets lane 1 only. Porting it means teaching the fork's
  `lend` the fixed layout: the developer's decision, not the merge's.
- **NVFP4 packs fail `Prefill::init`** (any native layer with `tail_off != 0`, with a message naming the merge). The
  fork's prompt walk, split (`moe_split_finish*`, `routed_sum`), FP16 path and `expert_rows` would read the experts
  without their per-expert scales. The decode kernels do take NVFP4 (`Fmt<40>`, the scale `* scale` / `* sd`, and
  `iq_dequant_expert_f16` falls back to upstream's two tail-scaled launches), but a pack the prompt path refuses cannot
  be served. generate.cpp does not refuse such a pack earlier (open, §7).
- `STRATA_ADAPT_OVERLAP` with the paired swaps: a new `aov_why` reason (the paired path never fills `pending`, so the
  verify gate would wait for layers never marked).
- `STRATA_PCIE_BALANCE` with the 4070 tier or `--exclusive-primary-experts`: guarded off (it would turn on PCIe reads
  beside a tier the fork allows only at share 0).
- `STRATA_PIN_BY_NEED` when the arena is not registered for CUDA (the 4070 tier, exclusive primary experts, or
  `STRATA_ARENA_PIN=0`): off, with that reason. Its padded layout also needs `!defer_load && pin_for_cuda`
  (placement-first writes packed offsets).
- `STRATA_MMQ_DIRECT` and `STRATA_MMQ_GROUP_SORT`: ignored, with one stderr line each; `slot_guard_bytes()` returns 0,
  so the expert cache is sized as before whatever the switch says.
- In a wave, `stop_at_chunk` is not honoured (the part is read to its end) and a forced chunk different from the
  lane's buffers fails the run with an error.

**Upstream features not taken, with the reason:**

- **653f0113, "gather experts in groups on short prompts too"** (`STAGE_GRP`, `stage_slots`, `group_resident`): it
  lives in upstream's per-expert `compute` / `flush`, which the fork replaced (#29 group gather in the streamed walk,
  #34 sub-products). A port groups the fork's staged walk and grows its ring from 8 to 32 slots (+24 x MAXBLOB of loan
  on every short layout), a reschedule of D2x's short Claude Code parts. Per the 0.1.42 precedent that is opt-in and
  measured first: a follow-up (§7).
- **NVFP4 in the prompt walk** (tails, `swiglu_scaled`, `down_row_scales`, `row_sd`, `fp4_activations`, `Xscale` /
  `Hscale`): refused at init instead (above). `Impl::Xscale/Hscale/grp_tail/row_sd` stay declared, never carved.
- **`quantize_scatter`** (c675f5b7, each token's MoE input quantized once and scattered to its K rows): the fork's #34
  per-token `Xtok` already does it.
- **`STRATA_REFILL_PARTS`, upstream's per-part relayout and #2090's halving loop in `lend`:** they size upstream's
  per-participant loan; the fork's `refill_lent` and `lend_slots(want, kv_end, wave)` (#122, #172) are a different
  rule. Upstream's `seg_lend` and `exact` were ported so `read_part`'s `plan.lend` compiles (generate.md hunks 21-25).
- **mtp.cpp's in-drafter `prefetch_ple`:** as at 0.1.42, one mechanism for one job (#44 D4).
- **Upstream's multi-pass `_run` pieces** (`segments`, `nf_retried`, `tail`, `close_retried`): the fork keeps its
  single-pass `_generate`; #879 is ported onto it (below).
- **PeerPrefill and everything on it** (#1996's edits inside it, its NVFP4 refusal, `up_ring` in it), the layer-split
  help (`kSplitHelpRing`, `single_chunk_`), the owned one-allocation ring: as at 0.1.42.
- Upstream's `#if defined(STRATA_PREFILL_MMQ)` guard on the `moe_mmq.hpp` include: the fork calls `mmq::warm()`
  unconditionally (#30).
- Upstream's `strata trace: read N tokens` line: the fork's "prompt part" line replaced it in 0.1.42.

**#879 non-finite logits, ported onto the fork's loop.** Upstream's retry lives in its multi-pass loop. The fork's
port is `Service._engine_once_more`, `_generate`'s first engine call: when the engine ends a pass with
`EngineNonFinite` before it wrote a token, and the client has not cancelled, the request is sent once more (upstream's
log line), and `engine.last` holds the retry's DONE. `_run` handles `EngineNonFinite` before `ValueError` (it is a
subclass): twice -> finish "error" with upstream's message (503 / error event); after tokens came out -> "length";
cancelled -> "cancel" / "stop". Each forgets the conversation's #200 carry entry. One difference from upstream: a
forced opening yields its tokens before the engine, so a retry that fires again ends as "length", not 503. In the
engine, the two `return 1` beside the `nonfinite_hit` break (generate.md hunk 29) are `serve_fatal()` (§4).

**The rest, taken as upstream has it (opt-in, or not on the D2x path):**

- **Engine flags and values:** `--prefill-pipe N` (#1441, layer split only), `--prompt-cache-spread`,
  `--conversation-cache-park-gain N`, `--pool-affinity measure`, `--host-core sibling` (ported into the fork's hybrid
  branch of `detect_cpu_topology_impl`; without the port it did nothing on this machine's hybrid CPU). GEN keys
  `img_keys` (#2228, a picture matched by its bytes' hash) and `pipe`.
- **Engine switches:** STRATA_EXPDEC* (grouped-expert decode kernels), STRATA_POOL_UMWAIT* (UMONITOR/UMWAIT parking,
  MSVC x64 + WAITPKG; unset, the fork's spin loop exactly), STRATA_ADAPT_OVERLAP / _SIGNAL and the other adaptive-tier
  details, STRATA_CKPT_POOL / _THREADS, STRATA_SLOT_*, STRATA_PIPED_* (prompt read beside decoding slots),
  STRATA_RAM_ADAPT, STRATA_SPLIT_OVERLAP and the layer-split / gfx12 set, STRATA_HANDOFF_PIPE(_WORKER),
  STRATA_VERIFY_ASYNC_LAUNCH, STRATA_FUSE_WAIT / _HITS (CUDA default: folded wait on sm_75 only), STRATA_F16_RANGE
  (HIP route), STRATA_XN_F16 (on only where `Gemm::bf16_converts_x()`, Turing), STRATA_HC_Q8 / STRATA_HC_REQ8,
  STRATA_PCIE_MIN1, STRATA_COMMIT_TIMING, STRATA_DRAFT_TIMING, the dump hooks (STRATA_DUMP_MOE_INPUT / _LAYER, ported;
  STRATA_QSA_DUMP_*), `borrow_spill` for `Prefill::init`. The full list is §9.
- **Serve:** image generation (Qwen-Image through stable-diffusion.cpp, `POST /v1/images/generations`, `GET
  /v1/images/files/<name>`, `/health` `image_generation`, `"images"` config); the elastic vision encoder (`elastic`,
  `elastic_idle_s`, `elastic_mib`); `before_load_timeout_s` / `_ignore_failure` / `_strict` (#1997); `env_to_user`
  (opt-in); joined args split and JSON errors that say where (#2267); `/unload?wait=1`; Retry-After on GpuBusy; engine
  start retries inside the GPU runtime set-up (`STRATA_START_RETRIES`); the crash note's CPU flags / NTSTATUS (#2324);
  STRATA_ADMIT_SHORT_FIRST / _MAX_WAIT_S; STRATA_DROP_CACHE_ON_EXIT; `reasoning_effort: false` as a level.
- **Setup:** families `nvfp4` and `qwen36` (preview), Unsloth UD-Q5_K_XL / UD-Q6_K_XL / Q8_0, UD-Q4_K_XL out of
  experimental; `--images` and its options; `engine_step()` (the fork's Gyro step moved into it); the stale-binary
  check (#2054); builds pass `-DSTRATA_Q6K_EXPERTS=ON -DSTRATA_MMQ_KQUANTS=ON`.
- **Hardware and platforms the fork does not build or run:** the Metal engine, the Qwen3.6 engine (`strata-q36`,
  `strata-q36ref`; `STRATA_HAVE_QWEN35`, `STRATA_Q36_CPU_POOL`), NVFP4 end to end, the sm_60 / sm_61 / sm_75 tables,
  gfx1150 (`STRATA_EXPERIMENTAL_GFX1150`), the gfx12 chunk rules, rocBLAS tuning.

## 3. Every conflict, file by file

Five areas were resolved in parallel, each into one notes file; `.gitattributes` was resolved outside them. Each notes
file holds every hunk with its side and reason, and the audits below; this section is their summary.

### `src/program/generate.cpp` (32 hunks; `generate.md`)

Resolved by a script from per-hunk resolutions (`resolve_gen.py`); the file stays CRLF, 18,780 lines.

- **Kept from the fork:** the PCIe default 0 with no probe (hunk 6); the whole placement block (hunk 7); the CUDA0
  `refill_lent` with its span (20) and the one loan `if (!refill(e)) return false;` (23); the tail-skip default off
  (14); the "prompt part" line (27); `run_prompt` and the wave lanes (26); the unguarded `moe_mmq.hpp` include (1).
- **Both sides, in order:** includes (2); `mem_mark`'s timeline step then upstream's `STRATA_TRACE` form (4);
  `CacheSlot` / `slot_*` (#56, #145) then upstream's `CheckpointPool` / `StageThreads` (5); the CPU ISA lines (8);
  upstream's `ModelGeometry` K (9); STRATA_PIN_BY_NEED before the fork's `open(..., pin_for_cuda, place_first, ...)`,
  with a new first `off_why` for a pageable arena (10); the fork's priority lines, upstream's callback warm-up and
  `pool_measure` (11); `slots[4]` and `ckpt_pool` (13); #229's paired serve stages and `adapt_nswaps` (15); the paired
  stage 1 then upstream's `aov` ordering, with `adapt_nswaps += swaps.size()` ported into the paired branch so
  upstream's swaps-per-window figure is not 0 on D2x (16); the cache-slot switch, the head-alias rebuild, `kv_quiesce`
  (18); `tl_req0` and `req_times` (19); the fork's timeline span around upstream's `adapt_nowait` / `aov` /
  `adapt_lag` chain and STRATA_RAM_ADAPT (28); #879's `nonfinite_hit` break and the merged-window `mtp.prefill`
  **before** the fork's `ple_ahead.start/push`, so a window that breaks out pushes nothing (29); the emit loop (30);
  the prefill sources line and upstream's MMQ_DIRECT read line (32).
- **Ported onto the fork's code:** `o.prefill_head_alias` as lane 0's `borrow_spill` (12); `seg_lend` and the
  `exact` parameter in the fork's `lend(tokens, kv_end, e, exact)`, `want = exact > 0 ? min(exact, prefill_chunk) :
  want_seg` (21, 22, 24); `plan_lend` from `resume_plan` (25); `read_to` (STRATA_ADMIT_MERGE_FIRST) with `run_prompt`
  (26); `req_times` sums and the `merge_at` trace line (27).
- **Theirs:** the serve loop's drain (STRATA_PIPE_BYIELD_NODRAIN), with `return serve_fatal()` (17); the adapt-thread
  join, the duplicate join dropped and the fork's "adapt join" span kept after upstream's (31).
- **Edits outside the hunks:** the five upstream-new serve-loop `return 1` -> `serve_fatal()` (§4); the
  `--prefill-head-alias` refusal; the `aov_why` paired reason; the STRATA_PCIE_BALANCE guard (in hunk 7's lambda).
- **Checked:** feature counts ours vs merged equal (`strata::timeline::` 58/58, `ExitTrace` 10/10, `end_engine` 6/6,
  `sec_paired` 10/10, wave 14/14, `req_slot` 9/9, `ban_` 20/20; `serve_fatal()` 40 -> 48); every `STRATA_*` name and
  `a == "--..."` flag of the fork's side is still in the file; the INFO line's format string re-counted (28 specifiers,
  28 arguments).

### `src/prefill/prefill.cpp` (39), `prefill.hpp` (2), `moe_mmq.hpp` (2), `moe_mmq.cu` (5), `kernels.hpp` (2), `kernels.cu` (1), `gemm.hpp` (1) (`prefill.md`)

Resolved in place (one resolution file per hunk, applied by a CRLF-keeping script), then post-edits on auto-merged
upstream code inside the fork's walk. Audited with the 0.1.42 method: every line the merge dropped from the fork is
replaced by upstream's equivalent or a listed port; the 135 upstream-added lines not in the result are all in "not
taken".

- **Kept from the fork:** `moe_bufs` (#34 `mmq_rows_cap`), `moe_set_bytes` and the MoE carve (APR `Xr`; the four NVFP4
  buffers not carved), the ring carve and counts (`ring_slots(ring_cap(T))`), `set_peer` declining, the routed-ids
  download in the "router sync" span, the #41 expert order, the #34 MoE input quantize with APR rotation and split
  send, the walk (`mmq_products`, `pt.mark`, `compute(j, stage_dev, sl)`, `expert_pos` release), the #29 one-launch
  FP16 dequant, the split finish / combine, the PLE read-ahead (`c0 + stride`), the timeline resolve block.
- **Both sides:** the no-MMQ stubs; the `Impl` tail (APR fields plus `tok_up`, `up_ring`); the destructor; the
  `--prefill-pipe` block after `bytes_needed`; `core::qsa_shapes(g)` with the #122 `stage_cells` refusal; upstream's
  `forced` / `planned_C` / local `C` / `log_plan` chunk choice (`single_chunk`, `helped`, `HelpScope` not taken);
  the wave's `stride` / `c0` with `stopped_at_ = -1`; `stop_at_chunk` **with `!wave`**; the QSA step span with
  `steps_up.upload`; both CPU-share arming conditions; the layer-split hand-off (`hand_ev_`, `hand_pieces_`); the
  STRATA_F16_RANGE report.
- **Post-edits** (auto-merged upstream code that would not compile or would misbehave in the fork's walk):
  - removed `stage_slots()` and every `STAGE_GRP` use, the `group_env` / `group_gather` / `group_resident` / `gg_now` /
    `stage_n` block, `% stage_n` -> `% STAGE`, the staged walk's look-ahead (653f0113: `stage_n = 32` would index
    `stage_dev[8..31]` of the fork's 8-slot ring);
  - `compute`'s `cslot` parameter, NVFP4 gather arguments and `last_of_group` memsets reverted;
  - base's `std::unique_ptr<PeerPrefill> pp / help_pp`, pulled back in by the split, removed;
  - the pageable branches of the ids download and `upload_tables` now go through `m.up_ring` (#1996), and
    `upload_tables` returns bool, checked at both call sites; STRATA_DUMP_MOE_INPUT / _LAYER ported after the tables;
  - `mmq_plan()`: STRATA_MMQ_DIRECT / GROUP_SORT print "ignored"; `slot_guard_bytes()` returns 0;
  - `Prefill::init` refuses NVFP4; a wave lane refuses `C != m.T`.
- **Headers:** `moe_mmq.hpp` drops the fork's `kGatherGroupMax` (upstream moved it above `WeightTab`; two would be a
  redefinition) and takes upstream's `gather_native(..., tail, tail_dst, zero_bytes)` with defaults; `prefill.hpp`
  adds `experts_direct` (and `merge_lane` sums it) and upstream's private members; `kernels.hpp` takes `f16img` and
  `row_sd = nullptr` beside the fork's #181 / #183 wire functions; `kernels.cu`'s `moe_combine_kernel` keeps the
  fork's `routed_sum` when `row_sd` is null (always on the fork); `gemm.hpp` keeps `warm_cublas` beside upstream's
  f16-range functions. `moe_mmq.cu` keeps the fork's TQ and IQ1_M cases before upstream's K-quant (`kq_ok()`) and
  NVFP4 cases, `gather_q8_rows`, and `expert_rows*`, with upstream's STRATA_MMQ_DIRECT block compiled in, unused.
- **Fork features checked present:** the split, its wire formats, `split_min()`, STRATA_SPLIT_ROUTED_MAX, the wave,
  the copy issuer, #29, #34, #41, #122, #31, #38, #33 spans, #5 P5a, the route trace, APR / Gyro, the IQ1_M tile,
  STRATA_MMQ_STREAM_K, `warm()` / `warm_cublas`, the 0.1.42 CPU-share rules, stream-ahead off, `RING_MAX = 512`.

### `src/core/*`, `src/platform/memory.cpp`, `src/ngram/ple_reader_test.cpp` and their headers (`core.md`)

- **`verify.cpp` (8) / `verify.hpp` (2):**
  - the PCIe part of the window graph keeps the fork's `if (pcie_share_)` block (K4) with upstream's three-way wait
    inside, under `pcie_share_ || m_bal_ != nullptr` (§4);
  - the CPU rows and hit add: `remote_opt_` (the fork), upstream's `fuse_rows` with `hits_in_combine`, else the fork's
    H2 `moe_hit_merge_mapped`; upstream's `dec_batch` `copy_rows_from_mapped` and trailing unconditional `moe_hit_add`
    **not taken** (H2 replaces both; taking the add would count the hits twice);
  - window start: the fork's "verify window" / "verify capture" spans around upstream's `pre` / `prelaunched_` /
    `refresh_ar()` and `chain_overlap()`;
  - staging and launch: upstream's structure (adapt gate, `HostGraphLaunch`, `NextLaunch`) with the fork's timeline
    inside `!pre`; `tl` taken after the adapt-gate wait so `ms_launch` does not absorb it; the "verify graph" GPU span
    skipped under the threaded launch;
  - the tail: the chained-stage return first, then the fork's `tt` / `ms_tail` / "verify tail", then
    `pcie_balance_window`; the commit: STRATA_COMMIT_TIMING's events and the fork's "commit graph" span;
  - the header: the fork's `decode_gpu_begin` / `decode_gpu_span` and upstream's `AdaptGate`; `set_pre_launch` beside
    `set_pcie_share` (the base's duplicate comment line dropped).
- **`expert_source.cpp` (9) / `.hpp` (1):**
  - `load_experts_gguf(..., threads, skip = nullptr, unbuffered = false, ready = nullptr, pad = nullptr)`: both
    parameters, with upstream's `dst_blob` in both readers' scatter;
  - the STRATA_DEFERRED_REGISTER predicate is `!defer_load && pin_for_cuda && !by_need && ...`; the arena is a
    four-way choice `reserve_only` / `ByNeed` / `Deferred` / the fork's `pin_for_cuda` constructor;
  - the load keeps the fork's `if (defer_load)` structure with upstream's three `pad` edits in its `else`;
  - `blob()`: the fork's guards (exclusive, deferred, the NVMe tier) then upstream's atomic `reads_.fetch_add`;
  - `host_at` / `host_commit` / `host_release` now address with `arena_off(l, x)` (identical unless padded);
    upstream's `blob_at()` is added **with the fork's guards** (null for an exclusive, deferred or NVMe-only expert)
    and without the counter side effects: a bare `base_ + arena_off` would hand the async adaptive tier a pointer into
    a GPU-owned expert's decommitted pages;
  - outside the hunks: the padded layout also needs `!defer_load && pin_for_cuda` (§4).
- **`mtp.cpp` (9, all in `MtpDrafter::draft`):** the fork's #44 D4 decision kept (no `prefetch_ple`, so upstream's
  `pss` / `do_ple` / `ple_prev` / `dth` / `PleT` dropped); upstream's `hms` timer, `progress_at` (#2065) texts,
  `launch_ms` (before the fork's event record, so it is not counted as launch time), `eb` / `ea` / `tl` /
  `t_pre_sync`, interleaved with the fork's "mtp round" / "mtp step" spans and `decode_gpu_begin`.
- **`pinned.hpp` / `pinned.cu` (1 each):** the fork's `pin_for_cuda` constructor, its `= delete` guard (#56) and
  `reserve_only`, plus upstream's `ByNeed` tag constructor, `by_need`, `pin_ranges`, `locked_ranges`. Upstream's body
  calls `reserve(...)` with seven arguments; the fork's `reserve()` has `bool allow_large_pages` fourth: it passes
  `true` (a by-need arena is CUDA-registered), as the Deferred constructor did at 0.1.42.
- **`memory.hpp` (1) / `memory.cpp` (2): both sides added `lock_resident_ranges`,** same name and signature (the
  fork's for `--lock-cpu-experts`, upstream's for pin by need's holes). One function: the fork's 4 KiB page rounding
  and null skip (its caller passes unaligned per-expert blobs) with upstream's 1 GiB `VirtualLock` pieces; Linux takes
  upstream's note strings.
- **`ple_reader_test.cpp` (1):** the fork's #213 `io_report` block first (it ends with a reset), then upstream's
  11,200-row `gather_batch` over three prefetch windows, so the big gather does not disturb the fork's counts.

### Kernels: `iq_kernels.cu` (8), `iq_kernels.hpp` (2), `native_expert.hpp` (1), `pool.cpp` (4), `CMakeLists.txt` (6) (`kernels.md`)

- **`native_expert.hpp`, `iq_kernels.hpp`:** union of fields (upstream's `tail_off`, the fork's Hadamard fields); no
  positional aggregate initialisation of either struct exists (grep). The fork's `hadamard_*` declarations and
  upstream's EXPDEC API side by side.
- **`iq_kernels.cu`:** the format lists are unions (`X(8) X(40) X(144) X(145)` for gate/up; upstream's
  `#ifdef STRATA_Q6K_EXPERTS` down list with the fork's `X(144) X(146)`; MMVQ `X(40) X(144) X(145) X(146)`), keeping
  upstream's `Fmt<40>`; `native_gu_kernel` / `native_down_kernel` keep the fork's `x_scales` / `h_scales` and take
  upstream's `* scale` / `* sd`, which are 1.0f without a tail (`s * 1.0f` is bitwise `s`); `dq_tq` (the fork's
  trellis) and `dq_nvfp4` both, the fork's closed explicitly; `is_iq()` unions. Outside the markers: the fork's
  one-launch `iq_dequant_expert_f16` (#29) falls back to upstream's two tail-scaled launches when `L.tail_off != 0`.
- **`pool.cpp`:**
  - the hybrid branch of `detect_cpu_topology_impl` keeps the fork's (xeno #2 had moved it into an `else`), with
    upstream's 5 `host_sibling` lines ported into it;
  - the park loop is upstream's STRATA_POOL_UMWAIT structure ending in the fork's `rest_` early-sleep condition;
  - `drain()` mode >= 5 and `run_split_multi_native` keep the fork's #147 `native_rows` / one-phase block with
    upstream's per-layer geometry (`nfmt_->n_ff` / `n_embd` in place of the constants `FF` / `H`). Upstream's
    `strata_qwen35` drives this pool with `n_ff` 512; for Flash-Next `n_ff == FF` (640) and `n_embd == H` (2560), so
    the D2x and Gyro arithmetic and task split are unchanged. The fork's `std::exp(-(double) g)` SwiGLU is kept.
- **`CMakeLists.txt`:** link and test-block unions (`strata_secondary_compute` with `host_graph_launch_test`; the IQ1_M
  MMQ test with `mmq_nvfp4_parity` / `mmq_expert_bench`); `strata_kernels_cpu` sources and their `/arch` / GCC flags
  hold both `tq_*` and `nvfp4_*`; the fork's `option(STRATA_NATIVE_EXPERTS ...)` dropped here because upstream moved
  it earlier (#2027, auto-merged). Every referenced new source exists.
- **Checked:** a line-multiset script (every line ours or theirs added vs base is in the merge, except the deliberate
  rewrites above); feature counts (`tail_off` 9, `expdec` 89, `STRATA_EXPDEC` 17, `had_block` 3, `x_scales` 16) match.

### `serve/server.py` (28), `serve/test_server.py` (1), `setup.py` (12), tools and docs (`serve.md`)

- **`server.py`:**
  - `_run`'s generation loop (hunks 12-15) is the fork's single pass over `_generate(state)`. Everything upstream
    changed there relative to base is #879, ported separately (§2). An AST diff of the merged `_run` against the
    fork's shows only the auto-merged sampling lines, `answer_room` and #2228;
  - `prepare(messages, tools, kwargs, max_new=None, priority=1, force=None, req=None, count_only=False, cancel=None)`:
    `priority` stays fifth positionally; `store=not count_only` carries upstream's count_tokens `store=False`; #567's
    `forget_prompt` on a context refusal ported as `try: _fit(...) except ValueError: forget; raise` (`EngineStarting`
    is a RuntimeError, so it keeps the prompt, as upstream intends); the image encode is upstream's elastic path with
    the fork's priority; `embeddings.keys` (#2228) before the fork's try/unlink around `write_temporary`;
  - `_openai` / `_anthropic` (20-24): #1999 moved — `cancel`, `rid` / `detached`, `arun.bind` and `_watch_client` now
    come before `prepare_carried(..., cancel=cancel)`; the fork's `with_slot`, `_meta`, think budget, forced opening
    and replay-cache events follow. Upstream's `events = anthropic_events(...)` / `_capture` not taken;
  - `free_vram_mib`: the fork's `index` parameter (#59) with upstream's `gpu_reader(dev, amd=...)`;
  - `/health`: the fork's fields plus `image_generation`; `/load`: Retry-After then the fork's 529; do_POST: the fork's
    `GpuBusy` handling with the Retry-After header; `run_with_mcp`: `cancel=` and `ClientGone` on both of the fork's
    prepare calls; count_tokens: the fork's `count_only=True`;
  - `main`: both sides' set-up (`HistoryStore`, checkpoints, ui paths; `before_load_*`, `split_joined_args`,
    `read_config`, `env_to_user`, `start_engine_watch()` and `start_vision_idle()`).
- **`test_server.py` (1):** both sides' test classes at the end of the file.
- **`setup.py` (12):** both sides' `HF_REVISIONS`, `MODELS`, `FAMILIES`; `SUPPORTED_GGUFS` merged; `engine_ok` is the
  fork's ggml stamp `and not stale_binary(...)`; the draft vocab default `thai` with upstream's `own_path_config`
  test; menus with both labels. Upstream factored step 4 into `engine_step()`: the fork's Gyro build is ported into it
  (`engine_step(..., gyro=False)`), and `recommended_size` gains a Gyro fallback to "S" (checked: 32 GB -> S, 40 GB ->
  S, 63.7 GB -> M, as before and as `test_the_default_size_follows_the_ram` expects).
- **`tools/check_isa_guard.py` (1, the whole file):** upstream committed it with CRLF and one stray `\r\r` (line 33),
  so git conflicted the whole file. Upstream's content with that CR fixed, plus the fork's
  `q2_0_gguf_rows_multi_avxvnni` in `WIDE_PUBLIC`.
- **`tools/iq_pack.py` (1):** the fork's `prism.hadamard` then upstream's NVFP4 `experts_bin`.
  **`tools/test_setup_choices.py` (1):** the hint names the merged list (checked by importing setup).
- **`docs/DETAILS.md` (2):** the fork's loop-guard paragraph and upstream's "Non-finite logits" bullet; upstream's
  0.1.45 sampling text with the fork's `(xeno)` preset sentence. **`README.md` (3):** below.
- All six Python files pass `py_compile` and `pyflakes`; the names pyflakes reports were there before the merge.

### `.gitattributes` (1 hunk, resolved outside the areas)

Both `-text` entries kept: the fork's `third_party/ggml/patches/*.patch` (#169, applied by `git apply`) and upstream's
`data/quant_quality/**` (tokenized byte for byte).

## 4. Semantic hazards checked beyond the markers

- **`return 1` inside the serve loop.** The fork's rule is that nothing in the loop unwinds `main` (#53, #45, #203; the
  0.1.42 §6a bug class). Every `return 1` between the serve `for (;;)` and the serve block's `return 0` was listed: the
  five upstream-new ones outside the markers, in #1164's borrow path (the failed parks, "restoring a borrowed
  conversation prefix") and the head-alias head reload (a failed load, "reloaded at another address"), and the four
  inside conflict hunks (17: 1, 18: 1, 29: 2) are `return serve_fatal();`. generate.md counts `serve_fatal()` 40 in
  the fork's side and 48 in the merged file; that +8 against the 5 + 4 sites listed was not reconciled in the notes.
- **`PinnedArena::reserve()`'s fourth parameter** (`bool allow_large_pages` on the fork): upstream's by-need body would
  not compile; it passes `true` (§3).
- **`lock_resident_ranges` defined by both sides:** one function, Windows and Linux (§3). Two definitions would not
  link.
- **A double hit add in the window graph:** H2 already adds the hits; upstream's trailing `moe_hit_add` is not taken,
  and `hits_in_combine` stays false on every branch but `fuse_rows`.
- **`STRATA_FUSE_WAIT=1` on CUDA** (opt-in) takes upstream's folded-wait copy for `dec_batch` windows. It relies on the
  copy zeroing the hit rows, which the fork's host no longer does for `dec_batch && !device_plan`:
  `copy_rows_from_mapped_after` launches the same kernel, which writes 0 to hit rows. Holds by reading; unmeasured.
- **The PCIe-share graph condition:** under `set_pcie_balance` the window plans each layer's PCIe count and the stamp is
  written only on this path, so the block runs on `pcie_share_ || m_bal_ != nullptr`. On D2x both are off, as before.
- **Pin by need's padded layout against the fork's addressing:** every `blob_offset` on the arena was listed;
  `host_at` / `host_commit` / `host_release` moved to `arena_off`; `read_expert` / `write_expert_pack` use file or
  packed offsets (correct as is); `load_experts_gguf_direct` (placement-first) writes packed offsets, so padding is
  refused with `defer_load`.
- **`blob_at()` under capacity mode's slab is not thread-safe** (a slot can move between the lookup and the copy).
  `--adapt-async` in arena mode and the slab should not be on together; **not enforced**.
- **Atomic `reads_`:** the header is `std::atomic<int64_t>`; the fork's `reads_ = 0` and `++reads_` compile.
- **Auto-merged 653f0113 in the fork's staged walk** (`stage_n = 32` against an 8-slot ring) and `compute`'s
  `last_of_group` / `cslot`: reverted (§3). Without it the walk would read null or stale stage buffers.
- **The local `C` in `run_impl`** shadows the namespace `C = 10240` for the rest of the function: the fork's side used
  no bare `C` there (0 uses), so nothing changes meaning.
- **New trailing parameters** (`gather_native`, the `GatherGroup` overload, `quantize`, `moe_combine`, `gr_norm*`) all
  have defaults; every fork call compiles unchanged.
- **NVFP4 through the fork's prompt path** would read unscaled weights: refused at init (§2).
- **STRATA_MMQ_DIRECT's slot guard** would resize the expert cache through `set_slot_guard`: `slot_guard_bytes()` is 0.
- **The wave against the pipe / forced chunks / `stop_at_chunk`:** guarded (§2).
- **`--prefill-head-alias` against the fork's loan:** refused (§2).
- **STRATA_ADAPT_OVERLAP against the paired swaps**, **STRATA_PCIE_BALANCE against the fork's tiers** (`pcie_given`
  is read before the fork forces `pcie_frac = 0`, so it is false by default and the opt-in would turn PCIe reads on),
  **STRATA_PIN_BY_NEED against a pageable arena:** each held off with a reason (§2).
- **`--pool-affinity measure` with placement-first:** `measure_pool_affinity` reads random experts' `blob()` before
  `load_rest`; in a deferred arena those pages are committed but unread, so the timing holds but it faults in about 2
  experts x 48 layers of pages a GPU tier may own. Opt-in, not on any hub key; not changed.
- **The `lend` signature:** one caller; a layer split never lends on the fork (start-time refusal), so `plan_lend` is
  0 there.
- **`lend_slots(16384)` / `(8192)` in upstream's #2090 gfx12 rule** call the fork's three-argument lambda with
  defaults (`kv_end = 0`, `wave = true`, the worst case): HIP gfx12 only, not built here.
- **Python signatures changed on both sides:** every `prepare(` caller (server.py, subagent.py, tests) passes
  `priority` fifth or by keyword; `free_vram_mib(` callers (images.py, the `vis[1]` index call); `_turn(` gained a
  `priority=1` parameter, matching the fork's `fifo.slot(priority)`.
- **Duplicates:** no duplicate `def`, route, MODELS / FAMILIES key, CLI option, CMake option or target (AST, pyflakes,
  grep); no flag parsed twice in generate.cpp beyond upstream's own groups; one "read N tokens" trace line, not two.
- **Identifiers:** every symbol the ported blocks of generate.cpp read was found declared before use (a token-level
  diff of identifiers new in the merged file); every identifier new in upstream's core files is in the result except
  the deliberately dropped `PleT` / `ple_t`.
- **Line endings:** every resolved file keeps CRLF (checked byte by byte); `CMakeLists.txt` keeps its UTF-8 BOM.
- **Not settled without a build or run:**
  - **STRATA_HC_REQ8's static scratch in `bf16_proj`** is shared by every caller; with the wave's two lanes (two
    threads, two streams) it would race. An opt-in single-GPU switch upstream; **not guarded**.
  - Timeline accuracy on opt-in paths: the adapt-gate wait has no span of its own; a prelaunched stage
    (STRATA_SPLIT_OVERLAP) and STRATA_VERIFY_ASYNC_LAUNCH have no "verify graph" span; a chained window has no
    "verify tail". None is on D2x.
  - With STRATA_ADMIT_MERGE_FIRST the "prompt part" line reports `[at, to)` while its time covers `[at, read_to)`
    (opt-in, batch admissions only).
  - The pageable ids download (`up_ring.d2h_sync`) drains the stream before the fork's "waiting for the GPU"
    `progress_at`, so on that non-default path the watchdog's text is one step stale.
  - The Anthropic thinking budget may be counted twice (§7).
  - **SYCL:** `sycl/` keeps its own snapshot copies; it has neither `ByNeed` nor the fork's `reserve_only`, and it
    still calls the two-argument `native_expert_scratch_bytes`. It did not compile against the fork's headers before
    this merge either. Not built (`STRATA_ENABLE_SYCL` is OFF). The Metal tree is not built either.

## 5. Verification

This section covers the tree of both steps (`f20ec6ab`). The first gates ran on 2026-10-19 and are in #268's comment
"First gates of the merged tree". Their data files are in `C:\Strata-exp\m145-record\`.

**Conditions.**

- Every arm uses the served config's own `--pool-priority 2 --process-priority 2`.
- One engine runs at a time, each as its own boot, in the order given.
- Every boot of both arms printed the expert arena's RAM warning: 23.7-25.7 GiB of RAM was available against the
  29.9 / 33.0 GiB the arena asked for. Other programs were holding RAM. Placement-first commits as it loads, and every
  boot finished. The condition was the same for both arms, but it is not an idle PC.

**Baselines: the merge parent, not the served exes.** The gates compare against builds of `main` `e5737a27`, the
merge's first parent, with the same ggml as the arm they are paired with. That is AGENTS.md's "same-session baseline":

- `C:\Strata-exp\run-main-e573u\strata.exe`: default ggml (`build-main-e573u`), sha256 `f422c7ba6f1d8edf1b5d93e771e17c3cb605738a9215e84eb0089a8374770b94`;
- `C:\Strata-exp\run-main-e573g\strata.exe`: agentionai ggml (`build-main-e573g`), sha256 `c12325731a12f9c6d041c600e0b4d8f3bc49581069db0e95614646a1a3d68a16`;
- the base arm's server is `C:\Strata-exp\src-main-e573\serve\server.py`, the merge arm's the worktree's. So each
  served gate compares engine and server together.

The served exes are not the baseline. `run-d2x-260` is `xeno/257-dequant` (`main` + #257, #247, #188, #260) and
`run-gyro-tq` is `xeno/defer` (`main` + 57 commits). A difference against them could come from those branches. The
caveat that remains is §7's: `main` lacks #247's PLE read-ahead lock and #188's wave ring, so the 23-turn served run
has the same crash risk on both arms.

**The merge's exes** (sha256 in §6): `C:\Strata-exp\run-m145u\strata.exe` (default ggml, the D2x line) and
`C:\Strata-exp\run-m145g\strata.exe` (agentionai ggml, the Gyro line).

### Build

- After the bugs of §6a and §B5 were fixed (`33ab01d5`, `8ed30815`), **`build-m145u` builds every target, and
  `build-m145g` builds the engine** (`strata`). The commit message of `8ed30815` says so; the Gyro dir's other
  targets were not built.
- The first build of step 1 stopped at `q36_engine.cu:1414`.
- The build of step 2 stopped on `Prefill::Impl::Xr` (C2086) and on POSIX `setenv` in three upstream tests.

### ctest (`build-m145u`, serial)

**186 of 196 passed, 3 skipped, 7 failed** (`C:\Strata-exp\m145-ctest.log`; ctest's summary "96% tests passed, 7
tests failed" counts the skips as passed, hence #268's 189 / 196). It ran on the tree of `8ed30815` (log 23:17), before
`f20ec6ab` (23:19).

Skipped (3): `qsa_slots_parity`, `expert_multi_test`, `isa_guard_check`.

Failed (7):

| test | cause | status |
|---|---|---|
| `ple_parity` | known, fails on `main` as well | unchanged |
| `xeno_pool_core_policy`, `xeno_cpu_dispatch_vnni`, `xeno_cpu_dispatch_forced_avx2` | they expected the fork's automatic worker count (13 on the i5-13500); v0.2.0's pool plan gives 9 (P-cores less the host's one, plus half the E-cores: 5 + 4, §B2) | `f20ec6ab`: the check asks `plan_pool_workers` |
| `native_hadamard_test`, `tq_cpu_parity`, `tq_cpu_parity_avx2` | they need a ggml with the trellis types; since v0.2.0 the default-ggml build compiles them too | `f20ec6ab`: they exit 77 there and ctest counts a skip |

Passed, among the gates the notes name:

- `native_grouped_parity`, `iq_multi_parity` and its `_kq_nov2k` / `_kq_swiglu` / `_kq_nosub16` variants;
- `mmvq_il_parity`, the eleven `native_expert_parity_*`;
- `ple_reader_selftest`, `pool_affinity_test`, `pool_plan_test`, `host_graph_launch_test`;
- `ggml_tq_overlay` (the overlay's configure-time test, §B2).

**Re-run after `1bd671c7`** (`build-m145u`, the tree with the test fixes of `f20ec6ab` and `1bd671c7`): **only
`ple_parity` fails**, the known one. The three trellis CPU tests skip (77) on the default ggml, and the pool-count
check passes against the plan (#268's comment "Gates, round 2").

Not run:

- **`native_q2_pool_hit_parity`** (ADR 0001, bit-exact) is not a ctest target, as at 0.1.42, so the log does not have
  it. gpu.md asks for it with `STRATA_GU_SWIGLU=1` because of §B2's guard. TODO.
- ctest of `build-m145g`, after building its test targets. TODO.

### pytest

**Final runs** (`C:\Strata-exp\m020-notes\pytest-fixes.md`, HEAD `484eaaf0`, Python 3.11.9):

- `python -m pytest serve tests/xeno -q -p no:cacheprovider`: **1 failed, 1802 passed, 28 skipped, 10637 subtests
  passed** (629 s). The one failure is `serve/test_lifeline.py`, which needs `psutil` (not installed on this box).
- Before the fixes: 30 failed (15 test functions). One was a real merge bug, fixed in `484eaaf0` (§6a): an Anthropic
  `thinking.budget_tokens` was cut twice. The rest were tests that assumed upstream's defaults (model-card sampling,
  `answer_room`, `fit_max_tokens`) or a stand-in tokenizer; each was adapted to the fork's default, with its reason in
  the notes.
- `pytest tools`: collection used to stop on the `tools/*_test.py` scripts; `conftest.py` skips them now (`484eaaf0`).
  13 merge-only failures were found in the tools tests; `443ba079` fixes setup's Gyro menu order (before NVFP4, as
  upstream), keeps the fork's Thai default (two `"cjk"` -> `"thai"`), lists Gyro-M before Gyro-S in the MCP fallback,
  and makes `test_setup_engine_hash` hermetic. Per-file re-runs pass, apart from the baseline's `SmallCardNote` and a
  flaky download timing test (1 in 4).

The first two attempts, for the record:

- `serve` + `tests/xeno` stopped at the command line: `unrecognized arguments: --timeout 600`. This box's pytest has no
  `pytest-timeout` (`C:\Strata-exp\m145-pytest.log`).
- The `tools/test_*.py` run stopped with an INTERNALERROR in collection, "llama.cpp's gguf-py was not found ... or set
  STRATA_GGUF_PY" (5 errors, `C:\Strata-exp\m145-pytest-tools.log`).

Those attempts' open points are closed by the final runs: `NonFiniteLogits` passes, `tools/test_setup_gyro.py` passes
(32), and the `fit_max_tokens` tests follow the fork's default.

### D2x state and replies

`ab145.py hash` (the 0.1.42 `ab212.py` shape) ran on the served D2x config, `max_tokens` 1 and `STRATA_STATE_HASH=1`.
It reads a 9,731-token prefix, then six parts that branch from it (Claude Code's request shape). Boots in the order
base / merge / merge / base.

**All 7 `STATE_HASH` lines are identical in all 4 boots**, every fingerprint (`gdn`, `ple`, `tail`, `pooled`, `kv`,
`mtp`, `stale`, `dead`, `pooled_full`). The replies are identical in every part.

Files: `ab145-hash-{base,new}-*.{json,log,server.txt}`; the results are in `ab212-hash-results.json` (the script keeps
its 0.1.42 output name).

The merge arm ran with every default the two steps turn on: the sm_120 set, `STRATA_GU_SWIGLU`, `STRATA_COMMIT_BATCH`,
`STRATA_NAN_GUARD`, the pin reserve, and the K-quant and trellis MMQ instances compiled in. So this is D2x's prompt-path
parity with those defaults on. Decode rounds are covered by the generate runs below.

**Boot lines** (`ab145-speed-*.log`): the same in every boot of both arms.

- "expert cache sizing: 9.38 GiB free, 700 MiB reserve, budget 11.25 GiB -> cap 8.70 GiB";
- 6485 CUDA0 slots, and 2992 secondary experts staged (4.00 GiB);
- "the prompt path borrows 2369 CUDA0 cache slots";
- "544 MiB of VRAM free with everything loaded".

So the K-quant and trellis MMQ instances cost D2x no VRAM that the boot shows (§2, §B4).

The private commit is 0.17-0.18 GiB higher on the merge, before the expert load and after it: "private commit 22.86 ->
43.26" and "22.88 -> 43.28" on the base, "23.05 -> 43.44" and "23.03 -> 43.43" on the merge. The cause is not split.

The merge's boots print two new lines:

- "pinned-memory headroom: holding 800 MiB ..." (§B2);
- "strata cpu pool: warning: 13 CPU expert workers are configured ... Recommended: --pool-workers 9" (§B2).

### D2x: Claude Code's parts (`ab145.py speed`)

The same config, prefix and parts, greedy, 32 tokens each, then a 400-token answer on a 68-token prompt.

**Round 1** (`ab145-speed-round1.json`), 8 boots, base / new / new / base / new / base / base / new. The table gives
medians of 4 per arm, as prompt read ms:

| part (tokens read) | base | merge | merge - base |
|---|---:|---:|---:|
| prefix (9,731) | 7,640 | 7,617 | -24 |
| 400 (465) | 1,306 | 1,239 | -66 |
| 700 (784) | 1,511 | 1,429 | -82 |
| 1000 (1,088) | 1,497 | 1,461 | -36 |
| 1412 (1,474) | 1,669 | 1,581 | -88 |
| 1800 (2,120) | 2,102 | 1,974 | -128 |
| 2400 (2,428) | 2,135 | 2,178 | +42 |

All 8 boots gave one reply set.

**Round 2, the bisect** (`ab145-bisect.out`, `ab212-speed-results.json`), 10 boots. It has base x2, merge x2, and the
merge with one default off per arm, x2 each: `STRATA_SM120_PERF=0`, `STRATA_NAN_GUARD=0`, `STRATA_COMMIT_BATCH=0`.

- The merge reads every part faster than the base again, by 31-149 ms, and the prefix by 360 ms (medians of 2: 400
  1,179 against 1,246; 1800 1,852 against 2,001).
- None of the three switches brings the base's times back: each switch arm's parts stay below the base's and within
  60 ms of the merge's.
- So the parts' gain comes from none of those three defaults. Which change gives it is not measured; a timeline of one
  part per arm would show it.

### D2x decode

**Served, one 400-token answer per boot:** these single answers do not settle decode.

| boots | tok/s | median |
|---|---|---:|
| base, round 1 | 75.6, 71.8, 77.1, 77.0 | 76.3 |
| merge, round 1 | 66.6, 69.4, 68.7, 72.2 | 69.1 |
| base, round 2 | 75.6, 76.2 | |
| merge, round 2 | 73.0, 67.9 | |
| `SM120_PERF=0` | 70.0, 70.6 | |
| `NAN_GUARD=0` | 70.8, 69.2 | |
| `COMMIT_BATCH=0` | 73.5, 80.1 | |

The served config runs adaptive swaps (`--adapt-swaps 8`), so each boot places experts differently.

- Every merge boot is slower than 5 of the 6 base boots, and 4 of the 6 are slower than all of them. That is a
  consistent lean, and repeated rounds count (the fork's rule), so it is not dismissed as noise.
- The generate runs below, with swaps off and identical outputs, show the opposite.
- No switch arm restores the base consistently.
- **Settled by the 23-turn served ABBA below:** the lean was a placement accident of that one answer. With swaps off,
  the two engines draft alike (base 207 / 299 and 212 / 315 accepted / offered, merge 207 / 300 twice) and the merge is
  as fast or faster (81.4 / 79.7 against 78.8 / 79.7). `STRATA_COMMIT_BATCH` was not the cause (`e2120ea6` made it
  opt-in anyway; that does no harm).

**Served, 23 turns** (`C:\Strata-exp\gyro-record\run-m145ab.sh`, `turns217-abm145-*`; order C1 F1 F2 C2 F3 C3 C4 F4;
C = `run-main-e573u`, F = `run-m145u` sha256 `68789427991f`; the same server `src-main` for both):

| arm | decode median | `ms_verify` | `ms_pool` | `ms_gpu_wait` | `ms_draft` | acceptance | 5060/4070/CPU | prompt read, 23 turns |
|---|---|---|---|---|---|---|---|---|
| C1 | 76.8 | 30.02 | 14.48 | 14.43 | 3.19 | 0.699 | 0.687/0.145/0.169 | 213.5 s |
| C2 | 79.2 | 29.90 | 14.45 | 14.31 | 3.10 | 0.698 | 0.685/0.148/0.167 | 217.3 s |
| F1 | 81.0 | 28.60 | 13.85 | 13.62 | 2.54 | 0.685 | 0.680/0.150/0.170 | 215.9 s |
| F2 | 82.8 | 28.92 | 14.06 | 13.72 | 2.57 | 0.705 | 0.679/0.146/0.174 | 214.4 s |
| F3 | 83.2 | 28.21 | 13.43 | 13.64 | 2.53 | 0.701 | 0.701/0.144/0.155 | 216.2 s |

- Per turn against the mean of C1 and C2: **x1.038, x1.058, x1.059**, with 17, 20 and 20 of 23 turns faster.
- The round is 1.0-1.8 ms shorter: `ms_gpu_wait` -0.7 ms and `ms_draft` -0.6 ms (a cheaper MTP round), the pool a little.
  Acceptance about 0.70 in both; prompt read about 215 s in both. Private commit at boot 43.27 -> 43.42 GiB (+0.15).
- C3, C4 and F4 did not finish: the PC's commit limit (83 GB) ran out with the desktop's processes holding 41.4 GB
  (base: `bad_alloc` in `conversation_checkpoint_save`; merge: `VirtualAlloc(MEM_COMMIT)` error 1455 in a paired swap).
  An environment limit hit by both arms, not a merge bug.

**Generate mode, swaps off** (`dec145.py`):

- Setup: the served D2x args with `--adapt-swaps 0`, 256 tokens, `--greedy`, ABBA per prompt with the switch arms in
  between.
- Files: `dec145-a.out`, `dec145-runs\<tag>\on.{stdout,stderr}`.
- **Outputs are identical in all 16 runs:** `2d2da1f7` (code, `n0-py-async`) and `10d41d91` (Thai, `n0-thai-net`).

| prompt | base (A ... A) | merge (B, B) | `COMMIT_BATCH=0` | `SM120_PERF=0` |
|---|---|---|---|---|
| code | 93.27, 90.89 | **101.29, 101.08** | 97.67, 99.69 | 98.91, 97.24 |
| Thai | 70.24, 75.01 | 72.90, 74.74 | 74.84, 73.98 | 74.11, 73.00 |

Counters on the code prompt, base -> merge (each pair's first run, from the same stdout):

- `tests/xeno/latency_breakdown.py`: the round goes from 31.92 to 29.39 ms;
- `dispatch detail` CPU pool: 10.990 -> 9.434 ms/round (second pair: 11.046 -> 9.525). This is the stage that moved most;
- `pool multi` gate/up: 10.860 -> 9.425 ms/round, 23.5 -> 27.1 GB/s;
- `verify window` wait for rings: 13.318 -> 12.921;
- the CPU pool's time per token is 3.69 -> 3.17 ms (CPU pool ms/round over 2.98 tokens per round);
- these are the same in both arms: tier hits primary 77.3 %, secondary 10.8 %, PCIe 0.0 %, CPU 11.9 % of 153,120 routed
  entries; drafts accepted 170 of 233 (0.730), 2.98 tokens per round; 3.69 distinct / 4.40 routed CPU experts per layer.

The CPU work per round is the same (the same experts, the same hit split). So the pool ran the same rows faster. Which
change in the pool did it is not measured. The candidates in the merge are step 2's per-job layout in `native_rows` and
the pool plan code (§B3), and step 1's per-layer geometry and STRATA_POOL_UMWAIT structure (§3). A pool timeline
(`STRATA_TIMELINE`) of one run per arm would split it.

On the Thai prompt the arms overlap (base 70.2 / 75.0, merge 72.9 / 74.7): no change is shown there.

### Gyro-M

**Parity with swaps on (first run, decides nothing).** `ab145g.py hash` ran with key 3's args: `strata-gyro-m-d2x.json`, with its profile
copied into the record folder so the served profile is not touched. Arms:

- base: `run-main-e573g`;
- new: `run-m145g`;
- ovl: `run-m145u`, the default ggml with step 2's trellis MMQ overlay.

Boots base / new / ovl / ovl / new / base. Results:

- **The replies are identical in all 6 boots.**
- The `STATE_HASH` lines differ even between the two base boots. Key 3 runs adaptive swaps (`--adapt-swaps 48`), so a
  boot is not reproducible, and the hashes decide nothing.
- Files: `ab145g-hash-*.{json,log,server.txt}`, the results in `ab145g-hash-swapson.json`.

**Parity with swaps off** (`ab145g.py hash`, key 3's args with `--adapt-swaps 0 --adapt-secondary 0 --expert-cache 5900
--secondary-expert-mib 4900`; boots base / new / ovl / ovl / new / base): **all 7 `STATE_HASH` fingerprints identical in
all 6 boots, and the replies too.**

**The overlay question: answered, yes.** The `ovl` arm is `run-m145u`, the default ggml with upstream's trellis MMQ
overlay. It is bit-identical to the agentionai-ggml builds above, and D2x keeps its own outputs on the same build (the
D2x hashes). So one default-ggml engine can serve D2x and Gyro-M (#258).

**The served Gyro-M engine** is `xeno/defer` (`main` + the Gyro-M work), not `main`. The merge was carried into it as
`xeno/defer-m145` (`de49e29e`: `xeno/dense-req-groups` + this branch at `1bd671c7`; 14 files, 45 hunks, all fork
against fork, notes `m020-notes\defer-{gpu,core}.md`; then `86094fb9` takes `484eaaf0` and `443ba079`, serve / setup /
tests only). Exe `run-gyro-d145u` (default ggml + overlay, sha256 `90938462db75`).

- State against the installed `run-gyro-tq` (`a9c70f32`, agentionai ggml, sha256 `c524c8d8ded7`), swaps off, boots
  tq / d145 / d145 / tq: all 7 fingerprints identical in all 4 boots; replies identical.
- Served 23-turn ABBA on key 3's config (`turns217-g145-*`, `turns217-g145b-*`): every boot of the merge has
  `ms_gpu_wait` 17.2-17.5 against 18.0-18.2 ms and `ms_draft` 2.88-2.94 against 3.48-3.58 ms. In its normal boots the
  merge is x1.007-1.019 per turn against the mean of six installed-exe boots (g145b), and x1.027-1.087 against one
  (g145). **3 of its 10 boots ran the CPU pool 1.5-3.2x slower per entry for the whole boot** (x0.556-0.825 per
  turn); the installed exe did so in none of its 7. RAM, E-cores and the pool code's Windows defaults are ruled out;
  the cause is open (#275).

### Not run yet, and not planned unless asked

TODO, part of the merge gate:

- `native_q2_pool_hit_parity` with and without `STRATA_GU_SWIGLU=1`.
- #275 (Gyro-M's slow-pool boots) before the Gyro-M install.

Not planned for the merge gate unless asked:

- the opt-in arms of §2 and §B2; capacity mode; a SAVE / RESTORE round trip;
- a HIP / Linux / SYCL / Metal build;
- the image generation path; Gyro with pictures; the app / Terminal / Control-mode setup paths.

## 6. Commits and the exe

- Base: `main` `e5737a27`. Branch `xeno/268-merge-0.1.45`.
- Upstream: `v0.1.45` (`477ac13c`), then `v0.2.0-beta` (`3d1fb480`).
- The commits, oldest first (`git log --first-parent 1faf709a^..f20ec6ab`):

| commit | what |
|---|---|
| `1faf709a` | step 1's merge commit (parents `e5737a27`, `477ac13c`); not built at this commit |
| `33ab01d5` | the four 2-argument `native_expert_scratch_bytes` callers; README / DETAILS say the fork's sampling defaults (§6a) |
| `f06366df` | step 2's merge commit (parents `33ab01d5`, `3d1fb480`); not built at this commit |
| `8ed30815` | the Windows build of v0.2.0: one `Prefill::Impl::Xr`; a `_putenv_s` shim for `setenv` in three upstream tests (§B5) |
| `f20ec6ab` | the test fixes: the pool-count check asks upstream's pool plan; the trellis CPU tests skip without trellis ggml types (§B5) |
| `e2120ea6` | `STRATA_COMMIT_BATCH` opt-in on the fork (=1). The commit gives a reason later shown wrong (§5, D2x decode); the opt-in is kept, it does no harm |
| `1bd671c7` | `native_hadamard_test`'s skip line kept its closing brace |
| `484eaaf0` | serve: one thinking budget for Anthropic requests again (§6a); tests follow the fork's defaults; `conftest.py` skips the `tools/*_test.py` scripts |
| `443ba079` | setup: the Gyro family keeps upstream's menu place and the fork's Thai default; MCP lists Gyro-M first; tools tests follow the fork |

- **Engine exes.** The first pair was copied 2026-10-19 23:09 from the tree of `8ed30815`; they were rebuilt 2026-10-20
  00:35 from `e2120ea6`. The commits after it change only serve, setup and tests, so the engine source of the second
  pair is that of `443ba079`.

| exe | ggml | sha256 (first, `8ed30815`) | sha256 (now, `e2120ea6`) |
|---|---|---|---|
| `C:\Strata-exp\run-m145u\strata.exe` (from `build-m145u`) | default pin `3cf03257`, with the trellis MMQ overlay (`build-m145u\ggml-tq-overlay`) | `b45be245dac7...` | `68789427991ff2dd299d877fe379d90ee13c43d49c03c8f62f16db96364a8c2b` |
| `C:\Strata-exp\run-m145g\strata.exe` (from `build-m145g`) | agentionai's (`STRATA_GGML_DIR`) | `3f03c991d1c4...` | `5c8cdcce6130727be98c24dc52d386916aa591ac4e7db78a0b549ed025ebb0c0` |
| `C:\Strata-exp\run-gyro-d145u\strata.exe` (from `build-defer145u`, `xeno/defer-m145` `de49e29e`) | default pin + the overlay | | `90938462db75fb3bdbc4f8ac05db03c4501ea632de7e38145a997c3eb2b38098` |

- The engine reports version 0.1.45.3: CMake `project(strata VERSION 0.1.45.3)`, from upstream 98b865c5.
- The baselines' exes are in §5.
- This report is not committed yet. It is in the worktree. Its file name names only the first step; a later commit may
  rename it.

## 6a. Bugs the merge introduced

**Auto-merge breakage found by the area sweeps before the first build** (the evidence for each is in the area's notes
and §4):

- code that would not compile against the fork's side: the duplicate `kGatherGroupMax` (prefill.md); `compute`'s
  `last_of_group`, not defined in the fork's `compute` (prefill.md); the by-need arena's 7-argument `reserve()` call
  (core.md); two `lock_resident_ranges` definitions (core.md); a `lend` without upstream's `seg_lend`, which
  `read_part` reads (generate.md hunk 21);
- behaviour the auto-merge broke: 653f0113's `stage_n = 32` indexing the fork's 8-slot staged ring (prefill.md); base's
  `pp` / `help_pp` members pulled back into `Impl` (prefill.md); the five serve-loop `return 1` (generate.md); a padded
  arena written at packed offsets by placement-first's loader (core.md); the PCIe-share block skipped under
  STRATA_PCIE_BALANCE (core.md); a double hit add had upstream's trailing `moe_hit_add` been taken (core.md);
- `tools/check_isa_guard.py` committed by upstream with a stray `\r\r` (serve.md).

**Found by the first build** (`build-m145u.out` line 1514, `build-m145g.out` line 1512):

- **`native_expert_scratch_bytes` with upstream's two-argument signature.** The fork's is `(cap_entries, n_ff, n_embd)`
  (iq_kernels.hpp:75; the fork's kernels size the scratch by `n_embd` too). Upstream's new code calls `(cap, n_ff)`:
  - `src/qwen35/q36_engine.cu:1414`: `error: too few arguments in function call` in both build dirs; now
    `(T * Kp, FF, N)`;
  - `src/kernels/expdec_bench.cpp:237` (an `EXCLUDE_FROM_ALL` target, so the default build would not reach it): now
    `(kMaxEntries, kFf, kEmbd)`.

  The 0.1.42 merge had the same class of break in `native_expert_bench.cpp:96`.
- **Two more call sites of the same kind, found by grep:** `src/kernels/nvfp4_decode_bench.cpp:107` `(E, FF)` and
  `src/kernels/nvfp4_expert_gpu_parity.cpp:82` `(T, FF)`. Both are in the default build under `STRATA_ENABLE_CUDA`
  (CMakeLists.txt:2085-2089). (The `sycl/` and `metal/` call sites are not built.)

  All four are fixed in `33ab01d5`. Step 1's tree was not built again after that. The next build was of the tree of
  both steps, and its errors are step 2's (§B5).

**Found while drafting this report: the docs described upstream's defaults, not the fork's opt-in.** Fixed in
`33ab01d5` (README.md, docs/DETAILS.md). The merge commit made model-card sampling and `answer_room` opt-in (§2), but
the text merged from upstream said they are on:

- `README.md:243-245`: "since upstream v0.1.45 a request without sampling settings, on a config without a
  `"sampling"` block, gets the model card's recommended sampling, not greedy decoding";
- `docs/DETAILS.md:877-881`: a thinking request with `max_tokens` of 1,024 or more "gets one" (a budget), and
  `"answer_room": false` turns it off;
- `docs/DETAILS.md:1408`: "With no block at all (0.1.45) a request without sampling keys gets the sampling the model
  card recommends".

Each now states the fork's rule: off unless `"model_card_sampling": true` / `"answer_room": true`.

**Found by the test runs and fixed in code:** the ctest failures of the tree of both steps came from step 2 (§5,
§B5). pytest found one real bug, fixed in `484eaaf0`:

- **An Anthropic `thinking.budget_tokens` was cut twice.** Upstream's `serve/frontend.py` (194c2688, auto-merged)
  copies it into `reasoning_budget_tokens`, so the fork's own budget handling (S3's `_think_budget` and #123's budget)
  saw it a second time. `484eaaf0` removes the copy; the fork's path reads the Anthropic field once, as before the
  merge. This was §7's "may be counted twice" follow-up; it is closed.

## 7. What is left

- **Whether D2x and Gyro move to the merge's exes:** the developer's call, on §5. D2x: a measured win with identical
  state. Gyro-M: wait for #275.
- **Before the served A/B: the #247 / #188 fixes.** The merge is built on `main`, which lacks `xeno/257-dequant`'s
  #247 PLE read-ahead lock and #188 wave ring. Those were cherry-picked there because a main-built engine crashed
  serving D2x (`11ded277`; the first `run-d2x-1018` install crashed at turn 11). There are two ways:
  - merge or cherry-pick that branch's five commits onto the merge branch before the 23-turn served run;
  - or run it against the `e5737a27` build (`run-main-e573u`, the baseline of §5's first gates), knowing the crash risk
    is the same on both arms and unmeasured.
- **Step 2's open items** are listed in §B9.
- **Refused combinations** (§2): `--prefill-head-alias` (a port of upstream's fixed head-alias layout into the fork's
  per-part `lend`); NVFP4 packs (the prompt path refuses them at `Prefill::init`; generate.cpp should refuse such a pack
  before loading, with the same reason, or the docs must say the fork does not run NVFP4); STRATA_ADAPT_OVERLAP with the
  paired swaps; STRATA_PCIE_BALANCE and STRATA_PIN_BY_NEED with the fork's tiers.
- **Follow-ups:**
  - **653f0113 as an opt-in port:** grouped gathers on the fork's staged walk (ring 8 -> 32 slots). Latency case: one
    wait and one event per group instead of per expert on short Claude Code parts. Measure the ceiling first (the
    per-expert waits of a short part in a timeline), then build it behind a switch.
  - **`serve/ui` port of `e8634865`** (§8).
  - The Anthropic budget counted twice: found by pytest and fixed in `484eaaf0` (§6a).
  - **`blob_at()` with capacity mode's slab** is not thread-safe; refuse `--adapt-async 1` with the slab, or document it.
  - **STRATA_HC_REQ8 with the wave** races on a static scratch; refuse it with the wave, or document it.
  - The K-quant CMake defaults: closed. §5's boot lines show no VRAM or slot cost on D2x.
- **Opt-ins not measured on the fork's walk:** every switch of §2's "the rest", and in particular `--pool-affinity
  measure`, `--host-core sibling`, STRATA_POOL_UMWAIT, STRATA_EXPDEC, STRATA_ADAPT_OVERLAP (unpaired),
  STRATA_CKPT_POOL / _THREADS, STRATA_RAM_ADAPT, the elastic vision encoder, `model_card_sampling` and `answer_room`.
- **Developer decisions:**
  - the model-card sampling and `answer_room`: kept opt-in by the merge commit; whether D2x or Capacity want them on;
  - the K-quant / Q6_K CMake defaults for the D2x and Gyro builds (no boot cost measured on D2x, §5);
  - `--prefill-head-alias`: port or keep refused;
  - the 0.1.42 questions still open: no switch restores upstream's PCIe probe default; the #626 host-thread pin has no
    switch back; helper caches with the 4070 tier are not refused.
- **Untested:** the #879 port (new code); `answer_room` and model-card sampling with their keys on; a non-finite retry
  with a forced opening (ends as "length", not 503, §2).
- **Cosmetic, left as it is:** after a NaN mid-reply that ends as "length" while still thinking, `_run`'s record also
  prints the "#530 reached max tokens while still thinking" hint.
- **Gates:**
  - `/code-review` (two axes) and `/scrutinize` on both merges' hand resolutions (`git show --remerge-diff 1faf709a`,
    `git show --remerge-diff f06366df`);
  - `/security-review` for the new routes (image generation, `/v1/images/files/`; step 2's `/v1/telemetry` and its read
    key).
- **`docs/BLUEPRINT.md`:** each step has a revision-log row and a "Not yet in the baseline" row (blueprint lines 29-30,
  376-377). When the tree is verified, the rows need the names of §9 and §B8 that reach D2x, and the facts of §2 and
  §B2.

## 8. The classic web app

Upstream changed `serve/web/` in 3 files, +86/-8 (`app.css`, `app.js`, `index.html`; `git diff --stat 61b3fb5d
477ac13c -- serve/web`), in two non-merge commits. The classic app is kept untouched and served at `/classic/` (xeno
#67). The merge did not touch `serve/ui/` (`git diff e5737a27 1faf709a -- serve/ui` is empty).

| commit | change | the new app (`serve/ui/`) |
|---|---|---|
| `e8634865` | the page sends no sampling until the user changes it (a stored 0.6 counts as not set), so the server's model-card defaults apply (#2685) | **needs a port.** The new app defaults to 0.6 and always sends temperature / top_p / top_k (`serve/ui/src/lib/chat.ts:96` `DEFAULTS`, `:750` the request body; `pages/chat/SettingsSheet.tsx:19-20` shared defaults). With `"model_card_sampling": true` the card's defaults would never apply to the new app's chats; with it off (the fork's default) the port changes nothing yet |
| `a5b3b6c2` | an Image generation tab (Qwen-Image through `/v1/images/generations`, hidden unless `/health` says `image_generation`) | optional: the new app has no images page, and upstream hides the feature in setup |

## README

`README.md` is the fork's own (#166); the conflict (3 hunks, and one auto-merged section) was resolved on the fork's
file:

- "last merged" bumped from v0.1.42 to **v0.1.45**;
- carried by hand from upstream's user-facing changes: Apple Silicon Macs (M1-M5, macOS 15+, `./setup.sh` builds the
  Metal engine, `xcode-select --install`); the models setup now offers (Unsloth UD-Q4_K_XL to Q8_0, UD-Q4_K_XL no
  longer experimental, NVFP4 packs, Qwen3.6-35B-A3B), in a subsection marked "upstream's, not measured by the fork";
  the sampling default change (#2685) with how to get greedy back; `BENCH.bat` / `./bench.sh`; the two opt-in one-card
  settings (`--adapt-async 1`, `STRATA_PIN_BY_NEED=1`), said to be unmeasured by the fork;
- not carried: upstream's speed figures (its RTX 5070 numbers among them), and the "Is it fast" paragraph's wording.

The README's sampling bullet described upstream's default. With the merge commit's opt-in it holds only with
`"model_card_sampling": true`; `33ab01d5` gave it that rule (§6a). Step 2's README changes are in §B7.

## 9. Configuration surfaces this merge adds (from `tools/blueprint_check.py`)

The script reads the staged diff, which is empty after the commit, so its own `reasons()` was run on `git diff -U0
e5737a27 1faf709a` (the merge against `main`). Grouped by the program that reads each name; upstream's own docs
(`docs/DETAILS.md`, `docs/MAC.md`, `docs/EXPERTDEC.md`) describe its names, and the ones that reach D2x, with their
defaults, are in §2. Nothing was removed.

**`strata` engine flags** (generate.cpp, 4): `--conversation-cache-park-gain`, `--prefill-head-alias` (refused),
`--prefill-pipe`, `--prompt-cache-spread`. New values of existing flags, which the regex cannot see: `--host-core
sibling`, `--pool-affinity measure`.

**GEN keys** (generate.cpp, 2): `img_keys`, `pipe`.

**Run-config keys** (serve/server.py, 11): `answer_room` (opt-in on the fork), `arch`, `before_load_ignore_failure`,
`before_load_strict`, `before_load_timeout_s`, `elastic`, `elastic_idle_s`, `elastic_mib` (the `vision` block),
`env_to_user`, `images`, `model_card_sampling` (the fork's, #268); plus the `images` block's own keys in
`serve/images.py` (16: `diffusion_model`, `gpu_policy`, `keep_files`, `llm`, `max_n`, `max_side`, `native`,
`need_mib`, `out_dir`, `quant`, `reload`, `sigmas`, `steps`, `timeout_s`, `uma`, `vae`). Checked against the tree:
every one is absent from `main`'s `serve/` and `setup.py`. (`args`, `backend`, `cwd`, `env`, `exe`, `gpu`, `lib_dirs`,
`log`, `model_name`, `vision`, `vram_elastic`, and setup.py's `api_key`, `cuda`, `gpus_asked`, `host`, `open_browser`,
`parallel`, `port` also show as "added" because new code names them; they are existing keys.)

**Other programs' flags and keys** (not `strata.exe`): the Qwen3.6 engine `src/qwen35/q36_engine.cu` (`--batch`,
`--batch-mtp`, `--bench-pp`, `--bench-tg`, `--chunk`, `--cpu-threads`, `--ctx`, `--draft-vocab`, `--dump-full-logits`,
`--emb`, `--expert-slots`, `--expert-vram`, `--force`, `--gpu`, `--greedy`, `--hybrid-rows`, `--max-context`,
`--max-new`, `--model`, `--mtp`, `--n`, `--native`, `--no-graph`, `--no-token-graph`, `--no-warmup`, `--pool-workers`,
`--prefill`, `--serve`, `--slots`, `--tokens`, `--vision`; GEN keys `img_keys`, `min_p`, `penalty_freq`,
`penalty_last_n`, `penalty_present`, `penalty_repeat`, `seed`, `temperature`, `top_k`, `top_p`) and its reference
`q36ref.cu` (`--act`, `--dump-layers`, `--dump-logits`, `--kernels`, `--tokens-file`, ...); the benches
`kq_cpu_bench.cpp` (`--mb`, `--nt`, `--pairs`, `--ram-gbs`, `--reps`, `--threads`) and `q8_gemv_bench.cpp` (`--bw`,
`--cols`, `--iters`).

**Environment variables.** The script's regex finds 162 names in the diff. 19 of them were already in `main`'s tree
and are only read in a new place (`STRATA_ARENA_LOCK`, `STRATA_BATCH_MTP`, `STRATA_DBG_GDN`, `STRATA_DRAFT_TIMING`,
`STRATA_EXP_MODE`, `STRATA_GDN_CHUNKED`, `STRATA_GDN_COMMIT_SPLIT`, `STRATA_GDN_SPLIT`, `STRATA_GFX12_DEFAULTS`,
`STRATA_GFX906_ATTN_QUERY_SWIZZLE`, `STRATA_LOGPOS` (the fork's own), `STRATA_MTP_FULL_HEAD`, `STRATA_NO_LARGEPAGES`,
`STRATA_NO_MMVQ_ROWS2`, `STRATA_Q8_PACKED`, `STRATA_REMOTE_SPIN`, `STRATA_SELECT_SIMT`, `STRATA_TEST_PAGEABLE`,
`STRATA_VERIFY_NODES`). 23 more are new to the tree but read through helpers the regex does not match (a quoted
`"STRATA_..."` in `src`, `include`, `serve`, `setup.py` or `CMakeLists.txt` that `main` does not contain). The new
names, 143 + 23 = 166, by area:

- **Decode, verify and MTP** (verify.cpp, mtp.cpp, kernels): `STRATA_BATCH_ATTN_ROWS`, `STRATA_BATCH_GDN_GROUPS`,
  `STRATA_BATCH_MTP_SERIAL`, `STRATA_BATCH_SELF_COMMIT`, `STRATA_COMMIT_BATCH` (on), `STRATA_COMMIT_TIMING`,
  `STRATA_COPY_HIP_PRIO`, `STRATA_DUMMY_NODES`, `STRATA_FUSE_CONVAB`, `STRATA_FUSE_HITS`, `STRATA_FUSE_WAIT`,
  `STRATA_MTP_HIP_PRIO`, `STRATA_MTP_PROFILE`, `STRATA_MTP_SIDE_STREAM`, `STRATA_MTP_STAGE0`, `STRATA_NAN_GUARD` (on),
  `STRATA_NAN_INJECT`, `STRATA_QFUSE_HC`, `STRATA_QFUSE_HC_LOCAL`, `STRATA_QIL`, `STRATA_QSA_SLOTS`,
  `STRATA_QSA_SPLIT_ATTN`, `STRATA_SIDE_MIXER`, `STRATA_SPLIT_WAIT_MS`, `STRATA_VERIFY_ASYNC_LAUNCH`; the expert
  kernels `STRATA_EXPDEC`, `STRATA_EXPDEC_D20`, `STRATA_EXPDEC_DOWN`, `STRATA_EXPDEC_DR`, `STRATA_EXPDEC_DRB`,
  `STRATA_EXPDEC_MINB`, `STRATA_EXPDEC_MINBLK`, `STRATA_EXPDEC_R`, `STRATA_EXPDEC_RB`, `STRATA_B6_MMVQ_ROWS`,
  `STRATA_MMVQ_IL_Q3K`, `STRATA_MMVQ_IL_T1`, `STRATA_Q2_WARP`, `STRATA_CPU_IQ_ALL_EXACT1`, `STRATA_NO_NVFP4_256`,
  `STRATA_NO_NVFP4_512`, `STRATA_NVFP4_PREFETCH`, `STRATA_DEQUANT_WIDE`; attention and selection
  `STRATA_ATTN_DAILY`, `STRATA_ATTN_MERGE_V2`, `STRATA_ATTN_V3`, `STRATA_QSA_ROWS`, `STRATA_SCORES_HALVE`,
  `STRATA_SCORES_THREAD`, `STRATA_SCORES_THREAD_TRACE`, `STRATA_SIMT2_TILE`, `STRATA_TOPK_REG_MIN_BLOCKS`,
  `STRATA_TOPK_SAMPLE`, `STRATA_TOPK_SPLIT`; the fused residual / hyper-connection kernels `STRATA_GR_DOWN_TG`,
  `STRATA_GR_NORM_FAST`, `STRATA_GR_UP75`, `STRATA_HC_PERSIST`.
- **Prompt path** (prefill.cpp, moe_mmq.cu, gemm.cu, attention): `STRATA_DENSE_F32`, `STRATA_DUMP_MOE_INPUT`,
  `STRATA_DUMP_MOE_LAYER`, `STRATA_F16_RANGE`, `STRATA_HANDOFF_PIPE`, `STRATA_HANDOFF_PIPE_WORKER`, `STRATA_MMQ_DIRECT`
  (ignored on the fork), `STRATA_MMQ_DIRECT_LOG`, `STRATA_MMQ_GROUP_SORT` (ignored on the fork), `STRATA_MMQ_KQ`,
  `STRATA_PA_V1_PREFETCH`, `STRATA_PA_V2T`, `STRATA_PF_F16`, `STRATA_PF_F16_BN`, `STRATA_PF_FUSED_SM75`,
  `STRATA_PREFILL_NVFP4` (refused on the fork), `STRATA_PREFILL_PIPE`, `STRATA_PROMPT_ATTN_IMMA`,
  `STRATA_PROMPT_ATTN_Q4_TC`, `STRATA_QSA_DUMP_LAYER`, `STRATA_QSA_DUMP_STEPS`, `STRATA_QUANT_GATHER`,
  `STRATA_XN_F16`.
- **Engine set-up, placement, adaptive tier, serve loop** (generate.cpp, expert_source.cpp, pinned.cu, layer.cpp,
  pool.cpp): `STRATA_ADAPT_ASYNC_BATCH`, `STRATA_ADAPT_ASYNC_WAIT`, `STRATA_ADAPT_CALLBACK_WARMUP`,
  `STRATA_ADAPT_CHECKRES`, `STRATA_ADAPT_EVICT_SYNC`, `STRATA_ADAPT_OVERLAP`, `STRATA_ADAPT_OVERLAP_RESIDENT`,
  `STRATA_ADAPT_OVERLAP_SLOW_US`, `STRATA_ADAPT_PICK_INLINE`, `STRATA_ADAPT_SIGNAL`, `STRATA_ADMIT_MERGE_FIRST`,
  `STRATA_ARENA_THP`, `STRATA_CKPT_POOL`, `STRATA_CKPT_THREADS`, `STRATA_FILE_CACHE_OLD_RULE`, `STRATA_GFX12_CHUNK`,
  `STRATA_HC_REQ8`, `STRATA_KV_HOST_REGISTER`, `STRATA_PCIE_AUTO_SPLIT`, `STRATA_PCIE_BALANCE` (held off with the
  fork's tiers), `STRATA_PCIE_FRAC_DEFAULT`, `STRATA_PCIE_MIN1`, `STRATA_PIN_BY_NEED` (held off with a pageable arena),
  `STRATA_PIN_BY_NEED_SKIP`, `STRATA_PIPED_CONCURRENT`, `STRATA_PIPED_CONCURRENT_PRIO`, `STRATA_PIPED_CONCURRENT_UNPIN`,
  `STRATA_PIPED_DECODE_MAX_MS`, `STRATA_PIPED_DECODE_MIN_MS`, `STRATA_PIPED_READ_PIECE`, `STRATA_PIPELINE_AUTO`,
  `STRATA_PIPELINE_AUTO_MIN_STAGES`, `STRATA_PIPE_BYIELD_NODRAIN`, `STRATA_PLE_AUTO_MMAP`, `STRATA_POOL_UMWAIT`,
  `STRATA_POOL_UMWAIT_C02`, `STRATA_POOL_UMWAIT_ESC`, `STRATA_PRIMARY_DEVICE`, `STRATA_PROMPT_DIRECT`,
  `STRATA_RAM_ADAPT`, `STRATA_SLOT_CKPT_MOVE`, `STRATA_SLOT_COPY_BATCHSYNC`, `STRATA_SLOT_COPY_DIRECT`,
  `STRATA_SPLIT_CHUNK16K`, `STRATA_SPLIT_LINK_ABS`, `STRATA_SPLIT_OVERLAP`, `STRATA_SPLIT_SPEED_SCALE`,
  `STRATA_SPLIT_UNIFORM`.
- **Server and setup:** `STRATA_ADMIT_MAX_WAIT_S`, `STRATA_ADMIT_SHORT_FIRST`, `STRATA_DROP_CACHE_ON_EXIT`,
  `STRATA_START_RETRIES`, `STRATA_EXPERIMENTAL_GFX1150`.
- **Older and other GPUs** (not reached on sm_89 / sm_120): `STRATA_SM60_TABLE`, `STRATA_SM61_TABLE`,
  `STRATA_SM75_EXP`, `STRATA_SM75_IL`, `STRATA_SM75_PERF`, `STRATA_TC_SM75`, `STRATA_RDNA2_SGEMM`,
  `STRATA_ROCBLAS_TUNING`, `STRATA_ROCBLAS_TUNING_V1`, `STRATA_ROCBLAS_VERBOSE`.
- **The Qwen3.6 engine** (`src/qwen35/q36_engine.cu`): `STRATA_Q36_BLAS_F32`, `STRATA_Q36_CHUNK`,
  `STRATA_Q36_DRAFT_VOCAB`, `STRATA_Q36_FIXED_SLOTS`, `STRATA_Q36_KEEP_PACKED`, `STRATA_Q36_NO_EXT`,
  `STRATA_Q36_NO_MMQ`, `STRATA_Q36_NO_MTP_CPU`, `STRATA_Q36_OLD_KERNELS`, `STRATA_Q36_PLACE_DEBUG`,
  `STRATA_Q36_READ_WINDOWS`, `STRATA_Q36_STREAM_ALL`, `STRATA_Q36_VDEBUG`, `STRATA_Q36_VERIFY`.
- **Tests and benches:** `STRATA_BENCH_QMIXED`, `STRATA_PARITY_DUMP`, `STRATA_TEST_BELL_SLEEP_US`,
  `STRATA_TEST_CONC_ON_HOST_CPU`, `STRATA_TEST_HANDOFF_DROP`, `STRATA_TEST_PINNED_SHARE`, `STRATA_TEST_SLOW_COPY_US`,
  `STRATA_TEST_UNMAP_GGUF`.

**CMake options and definitions** (not seen by the script): `STRATA_MMQ_KQUANTS` and `STRATA_Q6K_EXPERTS` now default
ON (§2); new `STRATA_MMQ_TAB` (OFF) and `STRATA_PARITY_GFX1100_QSA_SCORES` (OFF); new definitions `STRATA_KQ_DSO`,
`STRATA_HAVE_QWEN35`, `STRATA_Q36_CPU_POOL`, `STRATA_ROCBLAS_AVAILABLE`.

**New source files** (39): `include/strata/artifact/model_family.hpp`, `include/strata/core/family_layout.hpp`,
`include/strata/core/geometry.hpp`, `include/strata/core/host_graph_launch.hpp`, `include/strata/core/pinned_ring.hpp`,
`include/strata/core/sm75_series.hpp`, `include/strata/core/tc_sm75.hpp`, `include/strata/kernels/cpu/nvfp4_avx2.hpp`,
`include/strata/kernels/cpu/nvfp4_avx512.hpp`, `include/strata/prefill/mmq_kq_abi.hpp`,
`include/strata/qwen35/engine.hpp`, `serve/cache_drop.py`, `serve/fixtures/fake_sd_cli.py`, `serve/images.py`,
`src/kernels/b6_mmvq_bench.cpp`, `src/kernels/cpu/kq_cpu_bench.cpp`, `src/kernels/cpu/nvfp4_avx2.cpp`,
`src/kernels/cpu/nvfp4_avx512.cpp`, `src/kernels/expdec_bench.cpp`, `src/kernels/nvfp4_decode_bench.cpp`,
`src/kernels/q8_gemv_bench.cpp`, `src/kernels/qsa_rows_bench.cpp`, `src/kernels/router_chain_bench.cpp`,
`src/prefill/mmq_expert_bench.cpp`, `src/prefill/mmq_kq_lib.cu`, `src/prefill/mmq_kq_loader.cpp`,
`src/prefill/mmq_nvfp4_w4a4.cu`, `src/prefill/mmq_nvfp4_w4a8.cu`, `src/prefill/mmq_tab_instance.cu.in`,
`src/prefill/moe_f16_common.cuh`, `src/prefill/moe_f16_kq.cu`, `src/prefill/moe_f16_kq.hpp`,
`src/prefill/moe_mmq_tab.cuh`, `src/prefill/moe_mmq_tab.hpp`, `src/prefill/rocblas_tuning.hpp`,
`src/qwen35/q36_engine.cu`, `src/qwen35/q36_kernels.cuh`, `src/qwen35/q36_main.cpp`, `src/qwen35/q36ref.cu`. (The
script skips `metal/`, `sycl/`, `tools/`, `tests/` and files named `*_test.*`, `*_parity.*` or `test_*`.)

## B. Step 2: upstream v0.2.0-beta

Upstream `Niko1221/Strata` `v0.2.0-beta` merged on top of step 1 (`33ab01d5`). Merge commit `f06366df`. The working
notes are `C:\Strata-exp\m020-notes\{gyro-cpu,gpu,core,serve}.md` (every hunk with its side and reason) and
`blueprint-check.txt`. This section is their summary, in the structure of sections 1-4 and 6a-9. Verification and the
exes are in §5 and §6.

### B1. What moved

- **Base:** `xeno/268-merge-0.1.45` at `33ab01d5` (the fork's `main` `e5737a27` + v0.1.45 + the fix of §6a).
- **Upstream:** `v0.2.0-beta`, tag object `b73dbce3` -> commit `3d1fb480`. Published 2026-10-19 13:52 UTC by
  Niko1221 (release notes "Strata 0.2.0-beta", engine package 0.1.45.3).
  - The upstream is `Niko1221/Strata`. The `agention` remote (`agentionai/Strata`) carries only rc branches (#268).
  - `v0.2.0-beta` contains v0.1.45.
- **Merge base:** `477ac13c` = `v0.1.45`, the second parent of step 1's merge.
- **Size:** 777 commits (`git rev-list --count 477ac13c..3d1fb480`; 667 not merges).
- **Upstream's side:** `git diff --shortstat 477ac13c 3d1fb480`: 1,275 files, +219,191/-3,168. Most of it is new
  code the fork does not build or run:
  - `apps/`: 740 files, +121,209, the Strata app (Windows, Linux, Mac);
  - `strata_control/`: 45 files, +17,632, the app's control layer;
  - `strata_cli/`: 32 files, +15,206, the Terminal (`strata launch`, `strata connect`, ...);
  - `packaging/`: 23 files, +4,761; `docker/`: 11; `integrations/`: 9;
  - six README translations.
- **The rest, by directory:**
  - `tests/`: 90 files, +22,125;
  - `docs/`: 87 files, +7,790/-266;
  - `tools/`: 52 files, +6,162/-361;
  - `src/kernels/`: 34 files, +5,006/-415;
  - `sycl/`: 18 files;
  - `include/strata/`: 31 files, +1,628;
  - `setup.py`: +1,325/-184;
  - `src/core/`: 14 files, +1,242;
  - `src/prefill/`: 5 files, +1,125;
  - `serve/server.py`: +1,044/-125;
  - `src/program/`: +481;
  - `third_party/`: 3 files, +474 (`third_party/ggml-tq/`, the overlay's tile loader; the ggml pin is unchanged, §B2).
- **The fork's side:** `git diff --shortstat 477ac13c 33ab01d5`: 698 files, +110,244/-3,398. This is the fork's whole
  delta over v0.1.45 (the side the three-way merge weighs), not work since step 1.
- **The merge:** `git diff --shortstat 33ab01d5 f06366df`: 1,256 files, +215,689/-3,120. The fixes after it (`git diff
  --shortstat f06366df f20ec6ab`): 8 files, +18/-4.
- **Conflicts:** 26 files, 126 hunks (counted from `git show --remerge-diff f06366df`):

| file | hunks | what collided |
|---|---:|---|
| `README.md` | 30 | the fork's own README (#166) against upstream's app-centred redesign |
| `src/prefill/prefill.cpp` | 24 | upstream's APR / Gyro edits inside code the fork replaced (PeerPrefill, the per-expert compute / flush), NVFP4, pf_wmma, the "hc write" phase, `on_routes`, `fp16_bufs()`, against the fork's split, wave, #34 / #41 walk and its own APR adaptation (6d31c3fc) |
| `serve/server.py` | 18 | upstream's owned-server contract, bounded queue and deadline, terminal rows, `fit_max_tokens` on, `requested_slots`, replica GPU groups, against the fork's single-pass `_run`, priority FIFO, history store, `_ui_prefix`, MCP hub, model-card / `answer_room` opt-ins |
| `setup.py` | 12 | upstream's Control mode (`home()`, `acquisition`), `BUILD_REQUIREMENTS`, memory tiers, NVFP4-QAD, against the fork's Gyro build (`engine-gyro/`, agentionai ggml), `XENO_REQUIREMENTS`, `[recommended]` |
| `src/kernels/cuda/iq_kernels.cu` | 8 | upstream's `split_off<T>()`, sliced Hadamard launches, `gu_rows` 80-dot rule, against the fork's `x_scales` / `h_scales` (Q2_0 CPU order), #29 one-launch dequant |
| `src/program/generate.cpp` | 7 | the pin reserve around `arena_src.open`, RAM_ADAPT2, `planned_chunk`, the READY report, against the fork's open order, wave lanes, request metrics |
| `src/core/expert_source.cpp` | 4 | upstream's 149 cold-overlap batch loop and zero-copy jobs, against the fork's #95 / #11 NVMe overlap in `expert_dispatch_multi` |
| `src/kernels/cpu/native_expert.cpp` | 3 | upstream's `native_fmt` trellis geometry and `tq_rows_own`, against the fork's `native_fmt_tq` (#217 / #231) |
| `CMakeLists.txt` | 3 | the fork's `end_engine.cpp` and source order, against upstream's trellis MMQ overlay branch |
| `include/strata/core/expert_source.hpp`, `include/strata/kernels/iq_kernels.hpp`, `src/core/pinned.cu`, `src/core/verify.cpp`, `src/kernels/cpu/pool.cpp`, `src/prefill/kernels.cu`, `src/prefill/moe_mmq.cu`, `serve/telemetry.py`, `.gitattributes` | 1 each | see §B3 |
| `include/strata/kernels/cpu/hadamard.hpp`, `src/kernels/cpu/hadamard.cpp`, `include/strata/kernels/cpu/tq_cpu.hpp`, `src/kernels/cpu/tq_cpu.cpp`, `src/kernels/cpu/tq_avx2.cpp`, `src/kernels/cpu/native_hadamard_test.cpp`, `docs/GYRO.md`, `tools/test_setup_gyro.py` | 1 each (whole-file add/add) | both sides added the file through the same agentionai rc2 port (§B3) |

### B2. Upstream's changes that matter here, and what the fork does with them

D2x as served is §2's profile. The Gyro-M profile is `strata-gyro-m-d2x.json` (key 3, `run-gyro-tq`). For each item:
whether it reaches that path, and whether it is on. Unless a line says otherwise, "bitwise" is upstream's claim. The
check here is §5's parity, which ran with all of them on.

**Gyro: upstream's own port of rc2 meets the fork's.**

- Upstream ported agentionai's `rc2` commit by commit ("APR port", f47b0385 .. 86faf035). The fork had already merged
  the same rc2 patches (15878979 .. 50a19727, #217, 2026-10-13) and then adapted them into its own prompt walk
  (6d31c3fc) and CPU kernels (#220, #229).
- So most Gyro files are add/add on both sides with the same blobs. They were resolved against the shared port commit
  as a virtual base (gyro-cpu.md, "Method"). Where upstream made no change after the port, the fork's file was kept
  whole.
- What the merge takes from upstream's later work:
  - the scalar / arm64 trellis rows (34204c4a);
  - `tq_kernels_enabled()` true unless `STRATA_NO_TQ_KERNELS=1` (eca3aef3);
  - the `native_fmt` trellis geometry without asking ggml, with mixed trellis / non-trellis layers, and `tq_rows_own`
    (e1d11fe7);
  - the 65535-row Hadamard slices (02bbf51f; a 16K-token chunk passed `gridDim.y` without them);
  - the per-job layout in the pool (c85cc6dd);
  - the trellis MMQ overlay (below).
- The fork keeps:
  - the #220 AVX2 / AVX-VNNI kernels (`tq_avx2.cpp` byte-identical to the fork's);
  - `STRATA_HADAMARD_SCALAR` and the bit-exact `fwht_avx2_check`;
  - the #147 one-phase pool path and its SwiGLU expression;
  - its prompt walk's APR rotation.
- One copy of each function: §B5 lists the two duplicates the auto-merge made.

**Kept at the fork's value, with upstream's value behind a switch:**

| item | reaches D2x | on | why / switch to upstream's value |
|---|---|---|---|
| `fit_max_tokens` (f31b6efd: upstream shortens `max_tokens` to the room left, by default) | the D2x and Gyro configs set it themselves | as configured; off when unset | The fork keeps the strict 400 for a request over the room. `"fit_max_tokens": true` or `--fit-max-tokens` arms it; `--no-fit-max-tokens` still wins. D2x, `strata-gyro-m-d2x`, `-d2x-512k` and `strata-ista-d2x` already have it true, so they do not change. Capacity has no key and stays at 400 (serve.md §2) |
| Gyro pictures (`vision: True` upstream, engine 0.1.45.1+) | no | **off** | never tried with `engine-gyro/` and agentionai's ggml. `"vision": True` on MODELS `Gyro-M` / `Gyro-S` and FAMILIES `gyro` turns it on. `tools/strata_mcp.py`'s Gyro table still says True (§B9) |
| `STRATA_PF_WMMA` (Volta FP16 tiles in the prompt walk) | no | **ignored** | It lives in upstream's per-expert flush, which the fork replaced. `pf_wmma()` returns false and prints "STRATA_PF_WMMA is ignored: the fork's prompt expert walk has no WMMA tile path (merge 0.2.0)", as `STRATA_MMQ_DIRECT` at step 1. Without this, `STRATA_PF_WMMA=1` on D2x would mark IQ1_M layers `wmma_only` with no tile for them (core.md) |
| `model_card_sampling`, `answer_room` | yes | **off** | step 1's opt-ins, untouched by step 2 |

**Upstream defaults taken, on by default:**

| item | reaches D2x | on | why / switch |
|---|---|---|---|
| **The sm_120 set** (`include/strata/core/sm120_series.hpp`, ba2c2d5f): `STRATA_GU_SWIGLU=1`, `STRATA_SIDE_MIXER=2`, `STRATA_ATTN_DAILY=1`, `STRATA_GR_UP75=1`, `STRATA_GR_DOWN_TG` (=2, token pairs, on a 12.0 card with fewer than 80 SMs; the 5060 Ti has 36) | **yes, the 5060 Ti** (decided per device; the 4070, sm_89, is unchanged) | yes | Upstream calls each bitwise. Upstream measured them on an RTX PRO 5000 only: a 4-token verify round 13.70 -> 12.23 ms. `STRATA_SIDE_MIXER=2` moves the mixer's input-only work onto branch streams of the verify graph. The fork's graph also carries H2 `moe_hit_merge_mapped`, the K4 PCIe part, the 4070's launch / finish and the #44 spans, which upstream's parity did not exercise. Against the fork's rule (an upstream default stays only with same-session parity and an ABBA on D2x): §5 has identical `STATE_HASH`, replies and generate outputs with the set on, and a `STRATA_SM120_PERF=0` arm with no consistent speed change. `STRATA_SM120_PERF=0` turns the whole set off; each variable `=0` turns its own off. `STRATA_PF_FUSED` (rounds differently) is on only for 80+-SM cards, so it stays MMQ on the 5060 Ti |
| **The fork's extra guard on `launch_gu_swiglu`** (iq_kernels.cu, `gs_fork_ok`) | yes | the guard is always on | Upstream's fused gate/up + SwiGLU + q8_1 launch runs before the fork's Q2_0 CPU-order branch, and `gs_done` skips the rest. So on the 5060 Ti a Q2_0 layer with `x_scales` would have taken GPU-order SwiGLU, which breaks `native_q2_pool_hit_parity` (ADR 0001). A Q2_0 down with `x_scales` would read `h_scales` / `h_hx` that nothing wrote. A Hadamard-folded down needs `h` rotated as floats before q8_1. The guard `!(x_scales && (gu_type == 42 \|\| d_type == 42)) && had_block <= 0` matches the header's rule ("with `x_scales` ... the fused pass is not used"). Without `x_scales` and without a fold it is true, so D2x keeps upstream's default (gpu.md) |
| **The `gu_swiglu` 80-dot fix** (8837cb6f) | Gyro on the 5060 Ti | yes | `launch_gu_swiglu` assumed 80 dots per row. Gyro's trellis rows have 160, so Gyro-S answered with loops, garbage or nothing on every RTX 50: upstream got 0 of 8 right before the fix and 8 of 8 after, on an RTX PRO 5000. Trellis rows now take the full-warp kernel. Neither the fork nor step 1 had `launch_gu_swiglu`, so the served Gyro-M was never exposed: the path and its fix arrive together. The fork's own 80-dot guard in `launch_gu` (c4ba2887) is kept. `gu_rows` also excludes Q4_K / Q5_K (160 dots, 79ac3164) |
| **The trellis MMQ overlay** (b15b209a: `cmake/strata_ggml_tq_overlay.cmake`, `third_party/ggml-tq/`, CMake `STRATA_MMQ_TQ_OVERLAY` ON) | the build; Gyro's prompt path on a default-ggml build | yes | At configure time it writes a copy of the pinned ggml-cuda headers with the three trellis types routed like IQ2_XS. Only Strata's MMQ library sees the copy. With `STRATA_GGML_DIR` on agentionai's tree, that tree's own instances win and there is no overlay. Upstream's identity gate (R9700, 10 prompts, 4K / 20K): Gyro-S / M rc2 against upstream + overlay 10/10 identical; Q2_0 / IQ3_XXS with the overlay on identical on gfx1201 and sm_86. On the fork, every anchor was found once in the pinned tree with the IQ1_M patch applied (gpu.md §2), and `build-m145u` built it (`build-m145u\ggml-tq-overlay`, ctest `ggml_tq_overlay` passed). It costs D2x no VRAM or slots at boot (§5). Gyro-M on `run-m145u` is bit-identical to the agentionai-ggml builds (all 7 `STATE_HASH` fingerprints, swaps off, 6 boots; §5's overlay question). `-DSTRATA_MMQ_TQ_OVERLAY=OFF` drops it |
| **CPU pool worker plan** (5ece068f, 036c8f6e, bec4c838) | only without `--pool-workers` | yes | The engine picks the worker count from the physical cores (host socket, container quota, cap 32). On a hybrid CPU with more E-cores the plan is the P-cores less the host's one, plus half the E-cores: 5 + 4 = 9 on the i5-13500 (6 P + 8 E; setup's own count is P + E/2 = 10). D2x passes 13 and Gyro-M 5, so their counts are unchanged. D2x now prints at every start "strata cpu pool: warning: 13 CPU expert workers are configured ... Recommended: --pool-workers 9" (§5). The fork's measurement ("E-cores slow for the trellis pool; 5 P-core workers match 13", memory note) is for Gyro, not D2x's 13. Setup no longer writes a worker count on an ordinary CPU |
| **Pinned reserve** (4f74de9c, 4ff5cd40, `include/strata/core/pin_budget.hpp`) | **yes** (Windows CUDA) | yes | It holds 640 MiB soft + 160 MiB hard of pinned RAM before `arena_src.open`, hands the soft part back after it and the hard part at READY. Verify's mapped staging goes through `host_alloc_guarded` (one retry after the hard reserve is handed back). On D2x and Gyro-M the arena is pageable (the 4070 tier, exclusive primary experts) and registers nothing. So there the reserve protects nothing, and it costs about 800 MiB of pinned RAM through the open and 160 MiB until READY. Every merge boot prints "pinned-memory headroom: holding 800 MiB ..." (§5). `STRATA_PIN_RESERVE_MIB=0` skips it. Skipping it when `!pin_for_cuda && !place_first` is a code change not made (§B9) |
| **`end_leftover_engines(exe)`** (serve, every engine start) | yes | yes | It ends engines a lost server left: recorded ones whose starter is gone, or orphans of exactly this exe running `--serve`. On this PC that is a hazard for measurements: a second served engine of the same exe whose parent died is killed by a server start. `STRATA_KEEP_LEFTOVERS=1` keeps them |
| `apply_saved_tuning(cfg)` (serve, every strata start) | no preset store on the fork | yes | puts a saved preset (`strata_control/presets.py`) on the config; a no-op without one. `STRATA_NO_PRESET=1` |
| A start never acquires (`strata_control/acquisition.py`, `STRATA_NO_ACQUIRE`) | setup / start | yes | no engine update or compile at start; `engine_update_note()` instead, which on the fork also checks `engine-gyro/` (its source hash or `AGENTION_GGML_STAMP` changed: run UPDATE). No switch |
| Server: a non-loopback `--host` without an API key exits 2 (was a warning); close sends `STOP\nQUIT` and waits 20 / 5 / 20 s; children run under `lifeline.popen` (was `winjob.contain`) | yes | yes | upstream's. `STRATA_DROP_CACHE_ON_EXIT` now runs only on a second Ctrl+C, which looks like an upstream slip (opt-in) |
| mmvq_il 5-8 columns and sm_120 tables (75fabbd9) | yes, the 5060 Ti | yes | bitwise per upstream (`mmvq_il_parity` passed, §5); verify's `xil_` is sized `max(4, T)` |
| Setup: pictures on by default where the model reads them (`images_default`, 91baead7 / de910193); recommended size by RAM + VRAM tiers (aa82a886); 8-connection downloads with a pinned size and SHA-256 for every file (`STRATA_DL_CONNECTIONS`, `STRATA_ALLOW_UNPINNED`) | setup-written configs only | yes | The hand-made served configs are untouched. On this PC the picture reader would be the CPU one (16 GB primary < 24 GB). `--vision no`, or answer n |
| `STRATA_PREFILL_TIMING`: a new "hc write" phase (aef32e22) | the timing lines | yes | the write-back time moves out of the phase before it; parsers that sum phases are not affected |
| `tq_kernels_enabled()` and `native_fmt` (eca3aef3, e1d11fe7) | Gyro | yes | On an AVX2 / AVX-512 CPU the old expression was already true, and agentionai's ggml gives `vec_dot_type` Q8_0 and row bytes 34 / 26 / 30 per 128 for TQ2_T / TQK6 / TQK7. So the served Gyro-M's `NativeFmt` is the same (gyro-cpu.md). Changed: a CPU without AVX2 runs the scalar trellis kernel; `STRATA_NO_TQ_KERNELS=1` on a ggml without the trellis types is ignored instead of refusing the pack |

**Not on this PC's path** (checked, core.md §2):

- `STRATA_COLD_OVERLAP`, `STRATA_IO_PREFETCH`, `STRATA_IO_PF_GATE`: Linux defaults, file tier only (`linux_file_defaults()`).
- `STRATA_PREFILL_HOST_PLAN` (#2824): only for `--prefill auto` with the file tier. All served configs use `--pack`. A
  Windows file-tier run would have it on.
- `nv_defaults` (2725c23f): sm_86 / sm_60 only, and only when every card has the same compute capability.
- `STRATA_HIP_WMMA` on gfx12, `STRATA_PREFILL_WARM`: HIP only.

**One Gyro catalog in setup.** The auto-merge had silently made two:

- `HF_REVISIONS`, `GYRO_FILES` and `FAMILIES["gyro"]` were each defined twice;
- MODELS listed four Gyro sizes: upstream's `Gyro-S` / `Gyro-M` and the fork's `M` / `S`.

Now (serve.md, "Gyro: one catalog"):

- **MODELS** has upstream's `Gyro-M` / `Gyro-S`, with upstream's `ram_gb` 35.0 / 27.6 (counted against RAM + VRAM,
  `mem_total`) and `engine: GYRO_ENGINE (0,1,45,1)`. The fork adds `low_ram: False`, `recommended` (Gyro-M) and
  `vision: False`. Gyro-M comes first.
- **`GYRO_ALIASES = {"M": "Gyro-M", "S": "Gyro-S"}`:** `--model M|S` still works. `install_tag()` keeps the fork's
  names for an install: `strata-gyro-m.json`, `packs/gyro-m`, model name `qwen3.8-flash-next-gyro-m`.
  `choices_from_config` maps old configs.
- `HF_REVISIONS` keeps the fork's pin `8fa6f45c`, the revision the served Gyro-M came from (the same files; upstream's
  `1ada0842` is noted in a comment).
- Dropped: the fork's RAM-only `ram_gb` 48 / 36. With `mem_total`, a 32 GB RAM + 12 GB card now pre-selects Gyro-M.
  The fork's measured 48 GB for Gyro-M at 262K on two cards is only in README / docs now.
- Checked by importing setup (serve.md): sizes, `install_tag`, `recommended_size` at five RAM / VRAM points,
  `gguf_choice`, `choices_from_config`, `images_default`. The setup tests were not run (§5).

**The rest, taken as upstream has it (opt-in, or not on the D2x path):**

- **Server switches:** `--max-queue` / `max_queue`, `--queue-timeout` / `queue_timeout` (429 `queue_wait_timeout`, 503
  `server_draining`), `--shutdown-timeout`, `--ready-file`, `--owner-fd` (`STRATA_OWNER_ATTEST_KEY`), `--parallel`,
  `--batch-groups`, `--batch-telemetry` / `batch_telemetry` (BSTAT / PSTAT lines), `--offline`, `--script-cycle`,
  `--script-delay`.
- **Endpoints:** `GET /v1/telemetry` (`telemetry_read_key` / `STRATA_TELEMETRY_KEY`); `/v1/v1/messages[/count_tokens]`
  aliases.
- **Engine:** the 149 file tier (zero-copy cold batches, I/O prefetch gate, `STRATA_RAM_ADAPT2*` with a prompt seed
  through `Prefill::on_routes`, `STRATA_RAM_ADAPT_QUEUE`); `STRATA_BATCH_TELEMETRY`; `STRATA_COLD_TRACE`; the K-quant /
  Q8_0 Split traits behind `STRATA_KQ_SPLIT=1`; `STRATA_FUSE_DECODE`; `STRATA_GDN_CHUNK`; the per-request "adaptive
  tier drain" line; Volta `gemm_iq_f16_grouped`; sm_60 table staging.
- **Setup:** Control mode (`STRATA_CONTROL_INSTALL`, `STRATA_INSTALL_DIR`, `STRATA_SETTINGS_FILE`), the Mac app path,
  NVFP4-QAD, `ple-io ram` on Linux, the AMD HIP picture reader.
- **Not built or run here:** the app (`apps/`), the Terminal (`strata_cli/`), engine packages and the signed feed, AMD /
  Intel / Linux defaults.

### B3. Every conflict, file by file

Four areas were resolved in parallel, each into one notes file; `.gitattributes` was resolved outside them. In every
resolved file there is no marker left and CRLF is kept (byte-checked), and `CMakeLists.txt` keeps its BOM.

#### Gyro and CPU kernels (gyro-cpu.md)

- **`hadamard.hpp`, `hadamard.cpp`, `native_hadamard_test.cpp`, `tq_avx2.cpp`** (add/add, theirs == the shared port):
  the fork's file whole.
  - Added an x86-64 guard around the AVX2 Hadamard dispatch and around `fwht_avx2_check`'s body. Upstream's
    `metal/CMakeLists.txt` compiles `hadamard.cpp` without `tq_avx2.cpp`, so an arm64 build would miss
    `hadamard_rotate128_avx2`.
  - x86 arithmetic is unchanged.
- **`tq_cpu.hpp` / `tq_cpu.cpp`** (add/add, `git merge-file` against the port: clean):
  - kept from the fork: `tq_codebook_i16`, the #220 variants and `tq_rows_avx2_v`;
  - taken from upstream: `kTqBlock` and `tq_block_bytes()` (e1d11fe7), the `STRATA_TQ_X86` guards (34204c4a),
    `tq_kernels_enabled()` (eca3aef3).
- **`native_expert.cpp` (3):**
  - `native_fmt` is upstream's. The fork's `native_fmt_tq` and its local `tq_block_bytes` were deleted: they would be
    ambiguous with the header's new inline one, and upstream's version is a superset.
  - Two fork guards were ported into upstream's body: the negative / out-of-table type id (upstream would call
    `traits(-1)`), and the `kTqMaxN` (4096) width check (`TqAct::q[kTqMaxN]` would overflow).
  - `native_gu_rows` / `native_down_rows` take upstream's `tq_rows_own(...)`.
- **`pool.cpp` (1):**
  - the fork's `native_rows(mode_, ...)` (#147);
  - upstream's per-job layout ported into it (`mjobs_[e].fmt ? ... : nfmt_`, c85cc6dd). `per` and `native_quant` stay
    on `nfmt_`, because the job's fmt changes only `up_off` / `down_off`;
  - the fork's `std::exp(-(double) g)` SwiGLU is kept;
  - upstream's pool plan auto-merged outside the hunk.
- **`docs/GYRO.md`** (4 hunks after the virtual-base three-way):
  - upstream's 0.2.0 text, plus an "On this fork" paragraph: rc2 merged in #217; the fork builds its Gyro engine
    against agentionai's ggml, not upstream ggml + overlay, which is not measured here; Gyro-M is recommended;
  - the fork's setup and measurements;
  - upstream's sections renamed "Upstream's setup path and engine notes (v0.2.0)" and "Measured by upstream (the port,
    2026-10-15/16)";
  - the fork's `tools/gyro_setup.sh` section, kept because the script still exists here.
- **`tools/test_setup_gyro.py`** (add/add, two unrelated files): the fork's tests, moved to `Gyro-M` / `Gyro-S` (and
  `test_the_fork_names_are_aliases`), plus upstream's `GyroCatalog` / `GyroEngineVersionTest`.
  - Upstream's tests expect `vision` False, and no longer compare `vision` with `strata_mcp.py`.
  - The fake engine's version is 0.1.45.3.
  - `test_the_default_size_follows_the_ram` uses 20 GB RAM.
  - `py_compile` only; not run.

#### GPU kernels and CMake (gpu.md)

- **`iq_kernels.cu` (8):**
  - includes `<algorithm>` (theirs);
  - `is_iq()` in both builds (theirs, the same set in another order);
  - `launch_gu` / `launch_down`: the fork's `nullptr` (`x_scales` / `h_scales`) argument and upstream's
    `|| split_off<T>()`;
  - the fork's `iq_dequant_expert_f16` (#29, with step 1's NVFP4 tail), then upstream's sliced Hadamard block (the
    fork's unsliced copies dropped);
  - `gu_rows`: theirs, with the fork's trellis exclusion inside;
  - the SwiGLU / q8_1 pass: the fork's (`cpu_order`, `h_scales`, `h_hx`). Upstream's call has the 0.1.45 signatures and
    would not compile.
  - Outside the markers: the duplicate `dq_tq` was removed (§B5), and the `gs_fork_ok` guard was added (§B2).
- **`iq_kernels.hpp` (1):** the fork's `had_gu_block` in `NativeExpertLayout`, read by `expert_source.cpp` and
  `secondary_runner.cpp` (the 4070 rotates its own x).
- **`kernels.cu` (1):** only upstream's `swiglu_split_f16`. The fork had moved `swiglu_pair`; taking theirs too would
  have defined it twice.
- **`moe_mmq.cu` (1):** the fork's `STRATA_MMQ_IQ1M_TILE` case. Upstream's `is_tq` / `type_blck` helpers and its
  `STRATA_MMQ_TQ` cases auto-merged once each.
- **`pinned.cu` (1):** in `~PinnedArena`, the fork's `if (registered_bytes != 0)` guard around upstream's
  `pin_unregister(...)`, the pin budget's wrapper.
- **`CMakeLists.txt` (3):**
  - the fork's `end_engine.cpp` (#203) in `strata_kernels_cpu`;
  - the fork's source order;
  - upstream's `elseif(STRATA_MMQ_TQ_OVERLAY ...)` branch with ctest `ggml_tq_overlay`;
  - outside the markers: upstream's second copies of the `tq_avx2.cpp` / `tq_avx512.cpp` properties were removed, and
    the fork's AVX-VNNI define for `tq_avx2.cpp` (#220) is kept;
  - auto-merged: `project(strata VERSION 0.1.45.3)`, `src/core/nv_defaults.cpp`, the KQ_SPLIT / GU_SWIGLU variants of
    `iq_multi_parity`, `pool_plan_test`.

#### Core engine (core.md)

- **`prefill.cpp` (24):** mostly the fork's. Upstream's APR edits sit in PeerPrefill (`set_peer`, `split_help_*`,
  `peer_had_signs`), the per-expert compute / flush, NVFP4 and pf_wmma, all of which the fork replaced or does not have.
  Both sides were kept in these hunks:
  - `ring_slots`: upstream's `qsa_prompt_attn_matrix_cores_requested()` with the fork's split threshold;
  - the split plan, then `fp16_bufs()`;
  - the chunk line, then `warm_gemm_libraries` (HIP only);
  - `kPfWaitHost` (#31), then `kPfHcWrite`;
  - the wave wait and `ple_take()`, with upstream's `pt.mark(kPfPle, cs)` moved in front of the PLE wait;
  - the router sync span, then `on_routes(l, m.ids_h(), T*K)`.

  The carve and `moe_set_bytes` both follow `fp16_bufs(mp)`, so they count the same buffers. Post-edits:
  - the auto-merged pf_wmma block, which reads `on_peer` the fork's walk lacks, was removed;
  - `pf_wmma()` returns false (§B2).
- **`generate.cpp` (7):**
  - the fork's `crash_report.hpp` include, without its second `iq_kernels.hpp`;
  - the pin reserve around the fork's `arena_src.open(..., pin_for_cuda, place_first, pin_limit, ...)` (the other
    overload is `= delete`);
  - the fork's lane state, then `RamAdapt2`;
  - `sp2.should_stop` / `on_chunk`, then the RAM_ADAPT2 seed, **on lane 0 only** (two lanes would `+=` one array from
    two threads);
  - the fork's `INFO` lines, then `pin_sim_report` / `pin_reserve_release_hard()` before READY;
  - `lend`'s `want` with `planned_chunk(...)`. `pf_plan_chunk` is 0 unless the file-tier host plan ran, so it equals
    `equal_chunk`;
  - the fork's request metrics, then upstream's adaptive-drain line.

  It adds no `return 1` / `return 2` / `exit(` (`serve_fatal()` 49 -> 49).
- **`expert_source.cpp` (4) / `.hpp` (1):** in `expert_dispatch_multi`:
  - the fork's `tl_plan`, then upstream's `DecodeScope` and `cold_mode` pick;
  - the fork's #95 / #11 NVMe overlap, then upstream's 149 cold batch loop, with upstream's zero-copy job build ported
    into the fork's `add_entry`. The fork's `miss[128]` was renamed `nvme_miss`, because it shadowed upstream's new
    `miss` vector;
  - `run_jobs` only when nothing is pending, and `ms_cpu_pool += run_ms_extra`;
  - with `cold_mode == 0` (Windows, the arena) the function is the fork's, step for step;
  - the header has both sides' counters.
- **`verify.cpp` (1):** the fork's (the `had_gu` branch is the same on both sides; the fork's also has the Q2_0
  CPU-order `quantize_q8_1_rows_scaled`).

#### Serve, setup, README (serve.md)

- **`server.py` (18):**
  - `_run` keeps the fork's single-pass `_generate` loop (hunks 6, 7) and takes upstream's queue: the `terminal_requests`
    row and `queued` event, the FIFO deadline loop with the fork's priority (`self.fifo.acquire(priority, timeout=...)`,
    xeno #49 S6), and the outer `except QueueCancelled` / `QueueWaitExpired` / `Exception`;
  - the fork's three manual `queued -= 1` paths were dropped, because upstream's outer `finally` settles them;
  - the history row stays the fork's `summary_record` / `meta["id"]`;
  - `main` merges `with_component_engine`, lifecycle events and `--offline` with the fork's stream set-up, timeline,
    MCP hub and `--ban-ids` args;
  - `Service(...)` gets upstream's `launch_attestation`, `offline`, `max_queue`, `queue_timeout_s`, and keeps the fork's
    `fit_max_tokens=False` and opt-ins;
  - ported outside the hunks: `first_engine_token_s`; `except QueueWaitExpired` before `except ValueError`; `_fit`'s
    "max_tokens N shortened to M" line.
- **`telemetry.py` (1):** the fork's #237 per-card list, a superset of upstream's `name` field.
- **`setup.py` (12):**
  - both requirement sets;
  - `get_llama_cpp` with the fork's `agention` parameter and upstream's `home()`;
  - `update_gyro_engine` with `engine_update_note`;
  - `ensure_engine_for` with `acquisition.require(...)` and `engine_dir(tk, gyro)`;
  - `engine_step` with upstream's `if CONTROL: return control_engine_step(...)` first and a refusal of Gyro in Control
    mode (the app cannot compile `engine-gyro/`);
  - the ready-made-engine rebuild check stays `if ready and ...`: upstream's `or family == "gyro"` assumes upstream's
    ready-made Gyro engine;
  - then the one Gyro catalog (§B2).
- **`README.md` (30):** §B7.
- **`.gitattributes` (1):** both sides' lines: the fork's `*.pdf binary` and `tools/hooks/* text eol=lf`, upstream's
  `*.cmd` / `*.ps1 text eol=crlf` and `bin/strata text eol=lf`.

### B4. Semantic hazards checked beyond the markers

- **Duplicates from two ports of one patch set.** These are the class this step adds: both sides carry the same rc2
  code, so git can keep two copies of the same function where it sees no conflict. Checked:
  - `dq_tq` and `Xr` were found twice and fixed (§B5);
  - the CMake `tq_*` properties were in twice (harmless, removed);
  - `swiglu_pair` was avoided;
  - `tq_block_bytes` / `kTqBlock` are defined once (grep of `src` and `include`);
  - one `TqInfo`, `SplitTq`, `TqLane`, `hadamard_rows_kernel`;
  - the APR identifiers have the same counts in the fork's side and in the merged `verify.cpp` / `expert_source.cpp`.
- **The fused gate/up launch against the fork's Q2_0 CPU order and folded downs:** guarded (§B2). The ADR 0001 test
  with `STRATA_GU_SWIGLU=1` has not run (§5).
- **Signatures:**
  - every `native_gu_kernel` / `native_down_kernel` launch (4) carries the fork's extra argument;
  - every one of the 15 `native_expert_scratch_bytes` calls passes 3 arguments;
  - `swiglu_entries_kernel` / `quantize_q8_1_kernel` use the fork's form;
  - `native_gu_swiglu_kernel`'s 9-argument `row_dot2_80_sub16` call matches its definition;
  - `arena_src.open` keeps the fork's order;
  - `on_routes` matches `prefill.hpp`.
- **Names the fork lacks:** a script listed every identifier in the merged core files that the fork's side lacks. All
  are upstream locals or declared names, except `on_peer`, which was removed with the pf_wmma block.
- **Upstream lines missing from the result:** 73 in `prefill.cpp`, each one PeerPrefill / APR-peer, pf_wmma, NVFP4 or
  upstream's walk; 3 / 2 / 3 rewritten forms in generate / expert_source / verify.
- **Threads:** RAM_ADAPT2's seed runs on lane 0 only. `zc_fmt.resize` happens before any `nj.fmt` points into it.
- **Shadowing:** the NVMe `miss` -> `nvme_miss`. Prefill's outer `xsrc` was dropped and the fork's scoped one kept.
- **The pageable arena and the pin reserve:** `reserve_only` is a MEM_RESERVE VirtualAlloc, and `load_rest` registers
  nothing. So on D2x the reserve guards nothing (§B2).
- **Serve-loop exits:** no new `return 1` in `generate.cpp` (`serve_fatal()` 49 / 49).
- **Serve:**
  - every `self.fifo.acquire` in `_run` passes the fork's priority;
  - upstream's outer `finally` covers every path the fork's manual decrements covered (a ping to a closed consumer, a
    cancel while queued, an `ensure_loaded` failure);
  - the fork's `meta["id"]` stays the history id.
- **`--offline`** always refuses on the fork while the agent tools are on, because the hub is never None. Use
  `"agent": false`. Not changed.
- **Feature counts, the fork's side -> merged, equal:**
  - generate.cpp: `strata::timeline::` 58, `refill_lent` 4, `sp_wave` 7, `sec_paired` 10, `kv_end` 17;
  - prefill.cpp: `timeline::` 66, `SplitTier` 45, `split_send` 5, `had_sx` 9;
  - expert_source.cpp: `timeline::` 17, `secondary` 30, `materialize` 22.

  No `STRATA_*` name of the fork's is missing from the five core files.
- **The ggml pin:** `3cf03257` before and after. `ggml-common-tq.h` and `tq_lut_i8.cuh` are byte-identical on both
  sides.
- **Not settled without a run:**
  - the sm_120 set inside the fork's verify graph (§B2; §5's parity ran with it on);
  - the overlay's `STRATA_MMQ_TAB` fingerprint describes the pinned `mmq.cuh`, not the overlay's copy. The overlay adds
    only routing cases, so it should still hold; not verified;
  - timing under `STRATA_COLD_OVERLAP >= 1` (file tier): the "cpu pool" span is about 0 there while `ms_cpu_pool` gets
    `run_ms_extra`;
  - `native_hadamard_test` fails on a stock-ggml build when it is registered there. Since `f20ec6ab` it skips (§B5).

### B5. Bugs the merge introduced

**Found by the area sweeps before the build:**

- **`dq_tq<TY>` defined twice in `iq_kernels.cu`.** Both rc2 ports added the same 20-line function; git kept the
  fork's copy before `dq_nvfp4` and upstream's after it. The two were byte-identical, and upstream's was deleted (gpu.md).
- **Four Gyro sizes and two catalogs in `setup.py`** (`HF_REVISIONS`, `GYRO_FILES` and `FAMILIES["gyro"]` each twice;
  Python kept the last value at the first key's position). Resolved into one catalog (§B2).
- The CMake `tq_*` source properties were set twice (same values).
- An auto-merged pf_wmma block reading `on_peer`, which the fork's walk does not have (core.md).
- The NVMe `miss[128]` shadowing upstream's new function-scope `miss` (core.md).
- `README.md`: the auto-merge replaced "How fast is it?" with upstream's app line and added a second "Something went
  wrong?" section. Both were reverted (serve.md).

**Found by the build** (`8ed30815`):

- **`Prefill::Impl::Xr` declared twice** (MSVC C2086). Both sides ported rc2's APR MMQ path; one declaration kept.
- **POSIX `setenv` / `unsetenv` in three new upstream targets** (`bench/gdn_chunk_bench.cpp`,
  `src/kernels/gdn_chunk_parity.cpp`, `src/kernels/hc_persist_parity.cpp`). MSVC has neither; a `_putenv_s` shim
  under `_WIN32`.

With these, `build-m145u` builds every target and `build-m145g` builds the engine.

**Found by ctest** (`f20ec6ab`, §5):

- **`xeno_pool_core_policy` and its `_cpu_dispatch_vnni` / `_forced_avx2` variants** expected the automatic worker count
  to be every physical core but the host's (13). v0.2.0's plan gives 9, so the check now asks `plan_pool_workers`.
  Serving does not change, because the served configs set `--pool-workers`.
- **`native_hadamard_test`, `tq_cpu_parity`, `tq_cpu_parity_avx2`** need a ggml with the trellis types for their
  reference. Since v0.2.0 the default-ggml build compiles them too, where they failed.
  - They exit 77, and CMake's `SKIP_RETURN_CODE 77` counts that as a skip.
  - On the D2x build they now skip and do not run. They still run on `build-m145g`, whose ctest has not run (§5).
  - The overlay build's trellis CPU kernels are therefore not checked against a reference on the default-ggml build.

**Found by tests and fixed in code since:** none yet (pytest has not run).

**Found by reading, not fixed** (serve.md):

- Three serve tests assume upstream's `fit_max_tokens` default (True): `serve/test_context_overflow.py:91`,
  `test_prompt_encoder.py:119`, `test_monitor.py:114-116`.
- `serve/runconfig.py` says `fit_max_tokens` is "on by default", which is wrong on the fork.

### B6. The classic web app

`git diff --stat 477ac13c 3d1fb480 -- serve/web` is empty: v0.2.0 changed nothing in the classic app. The only log
entry, 07fe758a, is a merge that brought 0.1.45's web sampling, which is already in the base. Nothing to port. The merge
did not touch `serve/ui/` (`git diff 33ab01d5 f06366df -- serve/ui` is empty).

Still open from step 1: `e8634865` (send no sampling until the user changes it) -> `serve/ui/` (§8).

### B7. README

`README.md` is the fork's own (#166). All 30 hunks were resolved on the fork's file, because upstream redesigned its
README around the app, and the fork did not adopt that. The two auto-merged changes outside the hunks were reverted
(§B5).

Carried by hand, with no upstream speed figures:

- "last merged" v0.1.45 -> **v0.2.0-beta**.
- **Install:** upstream 0.2.0 is an app + Terminal + engine packages that carry upstream's engine; the fork installs from
  source. A start never downloads or compiles (UPDATE.bat after `git pull`). Downloads use 8 connections with resume and
  a pinned size + SHA-256 (`STRATA_DL_CONNECTIONS=1`, `STRATA_ALLOW_UNPINNED=1`). CMake / Ninja are needed only when
  compiling. No CPU worker count is written on an ordinary CPU.
- **Models:** setup's pre-selected size follows upstream's RAM + VRAM tiers, and Gyro-M stays the fork's
  recommendation. Pictures are on by default where the model reads them (CPU reader under 24 GB VRAM, `--vision no`),
  and Gyro is text-only on the fork. NVFP4-QAD is in "Other models"; "v0.1.43 to v0.2.0".
- **Gyro:** `--model Gyro-M` (`M` still works; old installs keep their config and pack); memory counted as RAM + VRAM
  (35 / 27.6 GB); Gyro-S below 35 GB.
- **Using it:** a base URL ending in `/v1` works (the `/v1/v1/messages` alias). The server will not listen beyond this
  PC without an API key. `max_tokens` stays a strict 400 on the fork (`"fit_max_tokens": true` is the opt-in; D2x has it
  on).
- **Something went wrong:** "still frozen after 10 minutes", "stopped while downloading: run it again", "engine older
  than this Strata expects: run UPDATE".

Not carried: upstream's speed figures and its app-first layout.

**Not reconciled:** upstream's six translations (`README.de.md`, `.es`, `.fr`, `.ja`, `.pt-BR`, `.zh-CN`) were already
in the fork's tree before this merge (they are in `main` `e5737a27`). The auto-merge updated them to upstream's 0.2.0 text (1,924 lines added, 974 removed in this
step). They translate upstream's README, not the fork's (§B9).

### B8. Configuration surfaces this step adds (from `tools/blueprint_check.py`)

`blueprint-check.txt` is the script's output on the staged merge. Grouped by the program that reads each name, as in §9.
Upstream's docs describe its names, and §B2 gives the ones that reach D2x, with their defaults.

**`strata` engine flags and GEN keys:**

- No new engine flag.
- The script reports a GEN key `file` in `src/core/expert_source.cpp`. It is a key of the cgroup `memory.stat` parser
  (149's RAM check), not a GEN key.
- It also reports `strata_tune` in `serve/server.py`. That is a `sampling` sub-key of the run config (a saved tuning's
  per-request settings), not an engine GEN key.

**Server switches and run-config keys** (`serve/server.py`):

- new keys: `batch_telemetry`, `max_queue`, `queue_timeout`, `telemetry_read_key` (none of them is in `33ab01d5`'s
  `serve/server.py`);
- also reported as "added" because new code names them, though they are existing keys (in `33ab01d5`'s server):
  `args`, `backend`, `lib_dirs`, `parallel`, `before_load`;
- flags: `--max-queue`, `--queue-timeout`, `--shutdown-timeout`, `--ready-file`, `--owner-fd`, `--parallel`,
  `--batch-groups`, `--batch-telemetry`, `--offline`, `--no-fit-max-tokens`, `--script-cycle`, `--script-delay`;
- endpoint `GET /v1/telemetry`.

**`setup.py`:** the script reports `args`, `host`, `api_key`, `draft_vocab`, `open_browser` as run-config keys; all
existed before.

**Environment variables:**

- **Engine set-up and the pinned budget:** `STRATA_PIN_RESERVE_MIB` (on, §B2), `STRATA_TEST_PIN_BUDGET_MIB`,
  `STRATA_TEST_PIN_LEFT_KIB`, `STRATA_SM120_PERF` (on, §B2), `STRATA_HIP_VMM`.
- **Decode kernels:** `STRATA_GU_SWIGLU` (on for sm_120), `STRATA_KQ_SPLIT`, `STRATA_IQ_STAGE_TABLES`,
  `STRATA_PF_WMMA2`, `STRATA_FUSE_DECODE`, `STRATA_GR_NORM_FAST`, `STRATA_Q8_GATHER_WIDE`, `STRATA_Q8_SM60`,
  `STRATA_Q8_SM60_BPS`, `STRATA_ARGMAX_ROWS_SM70`.
- **Prompt path:** `STRATA_PF_WMMA` (ignored on the fork), `STRATA_PREFILL_WARM`, `STRATA_COLD_TRACE`,
  `STRATA_GDN_CHUNK`, `STRATA_GDN_CHUNK_PROF`, `STRATA_GDN_REC_WARP`, `STRATA_HIP_WMMA_MIN_Q`,
  `STRATA_HIP_WMMA_MIN_CAP`.
  - The script reports `STRATA_HIP_WMMA` "removed" from `prefill.cpp` and `generate.cpp`. It is still read, in
    `src/kernels/cuda/qsa_prompt_attn.cu` (upstream moved it behind `qsa_prompt_attn_matrix_cores_requested()`), so
    nothing is removed from the tree.
- **The file tier and the adaptive tier** (`expert_source.cpp`, `generate.cpp`; Linux defaults or file tier only):
  `STRATA_COLD_OVERLAP`, `STRATA_IO_PF_GATE`, `STRATA_IO_PF_GATE_LO`, `STRATA_IO_PF_GATE_HI`,
  `STRATA_IO_PF_GATE_USEFUL`, `STRATA_IO_PF_SCHED`, `STRATA_IO_PF_MINCONF`, `STRATA_IO_PF_TARGET`,
  `STRATA_IO_PF_BUDGET_MB`, `STRATA_IO_PF_BUDGET_FRAC`, `STRATA_RAM_ADAPT_QUEUE`, `STRATA_RAM_ADAPT2`,
  `STRATA_RAM_ADAPT_PROMPT_W`, `STRATA_RAM_ADAPT_CACHED_W`, `STRATA_RAM_ADAPT_PER_LAYER`, `STRATA_RAM_ADAPT_SKIP_TOP`,
  `STRATA_PREFILL_HOST_PLAN`, `STRATA_PREFILL_PLAN_TOKS`, `STRATA_SSD_GBPS`, `STRATA_IO_STATS`.
- **Telemetry:** `STRATA_BATCH_TELEMETRY`.
- **Server** (`serve/`): `STRATA_KEEP_LEFTOVERS`, `STRATA_NO_PRESET`, `STRATA_TELEMETRY_KEY`, `STRATA_LIFELINE`,
  `STRATA_RUNTIME_PROFILE`. From serve.md, read through helpers the regex does not match: `STRATA_NO_ACQUIRE`,
  `STRATA_OWNER_ATTEST_KEY`.
- **Setup:** `STRATA_CONTROL_INSTALL`, `STRATA_INSTALL_DIR`, `STRATA_SETTINGS_FILE`, `STRATA_ALLOW_UNPINNED`,
  `STRATA_DL_CONNECTIONS`, `STRATA_NEED_ENGINE`, `STRATA_ALLOW_NO_ENGINE`.
- Named by the area notes, not by the script's output:
  - `STRATA_SIDE_MIXER`, `STRATA_ATTN_DAILY`, `STRATA_GR_UP75`, `STRATA_GR_DOWN_TG`: names from step 1 (§9), now with
    sm_120 defaults;
  - `STRATA_SM86_PERF`, `STRATA_SM60_PERF` (`nv_defaults`);
  - `STRATA_IO_PREFETCH`;
  - `STRATA_NO_TQ_KERNELS` (its meaning changed, §B2).

**CMake options:** `STRATA_MMQ_TQ_OVERLAY` (new, ON). It is not seen by the script.

**New source files** (the script's list; it skips `apps/`, `metal/`, `sycl/`, `tools/`, `tests/` and test-named files):

- `include/strata/core/batch_telemetry.hpp`, `gfx12_platform_defaults.hpp`, `nv_defaults.hpp`, `pin_budget.hpp`,
  `sm120_series.hpp`;
- `include/strata/kernels/cpu/pool_plan.hpp`;
- `include/strata/kernels/fuse_decode.hpp`, `quantize_act_dev.cuh`, `s_rowdev.cuh`, `swiglu.cuh`;
- `include/strata/prefill/gdn_chunk.hpp`, `include/strata/program/prefill_plan.hpp`;
- `serve/launch_identity.py`, `serve/lifeline.py`, `serve/loader_diag.py`, `serve/owner_guard.py`, `serve/ready.py`;
- `src/core/nv_defaults.cpp`;
- `src/kernels/cuda/q8_sm60.cuh`, `src/kernels/cuda/s_gemv_pair.cu`;
- `src/prefill/gdn_chunk.cu`.

Also new, outside the script's view: `cmake/strata_ggml_tq_overlay.cmake`, `cmake/ggml_tq_overlay_test.cmake`,
`third_party/ggml-tq/`, `src/kernels/cpu/pool_plan_test.cpp`, `strata_control/acquisition.py`.

### B9. What is left from step 2

Developer decisions:

- **The sm_120 set on D2x:**
  - keep upstream's default, as the merge does now;
  - or turn it off on the fork (one line in `sm120_series()`).

  §5 shows parity with it on and no consistent speed change from `STRATA_SM120_PERF=0` in generate mode. The served
  ABBA ran with it on (the merge's default) and is x1.038-1.059 per turn; it was not split by this switch.
- **The pin reserve on pageable-arena configs** (D2x, Gyro-M): about 800 MiB of pinned RAM through the arena's open
  that protects nothing there. Either skip it when `!pin_for_cuda && !place_first` (a code change), or set
  `STRATA_PIN_RESERVE_MIB=0` in the configs (the developer's files).
- **The pool warning on D2x** ("13 workers ... Recommended: 9"): it is printed at every start. Whether 9 or 13 is
  faster for D2x is not measured here; the fork measured 5 P-core workers only for Gyro's trellis pool.
- **One build for D2x and Gyro-M:** §5's overlay question holds (Gyro-M bit-identical on the default ggml + overlay),
  so the two-build split of #258 can end. Which exe each key runs is the developer's choice.
- **Gyro pictures:** kept off; upstream's are on with engine 0.1.45.1+.

Code and docs to align:

- `tools/strata_mcp.py`'s Gyro entries say `vision: True`, but setup says False.
- `serve/runconfig.py` says `fit_max_tokens` is "on by default".
- The three serve tests that assumed upstream's `fit_max_tokens` default: adapted in `484eaaf0` (§5, pytest).
- The six upstream README translations describe upstream's app, not the fork. Either drop them on the fork or add a
  line at the top of each that points to `README.md`.

Gates and hazards:

- `native_q2_pool_hit_parity` with `STRATA_GU_SWIGLU=1`, to check the `gs_fork_ok` guard.
- `end_leftover_engines` against this PC's measurement scripts: an A/B that starts a server while an orphaned engine of
  the same exe is running loses that engine. `STRATA_KEEP_LEFTOVERS=1` in the measurement scripts, or one engine at a
  time as now.
- `/code-review` and `/scrutinize` on `git show --remerge-diff f06366df` (§7).
- `docs/BLUEPRINT.md`: the step-2 row (line 377) needs §B8's names that reach D2x when the tree is verified.
