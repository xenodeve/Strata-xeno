# Merge report: upstream v0.1.38 into the fork's main (2026-10-04, #129)

Upstream `Niko1221/Strata` `v0.1.38` (`99f3dbd`, 2026-10-03) merged into the fork's `main` `2fefaae` (which holds
v0.1.37, #103, and everything up to #147) on `xeno/129-merge-0.1.38`, worktree `C:\Strata-exp\src-138`, build dir
`C:\Strata-exp\build-138` (`build-138.cmd`: CUDA 13.3, sm_89 + sm_120, tests on). Working records:
`C:\Strata-exp\merge-138-record\` (the three sides of `prefill.cpp`, the per-hunk resolutions, the scripts that
removed the unported peer prompt path, the build, ctest and pytest logs).

**Speed is not measured here.** No figure below is a speed claim. The greedy parity runs, the decode-cluster parity on
the served config, the same-session ABBA against `main` (prefill 8K/32K, the Claude Code shape sm119, decode) and the
#61 re-check are pending; the orchestrator runs them on the exe this merge built.

## 1. What moved

- Upstream since v0.1.37 (`db4f91a`): 56 commits, 53 files, +5,023/-304.
- The fork since the same base: 390 non-merge commits. 26 files changed on both sides; 15 conflicted, 72 hunks in diff3 style (the issue's 67 counts the default style, where `prefill.cpp` has 26):

| file | hunks | what collided |
|---|---|---|
| `src/prefill/prefill.cpp` | 30 | upstream's peer share of the prompt path (PeerPrefill), #372's group gather (`used_of`), #374's `ple_land`, 3a91b9c's stager wait, against the fork's split (#35/#113/#133/#142), wave, copy issuer, expert order (#41), #29 group gather, #31 PLE read-ahead |
| `src/program/generate.cpp` | 14 | `--peer-device` and its adaptive tier, `--adapt-decay`, #463 `apply_pending(!adapt_nowait())`, `STRATA_TRACE_ADAPT`, #286 batched pair reads, `ple_inflight` 256, `vram_reserve_given`, against the fork's 4070 tier, paired swaps, timeline spans, flag blocks |
| `src/kernels/cuda/iq_kernels.cu` | 6 | #363's group stride and fused SwiGLU+q8_1 against the fork's Q2_0 CPU-order arithmetic (`x_scales`, #1) |
| `src/core/expert_source.cpp` | 5 | the peer tier in the verify window's dispatch and #285 unbuffered arena reads against the fork's tier counters, 4070 finish, placement-first `skip` and deferred load |
| `serve/server.py` | 4 | the Host/Origin checks (upstream's security change) against the fork's #71 Host check, `loading_server`, timing report, `/status` |
| `src/core/pinned.cu` | 2 | `STRATA_NO_LARGEPAGES` on Linux and the hugetlb shortfall note against the fork's `allow_large_pages` |
| `src/core/verify.cpp` | 2 | #363's `gy` / `kPcieGroupRows` against the fork's `x_scales` and its skipped PCIe call (`pcie_share_`) |
| `include/strata/core/expert_source.hpp` | 2 | `load_experts_gguf`'s new parameter on both sides (`skip` / `unbuffered`) |
| `include/strata/kernels/iq_kernels.hpp`, `include/strata/prefill/moe_mmq.hpp`, `src/prefill/moe_mmq.cu` | 1 each | same-named additions with different signatures (`native_expert_grouped`'s last parameter, `gather_native_group`) |
| `src/core/mtp.cpp`, `src/kernels/decode_cluster_parity.cpp`, `src/ngram/ple_reader_test.cpp`, `serve/test_server.py` | 1 each | includes; the #548 sync both sides added; the selftest's row sizes against the fork's prefetch check; imports |

Merge commit, later commits and the exe sha256: §6.

## 2. Upstream's changes that matter here, and what the fork does with them

The issue's named items, each with whether it reaches the served D2x path (`strata-flash-next-d2x.json`: Swift 1.5
IQ2_XS, `--exclusive-primary-experts`, paired adaptive swaps, the 4070 tier, split + wave, `--pcie-frac 0`,
`--kv int8`, `--kv-resident 65536`) and whether it is on:

| item | reaches D2x | on | why |
|---|---|---|---|
| #372 grouped expert gather (one launch, one wait, one event per MMQ group) | yes, as the fork's own #29 | yes (fork #29) | the fork's streamed walk already gathers an MMQ group in one launch after one wait and releases its slots together, under the same switch `STRATA_PREFILL_GROUP_GATHER` (default on). Upstream's version (`used_of`, the `flush` in `compute`) was not taken: two mechanisms for one job in one walk. Upstream's `gather_native_group(const GatherGroup&, ...)` overload is kept beside the fork's (§3) but nothing calls it |
| #374 the first chunk's PLE rows beside layer 0, 256 at a time | read-ahead: yes, as the fork's own #31; queue depth: no | read-ahead yes (fork #31); `--ple-inflight` stays 64 | the fork's #31 already reads the first chunk's rows on a thread while the embeddings and layer 0 run (`STRATA_PREFILL_PLE_AHEAD`). Ported from upstream: the take is idempotent per chunk, taken at the stage's first layer from 1 on and at the end of a stage that ends before layer 1 (upstream's `ple_land` rule), and 4f7b3e8's checked upload. **Left off:** upstream's `--ple-inflight` default 256 (it moves the served SSD queue depth); `--ple-inflight 256` is the ABBA arm |
| #413 DeltaNet recurrence, three value heads per thread | yes (prompt path, sm_80+: the 5060 Ti) | yes, as upstream | same bits by upstream's construction; `gdn_rec_parity` (new) checks it on this machine (§5). `STRATA_GDN_KEYHEAD=0` is upstream's A/B |
| #452 QSA prompt attention, Q4_0 mode 4 (sm_80+) | no | n/a | only with `--kv q4_0`; D2x runs `--kv int8` |
| #463 greedy independent of the adaptive tier's copy timing | no | yes where it applies | `apply_pending(!adapt_nowait())` waits for the non-paired adaptive swaps' copies. D2x runs the **paired** swaps (`--exclusive-primary-experts --adapt-swaps 8`), which keep no `pending`: their stage 3 still publishes residency on a non-blocking `cudaEventQuery`, the same timing dependence #463 removed. So #463 does not reach D2x, and #61's re-check under #463 is n/a on D2x; adding the wait to stage 3 is a fork follow-up, not merge work. `STRATA_ADAPT_NOWAIT=1` is the A/B; `STRATA_TRACE_ADAPT` traces the non-paired path |
| #363 fewer launches in the verify window | the fused pass: yes; the group stride: no | yes | `native_expert_grouped` now runs SwiGLU and the q8_1 quantization as one pass over the call's own entries (bitwise, per upstream) for every caller without `x_scales`: CUDA0's VRAM call, the 4070's `SecondaryRunner`, `remote_experts`. The group stride (`kPcieGroupRows`) applies to the PCIe call only, which the fork skips at `--pcie-frac 0` (`pcie_share_`). The fork's Q2_0 CPU-order path (`x_scales`) keeps its two kernels and one block row per group. `STRATA_GROUPED_V1=1` restores the old launches |

The rest:

- **`--peer-device N` (9d9a251 and follow-ups): a second GPU as an adaptive expert-cache tier.** Kept for decode, off
  unless given. **Refused** with the fork's 4070 tier (`--secondary-expert-mib`), `--exclusive-secondary-experts` and
  `STRATA_PREFILL_EXPERT_SPLIT=1`, beside upstream's own refusals (`--layer-split`, `--expert-cache-device1..3`): both
  tiers live on the second card and both mark their entries kind 2 in the verify window's dispatch. **Not ported:**
  upstream's peer share of the prompt path (`PeerPrefill`: the peer's rows of each chunk, its ring, compact buffers,
  `STRATA_PF_PEER_*`). The fork's prompt path is restructured (split, wave, issuer, expert order), and that code could
  not be run here. `Prefill::set_peer` declines with a message, which upstream's own caller handles ("the prompt path
  stays on the primary GPU"), the same as upstream's `--peer-prefill-rows 0`; `fused_ring()` stays off with a peer
  (734273a), as upstream.
- **Unbuffered expert loads on Windows (#285/#286/#357/#362):** kept. The arena's start read and the file tier's
  in-place reads go past the file cache when it cannot keep the experts. Not reached on D2x: its arena is deferred
  (placement-first), and the fork's own `load_rest` path reads it. `STRATA_UNBUFFERED_LOAD` is upstream's switch.
- **The draft layer's batched K/V for a ring (ba790d6):** with `--kv-resident` the drafter's K/V is a ring, and upstream
  moves its prompt-time appends onto the batched path. That is the D2x prompt path, so the fork turns it **off by
  default**: `STRATA_MTP_BATCH_RING=1` is upstream's default and the ABBA arm. `STRATA_DRAFT_TIMING` kept.
- **The fused layout's smaller buffers only when the native kernels cover every layer (220e0e8):** not taken in
  `fused_ring()`. Upstream needs "every layer" because its fused layout shrinks the MoE buffers. The fork's `moe_bufs`
  keeps MMQ's full size (the 0.1.37 merge), so "any layer" stays safe, and the mixed served pack (three IQ1_M layers)
  keeps the opt-in fused kernels on its covered layers. Off on D2x anyway (`STRATA_PF_FUSED` unset).
- **`--adapt-decay` (c38dc71):** kept; the default 0.7 is the same float as before, bit for bit. The fork's paired decay
  sites (serve and generate) read the flag too.
- **Server security for a server with no API key:** the Host check on every request (DNS rebinding), the Origin check
  on `/v1/*`, JSON-only `/load` and `/unload`. Combined with the fork's #71 check (§3).
- **#496 a small card's automatic reserve** (`vram_reserve_given`): only inside `--expert-cache auto` when the cache
  would fall below what the prompt path needs; D2x passes `--expert-cache 8000`, so not reached. **#542** MMQ's device
  facts from `cudaDeviceGetAttribute`: the same values in a matched build (CUDA 13.3 here).
- **Small ones, taken as upstream has them:** Q5_0 experts on the GPU (#473), IQ4_XS on AVX-2 (#415), #496 small-card
  reserve, #542 MMQ device facts, #545/#530 server messages, `STRATA_GR_DOWN_MAX4` (opt-in), active QSA top-k on Turing
  (`STRATA_TOPK_*`), the HIP MTP prompt-pass default, `STRATA_NO_LARGEPAGES` on Linux, setup and docs.

## 3. Every conflict, file by file

### `src/prefill/prefill.cpp` (30 hunks)

Resolved in place, from the fork's side, then audited line by line against the fork's pre-merge file
(`merge-138-record/prefill.ours.cpp`): the only remaining differences are the ports listed here. Git had auto-merged
several upstream pieces into the fork's code; `merge-138-record/excise_peer_prefill.py` and `fix2.py` removed the
unported ones.

- **Kept from the fork:** the split (`SplitTier`, `split_row_layout` / `local_first`, `cuda0_owns`, `mark_routed`), the
  wave, the copy issuer thread, the expert order (`expert_at` / `expert_pos`), the #29 group-gather walk, the #31 PLE
  read-ahead, the compact MMQ sub-products, the stream entries' 4070 / pack tiers, the timeline spans.
- **Ported from upstream:** 3a91b9c (a stager buffer's first job of a generation waits for the previous generation's
  last DMA from it; skipped jobs do not wait); 4f7b3e8 (the PLE upload and its event checked); #374's `ple_land` rule
  folded into the fork's `ple_take` (taken once per chunk, at `max(LB, 1)`, and at the end of a stage); ba790d6's
  ring path in `draft_kv` (off by default, §2) and `STRATA_DRAFT_TIMING`; 734273a's `peer_portable()` guard in
  `fused_ring()`; the `cudaHostAllocPortable` flag with a peer; the stub of upstream's `gather_native_group` overload.
- **Not taken:** #372's `used_of` / `group_gather` / `flush` (the fork's #29 is the same mechanism); `PeerPrefill`,
  `m.pp`, the peer ring (`ps_on`, `p_issue_until`, `p_release_to`), the peer row layout (`on_peer`, `order_peer`,
  `rows_local`), `peer_now`, `PeTimer` (§2); 421a31d (it fixes a line of the peer path); 220e0e8 (§2).

### `src/program/generate.cpp` (14 hunks)

Includes and help text both sides. `ple_inflight` stays 64 with upstream's 256 as the arm (§2); `--ple-ahead` kept.
`adapt_every` keeps the fork's `-1` (resolved by mode) and gains `adapt_decay`. `private_commit_bytes()` and
`adapt_nowait()` both. The fork's flag block with upstream's `vram_reserve_given`. Hunk 7: the fork's placement-first
fill, then upstream's #286 batched pair reads ahead of the fork's `!place_first` loop. The 4070 tier's `set_peer_tier`
and the wave set-up, then upstream's `set_peer` (declines, §2). The adaptive candidate filter keeps the fork's 4070 and
lent-tail rules and adds upstream's `!peer.has(l, e)`. Both `apply_pending` sites keep the fork's timeline spans and
call `apply_pending(!adapt_nowait())`. `adapt_rounds` and the `STRATA_TRACE_ADAPT` lines go into the fork's
`adapt_primary` (the non-paired branch; the paired stages are not traced). New: the refusal of `--peer-device` with
the fork's 4070 tier and split.

### `src/kernels/cuda/iq_kernels.cu` (6) and `include/strata/kernels/iq_kernels.hpp` (1)

`native_expert_grouped(..., stream, const float* x_scales = nullptr, int64_t grid_groups = 0)`: the fork's parameter
first, then upstream's; upstream's two callers that passed `grid_groups` (`native_grouped_parity.cpp`) now pass
`nullptr, grid_groups`, and `verify.cpp` passes both. `native_gu_kernel` / `native_down_kernel` take upstream's group
stride with the fork's `x_scales` per entry. `q8_1_store` (upstream's device function) gains the fork's optional
`scales` / `hx`; with both null it is upstream's q8_1 bit for bit, and the fork's `quantize_q8_1_kernel(x, y, scales,
hx, n)` calls it. The launch: the Q2_0 CPU-order path (`x_scales` on a Q2_0 layer) keeps one block row per group
(`cap_groups`), `swiglu_entries_kernel(..., cpu_order)` and the scaled quantize; every other call takes upstream's
fused `swiglu_q8_1_entries_kernel` unless `STRATA_GROUPED_V1=1`. Upstream's `native_expert_scratch_bytes(cap, ff)`
callers (`peer_experts.cpp`, `native_grouped_parity.cpp`) take the fork's three-argument form (`n_embd`).

### `src/core/expert_source.cpp` (5) and `include/strata/core/expert_source.hpp` (2)

`load_experts_gguf(..., threads, const uint8_t* skip = nullptr, bool unbuffered = false)`; the unbuffered reader now
honours `skip` too. `ArenaExpertSource::open`: a deferred load (the fork's placement-first) reads nothing, as before;
otherwise upstream's #285 choice (unbuffered or buffered, its log line). The dispatch: upstream's `peer_entries` and
its race-free row zeroing beside the fork's `tier_entries`; after the CPU work the fork's 4070 `finish`, then the
peer's `finish`. The includes of both.

### `serve/server.py` (4) and `serve/test_server.py` (1)

One Host check. Upstream's `parse_request` runs it for every method, so it answers before the fork's `_host_ok` could;
`_host_ok` and the fork's two-argument `host_allowed` are gone. Upstream's `host_allowed` gains the fork's #71 rules: a
name with no dot (a LAN machine) and a `.local` name pass. The check also reads `svc.allowed_hosts` live, as the
fork's did (a name added after `serve()`). The fork's set-valued parse of `allowed_hosts` is gone; upstream's
`allowed_hosts_of` (list or string, plus `STRATA_ALLOWED_HOSTS`) is the one parser. **Visible change:** a refused Host
is now **403** (was 421 in the fork); `test_server.py`'s three assertions say so. `/unload` and `/load`: the fork's
`_foreign_origin` (403 for a foreign page whatever it sends), then upstream's `_own_page` (JSON only). The fork's
`loading_server` and upstream's host helpers both; the fork's timing report, upstream's #530 note and the fork's
`ascii()` raw log; `/status` keeps `loops_stopped`. The fork's web app sends `Content-Type: application/json` to
`/v1/chat/completions` (`serve/ui/src/lib/api.ts` `apiHeaders(true)`), so upstream's Origin rule does not refuse it.

### The rest

- `include/strata/prefill/moe_mmq.hpp` / `src/prefill/moe_mmq.cu`: both `gather_native_group` overloads. The fork's
  #29 form `(blobs, n, up_off, down_off, gu_half_bytes, ...)` returns void and falls back per expert inside the call;
  upstream's #372 form `(const GatherGroup&, up_off, gu_half_bytes, down_off, ...)` returns false. One
  `kGatherGroupMax = 16`. The fork's private `GatherGroup` / `copy16_group_kernel` were renamed `XenoBlobSet` /
  `copy16_blobs_kernel` (upstream's public `GatherGroup` has the name).
- `src/core/verify.cpp`: both arguments (above); the fork's `if (pcie_share_)` block keeps the PCIe call, which now
  passes `kPcieGroupRows`.
- `src/core/pinned.cu`: the fork's `allow_large_pages` first, then upstream's `STRATA_NO_LARGEPAGES` and the shortfall
  note (Linux only; not compiled here).
- `src/core/mtp.cpp`: both includes. `src/kernels/decode_cluster_parity.cpp`: upstream's comment on the same sync the
  fork had added (#548). `src/ngram/ple_reader_test.cpp`: upstream's two row sizes, then the fork's prefetch check.

## 4. Semantic hazards checked beyond the markers

- **Auto-merged upstream code that referenced unported mechanisms** (`used_of` in the issue paths, the fused block and
  `compute`; `m.pp` in the resident branch of the streamed walk; the peer timers): found by grep and by diffing the
  merged `prefill.cpp` against the fork's side, removed, and the diff re-read until only the listed ports remained.
- **Same-named functions with different signatures:** `gather_native_group` (overloads), `native_expert_grouped`
  (combined, every caller checked by grep), `native_expert_scratch_bytes`, `load_experts_gguf` (combined; a `bool`
  passed in the pointer's place would not compile), `host_allowed` (Python: the later definition would have silently
  replaced the earlier one; one left).
- **The verify window's `kind == 2`:** the fork's 4070 tier and upstream's peer both use it; the refusal keeps them apart.
- **Defaults that would move D2x:** `ple_inflight`, the drafter ring batch, the fused predicate, the adaptive decay;
  each kept at the fork's value (§2).
- **Line endings:** `git checkout -m` rewrote `generate.cpp` with LF; git normalises on commit (no diff in content).

## 5. Verification

- **Build:** `build-138.cmd --target strata`, then the whole tree (219/219 steps), CUDA 13.3, sm_89 + sm_120: no errors
  at the first build (`merge-138-record/build-strata-1.log`, `build-all-1.log`).
- **ctest** (serial, no `-j`, #140; PATH with CUDA `bind`): **105/107** (`merge-138-record/ctest-138.log`). The two
  failures are the known ones of the fork's main: `ple_parity` (the Q2_0 file was deleted) and `expert_multi_test` (no
  AVX-512 on this CPU). `ple_reader_selftest` passed (12.8 s). Upstream's new tests pass here: `native_grouped_parity`
  (#363: the group stride and the fused pass bitwise against the old launches), `gdn_rec_parity` (#413),
  `qsa_topk_active_parity`, `decode_cluster_parity` (on the 5060 Ti), `native_expert_parity_q4_K_q5_0` (Q5_0, #473).
  The fork's own pass too, among them `gpu_4070_iq_parity_layer{0,1,8,12,35}` and `gpu_4070_iq_graph_formats` (the 4070
  runner, which now takes the fused SwiGLU+q8_1 pass), `xeno_gather_group_parity` (#29's overload),
  `xeno_pool_fused_parity`, the split and combine parities. `native_q2_pool_hit_parity` is not registered (the Q2_0
  file is gone), so ADR 0001's bit-exact gate was not run; the Q2_0 CPU-order path it guards keeps its two kernels.
- **pytest `serve` + `tests/xeno`:** 1091 passed, 8 skipped, 219 subtests (`pytest-serve-xeno.log`), upstream's
  `serve/test_security.py` among them.
- **pytest `tools/test_*.py`** (with `STRATA_GGUF_PY` pointing at the build's llama.cpp `gguf-py`): 287 passed,
  1 skipped, 1 failed: `test_setup_draft_vocab.py::SmallCardNote::test_sizes_follow_the_shipped_subsets` (`KeyError:
  'cjk'`, the fork's `DRAFT_VOCABS` has no `cjk` subset since #55 W7). It fails the same way on the pre-merge `main`
  (`git archive HEAD`, the same command): not this merge's; a follow-up.
- **Not run here (pending, the orchestrator):** greedy raw-token parity against `main` on the D2x config, the
  same-session ABBA (prefill 8K/32K, the Claude Code shape sm119, decode) and #61 under #463. No engine was started on
  the GPUs beyond ctest.

## 6. What moved: commits and the exe

- Base: `main` `2fefaae`; upstream tag `v0.1.38` (`99f3dbd`); branch `xeno/129-merge-0.1.38`.
- The merge commit: `3933ed5` (parents `2fefaae`, `99f3dbd`); it adds this report. The commit after it corrects this
  report and notes the fork's peer limits in `docs/SECOND_GPU.md` (no code).
- Engine: `C:\Strata-expuild-138\strata.exe`, sha256 `beea3e729c886033b5ecf08adc58c518d5d84bb3871870ba3fac4dd2280b031c`
  (74,470,400 B, built from the merge's tree).

## 7. What is left

- Parity and ABBA (pending, the orchestrator): greedy ABBA against `main` on D2x, the decode-cluster parity, prefill
  8K/32K, the Claude Code shape (sm119), decode. Candidates for the ABBA arms: `--ple-inflight 256`,
  `STRATA_MTP_BATCH_RING=1`, `STRATA_GROUPED_V1=1` (against the default fused pass).
- #61 under #463: n/a on the paired swaps (§2); the fork follow-up is to wait for stage 3's copies (or to make the
  GPU and CPU expert products round alike).
- Upstream's peer share of the prompt path is not in the fork (§2). `docs/SECOND_GPU.md` (upstream's) says so under `--peer-prefill-rows`.
- Not run on this merge: `/simplify`, `/code-review`, `/scrutinize`; the #129 comment linking this report.
- `tools/test_setup_draft_vocab.py` fails on `main` too (`KeyError: 'cjk'`): a follow-up.

## 8. The classic web app

Nothing changed: upstream's diff has no file under `serve/web/` (xeno #67).
