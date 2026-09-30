# Merge report: upstream v0.1.30 into the 0.1.26 line (2026-10-01, #56)

The blueprint and checkpoint of this merge (`AGENTS.md`, "Write a merge report for every upstream merge"). It records
what moved, how every conflict was resolved and why, the three bugs the merge introduced, and what verified the
result. The review of upstream's releases that led to it is in #45 (comment 5918311153).

## 1. What moved

| | |
|---|---|
| base | `xeno/exp-upstream-0.1.26-dyn` at `8f8098e` (upstream `v0.1.26` = `ac8b251` + the fork's dynamic experts, D2x, #49 S7, the Thai draft subset) |
| merged | upstream tag `v0.1.30`: 112 commits after `v0.1.26`, 125 files, +11,397 / −880 |
| branch | `xeno/exp-upstream-0.1.30-dyn` (local, not pushed), worktree `C:\Strata-exp\src-dyn` |
| merge commit | `eb2ec1f` |
| files changed on both sides | 33 files, 18 of them with conflict markers |
| engine exe | `C:\Strata-exp\build-130\strata.exe`, copied as `C:\Strata-exp\run\strata-a8f70684094de8af.exe` (build dir `build-130`, `STRATA_BUILD_TESTS=ON`, script `C:\Strata-exp\build-130.cmd`) |
| pre-merge line | worktree `C:\Strata-exp\src-126` on `xeno/exp-upstream-0.1.26-dyn`, engine `strata-26f5974288933cb3.exe` |

## 2. Upstream's changes, and what the fork does with them

| upstream | taken? | notes |
|---|---|---|
| #197 sampled top-k split across the GPU (0.1.29) | yes | the fork's CJK ban (#49 S4) ported into the new kernels (§3, `sampler.cu`) |
| #187 / #188 QSA and GDN prompt kernels | yes | auto-merged |
| #154 correctness (bf16 NaN, s_gemv barrier, KV-stream mask, table guards) | yes | auto-merged |
| #210 tool-call values may contain `</parameter>` / `</tool_call>` | yes | auto-merged in `serve/` |
| #183 cancelled request fails the next (`6cad1fe`) | one copy | the same fix as the fork's #53 (`e988f4d`); the fork's line and comment kept |
| #224 CUDA fault in the prompt path exits at once | yes, as `serve_fatal()` | `serve_fatal()` already `_Exit`s |
| 1024-token streaming for 1K-4K prompts (`8acd17c`) | off the split layout only | the split layout (#32/#35) keeps 2048: its one-card MoE buffers and wave lanes are sized for it; not measured lower |
| #216 multi-GPU session carve, per-stage prompt loans | carve yes; loans for CUDA0 only | a layer split with prompt borrowing is refused at start (§3, lending) |
| #189 conversation cache | yes, opt-in, off | coexists with the fork's cache slots (§3, `want_cvec`) |
| #84 rope scaling past 262K | yes, opt-in | the fork's #29 QSA append overload dropped (§3) |
| #208 idle unload, `/load`, `/unload`, `--min-free-vram-mib`, `--before-load` | yes | `RequestGate` gained a non-blocking acquire (§3) |
| #199 the draft head inside the cache's reserve | yes | slot counts unchanged here: 6,222 primary in both arms |
| resident low-RAM variant, shared arena (#129), RDNA4, Pascal, RTX 20, Docker | yes, inert here | not used by this machine's configs |
| CJK draft subset (`6e153c9`, 106,299 ids) | **no** | the fork keeps its Thai subset (46,252 ids): the CJK guard bans Han output, so CJK drafts would cost ~110 MiB and 1-2 % English decode (upstream's figures) for nothing |
| AVX2 i-quant prefetch (`fa6310b`) | yes | covers the i-quant rows only, not the Q2_0 kernel this machine runs (#55 W4) |

## 3. Every conflict, file by file

| file | resolution | why |
|---|---|---|
| `CMakeLists.txt` | both options (`STRATA_BUILD_XENO_TESTS`, `STRATA_BUILD_CONVERSATION_TESTS`); the `xeno_qsa_append_batch` target removed | the test covered the dropped #29 overload |
| `setup.py` | upstream's `MIN_ENGINE = (0, 1, 30)`, the fork's `PY_PACKAGES` (+ `pypdf`, `pypdfium2` for #46) | |
| `tools/draft_vocab.py`, `data/draft_vocab.bin` | the fork's | the tool is upstream's plus a `thai` script; the subset is the Thai one (§2) |
| `src/core/verify.cpp` | the fork's timeline spans with upstream's `qsa_primary()` | |
| `src/kernels/ngram.cpp` | both includes | |
| `src/kernels/cuda/native_qsa_indexer.cu`, `include/strata/kernels/native_qsa_indexer.hpp` | **upstream's, whole files** | the fork's only change was #29's one-launch append overload, unused since the prompt path moved to upstream's C-2 batch; upstream's version adds rope scaling |
| `include/strata/core/pinned.hpp`, `src/core/pinned.cu` | one constructor: `(bytes, bounds, bool pin_for_cuda = true, uint64_t max_pinned_bytes = 0, shared_file = {}, shared_pack_hash = 0)`; `reserve()` takes both the fork's `allow_large_pages` and upstream's shared-file mapping | the two sides had different third parameters (the fork's `bool`, upstream's `uint64_t`) |
| `include/strata/core/expert_source.hpp`, `src/core/expert_source.cpp` | `open(..., err, bool pin_for_cuda, bool defer_load, uint64_t max_pinned_bytes, shared_arena_file)`; the fork's NVMe-tier virtuals and upstream's `pcie_layer` both kept; `defer_load ? reserve_only(...) : new PinnedArena(..., shared_arena_file, pack_hash)` | same parameter clash |
| `src/core/device.cu` | upstream's arch checks and RTX 20 floor (cc 7.5) with the fork's `allow_display_sm89` signature | |
| `src/core/mtp.cpp` | the fork's timeline GPU spans around upstream's coupled / plain graph choice | |
| `src/kernels/cuda/sampler.cu` | greedy and the old path: the fork's `kBan` templates; default path: upstream's split kernels **with the ban added** (`split_part`: a banned id is −inf after its penalty; `one_block`: skipped) | without it a sampled request (Claude Code's main turns) would have lost the CJK guard silently |
| `src/prefill/prefill.cpp` | both constants; `stream_all = T >= (g_split_layout ? STREAM_ALL_MIN : stream_all_min())`; the fork's buffer report with `qsa_primary()`; upstream's per-token append comment | §2, 1024-token streaming |
| `serve/server.py` | imports and fields: both; upstream's `ensure_loaded()` wrapped in the fork's `restarting` flag and `EngineDied` (a 529); `load()` answers 529 at once while restarting; the fork's stop-sequence tail plus #212's status pops; `/health`: the fork's 503 for a dead engine, not for an unloaded one, plus `loaded`; the cache-slot check before `svc.load()`; the fork's `cjk_ban` plus upstream's idle-unload settings | |
| `serve/server.py`, `RequestGate` (auto-merged, broke a test) | `acquire(priority, blocking=False)` takes the gate only when free and nobody waits | upstream's idle unload called `fifo.acquire(blocking=False)` on what it assumed was a Lock |
| `serve/test_server.py` | both test sets | |
| `src/program/generate.cpp` | see below | |

`generate.cpp` (16 hunks):

- **Two helper functions:** the fork's `parallel_copy` and upstream's `resident_stage_swaps`, both.
- **The argument parser's split chain:** both sides had split the `else if` chain (MSVC C1061), one with `parsed` and one with `matched`. Both are declared, the blocks nest, and each gets its closing brace.
- **Checks after parsing:** upstream's conversation-cache checks, plus the fork's `--ban-ids` check and upstream's HIP arch check.
- **CPU banner:** the fork's "native Q2_0 expert rows use AVX-VNNI", then upstream's no-AVX-512 line and its rope-config block.
- **The MTP / arena / pool / expert-cache block.** Upstream moved it; the fork's copy is kept and re-homed after upstream's session allocation (§4, bug 1). Upstream's two edits to it were ported in: `&& !early_remote`, and `o.shared_expert_arena` passed to `open`. Upstream's copy was deleted.
- **Adaptive swaps (both loops):** the fork's paired swap first, then upstream's `resident_stage_swaps` call.
- **INFO line:** `spec_min_p`, `ban`, then the `conversation_cache_*` fields. 20 specifiers against 20 arguments, counted with `count_info.py`. `cache_slots=` is on its own INFO line and is unchanged.
- **`err.clear()` (#53 against #183):** the fork's line.
- **Cache slots against `want_cvec`:** the fork's slot switch is kept. Its own control-vector block is dropped, because upstream now invalidates the session and checkpoints for a cvec switch after a parked conversation may have been restored, and keeping both would do it twice.
- **Lending.** Upstream's per-participant `pf_parts` setup computes the chunk and the first lendable slot, and `lend_first_now` and `lend_bytes` are restored. The fork's `lent_now` / `refill` / `lend` with the wave relayout replace upstream's `refill_one` / per-part `lend`. A layer split with prompt borrowing is refused at start, because a stage's loan would hold prompt buffers in slots its rows still name as experts.
- **#224:** `serve_fatal()`.
- **Adapt summary lines:** both.

## 4. Hazards checked beyond the markers

- **Silent `uint64_t` → `bool` conversion.** Upstream's call order puts `max_pinned_bytes` where the fork has `pin_for_cuda`. `= delete` overloads that take `uint64_t` in that slot were added to `ArenaExpertSource::open` and `PinnedArena`, so such a call fails to compile. They stay as guards. The build found no such call.
- **Serve-loop exits.** The #53 rule is that nothing in `while (next_line(line))` returns from `main`. `serve_returns.py` counts `return 1;` inside the loop, skipping lambdas. The 0.1.26 line had 0; the merge brought 5 new ones from the conversation-cache code, and all 5 are now `serve_fatal()`.
- **Duplicated features.** Upstream has no #175 `cache_slot`: its `cache_slot_off` / `conversation_cache_slots` are other things. #53 and #183 were deduplicated.
- **Per-file delta.** The fork's delta on 0.1.30 was compared with its delta on 0.1.26: `prefill.cpp` +1,551/−204 against +1,548/−205, `server.py` +593/−50 against +582/−49, `expert_source.cpp` +563/−6 against +562/−5. `generate.cpp` is +2,383/−289 against +2,223/−126; the difference is the re-homed block and the lending replacement.
- **Line endings.** The working copy has CRLF; git normalises to LF, as before.

## 5. Bugs the merge introduced, found and fixed before the merge commit

1. **Access violation at start (`0xC0000005`, after "301 native projection matrices").** The re-homed MTP block called `mtp.load(..., ss, ...)` before `session_init`, which upstream (#216) moved after the split search. Found by running the exe directly (`gen_pair.py`); the serve log showed only "the engine exited before it was ready". **Fix:** the block now sits after the session allocation. It still loads the drafter before the host arena, which is #45's WDDM constraint.
2. **Illegal address on the first 11K prompt in serve mode** (`prefill gr_broadcast: an illegal memory access was encountered`, both B boots). Generate mode passed on an 11.9K prompt with identical output, which pointed at the serve loan. Upstream's `part_slots` sized CUDA0's loan with `Prefill::bytes_needed` (one lane), while the fork's wave lays out two lanes, so lane 2's buffers ran past the cache. **Fix:** `part_slots` uses the fork's `prompt_bytes_needed`, which is `wave_bytes_needed` when the wave is on.
3. **`loading conversation cache slot failed`** (`check_cache_slots.py`, HTTP 400, then the engine exited). Upstream 0.1.30's checkpoint (`ConversationCheckpoint`) gained `dead` and `block_pos`, and `conversation_checkpoint_restore` copies them per layer. The fork's slot file did not carry them. **Fix:** `CacheSlot::Shape` and `slot_checkpoints` write and read both.

Also seen once, not reproduced: `secondary expert (7,350) slot 313 failed its fill: bytes differ`. On one boot of the final exe the 4070 tier's fill verification refused a slot, and the engine stopped rather than serve wrong bytes. The next boot of the same exe filled cleanly. It is the first occurrence in this session's logs (`engine-s7-130chk*-fillfail.log`). **Hypothesis:** the display card's known instability (TDR history); not investigated.

## 6. Verification

| gate | result | data |
|---|---|---|
| build (`build-130`, tests on) | pass | `scratchpad/live/build-130*.log` |
| `sampler_parity --selftest` on the split (default), one-block and old paths | all pass, including the ban fixtures (greedy, tie, sampled seeds 1-7), run on the 5060 | console output |
| `tests/xeno` | 109 passed | |
| `pytest serve` | 136 passed, 3 skipped (121 of the 0.1.26 line + upstream's new ones) | |
| generate mode, the same args, 0.1.26 exe against the merge | code256: output `09c72854` in both, 92.2 / 91.0 tok/s; 11,893-token prompt: `66ab11a9` in both | `gen_pair.py` |
| serve ABBA, the #54 request set, A = 0.1.26 line, B = merge, order A B B A | **20/20 outputs identical**: code `76ae5b9f`, thai `d91d1754`, read `d766075e`; drafts accepted identical (171/172, 87, 4) | `scratchpad/live/ab-130.jsonl` |
| decode speed | equal where the machine was in its normal state (the ABBA run before the loan fix: code 87.0 / 91.0 and 86.2 / 90.3 against 86.9 / 90.3 and 86.7 / 90.1; Thai 55.0 / 60.3 and 55.0 / 59.7 against 54.8 / 59.5 and 54.0 / 59.8) | `ab-130-void-loan.jsonl` (decode rows; its B read8k rows failed) |
| prompt read speed | **not settled**: the final ABBA's later boots ran in a slow machine state that hit both arms (A's last boot too) | `ab-130.jsonl` |
| `check_cache_slots.py` | PASS: independent reuse, cold-output parity, retrieval, recovery after an interrupted stream | console output |
| #53 cancel repro (`repro53.py`) | a 76K prompt dropped after 22 s, the next streamed and JSON requests answer (72919) | console output |
| final exe (`a8f70684094de8af`) on the #54 set | outputs `76ae5b9f` / `d91d1754` / `d766075e` | `ab-130-final.jsonl` |

**Not run:** `/code-review`, `/scrutinize`, `/simplify` on the merge; upstream's HIP tests; the conversation cache, rope scaling and idle unload in use; a layer split.

## 7. What is left

- **The slow machine state.** Decode sometimes runs ~40 % slower in serve mode, in both arms and across boots (code ~55 against ~87 tok/s). It first showed during the tier-7400 trial, and it recurs at 6400. Cause unknown; to be opened as an issue with its logs.
- **1024-token streaming on the split layout** needs its own measurement before it is enabled there.
- **Per-stage prompt loans** (upstream #216) with the wave are not supported: a layer split with borrowing is refused.
- **The served engine** stays on the daily branch until the developer moves it. This branch becomes the blueprint's baseline when it does; until then the merge is a row in "Not yet in the baseline".
- **Push and PR** wait for the developer's word, then `/code-review` and `/scrutinize` before merging to `main`.
