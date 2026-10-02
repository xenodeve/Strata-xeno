# Merge report: upstream v0.1.34 into the fork's main (2026-10-02/03, #100)

**Companion document.** `2026-10-02-upstream-0.1.34-blueprint.md` covers three things:
- Part A: upstream v0.1.34 area by area;
- Part B: upstream beside the fork beside the merge;
- Part C: the merge checkpoint.

This report is the merge's own record: what moved, every conflict, the hazards checked, the bugs found and the verification.

## 1. What moved

**Base and upstream.**
- **Base:** the fork's `main` at `1c2b520` (PR #98 merged). It is the 0.1.30 line with capacity mode, the N0 evidence and the review fixes.
- **Upstream:** `Niko1221/Strata` tag `v0.1.34` (`1678de3`, 2026-10-02), i.e. releases 0.1.31, 0.1.32, 0.1.33 and 0.1.34.
- **Merge base:** `v0.1.30` (`30ec18e`).

**Size.**

| | commits | files changed |
|---|---|---|
| upstream | 189 | 246 |
| fork | 255 | 238 |
| both sides | | 41 |

- On the fork side, the merge brings 248 files, +62,853 / −1,884.
- About 30 % of upstream's diff is community benchmark data (`bench/results/`).

**Conflicts and the build.**
- 25 files conflicted; 120 hunk decisions are recorded.
- Branch `xeno/exp-upstream-0.1.34`; the merge commit SHA is filled in at commit.
- Engine exe `build-134\strata.exe`: sha256 `9fc5e9d229c996f0…`. The first build was `725ae7c0dcae7d64…`, before two review edits.

## 2. Upstream's changes that matter here, and what the fork does with them

**Taken.**

| change | release | notes |
|---|---|---|
| Faster Q2_0 / IQ expert kernels (decode-once multi kernels, #241 #242) | 0.1.31 | Upstream says they are bitwise identical. Q2_0 without `x_scales` now runs them. |
| Native Q4_K / Q5_K / Q5_1 / Q8_0 experts | 0.1.31 | |
| Per-role GGUF layout (`native_experts.txt` v4) and `check_experts_gguf` | 0.1.31 | |
| `FileExpertSource` GGUF-in-place with transient blobs | 0.1.31 | |
| `--resident-budget-gib` (static RAM budget, mmap reads, router-predicted prefetch) | 0.1.31 | |
| The #266 late-request status fix, the #268 CJK tokenise speedup, Windows load speed (#230) | 0.1.31 | |
| `PoolAffinity` (#272) | 0.1.31 | |
| The pin cap only under WDDM, `STRATA_ARENA_PIN_GIB` (#243 #253) | 0.1.31 | |
| The prompt-path step-down retry `init_prompt_paths` (#253) | 0.1.31 | |
| Async commit (`Verifier::set_commit_async`, `STRATA_COMMIT_SYNC`) | 0.1.31 | |
| Per-layer cache admission (#369), setup that recommends instead of forcing, a portable vision encoder | 0.1.33 | |
| Hang-up cancel within ~1 s (#430 #431) | 0.1.34 | |
| MMQ `J_best=0` FP16 fallback (#420) | 0.1.34 | |
| AMD on Windows | 0.1.34 | Taken; not built here. |
| `tools/strata_mcp.py`, the README / docs split, `requirements.txt` pins | 0.1.34 | |

**Taken but unreachable in the fork.** These are 0.1.32's multi-GPU split refill pieces:
- refilling every card at once (`refill_issue` / `refill_wait`, `STRATA_REFILL_SERIAL`);
- the small-chunk own buffers (`STRATA_SPLIT_SMALL_OWN` / `MAX`).

The fork refuses a layer split that lends any card's cache (its split + wave prompt path lends CUDA0 only, #56). The merge widened that refusal from a CUDA0 loan to *any* stage loan, because upstream's #340 own-buffers rule could otherwise let a stage borrow unrefused. **This is a developer decision point:** to make them reachable, the fork would have to support per-stage loans with the wave.

**Not taken.** The `cjk` draft-vocab key: the fork removed it in `ac8b6be`. `DRAFT_VOCABS` is {thai, en, cyrillic}.

## 3. Every conflict, file by file

Full per-hunk record (both sides, the side kept, what was dropped and why): `C:\Strata-exp\merge-134-record\wf134.json` (compact: `wf134-hunks.txt`). It was produced by workflow `wf_2db5d0ab-5d4`: one resolver and one adversarial verifier per file group, then fix and re-verify. Summary:

**`src/program/generate.cpp` (16 records).**
- *Includes:* both.
- *Pin cap:* upstream's computation plus the fork's `pin_for_cuda`. The fork's `ArenaExpertSource::open` argument order is kept, together with its `= delete` guard against upstream's order. Upstream's pin info lines print only when the arena is pinned.
- *Pool:* `ExpertPool(workers, pin, host_works, spin_us, affinity)`.
- *Boot fill:* the fork's pipelined boot fill with upstream's per-layer admission ported in.
- *CUDA0 loan:* upstream's `lend_first >= 0` guard.
- *Prompt path:* upstream's `init_prompt_paths` retry. It re-inits **both wave lanes** (lane 2's init moved into the lambda; `sp2.reset()` was added), and CUDA0's `own_fits` uses `wave_bytes_needed` when the wave is on. The split+loan refusal is widened (see §2).
- *Read windows:* `trace_commit` sits right before `ver.commit`.
- *Refill and lend:* the fork's code, plus upstream's `STRATA_TRACE` timing lines.
- *End of decode:* `wait_commit` first.
- *Auto-merged code, fixed:*
  - upstream's two `wait_commit` exits inside the serve loop became `return serve_fatal();` (the #53 rule);
  - the request-start `wait_commit` moved above the fork's cache-slot switch.

**`serve/server.py`, `serve/test_server.py`, `serve/frontend.py` (31 records).**
- *Engine life:*
  - `StrataEngine._pump` keeps the fork's signature and takes upstream's order (`ended` before the `None` sentinel).
  - `restart()` runs the fork's #59 degraded logic on upstream's `close()` (QUIT, terminate, kill) instead of `kill()`.
- *Request life:*
  - upstream's #266 settle structure wraps the fork's loop;
  - one thinking/reasoning budget: the smaller wins, cut at a clean point;
  - `load(wait_for_restart)`: OpenAI requests queue during a (re)load, while `/v1/messages` keeps the fork's immediate 529;
  - `count_tokens` takes upstream's query-stripped dispatch, which also fixes the fork's 404 on `?beta=true`.
- *Answers:* a hung-up client's cut answer is never replayed (`cancel.client_gone`, new test); `_merge_legs` sums the tier counters too (new test).
- *Endpoints:* `/health` and `/api/health` keep the fork's alive logic. CORS fields come from upstream.
- *Tests:* both sides' classes are kept. Upstream's `ThinkingBudget` test class is renamed `ReasoningBudget`, because the fork's #49 S3 class owns the name.

**`src/prefill/prefill.cpp`, `kernels.hpp`, `moe_mmq.hpp` (17 records).**
- *Staging:* one `Stager::Job{src, bytes, from, l, e}` with three kinds: copy_blob (upstream's transient), memcpy, and an NVMe read from the pack (the fork's #11). Upstream's "no blob" error is dropped for unpinned experts: under #11 a null blob means a pack read.
- *Cleanup:* `Prefill::reset()` / `release()` follow upstream's structure, with the fork's cleanup and the #45 exit-trace probe.
- *Ring:* the HIP ring branch uses the fork's split-layout threshold.
- *Kernels:* the `_lo` BF16X2 parameters default to `nullptr`.

**`src/kernels/cuda/iq_kernels.cu`, `native_expert_parity.cpp` (8 records).**
- Both kernel families are kept. The dispatch is `if (type == 42 && x_scales) { the fork's CPU-order Q2 path } else switch (STRATA_GU_FMTS / STRATA_D_FMTS)`.
- The fork's `is_iq` one-liner is removed for upstream's superset.
- `iq_dequant_expert_f16` now accepts K-quants.

**`expert_source.{hpp,cpp}`, `expert_layout.{hpp,cpp}` (16 records).**
- Upstream's v4 per-role GGUF layout is taken. The fork's readers (`read_expert`, `expert_file(..., r)`, `submit_reads` grouped per source, `load_experts_gguf_direct`, `read_into`) open the per-role shard.
- The deferred open now runs `check_experts_gguf`.
- `expert_pool_dispatch_multi` runs upstream's prefetch block, then the fork's NVMe block. Each is a no-op on the other's source.
- The CPU feature probes keep all three: `cpu_avxvnni_ok`, `cpu_avx2_ok`, `cpu_name`.

**`pool.{hpp,cpp}` (6 records).**
- Upstream's `detect_cpu_topology` holds the fork's EfficiencyClass sort.
- The fork's non-empty `STRATA_POOL_SPIN_US` rule is kept.
- A stale, uncompilable auto-merged block was removed.

**`device.cu`, `device_main.cpp`, `verify.cpp` (7 records).**
- Upstream's `kMinCc` / `kNeed` floor, with the fork's `allow_display_sm89`.
- One argument loop with `--list-devices` plus the fork's secondary probes.
- In `Verifier::commit`, the fork's commit span comes first, then upstream's async-or-sync block (see §4).

**`ngram.cpp` (3 records).** One `PleTable::Impl` with FP8 rows and the fork's prefetch ring. The prefetch buffer is sized `n * rb` (see §5).

**`setup.py`, `tools/draft_vocab.py` (9 records).**
- Upstream's `MIN_ENGINE` and `requirements.txt` pins are taken.
- The fork's PDF packages are pinned in `XENO_REQUIREMENTS`, deduplicated against `requirements.txt`. The resolver added `typing_extensions==4.16.0; python_version < "3.11"` because pypdf needs it below 3.11. **That version pin is UNVERIFIED.** It does not apply on this PC's Python 3.11.
- No `cjk` (§2).

**`.gitattributes`, `AGENTS.md`, `CMakeLists.txt`, `docs/DETAILS.md` (7 records).**
- `AGENTS.md`: upstream's guidance first, then a bridge sentence, then every fork section.
- `CMakeLists.txt`: both sides' targets and tests, with the if/foreach depth checked.

## 4. Hazards checked beyond the markers

- **Serve-loop exits (the #53 rule).** `merge-134-record/serve_returns2.py`: the loop spans 6341-7437 in `generate.cpp`. Its only hit is a one-line lambda at 7171, a false positive. Upstream's two new `wait_commit` exits were converted to `serve_fatal()`.
- **Async commit against fork code that reads the session (`/scrutinize`).** Upstream's contract is that anything reading the session from another stream or the host after a commit calls `wait_commit()` first. Traced in the serve and generate decode loops:
  - checkpoint and slot saves sit after the request-start wait (6563) or the decode-end wait (7195); the cancel path breaks to the latter;
  - `mtp.draft` reads only the window's rows, its own K/V and scratch the commit graph does not write. The fork's `mtp.cpp` differs from upstream by 13 lines;
  - `PleAhead` reads host state only;
  - the adapt / 4070 swaps write expert slots, not session state;
  - the commit graph writes the GDN state and conv, the QSA indexer state and `ple.hist`; none of the fork's code between `commit` and the next `run` reads them.
  The greedy parity gate (§6) exercises the path with the 4070 tier on.
- **A measurement that changed meaning.** With async commit, `Verifier::ms_commit`, the `verify window … commit` counter and the commit span time only the launch; the GPU time moves into the next window's wait. **Compare decode stages across this merge only with `STRATA_COMMIT_SYNC=1` on both arms, or compare total ms/window.**
- **A capacity-mode policy changed.** In `prefill.cpp` `stage_one`, the second `blob()` call for unpinned experts was dropped. In capacity mode, a chunk below the stream-all size now raises the decayed-LFU score once per layer, not twice, and the "expert blobs read" count falls. #84's cache-policy numbers were measured with the old scoring, so a capacity-mode A/B across this merge carries this confound.
- **Restart time.** `StrataEngine.restart()` now ends the old engine with `close()`: QUIT, then 20 s, then terminate. The fork used to `kill()` at once. A #59 tier restore can take up to ~20 s longer if the engine hangs at exit (#45).
- **Silent argument-order changes.** The fork's `= delete` overloads on `ArenaExpertSource::open` and `PinnedArena` stay, so upstream's order would not compile. The build found no such call. Every `ExpertPool(` site, tests included, uses (n, pin, host, spin_us, affinity), checked by the spec reviewer.
- **Struct layouts the fork serialises** (cache slots, checkpoints). The hazard sweep found no new upstream field in `ConversationCheckpoint` since 0.1.30. One gap predates this merge: `slot_qsa` has no `kv_hybrid` (`--kv k8v4`) branch. It is recorded in §7.
- **Format gate.** Upstream's startup check `native_expert_supported(gu, d, n_embd, n_ff)` at `generate.cpp:2431` runs on every layer before any allocation. An unsupported pair therefore never reaches the fork's per-type capacity-mode gate.
- **Numerics.** The FP16 SwiGLU now saturates at ±65504; only tokens with huge activations change. `s2_expert_grouped` was rewritten; upstream says it is bitwise identical, and it has a parity test.
- **New configuration surfaces**, from `merge-134-record/new_surfaces.py`:
  - engine flags: `--coupled-draft`, `--no-coupled-draft`, `--embd-gguf`, `--pool-affinity`, `--resident-budget-gib`, `--split-skip-if-fits`; all present in the merged `generate.cpp`;
  - 57 new `STRATA_*` names, including CMake options and macros;
  - no flag or env var was removed.

## 5. Bugs the merge introduced, found and fixed before the merge commit

| # | bug | found by | fix |
|---|---|---|---|
| 1 | upstream's `return 1` inside the serve loop (async-commit waits) | hazard sweep / resolver | `serve_fatal()` |
| 2 | request-start `wait_commit` after the fork's cache-slot switch, so the switch could read a session still being committed | hazard sweep | moved above the switch |
| 3 | `PleTable::prefetch` buffer sized for 90 B rows while FP8 rows are 160 B: a heap overflow in Direct mode with the row cache | resolver (auto-merged code) | `n * rb` |
| 4 | `native_expert_parity.cpp` and `iq_multi_parity.cpp` called the fork's 3-arg `native_expert_scratch_bytes` with 2 args (compile error) | hazard sweep / verifier | pass `n_embd` |
| 5 | a stale auto-merged block in `pool.cpp` used an undeclared `cores` | resolver | removed |
| 6 | the `load_experts_gguf` declaration plus a definition with a default made two overloads | resolver | default moved to the header |
| 7 | the fork's readers indexed the per-layer `gguf_file`, which is per role in v4, so they read the wrong shard | resolver | per-role lookup |
| 8 | `tests/xeno/nvme_aligned.cpp`'s raw-byte "GGUF" fixture was rejected by `check_experts_gguf` | verifier | a real GGUF via `tests/core/gguf_fixture.hpp` |
| 9 | `test_anthropic_stream.py`'s `FakeService` lacked upstream's `model_for` | verifier | added |
| 10 | `tools/test_setup_draft_vocab.py` indexed the removed `cjk` key | verifier | `thai` |
| 11 | `expert_file(gguf, "", true, ...)` was a wrapper for `expert_gguf_file` | `/code-review` | direct call |
| 12 | the `load()` docstring said "a 503" where `/v1/messages` answers 529 | `/code-review` | corrected |

## 6. Verification

| gate | result | data |
|---|---|---|
| build (`build-134.cmd`, tests on) | 321/321, 0 errors (first try) | `C:\Strata-exp\build-134.log` |
| `pytest serve tests/xeno` | 368 passed, 5 skipped | console |
| `pytest tools` (with `STRATA_GGUF_PY` = the llama.cpp gguf-py) | 237 passed, 1 skipped | console |
| ctest | 81 / 97 pass (see below) | `C:\Strata-exp\ctest-134.log` |
| greedy output parity against pre-merge `main` (A = `run-main-1c2b520` `e2f8e042…`, B = this merge `9fc5e9d2…`), Swift 1.5 **IQ2_XS** (IQ3_XXS was deleted from disk; SHA-256 of both shards checked), the served capacity-mode args (`--ram-cache-gib 12`, the 4070 tier at 6400 MiB, SECONDARY COMPUTE, split + wave env), 256 greedy tokens, A B B A per prompt in one session | **12/12 identical**: thai-net `6f2763a8` ×4, py-async `3f8f5ec1` ×4, long8k (8K prompt, the prompt path) `46bf06b2` ×4 | `C:\Strata-exp\merge-134-record\parity134.out`, `parity-runs/*.stdout` (pack `D:\Github\Strata\packs\swift15-iq2_xs`, native_experts v3 from the pre-merge `tools/iq_pack.py`) |
| `/code-review` (standards + spec) | no hard violations; spec: the six high-risk areas trace clean | §8 |
| `/scrutinize` (async commit) | no violation of the wait contract found | §4 |

**ctest passes** include:
- `iq_multi_parity` (upstream's new multi kernels against the old ones, Q2_0 and the IQ types);
- `native_expert_parity_*` (Q4_K/Q8_0, Q5_K/Q8_0, Q4_K/Q5_1, Q5_1, Q8_0, BF16 embd, Q6_K refused);
- `sampler_parity` (split, one-block, old) and `qsa_parity`;
- `s2_expert_grouped_parity`, `xeno_gr_fused_parity`, `xeno_hit_merge_mapped`;
- `xeno_moe_layer_cross_arch`, `xeno_combine_cross_arch`, `xeno_combine_split_parity`, `xeno_frontier_combine_parity`;
- every `xeno_nvme_*` test (with the new GGUF fixture), `xeno_pool_spin`, `xeno_secondary_*`.

**The 16 failures** fail identically on the pre-merge `build-cap`:
- 13 need a model file that is no longer on disk: Swift IQ3_XXS (`gpu_4070_iq_*`, an uncaught `GgufFile` open, `0xC0000409`) or Q2_0 (`gpu_*q2*`);
- `ple_parity` (bench/micro fixtures) and `expert_multi_test` (it needs AVX-512, which this CPU lacks) were already failing before the merge.

## 7. What is left

- **Decision for the developer:** whether to support per-stage loans so upstream's 0.1.32 all-card refill and small-chunk own buffers become reachable (§2).
- **Pre-existing gap:** `slot_qsa` has no `kv_hybrid` branch. Refuse `cache_slot` with `--kv k8v4`, or add the branch.
- **Watch:** the #342 `superseded` counter with `ckpt_at` checkpoints; never set `STRATA_ARENA_PIN_GIB=auto` for D2x.
- **Tests to add:** K-quant cases for `tests/xeno/dequant_expert_fused.cpp` (the fork's single-launch dequant is now reachable for K-quant packs).
- **Speed** across the merge is not measured, because benchmarks are paused. Compare with `STRATA_COMMIT_SYNC=1` (§4).
- **Not done here:** AMD/HIP and the Windows HIP package (not built on this machine); setup's new paths.
- **The served engine** is not moved by this merge.

## 8. The review

- **`/code-review` standards:** no hard violations. Owed and done in this commit: this report and the BLUEPRINT row.
  - Fixed: smell 2, the `expert_file` wrapper.
  - Left as judgement calls: the reopen-on-name-change logic repeated in three readers; the null-pointer-kinded `Stager::Job`; the `cancel.client_gone` attribute; `_count_tokens` unlinking Service-staged files; the two fill loops' repeated `kNotResident` branch.
- **`/code-review` spec:** acceptance items 3 and 5 are met or in progress; 4, 6 and 7 are covered by this report, the parity run and the PR.
  - It flagged the LFU change, the `ms_commit` meaning, the slower `restart()` and the unreachable 0.1.32 pieces; all four are recorded above.
  - It corrected the ctest attribution: `ple_parity` and `expert_multi_test` were already failing before the merge.
  - Fixed: the docstring.
- **`/scrutinize`:** the async-commit wait contract holds on every fork path between `commit` and the next `run` (§4).
