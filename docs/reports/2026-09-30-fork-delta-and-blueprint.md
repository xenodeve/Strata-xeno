# Strata-xeno: every change from v0.1.2 through two upstream merges - upstream PR plan and system blueprint

Date: 2026-09-30. Tree read: `C:/Strata-exp/src-dyn`, branch `xeno/exp-upstream-0.1.26-dyn`, HEAD `5c51574`.
Nothing was built, run or checked out to write this report. The sources are `git log/show/diff/grep/merge-tree`,
the reports under `docs/reports/`, the merge checkpoint report (read from commit `070d1b8`, because it is not in the
`5c51574` tree), and read-only `gh issue view` on `xenodeve/Strata-xeno` and `Niko1221/Strata`.

**Evidence labels used throughout.**

- **MEASURED** quotes a number and names its source: a commit message, a report file or an issue comment.
- **VENDOR** is upstream's own claim, quoted from its commit, doc or PR comment. It was not measured here.
- **static reading** means a mechanism inferred from reading the code, not observed in a run.
- **UNMEASURED** marks a claim or number that has no source.
- A `file:line` is at `5c51574` unless it names another SHA.

**Measurement caveat for every number below.** Numbers from different days come from different builds and placements.
`510d06a` (`docs/reports/2026-09-28-ggml-cpu-build-mode.md`) shows that the same exe can produce different tokens
across sessions: the drift depends on boot state and was first seen in prefill, and its mechanism is unproved. Only
same-session ABBA pairs are valid comparisons. The noise gate that the commit messages cite is 13.6 %.

---

## 0. Summary

**What the fork is.** Strata-xeno is a fork of upstream Strata (`Niko1221/Strata`), a MoE inference engine for
Qwen3.8-Flash-Next with 48 layers, 512 experts and top-10 routing. It serves the model from a GPU expert cache, a CPU
expert pool and host RAM. Upstream targets one 12-24 GB NVIDIA GPU with 64 GB of RAM, and has an experimental AMD
HIP backend. The fork targets one machine:

- **Primary GPU:** RTX 5060 Ti 16 GB on a PCIe x4 link.
- **Secondary GPU:** RTX 4070 SUPER 12 GB on x16. It is also the display card.
- **CPU:** i5-13500 with AVX-VNNI and no AVX-512.
- **RAM:** 48 GB DDR5.
- **Model and OS:** the Q2_0 native pack only, on Windows 11.

**Lineage.**

- The fork's base is upstream v0.1.2 (`6da1f66`). We added 105 commits (`6da1f66..f7ecc89`).
- **Merge 1** (`2a9b26c`, #30) took upstream v0.1.20 (`c1e9033`). We then added 77 commits (`2a9b26c..7134ce9`).
- **Merge 2** (`6c09c5b`, #45) merged our branch into upstream `4c68013` (v0.1.26 plus docs). Two commits
  followed: `7b31aa6` (the exit-hang probe) and `5c51574` (the #48 fix).
- Our delta against `4c68013` is 184 non-merge commits, 190 files, +18,025/−469.
- Upstream is now at v0.1.27 (`a790805`), which is **not merged**.

**Headline features, by area.**

- **Memory ownership (#4).**
  - Exclusive GPU ownership of experts: a cached expert has no host copy.
  - Paired adaptive swaps under that ownership.
  - Placement-first cold start: a GPU-owned expert never enters RAM.
  - A tail file for the cache slots the prompt path borrows.
- **Second GPU.**
  - A 4070 decode expert tier, `SecondaryRunner`.
  - A dual-GPU prompt path in two parts:
    - "split": whole MoE layers run on the 4070.
    - "wave": two prompt lanes run over the split.
- **Numerics.**
  - Q2_0 GPU hits that are bit-exact with the CPU pool (#1).
  - MMQ without stream-k, so prompt MMQ products are bit-exact across sm_89 and sm_120 (#5).
- **Speed.**
  - Q2 kernel K0-K4.
  - An AVX-VNNI Q2 CPU row kernel.
  - CPU pool priority and rest.
  - LAZY CUDA module loading plus a warm-up.
  - PLE read-ahead in decode.
  - Group gather and group-local MMQ scratch in the prompt path.
- **Serving and diagnostics.**
  - Claude Code serving parity: `count_tokens`, the billing-header strip, stop sequences, the loop guard, PDFs.
  - A whole-pipeline timeline.

**Package-level result (MEASURED, #47 body and #45 steps 2-3 comment; D = merged, U = upstream 0.1.26).** These are
package numbers: #47 itself says "The D-vs-U numbers prove the package, not each mechanism".

- **RAM, one card** (same U/O/D session): working set 23.2 vs 33.2 GiB.
- **RAM, two cards** (same session): 16.5 vs 35.3 GiB against U's layer split, at equal code decode.
- **8K prompt, two cards vs U's best.** Like for like (same U/O/D session, generate mode, #45): D's split + wave
  4.1-4.2 s (1,961/1,888 tok/s) against U4, upstream on the 4070 alone, 4.6-4.8 s (1,669/1,733 tok/s): about −12 %,
  **below the 13.6 % gate**. #47's headline "D2x 4.1 s vs U3 serve 7.3 s" is **cross-session and cross-mode**: D2x is
  from m11 (merged engine only, generate mode, 8K borrowed chunks) and U3's 7.28/7.26 s from the earlier session in
  serve mode, which is slower (D1 reads 8K in 6.6 s generated vs 7.29 s served). U3's layer split is also held to
  2,048-token chunks by upstream's own code (`generate.cpp:1722`, `d733199`; `:1743-1746`, `928b0e0`). #47 keeps the
  caveat "U's best one card 1,700 tok/s"; D2x's 1,940 tok/s is about +14 % over that, across sessions.
- **Decode with the 4070 as the one card:** D4 63.6 vs U4 49.6 tok/s (#47, "mechanism not isolated").
- **Output determinism:** "D stable; U's thai changes run to run" (#47, "U's cause unknown").
- **Where U wins:** decode on the 5060 alone (U1 68.4 vs D1 64.6) and the CPU pool (22.9 vs 27.9 ms/round).

**PR shortlist (full ranking in §6).**

1. MMQ nsm=1 (opt-in).
2. Decode PLE read-ahead.
3. LAZY module loading plus warm-up (#47 PR 1).
4. AVX-VNNI Q2 rows (#47 PR 9).
5. Group-local MMQ scratch (#47 PR 6).
6. One gather launch per MMQ group.
7. Exclusive ownership plus paired swaps (#47 PR 2 and PR 3 merged).
8. Placement-first (#47 PR 4).
9. Tail file (#47 PR 5).
10. Mapped hit merge and pool rest.
11. Q2_0 CPU-order bit-exact GPU path with K0-K3 (opt-in).
12. Six small Claude Code serving PRs.
13. Exclusive secondary device caches (#47 PR 7).
14. Dual-GPU prompt path, split plus wave (#47 PR 8, an RFC that must carry `5c51574`).

**Dropped.** Upstream already has or rejected these:

- K4 `pcie_share` skip: upstream rejected the same idea from PR #109 (`1e4515c`, by another contributor) in `9a19f57`;
  our `6790f01` was never submitted.
- `--pcie-frac` default 0.
- Batched indexer append.
- Batched embedding.
- First-chunk PLE async (the pre-merge-1 version).
- The parallel stager.
- The copy-issuer thread.
- The #42 ports.

**Blockers before any multi-GPU PR.**

- The generate-mode exit hang (`7b31aa6`). §8 gives a static hypothesis.
- A static-reading race between PLE prefetch and the wave in serve.
- The cross-session prefix-reuse loss seen with split + wave on (#48; a wave-independent cause is also possible, §8).
- Mid-prompt checkpoints taken while the other wave lane advances the state (static reading, §8 item 13).
- Windows-only page decommit on a default-on path.
- A GPU identity hard-coded into the 4070 tier.

---

## 1. Lineage

```
f2a08d4 (v0.1.0)
   |
6da1f66 (upstream v0.1.2 = fork base, branch main)
   |\
   | \--105 ours (101 + 4 upstream cherry-picks)--> f7ecc89 (xeno/claude-q2-kernel)
   |                                                    \
   |--93 upstream (non-merge)--> c1e9033 (v0.1.20) ----> 2a9b26c  MERGE 1 (#30), parents f7ecc89 + c1e9033
   |                                |                      |
   |                                |                      +--77 ours--> 7134ce9
   |                                |                                       |   \--> 6167b6e, 070d1b8 (docs, checkpoint report)
   |                                |                                       |        --> 694ad82 (#46 serve fix) --> 12556b1,
   |                                |                                       |        e63d359, 55eb129, c54c337, 80cd0b3, 1620329
   |                                |                                       |        (#49 serve work) = xeno/claude-merge-0.1.20;
   |                                |                                       |        NONE of these is in HEAD (see below)
   |                                +--93 upstream--> 4c68013 (v0.1.26 + docs; tag v0.1.26 = ac8b251)
   |                                                    |    \
   |                                                    |     6c09c5b  MERGE 2 (#45), parents 4c68013 + 7134ce9
   |                                                    |       |
   |                                                    |     7b31aa6  exp(#45): exit-hang probe, STRATA_EXP_QUICK_EXIT
   |                                                    |       |
   |                                                    |     5c51574  fix(#48): per-lane drafter K/V  <-- HEAD
   |                                                    |
   |                                                    +--8 upstream--> a790805 (v0.1.27, NOT merged)
```

The table below comes from `git rev-list --count --no-merges` and `git diff --shortstat`, run for this report.

| segment | who | non-merge commits | diffstat (files, +/−) |
|---|---|---|---|
| `6da1f66..f7ecc89` | ours, 101 + 4 upstream cherry-picks (`8f2187d`, `c996648`, `dd05359`, `9aedbd2`) | 105 | 128, +9,481/−249 |
| `6da1f66..c1e9033` | upstream v0.1.2 → v0.1.20 (119 with merges) | 93 | 115, +14,987/−979 |
| `f7ecc89..2a9b26c` | what merge 1 brought into our branch | — | 116, +14,625/−1,093 |
| `2a9b26c..7134ce9` | ours after merge 1 | 77 | 81, +9,348/−333 |
| `7134ce9..070d1b8` | docs only, the checkpoint report (not merged; `origin/xeno/claude-merge-0.1.20` is here) | 2 | — |
| `070d1b8..1620329` | serve code on the local `xeno/claude-merge-0.1.20`, unpushed, **not in HEAD** (§1 below) | 7 | — |
| `c1e9033..4c68013` | upstream v0.1.20 → v0.1.26 + docs (109 with merges) | 93 | 166, +18,132/−610 |
| `4c68013..6c09c5b` | our side as merge 2 laid it on upstream | — | 190, +17,969/−459 |
| `6c09c5b..5c51574` | after merge 2: `7b31aa6`, `5c51574` | 2 | 2, +58/−12 |
| `4c68013..5c51574` | **our total delta vs upstream** | 184 (+2 merges) | **190, +18,025/−469** |
| `4c68013..a790805` | upstream v0.1.27, not merged | 8 | 12, +316/−55 |
| `6da1f66..5c51574` | everything since the fork base | — | 402, +50,665/−1,579 |

**Authorship.** 180 of the 184 non-merge commits in `4c68013..5c51574` are by `xeno`. The other four are the
upstream cherry-picks. Each is patch-equivalent to its upstream original (`git cherry` shows `-`):

| ours | upstream | what |
|---|---|---|
| `8f2187d` | `dd03276` | engine-queue race |
| `c996648` | `23f6b15` | conversation cache |
| `dd05359` | `c47e290` | pool sleep |
| `9aedbd2` | `9d9351b` | short-read |

We reverted `9aedbd2` in `eab0066`. The code came back through merge 1.

**Unmerged branches.** `xeno/afk-phase1` (`9ffb846`) and `xeno/phase1-candidate` (`8c28ad0`, a prefix of
afk-phase1) are not ancestors of HEAD.

- Their code is in HEAD as `9a2b807` (= `8c28ad0` by src/include/tests patch-id) and `dadb3a9` (= `9ffb846`).
- They differ only in `docs/OPEN-WORK-LEDGER.md` text, some report and ADR docs, and a one-line re-indent in `pool.cpp`.
- `git cherry` marks them `+` only because the ledger text differs.
- Three surveys agree that nothing on them is missing from HEAD. They can be deleted or kept as historical refs.

**`xeno/claude-merge-0.1.20` is NOT an ancestor of HEAD, and it carries code HEAD lacks.** `git merge-base
--is-ancestor xeno/claude-merge-0.1.20 5c51574` fails. `git log 7134ce9..xeno/claude-merge-0.1.20`, re-run during
verification (branch tip `1620329`; `origin/xeno/claude-merge-0.1.20` = `070d1b8`, so everything after it is local and
unpushed):

| commit | time (+0700) | what | in HEAD |
|---|---|---|---|
| `6167b6e`, `070d1b8` | — | docs: the merge checkpoint report | no |
| `694ad82` | 10:19 | fix(serve): a PDF or screenshot returned by Read inside a `tool_result` reaches the model (#46); `serve/frontend.py` +22/−1 (`_tool_result_content`), `serve/test_server.py` +44 | no (`git grep _tool_result_content 5c51574 -- serve` finds nothing) |
| `12556b1` | 10:38 | fix(serve): a dead engine fails the request as 529 `overloaded_error`; `disable_parallel_tool_use` (#49 S2, #48 defect 2) | no |
| `e63d359` | 10:45 | feat(serve): a thinking budget (#49 S3) | no |
| `55eb129` | 10:53 | feat(serve): llama-server-shaped timing block; `STRATA_TRACE_SSE` (#49 S5). A verifier saw it as `023b925`, which is no longer on any branch (rewritten) | no |
| `c54c337` | — | feat(serve): streamed requests go before queued non-streamed ones (#49 S6) | no |
| `80cd0b3` | — | feat(serve): coding sampling presets chosen in the run config (#49 S8) | no |
| `1620329` | 11:03 | feat(serve): CJK guard, server side (#49 S4) | no |

So HEAD `5c51574` still drops PDFs and screenshots inside a `tool_result` (#46, still OPEN). This branch moved while
this report was being written; the list above is a snapshot and must be re-run before use.

The other local `xeno/*` branches are ancestors of `5c51574`: afk-phase2/3/3-compute/4, afk-serving,
afk-vram-saturation, avxvnni, claude-prefill-mtp, claude-q2-kernel, codex-secondary-timing, exp-upstream-0.1.26 and
short-read-candidate. (Every one was checked with `git merge-base --is-ancestor`.)

**Merge conflict sizes.** Merge 1: 14 files, 73 conflict hunks. Merge 2: 14 files, 75 hunks, reproduced read-only
with `git merge-tree c1e9033 4c68013 7134ce9`. The `6c09c5b` message says "~74 hunks". The per-file split is in §4.2.

---

## 2. What upstream brought in, and how it overlaps ours

### 2.1 Merge 1: upstream v0.1.2 → v0.1.20 (`6da1f66..c1e9033`, absorbed in `2a9b26c`)

**Serving and conversation.**

- **`23f6b15` (v0.1.3), conversation cache.** Keeps the live session plus up to 6 checkpoints of the state that cannot
  be rewound: 36 GDN recurrences plus conv, 12 QSA indexer tails and the PLE history, about 118 MB each. A prompt then
  reads only its new suffix.
  - Flags: `--prompt-cache N` (6), `--prompt-cache-every` (16K), `--turn-token`.
  - New protocol lines: `RESUME`, `PP`, `REUSED`.
  - Follow-ups: `6fb2085` pins the root and rotates the rest LRU; `c1e9033` adds `--prompt-cache-root N` (min 2,048).
- **`9d9351b`, `--short-read N` (64).** Reads short prompt parts through the verify windows. VENDOR: "a one-word
  message waited ~1.4 s for its first token".
- **Per-request sampling**, PR #19 (`111c58a`…`3e0836f`), then `3e93670`, `72386cc` and `a49e810`. VENDOR
  (`4d39f7a`): "sampled decode 1.50 -> 42.7 tok/s".
- **Server fixes and features:**
  - `dd03276`: the queue race.
  - `a40fe52`: `max_tokens` −1 means the rest of the context.
  - `c22ca55`: the EngineDied restart.
  - `261d2d8`: `/v1/messages?beta=true` and mid-conversation system messages, which make Claude Code work.
  - `65bfaf7`: an MCP client.
  - `75e50e6`: per-request hit rate in `DONE`.
  - `09bff3f`: `setup --calibrate`, plus the GEN keys `pcie_frac=` and `spec_min_p=`.
  - The web app.

**Engine memory.**

- **`e02643e` (v0.1.5), KV streaming** (`--kv-resident`). VENDOR: "Q2_0 at 262K decodes 62.6 tok/s instead of 50.9".
- **`0680ab6` (v0.1.9)**, head before cache; `--vram-reserve-mib` default 700.
- **`6609960` (v0.1.15), `CUDA_MODULE_LOADING=EAGER` forced.** VENDOR: it fixes "out of memory: cudaFuncSetAttribute"
  for IQ3_XXS at 64K+ on 12 GB cards.
- **`6a6de69`**: the expert cache retries at 3/4 size on commit-limit failures.
- **`0bf3216` / `fb1fb57`**: `MEM_LARGE_PAGES`.

**Decode and CPU.**

- **`c47e290`**: pool sleep after 20 ms.
- **`06438fd` (v0.1.12)**: the pool claim protocol with 60 s bounded waits and a serve watchdog.
- **`f333c71` / `d2b05ef`**: Linux physical cores.
- **`ed14227` … `a3545f5`**: AVX2 multi-token i-quant rows in `iq_avx2.cpp`. They do not cover native Q2_0 (static
  reading).
- **`7a4b627` + `4befda7`, the PCIe-share probe.** Formula at `4c68013`:
  `bw>=20 ? 0.55 : bw<4 ? 0 : min(0.55, max(0.05, 0.55*bw/26))`.
- **`e6265c7` / `4d4014c` (v0.1.14)**: `--pcie-mode auto` uses the copy kernel for native packs.
  `data/expert-profile.bin` now ranks all 24,576 experts.
- **`a9047fd`**: the suffix drafter on by default.
- **`9e599c0`**: control vectors.
- **`be6d09b` / `a278979`**: `--kv q4_0`.

**Prompt path.**

- **`928b0e0` (v0.1.13), "about 2x faster long prompts".**
  - `--prefill auto`: chunks up to 8192, carved from borrowed cache slots.
  - Experts through llama.cpp MMQ.
  - A stream-all ring for chunks ≥ 2048.
  - Batched PLE, with the next chunk's PLE read on a thread.
  - Stager threads.
  - VENDOR: "Q2_0 32K prompt 572 -> 1290 tok/s".
- **`5917a05` / `b89c989`**: MMQ tail zeroing (fixes NaN) and no writes into borrowable slots.

### 2.2 Merge 2: upstream v0.1.20 → v0.1.26 + docs (`c1e9033..4c68013`, absorbed in `6c09c5b`)

**Upstream multi-GPU.**

- **`a4a5a37`, helper-GPU expert caches.**
  - Flags: `--expert-cache-device1..3`, `--expert-cache-remote-placement stripe|layer`.
  - Files: new `remote_experts.{hpp,cpp}`.
  - VENDOR (`docs/SECOND_GPU.md`): it "does not release the expert arena in system RAM".
  - Follow-ups: `9fba872` (spin-scheduled helper, zero-copy) and `ba79e39`.
- **Layer split** (`293c51c` … `2fb26a6`; engine `f1b1d96` = 0.1.21). Flags: `--layer-split`, `--split-device`,
  `"gpu": [0,2]`. Serve only. VENDOR (`docs/MULTI_GPU.md`): "Prompts gain the most (+18-20%)", and one GPU is
  byte-identical to 0.1.20.
- **`d9a4974`**: below 4 GB/s the PCIe share is 0.
- **`f36af4d`**: sm_80+ is accepted at run time.

**Prompt-path perf-review series** (upstream's C-/D-/E-/F- labels):

| commit | label | what | VENDOR claim |
|---|---|---|---|
| `5b7e316` | C-4 | one-launch embedding plus a DirectFile I/O pool (`STRATA_IO_THREADS`) | "PLE 591 -> 291 ms", byte-identical |
| `758eb1a` | C-1/C-2 | QSA select grid; indexer append in 3 launches | "4K 1063 -> 1255 (+18%)", byte-identical |
| `2575cb1` | C-3 | GDN conv tiled | identical bits |
| `66f4341` | D-2 | GDN recurrence split | "GDN 5118 -> 4264 ms" |
| `dce4598` / `a751715` / `4755942` | D-1 | QSA prompt attention on tensor cores | "+18.8%"; **not bitwise** |
| `581765a` | D-4 | lent-slot refill with queued copies | "162 -> 96 ms per prompt" |
| `cf68b00` | D-5 | per-chunk issuer thread | "1143 -> 1213 tok/s", byte-identical |
| `f0bbe97` | E-4 | drafter prompt pass without per-group sync | "1091 -> 905 ms" |
| `c6c6594` | E-9 | drafter prompt pass batched through MMQ (`Prefill::draft_kv`) | "0.9 s -> 0.07 s of a 32K prompt" |
| `731899f` | — | QSA select on tensor cores (3xTF32) | "+14.6% at 128K"; **not bitwise** |
| `fe609ce`, `882bb6d`, `b046845` | —, F-1, F-2 | mapped grouping tables; HC read and write fusion | bit-identical |

**Decode.**

- **`1e4515c` (PR #109, merged as `9a19f57`)**: batches the verify window's kernels, including
  `native_moe_combine_multi` (`verify.cpp:729`, `STRATA_DEC_BATCH`). VENDOR: "bit-identical". The `9a19f57` merge
  note matters to us: *"Left out: the session-wide --pcie-frac 0 skip of the PCIe path, because a request can ask for
  a PCIe share (pcie_frac=, used by --calibrate) and the window graph is captured once."*
- **`efddd74` / `0abcd5b`, E-6**: a device plan, opt-in (`STRATA_VERIFY_DEVICE_PLAN=1`).
- **`df6980d`, E-2**: AVX-512 i-quant prefetch.
- **`afdba4c`**: an incremental detokenizer.

**Other.**

- `--kv k8v4` (`2aa8f72` / `c48690a`).
- The AMD HIP backend, gfx1100 (`daa121b` and follow-ups).
- `--mmap-experts` for native packs (`de159b5`).
- Bulk fread (`5edb9d6` / `cc1d8e8`).
- GGUF checks (`c80131e`, `a349ec3`, `5427527`).
- Serve additions: `28e6c8e` `cached_tokens` and `/v1/status`; `d9fba59` `/models`, `/props`, `/slots` (llama.cpp-
  compatible discovery); `89e68f6` image sampling; `cb91f2a` image parts kept in tool/assistant messages (overlaps
  our #46 fix, §1); `61833e2` draft counts in timings.
- Prompt-path and layer-split fixes that touch our paths: `9295f64` (1,024-token chunks with no expert cache),
  `d6ff0b8` (an explicit `--expert-cache` leaves room for the prompt path under a layer split), `15345e6` (layer-split
  mid-prompt checkpoints taken per stage; the serve wave has the same hazard unfixed, §8 item 13).

### 2.3 Overlap analysis

**D1: upstream duplicates or supersedes ours.**

| ours | upstream | state at HEAD |
|---|---|---|
| `767fcfc` batched QSA indexer append (#29). MEASURED: 7473/7044/6983 → 6763/6750/6742 ms | `758eb1a` C-2 | upstream's is called (`prefill.cpp:2471`); our overload at `native_qsa_indexer.cu:270` is dead engine code, used only by `tests/xeno/qsa_append_batch.cpp` |
| `4935d9b` batched chunk embedding (#31) | `5b7e316` C-4 | replaced; `STRATA_PREFILL_EMBED_BATCH` is gone; `tests/xeno/embed_batch_parity.cu` is still built |
| `f7ecc89` PLE async read (#29) | `928b0e0` `ple_gather`/`ple_read` | replaced at merge 1 |
| `fdaddbe` four-thread stager | `928b0e0` `Stager` | replaced at merge 1; our NVMe `Job{l,e}` re-added |
| `c7a469e` copy-issuer thread (opt-in on O) | `cf68b00` D-5 (default on) | ours kept, **default flipped to on** in merge 2 (`prefill.cpp:2250-2257`) |
| `14c5975` port of `fe609ce`/F-1/F-2 (#42). MEASURED: 8K −2.4 % | the same upstream commits | our port removed; upstream's own A/B env vars remain (`STRATA_GROUP_COPY`, `STRATA_GR_UNFUSED`, present at `4c68013`) |
| `6790f01` K4 `pcie_share` skip | `1e4515c` `set_pcie_off` | **upstream rejected it** (`9a19f57`); ours stays at `verify.cpp:704` (§4.2, suspicious) |
| `704ac58` `--pcie-frac` default 0 | `7a4b627` / `4befda7` / `d9a4974` probe | CUDA0 probe dropped by us; the layer-split stage probe kept (`generate.cpp:2318`) |
| `ba0cade` `--pool-rest` | `c47e290` + `06438fd` | layered on upstream's protocol (`rest_` cleared in `publish()`, `pool.cpp:242`) |

**D2: partial overlap, or the same user problem solved differently.**

- **PLE read-ahead.** Upstream reads ahead for prompt chunks after the first. Ours adds decode windows (`7134ce9`)
  and the first prompt chunk (`4935d9b`).
- **NVMe capacity tier** (`--ram-cache-gib`, `bf0baf2`). Upstream's `de159b5` `--mmap-experts`, open PR #80
  `--tiered-experts` and #129 address the same problem. The maintainer on #80: "rebase them as a budgeted pinned tier on
  `FileExpertSource` instead of a second file-backed source". Ours lives in `ArenaExpertSource`, against that preference.
- **Second-GPU tier.** Ours (`secondary_*`) and upstream's `a4a5a37` coexist in one tree (`generate.cpp:504-519`,
  `:1511-1516`). Upstream's keeps the host copy. Ours is exclusive and Q2_0-only.
- **Split + wave vs upstream's layer split.** The architectures differ. MEASURED (#47 body): D2x 8K 4.1 s vs U3 serve
  7.3 s, a cross-session, generate-vs-serve package comparison confounded by chunk size (§0); U3 working set 35.3 vs
  D2 16.5 GiB. Maintainer on the similar PR #110: "let's first compare it with the layer
  split (`--gpus`) in an issue".
- **Claude Code serving.** Upstream has `261d2d8` (`?beta=true`) and `28e6c8e` (`cache_read_input_tokens`). None of
  our `count_tokens`, `signature_delta`, `stop_sequences`, billing strip, `LoopGuard` or `document_parts` exists at
  `4c68013` or `a790805` (grep).
- **Multi-slot conversation cache** (our PRD story 14, #7): upstream open #57 and PR #175.

**D3: conflicts, where the same path carries different assumptions.**

1. **Per-request PCIe share vs our K4 graph skip.** §4.2 rates it suspicious.
2. **`CUDA_MODULE_LOADING`.** Upstream forces EAGER, and still does at `a790805:src/program/generate.cpp:894-898`.
   We revert to LAZY and add a warm-up (`c868593`, `f4ce2bd`).
3. **The PCIe probe we dropped.** The #27 figure (3.24 vs 61.92 tok/s) was measured at `704ac58`, an ancestor of
   `f7ecc89`, when native packs used the DMA path. Upstream 0.1.14 made the copy kernel the default. Upstream's probe
   plus the copy kernel on our x4 link has **not been measured** on the merged engine (UNMEASURED).
4. **MMQ stream-k.** We set `nsm = 1` (`moe_mmq.cu:153-154`). E-9 (`c6c6594`) now routes the drafter's Q8_0 matrices
   through the same context, so nsm=1 affects the drafter's batched K/V as well (static reading).
5. **Decode numerics.** D's hashes (`60f0e5af`/`5b5884bc`) differ from O's (`050eee61`/`e7211023`). The checkpoint
   report §3.2/§4 attributes this to "for example the one-launch window combine". PR #109 claims bit-identical, so the
   attribution is **UNMEASURED**. No `STRATA_DEC_BATCH=0` A/B exists. Evidence against the combine attribution (#47
   dossier §4, m12): upstream with LAZY (U1L), which contains PR #109's combine, gives code hash `050eee61`, the same as
   O, and sizes 7,529 slots like O1 and D1; U1 (EAGER) gives `d6c283bc`. The dossier's own hypothesis is that cache
   size decides which experts the GPU computes, and so the rounding (UNMEASURED; a fixed `--expert-cache` tests it).
6. **E-9 `draft_kv` × our wave.** This is #48, fixed in `5c51574` (§4.3 B.0).
7. **Generate-mode exit hang.** It appears only after merge 2; see §8.
8. **Memory-ownership philosophy.** Upstream #141 maintainer: "the GPU holds a copy of the most-used ones, so the limit
   is your RAM". D-4, E-6 and upstream's adaptive swaps all assume a host copy exists.
9. **Two second-GPU tiers.** They share dispatch and setup. Combining them is untested (UNMEASURED).
10. **Not a conflict, an upstream rule our wave depends on:** `--no-prefill-borrow` caps chunks at 2048 even with
    `--prefill auto` (`generate.cpp:1743-1746`; `git blame` → upstream `928b0e0`, unchanged since v0.1.13). Our wave
    needs chunk ≥ 4096 (`generate.cpp:755`), so it is silently off under that flag. The same cap holds upstream's layer
    split to 2K chunks, because the split forces `o.no_prefill_borrow = true` (`:1722`, upstream `d733199`). MEASURED
    (checkpoint report §6.1): D2aw 11.17/11.23 s vs D2x 4.12/4.15 s at 8K.

**D4: upstream work that enables ours.**

- `928b0e0` is the substrate for the tail file, group-local scratch, group gather, nsm=1, split, wave and frontier.
- `a4a5a37` already puts a second-GPU tier with pinned staging and compact returns on main, so #47 PR 7 can target it
  instead of adding a third tier.
- Per-device plumbing from the layer split (`8729371` OnDevice, per-device shared-memory opt-ins) makes a second-card
  compute path expressible in upstream's terms (static reading).

**Open upstream PRs that overlap our series.**

| upstream PR | what | overlaps |
|---|---|---|
| #151 `--vram-experts` | maintainer: "rebase the store alone … Smaller PRs" | our PR 7 |
| #110 `--peer-device` | "compare with `--gpus` in an issue first" | our PR 7 and PR 8 |
| #153 | WDDM cache sizing | the sizing code of PR 2/4 |
| #80, #129 | tiered experts, shared arena backing | the NVMe tier |
| #148 | bitwise test of `native_mmvq` `multi_exact` | our `--mmvq-exact` |
| #154 (gputier) | correctness fixes, including `expert_pool_dispatch_multi` bounds in `expert_source.cpp` | will conflict next (+567 lines of ours in that file) |
| #149 (gputier) | a Q2_0 grouped kernel without 8-way bank conflicts and with `hx` computed once per token; the maintainer asked for it first | close to our K1/K3/K0 (static reading), but ours is in the CPU-order row kernel (`iq_kernels.cu:326+`) and gputier's in `s2_expert_grouped`; duplication risk |
| #175 | independent conversation caches | PRD story 14 (#7) |

### 2.4 What v0.1.27 (`a790805`) would touch

**Method.** `git merge-tree 4c68013 HEAD a790805`, legacy mode, stdout only.

**Files it changes (12).** `CMakeLists.txt`, `README.md`, `data/draft_vocab.bin`, `docs/DETAILS.md`,
`docs/MULTI_GPU.md`, `serve/server.py`, `serve/test_server.py`, `setup.py`, `src/core/device.cu`,
`src/kernels/cuda/fused_gr.cu`, `src/kernels/cuda/qsa_prompt_attn.cu`, and the new `tools/draft_vocab.py`.

**Textual conflicts: 2 hunks.**

1. **`setup.py`, `MIN_ENGINE` / `PY_PACKAGES`.** Take `(0,1,27)` and keep our `pypdf` and `pypdfium2`.
2. **`src/core/device.cu`, the runtime floor.** Take upstream's `cc_major*10+cc_minor < 75` (Turing port `aad5bb1`).
   The 4070 (sm_89) passes either way.

**Semantic checks needed after merging.**

- **`fused_gr.cu`.** Token slicing starts only below an 80 KB shared-memory opt-in. sm_89 and sm_120 report about
  99 KB (an NVIDIA spec figure, UNMEASURED here), so our bits should be unchanged (static reading). Re-run `21a2ed0`'s
  `gr_multi_parity`.
- **`serve/server.py`.** `77b70c0` (the image-marker rule) touches only the image-expansion loop. Run
  `serve/test_server.py` and our serve tests.

**Behaviour change: `data/draft_vocab.bin` grows from 40,525 to 106,299 ids** (`6e153c9`).

- VENDOR: CJK +15-38 % decode, English −1-2 %.
- Thai is not in `draft_vocab.py`'s SCRIPTS (han, kana, hangul, cjk_punct).
- Whether the draft subset holds Thai tokens at all is UNMEASURED. It is a hypothesis for thai decode (~50 tok/s)
  trailing code (~83-89) in D2x. `tools/draft_vocab.py --stats` **cannot** show it: it counts only the SCRIPTS keys
  (`a790805:tools/draft_vocab.py:24-30`, stats loop `:78-80`). A Thai count needs a U+0E00-U+0E7F range added to
  SCRIPTS, or a one-off count over the subset's ids.
- `319e4ef` makes setup replace an older shipped subset "at setup and at start" (a hand-made one is kept), which
  matters for pinning `draft_vocab.bin`.
- A daily-server merge must pin which `draft_vocab.bin` the rt dir uses. Otherwise the A/B compares vocab subsets, not
  engines.

**Hot files.** Nothing else in v0.1.27 touches our hot files: `generate.cpp`, `prefill.cpp`, `expert_source.cpp`,
`verify.cpp`, `secondary_*` and `pinned.cu`.

**Other v0.1.27 commits that bear on us.** `da77db6` (HIP-only toolchains build without CUDA headers; conflicts in
spirit with our unconditional `CUDA::cudart` and the CUDA headers in `secondary_*.cpp`, §4.2); `319e4ef` (setup
replaces an older shipped `draft_vocab` subset at setup and at start); `aaaafc6` (setup recompiles the engine for a
card it has no code for, at start); `aad5bb1` (sm_75 runtime floor).

---

## 3. Our changes: the feature catalogue

Features are numbered **P1-P18** (before merge 1, `6da1f66..f7ecc89`) and **Q1-Q22** (after merge 1,
`2a9b26c..5c51574`). Each entry gives the mechanism, commits, files at `5c51574`, flags and defaults, the measured
effect with its source, dependencies, and state.

### 3.1 Before merge 1

**P0. Upstream serving fixes we cherry-picked.**

- **Commits:** `8f2187d`, `c996648`, `dd05359`, `9aedbd2` (reverted by `eab0066`), plus our `8ee13c1` (tests) and
  `cc9d88f`.
- **Files:**
  - Checkpoints: `generate.cpp:1195`, `checkpoint_save` `:1232`.
  - Pool sleep: `pool.hpp:229-233`, `pool.cpp:232-247`, `:285-289`.
  - Tests: `tests/xeno/test_serve_{serialization,protocol}.py`, `pool_idle_sleep.cpp`.
- **Measured** (`docs/reports/2026-09-27-serving-upstream.md`):
  - A live run returned `REUSED 55`.
  - `pool_idle_sleep`: 312 → 31 ms of process CPU over 150 ms idle.
  - `--short-read 0` vs `64` changed output token 5, hence the revert.
- **State.** `--short-read` is back at HEAD with default 64 (`generate.cpp:379`), through merge 1. Whether it still
  diverges is **UNVERIFIED**.
- **Upstream:** already there.

**P1. AVX-VNNI path for native Q2_0 CPU expert rows.**

- **Mechanism.** `q2_rows_any` dispatches AVX-512 → AVX-VNNI → AVX2. AVX-VNNI is detected by CPUID.(7,1):EAX[4] plus
  the OS XMM/YMM state. The VNNI arm uses VEX `_mm256_dpbusd_avx_epi32`, which sums the same four u8×s8 products per
  dword, so it is bit-exact with AVX2 `maddubs+madd` (`f679806`).
- **Commits:** `f679806`; `10078cb` adds the test `tests/xeno/q2_isa_parity.cpp` and CTest gating.
- **Files:** `src/kernels/cpu/q2_avx2.cpp:38-66` (`row_multi<NT,Vnni>`, VNNI arm `:50-56`) and `:104`;
  `src/kernels/cpu/expert_layout.cpp:54-78` (`cpu_avxvnni_ok`, `q2_rows_any`).
- **Flags.** On when the CPU has it. `STRATA_FORCE_AVX2=1` disables it (and AVX-512) (`expert_layout.cpp:24`, `:56`).
- **Measured.**
  - `f679806`: "Rough on i5-13500 + RTX 5060 Ti (x4), codex using ~3.8 cores: decode 49/35/33/38 tok/s vs
    AVX2-forced 31/23/33/40". These were separate runs, not ABBA, under a concurrent CPU load, and 2 of the 4 prompts
    show no gain (33 vs 33) or a loss (38 vs 40).
  - #47 comment: pool bandwidth O1 36.3 vs U1 33.2 GB/s, with "different packs and expert counts".
  - The isolated kernel A/B is **UNMEASURED**.
- **Risk (static reading).**
  - `cpu_avxvnni_ok` returns `false` on non-MSVC builds (`expert_layout.cpp:66-68`).
  - `q2_avx2.cpp` is compiled with `-mavx2;-mfma;-mf16c` (`CMakeLists.txt:622`) and uses `_mm256_dpbusd_avx_epi32`
    without a `target` attribute, so it likely fails under GCC/Clang. **UNVERIFIED**: no Linux build was done.
  - Upstream `a790805` has no `_mm256_dpbusd_avx` or `avxvnni` (grep).

**P2. CPU pool core ordering by EfficiencyClass (Windows).**

- **Commits:** `ebcda26`, `10078cb`, `0ffea2b`.
- **Files:** `pool.cpp:40-68` (the sort block `:63-68`, the `stable_sort` call `:65-66`). The Linux `#else` branch
  `:71-105` is upstream's.
- **Flags:** always on under Windows.
- **Measured.**
  - `docs/reports/2026-09-27-core-policy.md`, rough separate boots: auto (13) 39.31 … 13 → 41.43 tok/s; "does not
    establish an optimal count".
  - `--pool-workers 5` (P-cores only) was worse than 6 (`2026-09-28-cpu-pool-sweeps.md`).
- **Upstream:** Windows-only, small (+24/−3). The gain is unproven.

**P3. Bit-exact native Q2_0 GPU expert path ("CPU order") and ADR 0001.**

- **Mechanism.**
  - For Q2_0 packs, GPU hits keep FP32 Q8 scales (`quantize_q8_1_rows_scaled`), CPU half-away rounding, the 8-lane
    accumulation order and one SwiGLU for both paths.
  - The CPU SwiGLU becomes `float(exp(double))`, which changes CPU-only output. ADR 0001 accepts that baseline.
  - Adds the `--cache-cpu-only` diagnostic.
- **Upstream's position** (`c1e9033` `generate.cpp:1653-1662`): the old warning was stale; the difference is rounding,
  95-98 % same top-1, and perplexity is equal.
- **Commits:** `9a2b807`, `dadb3a9`.
- **Files:** `verify.cpp:654`, `:694`; `iq_kernels.cu:905-970`; `pool.cpp:419` (double exp); `generate.cpp`
  `cache_cpu_only` `:262`, `:1550`, `:5867-5870`; tests `tests/xeno/native_q2_pool_hit_parity.cpp`, `cache_tokens.py`;
  `docs/adr/0001-q2-exp-baseline.md`.
- **Flags.** On for any all-Q2_0 native pack (`gu_type == 42 && d_type == 42`). There is no opt-out.
- **Measured** (`docs/reports/2026-09-27-cache-investigation.md`):
  - 4 real experts bit-exact (0/7,680).
  - `final256-{sky,thai,code,long}`: 256/256 IDs match between CPU-only and GPU-hit.
  - `9a2b807`: "66 multi-token/two-expert cases passed bit-exactly".
- **Hypothesis (#47, UNMEASURED):** with this, output no longer depends on the cache size.
- **Depends on:** nothing. P4, P7-P11 rely on it.

**P4. Q2_0 CPU-order kernel speed-ups K0-K3.**

- **Mechanism.**
  - K0: four 8-lane rows per warp.
  - K1: the `hx` zero-point computed once per entry.
  - K2: an exact `__dp4a` dot.
  - K3: a weight block loaded once for up to `Q2_MULTI`=4 entries.
- **Commits:** `ab93dfd`, `40d482b`, `cad5a92`, `6dd2867`; docs `510d06a`, `ce6dd2f`, `d5e0815`.
- **Files:** `iq_kernels.cu:322` (`q2_spread`), `:327-361`, `:425`, `:437`, `:473`, `:543`, `:897-903`.
- **Measured** (`docs/reports/2026-09-28-q2-rows-per-warp.md`, ABBA, dual config):
  - K0: code 45.0 → 61.1 tok/s, long 42.0 → 58.7. `wait for rings` 32.5 → 21.9 ms/round. 0/7,680 differ; a mutant
    fails 7,680/7,680.
  - K1 and K2: about −0.35-0.4 ms/round each, "Neither is claimed as a throughput gain".
  - K3: parity only; its speed is UNMEASURED.
- **Depends on:** P3.

**P5. Verify-window trims: K4 empty PCIe-share skip, and the mapped hit merge.**

- **Mechanism.**
  - K4: `set_pcie_share(false)` drops the `m_flagB_` wait and the second grouped launch when `pcie_num == 0` (`6790f01`).
  - `moe_hit_merge_mapped`: reads only the non-hit rows of the mapped CPU output. It is bitwise equal to
    `copy_from_mapped + moe_hit_add` (`94d347d`).
- **Commits:** `6790f01`, `94d347d`, `fea67c5`.
- **Files:** `verify.hpp:139`; `verify.cpp:704-706`, `:726`; `s2_expert_grouped.cu:764-769`;
  `tests/xeno/hit_merge_mapped.cpp`.
- **Measured.**
  - K4: "outputs identical to K3 (12/12 runs)"; speed UNMEASURED.
  - Merge (`docs/reports/2026-09-28-h2-primary-timeline.md`, ABBA): Thai 43.43 → 44.91 tok/s (+3.4 %); code
    69.20 → 69.87 with one run disturbed.
- **State.** K4 is the hunk upstream rejected (§4.2).

**P6. CPU pool scheduling.**

- **Mechanism.**
  - `--pool-priority` (`SetThreadPriority`).
  - `--process-priority` (`SetPriorityClass`).
  - `--pool-rest` (`ExpertPool::rest()` sleeps parked workers after each window).
  - `--lock-cpu-experts` (VirtualLock).
- **Commits:** `af4e19e`, `021eb5b`, `ce6dd2f`, `e0e505c`, `0a2d70d`, `ba0cade`, `3780ec5`.
- **Files:** `pool.cpp:126-136`, `:221`; `pool.hpp:88-91`, `:176`, `:227`; `generate.cpp:2440`, `:2443`, `:6592`,
  `:3264-3290`; `src/platform/memory.cpp:54`, `:116`; `tests/xeno/pool_rest.cpp`.
- **Defaults.** `--pool-priority 2` (HIGHEST), `--process-priority 0`, `--pool-rest 1`, `--lock-cpu-experts` off.
- **Measured.**
  - Priority (`cpu-pool-sweeps.md` sweep 2): code 57.0 → 66.8, Thai 26.9 → 36.7.
  - Process HIGH (`0a2d70d`): code 76.76 → 78.73.
  - Pool rest (`ba0cade`, same-session A/B): **rest alone**, at 6 workers, code 80.52 → 83.28 (+3.4 %), thai 50.55 →
    50.92 (+0.7 %), both below the 13.6 % gate. "13 workers + rest: code 89.22 (+10.8 %)" is against 6 workers with rest
    off, so it mostly measures the worker count; no run has 13 workers with rest off. Without rest the join wait rose
    0.9 → 19 ms/round.
  - Lock: no gain (`cpu-pool-sweeps.md`).
- **Finding (verified by grep).** `pool.rest()` is called only at `generate.cpp:6592`, the generate-mode loop. **Under
  `--serve` `--pool-rest` has no effect**, and neither does `arena_src.decay_scores()` (`:6593`). This was already true
  at `7134ce9` and `2a9b26c`, so it is not a merge loss.

**P7. The 4070 SUPER static Q2_0 expert tier and `SecondaryRunner`.**

- **Mechanism.**
  - After the primary fill, the next-ranked non-primary pairs are staged into a `SecondaryArena` on CUDA device 1.
    It is allocated once, touched, and checked against the lower of the CUDA and PCI-matched NVML free readings, with
    up to 8 retries.
  - Dispatch marks these pairs kind 2.
  - `SecondaryRunner::launch()` runs before the CPU pool. It does a packed pinned metadata H2D and an activation H2D,
    `quantize_q8_1_rows_scaled`, `native_expert_grouped` (the P3 path), then a compact D2H.
  - `finish()` scatters the rows.
  - A 100 ms monitor thread calls `_Exit(3)` on a free-VRAM breach.
  - The launch is a CUDA graph (`3ec52cb`). An async launcher is opt-in.
- **Commits:** `46c8616`, `5532ce6`, `9449d93`, `71be182`, `b6e909a`, `4e16323`, `4514316`, `08ca92b`, `abac914`,
  `de5537f`, `2657939`, `97b9443`, `d1431ea`, `acfaa3b`, `3ec52cb`, `aec0da7`, `3780ec5`, `2bbf77d`, `9f81cd0`.
- **Files:** `include/strata/core/secondary_{budget,vram,arena,profile,runner}.hpp`; `src/core/secondary_arena.cpp`
  (open `:55-140`); `secondary_runner.cpp` (launch `:161`, monitor `:459-525`, `_Exit(3)` `:486`, `:511`);
  `expert_source.cpp:917-993`; `generate.cpp:2974-3160`, `:1920-1930`; `device_main.cpp`; CMake `strata_secondary`,
  `strata_secondary_compute`.
- **Flags.** `--secondary-expert-mib` 0 (off; max 12288). `--secondary-free-floor-mib` 2560 (min 256).
  `--secondary-graph` 1. Needs native Q2_0, spec ≥ 2, the pool and `--pcie-frac 0` (verified at
  `generate.cpp:1920-1930`), plus `CUDA_VISIBLE_DEVICES=1,0`.
- **Measured.**
  - First run (phase3-compute.md): sky 20.47 → 25.16 tok/s.
  - CUDA graph (`3ec52cb`): launch 3.2 → 1.6 ms/round.
  - Packed metadata: host enqueue 7.356 → 4.558 ms/round, "not an end-to-end speedup".
  - Compact D2H: 1,336.91 → 345.87 MiB.
  - Async launcher: abandoned (`finish()` waited 4-5.8 ms).
- **Machine-bound (verified at `secondary_arena.cpp:59-77`).** It requires ordinal 1, `"5060 Ti"` on device 0,
  `"4070 SUPER"` on device 1 and sm 8.9. The error reads "requires CUDA_VISIBLE_DEVICES=1,0 (5060 Ti then 4070 SUPER
  sm_89)".

**P8. Exclusive primary ownership.**

- **Mechanism.**
  - After the primary fill, every slot is read back and byte-verified. Its ownership is published, and
    `PinnedArena::decommit_interior` frees the whole pages inside the blob.
  - `blob()` refuses a GPU-owned pair.
  - `fdaddbe` makes it the default and keeps host copies for the lendable tail. That part was later replaced by the Q10
    tail file.
- **Commits:** `28a610b`, `92f10c2`, `fdaddbe`.
- **Files:** `pinned.hpp:58-60`; `pinned.cu` decommit (Windows-only); `expert_source.cpp:1759`, `:1780`, `:1786`,
  `:1825`; `generate.cpp:1931-1941` (verified), `:2796`; `tests/xeno/exclusive_host_pages.cpp`.
- **Flags.** `exclusive_mode = -1` means on when eligible. Eligible = native Q2_0, spec ≥ 2, no mmap, no
  cache-cpu-only, pool on, `pcie_frac == 0`, a profile and an expert cache (verified).
- **Measured.**
  - `fdaddbe`: decode 64.68 vs 61.41; TTFT 27.54 vs 27.60 s; private 46.73 → 38.87 GiB (−7.86 GiB).
  - #47 package: one card, working set 23.2 vs 33.2 GiB.
- **Risk.** `pinned.cu:292` and `:322` return "exclusive host page release/re-commit is Windows-only in this slice"
  (verified). Combined with the default-on flag, a Linux boot with a Q2_0 pack would fail (static reading).

**P9. Adaptive swaps: paired primary swaps, 4070 swaps, threading, defaults.**

- **Mechanism.** Paired swaps are three non-blocking stages on the adapt thread, between windows:
  1. D2H the victim.
  2. Recommit it, copy it home, publish it CPU-owned; stage the newcomer and H2D it.
  3. Mark the newcomer resident and release its host pages.
  - Candidates are CPU-served experts only.
  - `--adapt-secondary` does H2D-only swaps into the 4070.
  - `parallel_copy` uses 4 threads.
- **Commits:** `f009130`, `a9e5c0e`, `292e3b8`, `4ef7967`, `87e9922`, `86b6419`, `5748dc5`, `e0cc53f`.
- **Files:** `generate.cpp:118-132`, `:1963-1970`, the generate block `:6138-6300`, and a serve duplicate
  `:4497-4700`.
- **Defaults** (verified): `adapt_swaps`/`adapt_every` → 8/1 under exclusive primary, else 96/4. `adapt_secondary` →
  8 when the tier is on, 0 under serve with an exclusive 4070.
- **Measured.**
  - `a9e5c0e`: code 63.5 → 68.4, Thai 40.1 → 42.3.
  - `86b6419`: code 73.91 → 75.37.
  - `5748dc5`: Thai 46.31 → 47.34 (this is the 4070 swaps' thread, secondary-tier code).
  - **Misattributed in #47:** #47's PR 3 row ("equal decode to copy-kept (84.75 vs 85.76)") quotes `0a0791d`, which
    compares the **4070 tier** exclusive against copy-kept, not primary paired swaps. The primary paired-swap evidence is
    `a9e5c0e` ("matching the non-exclusive swap that costs +8.5 GB RAM") and `86b6419`. The #47 row should be corrected.
- **Abandoned:** `--adapt-gate`; batches larger than 8.

**P10. Exclusive 4070 ownership, plus a peer copy in prefill.**

- **Commits:** `5213b90`, `ea04a91`, `0a0791d`, `ca80852`.
- **Files:** `expert_source.hpp:491`; `prefill.hpp:76`; `prefill.cpp:729`, `:1456`; `generate.cpp:1945-1947`,
  `:3077-3117`, `:4229`, `:5778-5788`.
- **Flags.** Default on when `--secondary-expert-mib > 0`, except `--serve` with `--adapt-secondary > 0` (verified,
  `generate.cpp:1945-1947`).
- **Measured.**
  - `0a0791d`: code 84.75 copy-kept vs 85.76 exclusive; peak in-use RAM 39.8 → 31.3 GB.
  - `853c6f2`: 35 % of the x4 prefill stream was peer copies.

**P11. Placement-first cold start.**

- **Mechanism.**
  - With any exclusive tier, the arena is `reserve_only`.
  - The GPU tiers fill straight from the pack through upstream's `DirectFile`, in pinned batches of 32 with a reader
    thread.
  - `load_rest` then commits and reads only the host-owned experts.
- **Commits:** `1d35b34`, `2bbf77d`, `d4de5a6`, `6aee9b9`, `9f81cd0`, `77bbd38`, `81564fa`.
- **Files:** `pinned.hpp:46-50`; `pinned.cu:225-231`; `expert_source.cpp:1262`, `:1359`, `:1431`, `:1438`, `:1564`
  (`commit_interior` `:1575`); `generate.cpp:1948`, `:2811-2960`, `:3030-3120`.
- **Measured.**
  - `1d35b34` (the placement-first-specific evidence): boot peak "46.3 GB peak during boot -> 29.6 steady" before,
    "30.5 GB: the peak is the steady state" after.
  - `d4de5a6`: peak 38.9 vs 46.6 GB; decode 65.64 vs 61.76. The control arm is `--no-exclusive-primary-experts`, which
    also turns off exclusive ownership (P8), so this conflates P8 and P11.
  - Boot to decode: about 161 s → 62 s (`2bbf77d`) → 48 (`6aee9b9`) → 42 (`9f81cd0`) → 18 s (`77bbd38`). The 161 s
    start point is the `1d35b34` build itself, whose 4070 fill took 104 s by sampling NVML per slot (`2bbf77d`);
    `2bbf77d` and `9f81cd0` are 4070-tier fixes. Primary-only fill times: 29.3 → 13.4 s (`6aee9b9`) → 5.3 s
    (`77bbd38`, from 11.2). No run compares boot time with and without placement-first (UNMEASURED). Output identical
    by md5 (`6aee9b9`).

**P12. NVMe capacity tier (opt-in).**

- **Commits:** `c2aabee` (N0 simulation), `bf0baf2`, `ed1972c`.
- **Files:** `expert_source.hpp:107-120`, `:436-455`; `expert_source.cpp:1595-1740`; `tests/xeno/nvme_capacity.cpp`.
- **Flags:** `--ram-cache-gib` 0.
- **Measured** (`bf0baf2`): 8 GiB vs unlimited, commit 30.3 vs 36.9 GiB, decode 49.57 vs 84.71 tok/s. That is why it
  is opt-in.

**P13. `--pcie-frac` default 0** (`704ac58`).

- **Files:** `generate.cpp:1919` (verified). The comment at `:1909-1910` says upstream's probe would pick 0.15 on this
  link.
- **Measured:** 8,024-token prompt, 3.24 tok/s with the old default vs 61.92 with 0 (#27).
- **Not for upstream** (§7).

**P14. Prompt path before merge 1.**

- **`853c6f2`, per-tier prefill source counters.** Alive at `generate.cpp:5860-5865`. MEASURED: 2K dual prompt, pageable
  9,729 (13.45 GB) plus peer 5,223 (7.22 GB).
- **`6ee0274`, `iq_dequant_expert_f16`**: gate/up and down dequantized in one launch. Alive at `iq_kernels.cu:872`,
  called at `prefill.cpp:2900`. MEASURED (before merge 1, single-GPU 2K serve prompt): 6733/6642/6649 →
  6598/6450/6431 ms; 0 of 4,915,200 values differ. Not present at `a790805` (grep). At HEAD the call sits on the
  non-MMQ FP16 branch only (`:2873` returns first when `use_mmq`), so it runs only with `STRATA_PREFILL_MMQ=0` or on
  layers without MMQ.
- **Superseded:** `767fcfc`, `f7ecc89` (PLE async) and the `fdaddbe` stager (see §2.3).
- **`d6a93df`, `--profile-prefill-range`.** Alive.
- **`a429af7`, cross-arch GEMM probe.** MEASURED: cuBLAS 5/72 bit-identical, mma 0/72, FP32 FMA 72/72. This is why the
  tensor-core cross-card prefill became opt-in (`63700d9`).

**P15. Claude Code serving (ours).**

- **What it adds:**
  - `POST /v1/messages/count_tokens` (`e6fd73e`, `5a890f7`, `1920133`).
  - The billing-header strip (`bc1b18f`, `9ccc276`).
  - `LoopGuard` (`6bff9c1`, `14c5708`, `0d6536f`).
  - `signature_delta` (`6bff9c1`).
  - PDF `document` blocks (`56881b8`).
  - `/health` 503 `engine_exited` (`e9c6958`).
  - `stop_sequences` (`d399916`).
  - Not in HEAD, on the local `xeno/claude-merge-0.1.20` only (§1): `694ad82` (#46, PDFs and screenshots inside a
    `tool_result` reach the model) and the #49 series `12556b1`, `e63d359`, `55eb129`, `c54c337`, `80cd0b3`,
    `1620329`.
- **Files:** `serve/frontend.py:184`, `:192-193`; `serve/pdf_blocks.py`; `serve/loop_guard.py`; `serve/server.py:554`,
  `:838-899`, `:1237`, `:1311`, `:1408-1414`, `:1468`, `:780`; `setup.py`; tests
  `tests/xeno/test_{count_tokens,billing_header,loop_guard,anthropic_stream,document_blocks}.py`.
- **Measured.** Tests only, red then green (`docs/reports/2026-09-27-claude-code-compat.md`). The latency benefit of
  the billing strip is UNMEASURED.
- **Finding (verified).** The `count_tokens` route compares `self.path.rstrip("/")` including any query
  (`server.py:1468`), while other routes strip `?…` first (`:1458`). A `count_tokens?beta=true` POST would miss the
  route. Whether Claude Code sends that query is UNVERIFIED.

**P16. Stats and diagnostics.** `tier hits` (`ad2d10e`), `secondary timing` (`c54c948`), `--secondary-profile-timing`
(`234ab7c`), `dispatch detail` (`0960655`), `--profile-decode-range` (`dbf6f45`), `--route-trace` (`3780ec5`),
`verify edges` (`3ec52cb`), `adapt detail` / `join wait` (`86b6419`), and `--mmvq-exact` (`e0cc53f`: "no faster"
with 0).

**P17. Tests and harnesses.**

- `tests/xeno/*` is gated by `STRATA_BUILD_XENO_TESTS` (`CMakeLists.txt:53`, `:166`, `:683-733`).
- `tests/xeno/perf/*`: `ab.py`, `launch_ab.py`, `route_tools.py`, `adapt_sim.py`, `blend_profile.py`, `mem_trace.py`,
  `nvme_sim.py`, and more.

**P18. Docs and process.** `AGENTS.md`, `CLAUDE.md`, `docs/OPEN-WORK-LEDGER.md`, the ADR, and the
`docs/reports/2026-09-2{7,8,9}-*` reports.

### 3.2 After merge 1

**Q1. Whole-pipeline timeline (`STRATA_TIMELINE`, #33) and the gpu0 decode lane (#44).**

- **Mechanism.**
  - Host spans from every thread and GPU lanes, anchored on one clock (on WDDM the narrowest of 8 anchors is kept),
    written as Chrome-trace JSON.
  - The server writes `<file>.server.json`.
  - `tests/xeno/perf/timeline.py` turns a trace into a budget.
- **Commits:** `1702f09`, `ed7e6ad`, `5e42b14`, `c9cd626`, `9c800c5` (instants), `915dba7`, `5425bbb`, `3d7bd1f`,
  `579efee`, `8bb3a00`.
- **Files:** `include/strata/timeline.hpp`, `timeline_gpu.hpp`, `src/platform/timeline.cpp`, `serve/timeline.py`,
  `tests/xeno/perf/{timeline,decode_paths,decode_gpu_layers,prefetch_sim}.py`.
- **Flags:** `STRATA_TIMELINE=<file>`, off by default.
- **Measured.**
  - About 6 % of an 8K prefill when on (`1702f09`).
  - When off: 8K 12.07 vs 12.27 s, code 84.42 vs 84.26 tok/s (`5e42b14`).
  - `c9cd626` retracted an earlier "copies stall 35.7 ms" as an instrument artefact.
- **Size:** +1,007/−88.

**Q2. MMQ without stream-k (nsm = 1), #5.**

- **Mechanism** (verified at `moe_mmq.cu:147-154`). The comment reads: stream-k "sizes its grid from the SM count (36
  on the 5060 Ti, 56 on the 4070 SUPER) and splits a tile's K range across blocks, summed in float". With `nsm = 1`,
  each tile gets its own block over the whole K range.
- **Commits:** `ff2bdae`, `a87644a`, `4a5c9ce`, `31943cf`, `48a0f67`, plus the `c7a469e` probe.
- **Files:** `src/prefill/moe_mmq.cu:153-154`; `tests/xeno/mmq_cross_arch.cu` (270 lines), `moe_layer_cross_arch.cu`
  (227), `gemm_cross_arch.cu`; `docs/reports/2026-09-29-mmq-cross-arch.txt`.
- **Flags:** default nsm=1. `STRATA_MMQ_STREAM_K` set to **any** value (even 0) restores stream-k: the code checks only
  `getenv(...) == nullptr` (`moe_mmq.cu:153`).
- **Measured.**
  - `ff2bdae`: 8 of 8 identical vs 1 of 8 with stream-k (0 of 6.4M values differ). 8K gate/up 713 vs 715 ms on the
    5060 Ti.
  - `31943cf`: one full MoE layer, 0 of 13,107,200 expert rows and 0 of 1,310,720 outputs differ. With stream-k,
    7,317,359 values differ.
- **Limit.** The trunk's floating GEMMs are not made exact. The 2-chunk vs 4-chunk 8K reads differ (#45 m11); that
  mechanism is UNMEASURED.
- **Upstream:** `a790805` has no nsm override (grep). Size: +8 lines in `moe_mmq.cu`, +3 in CMake.

**Q3. LAZY CUDA module loading plus cuBLAS/ggml warm-up (#30).**

- **Commits:** `f4ce2bd`, `c868593`.
- **Files:** `generate.cpp:1413-1418`, `:1871-1875`, `:2757-2762`; `gemm.cu:285` `warm_cublas`; `moe_mmq.cu:91`
  `mmq::warm()`.
- **Measured.**
  - `f4ce2bd` (committed 10:10, **before** `c868593` at 10:29, so the build still forced EAGER): TTFT 3856 → 1243 ms.
    This is the warm-up hiding EAGER's 2.76 s first `cublasCreate` ("LAZY: 0.16 s", same message). It is **not** a
    measurement of LAZY + warm-up.
  - `c868593`: private 42.20 (EAGER) vs 39.11 GiB (LAZY); about 0.2 GB more VRAM free per card.
  - #47 dossier m12, on upstream with LAZY only and no warm-up: working set −1.1 GiB, commit −1.6 GiB, +26 slots, but
    the **first 8K prompt was +7 % slower (5.40 → 5.80 s)**. #47: "PR 1 must be LAZY + that warm-up, measured again".
  - **UNMEASURED:** the TTFT and RAM effect of LAZY + warm-up on upstream. Upstream `a790805` has no warm-up at all
    (`git grep warm_cublas` finds nothing), so the warm-up is also a possible separate, smaller candidate: measure U vs
    U+warm-up and U vs U+LAZY+warm-up separately.
- **Upstream still forces EAGER** at `a790805` (`generate.cpp:894-898`, verified).

**Q4. Copy-issuer thread** (`c7a469e`). Merge 2 made it the default (`prefill.cpp:2245-2257`). MEASURED: below the
noise gate on our link (`c7a469e`). This is upstream's D-5, so it is **not a PR**.

**Q5. Batched chunk embedding** (`4935d9b`, `cb23774`). Replaced by upstream C-4. Not a PR.

**Q6. First-chunk PLE read-ahead in the prompt path** (`4935d9b`, `cb23774`).

- **Files:** `prefill.cpp:1928-1961`, `:2315`.
- **Flags:** default on; `STRATA_PREFILL_PLE_AHEAD=0` reverts.
- **Measured.**
  - `4935d9b`: 11.307/11.308/11.307 vs 11.689/11.656/11.805 s (−3.7 %), identical output `8c6c97e1`.
  - Q5 + Q6 + Q8 together (`cb23774`, paired against `c9cd626`: the batched embedding, the PLE read-ahead and the
    group gather): 10.84 vs 11.64 s (−6.9 %). Q5 is now superseded by upstream C-4.
  - **UNMEASURED on the merged engine.**
- **Why it is new to upstream (static reading).** Upstream calls `ple_gather` inline for the first chunk
  (`4c68013` `prefill.cpp:1045-1060`).

**Q7. The "wait host" phase and a gather probe** (`76f15be`). The stager-depth knob was removed in `cb23774`: ring and
buffer sweeps did not move the prompt. Diagnostic only.

**Q8. One gather launch per MMQ group on the streamed walk** (`63e0186`, `cb23774`).

- **Files:** `moe_mmq.cu:243` `gather_native_group`; `moe_mmq.hpp:71-75`; `prefill.cpp:89-90`, `:2258-2262`,
  `:2958-2975`, `:1071`; `tests/xeno/gather_group_parity.cu`.
- **Flags:** default on; `STRATA_PREFILL_GROUP_GATHER=0` reverts.
- **Measured.**
  - `63e0186`: 10.72/10.77/10.77 vs 11.22/11.13/11.21 s (−4.0 %), identical output.
  - Host "expert launches" 185 ms, from 1.2-3.4 s.
  - Its effect on the merged engine is UNMEASURED.
- **Upstream:** `a790805` has no `gather_native_group` (grep).

**Q9. Group-local MMQ scratch and activations quantized once per token** (#34 S3a/S3b, #32 S1).

- **Commits:** `c8f12fb`, `f801c5d`.
- **Files:** `prefill.cpp:1190` `mmq_rows_cap`, `:701`, `:1419`; `moe_mmq.cu:112`, `:127` `gather_q8_rows`;
  `tests/xeno/q8_row_gather.cu`, `combine_cross_arch.cu`.
- **Measured** (`f801c5d`): prefill 10.53 vs 10.86 s (−3.0 %); borrowed VRAM 2,951 vs 3,660 MiB at 8192; commit
  41.16 vs 41.86 GiB.
- **Overlap.** Upstream `4c68013` has 0-based per-group MMQ bounds but no `gather_q8_rows` or `mmq_rows_cap` (grep).
  The PR must be measured as U vs U+PR.

**Q10. Tail file for the lendable cache slots** (#34).

- **Commits:** `497777a`, `48e44be`.
- **Files:** `generate.cpp:307-312`, `:630` `TailFile`, `:770` `setup_tail_file`, `:817` (2 GiB disk guard), `:872`
  `refill_lent`, `:1959-1961` (verified: `tail_from_pack = o.tail_file && o.exclusive_primary_experts &&
  !o.mmap_experts`), `:3238`, `:5207`, `:5834`.
- **Flags:** default on; `--no-tail-file`.
- **Measured** (`497777a`): commit 41.16 → 37.90 GiB; refill 0.63-0.65 s at 4.8-5.1 GB/s, against 1.13-1.38 GB/s
  reading the GGUF.
- **Cost:** about 3.5 GB of disk.

**Q11. Dual-GPU prompt path "split"** (#32, #35; S4a/S4b, D1-D6).

- **Mechanism** (this is the simulator's `whole_4070` policy). For every MoE layer of a chunk ≥ 2048 tokens:
  - The 5060 keeps the trunk, router and shared expert.
  - The q8 activations and gates go down over x4 to pinned host memory, then to the 4070.
  - The 4070 runs every routed expert (`mmq::expert_rows`, sub-products ≤ 4096 rows) and `moe_routed_sum`.
  - The routed sum goes back up over x4.
  - The 5060 finishes with `moe_shared_finish`.
  - D4 replaces cross-card stream waits with host waits on blocking-sync events.
  - The output is byte-identical to the one-card path.
- **Commits:** `c8f12fb`, `31943cf`, `7db2963`, `b9d2f57`, `3744f9a`, `3834af2`, `1bb403f`, `f0db7af`, `e10e27f`.
  Docs: `fb71a35` (`docs/reports/2026-09-29-dual-gpu-prefill.md`), `862be8f`
  (`docs/reports/2026-09-30-dual-gpu-daily-config-decision.md`).
- **Files:** `prefill.cpp:88-96`, `:388-470` (`SplitTier`), `:934`, `:1144`, `:1219-1225`, `:1634`, `:1984-2012`,
  `:3046-3048`; `kernels.cu:449`, `:707`, `:711`; `generate.cpp:1949-1955`.
- **Flags.** `STRATA_PREFILL_EXPERT_SPLIT=1`, opt-in. It needs an exclusive 4070 tier and a native pack with every layer
  on MMQ.
- **Measured:**

| step | result | source |
|---|---|---|
| S4a | 10.98/11.04 → 10.72/10.72 s | `7db2963` |
| S4b | 11.07/11.10 → 9.41/9.54 s (−14.5 %) at tier 6912 | `b9d2f57` |
| D1 | 9.37/8.97 → 8.56/8.23 s; 0 of 1,310,720 differ | `3744f9a` |
| D4 | `wait copies` 447-709 → 116 ms | `3834af2` |
| D6 | borrowed slots 2245 → 2137 | `1bb403f` |
| D5 | host grouping 351 → 179 ms; end-to-end unchanged | `f0db7af` |

- **Decode cost of the smaller tier (#36).** Tier 6400 vs 8704: code −5.0 % (84.70 → 80.48), thai −2.9 %.
- **Trap (verified at `generate.cpp:1945-1947`, `:1955-1958`).** `--adapt-secondary N>0` under `--serve` makes
  `exclusive_secondary` false. That silently disables split and wave; the env vars are ignored with no message.
- **Trap: a stale 8,000-pair profile caps the 4070 tier at 1,761 experts (2.27 GiB)** (checkpoint report §5). This
  is **not** the profile HEAD ships. `data/expert-profile.bin` is 130,328 B (blob `bf4aeed`, 8,000 pairs) at `6da1f66`
  and `f7ecc89`, and 196,632 B (blob `a4ed7e5`, all 24,576 experts, since upstream `e6265c7`/`4d4014c`, v0.1.14) at
  `c1e9033`, `4c68013` and `5c51574` (`git ls-tree -l`). The 8,000-pair file is the old blob still sitting in the
  `D:/Github/Strata` checkout, which is on branch `xeno/avxvnni` (`f679806`, before merge 1); #35's reproduce line
  and the `hang-*` runs (`C:/Strata-exp/hang-*.stderr`: "profile D:\Github\Strata\data\expert-profile.bin: 8000
  ranked pairs") point at it. 1,761 is not a fixed cap: it is 8,000 minus the 5060's 6,239 slots. D2x uses
  `D:/Github/Strata/data/ranked-exl3only-profile.bin` (untracked, 196,632 B, `git hash-object` `c7d37560`), which is
  a different file from HEAD's `a4ed7e5`.

**Q12. Two-lane prompt wavefront "wave"** (#35 D7), with the #48 fix.

- **Mechanism.**
  - Two `Prefill` lanes on one session read alternate half-size chunks.
  - Chunk c's layer l waits (`WaveLink::wait_attn`) until chunk c−1 has handed that layer's MoE off. The 5060's trunk
    of one chunk then overlaps the 4070's MoE of the other.
  - The lanes share one expert stream (ring 512).
  - `5c51574` gives each lane its own `on_chunk` bound to its own `Prefill`: `serve_chunk(sp)` / `serve_chunk(sp2)`,
    and `drafter_rows(prefill)` / `drafter_rows(prefill2)`.
- **Commits:** `cc81498`, `15794d6`, `9bd207f`, `e10e27f`, `e34a1d0`, `db17e31`, `5c51574`.
- **Files:** `prefill.cpp:455-470`, `:845-882`, `:1774`, `:2060-2168`; `prefill.hpp:95-100`; `generate.cpp:753-764`
  (verified: `wave_on` = `g_prefill_wave && prefill_chunk >= 4096`), `:1955-1958`, `:4214-4232`, `:4417-4454`, `:4787`,
  `:5226-5249`, `:5759-5814`.
- **Flags.** `STRATA_PREFILL_WAVE=1`. It takes effect only with the split, an exclusive tier and chunk ≥ 4096.
- **Measured:**

| run | result | source |
|---|---|---|
| 8K, tier 6400, ABBA | 7.86/7.73 vs 8.61/8.59 s (−9.4 %) | `cc81498` |
| 15K | 14.04 vs 15.68 s | `cc81498` |
| served 8K | 7.90 vs 8.81 s | `9bd207f` |
| merged, D2x | 8K 4.12/4.15 s; code 83.6/86.5, thai 50.8/52.2; working set 19.2 GiB | checkpoint report §6.1 / #45 m11 |
| #48 fix, split + wave vs split only | identical outputs `dfa16be2` (16,405), `025c15f8` (8,213), `a2ab06c2` (12,309); 16,405: 9.90-9.95 vs 12.13-12.16 s | #48 root-cause comment |
| #48 fix, served Claude Code, 47,201 tokens | 26.5 s vs 46.0 s with split + wave off | #48 root-cause comment |

- **Survey disagreement, resolved.** The 18.3 s figure is **D2a** (no borrow, no split + wave), not D2: #45 m10 table,
  "D2a | 18.30 / 18.28 s (438, 4 chunks)". The same m10 table gives D2aw 11.11/11.19 s; m11 gives D2aw 11.17/11.23 s
  (checkpoint report §6.1). The #47 PR 8 row "8K 11.1 → 4.1 s" pairs D2aw from **m10** with D2x from **m11**: two
  sessions.
- **MEASURED (m10 logs): D2aw ran split only, not split + wave.** `C:/Strata-exp/m10-8k-3-D2aw.stderr` has
  "strata prefill: chunk 2048, 739 MiB … (own)", "prefill 8023 tokens in 4 chunks, 11114.6 ms" and **no** "prompt wave"
  line; `m10-8k-4-D2bwR.stderr` in the same session has "prompt wave: two lanes of 4096 tokens". This matches the static
  reading (`wave_on` needs chunk ≥ 4096, `generate.cpp:755`) and contradicts #45 m10's "split + wave on 2K chunks".
- **Consequence.** "11.1 → 4.1 s" is D2aw (2K chunks, split only) against D2x (8K borrowed chunks, split + wave). It
  measures borrowing and chunk size plus the wave, across two sessions; it is **not** the PR 8 effect. The
  single-variable numbers for split and wave are `b9d2f57` (S4b, −14.5 %), `cc81498` (wave, −9.4 %) and #48's split +
  wave vs split only (above).
- **Open:** see §8.

**Q13. Static expert order and the Dm frontier** (#41).

- **Commits:** `f1f0e01`, `48a0f67`, `8efa74b`, `78f373e`, `de69e64`.
- **Files:** `prefill.cpp:103-170`, `:423-432`, `:574-580`, `:1045`; `frontier.hpp`/`frontier.cu`.
- **Flags:** `STRATA_EXPERT_ORDER`, `STRATA_DM_FRONTIER`, `STRATA_DM_FRONTIER_FRAC` (0.6).
- **Measured.** Parity 0 of 1,310,720. The 4070 gains +703 MiB free. Prefill 7.45 vs 7.76 s, called "speed neutral"
  (goal-run plan). Not run on the merged engine.

**Q14. PLE read-ahead in decode** (#44 D4, `7134ce9`).

- **Mechanism.** `PleTable::prefetch` issues a token's 16 rows as soon as the token is known: the accepted token after
  the commit, and each draft with p ≥ `--spec-min-p` (`mtp.on_draft`). `gather_batch` first waits for the prefetches.
- **Files:** `generate.cpp:1374-1392`, `:2334-2335`, `:5479-5480`, `:6629-6630`; `ngram.cpp:298`;
  `mtp.cpp:704`, `:721`; test `src/ngram/ple_reader_test.cpp --prefetch`.
- **Flags:** `--ple-ahead` 1 (ours). `--ple-delay-us` is upstream's flag (present at `6da1f66`); `7134ce9` only uses it
  for fault injection.
- **Measured** (`7134ce9`).
  - Gate: prefetched floats are memcmp-equal to a plain gather; outputs identical.
  - ABBAABBA: code 87.58 → 87.69 (+0.1 %), **thai 53.71 → 54.91 (+2.2 %)**.
  - #44 follow-up: code 88.20 → 88.71 (+0.6 %).
  - **Measured before C-4 existed; the merged engine is UNMEASURED.**
- **Upstream** has only a page-prefetch A/B (`--no-ple-prefetch`), not cross-window read-ahead.

**Q15. `STRATA_SECONDARY_POKE`** (`f945079`). Code −3.1 %, thai +4.3 %: mixed, so opt-in. Our hardware only.

**Q16. `STRATA_HOT_TO_SECONDARY`** (`28431aa`). No gain. Diagnostic.

**Q17. Link and copy probes** (`92c461f`, `6873bac`).

- The 5060 Ti x4 does H2D 6.77-6.82 GB/s under decode; the 4070 x16 does 23.4-24.3 (`92c461f`).
- Batch copies cut host issue time about 5×, with device time unchanged (`6873bac`).

**Q18. Prefill route trace and 2-GPU simulators** (`6598377`, `d0a899e`, `f1f0e01`, `f801c5d`).

- `d0a899e`: the expert_split lower bound is 4.46 s vs 12.25 s today.
- `b9d2f57`: `whole_4070` wins all 48 layers.

**Q19. Logging.** Per-part prompt log (`8408f84`, #37), per-request decode metrics (`4d64b5c`, #9), and
`STRATA_PREFILL_BUFFERS` (`ff6002b`, which corrected #38's premise).

**Q20. Harnesses.**

- `cross_boot.py` (`e8a797a`, `77b7a64`): a failed capture had printed "first divergent token 0"; it now reports NOT
  COMPARED.
- `topic_switch.py` (`9c800c5`), `phase8_sampler_check.py` (`1e09806`), `phase7_session_check.py` (`5eb3497`: a 100K
  follow-up reuses 100,001 tokens), `serve_wave_parts.py` (`db17e31`).

**Q21. The #42 ports** (`14c5975`). Removed by merge 2 in favour of upstream's copies. Not a PR.

**Q22. Exit-hang probe and quick exit** (`7b31aa6`).

- **Flags:** `STRATA_EXIT_TRACE=1` and `STRATA_EXP_QUICK_EXIT=1`. The latter calls `TerminateProcess` at
  `generate.cpp:6967`, verified, with no `#ifdef _WIN32`.
- **Message mismatch (verified).** `7b31aa6`'s message claims a `remote_experts.cpp` change, but the commit touches only
  `prefill.cpp` and `generate.cpp` (+38/−2). The `remote_experts.cpp:187` three-argument `native_expert_scratch_bytes`
  change is in merge `6c09c5b`.

### 3.3 Not kept: experiments and reverts

| item | commit | reason (source) |
|---|---|---|
| `--short-read` (upstream #10) | `9aedbd2` → `eab0066` | token 5 diverged, 0 vs 64 (serving-upstream.md). Back through upstream with default 64 |
| `--secondary-async-launch` | `3780ec5` | E-core helper; `finish()` waited 4-5.8 ms (cpu-pool-sweeps.md) |
| `--lock-cpu-experts` | `3780ec5` | no gain on top of priority |
| `--pool-workers 5` (P-cores only) | `ebcda26` | worse than 6 |
| `--adapt-gate` | `a9e5c0e` | "swapped too little" |
| paired batch > 8 | `292e3b8` / #21 | slows the CPU pool |
| one-wave `gr_down` geometry | `21a2ed0` | no change |
| exact-MMVQ rows-per-block | `e0cc53f` | slower, removed |
| `--mmvq-exact 0` | `e0cc53f` | not faster |
| ggml-cpu SIMD mode | `510d06a` | < 4 % difference |
| second CUDA context before the arena | phase3-static-staging.md | MTP weights then failed to load |
| pinned host arena in dual mode | `b6e909a` | blocks the second CUDA context |
| per-layer NVML query | `abac914` → `de5537f` | 13.39 vs 21.12 tok/s; moved to a monitor thread |
| tensor-core prefill across cards | `a429af7` | not bit-exact between sm_89 and sm_120 |
| `STRATA_PREFILL_STAGE_BUFS` | `76f15be` → `cb23774` | no difference, pinned-RAM risk |
| stager threads 4/8/12, ring 96-512, mapped router ids, stream-query flushes | `c7a469e`, `b9d2f57` | none moved 8K prefill |
| lanes with their own streams | `cc81498` | 9.2-9.6 s vs 8.0 s |
| 64-deep stager | `cc81498` | one-lane path 18 % slower |
| ring 512 on the split | `3834af2` | about 500 MB of 4070 VRAM for 0.05 s |
| lazy refill of the tail | `f801c5d` (sim) | decode touches 69-82 % of the tail within 256 tokens |
| device-side route grouping and graph | `f0db7af` | "no critical-path target left" |
| batched indexer, PLE async, stager, batched embed, #42 ports | `767fcfc`, `f7ecc89`, `fdaddbe`, `4935d9b`, `14c5975` | superseded by upstream at a merge |

---

## 4. Merge resolutions and interaction risks

All of §4.3 and §4.4 is **static reading** unless a source is cited.

### 4.1 Merge 1 (`2a9b26c`, parents `f7ecc89` + `c1e9033`): 14 files, 73 hunks

| file | hunks | resolution | rating |
|---|---|---|---|
| `CMakeLists.txt` | 1 | links `strata_secondary`, `strata_spec`, psapi | clean |
| `README.md`, `docs/DETAILS.md` | 1 each | upstream text | clean |
| `mtp.hpp` | 1 | upstream `kv_restore()` added (used at `generate.cpp:5132`) | clean |
| `pool.hpp` / `pool.cpp` | 5 / 6 | upstream claim protocol; our `rest()` re-added (`pool.hpp:176`); `rest_` cleared in `publish()` before the epoch bump (`pool.cpp:242`) | clean |
| `prefill.hpp` | 1 | our `set_peer_tier` kept; our `ms_stage_*` counters dropped | intended (diagnostics) |
| `serve/server.py` | 12 | upstream 3-tuple `prepare()`, "0/−1 = rest of context" replaces our 1024-token default; loop guard kept (`:919`) | intended |
| `setup.py` | 1 | `MIN_ENGINE (0,1,20)`; package union | clean |
| `pinned.cu` | 1 | our `allow_large_pages` plus upstream `STRATA_NO_LARGEPAGES` | clean |
| `verify.cpp` | 2 | our `ms_tail` plus upstream sampling | clean |
| `iq_kernels.cu` | 2 | our `x_scales` signature plus upstream case 23 (`:929`, `:954`) | clean |
| `prefill.cpp` | 12 | our `class Stager` replaced by upstream's `struct Stager`, with our NVMe `Job{l,e}` re-added; PLE async replaced by upstream `ple_next`; peer branch in `issue_one` | intended |
| `generate.cpp` | 27 | parser split into two else-if chains (MSVC C1061, `:1428`, `:1537`); upstream PCIe probe dropped; LRU conversation cache taken; upstream refill/lend lambdas | intended |

- **A race the merge fixed** (MEASURED, `2a9b26c` message). The auto-merge left `copied` unrecorded on the stager
  branch, which gave NaN from layer 0. The merge now records `copied` on every branch.
- **One silent loss (log only).** `cpu_require_expert_support()` was kept throughout (`2a9b26c:generate.cpp:1312`,
  HEAD `:1972`). What merge 1 dropped is upstream's other branch of the same conflict, `61126e9` ("generate: say which
  expert kernels an AVX2-only CPU gets"): the `else if (!cpu_avx512_ok())` notice "this CPU has no AVX-512: the expert
  kernels run on AVX-2 / ggml-cpu vec_dot (STRATA_NO_IQ256 set)" for native packs (`c1e9033:generate.cpp:1146-1149`).
  At `2a9b26c` and HEAD only our Q2_0-only "native Q2_0 expert rows use AVX-512/AVX-VNNI/AVX2" line remains
  (`generate.cpp:1973-1978`), so non-Q2_0 native packs on AVX2-only CPUs lose the notice. The #47 dossier quotes U1's
  log with that exact line.
- **Non-conflict edits in the merge** (`2a9b26c` message): `serve/test_server.py`'s answer fixture made non-repeating
  (upstream's `"x" * 2000` answer is a loop our `LoopGuard` stops at 512 chars), and `tests/xeno/test_count_tokens.py`
  adapted to the 3-tuple `prepare()`.
- **Merge 1's verification** (`2a9b26c` message): full build; ctest 45/47 (the two failures also failed before the
  merge); pytest `tests/xeno` 32, serve + tools 60; same-session ABBA, single-GPU 2K serve prompt: output tokens
  identical to the pre-merge exe, prefill 6176/6115 → 5049/4954 ms.

### 4.2 Merge 2 (`6c09c5b`, parents `4c68013` + `7134ce9`): 14 files, 75 hunks

| file | hunks | resolution | rating |
|---|---|---|---|
| `generate.cpp` | 20 | see below | two suspicious items |
| `prefill.cpp` | 20 | see below | issuer default flip |
| `serve/server.py` | 7 | status/rate, timings, `stop_sequence` + `cache_read_input_tokens`; `stop =` collapsed to one line (~1284, cosmetic) | clean |
| `expert_source.cpp` | 6 | remote `begin` (`:905-912`) before our secondary claims (`:914-944`); remote claims `kind == -1`, secondary `kind < 0` rows left over: disjoint | clean |
| `CMakeLists.txt` | 5 | `strata_secondary_compute` links `CUDA::cudart` unconditionally (`:312`); `strata_secondary` already did (`:161`, ours since before merge 1) | behaviour change for a HIP build (UNMEASURED; all sites listed under "HIP build" below) |
| `verify.cpp` | 4 | our `pcie_share_` skip kept (`:704`); H2 merge on the non-device-plan branch (`:726`) | **suspicious** |
| `kernels.hpp` | 3 | comments; our duplicate F-1/F-2/`copy_i32` removed | clean |
| `expert_source.hpp` | 3 | `open(..., pin_for_cuda, defer_load, max_pinned_bytes)`; only caller `generate.cpp:2412` | clean |
| `native_qsa_indexer.cu` | 2 | both overloads kept; ours (`:270`) dead | clean, dead code |
| `pinned.cu` | 1 | `pinned.cu:152` returns early for pageable, before the cap | clean |
| `device.cu` | 1 | upstream's sm_80 floor; our flag unused (`device.cu:72`) | intended; removes upstream's old sm_120-only check (below) |
| `setup.py` | 1 | `MIN_ENGINE (0,1,26)`; version comment truncated mid-sentence | clean (cosmetic) |
| `verify.hpp` | 1 | our `set_pcie_share` labelled "(upstream)" (`:136-139`) | label is wrong |
| `pinned.hpp` | 1 | ctor `(bytes, bounds, pin_for_cuda, max_pinned_bytes)` | clean |

Two files were edited in the merge without a conflict:

- `src/core/remote_experts.cpp`: our three-argument `native_expert_scratch_bytes`.
- `src/prefill/kernels.cu`: our #42 copies removed.

**Suspicious 1: the `pcie_share_` skip that upstream rejected** (`verify.hpp:136-139`, `verify.cpp:704`).

- Upstream's `9a19f57` left it out because a request can carry `pcie_frac=` and the window graph is captured once.
- Our resolution sets the skip once, from `o.pcie_frac` (`generate.cpp:4476` serve, `:6138` generate).
- Failure path (static reading):
  1. The server starts with the default `--pcie-frac 0` and a pinned, device-aliased arena (no exclusive tiers, no
     secondary).
  2. A request carries `strata_tune {"pcie_frac": x>0}` (`generate.cpp:5277`; `server.py:292-297`).
  3. Dispatch computes `pcie_ok` and the share (`expert_source.cpp:831-832`), sets `kind=1` rows (`:852`) and zeroes
     them for the CPU. Their entries are appended to the same `P.dst` list, and `P.counts[1] = entries` covers both
     VRAM and PCIe entries (`:876-890`).
  4. The captured window never runs the PCIe grouped launch, so those rows' `hit_out` is never written this window.
  5. `merge_mapped_kernel` takes `hit_out[row]` for every row found in `dst` (`s2_expert_grouped.cu:430-448`), and
     `hit_out_` is never cleared (`verify.cpp:249`). So those rows take **stale `hit_out` values from an earlier layer
     or window**, rather than being dropped: wrong output with no error, and a bogus speed for calibrate.
- D2x is safe from this: the pageable arena has `device_alias == nullptr`, so `pcie_ok` is false.
- Whether calibrate is ever used on a pinned-arena config here is UNMEASURED.

**Suspicious 2: upstream's D-4 refill dropped silently.**

- `refill_lent` (`generate.cpp:872-988`) uses only `fill_slot_blocking` (`:880`, `:967`).
- `fill_slot_queued` is never called. `refill_blocking()` (`:112`) is unused, so `STRATA_REFILL_BLOCKING` is a no-op,
  and `sync_queued` (`:5211`, `:5838`) waits on an empty stream.
- The merge message says "lent-slot refill from the tail file" but does not mention D-4's VENDOR 162 → 96 ms
  (`581765a`: "Q2_0 on a 5070 (1,782 slots)"; the message does not say which arena kind).
- With our pageable arena, D-4 would gain little on the host-copy branch (static reading).

**Intended changes that have no evidence of our own.**

- **Issuer default on** (`prefill.cpp:2250-2257`). Our own number was below noise (`c7a469e`). It leaves dead lambdas
  `wait_issued` (`:2296`) and `give_back` (`:2300`).
- **sm_80 primary floor** (`device.cu:72`). The sm_120-only check was upstream's own through v0.1.20
  (`if (d.cc_major != 12)` at `6da1f66` and `c1e9033`, `device.cu:62`); we only added the sm_89 display exception
  (`allow_display_sm89`, `7134ce9:device.cu:62`). Upstream relaxed it in `f36af4d` (sm_80), and v0.1.27 relaxes it
  further to sm_75 (`aad5bb1`). That removes the check which also stopped the 4070 becoming primary by accident: with
  `CUDA_VISIBLE_DEVICES` misordered and no secondary tier, the 4070 would silently run as primary. The secondary tier
  re-checks the names (`secondary_arena.cpp:69-77`).
- **`STRATA_PREFILL_EMBED_BATCH` removed.** Both sides claim identical bits.
- **HIP build** (UNMEASURED, static reading). Wider than the merge hunk: `strata_secondary` links `CUDA::cudart`
  (`CMakeLists.txt:161`) and NVML (`:162-165`) unconditionally inside the GPU block (ours since before merge 1);
  `strata_secondary_compute` links `CUDA::cudart` (`:312`, the merge-2 resolution); `strata-device` (`:173`) and
  `strata` (`:343`) link `strata_secondary`; `secondary_arena.cpp`, `secondary_runner.cpp` and `secondary_vram.cpp`
  include `<cuda_runtime.h>` with no HIP guard. v0.1.27's `da77db6` ("make qsa_prompt_attn.cu compile on HIP-only
  toolchains") shows upstream now builds HIP without CUDA headers.
- **Stale help text.** `generate.cpp:514-515` says exclusive primary "Needs profile, --no-prefill-borrow, --adapt-swaps
  0", but D2x runs exclusive with borrowing (`:2796`). (`--secondary-free-floor-mib`'s help, "default 2560", matches the
  option default at `:235` and is not stale.)
- **Stale comment.** The layer-split stage probe (`generate.cpp:2315-2321`) says "the same rule as CUDA0's above", but
  that CUDA0 probe was dropped.

**Upstream numerics changed our outputs.** #45: "The merged engine gives its own 60f0e5af / 5b5884bc … differs from
Strata-xeno because of upstream's numerics (for example the window combine in one launch)". The attribution is
UNMEASURED (§2.3 D3.5).

### 4.3 Part B: features correct alone that may break together

**B.0 E-9 × wave (#48). Fixed in `5c51574`.**

- **Bug.** `sp2.on_chunk = sp.on_chunk` made lane 2's thread run lane 1's `Prefill::draft_kv`. `draft_kv` carves rows
  from `m.region` (`prefill.cpp:1540-1560`) and syncs `m.cs` (`:1626`), while lane 1 was using the same region and
  stream. The `src_dev` row table was overwritten with floats, and `gather_q8_rows` / `mul_mat_q` hit an illegal
  address.
- **MEASURED** (`5c51574` message): "rows out of range 23 of 25", 8 of 8 runs on 16,405 tokens; 0 of 3 crashes without
  MTP, 3 of 3 with it.
- **Fix.** Factory lambdas bound per lane (`generate.cpp:4417`, `:4787`, `:5804`).
- **What remains.**
  - `on_chunk` calls are serialised by `wait_tail` / `publish_tail` (`prefill.cpp:3148`, `:3155`).
  - Each lane writes a disjoint cell range.
  - `mtp.ms_prefill +=` (`prefill.cpp:1630`) runs from both lanes' threads, but inside `on_chunk`, which the
    `WaveLink` mutex and condition variable serialise (`wait_tail` / `publish_tail`, `prefill.cpp:840-848`), so it is
    not raced. #48 rules the concurrency out as H10.
  - Not covered by this fix: mid-prompt checkpoints taken from a lane's `on_chunk` while the other lane advances the
    in-place state (§8 item 13, static reading).
- **Verdict: safe.** Outputs `dfa16be2`, `025c15f8`, `a2ab06c2` are identical with and without the wave (MEASURED,
  #48 comment).

**B.1 D-4 × tail file.**

- **At HEAD:** D-4 is dormant, so this is safe.
- **If D-4 were ported into `refill_lent`:** queued copies from `bounce[buf]` (`:920-926`) would race batch k+2's
  reader thread refilling the same buffer (`:958-975`). Slots would get the wrong expert with no error.
- Tail slots cannot change under swaps: paired swaps skip `r[e] >= excl_keep_from` (`:4673`, `:6441`).
- **To settle:** compare read vs H2D in the existing refill line (`generate.cpp:980-986`). If the port is made, check
  `ExpertCache::verify_slot` over a refill of more than 64 experts.

**B.2 D-5 issuer × wave × exclusive tier.**

- With the split layout, `moe_cap(T) = min(T, 2047)` (`prefill.cpp:190-192`), so `m.ring = STAGE`, so
  `stream_all` is false (`:1982`) and `use_issuer` is false (`:2257`).
- **Wave + issuer:** not reachable with the Q2_0 pack.
- **Issuer + exclusive-4070 peer copies:** safe. `read_into` uses thread-local handles (`expert_source.cpp:1716-1738`).
- **To settle:** a D2x generate run with `STRATA_PREFILL_EXPERT_SPLIT` unset, arms I / I0 = `STRATA_PREFILL_ISSUER=0`
  in the order I I0 I0 I; compare the hashes.

**B.3 Upstream `a4a5a37` remote caches × our 4070 tier.**

- The rows are disjoint.
- No option guard rejects `--expert-cache-device1` together with `--secondary-expert-mib`.
- **Static reading: with the default exclusive 4070 tier the combination most likely fails at boot.**
  `RemoteExperts::open` picks ranked pairs that are not in the primary cache and not in `claimed`
  (`remote_experts.cpp:140-147`), and `claimed` holds only layer-split stage residency (`generate.cpp:3406-3409`), not
  the 4070 tier's. It then calls `source.blob()` on each pick and fails with "CUDA… experts: cache fill failed" on a
  null blob (`remote_experts.cpp:169-175`); the engine returns 1 (`generate.cpp:3412-3415`). The 4070 tier is filled
  first (`:2974-3160`) and, exclusive by default, its pairs get `exclusive_[i]=1` (`expert_source.cpp:1794-1802`), for
  which `ArenaExpertSource::blob()` returns `nullptr` (`:1825-1830`). With a ranked profile the first remote picks are
  exactly those 4070-owned pairs.
- Only with `--no-exclusive-secondary-experts`, or serve with `--adapt-secondary > 0`, does the duplicate case apply:
  both tiers can then hold the same expert (wasted VRAM on device 1, no double write).
- **Verdict:** dormant in D2x. A PR should add an option guard, or make `RemoteExperts` skip secondary-owned pairs.

**B.4 C-4 I/O pool × PLE row cache.**

- Outputs cannot change: the row cache stores file bytes.
- The completion order changes cache eviction, so speed may vary.
- `7134ce9`'s gains were measured before C-4. On the merged tree they are UNMEASURED.
- **To settle:** `ple_reader_test --prefetch`, then `STRATA_IO_THREADS=1` vs the default.

**B.4a C-4 × the tail reader's `thread_local DirectFile`. A hypothesis for the exit hang: at risk.**

- `thread_local strata::platform::DirectFile f;` (`generate.cpp:3251`) runs on every stager thread that reads a
  lent-tail expert (`expert_source.cpp:1717`).
- After C-4, each `DirectFile` owns 4 pool threads, and its destructor joins them (`direct_file.cpp:152-156`).
- MSVC runs `thread_local` destructors during thread detach under the loader lock, which the pool threads need in order
  to exit. So the stager thread could never finish exiting.
- This matches the recorded symptom: "all four return, the first join never does" (`generate.cpp:6963-6965` comment;
  `7b31aa6`). It also fits the timing: the hang was first seen after merge 2, and `DirectFile` had no pool before C-4.
- **Not measured directly. Supporting logs** (`C:/Strata-exp/hang-trace`, `hang-pa0`, `hang-pa1`, `hang2`,
  `hang-mmap` `.stderr`, 2026-09-30 05:52-06:14; `hang2.log`: "HUNG - killed after 100 s"): every run builds or uses
  the tail file ("tail file refill 192 experts") and logs prompt sources "nvme 37" (51 in `hang-mmap`). By static
  reading that "nvme" counter is the stager's `blob == nullptr` path (`prefill.cpp:2212`, `:2828`; tier names at
  `generate.cpp:5861`), which calls `ArenaExpertSource::read_into`, and `read_into` tries the tail reader first
  (`expert_source.cpp:1717`); the reader constructs its `thread_local DirectFile` only for a pair that is in the tail
  file (`generate.cpp:3246-3251`). So stager threads very likely did open a `thread_local DirectFile` in these runs.
  These runs are one-card ("GPU tiers own 7529 + 0"), with the stale 8,000-pair profile. `hang-mmap` still used the
  tail file and placement-first, so it does not isolate `--mmap-experts`.
- **To settle:**
  1. A CPU-only repro: a thread opens a `thread_local DirectFile` and returns; `join()` it.
  2. D2x generate with `STRATA_EXIT_TRACE=1`, default vs `--no-tail-file`.

**B.4b Our decode PLE prefetch × wave in serve. At risk.**

- The last decode window of a request pushes a prefetch (`generate.cpp:5479-5480`), so `ahead` is non-empty when the
  request ends.
- In the next wave request, both lanes' `ple_read` threads call `gather_batch` on the same `PleTable`
  (`prefill.cpp:1836`, `:1931-1938`). Both drain the **unlocked** `std::deque ahead` (`ngram.cpp:133-141`).
- Possible outcomes: "PleReader: unknown ticket" (`ple_reader.cpp:416/422`), or `pop_front` on an empty deque, which is
  undefined behaviour.
- Generate mode cannot hit this, because it does no decode before its prefill.
- Nuance (static reading, not re-checked): both lanes' `ple_read` threads call `gather_batch` on the shared `PleTable`
  even when `ahead` is empty. `PleReader` is internally locked, but upstream's `impl_->bytes_read +=` in the direct
  gather path (`ngram.cpp:325`) is also unlocked; that race is a stats counter only.
- **To settle:** 20+ back-to-back serve requests on one conversation with the wave on, each adding ≥ 8,192 tokens,
  with `--ple-ahead 0` as the control. Or a two-thread `PleTable` unit test.

### 4.4 Part C: other shared paths (54 upstream source commits × our features)

| upstream | × ours | risk |
|---|---|---|
| `de159b5` mmap-experts | exclusive / tail / placement-first | low: guarded by `!o.mmap_experts` |
| `c6c6594` E-9 | KV streaming / layer split / split tier | low: `draft_kv` refuses `kv_mode != 0` (`prefill.cpp:1494-1496`); `!multi_gpu` (`generate.cpp:4422`) |
| `c48690a` / `2aa8f72` k8v4 | wave / split | low, never exercised (D2x uses `--kv int8`) |
| F-1 / F-2 / `fe609ce` | split / wave | low; mapped tables skipped for split layers; identical in O and D (#48 H5) |
| HIP series | our CUDA-only additions | medium, upstream PR only |
| `731899f`, D-1 (`static bool attr[64]`) | wave lanes | formal UB, benign (`qsa_prompt_attn.cu:616`, `qsa_select.cu:471`) |
| `ce8789b` / `ade1aac` load hint | deferred load | low: deferred load reports `st.bytes = want` (`expert_source.cpp:1315`), so the printed load rate is meaningless |
| `57fa1fd` `--dump-routing` | `--route-trace` | low: two independent trace writers |
| `0abcd5b` E-6 | 4070 tier / H2 merge | low; E-6's residency check sees only CUDA0 (`verify.cpp:621`) |
| `66f4341` / `2575cb1` GDN | wave | low: layer-l waits order the GDN state (`prefill.cpp:2313`) |
| `758eb1a` C-2 | our batched indexer | ours is dead code |
| layer split series | our 4070 tier | medium if combined: secondary hard-codes device 1 (`set_peer_tier(...,1)`) |
| layer split | exclusive primary + paired swaps | cannot tell statically: `eligible` does not exclude `multi_gpu`; paired stages write `d_res` directly and skip `res_upload` (`:4613`, `:4655`) |
| `1e4515c` | `set_pcie_share` | suspicious (§4.2) |
| `9fba872` remote spin | our pinned pool workers | low, dormant |
| `f36af4d` sm_80 | upstream's old sm_120 guard, which our 4070 tier relied on | low to medium (§4.2) |
| `5b7e316` C-4 | `TailFile::read_slot` checks `c[0].ok` but not tag or bytes (`generate.cpp:636-640`) | **medium** (static reading): `DirectFile` reports an immediate end-of-file as a zero-byte completion with `ok = true` (`direct_file.cpp:118-121`, `e == ERROR_HANDLE_EOF`), so a truncated tail file would copy a stale bounce buffer into the expert slot with no error |
| `5b7e316` C-4 | thread count | 4 I/O threads per `DirectFile` × (PleReader + `g_tail` + each stager's `thread_local` + arena `dfiles_`) |

---

## 5. System blueprint

`UPSTREAM` means byte-identical to `4c68013`. `XENO` means a new file in our delta (145 files). `MIXED` means an
upstream file we modified (45 files). Source: `git diff --name-status 4c68013 5c51574`.

### 5.1 The model served (`include/strata/core/layout.hpp:27-58`)

- **Model:** Qwen3.8-Flash-Next, 48 layers.
- **Attention:** every 4th layer is full attention ("QSA", 12 layers); the other 36 are Gated-DeltaNet ("GDN").
- **Width:** `n_embd` 2560, hyper-connection `hc` = 4.
- **MoE:** on every layer, 512 experts, top-10, `n_ff` 640.
- **PLE:** a per-layer n-gram embedding table in GGUF shard 2 on SSD.
- **Drafter:** one MTP draft layer.
- **Sizes:** an expert blob is 1,382,400 B (`generate.cpp:652`); the Q2_0 arena is 34.0 GB (`setup.py`).

### 5.2 Process picture

```
 Claude Code / OpenAI client / web app
          |  HTTP (SSE)                          D2x: port 8091, launched by D:/Github/Strata/start-flash-next.ps1
          v                                      (sets CUDA_VISIBLE_DEVICES=1,0 -> CUDA0 = 5060 Ti, CUDA1 = 4070 SUPER)
 +----------------------- serve/server.py (Python, one process) [MIXED] ------------------+
 | Handler -> frontend.py (API -> chat messages, Qwen XML tool-call parser, template)      |
 |         -> Service.prepare (Jinja template + BPE tokenizer, tools/strata_tokenizer.py)  |
 |         -> Service.run (FIFO lock, LoopGuard [XENO], StopSequenceFilter [XENO], Detok)  |
 |         -> StrataEngine: stdin/stdout text protocol (GEN/GENI/STOP/QUIT <-> T/PP/DONE)  |
 |   optional: Vision, McpHub (serve/mcp.py), Telemetry, timeline [XENO]                   |
 +---------------------------------------+------------------------------------------------+
                                         | pipes (one resident sequence)
 +---------------------- strata.exe --serve (src/program/generate.cpp) [MIXED] -----------+
 | boot: weights -> session -> PLE -> MTP -> host arena -> CPU pool -> GPU0 cache (fill)  |
 |       -> 4070 tier (fill) [XENO] -> host load_rest [XENO] -> tail file [XENO] -> ...   |
 | per request: resume (live / checkpoint) -> prompt parts (windows | batched: borrow,    |
 |       split [XENO], wave [XENO]) -> checkpoints -> refill lent slots -> verify-window  |
 |       decode loop (MTP / suffix drafts) -> adaptive swaps between rounds -> DONE       |
 +------+-----------------------+----------------------+---------------------+----------+
        v                       v                      v                     v
  GPU0 5060 Ti (x4)      GPU1 4070 SUPER (x16)   CPU pool (13 workers   host RAM arena (pageable,
  trunk, KV, MTP,        exclusive expert tier,  + host thread,         reserve-only; host-owned
  expert cache           split-prefill experts   AVX-VNNI Q2 rows)      experts) + NVMe (pack, PLE,
                         [XENO]                  [MIXED]                tail file)
```

### 5.3 File responsibilities

| path | responsibility | owner |
|---|---|---|
| `src/program/generate.cpp` (6,969 lines) | engine driver: options, boot and placement, `--serve` loop, generate loop, adaptive tiers, checkpoints | MIXED +1,964/−111 |
| `src/prefill/prefill.cpp` (3,231) | batched prompt path: GEMMs, MMQ experts, streamed ring, stager, split, wave, Dm frontier, E-9 drafter K/V | MIXED +1,548/−205 |
| `src/prefill/frontier.cu` + `.hpp` | #41 Dm frontier | XENO (`78f373e`) |
| `src/prefill/moe_mmq.cu` + `.hpp` | llama.cpp MMQ for prompt experts; nsm=1 (`:153-154`) | MIXED |
| `src/prefill/kernels.cu`, `gemm.cu` | prompt kernels; `moe_routed_sum`, `moe_shared_finish`; `warm_cublas` | MIXED |
| `src/core/verify.cpp` + `.hpp` | speculative verify window: T tokens × 48 layers in one captured graph, bit-exact to greedy (`verify.hpp:1-20`) | MIXED |
| `src/core/mtp.cpp` + `.hpp` | MTP draft layer; `on_draft` hook | MIXED |
| `src/core/expert_source.cpp` + `.hpp` | expert sources; `expert_pool_dispatch_multi`; placement-first reads; exclusive release/recommit; NVMe tier | MIXED +562/−5 |
| `src/core/pinned.cu` + `.hpp` | host arena: VirtualAlloc, large pages, cudaHostRegister; `reserve_only`, `decommit_interior`, `commit_interior` | MIXED |
| `src/core/secondary_{arena,runner,vram}.cpp`, `secondary_{budget,profile}.hpp` | 4070 tier | XENO |
| `src/core/expert_cache.cpp` | GPU0 expert cache | UPSTREAM |
| `src/core/remote_experts.cpp` | upstream helper-GPU caches | MIXED (merge-only, 1 line) |
| `src/core/device.cu`, `device_main.cpp` | `strata-device`; our 4070 VRAM probe | MIXED |
| `src/kernels/cpu/q2_avx2.cpp`, `expert_layout.cpp` | Q2_0 rows; AVX-VNNI arm; ISA dispatch | MIXED (`f679806`) |
| `src/kernels/cpu/pool.cpp` + `.hpp` | CPU expert pool; P-cores first; priority; `rest()` | MIXED |
| `src/kernels/cuda/iq_kernels.cu` | native expert GPU kernels; CPU-order Q2_0 rows K0-K4 (`:332`) | MIXED |
| `src/kernels/cuda/s2_expert_grouped.cu` | grouped hit kernels; `moe_hit_merge_mapped` (`:764`) | MIXED |
| `src/kernels/ngram.cpp` | PLE table, `prefetch` (`:298`) | MIXED (`7134ce9`) |
| `src/platform/timeline.cpp`, `include/strata/timeline*.hpp` | #33 timeline | XENO |
| `src/platform/direct_file.cpp` | unbuffered overlapped reads (C-4 pool) | UPSTREAM |
| `serve/server.py` (1,903) | HTTP, engine client, OpenAI/Anthropic/MCP | MIXED |
| `serve/frontend.py` | messages, tool-call parser; billing strip; document blocks | MIXED |
| `serve/loop_guard.py`, `serve/pdf_blocks.py`, `serve/timeline.py` | loop guard; PDFs; server timeline lanes | XENO |
| `CMakeLists.txt` | our `strata_timeline`, `strata_secondary`, `strata_secondary_compute`, `STRATA_BUILD_XENO_TESTS` | MIXED (+179/−9) |
| `tests/xeno/**` (88 files) | parity tests, Python tests, `perf/` tools | XENO |

**CMake targets.**

- `strata` links `strata_engine strata_prefill strata_secondary strata_spec`, plus psapi.
- `strata_engine` links `strata_secondary_compute`.
- `strata_secondary` links cudart, and NVML when found (`STRATA_HAS_NVML`).
- `strata_timeline` is linked into `strata_core` and `strata_kernels_cpu`.

### 5.4 The life of a request

```
client            server.py                                  strata --serve (generate.cpp)                   devices
  | POST /v1/messages?beta=true (stream)
  |-------------->| _do_post: path split on '?' (:1458)
  |               | _anthropic (:1583): anthropic_to_messages
  |               |   - strip billing header (frontend.py:184)  [XENO]
  |               |   - document blocks (pdf_blocks.py:31)      [XENO]
  |               | Service.prepare (:741): render + encode
  |               | Service.run (:831): FIFO lock; restart dead engine
  |               | "GEN <max_new> [keys] <ids>\n" ---------->| next_line (:4871); parse keys (:4884-4924)
  |               |                                           | resume = live | longest prefix checkpoint (:5053-5087)
  |               |<---------------- "RESUME n" --------------| (:5146); apply_pending(true) (:5149)
  |               |                                           | parts [read_from,root_at) [..,turn_at) [..,n-1)
  |               |<--- "PP pos total ms tok/s" per chunk ----|   windows (<= --short-read 64) | batched:
  |               |                                           |   lend -> Prefill::run / run_wave  ---> GPU0 trunk,
  |               |                                           |   checkpoint_at(root/turn) (:5348)      4070 MoE
  |               |<---------------- "REUSED n" --------------| refill lent slots (tail file); (:5359)
  |               |<---------------- "T id" x (a+1) ----------| decode loop (:5402)                ---> GPU0 graph +
  | SSE events    | Detok -> OutputParser -> LoopGuard ->     |                                          4070 + CPU pool
  |<--------------| StopSequenceFilter -> content blocks      |
  |               |<-- "DONE gen prompt pms dms fin acc off reused hits look" (:5636)
  | message_delta | usage: input_tokens = n - reused, cache_read_input_tokens = reused (:1287-1291)
```

**Engine protocol** (`generate.cpp:4073-4089`, `:4816-4854`, `:4871-5712`):

- **Boot:** `INFO key=value…`, then `READY <ctx> stop`.
- **Server → engine:** `GEN` / `GENI` / `STOP` / `QUIT`.
- **Engine → server, per request:** `RESUME` → `PP`* → `REUSED` → `T`* → `DONE`, or `ERR`. Several paths
  `return 1` after `ERR`; the process exits and the server restarts it.
- **Watchdog:** `STRATA_WATCHDOG_S` (60).

**Prefix reuse** (`:5044-5146`). `resume` is `live` if it is a prefix of the prompt, else the longest checkpoint that
is a prefix.

- Every checkpoint that is longer than `resume` **or is not a prefix of this prompt** is erased (`:5083-5085`). So an
  interleaved request that diverges early (a Claude Code side request) wipes the chain (§8 item 2).
- `resume == 0` clears all checkpoints (`:5088-5096`).
- A checkpoint resume restores GDN, PLE history and indexer tails; the KV cells are positional and trusted.

**Prompt parts** (`:5286-5356`).

- `turn_at` is the last `<|im_start|>`. `root_at` is the first, when the prompt is read from 0 and that position is
  ≥ 2048.
- A part of ≤ 64 tokens goes through verify windows. Otherwise the part is lent slots, then `Prefill::run` or
  `run_wave`.
- A part log line is printed (`8408f84`). Lent slots are refilled before the first decode window.

**Borrowing** (`plan_lend`, `:4131-4153`). The chunk is the largest of {8192, …, 256} whose buffers take at most 85 %
of the cache slots, or 90 % with pinned experts. The buffers are carved from the **last** GPU0 slots.

**Decode per round** (`:5361-5519`):

1. `apply_pending`.
2. `ver.run(T, …)`: host dispatch per layer (§5.5); the 4070's partials and the CPU rows merge.
3. Accept the longest matching prefix.
4. Start the adapt thread.
5. `ver.commit(a+1)`, then `ple_ahead.push` [XENO].
6. Emit `T` lines.
7. `mtp.draft`, whose `on_draft` does the PLE prefetch [XENO].
8. Join adapt; check EOS or STOP.

The window is up to 6 tokens with `--spec 4` plus the suffix drafter (`:1785-1788`).

### 5.5 Memory hierarchy and decode dispatch

```
                          +----------- GPU0: RTX 5060 Ti 16 GB (sm_120, PCIe x4) ----------------------------+
 routed (layer,expert) -->| trunk, native dense/head, KV (int8), GDN state, PLE hist, MTP (size UNMEASURED), logits|
 entries per window       | EXPERT CACHE (ranked profile prefix): exclusive owner by default [XENO]; last k    |
                          | slots = "lendable tail" = prompt-path buffers while a prompt is read [UPSTREAM     |
                          | borrow + XENO tail-file refill]                                                    |
                          +-------------------------------------------------------------------------------------+
                          +----------- GPU1: RTX 4070 SUPER 12 GB (sm_89, x16, display card) [XENO] ------------+
                          | SECONDARY ARENA: next-ranked experts, exclusive by default; runner workspace;       |
                          | split-prefill buffers, ring, stager; free floor (D2x: 640 MiB)                      |
                          +-------------------------------------------------------------------------------------+
                          +----------- host RAM 48 GB ----------------------------------------------------------+
                          | arena reserves address space for all 24,576 experts; only HOST-OWNED ones are       |
                          | committed (placement-first) [XENO]. CPU pool computes misses. --ram-cache-gib -> NVMe |
                          +-------------------------------------------------------------------------------------+
                          +----------- NVMe --------------------------------------------------------------------+
                          | GGUF shard 1 (experts + dense), shard 2 (PLE, direct reads), tail-<key>.bin (~3.5 GB)|
                          +-------------------------------------------------------------------------------------+
```

`expert_pool_dispatch_multi` (`expert_source.cpp:780-1075`):

```
ids (n_tok x k) --> usage[layer*512+e] += 1                                    (:812-814)
                --> plan: host_res[l,e] >= 0 -> kind 0: GPU0 hit (in the window graph)
                          miss, pcie share   -> kind 1: PCIe read (D2x: pcie_frac 0 -> never)
                    publish plan to GPU before CPU work                         (:820-898)
                --> upstream remote helpers (--expert-cache-device*)            (:905-913)   [UPSTREAM]
                --> secondary_res[l,e] >= 0 -> kind 2: 4070 launch              (:912-947)   [XENO]
                --> act quantize                                                (:948-955)
                --> NVMe tier: materialize_batch                                (:956-976)   [XENO]
                --> CPU jobs -> pool.run_split_multi_native (13 workers + host) (:978-1030)
                --> secondary_runner.finish(out)                                (:1036-1052) [XENO]
tier_entries[0..3] = primary / 4070 / pcie / cpu                               (:993, :1021)
```

- **Tier split** (MEASURED, dual D2 decode on the code prompt): primary 56.9 %, 4070 28.0 %, CPU 15.1 % (checkpoint
  report §3.1).
- **Measured sizes:**

| arm | GPU0 tier | 4070 tier | host-owned | working set | source |
|---|---|---|---|---|---|
| D2 (tier 8704) | 6,239 slots / 8.03 GiB | 6,602 / 8.50 GiB | 15.11 GiB | 16.4 GiB | checkpoint report §3.1, §5 |
| ranked profile, tier 6400 | — | 4,854 / 6.25 GiB | 17.36 GiB | 18.8-19.2 GiB | same §5; #45 m10 |
| D2x | UNMEASURED here | — | — | 19.2 GiB at 8K | #45 m11 |

**Boot order.**

| step | code | note |
|---|---|---|
| options, validation | `:1419-1869` | unknown flag = error |
| LAZY CUDA [XENO] | `:1413-1418` | EAGER removed (`c868593`) |
| `--pcie-frac` → 0 [XENO] | `:1909-1919` | #27 |
| exclusive and placement decisions [XENO] | `:1920-1970` | verified |
| MTP before the host arena | `:2332-2348` | "WDDM can refuse the draft weights after mapping tens of GiB of host pages" (`:2330-2331`) |
| host arena | `:2405-2440` | `pin_for_cuda = false` with the 4070 tier or exclusive primary |
| cache sizing | `:2619-2757` | ×3/4 retry (#60) |
| placement-first GPU0 fill [XENO] | `:2811-2963` | reader thread → 32 pinned → H2D → D2H → memcmp → release |
| 4070 tier [XENO] | `:2974-3160` | monitor thread `:3148` |
| `load_rest` [XENO] | `:3177-3190` | host-owned only |
| tail file [XENO] | `:3231-3265` | — |
| serve setup | `:4155-4870` | lend, lanes, verifier, drafter, adapt, stdin thread, READY |

**Ownership rules [XENO].**

- Exclusive primary is on when eligible. Exclusive secondary is on with the tier.
- Placement-first applies whenever any tier is exclusive.
- The lendable tail is refilled from the tail file.

**Adaptive swaps.**

- Code: serve `:4485-4728`, generate `:6150-6490`.
- Usage decays ×0.7 per adapt call. A swap needs usage ≥ 2 and a gain ≥ 1.5.
- **D2x in serve:** paired primary swaps 8 every round; the 4070 tier is static (`adapt_secondary` resolves to 0).

### 5.6 KV cache and checkpoints

- **QSA KV** (`--kv fp16|int8|q4_0|k8v4`): D2x uses int8, all in VRAM (`--kv-resident 0`).
- **GDN state:** one in-place buffer.
- **MTP K/V:** its own, over the last 32,768 cells.
- **Checkpoints** (`:1197-1266`, `conv_cache.hpp`):
  - Contents: ids plus GDN, PLE and indexer tails, about 118 MB each.
  - `checks[0]` is pinned; the rest rotate LRU; the cap is 6.
  - Created mid-prompt (every 16,384 fresh tokens, in `serve_chunk`, `:4432-4450`), at the root (`:5299-5306`) and at
    the turn (`:5290-5293`).
  - `checkpoint_at` dedupes **by length only** (`:4384-4385`), then runs `cudaDeviceSynchronize` + `checkpoint_save`
    (`:4396`).

### 5.7 Configuration surfaces

**The 25 XENO flags** (set difference against `4c68013`):

`--adapt-gate --adapt-secondary --cache-cpu-only --exclusive-primary-experts --exclusive-secondary-experts
--lock-cpu-experts --mmvq-exact --no-exclusive-primary-experts --no-exclusive-secondary-experts --no-tail-file
--ple-ahead --pool-priority --pool-rest --process-priority --profile-decode-range --profile-prefill-range
--ram-cache-gib --route-trace --secondary-async-launch --secondary-expert-mib --secondary-free-floor-mib
--secondary-graph --secondary-profile-timing --secondary-stage-only --tail-file`

No upstream flag was removed. Upstream defaults we changed through a flag: `--pcie-frac` (−1 → 0), and
`--adapt-swaps`/`--adapt-every` (8/1 under exclusive primary).

**Default-on or flagless behaviour changes against `4c68013`** (upstream rule 7, "no silent default changes"; option
defaults verified at `generate.cpp:181`, `:237`, `:239`, `:251`, `:258`, `:312`):

| change | where | way back to upstream's behaviour |
|---|---|---|
| MMQ `nsm = 1` on every device | `moe_mmq.cu:153-154` | `STRATA_MMQ_STREAM_K` set to any value |
| CPU-order Q2_0 GPU path and the `exp(double)` CPU SwiGLU (changes CPU-only output too) | `pool.cpp:419`; `verify.cpp:654`, `:694` | **no opt-out** |
| LAZY module loading instead of forced EAGER | `generate.cpp:1413-1418` | `CUDA_MODULE_LOADING=EAGER` in the environment |
| exclusive primary when eligible (`exclusive_mode = -1`); eligible by default for any Q2_0 native pack with a profile and a cache, since `pcie_frac` now defaults to 0 | `:251`, `:1931-1941` | `--no-exclusive-primary-experts` |
| placement-first whenever a tier is exclusive | `:1948` | follows the above |
| tail file | `:312` | `--no-tail-file` |
| `--pool-priority` 2 (HIGHEST) | `:237` | `--pool-priority 0` |
| `--pool-rest` 1 (generate mode only, P6) | `:239` | `--pool-rest 0` |
| decode PLE read-ahead | `:181` | `--ple-ahead 0` |
| first-chunk PLE read-ahead in the prompt path | `prefill.cpp:1940-1943` | `STRATA_PREFILL_PLE_AHEAD=0` |
| one gather launch per MMQ group | `prefill.cpp:2258-2262` | `STRATA_PREFILL_GROUP_GATHER=0` |
| AVX-VNNI Q2 row dispatch | `expert_layout.cpp:73-77` | `STRATA_FORCE_AVX2=1` (also turns off AVX-512) |
| copy-issuer thread on (upstream's D-5 default, kept) | `prefill.cpp:2250-2257` | `STRATA_PREFILL_ISSUER=0` (per §4.3 B.2) |
| AVX2-only kernel notice for non-Q2_0 native packs dropped (log only) | `generate.cpp:1973-1978` | none (§4.1) |

**XENO environment variables:**

- Prompt-path features: `STRATA_PREFILL_EXPERT_SPLIT`, `STRATA_PREFILL_WAVE`, `STRATA_MMQ_STREAM_K`,
  `STRATA_DM_FRONTIER(_FRAC)`, `STRATA_EXPERT_ORDER`.
- Prompt-path diagnostics and A/B: `STRATA_PREFILL_{BUFFERS,COPY_THREAD,GROUP_GATHER,PLE_AHEAD,ROUTE_TRACE}`.
- Timeline: `STRATA_TIMELINE(_MAX_EVENTS)`.
- Exit hang: `STRATA_EXIT_TRACE`, `STRATA_EXP_QUICK_EXIT`.
- Placement and arena: `STRATA_ARENA_PIN`, `STRATA_HOT_TO_SECONDARY`.
- Probes: `STRATA_LINK_PROBE`, `STRATA_SECONDARY_POKE`.

**D2x** (`D:/Github/Strata/strata-flash-next-d2x.json`):

- **Flags** (read from the file during verification): `--pack D:\Github\Strata\packs\q2_0`, `--native <Q2_0 shard 1>`,
  `--ple-gguf <shard 2>`, `--expert-profile D:\Github\Strata\data
anked-exl3only-profile.bin` (untracked, 196,632 B,
  `git hash-object` `c7d37560`, 24,576 pairs; not HEAD's `a4ed7e5`), `--prefill auto` (borrowing on), `--spec 4`,
  `--spec-min-p 0.5`, `--mtp D:\Github\Strata\mtp
t`, `--kv int8`, `--max-context 131072`, `--pool-workers 13`,
  `--adapt-swaps 8`, `--adapt-every 1`, `--pcie-frac 0`, `--expert-cache 8000`, `--vram-reserve-mib 2400`,
  `--secondary-free-floor-mib 640`, `--secondary-expert-mib 6400`, `--exclusive-primary-experts`, `--pool-priority 2`,
  `--process-priority 2`. `cwd` is `D:\Github\Strata`, a checkout on branch `xeno/avxvnni` (`f679806`), so the
  profile, MTP and draft vocab come from that pre-merge-1 tree.
- **Config comment:** several GPUs in the server's `gpu` field make the server add `--layer-split`, so the dual setup
  comes from `CUDA_VISIBLE_DEVICES=1,0` in `start-flash-next.ps1`, not from `gpu`.
- **Environment:** `STRATA_PREFILL_EXPERT_SPLIT=1`, `STRATA_PREFILL_WAVE=1`.
- **Engine:** `dynfix.exe`, the merged engine plus `5c51574`.
- **Measured:** #45 m11 (`dyn.exe` sha256 `8462650abea384b2`): 8K read 4.12/4.15 s, decode code 83.6/86.5 and thai
  50.8/52.2, "An 8K read plus 256 tokens takes about 7.9 s".

**Build.** VS2022 with CUDA 13.3, Ninja, `-DSTRATA_BUILD_TESTS=OFF`, `-DSTRATA_BUILD_XENO_TESTS=ON`,
`"-DCMAKE_CUDA_ARCHITECTURES=89;120"` (`C:/Strata-exp/build-dyn.cmd`). So upstream's own C++/CUDA parity tests were
**not built or run** on D; #45 reports Python tests only (93). Upstream `3801f86` ("STRATA_BUILD_TESTS=ON works
without the unpublished tests/ tree") shows part of upstream's `tests/` is not published, which matters for rule 5
(a PR carries its parity test in upstream's layout).

### 5.8 Diagnostics

- **Per request:** a part log, request metrics by tier, decode hit rate.
- **At boot:** `mem_mark`, placement-first timings, tail-file lines.
- **Timeline:** `STRATA_TIMELINE`, read with `tests/xeno/perf/timeline.py`. `AGENTS.md` requires recording one timeline
  rather than hunting stage by stage.
- **Tests:** XENO C++/CUDA tests exist only with `STRATA_BUILD_XENO_TESTS` on Windows. #45 reports "Our Python tests
  pass (93)" on the merged tree.

---

## 6. Upstream PR candidates, ranked

### 6.1 Upstream's PR rules (upstream #149 and maintainer comments; there is no CONTRIBUTING file at `4c68013` or `a790805`)

1. **One package per PR, rebased on the current release.** #149: "one PR per package rebased on 0.1.26 works best".
   Rebase on v0.1.27 (`a790805`). Merged outside PRs are small: #44 +49/−2, #89 +140/−20, #109 +387/−10.
2. **The default path stays byte-identical, and the PR carries its parity test.** #149: "Each should keep the default
   path byte-identical and include its parity test."
3. **Maintainer gates.** "no-repeat on every quant, long prompts, 2-GPU" (#149). #120 asked for byte-identity on Q2_0,
   IQ3_XXS, IQ3_S and the Coder, on NVIDIA and AMD. A non-bitwise change comes with needles and a KL figure (`dce4598`,
   `731899f`).
4. **An env switch back to the old path**, e.g. `STRATA_*_OLD=1` or `STRATA_*=0`.
5. **Before/after measurement** (#38). Tests should fail on the old code.
6. **HIP/CMake hygiene.** Use `${_strata_gpu_runtime_target}`. Keep CUDA-only features refused on HIP (#148).
7. **No silent default changes, no dropped security checks, no log floods.** #39 was rejected for changing
   `adapt_every`/`adapt_swaps` "without saying so"; #38 for a per-swap stderr line.
8. **No unrelated files** (#38).
9. **Linux parity is expected** (#41).
10. **Expect churn.** The maintainer may fold an idea into his own commit (#65 → `c1e9033`). Upstream went from 0.1.21
    to 0.1.27 in one day.

**Our own rules (#47 body).**

- Branch off `upstream/main`.
- TDD in upstream's test layout.
- English only.
- The body holds the problem, the change, a U vs U+PR table and "Tested", and notes the Claude co-authorship.
- Opt-in first.
- The developer sees each body before it opens.
- Evidence limits to state: one machine, the Q2_0 pack only, no Linux run.

### 6.2 Reconciling with #47

| #47 PR | decision here |
|---|---|
| 1 LAZY + warm-up | kept (rank 3) |
| 2 exclusive ownership | **merged with 3** (proposed here; no recorded decision found in #45/#47/#48) → rank 7 |
| 3 paired swaps | merged into 2 |
| 4 placement-first | kept (rank 8) |
| 5 tail file | kept (rank 9) |
| 6 group-local MMQ scratch | kept (rank 5) |
| 7 exclusive secondary device caches | kept (rank 13), re-targeted at upstream's `--expert-cache-device1` |
| 8 dual-GPU prompt path | kept as RFC (rank 14); **must carry `5c51574`** (stated in the #48 comment: "The upstream wave PR (#47 PR 8) must carry this fix"; agent-written, not a recorded developer decision) |
| 9 AVX-VNNI Q2 rows (new) | added (rank 4); the #47 dossier comment lists it as "New candidate: PR 9" |
| stream-k PR (new) | added (rank 1): MMQ nsm=1, cross-card bit-exact at no measured cost (#5) (proposed here; not in #47) |
| not in #47 | added: decode PLE read-ahead (rank 2), group gather (rank 6), mapped hit merge + pool rest (rank 10), Q2 CPU-order path (rank 11), serve PRs (rank 12) |
| superseded / rejected | K4 skip, pcie default 0, batched indexer, batched embed, PLE async, stager, issuer, #42 ports (§6.4) |

The ranking weighs, in order: generic value to upstream's target user (one 12-24 GB GPU); strength of our evidence;
small size and low risk under the byte-identical rule; and how few prerequisites it has. **Speed deltas below the
13.6 % noise gate are quoted but marked "not established"**; the project's own commits treat gaps of that size as noise
(`0a0791d`: "The 2.5-3 % gap … was noise … under the 13.6 % gate"; `c7a469e`: "below the noise gate, so opt-in"). For
such PRs the body should lead with byte-identity and RAM/VRAM evidence.

### 6.3 The ranked list

**Rank 1: `prefill: MMQ without stream-k for cross-card bit-exact expert products` (the stream-k PR, #5).**

- **Contains:** `ff2bdae` (the `moe_mmq.cu:147-154` override, +8 lines, plus `tests/xeno/mmq_cross_arch.cu`), and the
  test extensions `a87644a` and `4a5c9ce` (same file). These use only APIs upstream has.
- **Not in this PR:** `31943cf`. It is not only a test: it extracts the prompt path's per-sub-product chain into a new
  `mmq::expert_rows` (`moe_mmq.hpp` +32, `moe_mmq.cu` +24, `prefill.cpp` +13/−16), and `expert_rows` calls
  `gather_q8_rows` (`c8f12fb`, rank 5). `moe_layer_cross_arch.cu:139` calls `mmq::expert_rows`. Upstream `a790805` has
  neither function (`git grep`). The one-layer test therefore needs rank 5 plus that refactor first, or a rewrite
  against upstream's lambda. (`31943cf` is also listed in rank 14.)
- **Who needs it:** multi-GPU users of the layer split or helper caches, whose cards have different SM counts. What
  nsm=1 gives is **card-to-card bit-identity of the MMQ products**, measured on sm_89 against sm_120. Upstream's docs
  claim only that "one GPU is byte-identical to 0.1.20, and the hand-off itself is bit-exact" (`docs/MULTI_GPU.md:116`
  at `4c68013`/`a790805`); they say nothing about multi-GPU non-identity. The run-to-run non-determinism we saw in
  upstream's layer split is **our own measurement** on D (D3u `75592447` / `f0b1a8da`, checkpoint report §4 item 3), with
  the cause not isolated. A single-card run-to-run benefit is **UNMEASURED**: on one card the pinned ggml sizes the
  stream-k grid from `nsm` and the tile count only, which are fixed per card (static reading of
  `D:/Github/Strata/third_party/llama.cpp/ggml/src/ggml-cuda/mmq.cuh:1442-1448`).
- **Evidence (MEASURED):**
  - `ff2bdae`: 8/8 identical vs 1/8; 0 of 6.4M values differ; gate/up 713 vs 715 ms at 8K on the 5060 Ti.
  - `31943cf` (needs rank 5, above): one full MoE layer, 0 of 13,107,200 rows.
- **Gate.** It changes default bits (stream-k on is today's default), so it goes in **opt-in** under a new name, e.g.
  `STRATA_MMQ_NO_STREAM_K=1`. Do **not** propose `STRATA_MMQ_STREAM_K=0`: today's code tests only whether the variable
  exists (`moe_mmq.cu:153`, `getenv(...) == nullptr`), so **any** value, including 0, restores stream-k; a PR that
  wants =0/=1 semantics must parse the value. Default-on is proposed separately with numbers. Also measure speed on a
  high-SM card; that is UNMEASURED (only the 36-SM 5060 Ti was measured). E-9 routes the drafter through the same
  context, so the drafter K/V must be covered too (static reading).
- **Review points (static reading).**
  - The override writes the process-global `ggml_cuda_info()` device table through a `const_cast`
    (`moe_mmq.cu:154`). In both trees `ggml_cuda_info()` is defined in `src/prefill/ggml_cuda_host.cu:61` and the
    only other reader found is `mmq::warm()` (`moe_mmq.cu:91`; `git grep ggml_cuda_info\|\.nsm` at `a790805` and
    `5c51574`), so the comment's "the device table is ours to set" holds today, but a reviewer will ask.
  - The effect depends on a ggml heuristic: with `nsm = 1` the NVIDIA ">= 90 % tile efficiency" branch launches one
    block per tile (`mmq.cuh:1442-1448`, pinned llama.cpp). A ggml bump could change that silently.
  - On **HIP** that branch is NVIDIA-only (`GGML_CUDA_CC_IS_NVIDIA(cc)`), so with `nsm = 1` the grid would be
    `nsm` = **one block** for the whole product. The switch must be refused or ignored on AMD (static reading,
    UNMEASURED).
- **Prerequisites:** none for `ff2bdae` + `a87644a` + `4a5c9ce`. **Size:** about 11 lines plus tests. **Risk:** low
  on NVIDIA.
- **Xeno-only parts to remove:** the tests need two architectures; provide a 1-card determinism variant.

**Rank 2: `decode: read the next window's PLE rows during commit and draft` (`--ple-ahead`).**

- **Contains:** `7134ce9`: `PleAhead` (`generate.cpp:1374-1392`), `PleTable::prefetch` (`ngram.cpp:298`),
  `mtp.on_draft` (`mtp.cpp:704`, `:721`), and `ple_reader_test --prefetch`.
- **Who needs it:** every user, since PLE lives on SSD for every pack. One GPU is enough.
- **Evidence (MEASURED, `7134ce9`):** outputs identical; memcmp gate. Speed: thai +2.2 %, code +0.1 % (ABBAABBA);
  code +0.6 % across two sessions. **All below the 13.6 % gate, so not established**; #44 itself says "The first
  session's +0.1 % was the drift".
- **Caveat:** it was measured before C-4. **Re-measure on U vs U+PR.** B.4 says the speed effect on the merged tree
  cannot be told statically.
- **Gate:** byte-identical by construction; include the memcmp test. The default can stay on, because output is
  unchanged (rule 2 is about bytes), but an off switch must exist (`--ple-ahead 0`).
- **Prerequisites:** none.
- **Risk:** medium until B.4b is closed. In serve with the wave, the unlocked `ahead` deque races. Upstream has no
  wave, but a PR must add a mutex, or document that `gather_batch` is single-caller, and should include a two-thread
  unit test.
- **Size:** about +133 including the test.

**Rank 3: `LAZY CUDA module loading, with cuBLAS/ggml warm-up during load` (#47 PR 1).**

- **Contains:** `c868593` (the LAZY revert, `generate.cpp:1413-1418`) and `f4ce2bd` (the warm thread `:1871-1875`,
  `:2757-2762`; `warm_cublas` `gemm.cu:285`; `mmq::warm` `moe_mmq.cu:91`).
- **Who needs it:** RAM-bound users; upstream issues #60, #133 and #141 are about commit and RAM.
- **Evidence.**
  - `c868593`: private −3.1 GiB, about +0.2 GB VRAM per card.
  - `f4ce2bd`: TTFT 3856 → 1243 ms, measured **under EAGER** on the merge-1 build (the warm-up hid EAGER's 2.76 s
    first `cublasCreate`); it does not measure LAZY + warm-up.
  - On upstream (#47 m12), LAZY alone: working set −1.1 GiB, commit −1.6 GiB, but the first 8K prompt +7 %. So the
    warm-up is mandatory.
  - **UNMEASURED:** the net TTFT and RAM effect of LAZY + warm-up on upstream. Measure U vs U+warm-up and U vs
    U+LAZY+warm-up. The warm-up alone may be a separate candidate (upstream has none).
- **Gate.** Output bytes are unchanged. Upstream forced EAGER (`6609960`) for 12 GB cards at 64K+ with ~30 MB free, so
  the PR must keep EAGER when the VRAM reserve is too small, or when the user sets `CUDA_MODULE_LOADING=EAGER`. Test on a
  12 GB card at 64K: we have none, so the maintainer's gate covers it.
- **Prerequisites:** none. **Size:** +52/−10. **Risk:** medium (upstream's original OOM).
- **Still valid:** `a790805` still forces EAGER (verified).

**Rank 4: `cpu: AVX-VNNI path for native Q2_0 expert rows` (PR 9).**

- **Contains:** `f679806` (`q2_avx2.cpp:38-66`, `:104`; `expert_layout.cpp:54-78`) and `10078cb`
  (`tests/xeno/q2_isa_parity.cpp`).
- **Who needs it:** Alder, Raptor and Arrow Lake CPUs (AVX-VNNI without AVX-512); upstream issues #142 (Arrow Lake
  285K) and #34 (13600KF). Not #103: that is a Linux Colab host ("6-core Xeons at 2.2 GHz with no VBMI"), where
  `cpu_avxvnni_ok` returns false (`expert_layout.cpp:66-68`).
- **Evidence.** Bit-exact by construction plus a parity test. Speed: rough separate runs only, with "codex using ~3.8
  cores" at the same time (`f679806`: 49/35/33/38 vs 31/23/33/40; 2 of 4 prompts show no gain or a loss). **An isolated kernel ABBA is required before opening, and is UNMEASURED.**
- **Build fixes needed (static reading, UNVERIFIED).**
  - `__attribute__((target("avxvnni")))`, or a separate TU with `-mavxvnni`, because `CMakeLists.txt:622` compiles the
    file with `-mavx2` only.
  - A GCC/Clang CPUID path, because `cpu_avxvnni_ok` returns `false` off MSVC (`expert_layout.cpp:66-68`).
- **Scope note:** it covers native Q2_0 only. IQ-pack users would need the same change in `iq_avx2.cpp`; offer that as
  a follow-up.
- **Size:** about 60 src lines plus a 52-line test. **Risk:** low.

**Rank 5: `prefill: group-local MMQ scratch, activations quantized once per token` (#47 PR 6).**

- **Contains:** `c8f12fb` (`gather_q8_rows`, `q8_row_bytes`) and `f801c5d` (`mmq_rows_cap`, sub-products);
  `tests/xeno/q8_row_gather.cu`. Split out the unrelated parts: `c8f12fb` also adds `tests/xeno/combine_cross_arch.cu`
  (the split's S2 block, rank 14), and `f801c5d` also adds `tests/xeno/perf/lazy_refill_sim.py` and its test.
- **Who needs it:** single-GPU users with `--prefill auto`. Less borrowed VRAM means more cache slots.
- **Evidence (`f801c5d`):** borrowed VRAM 3,660 → 2,951 MiB at 8K (the main value: more cache slots); commit 41.86 →
  41.16 GiB. Prefill −3.0 %, **below the 13.6 % gate, not established**.
- **Overlap:** upstream has 0-based per-group bounds. Measure U vs U+PR.
- **Gate:** byte-identical (row-gather test). **Size:** about +80/−41. **Risk:** low.

**Rank 6: `prefill: one gather launch per MMQ group on the streamed walk`.**

- **Contains:** `63e0186` (`gather_native_group`, `moe_mmq.cu:243`; `kGatherGroupMax`; the
  `STRATA_PREFILL_GROUP_GATHER=0` switch was already in `63e0186`); `tests/xeno/gather_group_parity.cu`. From the
  review commit `cb23774` take only the `static_assert(MMQ_GROUP <= kGatherGroupMax)`: the rest of `cb23774` is the
  batched-embedding switch and id check (superseded by upstream C-4), the PLE-ahead switch (Q6) and the stager-knob
  removal (Q7).
- **Who needs it:** native-pack users on a narrow link.
- **Evidence (`63e0186`):** byte-identical for group sizes 1..16; host "expert launches" 185 ms, from 1.2-3.4 s. The
  end-to-end −4.0 % 8K prefill is **below the 13.6 % gate, not established**.
- **Caveat.** It was measured before D-5 and C-4 changed the host-launch pressure. **Re-measure first.**
- **Prerequisites:** none. **Size:** core about +140. **Risk:** slot-release ordering, reviewed in `cb23774`.

**Rank 7: `experts: exclusive GPU ownership with paired adaptive swaps` (#47 PR 2 + PR 3, merged).**

- **Contains:** `28a610b`, `92f10c2`, `fdaddbe` (the default and borrow-tail parts, minus the superseded stager),
  `a9e5c0e`, `292e3b8`, `4ef7967`, `86b6419`, and from `e0cc53f` only the adapt-default hunk in `generate.cpp` (the
  commit also carries `--mmvq-exact` and `ab.py` / `h2_nsys.py` / `test_ab_parse.py` changes). `5748dc5` moves the
  4070 swaps (`--adapt-secondary`, `87e9922`) onto their own thread: it is secondary-tier code and belongs to rank 13.
  - Files: `pinned.{hpp,cu}` decommit/commit; `expert_source.cpp:1759-1825`; the paired stages in `generate.cpp`.
  - Test: `tests/xeno/exclusive_host_pages.cpp`.
- **Why merged:** exclusive ownership without paired swaps forces adaptive swaps off (#47 PR 2's text). The pair is the
  usable unit.
- **Who needs it:** RAM-limited single-GPU users (#60, #133, #141).
- **Evidence.**
  - `fdaddbe`: −7.86 GiB private, decode 64.68 vs 61.41.
  - `a9e5c0e`: paired swaps under exclusive primary, code 63.5 → 68.4, Thai 40.1 → 42.3, "matching the non-exclusive
    swap that costs +8.5 GB RAM"; `86b6419`: code 73.91 → 75.37 (run as "paired 8 + 4070 16", so the 4070 tier was on).
  Both are below the 13.6 % gate. (`0a0791d`'s 85.76 vs 84.75 is the 4070 tier, not
    this PR.)
  - Package note: #47's "one card, working set 23.2 vs 33.2 GiB" combines exclusive primary, placement-first, the
    tail file and LAZY (LAZY alone is −1.1 GiB on upstream, m12); #47 says these "prove the package, not each
    mechanism". **U vs U+this PR is UNMEASURED.**
- **Gate:** **opt-in** (our default −1 = on-when-eligible must become off). Outputs identical (`fdaddbe`).
- **Must fix before opening.**
  - Linux: `decommit_interior`/`commit_interior` are Windows-only (`pinned.cu:292`, `:322`). Needs an `madvise`
    path or an OS gate.
  - The Q2_0-only eligibility, which is tied to the P3 parity path.
  - Refactor the duplicated serve and generate adapt blocks (`generate.cpp:4497+` / `:6154+`) into one function.
  - Handle layer split × paired swaps, which skip `res_upload` (cannot tell statically).
- **Risk.** Upstream's #141 position is that duplication is deliberate: pitch it as opt-in for RAM-bound users.
  Overlaps #153's sizing code. **Size:** about 600 lines.

**Rank 8: `experts: placement-first cold start` (#47 PR 4).**

- **Contains:** `1d35b34`, `d4de5a6` (fill timing), `6aee9b9` (pipelined primary fill), `77bbd38` (unbuffered
  reads; it also speeds the 4070 fill 10.9 → 2.0 s, and any 4070 hunk stays out) (`reserve_only`, `read_experts`, `load_rest`, deferred `blob()`). **Not in this PR:** `2bbf77d` (touches only
  `secondary_arena.cpp`) and `9f81cd0` (the 4070 fill pipeline and `SecondaryArena::check_free_floor`): both are
  xeno-only 4070-tier code (rank 13 / §7).
- **Who needs it:** RAM-limited users (boot peak).
- **Evidence.**
  - `1d35b34`: boot peak 46.3 GB → 30.5 GB, "the peak is the steady state". This is the placement-first-specific
    number.
  - `d4de5a6` peak 46.6 → 38.9 GB is against `--no-exclusive-primary-experts`, which also switches off rank 7, so it
    conflates the two.
  - Boot time: the "161 → 18 s" series is mostly the 4070 fixes (the 161 s came from `1d35b34`'s own per-slot NVML
    sampling, 104 s). Primary-only fill 29.3 → 13.4 → 5.3 s (`6aee9b9`, `77bbd38`). **A boot-time benefit against a
    non-placement-first boot is UNMEASURED.** Output md5 identical (`6aee9b9`).
- **Prerequisite:** rank 7.
- **Must fix.** Linux: `commit_interior` in `load_rest` (`expert_source.cpp:1575`). Also, the deferred load reports
  `st.bytes = want` (`expert_source.cpp:1315`), which makes upstream's load-rate hint (`ce8789b`) meaningless.
- **Size:** about 472 insertions for the four kept commits (192 + 13 + 86 + 181); the earlier "577" included
  `2bbf77d` and `9f81cd0`. **Risk:** medium.

**Rank 9: `prefill: tail file for the borrowed cache slots` (#47 PR 5).**

- **Contains:** `497777a`, `48e44be` (`TailFile`, `setup_tail_file`, `refill_lent`).
- **Evidence (`497777a`):** commit 41.16 → 37.90 GiB at 8K; refill 4.8-5.1 GB/s vs 1.13-1.38 from the GGUF.
- **Prerequisites:** ranks 7 and 8.
- **Must address.**
  - D-4: our `refill_lent` replaced upstream's queued refill (§4.2). A port must keep D-4 for the non-exclusive path,
    and must not queue from the reused bounce buffers (B.1).
  - The exit-hang hypothesis B.4a (the tail reader's `thread_local DirectFile`) must be settled first, since it may be
    the cause.
  - About 3.5 GB of disk, and a 2 GiB free-disk guard.
- **Size:** +313/−18. **Risk:** medium.

**Rank 10: small verify and pool trims.**

- **10a. `verify: merge mapped CPU rows only where the GPU missed`** (`94d347d`, `tests/xeno/hit_merge_mapped.cpp`).
  - Who: anyone with CPU-served experts plus GPU hits.
  - Evidence: bitwise equal; a mutant fails. Speed: Thai +3.4 % ABBA (`h2-primary-timeline.md`), **below the 13.6 %
    gate, not established**.
  - Size: about 50 src lines (+48/−2, `git show --numstat 94d347d`) plus 63 test lines. Low risk.
  - Conflict site: upstream open PR #154 also changes `src/kernels/cuda/s2_expert_grouped.cu` (+4/−0), the file that
    holds `moe_hit_merge_mapped` (`gh pr view 154 --repo Niko1221/Strata --json files`).
  - Note: upstream E-6 took `moe_hit_add` in its own branch (`verify.cpp:710-727`).
- **10b. `pool: rest parked workers between verify windows`** (`ba0cade`, `tests/xeno/pool_rest.cpp`).
  - Evidence: `ba0cade`, rest alone at 6 workers: code 80.52 → 83.28 (+3.4 %), thai +0.7 %, below the gate. The
    "+10.8 %" (13 workers + rest vs 6 workers, rest off) includes the worker-count change and is not this PR's effect.
  - **Must fix first:** the rest is never called under `--serve` (only `generate.cpp:6592`, verified). Upstream serves
    by default, so the PR must wire it into the serve loop and be measured there.
  - Size: about 17 lines.
- **10c (optional, Windows-only). `--pool-priority`** (`af4e19e`). Opt-in with default 0, not our default 2.
  Evidence: code 57.0 → 66.8 (`cpu-pool-sweeps.md`).

**Rank 11: `experts: placement-independent Q2_0 output (CPU-order GPU rows) with K0-K3`.**

- **Contains:** `9a2b807` (P3) plus `ab93dfd`, `40d482b`, `cad5a92`, `6dd2867` (P4); the tests
  `native_q2_pool_hit_parity.cpp` and `cache_tokens.py`.
- **Who needs it:** Q2_0 users who want output independent of cache size and placement.
- **Evidence:** 256/256 CPU-only vs GPU-hit on four prompts; 66 cases bit-exact. K0 (ABBA, `q2-rows-per-warp.md`):
  code 45.0 → 61.1 tok/s, thai 32.5 → 35.9 (+10 %). K0 **recovers the cost of P3's own CPU-order kernel** (8-lane rows
  "left 24 of 32 lanes idle", same report); it is not a gain over upstream's kernel. Against upstream the package is
  slower on 5060-only decode (#47: D1 64.6 vs U1 68.4, "U faster on code"); U vs U+this PR is UNMEASURED.
- **Gate.** It **changes default output**, including CPU-only via `exp(double)` (`pool.cpp:419`), so it must be opt-in.
  The CPU `exp` change also needs its own switch.
- **Risk.** Upstream disputes the need (`c1e9033` note: 95-98 % same top-1, equal perplexity). Pitch it as
  determinism. It overlaps gputier's #149 grouped-kernel PR, which the maintainer asked for first: co-ordinate or wait.
- **Size:** about +347/−137 src.

**Rank 12: Claude Code / Anthropic serving parity, as six small PRs.**

| PR | commits |
|---|---|
| (a) `count_tokens` | `e6fd73e`, `5a890f7`, `1920133` |
| (b) billing-header strip | `bc1b18f`, `9ccc276` |
| (c) `stop_sequences` + `signature_delta` | `d399916`, `6bff9c1` part |
| (d) `/health` 503 on a dead engine | `e9c6958` |
| (e) `LoopGuard` | `6bff9c1`, `14c5708`, `0d6536f` |
| (f) PDF `document` blocks | `56881b8`; adds `pypdf`, `pypdfium2`; plus `694ad82` (#46: documents and images inside a `tool_result`), which is **not in HEAD** (§1) |

- **Cross-PR dependencies (verified with `git show --stat`).** The commits cross these boundaries: `1920133` (listed
  in (a)) is "Verify PDF token count over HTTP", a test of (f)'s PDF blocks (`tests/xeno/test_count_tokens.py` +20);
  `d399916` (c) edits `serve/pdf_blocks.py`, which (f) creates; `e9c6958` (d) adds its test to
  `tests/xeno/test_count_tokens.py`, which (a) creates; `6bff9c1` is split across (c) and (e). So either open (f)
  before (a) and (c), or rewrite the cross-cutting hunks and tests for each PR.
- **Later serve work not in HEAD.** `12556b1` (#49 S2: a dead engine fails the request as 529 `overloaded_error`,
  #48 defect 2; `disable_parallel_tool_use`), `e63d359` (S3 thinking budget), `55eb129` (S5 timing block), `c54c337`
  (S6), `80cd0b3` (S8), `1620329` (S4 CJK guard) sit on the local `xeno/claude-merge-0.1.20` only (§1). Which of them
  are upstream candidates is not assessed here.
- **Upstream overlap.** Upstream `cb91f2a` (in merge 2, "serve: keep image parts in tool/assistant messages")
  overlaps #46 / `694ad82`; the PR must build on it.
- **Who needs it:** anyone driving Strata from Claude Code or Anthropic SDKs. None of these exist upstream (grep at
  `a790805`).
- **Evidence:** tests only. The billing strip's prefix-reuse benefit is UNMEASURED. Measure `REUSED` across two Claude
  Code sessions with and without it.
- **Fix before opening (a).** Strip the query before matching (`server.py:1468` vs `:1458`).
- **(e)** changes behaviour on degenerate output, so it should be opt-in. **(f)** adds dependencies, so it should be
  optional.
- **Size:** about 316 src plus 340 test lines in total. **Risk:** low for the engine.

**Rank 13: `multi-GPU: exclusive ownership for --expert-cache-device1..3` (#47 PR 7).**

- **Contains:** the ideas of `5213b90`, `ea04a91`, `0a0791d` (exclusive 4070 plus paired 4070 swaps), and optionally
  `d1431ea` (packed metadata), `acfaa3b` (compact D2H) and `3ec52cb` (CUDA-graph launch), **re-implemented on upstream's
  `RemoteExperts`**, not on our `secondary_*`.
- **Who needs it:** multi-GPU users. Upstream's helper tier "does not release the expert arena in system RAM"
  (`docs/SECOND_GPU.md`).
- **Contains also:** `5748dc5` (the 4070 swaps on their own thread), moved here from rank 7.
- **Evidence (ours):** peak RAM 39.8 → 31.3 GB and code 84.75 vs 85.76 (`0a0791d`, same-session ABBA, 4 runs per
  arm). Package note: #47's "D2 16.5 vs U3 35.3 GiB" also includes exclusive primary, placement-first, the tail file,
  LAZY and, for U3, a different architecture (layer split); it is not this PR's number. **U vs U+this PR is
  UNMEASURED.**
- **Prerequisite:** rank 7.
- **Overlap:** #151 and #110. The maintainer wants smaller PRs.
- **Must drop (xeno-only):** the hard-coded names, ordinal and sm check (`secondary_arena.cpp:59-77`,
  `device_main.cpp:68-77`), the 2560 MiB display floor, the NVML dependency, and the Q2_0-only limit.
- **Size:** large. **Risk:** high.

**Rank 14: `RFC: dual-GPU prompt path — whole MoE layers on the second card, two-lane wave` (#47 PR 8).**

- **Contains:**
  - Split: `c8f12fb`, `31943cf`, `7db2963`, `b9d2f57`, `3744f9a`, `3834af2`, `1bb403f`, `f0db7af`, `e10e27f`.
  - Wave: `cc81498`, `15794d6`, `9bd207f`, `e34a1d0`, `db17e31`, and **`5c51574` (the #48 fix, required)**.
  - Optional frontier: `78f373e` and others.
- **Who needs it:** two-GPU users where the second card has the wider link.
- **Evidence.**
  - Single-variable (use these in the RFC): `b9d2f57` S4b split −14.5 % (11.07/11.10 → 9.41/9.54 s); `cc81498` wave
    −9.4 % (8K, tier 6400, ABBA); #48 split + wave vs split only at 16,405 tokens, 9.90-9.95 vs 12.13-12.16 s with
    identical outputs.
  - Package only (confounded): D2x 8K 4.12 s vs U3 serve 7.3 s and #47's "11.1 → 4.1 s" mix sessions, generate vs
    serve mode, and 8K borrowed chunks vs 2K chunks (§0, §3.2 Q12). Like for like in one session, D's split + wave
    4.1-4.2 s vs upstream's best one card (the 4070) 4.6-4.8 s, about −12 %, below the gate.
  - #48: 47,201-token Claude Code prompt in 26.5 s vs 46.0 s.
- **Form:** **an issue first** (the #110 precedent: "compare with `--gpus` in an issue"). Compare against the layer
  split on the same machine.
- **Prerequisites:** ranks 1, 7 and 13.
- **Blockers:** the generate-mode exit hang; B.4b; mid-prompt checkpoints under the serve wave (§8 item 13, the
  hazard upstream fixed for its layer split in `15345e6`); the cross-session reuse loss (§8 item 2); no unit test for
  WaveLink / ring gating (`9bd207f`); the WDDM cross-card sync design; the `--adapt-secondary` silent-disable trap;
  hard-coded device 1.
- **Size:** about +1,612/−456 src (not re-verified). **Risk:** high: after first landing, two hangs, two crashes and
  one use-after-free (`9bd207f`: one hang class, one crash (`std::terminate`), one use-after-free in serve; `e34a1d0`:
  one hang; #48: one crash, an illegal address).

### 6.4 Dropped: upstream superseded or rejected them

| ours | why dropped |
|---|---|
| K4 `pcie_share` skip (`6790f01`) | upstream rejected the same idea from PR #109 (`1e4515c`'s `--pcie-frac 0` skip) in `9a19f57` (the per-request `pcie_frac=` conflict); ours was never submitted. If kept locally, it needs a guard that refuses `pcie_frac>0` when the share was compiled out |
| `--pcie-frac` default 0 (`704ac58`) | upstream has the probe (`7a4b627`, `d9a4974`); file an issue with the x4 data point instead (§7) |
| batched indexer (`767fcfc`) | upstream C-2 (`758eb1a`); delete our dead overload `native_qsa_indexer.cu:270` |
| batched embedding (`4935d9b` part) | upstream C-4 (`5b7e316`) |
| PLE async (`f7ecc89`), stager (`fdaddbe`) | upstream `928b0e0` |
| copy-issuer (`c7a469e`) | upstream D-5 (`cf68b00`) |
| #42 ports (`14c5975`) | upstream `fe609ce`, `882bb6d`, `b046845` |
| fused dequant (`6ee0274`) | **low value to upstream.** At HEAD `iq_dequant_expert_f16` runs only on the non-MMQ FP16 branch of the streamed walk (`prefill.cpp:2711` `use_mmq`, `:2873` returns first, the call at `:2900`), i.e. with `STRATA_PREFILL_MMQ=0` (`:1185`) or on layers without MMQ. MMQ has been the default for native packs since upstream `928b0e0` (0.1.13). The −2 % was measured before merge 1, single-GPU 2K serve prompt, when FP16 dequant was the default. Its value on upstream's default path is none or UNMEASURED. 0 of 4.9M values differ |
| first-chunk PLE read-ahead in the prompt (`4935d9b` part) | **not dropped; a small candidate**, to be bundled with rank 2 or sent alone after re-measuring (Q6: −3.7 % on O, UNMEASURED on D) |

---

## 7. Not for upstream

| item | why |
|---|---|
| `--pcie-frac` default 0 (P13) | tied to our x4 link; upstream has a generic probe. Offer an **issue**: the probe picks 0.15 on this link (`generate.cpp:1909-1910` comment), and at `704ac58` that stalled decode (3.24 vs 61.92). Upstream's probe with the copy kernel is UNMEASURED here, and the link generation is UNMEASURED |
| the 4070 static tier as written (P7) | machine-bound (names, ordinal, sm 8.9, NVML, display floor); only the ideas go upstream (rank 13) |
| `STRATA_SECONDARY_POKE` (Q15) | WDDM display-card specific; mixed result |
| `STRATA_HOT_TO_SECONDARY` (Q16) | no gain |
| `--process-priority`, `--lock-cpu-experts` | desktop starvation; no gain |
| P-core ordering (P2) | Windows-only and unproven; possible later with a paired measurement |
| NVMe capacity tier (P12) | decode 49.57 vs 84.71; the maintainer prefers a `FileExpertSource` design (#80) |
| timeline (Q1) | large diagnostic (+1,007/−88); offer as an RFC or issue, not a PR |
| route trace, link/copy probes, simulators, stats counters (P16, Q17, Q18) | diagnostics; `--route-trace` and `tier hits` could ride along later |
| logging (Q19) | low priority; could fold into another PR |
| harnesses (P17, Q20) | local measurement tools |
| exit-hang probe (Q22) | an experiment bypass, not Linux-safe (`TerminateProcess` without `#ifdef`, `generate.cpp:6967`) |
| `--cache-cpu-only`, `--secondary-stage-only`, `--secondary-profile-timing` | A/B arms |
| Dm frontier and expert order (Q13) | follow-on to the RFC; speed neutral |
| `--mmvq-exact` | overlaps upstream PR #148 |
| docs, ADR, AGENTS.md/CLAUDE.md (P18) | evidence for PR bodies only |

---

## 8. Open questions and everything unverified

**Defects and hazards (static reading unless noted).**

1. **Generate-mode exit hang** (`7b31aa6`; checkpoint report §4.1). The root cause is unknown. **Hypothesis B.4a:** the
   tail reader's `thread_local DirectFile` owns C-4 pool threads and is destroyed during thread detach under the loader
   lock. Settle it with a CPU-only repro, or `--no-tail-file` with `STRATA_EXIT_TRACE=1`.
2. **Cross-session prefix reuse was lost in one served run with split + wave on** (#48). A new Claude Code session in
   the same folder reused **0** tokens with split + wave on (47,208 read in 26.7 s), where **split + wave off** reused
   **30,791** (17.2 s) (MEASURED, #48 root-cause comment). The arms change two features at once, n = 1 each, and the
   request history before each was not controlled. Not root-caused. Candidate mechanisms from the code (static
   reading):
   - **(lead, wave-independent)** The resume step erases every checkpoint that is longer than `resume` **or is not a
     prefix of the current prompt** (`generate.cpp:5083-5085`, `starts_with` compares token ids, `:5053-5059`), and
     `resume == 0` clears all (`:5088-5096`). Any interleaved request that diverges early (Claude Code's title,
     auto-mode classifier and side requests share the single slot: #49 stories 17 and 24) can therefore wipe the chain
     regardless of the wave. Upstream work on the same problem: #41 (closed; the maintainer asked that side calls keep
     `resume`), #57 and PR #175.
   - **(b)** The pinned conv-cache slot is `checks[0]`, the first inserted. When the system part exceeds 16,384 tokens
     (30,791 here), the mid-prompt checkpoint at 16,384 is inserted first, so the real `root_at` checkpoint is not
     pinned and can be LRU-evicted. Wave-independent.
   - **(d)** `resume == 0` clears every checkpoint (`:5088-5096`). Wave-independent.
   - **(e)** A restore failure exits the engine (`return 1`), losing all checkpoints. Wave-independent.
   - **(a)** and **(c)** (wrong state in a mid-prompt checkpoint; `checkpoint_at` dedupes by length only, `:4384-4385`)
     **cannot produce 0 reuse**: `starts_with` matches by ids, so a checkpoint with bad state would still be mounted,
     giving reuse > 0 and possibly wrong output. (a) is a correctness hazard in its own right: item 13 below.
   - **Instruments in the tree:** `RESUME`/`REUSED`, the part log, `STRATA_STATE_HASH(_GDN)`, `STRATA_CKPT_REREAD=1`.
   - **Proposed A/B:** split on, `STRATA_PREFILL_WAVE` unset vs set, under the same scripted request sequence
     (including the side requests), n ≥ 2 per arm.
3. **B.4b: race on `PleTable::Impl::ahead`** between two wave lanes in serve (`ngram.cpp:133-141`). Possible
   "PleReader: unknown ticket" or UB.
4. **The `pcie_share_` skip × per-request `pcie_frac=`** (§4.2): the PCIe rows would take stale `hit_out` values. Neutral in D2x.
5. **D-4 lost at merge 2**, and `STRATA_REFILL_BLOCKING` is a no-op.
6. **`--pool-rest` and `decay_scores()` are never called under `--serve`** (verified: only `generate.cpp:6592-6593`).
   The `ba0cade` numbers are generate-mode numbers; rest alone was +3.4 % code at 6 workers, and "+10.8 %" includes
   the 6 → 13 worker change.
7. **`count_tokens?query`** would miss the route (`server.py:1468`).
8. **Windows-only decommit/commit on a default-on path** (`pinned.cu:292`, `:322`). **`TerminateProcess` without an
   `_WIN32` guard** (`generate.cpp:6967`). **The AVX-VNNI intrinsic under `-mavx2`** (`CMakeLists.txt:622`). All three
   likely break Linux; no Linux build was done.
9. **Upstream's relaxed floor (sm_80 now, sm_75 in v0.1.27) removes the sm_120-only check** that also stopped the 4070
   becoming primary by accident (`device.cu:72`). The check was upstream's; only the sm_89 display exception is ours.
10. **HIP build** broken by unconditional CUDA links and headers in the whole secondary library: `CMakeLists.txt:161`,
    `:162-165` (NVML), `:312`, `:173`, `:343`, and the `<cuda_runtime.h>` includes in `src/core/secondary_*.cpp`.
    UNMEASURED. Rule 6's fix (`${_strata_gpu_runtime_target}` plus refusing the feature on HIP) must cover all of it.
11. **An engine that dies mid-prompt leaves the stream waiting** (#48 defect 2; #49 S2). Not fixed **in HEAD**. A fix
    exists on the local `xeno/claude-merge-0.1.20` as `12556b1` (a dead engine fails the request as 529
    `overloaded_error`), not merged into this tree (§1).
12. **Upstream behaviour, not ours:** the server spawns the engine without `CREATE_NO_WINDOW` (`server.py:168`; the
    same call is at `4c68013:serve/server.py:166`); our launcher hides the window.
13. **Serve wave × mid-prompt checkpoints: possible silent state corruption** (static reading, UNMEASURED). In a wave,
    lane B's chunk c+1 layer l starts as soon as chunk c's layer l is handed off (`prefill.cpp:2311-2313`,
    `wait_attn` per layer), and lane B waits for the tail only after its own layers (`:3148`). So when lane A's
    `on_chunk` for chunk c runs `serve_chunk`'s mid-prompt `checkpoint_at(done)` (`generate.cpp:4432-4449`), which does
    `cudaDeviceSynchronize` + `checkpoint_save(c, ss, g)` (`:4396`), the in-place GDN state and indexer tails may
    already hold part of chunk c+1. A later request that mounts that checkpoint would continue from wrong state with no
    error. Upstream fixed the same hazard for its own pipelined layer split in `15345e6` (in merge 2: "They were off
    across GPUs because the first stage is a chunk ahead when the last one reports"), with per-stage parts
    (`on_stage_chunk`, `generate.cpp:4374-4378`, `:4434-4446`, `:4461`). The wave path has no equivalent: at `:4446` a non-split run calls `checkpoint_at(done)` directly (`serve_chunk` takes
    the `multi_gpu` branch only for the layer split). It blocks the serve wave. Test: `STRATA_CKPT_REREAD=1` on a
    > 16,384-token wave prompt. Fix: follow `15345e6`, or skip mid-prompt checkpoints while a wave part runs.
14. **#46 is not in HEAD:** PDFs and screenshots returned inside a `tool_result` are dropped at `5c51574`; the fix
    `694ad82` is only on `xeno/claude-merge-0.1.20` (§1).

**Numbers that are UNMEASURED or not transferable.**

- **The isolated AVX-VNNI kernel speed** (PR 9 gate).
- **Carried over from before merge 2.** These gains were measured before merge 2 and are unmeasured on the merged
  engine:
  - decode PLE read-ahead (Q14);
  - first-chunk PLE read-ahead (Q6);
  - group gather (Q8);
  - fused dequant (`6ee0274`).
- **Merge-2 defaults.**
  - The issuer default flip on our link.
  - Upstream's PCIe probe plus the copy kernel on the x4 link.
- **Stream-k.** nsm=1 speed on high-SM cards.
- **What moved D's hashes against O's.** PR #109's combine, cache size, or something else. Isolate with
  `STRATA_DEC_BATCH=0` and a fixed `--expert-cache` on both arms.
- **The one-card −16 % prompt gap** (D 1,205 vs U 1,439 tok/s; checkpoint report §3.3). Not isolated.
- **Why the 2-chunk and 4-chunk 8K reads differ** (`705e4c9d` vs `4b0bfd47`). The trunk's floating GEMMs are the
  candidate.
- **Whether `--short-read 64` still diverges from 0.**
- ~~Whether D2aw ran the wave~~: settled, it did not (m10 logs, §3.2 Q12).
- ~~"D2 18.3 s"~~: settled, it is D2a in #45 m10 (§3.2 Q12).
- **v0.1.27's `draft_vocab.bin`:** its effect on thai (no Thai script in the subset), and English −1-2 %.
- **Combining helper caches (`--expert-cache-device1`) with our 4070 tier.**
- **Layer split × exclusive primary + paired swaps** (`res_upload` skipped).
- **The billing-strip latency benefit.**
- **Where the ranked profile came from.** Its name suggests EXL3 router counts; UNVERIFIED.
- ~~Whether upstream #16 was hand-merged~~: settled. `a4a5a37` is authored by Daerdaal and committed by Niko1221
  (2026-09-28 22:05 +0200, parent `c1e9033`); the PR was closed unmerged at 2026-09-28T20:45Z
  (`gh api repos/Niko1221/Strata/issues/16`). The maintainer applied it by hand.

**Not verified by the fix pass (open).**

- The sizes of ranks 12, 13 and 14 ("about 316 + 340 lines", "large", "+1,612/−456").
- The maintainer quotes attributed to upstream #80, #110 and #120.
- §2.3's open-PR overlap table beyond the PR titles, and §4.4's Part C table row by row.
- §5 in general (the blueprint diagrams and boot-order line ranges), except the lines corrected in the log below.
- Static reading, not checked: `excl_keep_from` (the lendable tail and the tail file) is found by halving
  `o.prefill_chunk` with only `k + 128 ≤ slots` (`generate.cpp:2796-2806`), while serve's `plan_lend` with `--prefill
  auto` tries {8192, 6144, …} under an 85/90 % cap (`:4131-4153`). The comment says they match; if they choose
  different chunks, lent slots below `excl_keep_from` would have no host copy and no tail-file entry, and `refill_lent`
  would read them from the GGUF (1.13-1.38 GB/s per `497777a`).
- #47's own tables carry two items this report corrects: the PR 3 row cites `0a0791d` (a 4070-tier result) and the
  dossier's CPU row lists #103 (a Linux Colab host that AVX-VNNI would not reach). #47 should be updated.

**Docs gap.** The merge checkpoint report (`docs/reports/2026-09-30-upstream-0.1.26-merge-checkpoint.md`) exists only
at `070d1b8` on `xeno/claude-merge-0.1.20`. It is not in the `5c51574` tree. Many citations in this report depend on
it.

---

## 9. Verification log

Two adversarial verifiers checked this report; a fix pass re-checked every item against git, the code at `5c51574`,
the `C:/Strata-exp` logs and the issues (read-only). **80 items checked** (55 claimed errors, 25 gaps). **54 errors
corrected, 1 rejected** (the verifier was wrong); **20 gaps added**, 2 moved to §8, 3 needed no change. Duplicate
findings from the two verifiers are merged into one line below.

- §1: `xeno/claude-merge-0.1.20` is not an ancestor of HEAD; it holds `694ad82` (#46) and the #49 serve series, none in HEAD. The ancestor list was completed.
- §3.2 Q11, §5.7: the 8,000-pair profile is the stale v0.1.2 blob in `D:/Github/Strata`; HEAD ships 24,576 pairs; 1,761 = 8,000 − 6,239. D2x's profile file and hash recorded.
- §3.2 Q12, §8: "18.3 s" is D2a in #45 m10; "11.1 → 4.1 s" pairs m10 with m11.
- §3.2 Q12: D2aw ran split only, now MEASURED from the m10 logs, so "11.1 → 4.1 s" is not the PR 8 effect.
- §0, §2.3, rank 14: D2x vs U3 relabelled cross-session, cross-mode and chunk-confounded; like-for-like −12 % (below the gate) and #47's "U's best one card 1,700 tok/s", D4 vs U4 and determinism rows added.
- Rank 1: `31943cf` moved out (it needs `mmq::expert_rows` and rank 5's `gather_q8_rows`).
- Rank 1: the "upstream documents multi-GPU non-identity" and single-card run-to-run claims removed; D3u is our own measurement.
- Q2, rank 1: `STRATA_MMQ_STREAM_K` restores stream-k at any value; the proposed `=0` switch withdrawn.
- Q3, rank 3: `f4ce2bd`'s TTFT is an EAGER + warm-up measurement; LAZY + warm-up on upstream marked UNMEASURED.
- P9, rank 7: `0a0791d` is 4070-tier evidence; rank 7 now cites `a9e5c0e` / `86b6419`; #47 PR 3 row flagged.
- Ranks 7, 13: #47's package RAM numbers moved to package notes; U vs U+PR marked UNMEASURED.
- P11, rank 8: `2bbf77d` and `9f81cd0` moved out as 4070 code; `1d35b34`'s boot-peak evidence used; the boot-time benefit marked UNMEASURED; size corrected to about 472.
- P6, rank 10b, §8.6: rest alone is +3.4 % at 6 workers; +10.8 % includes 6 → 13 workers.
- Rank 11: K0 described as recovering P3's own cost; thai +10 % added; the package against upstream is slower or UNMEASURED.
- P14, §6.4: fused dequant runs only on the non-MMQ path at HEAD; its value to upstream's default path is none or UNMEASURED.
- Ranks 2, 5, 6, 10a, §6.2: sub-gate speed deltas marked "not established".
- §6.2: "developer decision" labels replaced with their actual sources or "proposed here".
- Rank 12: cross-PR commit dependencies listed; `694ad82` and the later serve commits added.
- P1, rank 4: #103 removed as a beneficiary; the "codex using ~3.8 cores" note and the 2-of-4 no-gain prompts restored.
- §0, §6.4: K4 is "the same idea from PR #109" that upstream rejected, not our commit.
- §8 item 2: the arms are split + wave on vs off, n = 1; the non-prefix checkpoint erase (`generate.cpp:5083-5085`) added as the leading wave-independent candidate; (a) and (c) cannot give 0 reuse; §5.4's erase rule completed.
- §8 item 13 (new): the serve wave's mid-prompt checkpoints are a static-reading corruption hazard; upstream fixed the same hazard for the layer split in `15345e6`.
- Ranks 5, 6: unrelated content in `c8f12fb`, `f801c5d` and `cb23774` split out.
- Rank 7: `5748dc5` moved to rank 13; only the adapt-default hunk of `e0cc53f` kept.
- Line and size cites corrected: `pool.cpp:63-68` / `:71-105`, `generate.cpp:880`, `expert_source.cpp:1315`, rank 10a about 50 src lines, `prefill.cpp:1626`, `generate.cpp:4396`, `expert_source.cpp:905-913`, `frontend.py:184`, the boot-table MTP row `:2330-2348`.
- Q14: `--ple-delay-us` is upstream's flag.
- Rank 14: the risk count is now two hangs, two crashes and one use-after-free.
- §2.4: `--stats` cannot count Thai; how to count it added.
- Q6: −6.9 % is Q5 + Q6 + Q8 together.
- §4.1: merge 1 dropped upstream `61126e9`'s AVX2-only notice (log only); `cpu_require_expert_support` was never absent. The merge's verification and non-conflict edits added.
- §4.3 B.0: the `ms_prefill` "benign race" removed (serialised by the WaveLink mutex; #48 H10).
- §4.3 B.3: with an exclusive 4070 tier, helper caches most likely fail at boot (static reading); duplication applies only to a non-exclusive tier.
- §4.2 Suspicious 1: the rows take stale `hit_out` values rather than being dropped; `kind=1` is at `:852`.
- §4.2, §8.10: every HIP-breaking site listed.
- §5.7: a table of default-on and flagless behaviour changes against `4c68013` added.
- §3.2 D3.10: recast as an upstream rule (`928b0e0`) that our wave depends on, which also caps U3 at 2K chunks.
- §2.2: endpoints attributed to `d9fba59`, `89e68f6` and `28e6c8e`.
- §2.3 D1: `STRATA_GROUP_COPY` / `STRATA_GR_UNFUSED` are upstream's.
- §4.2 Suspicious 2: "pinned arena" removed from the `581765a` citation.
- §4.2, §8.9: the sm_120-only guard was upstream's; only the sm_89 exception is ours.
- §8.12: `CREATE_NO_WINDOW` labelled upstream behaviour.
- §8: upstream #16 settled (hand-applied by the maintainer).
- §5.5: the "MTP ~0.9 GB" figure marked UNMEASURED.
- **Rejected:** §3.1 P6 "Process HIGH (`0a0791d`)". The report already cites `0a2d70d`, which is where the 76.76 → 78.73 figures are (`git log --all --grep=76.76`).
- **Gaps added:** hang-log evidence for B.4a; ggml heuristic, global device table and HIP one-block risk for rank 1; `15345e6`; the checkpoint erase rule; `-DSTRATA_BUILD_TESTS=OFF` and `3801f86`; U1L's `050eee61` against the combine attribution; v0.1.27 commits `da77db6`, `319e4ef`, `aaaafc6`, `aad5bb1`; merge-2 commits `cb91f2a`, `9295f64`, `d6ff0b8`, `61833e2`; PR #154's `s2_expert_grouped.cu` hunk; the D2x flags, cwd and comment; `TailFile` EOF raised to medium; the B.4b stats-counter nuance; the warm-up as a separate candidate.
- **Gaps moved to §8:** the unverified sizes, quotes and tables; the `excl_keep_from` vs `plan_lend` chunk mismatch.
- **Gaps with no change:** confirmations of load-bearing claims and SHAs; the uncommitted #49 S2 edits in the worktree, now committed as `12556b1`; the `--secondary-free-floor-mib` help text, which matches its default of 2560 (`generate.cpp:235`, `:511`) and is not stale.
- **Tooling note.** A copy of `generate.cpp` staged under `/tmp` during this pass was overwritten by another process. Every `generate.cpp` line cited above was re-checked against `git show 5c51574:src/program/generate.cpp` (6,969 lines).
