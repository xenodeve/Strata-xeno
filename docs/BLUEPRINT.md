# Strata-xeno system blueprint

How the engine and the server fit together: the processes, the life of a request, the memory hierarchy, the
checkpoints, every configuration surface and the diagnostics. Read it before changing anything structural; a new
engineer or an upstream reviewer should be able to start here.

**Keep this file current.** The rules are in [Keeping this file current](#keeping-this-file-current) below and in
`AGENTS.md`. A blueprint that describes last month's code is an instrument that returns a plausible wrong answer.

## Baseline and revision log

The sections below describe the code at the **baseline**. `file:line` references are at that commit unless a line
names another. Changes that are on other branches and not yet in the baseline are listed in
[Not yet in the baseline](#not-yet-in-the-baseline).

| date | baseline | what changed in this file | verified by |
|---|---|---|---|
| 2026-09-30 | `5c51574` on `xeno/exp-upstream-0.1.26-dyn` (upstream 0.1.26 + our dynamic experts + the #48 fix; the engine the D2x server runs on :8091) | extracted from `docs/reports/2026-09-30-fork-delta-and-blueprint.md` §5 (`534ffd3`) | two adversarial verifiers and a fix pass (that report's §9) |
| 2026-09-30 | the merge of `xeno/claude-merge-0.1.20` (`f0f5d7a`) into `xeno/exp-upstream-0.1.26-dyn` (#49 S7 prerequisite) | the line now carries every row of "Not yet in the baseline"; the rows stay listed until a build of this merge re-checks their `file:line` | not built yet; `pytest serve` 102 passed |
| 2026-10-01 | unchanged (the 0.1.30 merge `eb2ec1f` is not served yet) | the merge's row under "Not yet in the baseline"; its record is `docs/reports/2026-10-01-merge-upstream-0.1.30.md` | serve ABBA 20/20 identical outputs, `check_cache_slots` PASS (#56) |

**Labels.** `UPSTREAM` means byte-identical to upstream `4c68013`. `XENO` means a file new in our delta. `MIXED` means
an upstream file we modified. "Static reading" marks a statement from the code that no run has confirmed.

**Branches.** The baseline engine is the merged one (0.1.26-based). The daily branch `xeno/claude-merge-0.1.20` is
0.1.20-based: its engine differs as the fork-delta report §4 records, and its `serve/` has the work listed under
[Not yet in the baseline](#not-yet-in-the-baseline).

`UPSTREAM` means byte-identical to `4c68013`. `XENO` means a new file in our delta (145 files). `MIXED` means an
upstream file we modified (45 files). Source: `git diff --name-status 4c68013 5c51574`.

## 1. The model served (`include/strata/core/layout.hpp:27-58`)

- **Model:** Qwen3.8-Flash-Next, 48 layers.
- **Attention:** every 4th layer is full attention ("QSA", 12 layers); the other 36 are Gated-DeltaNet ("GDN").
- **Width:** `n_embd` 2560, hyper-connection `hc` = 4.
- **MoE:** on every layer, 512 experts, top-10, `n_ff` 640.
- **PLE:** a per-layer n-gram embedding table in GGUF shard 2 on SSD.
- **Drafter:** one MTP draft layer.
- **Sizes:** an expert blob is 1,382,400 B (`generate.cpp:652`); the Q2_0 arena is 34.0 GB (`setup.py`).

## 2. Process picture

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

## 3. File responsibilities

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

## 4. The life of a request

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
  interleaved request that diverges early (a Claude Code side request) wipes the chain (fork-delta report §8 item 2).
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
2. `ver.run(T, …)`: host dispatch per layer (§5); the 4070's partials and the CPU rows merge.
3. Accept the longest matching prefix.
4. Start the adapt thread.
5. `ver.commit(a+1)`, then `ple_ahead.push` [XENO].
6. Emit `T` lines.
7. `mtp.draft`, whose `on_draft` does the PLE prefetch [XENO].
8. Join adapt; check EOS or STOP.

The window is up to 6 tokens with `--spec 4` plus the suffix drafter (`:1785-1788`).

## 5. Memory hierarchy and decode dispatch

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

**The CPU pool under `--serve`** (`docs/reports/2026-09-30-cpu-pool-and-kernel-survey.md` §2): an idle worker spins
`_mm_pause` for 20 ms (`pool.hpp:167`, `STRATA_POOL_SPIN_US`) before it sleeps, and the serve loop never calls `rest()`
(only generate mode does, `generate.cpp:6592`), so during decode the 13 workers spin at priority 15 (HIGH class +
HIGHEST) on logical CPUs 2, 4, 6, 8, 10 and 12-19, with the host on 0. Between requests and during batched prompt reads
they sleep.

## 6. KV cache and checkpoints

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

## 7. Configuration surfaces

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
| copy-issuer thread on (upstream's D-5 default, kept) | `prefill.cpp:2250-2257` | `STRATA_PREFILL_ISSUER=0` (per the fork-delta report §4.3 B.2) |
| AVX2-only kernel notice for non-Q2_0 native packs dropped (log only) | `generate.cpp:1973-1978` | none (fork-delta report §4.1) |

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

## 8. Diagnostics

- **Per request:** a part log, request metrics by tier, decode hit rate.
- **At boot:** `mem_mark`, placement-first timings, tail-file lines.
- **Timeline:** `STRATA_TIMELINE`, read with `tests/xeno/perf/timeline.py`. `AGENTS.md` requires recording one timeline
  rather than hunting stage by stage.
- **Tests:** XENO C++/CUDA tests exist only with `STRATA_BUILD_XENO_TESTS` on Windows. #45 reports "Our Python tests
  pass (93)" on the merged tree.

## Not yet in the baseline

Work committed on other branches that changes what this blueprint describes. When the baseline moves past it, fold it
into the sections above and delete the row.

| branch, commit | area (section) | what changes |
|---|---|---|
| `xeno/claude-merge-0.1.20` `694ad82` (#46) | request life (§4), `frontend.py` | a `document` or `image` inside a `tool_result` reaches the model (`_tool_result_content`); images stay image items with vision, else a note |
| same, `12556b1`, `284bb19` (#49 S2, #48) | request life (§4), engine protocol | `StrataEngine.generate` polls the process at every quiet heartbeat (`QUIET_S`) and in the STOP-and-drain; a dead or restarting engine is `529 overloaded_error` on `/v1/messages`; each stdout pump owns its process and queue; `disable_parallel_tool_use` |
| same, `e63d359`, `284bb19` (#49 S3) | request life (§4) | thinking budget: `Service._generate` stops the engine once the budget is spent inside the thinking block, then continues from prompt + written + drained tokens + CLOSE; the DONE figures of both calls are merged; non-streamed requests are side requests (low effort, 1,024 budget, `STRATA_SIDE_BUDGET`) |
| same, `55eb129` (#49 S5) | diagnostics (§8) | the llama-server-shaped timing block per request (`serve/timing_line.py`); `STRATA_TRACE_SSE=<file>` |
| same, `c54c337`, `284bb19` (#49 S6) | request life (§4) | the FIFO lock became `RequestGate`: streamed requests (priority 0) before non-streamed (1); image encoding takes the slot at the request's priority |
| same, `80cd0b3` (#49 S8) | configuration (§7) | `"sampling": {"preset": "deterministic" / "balanced" / "reasoning"}` in the run config |
| same, `1620329`, `284bb19` (#49 S4, server) | configuration (§7), request life (§4) | `serve/cjk_guard.py`: `"cjk_guard": true` writes the Han ids next to the config and starts the engine with `--ban-ids`; `ban=1` per request when the current turn has no Han and does not name Chinese; `cjk_chars` in `/metrics` |
| same, `5f12d89` (#49 S4, engine; built and checked: `sampler_parity`, engine seam check) | engine protocol, decode (§4, §5) | `--ban-ids FILE`, INFO `ban=<count>`, GEN/GENI key `ban=1`; `SamplerParams.ban` (a device bitmap; the sampler kernels are templated so no ban is the old code); the verifier re-picks a banned request's window |
| `xeno/exp-upstream-0.1.26-dyn`, upstream PR #175 (`9bc9a8e`, `fcf93ec`, ported; #49 S7) | engine protocol and checkpoints (§4, §6) | the GEN key `cache_slot=0..3` (the server sends `strata_cache_slot`, default 0): an inactive slot's positional KV, recurrent state and prefix checkpoints are snapshotted to a delete-on-close file and restored on return; INFO `cache_slots` / `cache_storage`; layer-split engines advertise one slot. The server picks the slot itself (`CacheSlots` in `serve/server.py`): a prompt family (its first 512 tokens) keeps its slot, a new family takes a free one or the least recently used; a client's own `strata_cache_slot` wins |
| same (#49 S7 follow-up, decided by the developer 2026-09-30) | request life (§4) | the classifier's fast stage: when the last user message (thinking off) says the reply "MUST begin with <X>" or "Respond with <X>... ONLY" (the real fast stage: `<severity>N</severity>`), the server writes `<X>` for the model and the model continues it (`serve/forced_opening.py`, `_opening` in `Service._generate`); such a request, when greedy, is answered again from its last answer instead of generated again (`Service.replay_key`, 4 kept; never for any other request, so a repeated benchmark prompt is always timed). `STRATA_DEBUG` also logs, per request, the tokens it shares with its family's last prompt and its turn boundaries. The server sends that shared length as the GEN key `ckpt_at=P`, and the engine reads the prompt in one more part (`prompt part shared`) and keeps a checkpoint at P when it lies between the resume point (or the root) and the last turn start: a transcript that grows inside one message (the classifier's) had no checkpoint past the root |
| same (2026-09-30, from live use) | process start, request life (§3, §4) | the port is bound before the model loads: `loading_server` answers every POST with 529 `overloaded_error` and every GET with 503 until `serve()` takes the port (a refused connection made Claude Code back off, once to "retry in 29m"; EXL3's `a04b381` did the same). A side request (`think_budget.is_side`) is now non-streamed **and without tools**: Claude Code resends a main turn without streaming after a failed stream, and that one's thinking was closed at 1,024 tokens |
| same, `6f646e8` (#55 W7, 2026-10-01) | model (§1, the drafter), setup | `data/draft_vocab.bin` (the MTP draft head's token subset) gains all 5,741 Thai tokens: 40,525 -> 46,252 ids, draft head 68.0 -> 77.6 MiB. The old subset held 14 Thai tokens, so Thai answers drafted almost nothing; Thai decode +20-35 %, the same output. `tools/draft_vocab.py` (upstream 0.1.27 `319e4ef`) with a `thai` script. Since the 0.1.30 merge (#56), setup's `refresh_draft_vocab` (upstream 0.1.27) installs it at setup and at every start over a shipped subset (the old 40,525-id one included); `--draft-vocab en` (`data/draft_vocab_en.bin`) keeps the English/code one; a subset made by hand is kept |
| `xeno/exp-upstream-0.1.30-dyn` (#56, 2026-10-01): upstream v0.1.30 merged into the 0.1.26 line | every section: the engine, the sampler, the prompt path's loans, setup | upstream 0.1.27-0.1.30 under the fork: sampled top-k split across the GPU (#197; the CJK ban ported into its kernels), the multi-GPU session carve and per-stage prompt loans (#216; a layer split with prompt borrowing is refused, as the fork's wave lanes live in CUDA0's loan only), the conversation cache (#189, opt-in, off), rope scaling (#84, opt-in), idle unload (#208; `RequestGate` gained a non-blocking acquire), 1024-token streaming off the split layout (the split layout keeps 2048), #210 tool-call parsing, #154 fixes. The fork keeps its MTP/arena block (moved after upstream's session allocation), its lending and wave, its cache slots (now carrying the checkpoint's `dead`/`block_pos`) and `serve_fatal()`. Greedy output identical to the 0.1.26 line on the #54 set. Full record: `docs/reports/2026-10-01-merge-upstream-0.1.30.md` |
| `xeno/11-capacity-iq` `e779f5c`, `684ff15`, this commit (#11, 2026-10-01) | expert placement and tiers (§5), configuration (§7) | `include/strata/core/placement_formats.hpp`: `all_q2` / `all_native` from the layout's formats and `iq_supported`. The automatic exclusive-primary default stays all-Q2_0; exclusive primary, and with it placement-first and the `--ram-cache-gib` NVMe tier, is allowed on any native pack when requested by `--exclusive-primary-experts` or by `--ram-cache-gib` itself (`generate.cpp` eligibility block). `--ram-cache-gib` without placement-first now stops with exit 2 instead of being ignored. The 4070 tier takes any native pack too: `SecondaryRunner` quantizes the activations as the 5060's verify path does (plain q8_1 unless Q2_0/Q2_0) and keeps one table of captured graphs per layer format (`xeno_secondary_iq_parity`: 0 differing floats on 5 layers, all 7 formats); every 4070 swap path (paired, non-paired, serve) skips a newcomer larger than its victim's slot (`SecondaryArena::fits`), since its pairs span layers and native blobs differ per layer. i-quant output is not run-to-run bit-reproducible (#61) |
| `xeno/11-multi-nvme` (#62, 2026-10-01) | expert placement and tiers (section 5), configuration (section 7), diagnostics (section 8) | `--expert-mirror DIR` (repeatable): byte-identical copies of the expert files on other drives. `ArenaExpertSource::read_experts_to` sends each NVMe-tier miss, all its ranges, to the copy with the fewest bytes queued in the batch; a copy is checked on first use (size, first/last MiB, 8 sampled pages) and a mismatch fails the read. Per-file stats (`nvme_file_stats`): the `nvme file` lines (generate) and `strata serve: nvme file` (serve). Every read through `read_experts_to` uses the copies: NVMe-tier misses, the placement-first boot fill, the tail file and the lent-slot refills, so a wrong mirror fails at boot. The reads are polled per copy, so each copy is timed on its own (a `nvme copy read` timeline span per copy and batch, per-read p50/p99/max). Without `--ram-cache-gib` the flag stops the run. Not routed: the single `materialize` (`read_expert`), the prompt path (`read_into`), `load_rest`, the PLE reader. Crash fix in the same branch: `ArenaExpertSource::hold` keeps a paired swap's gathered newcomers resident until their copies are done (`evict_one` skips held experts) |
| `xeno/11-multi-nvme` (2026-10-01) | diagnostics (section 8) | `src/platform/crash_report.cpp`: `main()` installs an unhandled-exception filter that prints the exception, the faulting module + offset and a DbgHelp-symbolized stack (`strata crash:` lines) to the engine log, writes `strata-crash-<pid>.dmp` in the working directory, and lets the exception continue (exit code and WER unchanged). `STRATA_TEST_CRASH=1` crashes on purpose. Added after a served engine died with 0xC0000005 and left only its exit code |
| `xeno/84-n0-trace` `f15af2a`, `c64dfb2`, `a4d1d89` (#85, #86, PRD #84, 2026-10-02) | diagnostics (§8) | `--dump-routing` writes through `include/strata/core/routing_trace.hpp` (format 2), in the existing record shape (negative layer = tag), so `tools/make_profile.py` skips the tags: a format tag (`-2`) first, then the GPU-owned set (`-4`, `ArenaExpertSource::owned_by_gpu`) and the host tier after the boot fill (`-6`, capacity mode) - a boot snapshot, adaptive swaps are not traced - a request tag (`-3`) per served GEN, and after each verify window a commit tag (`-1`: window, positions, accepted, phase 0 prompt windows / 1 decode) from `trace_commit` at the generate loop and both serve loops; tests `xeno_routing_trace`, `tools/test_make_profile_tags.py`; N0 v2 replay `tests/xeno/perf/n0sim.py` (`tests/xeno/test_n0sim.py`) |
| `xeno/84-n0-trace`, this commit (#95, 2026-10-02) | decode dispatch (§5), configuration (§7), diagnostics (§8) | a layer's NVMe-tier misses overlap the CPU pool: `expert_pool_dispatch_multi` (`expert_source.cpp:1169-1291`) submits the misses with `ArenaExpertSource::materialize_begin` (`:2193`, `host_mu_` held in `mat_lock_` until `materialize_end` `:2229`), runs the resident experts, then `materialize_end` publishes the reads and a second pool run serves the deferred entries; an `EndBatch` guard (`:1201`) ends the batch on every early return. `read_experts_to` = `submit_reads` (`:1877`) + `collect_reads` (`:1942`); `materialize_batch` = begin + end. Default on; `STRATA_NVME_OVERLAP=0` restores the serial order (§7 fork-delta row when folded in). `nvme tier ... ms/round waiting` (`generate.cpp:7736`, was `reading`) now counts the exposed wait only; timeline spans `nvme collect` and `cpu pool misses` (`:1286`), and `4070 finish` starts after the CPU work (`:1330`); `dispatch jobs` no longer holds the read. Test `xeno_nvme_overlap`; measurements on #95 |

## Keeping this file current

**Update this file in the same commit as the change** when a change does any of these:

1. adds, removes or renames a process, a thread that does work every request or every round, or a file with a
   responsibility of its own (§2, §3);
2. changes the life of a request: the HTTP path, the engine protocol (`GEN` keys, reply lines), prefix reuse, the
   prompt parts, the decode round (§4);
3. changes where experts live or how decode dispatches them: tiers, profiles, swaps, the CPU pool (§5);
4. changes the KV cache or the checkpoints (§6);
5. adds, removes or changes the default of a flag, a JSON config key or an environment variable (§7);
6. adds or changes a diagnostic: a log line, a counter, a timeline span, a test harness (§8);
7. merges upstream or moves the served engine to another branch or commit (the baseline).

**How:**

- Edit the section the change belongs to. Keep every statement tied to a `file:line`, a commit or a log line; mark a
  statement no run has confirmed as "static reading".
- A change on a branch that is not the baseline goes into [Not yet in the baseline](#not-yet-in-the-baseline) first,
  one row per commit or feature, and moves into the sections when the baseline includes it.
- When the baseline moves (a merge, a new served engine), re-check every `file:line` that the moved code touches,
  then add a row to the revision log with the new commit and what was re-checked.
- Do not copy measurements here: link the issue or report that holds them. This file says how the system works; the
  register and the reports say how fast it is.
- When a change in the list above needs no edit, the commit message says why in a trailer:
  `Blueprint: n/a - <why>`. `tools/hooks/commit-msg` (`git config core.hooksPath tools/hooks`) refuses a commit
  whose diff adds or removes an engine flag, a GEN key, a `STRATA_*` variable, a run-config key or a source file
  without this file or that trailer.
