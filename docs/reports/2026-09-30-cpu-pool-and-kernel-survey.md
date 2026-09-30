# CPU expert pool and Q2 kernel survey: why the desktop stutters, and what it would take to run on fewer workers (#51)

**Date:** 2026-09-30. **Issue:** #51 (asked by the developer: fewer than 13 pool workers without losing decode tok/s, because
the workers compete with the Windows desktop and it stutters). **Scope:** read-only. Nothing was built, and nothing was run
against the live server. **Code:** `C:/Strata-exp/src-dyn` at `5c51574` (the merged engine the D2x server runs on :8091);
the daily branch has the same pool code except where noted.

**How this was made.** A workflow (`wf_0ce748e8-824`) ran four read-only surveys: idle behaviour, thread placement and
priority, the per-expert kernel cost, and the gap to upstream's pool. A fifth agent then checked their claims against
the code and existing artifacts, and ranked a plan. The key claims were checked by hand afterwards (`pool.cpp:273-292`,
`pool.hpp:167`, the single `pool.rest()` call site, the D2x config's priority flags). The raw survey texts are kept
verbatim in the appendices. Where a survey and the checker disagree, the checker's correction in §7 wins.

**Labels.** MEASURED means taken from an existing log, trace, disassembly or system query, with its source. STATIC means
read from the code. UNMEASURED means no run has confirmed it. Nothing here passed a same-session ABBA for the goal.

## 1. Answer in one paragraph

A faster kernel **alone gives no CPU back to the desktop**. During decode under `--serve`, all 13 workers spin at
priority 15 between their short bursts of work (a worker is busy about 16 % of a round). A faster kernel makes the
bursts shorter and the spinning longer. What frees CPU for the desktop is idle workers that sleep or yield, core parking
not hiding the free logical CPUs, or fewer pinned workers. The kernel's job is tok/s per worker, which is what makes
fewer workers affordable. The stutter itself has three candidate causes, and only one of them involves the pool.

## 2. What the pool does under `--serve` (verified)

- **The park loop** (`src/kernels/cpu/pool.cpp:273-292`) spins on `_mm_pause` and reads the clock once every 1,024
  pauses. It never yields. After `kSpinBeforeSleep` = **20 ms** (`include/strata/kernels/cpu/pool.hpp:167`;
  `STRATA_POOL_SPIN_US` overrides it, read once at `pool.cpp:203-204`) it sleeps on a condition variable. The 20 ms
  restarts at every park, i.e. after every phase.
- **`rest()` skips the spin, but `--serve` never calls it.** The only call is `generate.cpp:6592`, inside the plain
  generate decode loop. The serve windows (`generate.cpp:5184` short prompt parts, `:5447` decode) do not call it
  (daily branch: the same single call at `:5636`). The `--pool-rest` help text ("sleep between verify windows",
  `:531`) is wrong for serve. The commit that added it (`ba0cade`) only touched the generate loop, so the omission looks
  accidental, not a decision.
- **Consequence by phase** (STATIC plus the live log):

| phase | pool runs? | worker state | CPU |
|---|---|---|---|
| waiting for the next request | no | spin 20 ms, then asleep | about 0 after 20 ms |
| batched prompt read | no (`prefill.cpp` never uses `ExpertPool`) | asleep | about 0 |
| short prompt parts (<= 64 tokens, through windows) | yes | spin or work | 13 workers + host at 100 % |
| decode | yes | spin or work for the whole decode (gaps between windows are a few ms, under 20 ms) | 13 workers + host at 100 % at priority 15 |

- **Placement** (MEASURED, a `GetLogicalProcessorInformationEx` query): P0-P5 are logical {0,1} ... {10,11}
  (EfficiencyClass 1, SMT); E0-E7 are 12-19. `physical_cores(true)` sorts P first and drops the first core
  (`pool.cpp:32-108`), so the 13 workers are pinned (`SetThreadAffinityMask`, `:115`) to **2, 4, 6, 8, 10 and 12-19**;
  the host is pinned to 0 (`session.cpp:516-518`). Only the six SMT siblings 1, 3, 5, 7, 9, 11 are left.
- **Priority:** the D2x config passes `--pool-priority 2 --process-priority 2` (verified in
  `strata-flash-next-d2x.json`): HIGH class (`generate.cpp:2439-2441`) + THREAD_PRIORITY_HIGHEST per worker
  (`pool.cpp:221`) = base priority **15**, the top of the non-realtime range (a Windows fact). `--pool-priority` also sets
  the **host** thread (`generate.cpp:2444`), so any priority arm moves the critical-path host too. Upstream `4c68013`
  has neither option.
- **Core parking is active** (MEASURED, read-only): the plan is Ultimate Performance with "core parking min cores" 4 %
  (Class 1: 0 %). A sample of `\Processor Information(0,*)\Parking Status` while the engine was idle showed 14-16 of the
  20 logical CPUs parked, including siblings 1, 3, 5 and 7. Whether the siblings stay parked during decode is
  UNMEASURED.

## 3. Where a round's time goes (existing artifacts)

**Timeline `tl1`** (`C:/AI/UsersxenodAppDataLocalTempstrata-claude-stagetl1.json`, 13 workers, generate mode, rest on,
16 rounds, 2026-09-29; `tl1b` agrees within 1-3 %; exe and config were not recorded):

| measure | value |
|---|---|
| round / verify window | 34.50 / 31.12 ms |
| pool phases per round | gate/up 4.463 + quantize 0.114 + down 2.256 = 6.83 ms (19.5 % of the round) |
| gate/up phase | mean 97.7 µs, p50 78.4, p90 190.8 |
| down phase | mean 49.4 µs, p50 39.3, p90 95.9 |
| wake while spinning | P p50 0.3 / p90 0.5 µs; E 0.5 / 0.6 µs |
| wake after sleep | P p50 56.7 / p90 93.3 / max 362 µs; E p50 70.6 / p90 153.5 / max 275.5 µs |
| worker utilisation inside phases | 0.819 (the rest is tail imbalance) |
| gaps between layers inside a window (n = 715) | p10 254, p50 372, p90 571, p99 793, max 1,059 µs |
| gaps between windows | p10 4.18, p50 6.74, p90 23.6 ms |

Sleep time per round if a worker slept after spinning T µs: T = 200 → 17.4 ms asleep of 34.5; 500 → 9.27;
1,000 → 7.89; 2,000 → 6.95.

**D2x, generate mode** (`C:/Strata-exp/m11-*-D2x.stdout`, 13 workers, rest on):

| run | tok/s | round (ms) | CPU pool (ms/round) | pool share | rows GB/s | distinct / routed per layer | join wait (ms/round) |
|---|---|---|---|---|---|---|---|
| code-1 | 83.62 | 40.8 | 11.29 | 27.7 % | 37.4 | 6.26 / 7.87 | 1.451 |
| code-6 | 86.54 | – | 10.97 | – | 38.4 | – | 1.407 |
| thai-1 | 50.76 | 27.9 | 5.63 | 20.2 % | 39.8 | 3.32 / 3.85 | 2.053 |
| 8k-1 | 68.08 | 31.1 | 4.16 | 13.4 % | 41.3 | – | 0.326 |
| 8k-6 | 66.08 | – | 4.48 | – | 38.2 | – | 1.330 |

The same arm varies 3.0-3.4 % inside one session, so a two-run arm cannot resolve effects under about 4 %. In
code-1 the `pool` counter is 15.97 ms/round but only 11.28 ms is rows; the rest is the plan (2.77 ms, containing the
4070 `launch` 2.56 ms, serial before the pool publishes), activation quantize 0.35, jobs 0.63 and the 4070 finish 0.88,
with the workers parked through all of it.

**The live serve log** (`D:/Github/Strata/strata-flash-next-d2x.log`, the boot from line 228): 294 requests, 295,899
tokens, 5,743 s of decode; `cpu experts` 1,706 s = 29.7 % of decode (an upper bound: it times the whole dispatch,
`generate.cpp:995-997`); CPU entries 21.8 % of all. Per-request tok/s (n = 204 requests of 200+ tokens) has mean 56.75,
sd 13.19, range 33.8-83.8, so it is not usable as an A/B metric. The log has no wall-clock timestamps.

## 4. The kernel (`src/kernels/cpu/q2_avx2.cpp`)

- **Geometry** (`expert.hpp:34-45`): H 2560, FF 640, 64-weight blocks of 18 B. Gate and up rows are 40 blocks (720 B),
  down rows 10 blocks (180 B); an expert is 76,800 blocks = 1,382,400 B. The pack is native Q2_0 on every layer, so the
  pool runs modes 5 (gate/up + SwiGLU) and 6 (down) (`pool.cpp:400-436`).
- **Per layer** (`expert_source.cpp:779-1061`, `pool.cpp:531-563`): the host quantizes the activation; builds one job per
  distinct expert; phase 5 runs `mtasks_ = 3 × (13 + 1) = 42` equal row-range tasks, each doing gate rows then up rows as
  two passes into thread-local buffers, then a scalar SwiGLU with `(float) exp((double))` (`:419`, in a TU compiled
  without `/arch`); the host quantizes the down input serially; phase 6 runs the down rows. That is 96 phase barriers per
  round.
- **Disassembly** (MEASURED, `dumpbin /disasm` of `build-dyn/.../q2_avx2.cpp.obj`, `rows<1,1>` block loop): **41
  instructions per 64-weight block, about 33 vector-ALU µops**: 3 vpsrlw, 4 vpand, 8 vpunpck, 2 vinsertf128, vmovd,
  vcvtph2ps, 3 vmulss, 2 vbroadcastss, 2 vpdpbusd, 2 vcvtdq2ps, 2 vfmadd231ps, 2 vaddss. No scalar FMA contraction:
  the `corr` chain is a separate mul and add (0 `vfmadd*ss`, 80 packed `vfmadd*ps` in the object).
- **Bounds (STATIC):** at NT = 1 the two dependent FMAs per block set a floor of about 8 cycles per block on one
  accumulator chain; about 33 µops is roughly 9-11 cycles of throughput on a P-core, so latency and throughput are
  close. On E-cores (Gracemont) 256-bit ops split in two, so a block costs roughly 2-2.5× more and the chain does not
  bind. About 23 % of the 16-byte code loads cross a cache line (18-byte block stride).
- **Tokens per job** (MEASURED from generate logs): mean NT 1.15-1.24; windows of 5-6 tokens almost never occur
  (`T5:0 T6:0` in every m11 D2x run). So about 80 % of jobs are NT = 1 and the NT 5-8 path is effectively unused.
- **Compute or bandwidth bound:** throughput scales linearly with added E-cores (#16: 6 → 9 workers +2.33 GB/s per
  E-core, 9 → 13 +2.2), with no knee, and 13 workers reach 42.8 GB/s against about 112 GB/s theoretical (DDR5-7000 dual
  channel). That points at per-core limits, but it does **not** separate instruction limits from per-core memory
  parallelism (checker's correction); the M-D microbenchmark (§6) with L2-resident against DRAM-streamed data settles it.
- **SwiGLU and quantizers are small:** gate/up ms ÷ down ms = 2.002 (code) and 2.011 (thai) against a byte ratio of
  exactly 2.0, so the SiLU is a few percent of gate/up; the two quantizers are about 1.1 % of a round, the down quantize
  0.114-0.18 ms/round.

## 5. The upstream gap (22.9 vs 27.9 ms/round) is placement, not the pool

The figure (#45 step 1; fork-delta report line 85) compares upstream U1 with **O1** (`7134ce9`, not the merged engine),
one card, same session. O1's pool received **34 % more distinct experts per layer** (14.46 against 10.81) because of a
different profile (8,000 pairs against upstream's 24,576), a slower adaptive-swap rate (8 per round against 96 every 4)
and no PCIe share of misses (exclusive mode requires `pcie_frac == 0`, a measured choice, #27). Per distinct expert our
pool is as fast or faster: 1.84-2.21 ms per distinct expert per layer against upstream's 1.89-2.12 (m1, m9, m12), and
the byte ratio 959/717 MB per round = 1.337 equals the distinct ratio 1.338. The merged engine with upstream's profile
matched upstream in serve (m8: code 70.97 / 70.85 against 71.35 / 70.10 tok/s). The VNNI advantage over AVX2 is **not
established**: m1 shows O1 +9-12 % GB/s, m9 does not, and both are under the 13.6 % gate. D2x already uses the
24,576-pair ranked profile, so this lever is already taken in the live config.

## 6. What stutters, and the ranked plan

**Three candidate causes (UNMEASURED which):** (1) the pool spinning at priority 15 on 14 physical cores during
decode; (2) core parking hiding the six siblings; (3) the display card (the 4070 is CUDA device 1, `start-flash-next.ps1:38`;
the log shows "SECONDARY COMPUTE" and 4,400-4,800 experts per prompt part "from the 4070's slots") computing experts,
plus 4 unpinned stager threads at base priority 13 during prompt reads (`prefill.cpp:641`, `:1290`). Only (1) involves
the worker count or the kernel.

**Speed-up of the rows needed to hold D2x decode at fewer workers** (ratios from #16's rows GB/s; 10 and 8 interpolated
at 2.2 GB/s per E-core): 10 workers 1.18×, 9 1.26×, 8 1.35×, 6 1.58×. Without it, a static extrapolation on D2x code
gives 9 workers about −6.6 % tok/s and 6 workers about −14 % (#16's measured losses in an older config were smaller: 9
−2.7 %, 6 −6.7 %).

| rank | change | bit-exact | needs | status |
|---|---|---|---|---|
| P0 | a passive scheduler-lateness probe: one thread pinned to each logical CPU at NORMAL priority plus unpinned ones, each on a 1 ms waitable timer, logging lateness p50/p99/max, parking status per CPU, `dynfix` CPU and GPU use every 100 ms; classify by phase (decode / prompt read / idle); optionally PresentMon and a 60 s WPR capture | n/a | a tiny exe; can run during normal use | attributes the stutter; do first |
| P1 | core parking off: `powercfg /setacvalueindex SCHEME_CURRENT SUB_PROCESSOR CPMINCORES 100` (and `CPMINCORES1`), `powercfg /setactive SCHEME_CURRENT` | yes | the developer's decision (system setting, reversible) | hypothesis: siblings stay available to unpinned desktop threads |
| P2 | `STRATA_POOL_SPIN_US` >= 1,100 in the D2x env (above tl1's max gap between layers, 1,059 µs; settle the value at 40K+ context where gaps grow) | yes | a server restart | below ~1,060 µs workers would sleep every layer: 46 wakes × 57-154 µs = 2.6-7 ms/round |
| P3 | call `rest()` after the serve windows (`generate.cpp:5184`, `:5447`) and at request end; add `join wait`, pool ms/round, rows GB/s and the sleep count to the serve `request metrics` line | yes | a build | supersedes P2; #16 measured `join wait` 19 ms/round without rest (older config) |
| P4 | a parked worker drops itself to `THREAD_PRIORITY_IDLE` after 20-50 µs and restores the pool priority before `parked_.fetch_sub` and before claiming; the desktop can then preempt it in the ~50 % of a round between layers | yes | a build; medium risk (scheduler, Thread Director, boosts) | the main lever that keeps 13-worker throughput; red tests listed in Appendix E |
| P5 | tail balance: guided task sizes (large first, down to >= 8 rows) or 6-8 × threads tasks in `pool.cpp:402-406`, `:539-541` | yes (rows are independent) | a build | bound: <= ~18 % of phase time (≤ 1.2 ms/round tl1, ≤ 2.0 D2x code); realistic about half |
| P6 | kernel: **K1** a cheaper unpack (`vbroadcasti128` + `vpshufb` replicate + `vpand` + `vpsrlw` + `vpblendw` + a `vpshufb` LUT: 17 → 10 ops, exhaustively testable: 256 values × 16 positions); **K2** gate+up two-row interleave with an in-register SwiGLU (removes the latency floor and the 40 KB thread-local buffers); **K3** hoisted scale broadcasts and a precomputed `hx[2b]+hx[2b+1]`, keeping `corr` as a separate mul+add; **K4** down rows interleaved 2-4 at a time | yes | the M-D microbenchmark first | static estimate 1.2-1.35× on P-cores, 1.0-1.1× on E-cores |
| P7 | worker sweep 13 / 12 / 10 / 8 / 6 (P-first order drops E-cores: 10 = 5P+5E, 8 = 5P+3E, 6 = 5P+1E), plus 12 workers with `--secondary-async-launch` (the helper then pins to logical 19 and takes the 2.56 ms serial 4070 launch off the pool's path) | yes | the GPU | #51's acceptance run |
| P8 | `--process-priority 1` (ABOVE_NORMAL: workers and host at 12, below dwm's 13) | yes | a restart | ranked below P4: it makes WORKING workers preemptible (the 57.0 → 66.8 swing in `2026-09-28-cpu-pool-sweeps.md` sweep 2) |
| P9 | UMWAIT / TPAUSE in the park | yes | WAITPKG and the OS limit are UNVERIFIED | lowers spin power only; worth it only if P0 shows PL1 throttling |
| P10 | N1 plane layout (activations pre-permuted so the unpack interleave disappears) | **no** | the developer's decision | static 1.5-1.8×; breaks ADR 0001 and the CPU/GPU Q2_0 parity (#1); only if P4-P7 cannot reach the target |

**Expected outcome (hypothesis):** 10 workers look reachable with P5 + K1-K3 (1.18× needed); 8-9 need the top of
those ranges; 6 is not reachable bit-exactly on these estimates. If P1, P3 and P4 fix the desktop at 13 workers, the
worker count becomes a pure tok/s choice.

**Measurements the plan needs:**
- **M-A** engine generate ABBA (GPU, live server stopped): D2x args, the m11 prompts (code, thai, 8k), 256 tokens,
  ABBAABBA, identical output hash, exe sha recorded; judge on `verify window`, `pool multi`, `dispatch detail`,
  `join wait`; at least 4 pairs (same-arm spread 3.0-3.4 %).
- **M-B** serve ABBA with P3's counters: a fixed request set at 1K, 8K and 32K context, same session, both orders.
- **M-C** desktop: the P0 probe's p99 / p99.9 / max per phase, sibling parking during decode, PresentMon frame-time p99;
  the baseline from P0 first.
- **M-D** kernel microbenchmark (CPU only): today's `rows<1..4>` against K1-K4 on 1 P-core, 1 E-core and 14 threads;
  L2-resident and >1 GB DRAM-streamed data (decides compute- vs bandwidth-bound); a `memcmp` gate first with a mutant
  that must fail; >= 200 ABBA iterations; p10 / p50 / p90 ns per block.
- **M-E** pool microbenchmark (CPU only): the real `ExpertPool` with synthetic jobs at tl1's cadence (48 × (98 + 49 µs),
  ~370 µs between layers, ~6.7 ms between windows); arms spin, rest, `SPIN_US` ∈ {200, 500, 1500}, P4 with T ∈ {20, 50},
  P5; plus a NORMAL load needing 2 ms of CPU every 16.7 ms; report phase p50/p99 and the load's missed deadlines.
- A deep-context `STRATA_TIMELINE` of D2x at 32K-64K for the real gap distribution (P2, P4).

**What can happen when:** now, observing only (P0 probe, parking samples, PresentMon); written now as code with red
tests but built later (P3, P4, P5, K1-K4, the M-D and M-E harnesses; a build takes minutes of CPU); with the CPU idle
(M-D, M-E); with the GPU and the server stopped (M-A, M-B, P2 and P7 sweeps, P8, the deep trace); the developer's call
(P1, P10).

## 7. Corrections the checker made to the surveys (the appendices keep the originals)

- **"About 70-80 % spin"** refined from the trace: a worker is busy about 16 % of a round (84 % spin under serve without
  rest); on D2x code the pool is 27.7 % of the round, so about 77 % spin (static from counters).
- **"The hidden ~1/3 of pool time is phase sync and wake" is wrong.** A spinning wake is 0.3-0.5 µs and the host-serial
  down quantize is 2.5 µs per layer (0.114 ms/round); the recoverable loss is tail imbalance, about 18 % of worker time
  inside phases.
- **"B6 (the barrier / serial quantize) matters" is wrong:** 0.114-0.18 ms/round, under 0.5 % of a round.
- **"Only the non-bit-exact N1 removes the unpack interleave" is incomplete:** K1 is a bit-exact cheaper unpack.
- **"Compute-bound" is unproven:** linear E-core scaling does not separate compute from per-core memory parallelism.
- **"13 workers = 42.8 GB/s"** is #16's older config; D2x measures 37.4-41.3 GB/s (m11).
- **The desktop is confined to the six siblings** is right statically, but parking (§2) and #16's 19 ms/round
  `join wait` suggest they are not reliably usable in practice (hypothesis).
- **Missed by the scheduling survey:** `--pool-priority` also moves the host thread (`generate.cpp:2444`).
- **Correction to #45 step 1:** "upstream's `--expert-cache auto` puts more experts on the card" is contradicted by the
  logs (7,503 vs 7,529 slots); the difference is the profile, the swap rate and the PCIe share (§5).

## 8. Open questions

- Which of the three causes produces the stutter (P0).
- Whether parking keeps the siblings from desktop threads during decode.
- P4's cost and effect; the real gains of P5 and K1-K4.
- Compute-bound or per-core-memory-bound (M-D).
- D2x tok/s at fewer workers (only #16's older config was measured).
- `STRATA_POOL_SPIN_US` behaviour at deep context; WAITPKG availability.

## Sources

- Code: `C:/Strata-exp/src-dyn/src/kernels/cpu/pool.cpp`, `include/strata/kernels/cpu/pool.hpp`,
  `src/kernels/cpu/q2_avx2.cpp`, `src/program/generate.cpp`, `src/core/verify.cpp`, `src/core/expert_source.cpp`.
- Data: `C:/Strata-exp/m1*.stdout`, `m8.log`, `m9*.stdout`, `m11-*-D2x.stdout`, `m12*`;
  `C:/AI/UsersxenodAppDataLocalTempstrata-claude-stagetl1.json`, `tl1b.json`; `D:/Github/Strata/strata-flash-next-d2x.log`;
  `%TEMP%/strata-claude-p0/*.stdout`, `%TEMP%/strata-claude-rest/*`; `docs/reports/2026-09-28-cpu-pool-sweeps.md`; #16, #45.
- The analysis scripts and the disassembly listing (`tl2.py`, `tl3.py`, `tl4.py`, `topo.py`, `q2.asm`) were written to the
  session scratchpad and are **not** kept in the repository; the numbers they produced are recorded above.

---

The appendices below are the survey agents' reports, verbatim. They were written for an engineer, not edited for this
report, and some of their claims are corrected in §7.

## Appendix A. Idle behaviour under --serve (survey, verbatim)

**Idle pool workers under `--serve` (src-dyn @ 5c51574, read-only survey)**

Short answer: when there is no expert work, a worker spins with `_mm_pause` for 20 ms, then blocks on a condition variable. It never yields in between. Between requests and during batched prompt reads it is asleep, so it costs about 0 CPU. During decode it never reaches the 20 ms timeout, so all 13 workers and the host stay at 100 % on their pinned cores at Windows priority 15 for the whole decode. `pool.rest()` is not called anywhere in the `--serve` path.

##### 1. What the worker does when idle (pool.cpp / pool.hpp)

- **Park loop** (`src/kernels/cpu/pool.cpp:275-291`):
  - `while (epoch_ == seen) { _mm_pause(); ... }`. There is no `yield`, `SwitchToThread` or `Sleep(0)` anywhere in the loop.
  - The clock is read only every 1024 pauses (`:280`, `(++spins & 1023u) != 0`).
  - The worker sleeps only if `rest_` is set, or if `steady_clock::now() - parked_at >= spin_before_sleep_` (`:281-282`).
  - It then takes `sleep_mu_`, sets `wstate = kSleeping`, does `sleepers_++`, and calls `sleep_cv_.wait(...)` (`:283-290`). This is a real OS block.
- **Spin length:** `kSpinBeforeSleep{20}` ms (`include/strata/kernels/cpu/pool.hpp:167`). The env knob `STRATA_POOL_SPIN_US` overrides it; it is read once in the constructor (`pool.cpp:203-204`) and the header calls it "a test knob".
- **The 20 ms restarts at every layer.** `parked_at` is taken at the top of each park (`pool.cpp:275`), i.e. after every drain. So a worker sleeps only after a gap longer than 20 ms with no publish.
- **Wake-up:** `publish()` clears `rest_` (`:242`), bumps `epoch_` (seq_cst), and calls `notify_all` under `sleep_mu_` only if `sleepers_ != 0` (`:244-248`).
- **Sleepers count as parked.** A sleeping worker still counts in `parked_`, so `wait_parked` does not wait for it to wake. The host publishes and drains. A late waker's claim fails because of the epoch-tagged `head_` CAS (`pool.cpp:314-323`; pool.hpp:19-28, issue #29). Sleeping mid-request is therefore safe by design.
- **Host thread:** `wait_done` and `wait_parked` are pure `_mm_pause` spins with a 60 s stall abort (`pool.cpp:339-379`, `kStall` at pool.hpp:170). The host also drains jobs (`host_works_`, `:466`).
- **Where the threads run** (static reading of `physical_cores`, `pool.cpp:32-108`):
  - The list is one logical CPU per physical core, sorted fastest class first (EfficiencyClass descending).
  - Workers use `physical_cores(true)`, which drops the first core. On the i5-13500 that is 14 − 1 = 13, one per core: P1–P5 plus E0–E7. "Every physical core" is the default when `--pool-workers` is 0 (generate.cpp:535).
  - The host is pinned to `physical_cores(false)[0]` (`src/core/session.cpp:516-518`).
  - Pinning is a hard `SetThreadAffinityMask` (`pool.cpp:115`).
  - Free logicals are only the 6 hyperthread siblings of the P-cores. The exact logical numbering on this machine is UNVERIFIED.
- **Priority:**
  - The live config has `--pool-priority 2` and `--process-priority 2`.
  - `SetPriorityClass(HIGH_PRIORITY_CLASS)` at `generate.cpp:2439-2441`.
  - `set_worker_priority(2)` and host `SetThreadPriority(HIGHEST)` at `generate.cpp:2443-2444`, applied per worker at `pool.cpp:221`.
  - HIGH (base 13) + HIGHEST (+2) = 15, the top non-realtime level. That this level is 15 is a Windows fact, not measured here.
  - Static reading: a spinning worker at 15 is always ready, so no normal-priority thread (8–10) can run on that logical while it spins.
  - Upstream `4c68013` has no `pool_priority` or `process_priority` option (grep of `4c68013:src/program/generate.cpp` found none).

##### 2. `rest()` and why it only works in generate mode

- `void rest() { rest_.store(true, seq_cst); }` (`pool.hpp:176`). The flag is cleared by the next `publish()` (`pool.cpp:242`). It only skips the 20 ms spin. The worker still checks the flag once every 1024 pauses; at about 140 cycles per pause (Intel vendor figure for Skylake-era cores) that is roughly 30 µs on a P-core. UNMEASURED; E-core pause latency unknown.
- **Only call site:** `generate.cpp:6592`, `if (o.pool_rest) pool.rest();`, inside the plain generate decode loop (the `ver.run` at `:6583`). `decay_scores()` right after it (`:6593`) is also generate-only.
- **The `--serve` block** runs from `generate.cpp:4155` to `return 0;` at `:5704-5705`. Its window calls do not call `rest()`:
  - `:5447`, the decode loop.
  - `:5184`, `read_windows`, which reads short prompt parts of up to `--short-read` 64 tokens through decode windows.
  - `:4343`, `win_pool_fn`.
- **Why:** commit `ba0cade` added four lines to generate.cpp (option, help, parse, one call). The commit message says "The verify loop now calls pool.rest()" and says nothing about serve. The serve loop is a separate copy of the decode loop, so there is no sign the omission was deliberate. The blueprint report (§P6, line ~591) says the call was also missing at `7134ce9` and `2a9b26c`, so it is not a merge loss.
- The daily branch `merge-0.1.20` (@ `284bb19`) has the same single call site, at `generate.cpp:5636`.
- The help text "pool workers sleep between verify windows" (`generate.cpp:531`) is wrong for `--serve`.

##### 3. Behaviour by phase under `--serve` (static reading plus the live log)

| Phase | Does the pool run? | Worker state | CPU |
|---|---|---|---|
| Waiting for the next request | no; main thread blocks on `in_cv.wait` (`generate.cpp:4776-4783`), stdin thread blocks in `_read` (`:4751`) | spin 20 ms after the last publish, then block | about 0 after 20 ms; 13 × 20 ms ≈ 0.26 core-s per request end |
| Batched prefill | no; `src/prefill/prefill.cpp` never references `ExpertPool` (grep), host waits use `cudaEventBlockingSync` (`prefill.cpp:269`, `:560-590`) | asleep after 20 ms | about 0 from the pool |
| Short prompt parts ("windows", ≤64 tokens; e.g. log "new turn: 4 tokens ... (windows) in 50.9 ms") | yes, through `ver.run` at `:5184` | spinning or working | 13 + host at 100 % |
| Decode | yes | spinning or working the whole time; gaps between windows are a few ms, below the 20 ms threshold | 13 + host at 100 % at priority 15 |
| Gaps inside a window (between layers, or layers with no CPU jobs; `run_split_multi_native` returns without publishing when `n<=0`, `pool.cpp:532`) | no | spinning | 100 % |

**From the live log** (`D:/Github/Strata/strata-flash-next-d2x.log`, awk over the "generated in" and "request metrics" lines):
- 247 requests, 265,428 tokens generated, 5,236.8 s of decode wall time.
- "cpu experts" totals 1,527.0 s, which is **29.2 % of decode wall**. This is an upper bound on pool work: `cpu_ms` times the whole dispatch, including planning, the 4070 launch and finish (`generate.cpp:995-997`; `expert_source.cpp:1024-1034`).
- Batched prompt time is 1,105.7 s, during which the pool sleeps.
- Example request: 4,584 tokens in 62,751 ms, cpu experts 24,783.8 ms (39.5 %).
- Memory note (#44): a code round is 40.1 ms, of which the pool is 7.8 ms, about 19 %.
- Estimate (UNMEASURED, derived from the above): each worker spends about 70–80 %+ of decode time in `_mm_pause` spin. That is about 13 × 5,237 s ≈ 68,000 core-s of priority-15 occupancy in this log, roughly 48,000+ of it spin.
- Between requests the pool uses about 0 CPU. So the answer to "13 cores burned while waiting for Claude Code or while the GPU prefills" is **no**. The answer to "during every decode" is **yes**.
- The log has no timestamps, so the time between requests cannot be measured from it.
- No "pool" or stall lines in either log. The serve path prints no `join wait` or pool-phase counters; they exist only in generate mode (`generate.cpp:6713-6754`, `:6888`).

##### 4. What `--pool-rest` measured

Source: `ba0cade`, Strata-xeno #16, A/B `strata-claude-rest`, exe `83b8cbabe6fa`, same session, HIGH class, **generate mode**, outputs identical in 16 runs.

| Arm | code tok/s | thai tok/s | pool ms/round | GB/s |
|---|---|---|---|---|
| 6 workers, rest off | 80.52 | 50.55 | 10.63 | 25.5 |
| 6 workers, rest on | 83.28 (+3.4 %) | 50.92 (+0.7 %) | 10.07 | 27.0 |
| 9 workers, rest on | 86.84 | 53.20 | 8.03 | 34.0 |
| 13 workers, rest on | 89.22 | 53.75 | 6.38 | 42.8 |

- Both +3.4 % and +0.7 % are below the 13.6 % noise gate.
- The "+10.8 %" figure compares 13 workers + rest against 6 workers without rest, so it mostly measures worker count. No run has 13 workers with rest off (blueprint report `:586-588`, `:1843-1844`).
- The mechanism behind `rest` was the first 13-worker attempt: `join wait` rose from 0.9 to 19 ms/round and thai fell to 30.2, because spinning HIGHEST workers on every core kept the adapt thread from running between windows (`tests/xeno/pool_rest.cpp:1-4`, #16).
- Unit tests:
  - `pool_rest`: four 30 ms idle pools use about 0 ms of CPU with `rest()`; a no-op mutant uses 156 ms (`ba0cade`).
  - `pool_idle_sleep`: 312 → 31 ms of process CPU over 150 ms (blueprint `:481`).
- 9 against 13 workers with rest in generate mode: code −2.7 %, thai −1.0 %, both within noise. This is the most relevant existing data for a lower worker count.

##### 5. What it would take to make rest or sleep work under `--serve`

1. **Wire it in (about 2–3 lines).** Add `if (o.pool_rest) pool.rest();` after the `ver.run` at `generate.cpp:5447` (decode) and at `:5184` (`read_windows`). This also puts the workers to sleep straight after a request's last window instead of 20 ms later. Consider `arena_src.decay_scores()` there too (`:6593` does it in generate mode).
   - This frees only the gap between windows: MTP draft about 2.5 ms, commit, adapt spawn and join, `apply_pending`, suffix propose. Per the memory note that is roughly 3–8 ms of a 25–40 ms round. Static reading, UNMEASURED.
   - It does not touch the spin between layers inside a window, which is most of the time.
   - Cost: one condition-variable wake of 13 workers per window. Wake latency is UNMEASURED; the only nearby figure is about 0.3 ms per Windows blocking-sync round trip for the GPU (`remote_experts.cpp:73-74`, a different mechanism).
   - It needs its own serve ABBA. The 13-worker serve arm without rest was never measured, and serve prints no `join wait`. Add that counter to the serve per-request metrics first.
2. **Free the desktop during the window too (candidates, all UNMEASURED):**
   - **(a)** `STRATA_POOL_SPIN_US=<N>` in the config's `env`. It needs no build: workers sleep after max(N µs, 1024 pauses) of idle. It works in serve today and the #29 claim protocol covers sleeping mid-request. Sweep N over something like 2000 / 500 / 100 against decode tok/s.
   - **(b)** Yield inside the park loop (`SwitchToThread` every K pauses), or drop the thread to normal priority while parked and raise it again on wake. That is a syscall pair per phase, and there are 2 phases × up to 48 layers per window.
   - **(c)** Drop `--process-priority 2`, which takes the pool from 15 to 13 and is still above normal. Measured cost: 76.76 → 78.73 code (`0a2d70d`).
   - **(d)** Fewer workers, so the E-cores are left completely free. With 9 workers the order of `physical_cores` leaves E4–E7 and the 6 siblings unpinned.
   - Priority itself measured code 57.0 → 66.8 (`cpu-pool-sweeps.md` sweep 2, #14). Lowering it is a real trade-off.
3. **Measurements still needed** (CPU or GPU work, so they wait for the developer):
   - Serve ABBA of rest on/off at 13 workers.
   - `STRATA_POOL_SPIN_US` sweep.
   - Worker sweep 13 / 10 / 8 / 6 with rest wired in, plus a desktop-responsiveness check (#51 acceptance).

##### Relevant paths

- `C:/Strata-exp/src-dyn/src/kernels/cpu/pool.cpp`
- `C:/Strata-exp/src-dyn/include/strata/kernels/cpu/pool.hpp`
- `C:/Strata-exp/src-dyn/src/program/generate.cpp` (lines 2439-2445, 4155-5705, 5184, 5447, 6592)
- `C:/Strata-exp/src-dyn/src/core/session.cpp:510-519`
- `C:/Strata-exp/src-dyn/src/core/remote_experts.cpp:73-79`
- `C:/Strata-exp/src-dyn/tests/xeno/pool_rest.cpp`, `pool_idle_sleep.cpp`, `pool_core_policy.cpp`
- `D:/Github/Strata/.worktrees/merge-0.1.20/docs/reports/2026-09-30-fork-delta-and-blueprint.md` (lines 469-481, 575-593, 1842-1846)
- `D:/Github/Strata/strata-flash-next-d2x.log`
- Issues: xenodeve/Strata-xeno #51 (open, this goal), #16 (worker count and CPU Sets), #14 (priority)

## Appendix B. Thread placement and priority (survey, verbatim)

**Where the D2x pool's threads run, at what priority, and which knobs make it a better desktop neighbour.** Read-only survey. Code is `C:/Strata-exp/src-dyn` at `5c51574` unless another path is given.

#### 1. Machine topology (queried live with `GetLogicalProcessorInformationEx`, no engine involved)

| LP (logical processor) | Physical core | EfficiencyClass | L2 cache |
|---|---|---|---|
| 0,1 | P0 (SMT pair) | 1 | 1280 KB, own |
| 2,3 | P1 | 1 | own |
| 4,5 | P2 | 1 | own |
| 6,7 | P3 | 1 | own |
| 8,9 | P4 | 1 | own |
| 10,11 | P5 | 1 | own |
| 12,13,14,15 | E0 to E3 | 0 | 2048 KB, shared by the four (mask `0xF000`) |
| 16,17,18,19 | E4 to E7 | 0 | 2048 KB, shared by the four (mask `0xF0000`) |

- L3 is 24 MB, shared by all 20 LPs.
- The OS reports P-cores first, as upstream's order also does.

#### 2. How the placement is computed

- **One LP per physical core.** `physical_cores()` takes the first set bit of each core's mask (`src/kernels/cpu/pool.cpp:49-56`). So it picks LPs 0,2,4,6,8,10 for the P-cores and 12 to 19 for the E-cores. It never picks the odd SMT siblings 1,3,5,7,9,11.
- **P2 sort.** It then does a `stable_sort` by EfficiencyClass, highest first (`pool.cpp:63-68`, commit `ebcda26`). On this CPU the order does not change, because the OS already lists P-cores first.
  - Upstream `4c68013` has no sort (`pool.cpp` there, lines 30-55). The placement is therefore the same as upstream on this machine.
- **Core 0 goes to the host.**
  - `ExpertPool` builds its core list with `physical_cores(true)` (`pool.cpp:205`), which drops the first entry (`pool.cpp:106`). That leaves `[2,4,6,8,10,12,13,…,19]`: 13 entries.
  - `n_ = 13` comes from `--pool-workers 13` in the D2x config.
  - Worker `i` is hard-pinned to `cores[i]` with `SetThreadAffinityMask(1<<core)` (`pool.cpp:112-115`, `:218-220`).
- **The host thread is pinned to LP0.** `SessionLoopScratch::init` pins it with `physical_cores(false)[0]` (`src/core/session.cpp:516-518`). It is called once at `generate.cpp:3988`, which is before the serve branch at `:4155`, so it is the same thread.
- **The host also drains.** `host_works=!o.no_host_worker` (`generate.cpp:2445`), and the log line 25 reads `strata generate: 13 expert-pool workers + the host thread`.

**Result (static reading, confirmed by a live snapshot below):**

| Thread | LP | Core |
|---|---|---|
| host | 0 | P0, first thread |
| w0 to w4 | 2,4,6,8,10 | P1 to P5, first thread |
| w5 to w12 | 12 to 19 | all 8 E-cores |

- **No physical core is free.** All 14 cores carry a pinned engine thread (13 workers plus the host).
- The only LPs without a pinned engine thread are the SMT siblings 1,3,5,7,9,11. Each shares its core with a worker or with the host.
- **No P-core is left for the desktop or for the 4070's display work.** P0 carries the host thread, which spin-waits on the GPU ring (`verify.cpp:1026-1027`, a `_mm_pause` loop). The inline 4070 launch also runs on that host thread.
- With `--pool-workers 13` the `--secondary-async-launch` helper would get index 13, which is out of range, so it would run unpinned (`generate.cpp:3140-3142`). That flag is off in D2x anyway.

#### 3. Priority

- **What the config asks for.** The D2x args are `--pool-priority 2 --process-priority 2`, and the env sets no pool or priority variables. The launcher `start-flash-next.ps1` only sets `CUDA_VISIBLE_DEVICES` (line 39).
- **What the code does with it.**
  - `SetPriorityClass(HIGH_PRIORITY_CLASS)` (`generate.cpp:2439-2441`). The process base becomes 13.
  - `set_worker_priority(2)` and `set_current_thread_priority(2)` (`generate.cpp:2443-2444`, `pool.cpp:126-136`) set `THREAD_PRIORITY_HIGHEST`. That puts the workers and the host at **15**.
- **Live snapshot** (`Get-Process dynfix`, which only reads, PID 22408):

| Thread group | Count | Priority |
|---|---|---|
| Process `BasePriority` | – | 13 |
| Base 15 (13 workers + host) | 14 threads | 15 |
| Base 13 (CUDA, 4-thread stager, adapt, monitor, PLE reader and others; dynamic 13-15) | 93 threads | 13 |
| `dwm.exe` for comparison | – | base 13 (`csrss` is 13 too) |

- **What this means for the desktop (static reading).**
  - 15 is the top of the dynamic range, and a boosted normal-class thread is capped at 15.
  - So `dwm`, explorer and foreground apps cannot preempt a worker on its LP. Only real-time and MMCSS threads (priority 16 and up), interrupts and DPCs can.
  - The pinned workers never migrate, so the desktop is confined to the six SMT siblings.
  - Even the 93 base-13 engine threads tie with `dwm` and outrank every normal-class app (base 8).
- **Stale claim.** Issue #14 says in its trade-off check that "E-cores stay free". That was written at 6 workers. At 13 workers every E-core is pinned, so it no longer holds.

#### 4. Spinning: the likely stutter mechanism (static reading)

- **What a parked worker does.** It spins `_mm_pause` for `kSpinBeforeSleep` = 20 ms and only then sleeps (`pool.hpp:167`, `pool.cpp:275-291`).
- **`--pool-rest` does nothing under `--serve`.** `pool.rest()` is called only at `generate.cpp:6592`. The serve branch runs from `:4155` to `:5705` and returns there (`:5704 return 0`), and grep finds no other `rest()` call.
  - The serve decode path uses the pool through `ver.run(... win_pool_fn, win_pool_user ...)` at `generate.cpp:5184` and `:5447`, but never rests it.
  - The blueprint §P6 finding and issue #51 lead 1 are therefore **confirmed on the live exe's source**.
- **Consequence during a served response.**
  - Each layer publishes a batch every sub-millisecond, and the gaps between rounds (draft, commit, adapt) are a few ms. Both are far below 20 ms, so the 13 workers and the host hold all 14 physical cores at priority 15 for the whole response, plus 20 ms after it.
  - Rough duty cycle, from the memory note's #44 numbers (code round 40.1 ms, pool 7.8 ms/round; measured on the daily branch, not on this exe): the workers do expert work about 19 % of a round and spin at priority 15 for the other ~80 %. **UNMEASURED on D2x.**
  - Between requests they do sleep after 20 ms (`pool.cpp:281-288`). The engine was idle at the time of the snapshot (all threads in Wait).
- **Measured precedent** (`ba0cade`, #16): with 13 workers at HIGHEST and no rest, the adapt thread could not run and `join wait` rose from 0.9 to 19 ms/round. Serve mode today has that same no-rest condition.

#### 5. Knobs that make the pool a better neighbour without fewer workers

| # | Knob | How | Cost (static reading) | Measured? |
|---|---|---|---|---|
| A | `--process-priority 0` (drop HIGH class) | Config only | Workers become 8+2 = 10, below `dwm`'s 13. The 93 helper threads drop to 8. Cost: HIGH vs normal class was code 76.76→78.73 (+2.6 %) and thai 48.72→49.33, below the 13.6 % gate | **Yes**, `0a2d70d` (same-session, 6 workers, not D2x). The blueprint's upstream-hold table (line 1965) lists `--process-priority` under "desktop starvation; no gain". The memory note lists "`--process-priority 2` and 13 workers for serving" as parked for the developer |
| B | `--pool-priority 1`, `0` or negative | Config only (`atoi`, applied when ≠0: `pool.cpp:132`) | At HIGH class, 1 gives 14 (still above `dwm`) and 0 gives 13 (ties with `dwm`, so boosts decide). At normal class, 0 gives 8 | Priority 2 vs 0 at normal class: code 57.0→66.8 (+17 %), thai 26.9→36.7 (+36 %). The loss came from other programs preempting the pinned workers (`cpu-pool-sweeps.md` sweep 2, #14). 1 and negative values: **UNMEASURED** |
| C | `STRATA_POOL_SPIN_US` env (for example 500-2000 µs) | **Already in the live exe** (`pool.cpp:203-204`); config `env` only | Parked workers sleep after the given spin instead of 20 ms. This stops the spin between rounds and after responses. The risk is an OS wake-up on the critical path when the gap exceeds the spin; `pool.hpp` says a wake costs more than the µs batch gaps. The clock is checked every 1024 pauses (`pool.cpp:280`), so the timing is coarse. Mid-request sleeping is already survivable (#29) | **UNMEASURED.** It is a test knob. The closest data is `ba0cade`'s rest A/B (generate mode) |
| D | Wire `pool.rest()` into the serve loop after each verify window | Code change (~1 line near `generate.cpp:5447`); blueprint rank 10b | Removes the 20 ms spin between windows. Expected to match rest in generate mode | Rest alone at 6 workers, generate mode: code 80.52→83.28 (+3.4 %), thai +0.7 %. At 13 workers there is no rest-off arm. The join-wait effect is measured (0.9 vs 19 ms/round). **Serve UNMEASURED** |
| E | Keep LP0/P0 free (move the host) | Code change (`session.cpp:516`) | Impossible at 13 workers plus the host on 14 physical cores unless threads share SMT siblings. `pool.hpp` measured an unpinned host costing pool bandwidth: 36.32 vs 26.9 GB/s | The reservation is measured. Moving the host off LP0 is **UNMEASURED**. LP0 likely also receives driver DPCs (inference) |
| F | E-cores first (reverse the P2 sort) | Code change (`pool.cpp:65-66`) | Same set of cores at N=13, so no effect today. It only matters for N<13: for example, 8 workers on E0-E7 would leave P1-P5 free. E-cores share one L2 per four cores, so per-worker bandwidth is lower | Only P-first data: `--pool-workers 5` (P-only) was worse than 6 (5P+1E); code 59.6 vs 64.9 (`cpu-pool-sweeps.md` sweep 1). Rough separate boots: 4/6/10/13 gave 39.28/37.61/41.08/41.43 tok/s (`2026-09-27-core-policy.md`). E-first is **UNMEASURED** |
| G | No SMT siblings | Already the case (`pool.cpp:52-56`) | – | Soft affinity allowed onto siblings was worst: arm 2 (P-cores and their siblings) gave 24.4/21.5 GB/s against the hard pin's 43.8/42.3 (#16) |
| H | CPU Sets soft affinity (`SetThreadSelectedCpuSets`) | Patch at `%TEMP%/strata-claude-stage/cpusets.patch`, not committed | Lets the scheduler stack workers | **Measured and rejected** (#16): hard pin 76.77/74.77 tok/s vs soft 74.77/69.08, 69.64/67.51 and 75.39/74.08 |
| I | EcoQoS (`SetThreadInformation` with `ThreadPowerThrottling`) | Not used anywhere (grep) | With hard pinning it would mainly lower clocks, which hurts a bandwidth-bound pool (inference) | **UNMEASURED** |
| J | Process affinity mask from the launcher (exclude an LP) | External | `SetThreadAffinityMask` fails for a core outside the process mask, and `pin_this_thread` ignores the failure (`pool.cpp:115`). That worker would run unpinned and could stack on another worker (static reading) | **UNMEASURED.** Not advised as-is |
| K | `--lock-cpu-experts` | – | Protects against page trimming, not preemption | No gain on top of priority (`cpu-pool-sweeps.md`) |

`SetThreadPriorityBoost`, `SetThreadIdealProcessor` and `THREAD_MODE_BACKGROUND` do not appear anywhere in the code.

#### 6. Where the numbers came from

- **Config:** `D:/Github/Strata/strata-flash-next-d2x.json` (`--pool-workers 13`, `--pool-priority 2`, `--process-priority 2`; env has only `STRATA_PREFILL_EXPERT_SPLIT=1` and `STRATA_PREFILL_WAVE=1`).
- **Log:** `D:/Github/Strata/strata-flash-next-d2x.log` lines 25/87 (`13 expert-pool workers + the host thread`) and line 30 (`stager 4 threads`).
- **Tests:** `tests/xeno/pool_core_policy.cpp` checks the ordering and that the automatic count is 13. `pool_rest.cpp` and `pool_idle_sleep.cpp` only test that idle workers sleep, not serve mode.
- **Issue #51** (open, no comments) already names leads 1 and 4 of this survey.

## Appendix C. Per-expert kernel cost (survey, verbatim)

FINDINGS: per-expert CPU cost of the served pool (READ-ONLY static reading plus existing logs; nothing was built or run)

All paths are under `C:/Strata-exp/src-dyn` (HEAD `5c51574`) unless another root is named. Labels used below: MEASURED (with its source), STATIC (inferred from reading the code), UNMEASURED, UNVERIFIED.

#### Bottom line

- **The pool is compute-bound, not DRAM-bound.** Adding E-core workers raises pool throughput linearly with no knee (#16 table), and 13 workers reach only 42.8 GB/s against about 112 GB/s theoretical.
- **About 80 % of CPU jobs run one token**, so the unpack of the 2-bit codes cannot be shared across tokens.
- **The largest bit-exact lever** is doing gate and up rows together, with an in-register SwiGLU and a vectorised SiLU that has been checked on every float input. Static estimate: 1.15–1.3× on the kernel. That would be enough for about 9 workers at today's pool bandwidth.
- **Going down to about 6 workers** needs about 1.65×. Only the non-bit-exact plane layout (N1) reaches that on paper, and it breaks CPU/GPU parity and ADR 0001.
- **The desktop stutter has a likely cause that no kernel change touches** (§7). Under `--serve`, `pool.rest()` is never called. The 13 workers spin for up to 20 ms between phases at THREAD_PRIORITY_HIGHEST inside HIGH_PRIORITY_CLASS, so they hold every physical core for the whole of decode.

#### 0. What the live server runs

- **Config** (`D:/Github/Strata/strata-flash-next-d2x.json`): `--pool-workers 13`, `--spec 4`, `--spec-min-p 0.5`, `--pool-priority 2`, `--process-priority 2`, `--pcie-frac 0`, `--secondary-expert-mib 6400`.
- **ISA path:** the live log `strata-flash-next-d2x.log:1` says "native Q2_0 expert rows use AVX-VNNI". Dispatch is `expert_layout.cpp:73-78` (`q2_rows_any`: AVX-512, then AVX-VNNI, then AVX2). The quantizer dispatch is `:80-83` (`act_quant_any`, which falls back to `act_quant_q8_1_avx2` on this CPU).
- **Pack:** `packs/q2_0/native_experts.txt` lists every layer as `42 42` (gate/up and down are both Q2_0) with `blob_bytes 1382400`. The pool therefore takes the native modes 5 and 6 (`pool.cpp:400-436`), not modes 3 and 4.
- **Geometry** (`include/strata/kernels/cpu/expert.hpp:34-45`):
  - H = 2560, FF = 640, QK = 64, QKA = 32, BB = 18 bytes per 64-weight block.
  - Gate and up rows are 40 blocks (720 B). Down rows are 10 blocks (180 B).
  - Per expert: gate 25,600 blocks, up 25,600, down 25,600, so 76,800 blocks = 1,382,400 B.
- **Window length:**
  - The suffix drafter is on by default (`suffix_draft = 3`, `generate.cpp:384`). It raises `o.spec` to min(4+2, 8) = 6 and sets `mtp_max_t = 4` (`generate.cpp:1785-1787`).
  - Live log: "verify: window up to 6 tokens", with 1-, 2-, 3-, 4- and 6-token windows captured.
  - `MAXT = kVerifyMaxT = 8` (`expert.hpp:131`, `verify_kernels.hpp:21`).
- **Cores:**
  - `physical_cores(true)` ranks cores by EfficiencyClass (P first) and drops the first one (`pool.cpp:32-107`, `:205`). The 13 workers are therefore 5 P-cores and 8 E-cores, hard-pinned (`:218-220`).
  - The host thread is pinned to the first P-core (`src/core/session.cpp:510-518`) and drains jobs too (`pool.cpp:466`).
  - No SMT siblings are used, so the desktop gets only the 6 hyperthread siblings of the P-cores.

#### 1. What the pool does per layer (native Q2_0), with lines

`expert_pool_dispatch_multi` (`src/core/expert_source.cpp:779-1061`):

1. **Host quantizes the input activation**, one call per token: `act_quant_any(x, H)` (`:950-951`). 80 chunks per token.
2. **Host builds the job list** (`:980-1022`). A job is one distinct expert, with `nt` = the number of window tokens routed to it (`jb.nt`, `:1015-1019`).
3. **`run_split_multi_native`** (`pool.cpp:531-563`) runs three steps per batch of up to 96 experts:
   - **Phase 5, gate/up rows** (`:543`). `mtasks_ = 3 × (13+1) = 42` equal row-range tasks over `nb × 640` rows (`:539-541`). Each task, per expert segment (`pool.cpp:408-419`):
     - `q2_rows_any(gate, …, gp, r0, r1)` (`:415`), then `q2_rows_any(up, …, up, r0, r1)` (`:416`). These are two separate passes over the same row range, results going to thread_local `gbuf`/`ubuf` [8][640] floats (40,960 B per thread, `:410`).
     - SwiGLU, scalar, in a TU compiled without `/arch` (CMakeLists.txt:610-617 sets `/arch` only on `expert.cpp` and `q2_avx2.cpp`):
       `sb.ff[t][r] = (g / (1.f + (float) std::exp(-(double) g))) * u` (`:419`)
     - That is 640 double `exp` calls per (expert, token).
   - **Down quantize, serial on the host, between the phases** (`:545-548`): `act_quant_any(ff[t], FF)`. 20 chunks per (expert, token). All 13 workers wait parked meanwhile.
   - **Phase 6, down rows** (`:551`): `q2_rows_any(down, …, 10 blocks, a2, nt, out, r0, r1)` (`:424-429`).
4. **Sync cost:** each phase costs `wait_parked` + `publish` + wake + `wait_done` + `wait_parked` (`:461-471`). That is 2 phases × 48 layers = 96 phase barriers per round.

##### Kernel: `src/kernels/cpu/q2_avx2.cpp`

- **`unpack64`** (`:24-35`): 16 bytes of codes become 64 u8 codes. 1 load, 3 `psrlw`, 4 `pand`, 4 `punpck*bw`, 4 `punpck*wd`, 2 `vinserti128` (`set_m128i`). About 18 µops, shared by all NT tokens.
- **`row_multi<NT,Vnni>`** (`:37-70`), per 64-weight block:
  - Shared: `h2f(blk)` (`:17-21`: 16-bit load, `movd`, `vcvtph2ps`; about 4 µops), then `unpack64`.
  - Per token (`:48-63`):
    - 2 × `vpdpbusd` from a zeroed register with a memory operand (`:54-55`).
    - 2 × `vcvtdq2ps`.
    - 2 × (scalar `d*scale`, then `vbroadcastss`) via `_mm256_set1_ps(d * a[t]->scale[2b(+1)])` (`:60-61`).
    - 2 × dependent `vfmadd` on `acc[t]` (`:60-61`).
    - Scalar correction `corr[t] += d * (hx[2b] + hx[2b+1])` (`:62`): 2 loads, add, mul, add.
  - End of row (`:65-69`): 128+128 add, `movehl` add, `movehdup` add, minus `corr`. The summation order is ((a0+a4)+(a2+a6)) + ((a1+a5)+(a3+a7)). This is exactly the GPU CPU-order kernel's `shfl_down` 4/2/1 order (`src/kernels/cuda/iq_kernels.cu:350-354`).
- **`rows_multi_t`** (`:83-97`): cases 1-4 are direct. `default` splits into groups of 4 and calls again over the whole r0..r1 range (`:91-95`). So NT = 5-8 re-runs `h2f` + `unpack64` and re-streams the task's weights (about 50 KB per matrix per task, from L2) once per group.
- **`act_quant_q8_1_avx2`** (`:109-147`): per 32-value chunk, a vector abs-max, scalar `amax/127.f` and `1.f/s`, vector round-half-away plus clamp, then **a scalar 32-iteration int32→int8 store loop** (`:138`) and a vector sum.

##### Static µop count per 64-weight block (VNNI arm; my count from the source, not from disassembly)

| NT | µops per block | bytes per µop | note |
|---|---|---|---|
| 1 | ≈ 22 shared + ≈ 18 per token + ≈ 3 loop = **≈ 43** | 0.42 | ≈ 2.4 µops per weight byte |
| 1.24 (measured code mean) | ≈ 47 | | |
| 4 | ≈ 97 | | |

- **Latency floor at NT = 1:** the two dependent FMAs per block (`:60-61`) give at least 8 cycles per block (FMA latency 4) on one `acc` chain. The throughput estimate for about 43 µops is about 9-11 cycles on a P-core. The latency chain and the throughput limit are close, and a 40-block row (about 1,700 µops) exceeds the out-of-order window, so row-to-row overlap is limited. STATIC.
- **E-cores (Gracemont):** 256-bit operations split into two 128-bit µops, so a block is roughly 2-2.5× more costly. The chain does not bind there. STATIC.
- **Split loads:** block stride is 18 B and row start mod 64 ∈ {0, 16, 32, 48}. About 23 % of the 16-byte code loads cross a cache line. STATIC.
- **Activation traffic stays in L1:** per token per block, 64 B of activations plus 16 B of scale/hx, against 18 B of weights. About 8-10 loads per block at NT = 1, not binding on GLC's 3 load ports. STATIC.

##### Other per-expert costs

- **SiLU:** 640 × (`std::exp(double)` + `vdivss` + mul) per (expert, token). At about 40-80 cycles each, that is about 25-50 k cycles per (expert, token), against about 0.5-0.6 M cycles for the NT = 1 gate/up rows: about 4-8 % of gate/up. STATIC.
  - **Measured cross-check:** gate/up ms ÷ down ms = 4.914/2.454 = **2.002** (code, 13 workers, `%TEMP%/strata-claude-p0/rev-code-1-B.stdout`) and 2.968/1.476 = **2.011** (thai, `def-thai-1-A.stdout`). The byte ratio is exactly 2.0.
  - So per weight byte, gate/up (SiLU + `gbuf`/`ubuf`) costs the same as down (4× more rows per byte, so 4× the row-end reductions). The SiLU is a few percent, not a large term.
- **Quantizers (MEASURED, `rev-code-1-B.stdout`):**
  - "pool multi … quantize 0.121 ms/round" (the down-input quantize).
  - "dispatch … activation quantize 0.289 ms/round" (includes the input quantize).
  - A round is 2838.9 ms / 76 windows ≈ 37.4 ms, so the two together are about 1.1 % of a round. Both run serially on the host.

#### 2. Tokens per expert (NT): MEASURED from existing generate-mode logs

| run (`%TEMP%/…`) | workers | windows (T = window length) | CPU experts per layer, distinct / routed | mean NT |
|---|---|---|---|---|
| `strata-claude-p0/rev-code-1-B` | 13 | T1:3 T3:4 T4:69 T5:0 T6:0 | 4.55 / 5.64 | 1.24 |
| `strata-claude-p0/def-thai-1-A` | 13 | T1:105 T2:16 T3:6 T4:42 T5:1 T6:0 | 2.76 / 3.31 | 1.20 |
| `strata-claude-rest/code-*` | 6/9/13 | T1:3 T3:4 T4:70 | 4.04 / 4.86 | 1.20 |
| `strata-claude-rest/thai-*` | 6/9/13 | T1:117 T2:18 T3:13 T4:34 | 2.09 / 2.40 | 1.15 |

- **Consequence:** about 80-85 % of jobs are NT = 1 (inferred from the means; the histogram is UNMEASURED). NT ≥ 5 needs a 5-6-token lookup window and 5+ tokens on the same expert. One T5 window in 170 appears, so NT 5-8 cases in the kernel are effectively never reached.
- **Serve-mode caveat:** these are generate-mode runs at tier 8704 or older configs. The live serve log only gives entries: 301,533 CPU entries over about 868 windows ≈ 7.2 routed per layer-window (`strata-flash-next-d2x.log`, the request "2769 generated … cpu 301533; cpu experts 17429.5 ms"). The distinct count under serve is UNMEASURED; `d.multi_entries` is counted but the job count is not accumulated there.

#### 3. Compute-bound or DRAM-bound

**Scaling with workers, same-session (#16 table; rest runs):**

| workers (+ host) | CPU rows GB/s | pool code ms/round | code tok/s | thai tok/s |
|---|---|---|---|---|
| 6 | 27.0 (#16); 25.6-25.8 (rest runs) | 10.07 | 83.28 | 50.92 |
| 9 | 34.0 (#16); 32.0-35.9 | 8.03 | 86.84 | 53.20 |
| 13 | 42.8 (#16); 41.0-42.5 | 6.38 | 89.22 | 53.75 |

- **Per added E-core:** 6 → 9 adds 3 E-cores for +7.0 GB/s (2.33 each). 9 → 13 adds 4 E-cores for +8.8 GB/s (2.2 each). Linear, no saturation knee.
- **Per P-core** (6 P + 1 E threads at 6 workers): about (27.0 − 2.3)/6 ≈ 4.1 GB/s. This figure includes phase sync, because the "GB/s over the rows phases" is `multi_bytes / (gu_ms + down_ms)` (`generate.cpp:6713-6716`).
- **Against DRAM:** DDR5-7000 × 2 channels × 8 B = 112 GB/s theoretical. 42.8 GB/s is 38 % of that; practical read bandwidth on this box is UNMEASURED.
- **Per-core miss-level parallelism:** Little's law with about 48 L2 outstanding misses × 64 B / ~90 ns ≈ 34 GB/s per core, far above the 2-4 GB/s per thread observed. STATIC.
- **Verdict (STATIC reading plus the #16 numbers):** the pool is bound by per-core instructions and latency, not by DRAM bandwidth.
- **Hidden fixed cost:** the static kernel-only estimate is about 6 P × ≈ 7 GB/s + 8 E × ≈ 2.5-3 GB/s ≈ 60-65 GB/s, against 41-42.8 measured. The gap of about 1/3 would be phase sync (96 barriers per round), wake latency and tail imbalance, and it does not shrink with a faster kernel. UNMEASURED; the #33 timeline "wake" and "drain" spans per worker would split it.
  - Tail mechanism: tasks are equal-sized (`pool.cpp:402-406`), so an E-core that claims a late task at about half P speed sets the phase end.
- **Speedup needed to drop workers and keep 13-worker throughput** (measured GB/s ratios, same session):
  - 9 workers: 42.5/35.9 to 42.5/32.0 = **1.18-1.33×**.
  - 6 workers: 42.5/25.7 = **1.65×**.
  - Decode cost of 9 workers today (#16, same session): code −2.7 %, thai −1.0 %, both below the 13.6 % gate.

#### 4. Bit-exact speedups, ranked (output identical to today, so GPU Q2_0 parity #1 / ADR 0001 is kept)

**The bit-exact constraint, from the code:**

- Each 32-lane dword of `vpdpbusd` must hold exactly the four products for values 4j..4j+3, within the right 32-value chunk. Integer sums are exact, so the order inside a dword is free.
- Per-lane FP32 `fma` order: chunk 0 then chunk 1, block by block.
- The scalar `corr` chain order.
- The final reduction order above.
- `(float) exp((double))` for the SiLU.
- The quantizer's scalar `amax/127.f` and `1/s`.
- A consistent permutation of lanes that is undone before the reduction is also allowed. STATIC.

##### B1. Fuse gate and up rows in one loop, with an in-register SwiGLU (also acts as 2-row unrolling)

- **Mechanism:** process gate row r and up row r together. Mode 5 always needs both over the same r0..r1 (`pool.cpp:415-416`).
- **Savings (STATIC):**
  - One activation load feeds both `dpbusd`s (register operand instead of 2 memory operands).
  - `hx[2b] + hx[2b+1]` is computed once for both rows.
  - Two independent FMA chains remove the 8-cycle-per-block latency floor at NT = 1.
  - SiLU runs on the two finished row sums directly, so `gbuf`/`ubuf` (40 KB of thread_local, `:410`) and a second pass go away.
  - Registers: 2 × NT accumulators + 4 code vectors fits 16 ymm for NT ≤ 4 (NT = 4: 8 acc + 4 + temps, tight; check for spills).
- **Why it stays bit-exact:** each row's operation sequence is unchanged. Only interleaving and sharing of identical subexpressions change.
- **Expected (STATIC):** about 43 → 38 µops per block at NT = 1 (−12 %) plus the latency-floor removal. **About 10-20 % on phase 5**, which is 2/3 of rows time, so about 7-13 % of pool rows. The same 2-row interleave applies to down rows (B5).
- **Measure:** see §6. Arms: `rows<1>` gate, then `rows<1>` up, then the scalar SiLU loop (today's `pool.cpp:415-419` copied verbatim), against the fused kernel. Compare output bytes with `memcmp`, all NT ∈ {1, 2, 3, 4}.

##### B2. Vectorised SiLU equal to `(float) std::exp((double) -g)` on all 2^32 inputs

- **Mechanism:** AVX2 double-precision exp, 4 lanes; convert to float; then vector `1.f + e`, `vdivps`, `vmulps`. Each is a single correctly-rounded IEEE op per lane, identical to the scalar ops. This moves the loop into the `/arch:AVX2` TU.
- **Bit-exactness is established by exhaustive check, not argument:** loop over all 4,294,967,296 float bit patterns and compare the full SwiGLU denominator (and `inf`/`NaN`/overflow cases, e.g. g < −88 → `exp` overflows the float) against the scalar expression compiled by the same MSVC 14.44.35207 UCRT. About 4.3e9 × 20-40 ns ≈ 1.5-3 min per core, parallelisable.
- **Transitivity:** the CPU output is unchanged, so parity with the GPU's `(float) exp((double) -g)` (`iq_kernels.cu:512`) is unchanged too, without touching the GPU.
- **Expected (STATIC):** exp is about 4-8 % of gate/up (bounded by the 2.00 per-byte gate/up:down ratio in §1). About 85 % of that is removed, so **about 3-6 % of phase 5, about 2-4 % of pool rows**.
- **Measure:** the SiLU loop alone on 640 × NT real `gate`/`up` float rows captured from a run (or normally distributed values with the measured variance), ns per element; then inside the B1 kernel.

##### B3. Take `d × scale` and broadcasts off port 5; hoist the per-token-per-block invariants

- **Mechanism:**
  - Per block, broadcast `d` to a vector once (shared by all tokens and both chunks).
  - Per token, `vbroadcastss ymm, [scale]` from memory (a pure load µop), then `vmulps(d_v, s_v)`. Each lane computes the same IEEE product as the scalar `d * scale` at `:60-61`, so it is bit-exact.
  - Precompute `hxs[t][b] = hx[2b] + hx[2b+1]` once per activation (same op, same result), saving 1 load + 1 add per token-block.
- **Expected (STATIC):** about 2-4 fewer µops per token-block; moves 2 p5 shuffles per token-block to the load ports. **About 3-8 %** if port 5 is the binding port. The unpack alone puts about 10 shuffle/insert µops on p1/p5. UNMEASURED.
- **Caveat UNVERIFIED:** whether MSVC contracts `d * (…) + corr` into an FMA today. MSVC 14.44, `/fp:precise` default, no `/fp:contract` in CMakeCache `CMAKE_CXX_FLAGS=/DWIN32 /D_WINDOWS /EHsc`, so it should not. Confirm with `dumpbin /disasm` on the `q2_avx2.obj` hot loop before changing the scalar chain.

##### B4. Software prefetch

- **Mechanism:** `_mm_prefetch` of the rows about 1-2 KB ahead, plus the first lines of the next task's gate/up/down ranges at task start (each task covers about 69 rows ≈ 50 KB per matrix, code run).
- **Expected (STATIC):** **0-3 %**. The kernel is compute-bound, and the L2 streamer already covers sequential streams at 2-4 GB/s per core. The only gain is at task starts and 4 KB page crossings, where the Intel L2 streamer re-arms.
- **Measure:** A/B the same kernel with and without prefetch, at 1 thread and at 14 pinned threads (P+E), on a 1+ GB weight buffer so the data comes from DRAM.

##### B5. Down rows: 2-4 rows interleaved, and a batched row-end reduction

- **Mechanism:** down rows are only 10 blocks, so the per-row reduction and store (about 8 µops + NT stores) is about 4-5 % of down.
  - Interleave rows to break the FMA chain (same as B1).
  - For 4-8 rows, do the reduction as a transpose-add that keeps each row's order ((a0+a4)+(a2+a6))+((a1+a5)+(a3+a7)).
- **Expected (STATIC):** **about 5-10 % of phase 6**, which is about 1/3 of rows time.

##### B6. Vectorised int8 pack in the quantizer, and the host-serial problem

- **Mechanism:** replace the scalar store loop at `q2_avx2.cpp:138` with `packs_epi32` → `packs_epi16` → `permute4x64`. Values are already clamped to [−127, 127], so the saturating pack is the identity. Bit-exact.
- **Expected:** removes about half of each chunk's µops. The measured total is about 0.41 ms/round (0.121 + 0.289, `rev-code-1-B`) of about 37.4 ms, so **about 0.5 % of a round**.
- **The bigger point:** the down quantize (`pool.cpp:545-548`) is serial on the host while 13 workers wait. Moving it into the workers (each (expert, token) quantized by whichever thread finishes that expert's last phase-5 task, or by the first phase-6 claimant) is also bit-exact. Its effect is about the barrier, not the kernel. UNMEASURED.

##### B7. Layout: aligned structure-of-arrays repack of host-owned experts

- **Mechanism:** at arena commit, store fp16 `d` separately (8 per `vcvtph2ps ymm` instead of 8 × (`movzx` + `movd` + `cvtph2ps`)) and 16 B-aligned codes (no split loads). Same values, so bit-exact.
- **Expected (STATIC):** about 3 µops per block (about 7 % at NT = 1), plus the removal of about 23 % split loads.
- **What it breaks** (format, not numerics):
  - The host blob is also the source for adaptive swaps into VRAM and the 4070 (`--adapt-swaps 8`) and for placement-first loads.
  - GPU kernels read the GGUF block form, so it needs a CPU-only copy or a convert-on-swap.
- **Cost:** 8 swaps per round × 1.38 MB of conversion. Medium effort, lowest rank for effort versus gain.

##### B8. NT 5-8 cases without re-unpack

- **Mechanism:** `rows<5..6,Vnni>` direct templates instead of `q2_avx2.cpp:91-95`. Registers: 6 acc + 2 codes + temps ≈ 12 ymm; NT = 8 risks spills.
- **Expected:** saves about 22 µops per block plus an L2 re-read, but only for NT ≥ 5. That was 0 of 76 code windows and 1 of 170 thai windows (§2). **About 0 % overall (MEASURED frequency).** Implement only for completeness.

##### B9. Large pages

- **Expected (STATIC):** **0-2 %**. One 4 KB page is about 228 blocks ≈ 2 k P-core cycles, against a walk of about 20-40 cycles. The Intel L2 streamer does not cross 4 KB boundaries whatever the page size.
- **Cost:** SeLockMemoryPrivilege plus non-pageable memory (memory note `strata-large-pages-tradeoff`, "developer decides").
- **NUMA:** single socket, so NUMA placement does not apply. The E-core cluster (4 cores sharing one L2 and one ring stop) is the only placement question, and it is a scheduling choice, not a kernel one.

##### B10 (scheduling, bit-exact, adjacent to the kernel). Tail balance on mixed P/E cores

- **Mechanism:** equal tasks (`pool.cpp:402-406`, `mtasks_ = 3 × threads`, `:540`) let an E-core finish last. Options: finer grain at the end, or P-cores claiming tail tasks first.
- **Expected:** UNMEASURED; bounded by the about 1/3 gap in §3. This becomes more important with fewer workers.

**Combined static estimate:** B1 + B2 + B3 + B5 ≈ 1.15-1.3× on rows time. That meets the 9-worker requirement (1.18-1.33×) but not 6 workers (1.65×). All UNMEASURED.

#### 5. Non-bit-exact options

##### N1. Plane layout / activation pre-permutation (drops the unpack interleave)

- **Mechanism:**
  - Build `X = [P0 | P1]` and `Y = [P2 | P3]` from one `vbroadcasti128` plus `vpsrlvd` shifts {0,2} and {4,6} plus 2 `pand`. About 5 µops per 64 codes instead of about 18.
  - Pre-permute the ActQ bytes to plane order in the quantizer (free: a `pshufb` at store time).
  - Every dword then mixes values 4j+m of 4 different bytes, and each vector spans both chunks. The scale vector per token-block becomes a lane pattern [s0,s0,s1,s1,…] (one `vbroadcastsd` of the adjacent `scale[2b..2b+1]` pair plus one permute).
  - Optional further step: chain the two `dpbusd` into one int32 accumulator. That halves the `cvt`/`fma` per token-block and the FMA chain.
- **Expected (STATIC):** NT = 1 goes from about 43 to about 24-28 µops per block, **about 1.5-1.8×**. Reaches the 6-worker requirement on paper.
- **What it breaks:**
  - Lane grouping of the integer sums and the FP32 accumulation order. That changes CPU-only output and breaks bit-exactness with the GPU CPU-order path (`iq_kernels.cu:322-357`: `q2_spread` per byte plus the 8-lane order).
  - The GPU kernel must be changed in lockstep, or parity #1 and ADR 0001 (`docs/adr/0001-q2-exp-baseline.md`) lapse.
  - The AVX-512 path (`expert.cpp`) and the AVX2 arm must match too, or output depends on the host CPU.
  - Invalidated: `tests/xeno/q2_isa_parity.cpp`, `native_q2_pool_hit_parity.cpp`, `cache_tokens.py`, the final256 ID checks and every A/B that uses output hashes as identity.
  - Needs the project's quality gate (the maths is equivalent but the rounding differs).

##### N2. Fast float SiLU (`__expf`-class polynomial)

- About 1-2 % more than B2. It changes output (the non-CPU-order GPU branch at `iq_kernels.cu:514` already does this). Not worth breaking ADR 0001 when B2 exists.

##### N3. Split `acc` per chunk (two accumulators)

- Breaks the FP32 order. B1 gets the same ILP bit-exactly, so there is no reason to do this.

#### 6. Microbenchmark design (CPU-only, no engine, no GPU)

**Harness:**

- Extend the pattern in `tests/xeno/q2_isa_parity.cpp` (deterministic weight bytes, fp16 scales, `act_quant_q8_1_avx2` activations, bitwise compare).
- Weights: either a real layer's blobs read from `packs/q2_0` (or the GGUF at the `native_experts.txt` offsets, read-only) or synthetic ones.
- Use ≥ 1 GB of weights so the data really comes from DRAM, and walk experts in a random order.

**Functions under test:**

- (a) Today's mode-5 body (`pool.cpp:408-419`, copied verbatim, including the scalar SiLU) against B1/B2/B3 variants.
- (b) Today's `q2_rows_any` for down (10 blocks) against B5.
- (c) `act_quant_q8_1_avx2` against B6.
- (d) The SiLU loop alone.
- (e) An exhaustive 2^32 check for B2.

**Workload shape:**

- NT ∈ {1, 2, 3, 4} weighted by the measured mix (about 80 % NT = 1).
- 4-5 experts per layer-phase.
- Row ranges of about 69 rows per task (code run: 4.55 × 640 / 42).

**Threads:** pinned exactly like `physical_cores(true)`. Run three configs: 1 P-core, 1 E-core, and {6, 9, 13} workers + host with the same claim protocol. The best option is to call `ExpertPool::run_split_multi_native` itself with fake `ExpertJobMulti` jobs and a `NativeFmt` 42/42, so sync is included.

**Gate:** `memcmp` of all outputs (`ff` and `out`) against the baseline, for every NT and every seed, before any timing counts. Add a mutant (for example, swap one FMA order) that must fail.

**Timing and ordering:**

- ABBA within one process, alternating variant order per iteration, at least 200 iterations per arm.
- Report median and p10/p90 ns per block and GB/s. Record the exe sha, per the Strata parity note.
- Thread priority as in serve (HIGHEST inside HIGH class).
- Desktop idle, no game running.

**Counters, if available:** µops retired per block via Intel PCM or VTune (UNVERIFIED availability on this box). That would turn the static µop estimates here into measured ones.

**Engine confirmation afterwards (the developer's run, not this task):** a generate-mode ABBA on the D2x config at 13 and 9 workers. Read the "pool multi" and "verify window … pool" lines and tok/s, with output hashes identical.

#### 7. The desktop-neighbour findings (outside the kernel, directly on the stutter goal)

- **`pool.rest()` has one call site:** `generate.cpp:6592`, in the generate-mode loop (grep here; the blueprint report §3.1 P6 and §8 item 6 say the same). Under `--serve`, parked workers spin `_mm_pause` for up to `kSpinBeforeSleep = 20 ms` (`pool.hpp:167`; loop at `pool.cpp:277-291`) before sleeping.
  - A decode round is about 37 ms, and phases come every few hundred µs within a window.
  - So in serve, the 13 workers are effectively always spinning during generation.
- **They spin at elevated priority:** `--pool-priority 2` gives THREAD_PRIORITY_HIGHEST (`pool.cpp:130-133`, `:221`), and `--process-priority 2` gives HIGH_PRIORITY_CLASS (`generate.cpp:2440`). They are hard-pinned to all 13 non-host physical cores.
- STATIC reading: this probably costs the desktop more than the worker count itself, and no kernel speedup changes it.
- **Bit-exact, zero-kernel options** (all UNMEASURED):
  - Call `rest()` at the serve loop's window end.
  - Shorten the spin under serve.
  - Drop `--process-priority` to 0 or 1 (it measured code 76.76 → 78.73 in `0a2d70d`, which is inside noise).

#### 8. Gaps

- No disassembly was read; every µop and cycle count above is STATIC.
- The per-expert NT histogram and the serve-mode distinct count are UNMEASURED.
- The split of pool rows time into kernel versus sync/tail is UNMEASURED (the #33 timeline would give it).
- Practical DRAM read bandwidth on this box is UNMEASURED.
- The i5-13500 all-core clocks under this load are UNMEASURED.
- The per-phase GB/s includes barrier time.
- A background `grep` over `%TEMP%` that I started (read-only search) may still be running; it has no side effects.

## Appendix D. The upstream gap (survey, verbatim)

#### Verdict

The gap between upstream and our CPU pool (22.9 vs 27.9 ms/round) is real in wall time, but it is not a like-for-like comparison. Our CPU pool received about 34 % more distinct experts per layer: 14.46 against 10.81. Per expert, our pool and kernel are as fast as upstream's or faster. None of our pool or kernel changes (P2, P3, P6, VNNI) accounts for the gap. It comes from expert placement: a different profile file, a slower adaptive-swap rate, and a PCIe miss share that our exclusive mode forbids.

#### 1. Where the number comes from, and under what conditions

- **Source:** #45, step-1 comment, "Per-stage latency (code, ms/round)" table: `CPU pool | 22.9 (U1) | 108.3 (U2) | 27.9 (O1) | 13.4 (O2)`. It is copied into the fork-delta report at `docs/reports/2026-09-30-fork-delta-and-blueprint.md:85`.
- **The arm is O1, not D.** O1 = Strata-xeno `7134ce9`, run as `%TEMP%\strata-claude-p0\d4def.exe`. It is not the merged engine `5c51574`.
- **Harness:** `C:\Strata-exp\m1.py`, log `m1.log`, raw counters in `C:\Strata-exp\m1-code-{1,10}-U1.stdout` and `m1-code-{4,7}-O1.stdout`.
- **Same between the arms:**
  - one session, order U1 U2 U3 O1 O2 O2 O1 U3 U2 U1;
  - one card (5060 Ti), HIGH class set by `run_up.py`;
  - 13 workers + the host thread (both stderr files: "13 expert-pool workers + the host thread");
  - `--spec 4 --spec-min-p 0.5 --max-context 16384 --kv int8`;
  - 76 rounds and about 3.4 tokens per round in both;
  - equal expert size, 1.38 MB largest blob in both.
- **Different between the arms:**

| | U1 (upstream `4c68013`) | O1 (`7134ce9`) | source |
|---|---|---|---|
| pack | `C:\Strata-exp\pack-q2_0` | `D:\Github\Strata\packs\q2_0` | stderr line 3 |
| profile | `C:\Strata-exp\src\data\expert-profile.bin`, **24,576 pairs** | `D:\Github\Strata\data\expert-profile.bin`, **8,000 pairs** | `m1.py` OURARGS `.replace(...)`; stderr "ranked pairs" |
| cache slots | 7,503 | 7,529 | stderr "expert cache ... slots" |
| PCIe share of misses | `pcie_frac 0.15` (probe); "1.25 distinct experts per layer read over PCIe" | 0 ("pcie 0.0%") | stdout |
| adaptive tier | "1824 experts swapped ... every 4 rounds" (96/4) | "304 experts swapped ... every 1 rounds" (exclusive 8/1) | stdout |
| exclusive / tail | no | "slots 5239.. are the lendable tail" | stderr |
| pool priority | none | `--pool-priority 2` by default at `7134ce9` | report §P6 |
| CPU row kernel | "the expert kernels run on AVX-2" | "native Q2_0 expert rows use AVX-VNNI" | stderr line 1-2 |
| **CPU experts per layer** | **10.81 distinct / 14.75 routed** | **14.46 distinct / 20.36 routed** | stdout `verify window` |

#### 2. Normalised per expert, from the existing logs (no new runs)

I computed ms per distinct expert per layer as `pool` divided by `distinct`. The table covers every `C:\Strata-exp\m*-*.stdout` with the arms U1, O1 or D1.

| run | pool ms | distinct | GB/s | ms/distinct |
|---|---|---|---|---|
| m1-code-1-U1 | 22.90 | 10.81 | 33.2 | 2.118 |
| m1-code-10-U1 | 22.55 | 10.81 | 33.8 | 2.086 |
| m1-code-4-O1 | 27.86 | 14.46 | 36.3 | 1.927 |
| m1-code-7-O1 | 26.56 | 14.46 | 37.9 | 1.837 |
| m9-code-1/10-U1 | 21.00 / 22.04 | 10.81 | 36.5 / 34.6 | 1.943 / 2.039 |
| m9-code-2/9-O1 | 27.34 / 31.91 | 14.46 | 37.0 / 31.5 | 1.891 / 2.207 |
| m9-code-3/8-D1 (merged `dyn.exe`) | 26.98 / 30.23 | 14.54 | 37.7 / 33.3 | 1.856 / 2.079 |
| m12-code U1 / U1L (4 runs) | 20.30-21.16 | 10.76-10.81 | 36.3-37.5 | 1.886-1.957 |
| m1-thai U1 (2 runs) | 7.06 / 9.57 | 3.52 / 3.59 | 36.0 / 26.7 | 2.007 / 2.666 |
| m1-thai O1 (2 runs) | 11.09 / 9.82 | 5.85 | 37.4 / 42.1 | 1.895 / 1.679 |

- **The work accounts for the whole gap.** For the rows, U1 moves 21.6 ms × 33.2 GB/s = 717 MB/round and O1 moves 26.4 ms × 36.3 GB/s = 959 MB/round. That ratio is 1.337, and the distinct-expert ratio 14.46/10.81 is 1.338. The pool's "GB/s" is distinct experts × `BLOB` over the rows time, so it is a direct per-expert speed.
- **The merged engine matches upstream once placement matches.** In `m8.log` (serve, one card, same session), D1u is the merged engine with upstream's profile and 7,287 slots.
  - code: D1u 70.97 / 70.85 tok/s against U1s 71.35 / 70.10;
  - 8K: 70.93 / 71.85 against 65.40 / 65.76.
- **The VNNI advantage is not established.** m1 shows O1 about +9-12 % GB/s over U1, but m9 does not (O1 37.0 / 31.5 against U1 36.5 / 34.6). Both are below the 13.6 % gate.

#### 3. What does and does not cost CPU time per round

- **P2, P-core ordering** (`pool.cpp:40-68` at `5c51574`): no effect on this CPU. `physical_cores(true)` skips the first core (`pool.cpp:204`), so 13 workers land on 5 P-cores + 8 E-cores. By static reading, upstream's enumeration order already lists P-cores first on the i5-13500. Unmeasured as an isolated A/B.
- **P6, priority and rest:** no CPU cost; `--pool-priority 2` made the pool faster in `2026-09-28-cpu-pool-sweeps.md` sweep 2 (code 57.0 → 66.8). One finding matters here: `pool.rest()` is called only at `generate.cpp:6592`, so under `--serve` it never runs (the report verified this by grep).
- **P3, SwiGLU with `exp(double)`** (`pool.cpp:419`): not a measurable part of the gap. Both builds call a scalar exp:
  - `dumpbin /disasm` of upstream's `build\...\pool.cpp.obj` shows one call to `expf`; ours (`build-dyn`) shows one call to `exp`. Neither calls a vectorised `__vdecl_*` routine.
  - `pool.cpp` is compiled with no `/arch` flag (`build-dyn/build.ninja:2605-2608`: `/O2 /Ob2`).
  - Volume: FF = 640 values per routed entry. That is about 625k exp calls per round in O1 and about 242k in D2x (7.87 routed × 48 × 640). The float-vs-double difference is UNMEASURED; by static reading it is well under 1 ms/round.
- **VNNI rows** (`q2_avx2.cpp:50-56`): if anything a small gain (table above).
- **Placement, which is the whole gap:**
  - **Profile:** 8,000 pairs against 24,576. With nearly equal slot counts, the 8,000-pair ranking leaves 3.65 more distinct experts per layer on the CPU.
  - **Adaptive-swap rate:** upstream 96 every 4 rounds (1,824 per run) against our exclusive 8 every 1 (304). This is placement, not pool cost.
  - **PCIe share:** U1 moved 1.25 distinct experts per layer off the CPU over PCIe. Exclusive mode requires `pcie_frac == 0` (report §P8 eligibility), so our default cannot use that path by design. `--pcie-frac 0` was a measured choice (#27: the PCIe path had collapsed serve 8K decode to 3.24 tok/s).
  - **Tail file:** not a cause. The lent slots are refilled before decode ("192 lent slots refilled in 82.9 ms").
- **The 4070 tier moving hot experts off the CPU** is the reason O2 is 13.4 ms, not a cost.
- **Correction to #45 step 1:** its text says "upstream's `--expert-cache auto` puts more experts on the card". The logs contradict this: the slot counts are equal (7,503 vs 7,529). The difference is the profile, the adaptive-swap rate and the PCIe share.

#### 4. The cheapest experiment that splits it

**Zero runs.** Section 2 already splits the gap: ms per distinct expert and GB/s are equal or better for O1 and D1 in the same sessions (m1, m9).

**If you want it confirmed on the counters:** one card, generate mode, same session, order A B C C B A, 256 tokens code. About 6 runs of about 30 s.
- A: `dynfix.exe` with the m1 OURARGS (8,000-pair profile).
- B: A with `--expert-profile C:\Strata-exp\src\data\expert-profile.bin`.
- C: upstream U1.

Read `verify window ... CPU experts distinct` and `pool multi ... GB/s`.
- Static prediction: B's distinct count falls to about 11-12 and its pool to about 21-23 ms. The remaining difference would be the PCIe share (about 1.25 distinct per layer), which exclusive mode cannot take.
- A vs B isolates the profile. B vs C isolates PCIe plus the swap rate.
- `STRATA_FORCE_AVX2=1` on B isolates VNNI in the same session.

**Caveat:** this uses the GPU. It needs the live `dynfix.exe` on port 8091 stopped, or the 31 GB RAM cap respected. I did not run it.

#### 5. What this means for going from 13 workers to 6-10

Everything in this section is static reading or from existing logs; nothing here was measured for the goal.

**The live pool does little work per round:**
- Live config D2x (m11-code-1-D2x.stdout): the `pool` counter is 15.97 ms/round, but only 11.28 ms of it is rows (gate/up 7.38 + quantize 0.18 + down 3.72).
- The rest:
  - plan 2.77 ms, which contains the 4070 `launch` 2.56 ms, serial before the pool publishes (`expert_source.cpp:935-948`, before `c1`);
  - act quantize 0.35 ms, jobs 0.63 ms, 4070 finish 0.88 ms.
- The workers sit parked during all of these.

**Workers spin at the top of the priority range:**
- A parked worker spins for `kSpinBeforeSleep{20}` ms (`pool.hpp:167`), with `STRATA_POOL_SPIN_US` as a knob (`pool.cpp:203`).
- Under `--serve` nothing calls `rest()`, and the gaps between windows are shorter than 20 ms. So during decode 13 threads spin on 13 physical cores at THREAD_PRIORITY_HIGHEST in HIGH class (`--pool-priority 2 --process-priority 2`, `strata-flash-next-d2x.json`; `generate.cpp:2440`). That makes base priority 15, from Windows documentation.
- Only the 6 HT siblings remain for the desktop.
- This is the likely cause of the stutter; UNMEASURED.
- One old trace (`C:\AI\UsersxenodAppDataLocalTempstrata-claude-stagetl1.json`, 16 decode rounds, config unknown) shows:
  - `cpu pool` is 6.85 ms of a 31 ms verify window;
  - the median `wake` is 0.4 µs, so the workers were spinning, not asleep.

**Live serve share:** summing the current boot's request metrics in `strata-flash-next-d2x.log` from line 205 gives 260 requests, `cpu experts` 1,567 s against 5,345 s of decode (29.3 %), and CPU entries 21.7 % of routed.

**The kernel's likely limit:** the pool does not saturate RAM bandwidth.
- About 3.3 GB/s per thread: 42.8 GB/s at 13 workers + rest (memory note), against 19-24 GB/s at 6 workers (`cpu-pool-sweeps.md`).
- NT = 1 dominates (routed/distinct ≈ 1.26-1.4). `row_multi<1>` (`q2_avx2.cpp:38-68`) runs one `acc` chain with 2 dependent FMAs per 18-byte block, plus a scalar `corr` add chain. By static reading it is bound by that chain's latency.
- **Candidate kernel lever:** interleave 2-4 rows per iteration in `rows<NT,Vnni>`. Each row keeps its own accumulation order, so it stays bit-exact with the CPU-order GPU path of ADR 0001. The activation loads are shared. Gain UNMEASURED.

**Levers that need no rebuild, for the 6-10 worker goal:**
- `STRATA_POOL_SPIN_US` (about 1-2 ms).
- `--pool-priority 1`.
- `--pool-workers 10`. The sort drops E-cores first, so 10 workers = 5P + 5E.

Each should be an ABBA on the dual config in both orders, per the memory note's drift warning.

**Rebuild levers:**
- a `pool.rest()` call in the serve loop, mirroring `generate.cpp:6592`;
- the async 4070 launch on a P-core. It was measured only on an E-core, where `finish()` waited 4-5.8 ms, so it was left opt-in.

## Appendix E. The checker's review and ranked plan (survey, verbatim)

### Pool review for #51: map claims checked, and a ranked plan for fewer workers with no desktop stutter

#### 0. Summary
1. **The cause of the stutter has not been measured, and the pool spin is only one of three candidates.**
   - **Pool spin.** During decode the workers spin at priority 15 on all 14 physical cores.
   - **The display card computes experts.** CUDA device 1 is the 4070 SUPER (`start-flash-next.ps1:38`). The log shows `expert_split (whole layer) on device 1: 1755 MiB` and about 4,400–4,800 experts per prompt part "from the 4070's slots" (`strata-flash-next-d2x.log` lines 38–45). Line 22 says "SECONDARY COMPUTE", so the display card also runs expert kernels during decode.
   - **Prefill stager threads.** 4 unpinned threads at base 13 copy host experts to the 4070 during prompt reads (`prefill.cpp:641`, `:1290`).
   - Only the first candidate involves the worker count or the kernel.
2. **A faster kernel does not free a single CPU for the desktop by itself.** Idle workers keep spinning at priority 15. A faster kernel shortens the pool's work, so the workers spend more of each round spinning.
   - What frees CPUs for the desktop: idle workers sleep or yield, core parking stops hiding the free CPUs, or fewer workers are pinned.
   - The kernel's job is tok/s per worker, which is what makes fewer workers affordable.
3. **Core parking is active on this machine.** This is new, read-only evidence.
   - The active plan is Ultimate Performance. The setting "core parking min cores" is 4 % and its Class 1 variant is 0 %. The performance check interval is 15 ms.
   - I read `\Processor Information(0,*)\Parking Status` at 12:54:44–46 while the engine was idle (the log stayed at 2,434 lines). 14–16 of the 20 logical CPUs were parked, including the SMT siblings 1, 3, 5 and 7.
   - Hypothesis: during decode the desktop does not reliably get even the 6 "free" siblings. This would fit #16's first 13-worker attempt, where `join wait` rose from 0.9 to 19 ms/round for the adapt thread (base 13).
   - This mechanism is UNMEASURED.

#### 1. Claims checked (code at `C:/Strata-exp/src-dyn` @ 5c51574)

##### MAP idle
- **VERIFIED:**
  - park loop at `pool.cpp:275-291`; clock read every 1024 pauses at `:280`; no yield;
  - sleep path at `:283-290`;
  - `kSpinBeforeSleep{20}` at `pool.hpp:167`;
  - `STRATA_POOL_SPIN_US` at `pool.cpp:203-204`;
  - `publish` at `:236-249`; epoch-tagged claim at `:314-323`;
  - `wait_parked` and `wait_done` are pure pause spins at `:339-379`;
  - the host drains at `:466`;
  - the only `pool.rest()` is `generate.cpp:6592`. The serve block runs from `:4155` to `return 0` at `:5704`, and its windows are the `ver.run` calls at `:5184` and `:5447`;
  - the daily worktree has the same single call site at `generate.cpp:5636` (@284bb19);
  - `--pool-rest` defaults to 1 (`:239`), and its help text (`:531`) is wrong for `--serve`;
  - `short_read = 64` at `:379`; `in_cv.wait` at `:4779-4780`; prefill never uses `ExpertPool` (grep).
- **REFINED with a trace.** The map estimated 70–80 %+ spin. Measured in tl1 (§2A): pool phases are 19.5 % of a round, and workers are busy 81.9 % of phase time, so a worker is busy about 16 % of the round. Under serve (no rest) that leaves about 84 % spinning. On D2x code the CPU pool is 27.7 % of the round (m11), so about 77 % spinning (static from counters).
- **VERIFIED:** the logical CPU numbering, which the map had left UNVERIFIED. My `GetLogicalProcessorInformationEx` query: P0–P5 = logical CPUs {0,1} … {10,11} (EfficiencyClass 1, SMT); E0–E7 = 12–19 (class 0). Workers get 2,4,6,8,10,12–19 and the host gets 0.

##### MAP scheduling
- **VERIFIED:**
  - `physical_cores` at `pool.cpp:32-108`; stable_sort at `:65-66`; skip_first at `:106`;
  - hard pin at `:115`; per-worker priority at `:221`;
  - `SetPriorityClass` at `generate.cpp:2439-2441`;
  - the host pin at `session.cpp:516-518`;
  - the async-launch helper is unpinned at 13 workers (`generate.cpp:3141-3143`: `pool_workers < cores.size()` is false);
  - the host spin on the GPU ring at `verify.cpp:1026-1027`, with `cudaStreamQuery` every 2 ms;
  - `0a2d70d`: HIGH vs normal class, code 76.76 → 78.73 and thai 48.72 → 49.33 (commit message);
  - the "E-cores stay free" text is present in #14's body.
- **MISSED:** `set_current_thread_priority(o.pool_priority)` also sets the HOST thread (`generate.cpp:2444`). Any `--pool-priority` arm therefore also moves the critical-path host.
- **QUALIFIED:** "the desktop is confined to the six SMT siblings" is right in static terms. The parking sample (§0.3) and #16's 19 ms/round `join wait` suggest the siblings are not reliably usable in practice (hypothesis).
- **NOT RE-CHECKED:** the thread-priority snapshot of the live `dynfix` process, to avoid touching it.

##### MAP kernel
- **VERIFIED:**
  - dispatch at `expert_layout.cpp:73-77`;
  - `/arch` only on `expert.cpp` and `q2_avx2.cpp` (`CMakeLists.txt:616-617`), so `pool.cpp` has no `/arch`;
  - the mode 5/6 bodies at `pool.cpp:400-436`; SiLU double `exp` at `:419`; `mtasks_ = 3*threads` at `:540`;
  - host-serial down quantize at `:545-548`;
  - NT 5–8 regrouping at `q2_avx2.cpp:91-95`; the scalar int8 store at `:138`;
  - the row-end reduction order at `:65-69`;
  - `suffix_draft = 3` at `:384`; the spec bump at `:1785-1789`.
- **CONFIRMED, previously an estimate.**
  - The µop count: `dumpbin /disasm` of `build-dyn/.../q2_avx2.cpp.obj`, `rows<1,1>` block loop at 0xC0–0x18A.
  - It has 41 instructions per block, about 33 of them vector-ALU µops.
  - Breakdown: 3 vpsrlw, 4 vpand, 8 vpunpck, 2 vinsertf128, vmovd, vcvtph2ps, 3 vmulss, 2 vbroadcastss, 2 vpdpbusd, 2 vcvtdq2ps, 2 vfmadd231ps, 2 vaddss.
- **RESOLVED (was UNVERIFIED):** FMA contraction. The object has 0 scalar `vfmadd*ss` and 80 packed `vfmadd*ps` in 3,423 lines, so the `corr` chain is a separate mul and add today.
- **WRONG:** "the hidden ~1/3 is phase sync and wake".
  - tl1 wake while spinning: p50 0.3 µs (P-cores) and 0.5 µs (E-cores).
  - The host-serial quantize is 2.5 µs mean per layer (0.114 ms/round).
  - The recoverable loss is tail imbalance: 18 % of worker time inside phases.
- **WRONG:** "B6 is about the barrier". The down quantize measures 0.114 ms/round (tl1) and 0.16–0.18 ms/round (D2x m11), under 0.5 % of a round.
- **INCOMPLETE:** the map says the unpack interleave can only be removed by the non-bit-exact N1. A bit-exact cheaper unpack exists (K1, §3).
- **UNPROVEN:** "compute-bound". Linear E-core scaling does not separate per-core compute limits from per-core memory parallelism limits. See the M-D microbenchmark.
- **UPDATED:** "13 workers = 42.8 GB/s" is from #16's older config. D2x measures 37.4 / 38.4 (code), 39.8 (thai) and 41.3 / 38.2 GB/s (8k), all in m11.
- **VERIFIED:** B8 is about 0 %. The window histogram is `T5:0 T6:0` in all m11 D2x runs.

##### MAP upstream-gap
- Correct for O1 vs U1, but it does not apply to the live config.
- D2x already uses the 24,576-pair ranked profile (log line 9: `ranked-exl3only-profile.bin: 24576 ranked pairs`), so the profile lever is already taken.
- Its D2x counters check out against `C:/Strata-exp/m11-code-1-D2x.stdout`:
  - `pool 15.971`; `gate/up 7.385 quantize 0.180 down 3.715`;
  - `plan 2.768`; `secondary timing launch 2.558`;
  - `CPU pool 11.291`; `join wait 1.451`.
- `cpu_ms` covers the whole dispatch: VERIFIED at `generate.cpp:995-997`.

#### 2. New data from existing artifacts (no runs)

##### A. Timeline traces
Source: `C:/AI/UsersxenodAppDataLocalTempstrata-claude-stagetl1.json`.
- 13 workers, generate mode, rest on, 16 rounds, written 2026-09-29 12:12. The exe and config were not recorded.
- `tl1b` reproduces these figures within 1–3 %.

| Measure | Value |
|---|---|
| Round | 34.50 ms |
| Verify window | 31.12 ms |
| Pool phases per round | gate/up 4.463 + quantize 0.114 + down 2.256 = 6.83 ms (19.5 % of the round) |
| Gate/up phase | mean 97.7 µs, p50 78.4, p90 190.8 |
| Down phase | mean 49.4 µs, p50 39.3, p90 95.9 |
| Wake while spinning | P p50 0.3 / p90 0.5 µs; E 0.5 / 0.6 µs |
| Wake after sleep (first phase after a gap over 3 ms) | P p50 56.7 / p90 93.3 / max 362 µs; E p50 70.6 / p90 153.5 / max 275.5 µs |
| First phase of a window vs others | 299.7 vs 71.0 µs mean |
| Idle tail per worker, gate/up | P 21.3 µs mean (p90 48.2); E 16.8 µs |
| Idle tail per worker, down | P 12.2 µs; E 9.7 µs |
| First-to-last drain end, gate/up | 27.7 µs mean (p90 57.1) |
| Worker utilisation inside phases | 0.819 |
| Gaps between layers inside a window (n = 715) | p10 254, p50 372, p90 571, p99 793, max 1,059 µs |
| Gaps between windows | p10 4.18, p50 6.74, p90 23.6 ms |

Time each worker would sleep per round if it slept after spinning for T µs:

| T (µs) | Sleeps per round | Asleep per 34.5 ms round |
|---|---|---|
| 200 | 45.6 | 17.4 ms |
| 500 | 10.1 | 9.27 ms |
| 1,000 | 1.1 | 7.89 ms |
| 2,000 | 0.9 | 6.95 ms |

##### B. D2x in generate mode
Source: `C:/Strata-exp/m11-*-D2x.stdout`. 13 workers, rest on.

| Run | tok/s | Round (ms) | CPU pool (ms/round) | Pool share | Rows GB/s | Distinct / routed per layer | Join wait (ms/round) |
|---|---|---|---|---|---|---|---|
| code-1 | 83.62 | 40.8 | 11.29 | 27.7 % | 37.4 | 6.26 / 7.87 | 1.451 |
| code-6 | 86.54 | – | 10.97 | – | 38.4 | – | 1.407 |
| thai-1 | 50.76 | 27.9 | 5.63 | 20.2 % | 39.8 | 3.32 / 3.85 | 2.053 |
| 8k-1 | 68.08 | 31.1 | 4.16 | 13.4 % | 41.3 | – | 0.326 |
| 8k-6 | 66.08 | – | 4.48 | – | 38.2 | – | 1.330 |

- The same arm varies by 3.4 % (code) and 3.0 % (8k) within one session. An arm with two runs cannot resolve effects under about 4 %.

##### C. Live log
Source: `D:/Github/Strata/strata-flash-next-d2x.log`, the boot starting at line 228.
- 294 requests, 295,899 tokens, 5,743.4 s of decode.
- `cpu experts` totals 1,705.9 s, which is 29.7 % of decode. This is an upper bound.
- CPU entries are 21.8 % of all entries.
- Requests of 200 tokens or more: n = 204, tok/s mean 56.75, sd 13.19, range 33.84–83.79.
- Per-request tok/s is therefore not usable as an A/B metric.
- Neither log has wall-clock timestamps.

##### D. Speedup needed on rows time to hold D2x at N workers
Ratios come from #16's rows GB/s (13: 42.8, 9: 34.0, 6: 27.0). The 10 and 8 rows are interpolated at 2.2 GB/s per E-core.

| Workers | Speedup needed |
|---|---|
| 10 | 1.18× |
| 9 | 1.26× |
| 8 | 1.35× |
| 6 | 1.58× |

- Without it, a static extrapolation on D2x code gives: 9 workers about +2.9 ms/round (about −6.6 % tok/s), 6 workers about +6.6 ms (about −14 %).
- #16's own measured losses were smaller (9: −2.7 %, 6: −6.7 %) because the pool was a smaller share of its round (about 16.5 %).

#### 3. Ranked plan

Ranked by: effect on the desktop, then tok/s per worker, then cost, then risk. "Bit-exact" means the output is byte-identical.

##### P0. Attribute the stutter first (tool only; passive; can run now during the developer's normal use)
- **Change:** a new stand-alone `sched_probe.cpp` (for example `tests/xeno/tools/`).
  - One thread pinned to each logical CPU 0–19 at NORMAL priority, 4 unpinned NORMAL threads, and 1 unpinned thread at priority 13.
  - Each waits on a 1 ms high-resolution waitable timer and records its lateness with QPC.
  - Every 100 ms it logs:
    - per-thread lateness p50, p99 and max;
    - which logical CPUs the unpinned threads ran on;
    - PDH `Parking Status` per CPU and `\Process(dynfix)\% Processor Time`;
    - NVML utilisation per GPU (all read-only).
- **Classify each 100 ms interval by phase:**
  - decode: dynfix at about 13+ cores busy;
  - prompt read: 4070 busy and dynfix CPU low;
  - idle.
- **Also run:**
  - PresentMon frame times on a 60 Hz animation;
  - optionally a 60 s `wpr -start CPU -start GPU` capture over one prompt read plus a decode (dwm Ready Time; the 4070's GPU queue).
- **Decision rule:**
  - spikes only in decode, or siblings parked during decode → the pool or parking;
  - spikes during prompt reads → the display card or the stagers; pool work will not fix these.
- **Bit-exact:** n/a. **Build:** a tiny exe. **GPU:** none. **Risk:** none. **Status:** hypothesis test.

##### P1. Core parking off (a system setting, the developer's decision, reversible)
- **Change:** `powercfg /setacvalueindex SCHEME_CURRENT SUB_PROCESSOR CPMINCORES 100`, the same for `CPMINCORES1`, then `powercfg /setactive SCHEME_CURRENT`.
- **Effect (hypothesis):** the siblings stay available to desktop threads that are not pinned. Pinned workers are unaffected.
- **Proof:**
  - the P0 probe before and after, during decode;
  - one generate-mode run at 13 workers with `--pool-rest 0`: `join wait`, parking on vs off. It should fall from about 19 ms/round toward about 1.4 if parking was the starvation mechanism.
- **Bit-exact:** yes. **Build:** no. **Cost:** idle power, UNMEASURED; the plan already has minimum processor state at 100 %.

##### P2. `STRATA_POOL_SPIN_US` in the D2x config env (no build; the live exe reads it at `pool.cpp:203-204`)
- **Change:** set it to at least 1,100 µs, above the tl1 maximum gap between layers of 1,059 µs. Settle the value from a deep-context timeline, because serve runs at 40K+ tokens and the gaps between layers grow with depth.
- **Effect:** about the same as rest in serve. Workers sleep between windows (about 7.9 of 34.5 ms per round in tl1) and after each request.
- **Cost:** one wake per window (57–71 µs p50), the same as generate mode pays today.
- **Warning:** below about 1,060 µs, workers sleep at every layer. That is 46 wakes per round × 57–154 µs = 2.6–7 ms/round, or 7–20 % (bound from tl1, UNMEASURED).
- **Bit-exact:** yes. **Resources:** server restart and GPU for the serve ABBA (M-B).

##### P3. Wire rest into `--serve`, and add serve pool counters (code, a few lines; supersedes P2)
- **Change:** `if (o.pool_rest) pool.rest();` after `ver.run` at `generate.cpp:5447` and `:5184`, and at the end of each request.
- **Counters:** add `join wait`, pool ms/round, rows GB/s and the sleep count to the serve `request metrics` line. Serve prints none of these today.
- **Tests:** a test on the counter formatter. The wiring itself is proven by the serve timeline, where each window's first phase shows a wake of about 60 µs.
- **Bit-exact:** yes. **Build:** yes. **GPU:** for M-B.
- **Expected:** `join wait` near m11's 1.4–2.1 ms/round. The no-rest condition measured 19 ms/round in #16 (older config), so a tok/s gain is possible in serve. UNMEASURED.

##### P4. Park at idle priority: the main desktop lever that keeps 13-worker throughput
- **Change:** in `ExpertPool::worker` (`pool.cpp:275-291`):
  - after a worker has been parked longer than T (20–50 µs, so the gate/up→down gap never pays it), it lowers itself to `THREAD_PRIORITY_IDLE` (or LOWEST) and keeps spinning;
  - when it sees a new epoch it restores the pool priority BEFORE `parked_.fetch_sub` and before claiming.
- **Why it is safe:** a worker never holds a task at low priority. A worker that is preempted while parked just misses that phase; the #29 claim protocol already handles late workers.
- **Effect (hypothesis):**
  - a desktop thread can preempt the worker instantly during the gaps between layers (about 17–18 ms of a 34.5 ms round in tl1) and between windows;
  - with no desktop load the wake stays at 0.3–0.5 µs.
- **Cost:** 2 `SetThreadPriority` calls per worker per layer, UNMEASURED. The raise happens before each layer's first claim.
- **Red tests:**
  - (a) the priority is lowered after T and restored before any claim; a no-op mutant fails;
  - (b) a NORMAL thread pinned to a worker's logical CPU gets CPU within 10 ms while the pool is parked (today about 0);
  - (c) zero claims while lowered;
  - `pool_rest`, `pool_idle_sleep` and `pool_core_policy` stay green.
- **Bit-exact:** yes. **Build:** yes. **Measure:** M-E on CPU only, then an engine ABBA on GPU. **Risk:** medium (scheduler, Thread Director, priority boosts).

##### P5. Tail balance (bit-exact; raises tok/s per worker, more so at fewer workers)
- **Change:** in `drain` modes 5/6 (`pool.cpp:402-406`) and `run_split_multi_native` (`:539-541`), use guided task sizes (large first, down to at least 8 rows) or 6–8 × threads equal tasks.
- **Bound (from tl1):** perfect balance saves at most about 18 % of phase time. That is at most 1.2 ms/round in tl1 and at most 2.0 ms/round on D2x code (rows 11.3). A realistic outcome is about half. UNMEASURED.
- **Bit-exact:** rows are independent, so any partition gives identical output.
- **Tests:** every row covered exactly once for random (n experts, threads, schedule), plus the existing native/multi parity tests.
- **Measure:** M-E, then M-A (`pool multi` gate/up and down ms/round).

##### P6. Kernel levers (bit-exact; run the M-D microbenchmark first)
- **K1, cheaper unpack (new).**
  - Replace `unpack64`'s 17 vector ops with `vbroadcasti128` plus, per half: `vpshufb` replicate (lane0 `[0×4,1×4,2×4,3×4]`, lane1 `[4..7]×4`; hi half 8–15), `vpand [03,0C,30,C0]`, `vpsrlw 4`, `vpblendw 0xAA`, and a `vpshufb` LUT `{0→0,1→1,2→2,3→3,4→1,8→2,12→3}`. That is 10 ops and one load.
  - The lo/hi registers come out identical by construction. The test is exhaustive: each output byte depends on one input byte, so 256 values × 16 positions covers everything.
  - STATIC: 33 → about 26 vector µops per block on P-cores, then the 8-cycle FMA latency floor binds, which is why K2 is needed.
  - On E-cores (Gracemont) 256-bit ops split in two, so it is about neutral there. The P-core or E-core variant can be chosen per worker, and mixing them is safe.
- **K2:** gate+up 2-row interleave with an in-register SwiGLU (the kernel map's B1).
- **K3:**
  - broadcast `d` once per block and use `vbroadcastss` of the scale from memory with `vmulps` (an IEEE-identical product);
  - precompute `hx[2b]+hx[2b+1]` once per activation;
  - keep `corr` as a separate mul and add (verified above).
- **K4:** interleave down rows 2–4 at a time (the kernel map's B5).
- **Combined STATIC estimate:** about 1.2–1.35× on P-cores and about 1.0–1.1× on E-cores. UNMEASURED.
- **Low value:** B2 SiLU (at most 2–4 % of rows), B6 (at most 0.5 %), B8 (0 %), B4 (0–3 %), B9 (the developer's decision).
- **Build:** a microbenchmark only, then M-A on GPU.

##### P7. Worker count sweep and async launch (config on the live exe)
- **Sweep:** `--pool-workers` 13/12/10/8/6. The P-first order drops E-cores: 10 = 5P+5E, 8 = 5P+3E, 6 = 5P+1E.
- **Extra arm:** 12 workers plus `--secondary-async-launch`. The helper then pins to logical CPU 19 (`generate.cpp:3141-3143`).
  - It moves the 2.56 ms/round serial 4070 launch (m11) off the pool's critical path.
  - Recorded: +1.9 % code on an E-core (memory note, daily branch).
  - Under `--serve`: UNVERIFIED.
  - Putting it on a P-core needs a code change at `:3141`.
- **Measure:** M-A, M-B and M-C at each step. This is #51's acceptance run.

##### P8. `--process-priority 1`
- **Effect:** ABOVE_NORMAL class puts workers and host at 12, below dwm's 13.
- **Measured nearby:** only HIGH vs NORMAL (+2.6 %, `0a2d70d`); ABOVE_NORMAL is UNMEASURED.
- **Why it ranks below P4:** it makes WORKING workers preemptible. That is the mechanism behind the measured 57.0 → 66.8 loss in `cpu-pool-sweeps.md` sweep 2.

##### P9. UMWAIT / TPAUSE in the park
- WAITPKG on the i5-13500 and the OS UMWAIT limit are both UNVERIFIED.
- It only lowers spin power; it does not give the CPU back to the scheduler.
- Worth trying only if P0's power and clock logs show PL1 throttling during long decodes.

##### P10. N1 plane layout (not bit-exact)
- STATIC 1.5–1.8×.
- Breaks ADR 0001 and CPU/GPU parity, so it is a developer decision.
- Only if P4–P7 cannot reach 6 workers.

##### Expected outcome (hypothesis)
| Workers | Outlook |
|---|---|
| 10 | Reachable with P5 + K1–K3 (needs 1.18×) |
| 8–9 | Needs the top of those ranges (1.26–1.35×) |
| 6 | Not reachable bit-exactly on these estimates; needs 1.58× |

If P1, P3 and P4 fix the desktop at 13 workers, the worker count becomes purely a tok/s choice.

#### 4. Measurements
- **M-A, engine generate ABBA** (GPU; the live server must be stopped):
  - D2x arguments, the m11 prompts (code, thai, 8k), 256 tokens, order ABBAABBA, identical output hash, exe sha recorded.
  - Judge on the counters `verify window`, `pool multi`, `dispatch detail` and `join wait`.
  - Use at least 4 pairs, because the same arm varies 3.0–3.4 %.
- **M-B, serve ABBA:** needs P3's counters. A fixed request set at 1K, 8K and 32K context, same session, both orders.
- **M-C, desktop:**
  - P0 probe p99, p99.9 and max per phase;
  - how often the siblings are parked during decode;
  - PresentMon frame-time p99.
  - Take the baseline from P0 before setting thresholds.
- **M-D, kernel microbenchmark** (CPU only):
  - today's `rows<1..4>` against K1–K4;
  - on 1 P-core, 1 E-core and 14 threads;
  - with the data L2-resident and streamed from more than 1 GB of DRAM (this decides compute-bound or not);
  - `memcmp` gate first; a mutant must fail;
  - at least 200 ABBA iterations; report p10, p50, p90 ns per block.
- **M-E, pool microbenchmark** (CPU only):
  - the real `ExpertPool` with synthetic jobs at tl1's cadence: 48 × (98 + 49 µs), gaps of about 370 µs between layers and about 6.7 ms between windows;
  - arms: spin, rest, `SPIN_US` ∈ {200, 500, 1500}, P4 with T ∈ {20, 50}, and P5;
  - plus a NORMAL load that needs 2 ms of CPU every 16.7 ms;
  - report phase p50/p99 and the load's missed deadlines.
- **Deep-context timeline:** STRATA_TIMELINE on D2x at 32K–64K, to get the real gap distribution for P2 and P4.

#### 5. What can be done now, and what needs the machine idle
| When | Items |
|---|---|
| Can run now (observe only; no GPU, no ports, never touches dynfix) | P0 probe, parking sampling, PresentMon |
| Can be written now (code plus red tests, not run) | P3, P4, P5, K1–K4 with the exhaustive and bitwise tests, the M-D and M-E harnesses. Building takes several minutes of CPU, so build when the developer is not decoding. |
| Needs the CPU idle, no GPU | M-D, M-E, and the timing runs in the tests |
| Needs the GPU and the live server stopped (the developer's call) | M-A, M-B, the P2 sweep, the P7 sweep, P8 arms, the deep-context trace |
| Needs a system setting change (the developer's call) | P1 |

#### 6. Hypotheses, stated plainly
- Stutter mechanism: pool vs parking vs display card vs stagers.
- Parking starves desktop threads during decode.
- P4's cost and effect.
- The gain sizes of P5 and K1–K4.
- The pool is compute-bound.
- D2x tok/s at fewer workers (only #16's older config was measured).
- `SPIN_US` behaviour at deep context.
- WAITPKG availability.

The figures from tl1, m11, the live log, the disassembly, the topology query and the parking sample were measured by reading existing artifacts.

#### Files
- Code: `C:/Strata-exp/src-dyn/src/kernels/cpu/pool.cpp`, `C:/Strata-exp/src-dyn/include/strata/kernels/cpu/pool.hpp`, `C:/Strata-exp/src-dyn/src/kernels/cpu/q2_avx2.cpp`, `C:/Strata-exp/src-dyn/src/program/generate.cpp`, `C:/Strata-exp/src-dyn/src/core/verify.cpp`
- Data: `C:/Strata-exp/m11-code-1-D2x.stdout` (and `-6`, `thai-1`, `8k-1/6`), `C:/AI/UsersxenodAppDataLocalTempstrata-claude-stagetl1.json`, `tl1b.json`, `D:/Github/Strata/strata-flash-next-d2x.log`
- My analysis scripts and the disassembly: `C:/Users/xenod/AppData/Local/Temp/claude/C--AI/37b6d362-d11e-4264-aed6-72fd7cd554f3/scratchpad/` (`tl2.py`, `tl3.py`, `tl4.py`, `topo.py`, `q2.asm`)
