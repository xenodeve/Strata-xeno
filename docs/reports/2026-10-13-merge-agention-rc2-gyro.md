# Merge report: agentionai/Strata `rc2` (Gyro, APR trellis quants) into Strata-xeno main, 2026-10-13

Tracking issue: xenodeve/Strata-xeno #217. Branch `xeno/gyro`, worktree `C:\Strata-exp\src-gyro`.

## 1. What moved

| | |
|---|---|
| fork base | `main` `27d8d5d6` (upstream v0.1.42 merged, #212; the deployed D2x engine `run-214f`) |
| merged | `agention/rc2` `9ff02c9` = Niko1221/Strata v0.1.40.3 (`d5ea7133`) + 35 commits by agentionai |
| merge base | `d5ea7133` (an ancestor of main, so a 3-way merge of exactly the 35 commits) |
| rc2's delta | 36 files, +3,487 / -53 (engine, kernels, packer, docs, `tools/gyro_setup.sh`) |
| commits | `aab9fbcf` (the merge), `6d31c3fc` (phase 2: rotation on the fork's MMQ path) |
| exes | stock ggml `build-gyro`: sha256 `db3b86100648dfea...` (Swift D2x exe); agention ggml `build-gyro-ag`: `e5b1f4fbe4808af3...` (Gyro exe) |
| toolchain | Windows 11, MSVC 14.44, CUDA 13.3, sm_89 + sm_120, Ninja |

## 2. What rc2 brings

- **Trellis expert types** TQ2_T (144), TQK6 (145), TQK7 (146) of agentionai's llama.cpp fork: 128-value blocks, a
  32768-state trellis over a 2048-point fp16 codebook (`third_party/ggml/ggml-common-tq.h`).
  - CUDA: decode through an int8 copy of the codebook with one global scale plus dp4a (`tq_lut_i8.cuh`;
    `Fmt<144..146>`, `Split<144..146>` for the decode-once kernels). FP16 prompt dequant `dq_tq` through the fp16
    codebook.
  - CPU: `tq_cpu.cpp` (scalar), `tq_avx2.cpp`, `tq_avx512.cpp`, decode-and-dot against Q8_0 activations through a
    32768 x 4 int16 state table. `STRATA_TQ_ISA` caps the ISA; `STRATA_NO_TQ_KERNELS=1` falls back to ggml-cpu's
    vec_dot.
- **APR** (Agention Precision Rotor): the routed experts are stored in a block-128 normalized Sylvester-Walsh-Hadamard
  basis, with optional explicit signs (`prism.hadamard.*` metadata). Every expert path must rotate its input:
  gate/up `H (s * x)`, down `H (s * h)`.
  - `iq_pack.py` writes `hadamard.txt` and marks `native_experts.txt` v5.
  - `expert_layout_load` refuses an inconsistent pack and a folded Q2_0 expert.
- The published files run unchanged: a Q6_K token embedding (Gyro-S) and a Q8_0 PLE table that stays on disk
  (about half of each file).
- `native_expert_parity --load-check`, `--synthetic[-hadamard] tq*`; `tq_cpu_parity`; `native_hadamard_test`;
  `tools/test_iq_pack.py`.

## 3. Conflicts, file by file

| file | kept | ported / why |
|---|---|---|
| `CMakeLists.txt` | both | the fork's `end_engine.cpp`, `q2_avx1.cpp`, `iq_avx1.cpp` plus rc2's `hadamard.cpp`, `tq_*.cpp` |
| `src/core/remote_experts.cpp` | both | the fork's cache read stream plus rc2's per-device sign vectors |
| `src/core/verify.cpp` | both | rc2's rotated `nat_xr_` branch first, then the fork's Q2_0 scaled quantize (`hit_xs_`), then the plain q8_1. A folded layer is never Q2_0 (refused at load), so the branches do not overlap |
| `src/kernels/cuda/iq_kernels.cu` | both | `launch_gu`: rc2's local sub-16 guard (qk 128) with the fork's extra `native_gu_kernel` argument. SwiGLU: the fork's CPU-order path (`cpu_order`, Q2_0 scales/hx) or an APR layer runs the separate passes, with the rotation of h between them (the Q2_0 arguments stay off for TQ). The fork's `iq_dequant_expert_f16` (#29) beside rc2's `hadamard_*`; `dq_tq` makes it take TQ |
| `src/prefill/moe_mmq.cu` | both | the fork's `STRATA_MMQ_IQ1M_TILE` case plus rc2's `STRATA_MMQ_TQ` cases |
| `src/program/generate.cpp` | both | includes |
| `src/prefill/prefill.cpp` | **ours** | the fork rewrote the prompt path (+2,283 / -1,188 since v0.1.40.3: the expert split, the wave, `Xtok`, the CPU share, no `PeerPrefill`). rc2's 97 lines were re-applied by hand where they apply (section 4) |

## 4. Fork-only work the merge needed

- **Prompt path.**
  - `init`: per-layer device sign vectors, plus rc2's refusal of experts other than 2560 x 640.
  - FP16 path: `Xs` and `Hh` rotated in fp16 storage with fp32 arithmetic.
  - Phase 2: `Xtok` is quantized from the rotated `m.Xr`, and H is rotated between `expert_rows_gate_up` and
    `expert_rows_down` on CUDA0 and in `split_run` on the 4070 (`SplitTier::had_sh`). So with an agention ggml tree
    the expert split and the wave run Gyro's layers.
  - A folded layer never takes the fused path.
  - **Not taken:** rc2's rotations in `PeerPrefill` and the layer-split helper (the fork does not have them; merge
    0.1.38 / 0.1.42).
- **The 4070 tier (`SecondaryRunner`)**, which rc2 does not have:
  - It rotates its private `device_x_` (gate/up) and passes its own device's down signs.
  - `NativeExpertLayout::had_gu_block` is new; `expert_source.cpp` sets it.
  - The runner's CUDA-graph key carries the folded flags.
- **Stock ggml.** rc2 asks ggml-cpu's traits about types 144-146, which exist only in agentionai's tree; on the
  stock tree that reads past the table. `native_fmt` now describes a TQ layer itself (Q8_0 activations, 128-value
  blocks, row bytes 34 / 26 / 30 per block) when ggml has no such type, and refuses `STRATA_NO_TQ_KERNELS` there.
- **MSVC:** `native_hadamard_test` used `__builtin_popcount`.

## 5. Semantic hazards checked

- **Every call site that quantizes expert activations or launches `native_expert_grouped`** was listed (grep of
  `native_expert_grouped(`, `quantize_q8_1_rows`, `native_quant_act/_h`, `from_float`).
  - Covered: verify, remote, peer, the 4070 runner, the prompt path.
  - The CPU pool's own quantizers (`act_quant_q8_1`, `act_quant_any`) serve Q2 only.
- **`mmq::supported`** returns false for 144-146 without `STRATA_MMQ_TQ`. `mmq_plan` therefore never asks ggml for a
  TQ block size on the stock tree.
- **`iq_dequant_expert_f16`** indexes 256-value calls across the whole matrix, so the 640-wide down rows (5 TQ
  blocks) decode correctly (dq_tq reads blocks 2i, 2i+1).
- **Buffer accounting:** `m.Xr` is taken in `carve` and counted in `moe_set_bytes` at the same place.

## 6. Verification

| gate | result |
|---|---|
| build, stock ggml (all targets) | ok |
| build, agention ggml (`strata`, the TQ tests) | ok |
| `tq_cpu_parity` | 0 failures (AVX2 = scalar bitwise; vs ggml-cpu rel <= 1.8e-7) |
| `native_hadamard_test` | ok (vs llama.cpp's graph 1.0e-7) |
| `native_expert_parity --synthetic` tqk6/tqk7, tq2_t | 0 failures (GPU rel 2.10e-2 / 2.21e-2, bound 3e-2) |
| `native_expert_parity --synthetic-hadamard` (both) | 0 failures (GPU rotation = CPU bitwise) |
| D2x default path (Swift 1.5 IQ2_XS), STATE_HASH, base `run-214f` vs gyro `run-gyro-a` (`db3b8610`), boots base, gyro, gyro, base (`gyro-record/par217.py hash`) | identical: 7 of 7 hash lines (a ~8K prefix and six branching parts) equal in all four boots, replies equal |
| Gyro-M pack (`iq_pack.py --compat-bf16`, agentionai gguf-py), from this branch and from rc2 | the two packs are byte-identical (`diff -rq`); 48 folded layers, 1079 tensors, 303 from the GGUF |
| Gyro-M `native_expert_parity --load-check` (build-gyro-ag) | 0 failures: experts v5 tq2_t 29.88 GiB, `check_experts_gguf` ok, 303 tensors with a kernel, PLE Q8_0 direct I/O ok (`loadcheck-m.log`) |
| Gyro-M end to end, rc2 vs xeno, same session (ABBA, section 7) | xeno faster at every prompt step and on decode; replies coherent; rc2 vs xeno diverge after 16-114 characters in 6 of 7 text replies (#61 class) |

Logs: `C:\Strata-exp\gyro-record\` (`bench217.py`; `bench217-<session>-results.json`, `bench217-<session>-<arm>-<i>.log`).

## 7. Gyro-M: rc2 against this branch, and the config

**Workload** (`gyro-record/bench217.py`, one server boot per arm, port 8092):
- a ~9.7K-token prefix, then six parts that branch from it (Claude Code's shape);
- a 35.6K-token prompt;
- a 400-token decode of a short prompt.

Greedy. Every xeno arm passes `--pool-priority 2 --process-priority 2`. rc2 has neither flag, so the harness set its process to the HIGH class from outside; its pool threads stayed at normal priority.

**Exes:** rc2 `build-rc2\strata.exe` `e4e7b26d`; xeno `run-gyro-ag\strata.exe` `e5b1f4fb`.

**rc2 vs xeno, session ab1** (order rc2, xeno131, xeno, rc2peer, rc2peer, xeno, xeno131, rc2; ms, two runs each):

| | rc2 (GYRO.md: 131K, auto cache, rc2's profile) | xeno, D2x flags, 262K, tier 4100, Swift's ranked profile | change |
|---|---|---|---|
| prefix 9,731 | 11,658 / 11,621 | 6,401 / 6,400 | -45 % |
| parts 465-2,442 fresh | 2,648-5,061 | 1,092-2,003 | -59..-72 % |
| long 35,594 | 28,653 / 28,252 | 18,854 / 18,777 | -34 % |
| decode 400, tok/s | 37.5 / 37.5 | 48.0 / 48.2 | +28 % |
| 32-token replies, mean tok/s | 41.1 | 38.6 | -6 % (not explained) |
| private commit, GiB | 48.7 (RAM left: 1.6 GB) | 45.4 | |
| boot, s | 40.4 | 19-22 | |

`rc2peer` (`--peer-device 1 --peer-reserve-mib 2560`) did not boot (`peer experts: stream: out of memory`).

**Config sweep** (sw1, sw2, sw3, cr1; each arm two boots, ABBA):

| arm | decode 400 | 32-token mean | parts 465 / 788 | long 35.6K | commit | note |
|---|---|---|---|---|---|---|
| 131K, 4100, ranked (D2x as is) | 48.7 / 46.9 | 39.1 | 1,078 / 1,367 | 19,026 | 43.5 | |
| 131K, 4100, shipped profile | 40.6 / 47.1 | 48.5 | 1,043 / 1,336 | 18,990 | 43.5 | |
| 131K, 6400, ranked | 49.5 / 49.3 | 44.6 | 1,025 / 1,274 | 19,037 | 43.5 | |
| 131K, 6400, shipped profile | 51.6 / 52.3 | 55.3 | 986 / 1,242 | 19,083 | 43.5 | |
| 262K, 6400, shipped profile | 51.4 / 52.8 | 53.9 | 997 / 1,259 | 19,036 | 45.4 | 4070 peak 9,265 MiB |
| 131K, 8000, shipped profile | 57.9 / 57.6 | 55.8 | 928 / 1,194 | 19,047 | 43.5 | 4070 budget 1.1 GiB during a prompt |
| 262K, 7200, shipped profile | 53.3 / 55.3 | 56.1 | 990 / 1,255 | 19,029-19,255 | 45.4 | 1 crash in 3 boots (#219); 4070 peak 10,069 MiB |

**What the counters say:**
- The shipped profile fits Gyro-M better on prose. On the document prompts it cuts the CPU share of the short replies from 14-33 % to 3-13 %; on the code decode it is even.
- Tier 6400 stages 5,140 experts on the 4070 instead of 3,292, and decode CPU entries drop by a third. The commit line does not move, but the box's available RAM rises 2.3 GB.
- 262K reads and decodes like 131K, at +1.9 GiB of commit.

**Proposed config** (not installed; the configs are the developer's): `C:\Strata-exp\gyro-record\strata-gyro-m-d2x.json`.
- It is D2x's flags with the Gyro-M files, vision off, tier 6400 and the shipped profile, at 262K like D2x.
- The file itself, booted twice through `serve/server.py` (session prop): the 9.7K prefix 6.40 / 6.47 s, the 35.6K prompt 18.99 / 19.05 s, decode 50.1 / 50.6 tok/s, the 4070 at most 9,277 MiB used, replies equal in 8 of 8, no crash.
- Against D2x's flags as they are (same sessions): decode +8 %, short replies +40 %, short parts -8..-9 %, long prompt equal.
- Against rc2: every prompt step 34-72 % faster (ab1, same session, with D2x's flags). Decode about +39 % with the proposed config, but that number is across sessions (rc2 in ab1, the proposal in sw2/sw3).

**Gyro-M against Swift 1.5 on this engine** (cr1, tier 7200, the same decode): 55.3 against 91.4 tok/s, with the same windows (172 / 170), the same draft acceptance (231 / 230 of ~300) and about the same CPU entries (30.2K / 27.7K). The CPU pool's time per entry is 123 us against 59 us, about 11.8 of the 16.3 ms/window gap (#220).

**What is left:**
- #218: greedy replies differ between boots of the same exe and config (131K and 262K). STATE_HASH is needed to find where.
- #219: one access violation at tier 7200 at the start of the long prompt; the crash report printed no stack or dump.
- #220: the TQ CPU kernel's cost per expert entry. A timeline of the pool lane comes before any fix.
- The 32-token replies after each part are about 6 % slower than rc2's with D2x's flags. With the proposed config they are faster (55 against 41 tok/s), but that comparison is across sessions.
- Not run: Gyro-S / Gyro-L, vision with Gyro's `mmproj-F16.gguf`, the RAM budget (no arm is under 42 GB: 43.5 at 131K, 45.4 at 262K).
- This branch's stock-ggml exe serves Swift D2x (gate in section 6). The Gyro exe needs agentionai's ggml tree, and the Swift D2x path was not gated on that exe.

## 8. Classic web app, README

- `serve/` is not touched by rc2: nothing to port to `serve/ui/`.
- README: unchanged for now. A Gyro line goes in once a config is chosen.
