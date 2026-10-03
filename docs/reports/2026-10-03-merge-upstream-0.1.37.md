# Merge report: upstream v0.1.37 into the fork's main (2026-10-03, #103)

Upstream `Niko1221/Strata` `v0.1.37` (`db4f91a`, 2026-10-02) merged into the fork's `main` `7bf0892` (which holds
v0.1.34, #100 / PR #101) on `xeno/exp-upstream-0.1.37`. Working records: `C:\Strata-exp\merge-137-record\` (issue
body, the conflicted files from the trial merge, the scripts that wrote the prefill port, the parity runs).

Speed is out of scope: benchmarks are paused, so no figure below is a speed claim. Every correctness figure names the
run it came from.

## 1. What moved

- Upstream since v0.1.34: 0.1.35, 0.1.36, 0.1.37 - 40 commits, 47 files, +6,060/-374.
- Sixteen files changed on both sides; five conflicted (`git merge-tree`, then the real merge):

| file | conflict hunks (diff3) | what collided |
|---|---|---|
| `src/prefill/prefill.cpp` | 20 | upstream's fused prompt experts (#136) against the fork's MMQ path (compact rows, per-token `Xtok`, the split layout #35, the expert order #41) |
| `serve/server.py` | 7 | the silent-engine watchdog (#481) against the fork's `QUIET_S` heartbeat, the #48 dead-process check and the fork's `_generate` loop |
| `src/program/generate.cpp` | 3 | upstream's continuous PCIe share (#485) against the fork's x4 override (xeno #27); PLE prefetch (#44 D4); the cancelled-prompt log (#471) against the timeline |
| `src/kernels/cuda/sampler.cu` | 1 | the cluster argmax against the fork's ban mask (#49 S4) |
| `serve/frontend.py` | 1 | the system message: the fork's billing-header strip against upstream's `_object_list` (#460) |

## 2. Upstream's changes that matter here, and what the fork does with them

- **Fused int8 prompt experts (#136, 0.1.36).** Upstream runs them by default on the Q2_0 pack and opt-in
  (`STRATA_PF_FUSED=1`) on the native IQ packs. Their kernels cover gate/up IQ2_XXS/IQ2_S and down type 42, so 45 of
  the 48 layers of the fork's Swift 1.5 IQ2_XS pack (`packs/swift15-iq2_xs/native_experts.txt`; the three IQ1_M layers
  keep the FP16 path). **Kept, opt-in on the native packs as upstream has it**, with three fork limits (§3).
- **Cluster decode kernels (0.1.36, sm_90+).** The QSA top-k and the greedy argmax on thread-block clusters. The
  5060 Ti (sm_120) takes them; the 4070 (sm_89) does not. **Kept**, except that a request with a ban mask keeps the
  one-block argmax (§3).
- **`--expert-profile-save` / `--expert-profile-save-every` (#477).** **Kept** as upstream has it; it sits beside the
  fork's ranked profile (#45) and is off unless given.
- **Serve: the silent engine is ended (#481), its last 20 log lines on a start failure (#496), `messages` as a JSON
  string (#460), the cancelled prompt's tokens (#471), the draft counts in `/metrics` (#457).** **Kept**, #481
  combined with the fork's loop (§3).
- **PCIe probe: best of four bursts and a continuous share (#485).** Kept for the layer-split stages; CUDA0 keeps the
  fork's `pcie_frac 0` default (§3).
- **Windows AMD / HIP, setup, `UPDATE.bat`.** Taken as upstream has them; nothing in the fork runs them.

## 3. Every conflict, file by file

### `src/prefill/prefill.cpp`

The diff3 hunks misalign because upstream re-indented the MoE walk into an `else`, so the file was rebuilt from the
fork's side with upstream's v0.1.34..v0.1.37 diff applied by hand (`merge-137-record/patch_prefill.py`, then
`simplify_prefill.py` after the review):

- headers, the stubs for builds without the kernels, `fused_ring()`, `ring_slots()`'s fused ring: upstream's.
- **One predicate for the fused layout.** `fused_ring()` also requires `!g_split_layout` (the split layout's one-card
  buffers hold `moe_cap(T) < T` tokens and its big chunks run on the peer card; the wave runs only with the split
  layout, `generate.cpp:2588-2590`) and the id expert order (`STRATA_EXPERT_ORDER`, #41: the fused walk matches the
  streamed sequence by expert id). `fused_layout(T)` sizes the buffers from it; carve records the result in
  `m.fused_bufs`, and the dispatch runs the fused path only into buffers carved for it, never on a split chunk.
- **`moe_bufs` keeps MMQ's full size.** Upstream shrinks MMQ's share to a last chunk below `stream_all_min()` (P3);
  here a layer the fused kernels do not take (the IQ1_M FP16 layers, an MMQ format they do not cover) runs the whole
  chunk in the same buffers, so each buffer is `max(the fork's MMQ size, the fused size)`. **Dropped: P3's smaller
  buffers**, and with them upstream's 1024-slot ring for the Q2_0 pack: `RING_MAX` and `ring_cap()` stay 512.
- **The dispatch.** Upstream's fused block runs before the fork's host grouping; the fork's grouping, route trace,
  split hand-off and walks moved unchanged into its `else`. The NaN report skips the fused path's int8 buffers.

With `STRATA_PF_FUSED` unset on a native pack `fused_ring()` is false (`moe_fused_iq.cu` `native_supported` needs
`=1`), so every size and launch is the pre-merge fork's: §6's parity runs check exactly that.

### `serve/server.py`

- `StrataEngine.generate()`: upstream's #481 allowance (`allow`, `heard`, the PP-chunk rule) with the fork's
  `QUIET_S` heartbeat through one helper `_wait_s`; a line wait that runs out raises `EngineSilent` only while the
  process lives (one that exited is #48's `EngineDied`, as before); the STOP drain keeps the #48 check and gains the
  deadline. `death_note()` keeps the fork's `_log_tail()` and gains `silent_note`.
- The request loop keeps the fork's single `try` around `_generate` (the thinking-budget passes live there); upstream's
  `leaving` flag and `EngineSilent` on `gen.close()` moved into its `except`/`finally`, `timeline.complete` in a nested
  `finally`; `_say_died` replaces the fork's inline copy. `main()` checks `engine_silence_s` before the start.

### `src/program/generate.cpp`

- PLE prefetch (#44 D4) and `pcie_frac_for_gbps` both kept.
- CUDA0's PCIe share: the fork's block (`pcie_frac 0` unless given: the x4 link collapsed decode to 3.24 tok/s, #27).
  **Dropped for CUDA0: upstream's per-pack probe**; the layer-split stages still probe with #485's formula.
- The request's timeline record, then upstream's #471 "N of M read" log.

### `src/kernels/cuda/sampler.cu`

The cluster argmax knows no ban mask, so `p.ban` keeps `sampler_greedy_kernel<true>`; without one, the cluster kernel
or `sampler_greedy_kernel<false>`. Cost: with the CJK guard on (most requests) the greedy argmax stays one-block - a
follow-up, not a merge fix.

### `serve/frontend.py`

The fork's system-message lines (the billing-header strip) with upstream's `_object_list` loop header (#460).

## 4. Upstream's test, fixed here

`decode_cluster_parity` (upstream's) had two faults, both found on this machine:

- **It raced.** `Dev::put` is a pageable `cudaMemcpy`, which may return before its DMA lands, and the cases run on a
  non-blocking stream; the graph replays then read the previous replay's logits. Pristine v0.1.37
  (`C:\Strata-exp\build-up137`) failed 5 and 4 of 6 replays on the 5060 Ti in two runs; the merge's copy failed 5/6
  three times; with a `cudaDeviceSynchronize` in `put` it passed 6/6 three times. The kernels are identical.
- **A skip passed.** On a card below sm_90 it printed SKIP and returned 0, and CUDA0 here is the 4070 (sm_89): ctest
  showed "Passed" without testing anything. It now tests the first sm_90+ card and returns 77
  (`SKIP_RETURN_CODE 77`) when there is none.

The race is reported upstream as Niko1221/Strata#548: unmodified v0.1.37 failed 5 of 6 replays in three runs, and
the same v0.1.37 with only the `Dev::put` sync passed 6/6 in three runs (`merge-137-record/up-race-*.txt`).

## 5. Review

- `/simplify` (four angles): the fused predicate written twice became one (`fused_ring()` + `m.fused_bufs`), so the
  ring no longer grows for a fused path that cannot run; `ring_cap()` 512; one wait helper in the server; the sync
  moved into `Dev::put`. Skipped: the cluster argmax with a ban (a feature), the duplicated `<true>`/`<false>` launches
  (they predate the merge), plan-time call counts.
- `/code-review`: standards - stale comments and tags fixed; spec - no upstream change in the five files lost; the
  first fused arm did not run the fused kernels (§6).
- `/scrutinize`: traced the fused dispatch against the split, the wave and the host grouping; the open point,
  `pytest serve`, now passes (§6).

## 6. Verification

- **Build:** `build-137.cmd` (CUDA 13.3, sm_89 + sm_120) passes.
- **ctest:** 94/96 (`ctest-137b.log`). The two failures, `ple_parity` and `expert_multi_test`, fail on the pre-merge
  build too (`ctest-134.log`). New and passing: `prefill_fused_moe_test`, `prefill_fused_iq_test`,
  `expert_profile_save_test`, `decode_cluster_parity` (on the 5060 Ti), the `gpu_4070_iq_*` tests (now on IQ2_XS).
- **pytest serve:** 245 passed, 5 skipped, 135 subtests passed (after the `serve/frontend.py` resolution in §3).
- **Greedy parity** (`merge-137-record/parity137.py`; Swift 1.5 IQ2_XS, the served capacity config
  `strata-swift-capacity.json`, generate mode, 256 tokens; A = `run-main-1c2b520` sha `e2f8e042`, the served exe,
  recorded identical to `7bf0892` in #100; B = the merge):
  - first build (sha `b00941d8`), A B B A on three prompts: 12/12 identical, the same hashes as #100
    (`6f2763a8`, `3f8f5ec1`, `46bf06b2`);
  - the same build, split and wave off (`a b f a`): `a` = `b` on all three prompts. `f` (`STRATA_PF_FUSED=1`) ran the
    fused kernels only on n0-long8k (8192-token chunk) and matched there (`15424127`); on the two short prompts the
    chunk is 256, the fused kernels do not run, and the output changed (`ce9960c9`, `11ac63a3`) because the opt-in
    sizes the lendable tail for the fused layout (it starts at slot 1574, not 1724) - the residency effect of #61;
  - the reviewed build (sha `c84174a9`, after `/simplify` and `/code-review`), A B B A on three prompts: 12/12
    identical, the same hashes; split and wave off on n0-long8k, `a b f a`: all four `15424127`, `f` on the fused
    kernels.

## 7. What is left

- The cluster argmax under a ban mask (most served requests): #105.
- The fused path on the native packs is upstream's opt-in and stays so; whether it is faster here is a separate
  measurement (benchmarks are paused), as is its cost of ~150 lendable cache slots.
- `decode_cluster_parity`'s skip-as-pass (exit 0 on SKIP) is not reported upstream; the race is (#548).
- #102 (loans on a split) rebases onto this merge.
