# Phase 3 static expert staging checkpoint

Scope: engine #3 / tracker #12. Branch `xeno/afk-phase3`. This checkpoint **only stages read-only experts on the RTX 4070 SUPER**. It does not route expert compute there, reduce RAM, or claim a faster dual-GPU result.

## Implementation

`--secondary-expert-mib N` is an opt-in, staging-only flag; default zero leaves the single-GPU path. The startup path first fills the primary 5060 Ti cache from the routing profile, then selects the next ranked, non-primary `(layer, expert)` pairs. The selection test rejects duplicate/out-of-range pairs and preserves rank order. A bounded `SecondaryArena` allocates on CUDA device 1, touches and checks the allocation, then fills and reads back every selected native Q2_0 blob. It checks the lower of CUDA and PCI-matched NVML free memory after each fill. The log explicitly says `STAGING ONLY, no secondary compute`. `CUDA_VISIBLE_DEVICES=1,0` is required; the ignored user launcher/config were not changed.

The native Q2_0 expert kernel was also run **standalone** on the display 4070 SUPER (`CUDA_VISIBLE_DEVICES=0`) against real layer-0/expert-7 boundary input and layer-47/expert-361 input. Each comparison reported 0/7,680 differing output floats versus the CPU pool. This proves the existing kernel arithmetic works on `sm_89` for these inputs, not that cross-device transport/dispatch works.

## Integration diagnosis

With the existing 31.64 GiB host expert arena CUDA-registered/locked, three bounded model attempts with 64 MiB secondary staging failed before secondary allocation: `cudaMemGetInfo` on CUDA device 1 returned `out of memory` while NVML still reported about 9.6 GiB free on the 4070 SUPER; `cudaPeekAtLastError` was `no error` before the query. The expert arena startup log reported whole-arena CUDA registration failure, followed by approximately 29 GiB successfully pinned in layer slices and another ~2 GiB OS-locked. Earlier code already clears handled `cudaHostRegister` errors, so a stale error is not established as the cause.

Two differential probes narrowed the issue:

- Setting `STRATA_ARENA_LOCK=0` while leaving CUDA registration enabled still failed at the secondary context query.
- Initializing the second CUDA context before the large host arena succeeded, but MTP dense weights on the primary 5060 Ti later failed to load; this ordering change was removed.

The opt-in secondary mode now leaves the large host expert arena pageable (no `cudaHostRegister` and no OS lock). This allowed the secondary context and a full one-token model run. A 64 MiB cap staged 48 next-ranked experts, verified every blob, and reported 9.54 GiB lower free VRAM on the 4070 SUPER. A 96-token sky run reported 9.47 GiB lower free at staging and matched the accepted single-GPU greedy IDs 96/96. Artifacts: `%TEMP%\strata-phase3-stage-pageable` and `%TEMP%\strata-phase3-stage-pageable-sky96`.

These observations support a host registration/commit-pressure interaction; they do not identify the exact Windows/CUDA limit. During the dual-card run, a spot check found about 1.4 GiB free physical and 1.2 GiB free virtual memory. Phase 4's planned exclusive expert tier may relieve that host-memory pressure, but it is not implemented yet.

## Rough performance, not a final verdict

One 96-token sky run with pageable host arena and staged but unused secondary experts measured 27.67 decode tok/s and 11.34 prefill tok/s. The earlier accepted single-card pinned run measured 31.64 and 15.99. A separate single-card pageable A/B probe using `STRATA_ARENA_PIN=0` matched 96/96 greedy IDs and measured 34.66 decode / 14.14 prefill tok/s. These are unpaired runs under changing desktop/CPU load; the dual-card decode slowdown cannot be assigned solely to pageability or the second CUDA context. Secondary compute has not been added, so none of these is a dual-card performance result. Before making a default choice, run paired measurements after secondary computation and host-RAM reduction.

## Next proof gate

Use the verify-window host callback's pinned activation/routing and output seam. The primary GPU publishes a graph plan before CPU work; the secondary GPU must claim only remaining expert groups, compute each once, and return each partial to its router-index row before that output is consumed. The existing native grouped Q2_0 kernel accepts multiple tokens and FP32 activation scales. Add a focused device-1 multi-token parity test and explicit cross-device transfer/event ordering before using the static tier in normal generation. Keep the 2560 MiB reserve checks through that path.

## Verification at checkpoint

The dual-architecture `strata` target built. `ctest -R xeno_` passed 8/8 and Python serving tests passed 23/23. The manual device-1 arena smoke test passed after rebuilding. The ordinary single-5060 path with no secondary flag still used its original CUDA-registered/pinned arena and produced the accepted first sky token (`29108`), artifact `%TEMP%\strata-phase3-single-regression1`. These checks do not exercise secondary compute.
