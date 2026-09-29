# Static 4070 SUPER expert tier — design

Status: implementation design for Strata-xeno Phase 3; performance remains unmeasured.
Source: engine issue #3, tracker issue #12, three-tier design tracker #2, accepted Q2_0 baseline ADR 0001. The developer authorized an AFK run and set a new display-card safety floor on 2026-09-27.

## Goal and boundary

Keep trunk, recurrent/attention state, MTP and primary expert cache on the 5060 Ti (`cuda:0` when `CUDA_VISIBLE_DEVICES=1,0`). Add a static set of profile-ranked Q2_0 experts on the RTX 4070 SUPER (`cuda:1`). During native verify windows, each routed expert is computed on exactly one of primary GPU, secondary GPU, or CPU. Greedy output must match the accepted single-GPU baseline token for token. This phase does not reduce the host expert arena or implement rotating swaps/prefill; those are Phases 4 and 5.

The display GPU must retain **at least 2.5 GB of free VRAM**. Use 2560 MiB as the code floor, which is conservative relative to decimal 2.5 GB. Query free memory on the secondary CUDA device immediately before allocation, physically touch allocations under WDDM, synchronize, and query again. Reduce the number of ranked expert slots or refuse the tier if the floor is not met. Log measured free MiB after placement. A changing desktop workload can consume VRAM later; sample free memory before secondary work and disable/retire the tier on a breach rather than launching new secondary work. Do not claim an unconditional margin against external desktop allocations between samples.

Preflight on this WDDM desktop found CUDA `cudaMemGetInfo` reporting about 10.810 GiB free while NVML reported about 9.8 GiB free for the same 4070 SUPER. Match the CUDA ordinal to the NVML device by PCI bus ID and use the lower nonzero value for planning and post-touch checks. If either query fails, refuse secondary placement. NVIDIA documents that [CUDA free memory is an estimate that may not all be allocatable](https://docs.nvidia.com/cuda/cuda-driver-api/cuda_driver_api/group__CUDA__MEM.html) and that [WDDM manages device memory](https://docs.nvidia.com/deploy/nvml-api/api/group__nvmlDeviceQueries.html); the exact source of the observed gap is not established.

## Existing seams and selected path

`expert_pool_dispatch_multi` already sees published per-layer activation and router IDs on the host. It plans primary-GPU resident/PCIe groups, and its remaining rows go through the CPU pool. Keep the primary graph and `host_res` unchanged. A new secondary tier will claim a subset of those CPU rows, grouped by distinct expert, launch Q2_0 work on device 1, and return each partial to the same host output row before the existing H2D copy to device 0. This avoids inserting cross-device nodes into the primary CUDA graph. The source blob remains in RAM in Phase 3; secondary residency is a read-only copy.

Secondary activation and result staging must be pinned. The host publishes data only after the primary graph's doorbell has made activation/routing visible. Device-1 H2D, kernel and D2H use its own stream; an event or stream synchronization gates consumption of the result rows. CPU jobs for other experts may run concurrently. The route decision is made once per distinct expert, then applied to every token's occurrence of that expert. Router weights remain solely in the existing primary-GPU combine step. A failure in the secondary path must latch a request error or safely fall back before any partial is consumed; it cannot silently emit zeros or compute an expert twice.

## Other approaches considered

- Moving primary graph nodes onto the second GPU would make cross-device graph capture/event ordering the first problem. The host verify-window seam already publishes the inputs needed, so use that seam first.
- Making the 4070 tier exclusive of RAM would save host memory but prevent safe CPU fallback and is the Phase 4 contract. Keep Phase 3 as a copy.
- Filling all remaining 4070 VRAM maximizes slots but violates the display-card safety constraint. A measured, touched allocation and 2560 MiB floor take precedence over hit rate.

## Proof gates

1. Dual-device preflight identifies CUDA device 0 as the 5060 Ti and device 1 as the 4070 SUPER when the visible-device order is `1,0`; no model load occurs during this probe.
2. Pure capacity tests cover free memory below/equal/above the 2560 MiB floor, slot-size alignment, overflow, and profile truncation. A GPU allocation test touches its bytes and verifies measured post-allocation headroom, then frees only its own allocation.
3. A small real Q2_0 expert test compares secondary-device partials to the accepted CPU/GPU expert output bit for bit, including quantization boundaries and repeated expert IDs across tokens.
4. Fixed-prompt token parity checks the accepted baseline at 96 tokens first, then four 256-token prompts. Capture raw IDs and exact binary/config hashes; do not infer quality from parity alone.
5. Record 4070 free VRAM before load, after touch, after model ready and during runs. If any sample falls below 2560 MiB, reduce or disable the tier and report that configuration as failed. Rough TTFT/decode measurements follow correctness; precise ABBA comes later.

## Open technical risks to falsify early

- The native verify-window host activation or result buffers might not be pinned for the entire secondary transfer lifetime. Confirm their allocation and reuse before asynchronous transfers.
- The grouped Q2_0 kernel currently gets primary-device buffers. Prove it accepts separate device-1 scratch/scale/slot buffers without hidden primary-device state.
- A synchronous wait on device 1 might erase any decode speed gain even when correctness passes. Measure overlap and the cross-card term before optimizing it.
