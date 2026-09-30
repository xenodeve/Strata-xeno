// #31 probe (not a test): the prompt path's per-expert MMQ gather (mmq::gather_native, one 1.38 MB expert from a ring
// slot into its group slot) shows 0.104 ms per expert on the prefill timeline ("dequant", 2.1 s of an 8K prompt).  A
// VRAM-to-VRAM copy of that size should take a few microseconds.  This times the same call on VRAM buffers, alone and
// back to back, with the Q2_0 layout the pack uses (gate/up halves of 460,800 bytes, down 460,800 bytes).
// CUDA_VISIBLE_DEVICES=1 runs it on the 5060 Ti.
#include "strata/prefill/moe_mmq.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>

int main() {
    const size_t half = 460800, down = 460800, blob = 2 * half + down;
    const int slots = 64, group = 8;
    uint8_t *ring = nullptr, *gu = nullptr, *dn = nullptr;
    cudaMalloc(&ring, blob * slots);
    cudaMalloc(&gu, 2 * half * group + 4096);
    cudaMalloc(&dn, down * group + 4096);
    cudaMemset(ring, 7, blob * slots);
    cudaStream_t s;
    cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
    cudaEvent_t a, b;
    cudaEventCreate(&a);
    cudaEventCreate(&b);
    for (int pass = 0; pass < 3; ++pass) {
        for (int n : {1, 8, 425}) {
            cudaEventRecord(a, s);
            for (int j = 0; j < n; ++j) {
                const uint8_t* src = ring + (size_t) (j % slots) * blob;
                const int q = j % group;
                strata::prefill::mmq::gather_native(src, src + half, half, src + 2 * half, down, gu + (size_t) q * 2 * half,
                                                    dn + (size_t) q * down, s);
            }
            cudaEventRecord(b, s);
            cudaEventSynchronize(b);
            float ms = 0;
            cudaEventElapsedTime(&ms, a, b);
            std::printf("pass %d: %3d gathers in %.3f ms = %.4f ms each (%.1f GB/s read)\n", pass, n, ms, ms / n,
                        blob * n / (ms * 1e-3) / 1e9);
        }
    }
    // the Strata-pack Q2_0 transcode (planes -> GGUF blocks), the gather a non-native pack's layer uses
    for (int pass = 0; pass < 2; ++pass) {
        const int n = 425;
        cudaEventRecord(a, s);
        for (int j = 0; j < n; ++j)
            strata::prefill::mmq::gather_strata_q2(ring + (size_t) (j % slots) * blob, gu + (size_t) (j % group) * 2 * half,
                                                   dn + (size_t) (j % group) * down, s);
        cudaEventRecord(b, s);
        cudaEventSynchronize(b);
        float ms = 0;
        cudaEventElapsedTime(&ms, a, b);
        std::printf("strata_q2 pass %d: %d transcodes in %.3f ms = %.4f ms each\n", pass, n, ms, ms / n);
    }
    const cudaError_t e = cudaGetLastError();
    std::printf("%s\n", e == cudaSuccess ? "ok" : cudaGetErrorString(e));
    return 0;
}
