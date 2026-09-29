// #29 prefill launch count: one native expert's gate/up and down dequantized by a single launch must be
// byte-identical to the two launches it replaces (iq_dequant_gu_f16 + iq_dequant_f16). The prompt path dequantizes
// every non-resident and resident routed expert of every layer (~20k per 2K prompt), two launches each.
#include "strata/kernels/iq_kernels.hpp"

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

using namespace strata::kernels;

namespace {
bool ok(cudaError_t e, const char* what) {
    if (e == cudaSuccess) return true;
    std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e));
    return false;
}

// random blocks; every 2-byte group that could be an fp16 scale is kept finite and small
std::vector<uint8_t> random_blob(size_t n, uint32_t seed) {
    std::vector<uint8_t> b(n);
    for (size_t i = 0; i < n; ++i) {
        seed = seed * 1664525u + 1013904223u;
        b[i] = (uint8_t) (seed >> 24);
    }
    for (size_t i = 1; i < n; i += 2) b[i] &= 0x3b;   // high byte of any fp16 lane: exponent < 15, finite
    return b;
}
}  // namespace

int main() {
    const int64_t n_embd = 2560, n_ff = 640;
    const int types[][2] = {{42, 42}, {23, 42}, {16, 18}};   // {gate/up, down}: the Q2_0 pack and two mixed IQ packs
    int failures = 0;
    cudaStream_t st;
    if (!ok(cudaStreamCreate(&st), "stream")) return 1;
    for (const auto& ty : types) {
        const NativeExpertLayout L = native_expert_layout(ty[0], ty[1], n_embd, n_ff);
        const std::vector<uint8_t> blob = random_blob(L.bytes, 99u + (uint32_t) ty[0]);
        uint8_t* d_blob = nullptr;
        uint16_t *gu_a = nullptr, *d_a = nullptr, *gu_b = nullptr, *d_b = nullptr;
        const size_t gu_n = (size_t) (2 * n_ff * n_embd), d_n = (size_t) (n_embd * n_ff);
        if (!ok(cudaMalloc((void**) &d_blob, L.bytes), "blob") || !ok(cudaMalloc((void**) &gu_a, gu_n * 2), "gu_a") ||
            !ok(cudaMalloc((void**) &d_a, d_n * 2), "d_a") || !ok(cudaMalloc((void**) &gu_b, gu_n * 2), "gu_b") ||
            !ok(cudaMalloc((void**) &d_b, d_n * 2), "d_b")) return 1;
        cudaMemcpy(d_blob, blob.data(), L.bytes, cudaMemcpyHostToDevice);
        cudaMemset(gu_b, 0xff, gu_n * 2);
        cudaMemset(d_b, 0xff, d_n * 2);
        iq_dequant_gu_f16(L.gu_type, d_blob, d_blob + L.up_off, n_ff, n_embd, gu_a, st);
        iq_dequant_f16(L.d_type, d_blob + L.down_off, n_embd * n_ff, d_a, st);
        iq_dequant_expert_f16(L, d_blob, gu_b, d_b, st);
        if (!ok(cudaStreamSynchronize(st), "sync")) return 1;
        std::vector<uint16_t> a(gu_n + d_n), b(gu_n + d_n);
        cudaMemcpy(a.data(), gu_a, gu_n * 2, cudaMemcpyDeviceToHost);
        cudaMemcpy(a.data() + gu_n, d_a, d_n * 2, cudaMemcpyDeviceToHost);
        cudaMemcpy(b.data(), gu_b, gu_n * 2, cudaMemcpyDeviceToHost);
        cudaMemcpy(b.data() + gu_n, d_b, d_n * 2, cudaMemcpyDeviceToHost);
        size_t diff = 0, nonzero = 0;
        for (size_t i = 0; i < a.size(); ++i) {
            diff += a[i] != b[i];
            nonzero += a[i] != 0;
        }
        std::printf("types gu %d / down %d: %zu of %zu fp16 values differ (%zu nonzero in the reference)\n", ty[0], ty[1],
                    diff, a.size(), nonzero);
        failures += diff != 0 || nonzero == 0;
        cudaFree(d_blob); cudaFree(gu_a); cudaFree(d_a); cudaFree(gu_b); cudaFree(d_b);
    }
    std::printf(failures ? "FAIL\n" : "PASS\n");
    return failures ? 1 : 0;
}
