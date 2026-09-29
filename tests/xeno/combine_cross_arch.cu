// #32 S2: expert_split combines each token on the 4070 (sm_89) instead of the 5060 (sm_120).  moe_combine (a per-token
// fmaf chain over k = 0..9 plus the gated shared expert) must write the same bytes on both cards for the same inputs.
// CUDA_VISIBLE_DEVICES=1,0: device 0 is the 5060 Ti, device 1 the 4070 SUPER.
#include "strata/prefill/kernels.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

int main() {
    int n_dev = 0;
    cudaGetDeviceCount(&n_dev);
    if (n_dev < 2) { std::printf("needs two devices (CUDA_VISIBLE_DEVICES=1,0)\n"); return 1; }
    const int64_t T = 512, K = 10, N = 2560, rows = T * K;
    std::mt19937 rng(2);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> Dm((size_t) (rows * N)), w((size_t) rows), shared((size_t) (T * N)), sg((size_t) T);
    for (auto& v : Dm) v = nd(rng);
    for (auto& v : w) v = std::abs(nd(rng)) * 0.2f;
    for (auto& v : shared) v = nd(rng);
    for (auto& v : sg) v = nd(rng) * 3.0f;
    std::vector<int32_t> slot((size_t) rows);
    for (size_t i = 0; i < slot.size(); ++i) slot[i] = (int32_t) i;
    std::shuffle(slot.begin(), slot.end(), rng);
    std::vector<std::vector<float>> out(2, std::vector<float>((size_t) (T * N)));
    for (int dev = 0; dev < 2; ++dev) {
        cudaSetDevice(dev);
        float *dD, *dw, *dsh, *dsg, *dbo;
        int32_t* dslot;
        cudaMalloc(&dD, Dm.size() * 4);
        cudaMalloc(&dw, w.size() * 4);
        cudaMalloc(&dsh, shared.size() * 4);
        cudaMalloc(&dsg, sg.size() * 4);
        cudaMalloc(&dbo, (size_t) (T * N) * 4);
        cudaMalloc(&dslot, slot.size() * 4);
        cudaMemcpy(dD, Dm.data(), Dm.size() * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(dw, w.data(), w.size() * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(dsh, shared.data(), shared.size() * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(dsg, sg.data(), sg.size() * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(dslot, slot.data(), slot.size() * 4, cudaMemcpyHostToDevice);
        strata::prefill::moe_combine(dD, dslot, dw, dsh, dsg, dbo, T, nullptr);
        cudaMemcpy(out[(size_t) dev].data(), dbo, (size_t) (T * N) * 4, cudaMemcpyDeviceToHost);
        const cudaError_t e = cudaGetLastError();
        if (e != cudaSuccess) { std::printf("device %d: %s\n", dev, cudaGetErrorString(e)); return 1; }
        cudaFree(dD); cudaFree(dw); cudaFree(dsh); cudaFree(dsg); cudaFree(dbo); cudaFree(dslot);
    }
    size_t diff = 0;
    for (size_t i = 0; i < out[0].size(); ++i) diff += std::memcmp(&out[0][i], &out[1][i], 4) != 0;
    std::printf("moe_combine sm_120 vs sm_89: %zu of %zu values differ\n%s\n", diff, out[0].size(),
                diff == 0 ? "PASS" : "FAIL");
    return diff == 0 ? 0 : 1;
}
