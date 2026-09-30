// #32 S1: the 4070 gets each token's activations once (q8_1 in MMQ's layout, quantized per token) and gathers the
// rows its experts need, instead of receiving T*K quantized rows.  mmq::gather_q8_rows over the per-token quantized
// rows must write exactly the bytes mmq::quantize writes when it quantizes the gathered rows itself (a row's q8_1
// blocks depend only on that row), for the weight types whose q8 layouts differ (the ds layout follows the type).
#include "strata/prefill/moe_mmq.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

int main() {
    namespace mmq = strata::prefill::mmq;
    const int64_t T = 300, cols = 2560, n = 900;
    std::mt19937 rng(32);
    std::normal_distribution<float> nd(0.0f, 1.5f);
    std::vector<float> x((size_t) (T * cols));
    for (auto& v : x) v = nd(rng);
    std::vector<int32_t> ids((size_t) n);
    for (auto& i : ids) i = (int32_t) (rng() % (uint32_t) T);
    float* d_x = nullptr;
    int32_t* d_ids = nullptr;
    cudaMalloc(&d_x, x.size() * 4);
    cudaMalloc(&d_ids, ids.size() * 4);
    cudaMemcpy(d_x, x.data(), x.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_ids, ids.data(), ids.size() * 4, cudaMemcpyHostToDevice);
    const size_t bytes_n = mmq::q8_bytes(n, cols), bytes_t = mmq::q8_bytes(T, cols);
    uint8_t *a = nullptr, *b = nullptr, *tok = nullptr;
    cudaMalloc(&a, bytes_n);
    cudaMalloc(&b, bytes_n);
    cudaMalloc(&tok, bytes_t);
    int fails = 0;
    for (int t : {42, 16, 21, 23}) {   // Q2_0 and IQ types (their q8 ds layouts differ)
        cudaMemset(a, 0x11, bytes_n);
        cudaMemset(b, 0x11, bytes_n);
        mmq::quantize(d_x, d_ids, a, t, cols, cols, n, nullptr);          // today: quantize the gathered rows
        mmq::quantize(d_x, nullptr, tok, t, cols, cols, T, nullptr);      // S1: each token once...
        mmq::gather_q8_rows(tok, T, d_ids, n, cols, b, nullptr);          // ...then gather the rows
        std::vector<uint8_t> ha(mmq::q8_row_bytes(cols) * (size_t) n), hb(ha.size());
        cudaMemcpy(ha.data(), a, ha.size(), cudaMemcpyDeviceToHost);
        cudaMemcpy(hb.data(), b, hb.size(), cudaMemcpyDeviceToHost);
        const cudaError_t e = cudaGetLastError();
        size_t diff = 0;
        for (size_t i = 0; i < ha.size(); ++i) diff += ha[i] != hb[i];
        std::printf("type %2d: %zu of %zu bytes differ%s%s\n", t, diff, ha.size(), e != cudaSuccess ? ": " : "",
                    e != cudaSuccess ? cudaGetErrorString(e) : "");
        fails += diff != 0 || e != cudaSuccess;
    }
    std::printf(fails == 0 ? "PASS\n" : "FAIL\n");
    return fails == 0 ? 0 : 1;
}
