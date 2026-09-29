// #42 (upstream F-1 882bb6d, F-2 b046845): the fused hyper-connection kernels must write the same bytes as the
// unfused ones they replace, or the prompt path's output changes.
//   F-1: gr_norm_rs + gr_mix_r   against gr_norm + gr_mix          (mixed FP32/BF16/FP16, xn16)
//   F-2: gr_write_norm_rs        against gr_write, then gr_norm_rs  (R, rs, xn16)
// Random rows at the model's shapes (n_embd 2560, hc 4); run on the current device (CUDA_VISIBLE_DEVICES picks it).
#include "strata/prefill/kernels.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace {
constexpr int64_t N = 2560, HC = 4, D = N * HC;
template <typename V>
void* up(const V& v) {
    void* d = nullptr;
    cudaMalloc(&d, v.size() * sizeof(v[0]));
    cudaMemcpy(d, v.data(), v.size() * sizeof(v[0]), cudaMemcpyHostToDevice);
    return d;
}
template <typename V>
V down(const void* d, size_t n) {
    V v(n);
    cudaMemcpy(v.data(), d, n * sizeof(v[0]), cudaMemcpyDeviceToHost);
    return v;
}
template <typename V>
size_t differ(const V& a, const V& b) {
    size_t n = 0;
    for (size_t i = 0; i < a.size(); ++i) n += std::memcmp(&a[i], &b[i], sizeof(a[i])) != 0;
    return n;
}
}  // namespace

int main() {
    using namespace strata::prefill;
    const int64_t T = 1024;
    const float eps = 1e-6f;
    std::mt19937 rng(42);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> R((size_t) (T * D)), w((size_t) D), w2((size_t) D), gated((size_t) (T * D)), bo((size_t) (T * N)),
        inj((size_t) (T * HC));
    for (auto* v : {&R, &gated, &bo, &inj}) for (auto& x : *v) x = nd(rng);
    for (auto* v : {&w, &w2}) for (auto& x : *v) x = 1.0f + 0.1f * nd(rng);
    auto *dR = (float*) up(R), *dw = (float*) up(w), *dw2 = (float*) up(w2), *dg = (float*) up(gated),
         *dbo = (float*) up(bo), *dinj = (float*) up(inj);
    float *xn = nullptr, *rs = nullptr, *m1 = nullptr, *m2 = nullptr;
    uint16_t *xa = nullptr, *xb = nullptr, *b1 = nullptr, *b2 = nullptr, *h1 = nullptr, *h2 = nullptr;
    cudaMalloc((void**) &xn, (size_t) (T * D) * 4);
    cudaMalloc((void**) &rs, (size_t) (T * HC) * 4);
    for (float** p : {&m1, &m2}) cudaMalloc((void**) p, (size_t) (T * N) * 4);
    for (uint16_t** p : {&xa, &xb}) cudaMalloc((void**) p, (size_t) (T * D) * 2);
    for (uint16_t** p : {&b1, &b2, &h1, &h2}) cudaMalloc((void**) p, (size_t) (T * N) * 2);
    // F-1
    gr_norm(dR, dw, eps, xn, xa, T, nullptr);
    gr_mix(xn, dg, m1, b1, T, nullptr, h1);
    gr_norm_rs(dR, dw, eps, rs, xb, T, nullptr);
    gr_mix_r(dR, rs, dw, dg, m2, b2, T, nullptr, h2);
    cudaDeviceSynchronize();
    const size_t f1 = differ(down<std::vector<float>>(m1, T * N), down<std::vector<float>>(m2, T * N)) +
                      differ(down<std::vector<uint16_t>>(b1, T * N), down<std::vector<uint16_t>>(b2, T * N)) +
                      differ(down<std::vector<uint16_t>>(h1, T * N), down<std::vector<uint16_t>>(h2, T * N)) +
                      differ(down<std::vector<uint16_t>>(xa, T * D), down<std::vector<uint16_t>>(xb, T * D));
    // F-2: two copies of R, one written then normed, the other through the fused kernel
    auto* dR2 = (float*) up(R);
    float* rs2 = nullptr;
    cudaMalloc((void**) &rs2, (size_t) (T * HC) * 4);
    gr_write(dR, dbo, dinj, HC, T, nullptr);
    gr_norm_rs(dR, dw2, eps, rs, xa, T, nullptr);
    gr_write_norm_rs(dR2, dbo, dinj, HC, dw2, eps, rs2, xb, T, nullptr);
    cudaDeviceSynchronize();
    const size_t f2 = differ(down<std::vector<float>>(dR, T * D), down<std::vector<float>>(dR2, T * D)) +
                      differ(down<std::vector<float>>(rs, T * HC), down<std::vector<float>>(rs2, T * HC)) +
                      differ(down<std::vector<uint16_t>>(xa, T * D), down<std::vector<uint16_t>>(xb, T * D));
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { std::printf("%s\n", cudaGetErrorString(e)); return 1; }
    int dev = 0;
    cudaGetDevice(&dev);
    cudaDeviceProp pr{};
    cudaGetDeviceProperties(&pr, dev);
    std::printf("%s: F-1 (gr_norm_rs + gr_mix_r) %zu values differ, F-2 (gr_write_norm_rs) %zu values differ\n%s\n",
                pr.name, f1, f2, f1 == 0 && f2 == 0 ? "PASS" : "FAIL");
    return f1 == 0 && f2 == 0 ? 0 : 1;
}
