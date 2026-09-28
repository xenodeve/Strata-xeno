// fused_gr_read_multi promises "every token's outputs are bitwise fused_gr_read(a[t])" (fused_gr.hpp), and the
// verify window relies on it for ADR 0001 parity. This test pins that promise before the down kernel's block
// geometry changes (H2: 41 blocks on 36 SMs ran in two waves).
#include "strata/kernels/fused_gr.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace {
constexpr int N = 2560, HC = 4, D = N * HC, LR = 320;

uint16_t bf16(float f) { uint32_t u; std::memcpy(&u, &f, 4); return (uint16_t) (u >> 16); }

template <class T> T* upload(const std::vector<T>& v) {
    T* d = nullptr;
    cudaMalloc(&d, v.size() * sizeof(T));
    cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice);
    return d;
}
template <class T> std::vector<T> download(const T* d, size_t n) {
    std::vector<T> v(n);
    cudaMemcpy(v.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost);
    return v;
}

struct Tok {   // one token's buffers, for the single-token run and the multi run separately
    float *R, *lo, *rs, *inj, *mixed;
};

int run_case(int T, bool apply, bool inject, std::mt19937& rng) {
    std::normal_distribution<float> nd(0.0f, 1.0f);
    auto randv = [&](size_t n, float scale) { std::vector<float> v(n); for (auto& x : v) x = nd(rng) * scale; return v; };
    auto randw = [&](size_t n) { std::vector<uint16_t> v(n); for (auto& x : v) x = bf16(nd(rng) * 0.02f); return v; };
    const uint16_t* w_down = upload(randw((size_t) LR * D));
    const uint16_t* w_up = upload(randw((size_t) D * LR));
    const uint16_t* w_inject = inject ? upload(randw((size_t) HC * D)) : nullptr;
    const float* w_norm = upload(randv(D, 1.0f));
    std::vector<std::vector<float>> R0(T), bo(T), ip(T);
    for (int t = 0; t < T; ++t) { R0[t] = randv(D, 1.0f); bo[t] = randv(N, 0.5f); ip[t] = randv(HC, 1.0f); }

    auto make = [&](int t) {
        Tok k{};
        k.R = upload(R0[t]);
        cudaMalloc(&k.lo, LR * 4); cudaMalloc(&k.rs, HC * 4); cudaMalloc(&k.inj, HC * 4); cudaMalloc(&k.mixed, N * 4);
        cudaMemset(k.inj, 0, HC * 4);
        return k;
    };
    std::vector<const float*> bo_d(T), ip_d(T);
    for (int t = 0; t < T; ++t) { bo_d[t] = upload(bo[t]); ip_d[t] = upload(ip[t]); }
    auto args = [&](const Tok& k, int t) {
        strata::kernels::FusedGrArgs a;
        a.R = k.R; a.R_out = k.R; a.apply = apply; a.bo_prev = bo_d[t]; a.inj_prev = ip_d[t];
        a.w_norm = w_norm; a.w_down = w_down; a.w_up = w_up; a.w_inject = w_inject;
        a.lo = k.lo; a.rs = k.rs; a.inject_out = k.inj; a.mixed = k.mixed;
        return a;
    };
    std::vector<Tok> single(T), multi(T);
    std::vector<strata::kernels::FusedGrArgs> ma(T);
    for (int t = 0; t < T; ++t) {
        single[t] = make(t);
        strata::kernels::fused_gr_read(args(single[t], t), nullptr);
        multi[t] = make(t);
        ma[t] = args(multi[t], t);
    }
    float* xn = nullptr;
    cudaMalloc(&xn, (size_t) T * D * 4);
    strata::kernels::fused_gr_read_multi(ma.data(), T, xn, nullptr);
    if (cudaDeviceSynchronize() != cudaSuccess) { std::fprintf(stderr, "cuda error\n"); return 1; }

    int bad = 0;
    auto cmp = [&](const char* what, int t, const float* a, const float* b, size_t n) {
        const auto x = download(a, n), y = download(b, n);
        if (std::memcmp(x.data(), y.data(), n * 4) != 0) {
            for (size_t i = 0; i < n; ++i)
                if (std::memcmp(&x[i], &y[i], 4) != 0) {
                    std::fprintf(stderr, "T=%d apply=%d inject=%d token %d %s[%zu]: single %a multi %a\n", T, apply,
                                 inject, t, what, i, x[i], y[i]);
                    break;
                }
            ++bad;
        }
    };
    for (int t = 0; t < T; ++t) {
        cmp("lo", t, single[t].lo, multi[t].lo, LR);
        cmp("rs", t, single[t].rs, multi[t].rs, HC);
        cmp("mixed", t, single[t].mixed, multi[t].mixed, N);
        cmp("R_out", t, single[t].R, multi[t].R, D);
        if (inject) cmp("inject", t, single[t].inj, multi[t].inj, HC);
    }
    return bad;
}
}  // namespace

int main() {
    std::mt19937 rng(11);
    int bad = 0;
    for (int T : {1, 2, 3, 4, 8}) bad += run_case(T, true, true, rng);
    bad += run_case(4, false, false, rng);
    if (bad != 0) { std::fprintf(stderr, "gr_multi_parity: %d mismatches\n", bad); return 1; }
    std::printf("gr_multi_parity: multi == single, bit for bit\n");
    return 0;
}
