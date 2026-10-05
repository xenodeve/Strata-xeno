// #181: the expert split's routed sum crosses the x4 as bf16 (STRATA_SPLIT_BO_BF16=1).  The 4070's
// moe_routed_sum_bf16 must write exactly bf16-RNE(moe_routed_sum), and the 5060's moe_split_finish_bf16 must write exactly
// moe_split_finish on the widened values - so the only change to the layer's output is that one rounding, which must
// stay within half a bf16 ulp of the routed partial (relative 2^-8) of the fp32 path.  A NaN partial
// must cross as a NaN.  Runs on the current device.
#include "strata/kernels/bf16_bits.hpp"
#include "strata/prefill/kernels.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace {
template <typename V>
void* up(const V& v) {
    void* d = nullptr;
    cudaMalloc(&d, v.size() * sizeof(v[0]));
    cudaMemcpy(d, v.data(), v.size() * sizeof(v[0]), cudaMemcpyHostToDevice);
    return d;
}
template <typename V>
void down(V& v, const void* d) { cudaMemcpy(v.data(), d, v.size() * sizeof(v[0]), cudaMemcpyDeviceToHost); }

constexpr int64_t T = 512, K = 10, N = 2560, ROWS = T * K;

}  // namespace

int main() {
    std::mt19937 rng(181);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> Dm((size_t) (ROWS * N)), w((size_t) ROWS), shared((size_t) (T * N)), sg((size_t) T);
    for (auto& v : Dm) v = nd(rng);
    for (auto& v : w) v = std::abs(nd(rng)) * 0.2f;
    for (auto& v : shared) v = nd(rng);
    for (auto& v : sg) v = nd(rng) * 3.0f;
    std::vector<int32_t> slot((size_t) ROWS);
    for (size_t i = 0; i < slot.size(); ++i) slot[i] = (int32_t) i;
    std::shuffle(slot.begin(), slot.end(), rng);
    const int32_t R = (int32_t) (ROWS / 2);   // half the pairs on each card
    auto *dD = (float*) up(Dm), *dw = (float*) up(w), *dsh = (float*) up(shared), *dsg = (float*) up(sg);
    auto* dslot = (int32_t*) up(slot);
    float *ds = nullptr, *dbo = nullptr;
    uint16_t* ds16 = nullptr;
    cudaMalloc((void**) &ds, (size_t) (T * N) * 4);
    cudaMalloc((void**) &dbo, (size_t) (T * N) * 4);
    cudaMalloc((void**) &ds16, (size_t) (T * N) * 2);
    std::vector<float> sum((size_t) (T * N)), fin32((size_t) (T * N)), fin16((size_t) (T * N));
    std::vector<uint16_t> sum16((size_t) (T * N));
    int bad = 0;
    // 1. the 4070's half: bf16-RNE of the fp32 routed sum, bit for bit
    strata::prefill::moe_routed_sum(dD, dslot, dw, ds, T, nullptr, 0, R);
    strata::prefill::moe_routed_sum_bf16(dD, dslot, dw, ds16, T, nullptr, 0, R);
    down(sum, ds);
    down(sum16, ds16);
    size_t d1 = 0;
    for (size_t i = 0; i < sum.size(); ++i) d1 += sum16[i] != strata::kernels::bf16_from_f32(sum[i]);
    std::printf("moe_routed_sum_bf16 vs bf16(moe_routed_sum): %zu of %zu differ - %s\n", d1, sum.size(),
                d1 == 0 ? "PASS" : "FAIL");
    bad += d1 != 0;
    // 2. the 5060's half: moe_split_finish on the widened values, bit for bit
    std::vector<float> wide((size_t) (T * N));
    for (size_t i = 0; i < wide.size(); ++i) wide[i] = strata::kernels::f32_from_bf16(sum16[i]);
    cudaMemcpy(dbo, wide.data(), wide.size() * 4, cudaMemcpyHostToDevice);
    strata::prefill::moe_split_finish(dD, dslot, dw, dsh, dsg, dbo, T, R, nullptr);
    down(fin32, dbo);
    cudaMemset(dbo, 0xFF, (size_t) (T * N) * 4);   // NaN: every value must be written
    strata::prefill::moe_split_finish_bf16(dD, dslot, dw, dsh, dsg, ds16, dbo, T, R, nullptr);
    down(fin16, dbo);
    size_t d2 = 0;
    for (size_t i = 0; i < fin16.size(); ++i) d2 += std::memcmp(&fin16[i], &fin32[i], 4) != 0;
    std::printf("moe_split_finish_bf16 vs moe_split_finish(widened): %zu of %zu differ - %s\n", d2, fin16.size(),
                d2 == 0 ? "PASS" : "FAIL");
    bad += d2 != 0;
    // 3. against the fp32 path: the only change is the routed partial's rounding (half a bf16 ulp: relative 2^-8)
    cudaMemcpy(dbo, sum.data(), sum.size() * 4, cudaMemcpyHostToDevice);
    strata::prefill::moe_split_finish(dD, dslot, dw, dsh, dsg, dbo, T, R, nullptr);
    std::vector<float> exact((size_t) (T * N));
    down(exact, dbo);
    size_t off = 0;
    double worst = 0.0;
    for (size_t i = 0; i < exact.size(); ++i) {
        const double bound = std::ldexp(std::abs((double) sum[i]), -8) + 1e-6;   // half a bf16 ulp of the partial
        const double e = std::abs((double) fin16[i] - (double) exact[i]);
        worst = std::max(worst, e / (std::abs((double) sum[i]) + 1e-30));
        off += !(e <= bound);
    }
    std::printf("bf16 path vs fp32 path: %zu beyond half a bf16 ulp of the routed partial (worst %.2e relative) - %s\n",
                off, worst, off == 0 ? "PASS" : "FAIL");
    bad += off != 0;
    // 4. a NaN in the partial crosses as a NaN (the add-and-shift rounding turned 0x7FFFFFFF into -0, a plausible
    //    value that hid it from STRATA_DBG_NAN): token 1's first pair reads a row of NaN, so all its values are NaN
    {
        const size_t row = (size_t) slot[(size_t) K];   // pair (token 1, k 0) - below R, so on this card
        std::vector<float> Dn = Dm;
        std::fill(Dn.begin() + (ptrdiff_t) (row * N), Dn.begin() + (ptrdiff_t) ((row + 1) * N), std::nanf(""));
        cudaMemcpy(dD, Dn.data(), Dn.size() * 4, cudaMemcpyHostToDevice);
        strata::prefill::moe_routed_sum(dD, dslot, dw, ds, T, nullptr, 0, R);
        strata::prefill::moe_routed_sum_bf16(dD, dslot, dw, ds16, T, nullptr, 0, R);
        down(sum, ds);
        down(sum16, ds16);
        size_t nan32 = 0, nan16 = 0;
        for (int64_t n = 0; n < N; ++n) {
            nan32 += std::isnan(sum[(size_t) (N + n)]);
            nan16 += std::isnan(strata::kernels::f32_from_bf16(sum16[(size_t) (N + n)]));
        }
        std::printf("a NaN partial (token 1): fp32 %zu of %lld NaN, bf16 %zu NaN - %s\n", nan32, (long long) N, nan16,
                    nan32 == (size_t) N && nan16 == (size_t) N ? "PASS" : "FAIL");
        bad += !(nan32 == (size_t) N && nan16 == (size_t) N);
    }
    for (void* p : {(void*) dD, (void*) dw, (void*) dsh, (void*) dsg, (void*) dslot, (void*) ds, (void*) dbo, (void*) ds16})
        cudaFree(p);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { std::printf("%s\n", cudaGetErrorString(e)); return 1; }
    return bad == 0 ? 0 : 1;
}
