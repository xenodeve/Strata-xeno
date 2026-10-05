// #183: the expert split's routed partial crosses the x4 as Q8 (STRATA_SPLIT_BO=q8): per token, groups of
// kSplitQ8Group values, each as uint8 codes plus the group's fp32 minimum and step (asymmetric).  The 4070's
// moe_routed_sum_q8 must keep every value within half a step of moe_routed_sum's, with the group's exact minimum, and
// the 5060's moe_split_finish_q8 must write exactly moe_split_finish on the dequantized partial (min + q * step, one
// fma).  A constant group (step 0) must come back exactly.  Runs on the current device.
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
constexpr int64_t G = N / strata::prefill::kSplitQ8Group;
}  // namespace

int main() {
    std::mt19937 rng(183);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> Dm((size_t) (ROWS * N)), w((size_t) ROWS), shared((size_t) (T * N)), sg((size_t) T);
    for (auto& v : Dm) v = nd(rng);
    for (auto& v : w) v = std::abs(nd(rng)) * 0.2f;
    for (int64_t k = 0; k < K; ++k) w[(size_t) k] = 0.0f;   // token 0: every gate 0, so every group is constant (0)
    for (auto& v : shared) v = nd(rng);
    for (auto& v : sg) v = nd(rng) * 3.0f;
    std::vector<int32_t> slot((size_t) ROWS);
    for (size_t i = 0; i < slot.size(); ++i) slot[i] = (int32_t) i;
    std::shuffle(slot.begin(), slot.end(), rng);
    const int32_t R = (int32_t) (ROWS / 2);
    const size_t wire_bytes = strata::prefill::split_q8_bytes(T);
    auto *dD = (float*) up(Dm), *dw = (float*) up(w), *dsh = (float*) up(shared), *dsg = (float*) up(sg);
    auto* dslot = (int32_t*) up(slot);
    float *ds = nullptr, *dbo = nullptr;
    uint8_t* dwire = nullptr;
    cudaMalloc((void**) &ds, (size_t) (T * N) * 4);
    cudaMalloc((void**) &dbo, (size_t) (T * N) * 4);
    cudaMalloc((void**) &dwire, wire_bytes);
    std::vector<float> sum((size_t) (T * N)), fin_ref((size_t) (T * N)), fin((size_t) (T * N));
    std::vector<uint8_t> wire(wire_bytes);
    int bad = 0;
    strata::prefill::moe_routed_sum(dD, dslot, dw, ds, T, nullptr, 0, R);
    strata::prefill::moe_routed_sum_q8(dD, dslot, dw, dwire, T, nullptr, 0, R);
    down(sum, ds);
    down(wire, dwire);
    // the wire: T*N codes, then T*G minimums, then T*G steps
    const uint8_t* q = wire.data();
    const float* mn = (const float*) (wire.data() + T * N);
    const float* st = mn + T * G;
    std::vector<float> deq((size_t) (T * N));
    size_t off = 0, minbad = 0, const_bad = 0;
    for (int64_t t = 0; t < T; ++t)
        for (int64_t g = 0; g < G; ++g) {
            const int64_t i0 = t * N + g * strata::prefill::kSplitQ8Group;
            const float lo = *std::min_element(&sum[(size_t) i0], &sum[(size_t) i0] + strata::prefill::kSplitQ8Group);
            minbad += mn[t * G + g] != lo;
            for (int64_t j = 0; j < strata::prefill::kSplitQ8Group; ++j) {
                const size_t i = (size_t) (i0 + j);
                deq[i] = std::fmaf((float) q[i], st[t * G + g], mn[t * G + g]);
                const double e = std::abs((double) deq[i] - (double) sum[i]);
                off += !(e <= 0.5 * (double) st[t * G + g] * (1.0 + 1e-5) + 1e-7);
                if (t == 0) const_bad += deq[i] != sum[i];
            }
        }
    std::printf("moe_routed_sum_q8: %zu of %zu values beyond half a step, %zu group minimums inexact, token 0 (constant) "
                "%zu inexact - %s\n", off, sum.size(), minbad, const_bad, off + minbad + const_bad == 0 ? "PASS" : "FAIL");
    bad += off + minbad + const_bad != 0;
    // the 5060's finish: moe_split_finish on the dequantized partial, bit for bit
    cudaMemcpy(dbo, deq.data(), deq.size() * 4, cudaMemcpyHostToDevice);
    strata::prefill::moe_split_finish(dD, dslot, dw, dsh, dsg, dbo, T, R, nullptr);
    down(fin_ref, dbo);
    cudaMemset(dbo, 0xFF, (size_t) (T * N) * 4);   // NaN: every value must be written
    strata::prefill::moe_split_finish_q8(dD, dslot, dw, dsh, dsg, dwire, dbo, T, R, nullptr);
    down(fin, dbo);
    size_t d2 = 0;
    for (size_t i = 0; i < fin.size(); ++i) d2 += std::memcmp(&fin[i], &fin_ref[i], 4) != 0;
    std::printf("moe_split_finish_q8 vs moe_split_finish(dequantized): %zu of %zu differ - %s\n", d2, fin.size(),
                d2 == 0 ? "PASS" : "FAIL");
    bad += d2 != 0;
    std::printf("wire %zu bytes for %lld tokens (fp32 %lld, bf16 %lld)\n", wire_bytes, (long long) T,
                (long long) (T * N * 4), (long long) (T * N * 2));
    for (void* p : {(void*) dD, (void*) dw, (void*) dsh, (void*) dsg, (void*) dslot, (void*) ds, (void*) dbo, (void*) dwire})
        cudaFree(p);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { std::printf("%s\n", cudaGetErrorString(e)); return 1; }
    return bad == 0 ? 0 : 1;
}
