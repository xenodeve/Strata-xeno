// #32 D1: the 4070 need not receive the shared expert's output.  moe_combine is
//   bo = (fmaf chain over k = 0..9 of w * Dm rows) + shared * sigmoid(sg)
// so the 4070 can compute only the routed sum (moe_routed_sum) and the 5060 finish it with its own shared output
// (moe_shared_finish), saving 84 MB over x4 per layer at 8K.  The split must write the same bytes as moe_combine on
// the 5060: the routed sum on the 4070 (sm_89), the finish on the 5060 (sm_120), against moe_combine on the 5060.
// The finish may be contracted to an FMA by the compiler; this test is what says it matches.
// CUDA_VISIBLE_DEVICES=1,0: device 0 is the 5060 Ti, device 1 the 4070 SUPER.
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
}  // namespace

int main() {
    int n_dev = 0;
    cudaGetDeviceCount(&n_dev);
    if (n_dev < 2) { std::printf("needs two devices (CUDA_VISIBLE_DEVICES=1,0)\n"); return 1; }
    const int64_t T = 512, K = 10, N = 2560, rows = T * K;
    std::mt19937 rng(34);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> Dm((size_t) (rows * N)), w((size_t) rows), shared((size_t) (T * N)), sg((size_t) T);
    for (auto& v : Dm) v = nd(rng);
    for (auto& v : w) v = std::abs(nd(rng)) * 0.2f;
    for (auto& v : shared) v = nd(rng);
    for (auto& v : sg) v = nd(rng) * 3.0f;
    std::vector<int32_t> slot((size_t) rows);
    for (size_t i = 0; i < slot.size(); ++i) slot[i] = (int32_t) i;
    std::shuffle(slot.begin(), slot.end(), rng);
    const size_t out_bytes = (size_t) (T * N) * 4;
    // the reference: moe_combine on the 5060
    std::vector<float> ref((size_t) (T * N)), got((size_t) (T * N)), sum((size_t) (T * N));
    cudaSetDevice(0);
    {
        auto *dD = (float*) up(Dm), *dw = (float*) up(w), *dsh = (float*) up(shared), *dsg = (float*) up(sg);
        auto* dslot = (int32_t*) up(slot);
        float* dbo = nullptr;
        cudaMalloc((void**) &dbo, out_bytes);
        strata::prefill::moe_combine(dD, dslot, dw, dsh, dsg, dbo, T, nullptr);
        cudaMemcpy(ref.data(), dbo, out_bytes, cudaMemcpyDeviceToHost);
        for (void* p : {(void*) dD, (void*) dw, (void*) dsh, (void*) dsg, (void*) dslot, (void*) dbo}) cudaFree(p);
    }
    // the split: routed sum on the 4070 ...
    cudaSetDevice(1);
    {
        auto *dD = (float*) up(Dm), *dw = (float*) up(w);
        auto* dslot = (int32_t*) up(slot);
        float* ds = nullptr;
        cudaMalloc((void**) &ds, out_bytes);
        strata::prefill::moe_routed_sum(dD, dslot, dw, ds, T, nullptr);
        cudaMemcpy(sum.data(), ds, out_bytes, cudaMemcpyDeviceToHost);
        for (void* p : {(void*) dD, (void*) dw, (void*) dslot, (void*) ds}) cudaFree(p);
    }
    // ... finished on the 5060, in place (bo holds the routed sum on entry)
    cudaSetDevice(0);
    {
        auto *dsh = (float*) up(shared), *dsg = (float*) up(sg), *dbo = (float*) up(sum);
        strata::prefill::moe_shared_finish(dsh, dsg, dbo, T, nullptr);
        cudaMemcpy(got.data(), dbo, out_bytes, cudaMemcpyDeviceToHost);
        for (void* p : {(void*) dsh, (void*) dsg, (void*) dbo}) cudaFree(p);
    }
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { std::printf("%s\n", cudaGetErrorString(e)); return 1; }
    size_t diff = 0, bad = 0;
    for (size_t i = 0; i < ref.size(); ++i) {
        diff += std::memcmp(&ref[i], &got[i], 4) != 0;
        bad += !std::isfinite(ref[i]);
    }
    std::printf("routed sum on sm_89 + shared finish on sm_120 vs moe_combine on sm_120: %zu of %zu values differ "
                "(%zu non-finite)\n%s\n", diff, ref.size(), bad, diff == 0 && bad == 0 ? "PASS" : "FAIL");
    return diff == 0 && bad == 0 ? 0 : 1;
}
