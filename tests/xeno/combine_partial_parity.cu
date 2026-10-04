// #133: CUDA0 computes the routed experts it already holds and the 4070 only the rest, so each card sums a disjoint
// subset of a token's ten (token, k) pairs.  A pair a card does not own carries slot -1 there, and its expert row on that
// card is never computed (poisoned with NaN here).  The 4070's moe_routed_sum over its own pairs, then moe_split_finish
// on the 5060 (its own pairs + the shared expert onto the 4070's partial), must equal moe_combine up to the rounding of
// the split FMA chain - and, when the 5060 owns nothing, write moe_shared_finish's bytes (#35 D1's split unchanged).
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

constexpr int64_t T = 512, K = 10, N = 2560, ROWS = T * K;

// the 4070's routed sum over the pairs it owns, then the 5060's finish over its own; `local[i]`: pair i is the 5060's
std::vector<float> split_run(const std::vector<float>& Dm, const std::vector<int32_t>& slot, const std::vector<float>& w,
                             const std::vector<float>& shared, const std::vector<float>& sg,
                             const std::vector<char>& local) {
    const float nan = std::nanf("");
    std::vector<int32_t> slot_peer(slot), slot_local(slot);
    std::vector<float> dm_peer(Dm), dm_local(Dm);
    for (int64_t i = 0; i < ROWS; ++i) {
        const int64_t row = slot[(size_t) i];
        if (local[(size_t) i]) {   // the 4070 never computes this row
            slot_peer[(size_t) i] = -1;
            std::fill(dm_peer.begin() + row * N, dm_peer.begin() + (row + 1) * N, nan);
        } else {                   // nor the 5060 this one
            slot_local[(size_t) i] = -1;
            std::fill(dm_local.begin() + row * N, dm_local.begin() + (row + 1) * N, nan);
        }
    }
    const size_t out_bytes = (size_t) (T * N) * 4;
    std::vector<float> sum((size_t) (T * N)), got((size_t) (T * N));
    cudaSetDevice(1);
    {
        auto *dD = (float*) up(dm_peer), *dw = (float*) up(w);
        auto* dslot = (int32_t*) up(slot_peer);
        float* ds = nullptr;
        cudaMalloc((void**) &ds, out_bytes);
        strata::prefill::moe_routed_sum(dD, dslot, dw, ds, T, nullptr);
        cudaMemcpy(sum.data(), ds, out_bytes, cudaMemcpyDeviceToHost);
        for (void* p : {(void*) dD, (void*) dw, (void*) dslot, (void*) ds}) cudaFree(p);
    }
    cudaSetDevice(0);
    {
        auto *dD = (float*) up(dm_local), *dw = (float*) up(w), *dsh = (float*) up(shared), *dsg = (float*) up(sg);
        auto* dslot = (int32_t*) up(slot_local);
        auto* dbo = (float*) up(sum);
        strata::prefill::moe_split_finish(dD, dslot, dw, dsh, dsg, dbo, T, nullptr);
        cudaMemcpy(got.data(), dbo, out_bytes, cudaMemcpyDeviceToHost);
        for (void* p : {(void*) dD, (void*) dw, (void*) dsh, (void*) dsg, (void*) dslot, (void*) dbo}) cudaFree(p);
    }
    return got;
}
}  // namespace

int main() {
    int n_dev = 0;
    cudaGetDeviceCount(&n_dev);
    if (n_dev < 2) { std::printf("needs two devices (CUDA_VISIBLE_DEVICES=1,0)\n"); return 1; }
    std::mt19937 rng(133);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> Dm((size_t) (ROWS * N)), w((size_t) ROWS), shared((size_t) (T * N)), sg((size_t) T);
    for (auto& v : Dm) v = nd(rng);
    for (auto& v : w) v = std::abs(nd(rng)) * 0.2f;
    for (auto& v : shared) v = nd(rng);
    for (auto& v : sg) v = nd(rng) * 3.0f;
    std::vector<int32_t> slot((size_t) ROWS);
    for (size_t i = 0; i < slot.size(); ++i) slot[i] = (int32_t) i;
    std::shuffle(slot.begin(), slot.end(), rng);
    const size_t out_bytes = (size_t) (T * N) * 4;
    std::vector<float> ref((size_t) (T * N)), shared_only((size_t) (T * N));
    cudaSetDevice(0);
    {   // the reference: moe_combine on the 5060; and #35 D1's finish (the 4070's full routed sum + the shared term)
        auto *dD = (float*) up(Dm), *dw = (float*) up(w), *dsh = (float*) up(shared), *dsg = (float*) up(sg);
        auto* dslot = (int32_t*) up(slot);
        float* dbo = nullptr;
        cudaMalloc((void**) &dbo, out_bytes);
        strata::prefill::moe_combine(dD, dslot, dw, dsh, dsg, dbo, T, nullptr);
        cudaMemcpy(ref.data(), dbo, out_bytes, cudaMemcpyDeviceToHost);
        strata::prefill::moe_routed_sum(dD, dslot, dw, dbo, T, nullptr);
        strata::prefill::moe_shared_finish(dsh, dsg, dbo, T, nullptr);
        cudaMemcpy(shared_only.data(), dbo, out_bytes, cudaMemcpyDeviceToHost);
        for (void* p : {(void*) dD, (void*) dw, (void*) dsh, (void*) dsg, (void*) dslot, (void*) dbo}) cudaFree(p);
    }
    int bad = 0;
    auto close = [&](const std::vector<float>& got, const char* what) {
        size_t nonfinite = 0, off = 0;
        double worst = 0.0;
        for (size_t i = 0; i < ref.size(); ++i) {
            if (!std::isfinite(got[i])) { ++nonfinite; continue; }
            const double e = std::abs((double) got[i] - (double) ref[i]) / std::max(1.0, std::abs((double) ref[i]));
            worst = std::max(worst, e);
            off += e > 1e-5;
        }
        const bool ok = nonfinite == 0 && off == 0;
        std::printf("%s: %zu non-finite, %zu beyond 1e-5 (worst %.2e) - %s\n", what, nonfinite, off, worst,
                    ok ? "PASS" : "FAIL");
        bad += !ok;
    };
    std::vector<char> local((size_t) ROWS);
    std::bernoulli_distribution half(0.5);
    for (auto& v : local) v = half(rng);
    close(split_run(Dm, slot, w, shared, sg, local), "random ownership");
    for (int64_t i = 0; i < ROWS; ++i) local[(size_t) i] = (i % K) % 2;   // alternating ranks within a token
    close(split_run(Dm, slot, w, shared, sg, local), "alternating ranks");
    std::fill(local.begin(), local.end(), 1);                               // the 4070 owns nothing
    close(split_run(Dm, slot, w, shared, sg, local), "all on the 5060");
    std::fill(local.begin(), local.end(), 0);                               // the 5060 owns nothing: #35 D1's bytes
    {
        const std::vector<float> got = split_run(Dm, slot, w, shared, sg, local);
        size_t diff = 0;
        for (size_t i = 0; i < got.size(); ++i) diff += std::memcmp(&got[i], &shared_only[i], 4) != 0;
        std::printf("all on the 4070 vs moe_routed_sum + moe_shared_finish: %zu of %zu values differ - %s\n", diff,
                    got.size(), diff == 0 ? "PASS" : "FAIL");
        bad += diff != 0;
    }
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { std::printf("%s\n", cudaGetErrorString(e)); return 1; }
    return bad == 0 ? 0 : 1;
}
