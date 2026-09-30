// #41 gate 3: a Dm frontier must write the same routed sum as moe_routed_sum over the whole Dm, byte for byte.
// Rows arrive a sub-product at a time (as mmq::expert_rows writes them); plan_frontier decides, on the host, which
// (t, k) rows commit into the token's running sum at once (strictly in k order, the same fmaf chain) and which wait
// in a pool slot; frontier_commit and frontier_copy follow the plan.  A synthetic layer: 512 tokens, top-10 of 64
// experts, rows laid out by expert as the prompt path lays them out, arriving in blocks of 700 rows.
#include "strata/prefill/frontier.hpp"
#include "strata/prefill/kernels.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <random>
#include <vector>

namespace {
constexpr int64_t T = 512, K = 10, N = 2560, NEX = 64, BLOCK = 700;
template <typename V>
void* up(const V& v) {
    void* d = nullptr;
    cudaMalloc(&d, v.size() * sizeof(v[0]));
    cudaMemcpy(d, v.data(), v.size() * sizeof(v[0]), cudaMemcpyHostToDevice);
    return d;
}
}  // namespace

int main() {
    using namespace strata::prefill;
    std::mt19937 rng(41);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    // routing: top-10 distinct experts per token; rows by expert (stable in (t, k) order), slot = (t, k) -> row
    std::vector<int32_t> ex((size_t) (T * K));
    for (int64_t t = 0; t < T; ++t) {
        std::vector<int32_t> p((size_t) NEX);
        std::iota(p.begin(), p.end(), 0);
        std::shuffle(p.begin(), p.end(), rng);
        for (int64_t k = 0; k < K; ++k) ex[(size_t) (t * K + k)] = p[(size_t) k];
    }
    std::vector<int32_t> cnt((size_t) NEX, 0), off((size_t) NEX + 1, 0), slot((size_t) (T * K));
    for (int32_t e : ex) ++cnt[(size_t) e];
    for (int64_t e = 0; e < NEX; ++e) off[(size_t) e + 1] = off[(size_t) e] + cnt[(size_t) e];
    std::vector<int32_t> at(off.begin(), off.end() - 1);
    for (int64_t i = 0; i < T * K; ++i) slot[(size_t) i] = at[(size_t) ex[(size_t) i]]++;
    std::vector<float> Dm((size_t) (T * K * N)), w((size_t) (T * K));
    for (auto& v : Dm) v = nd(rng);
    for (auto& v : w) v = std::abs(nd(rng)) * 0.2f;
    // the reference: the routed sum over the whole Dm
    auto *dDm = (float*) up(Dm), *dw = (float*) up(w);
    auto* dslot = (int32_t*) up(slot);
    float *ref = nullptr, *acc = nullptr;
    cudaMalloc((void**) &ref, (size_t) (T * N) * 4);
    cudaMalloc((void**) &acc, (size_t) (T * N) * 4);
    moe_routed_sum(dDm, dslot, dw, ref, T, nullptr);
    // the frontier: rows arrive in blocks of BLOCK rows, in row order
    std::vector<std::pair<int64_t, int64_t>> subs;
    for (int64_t r0 = 0; r0 < T * K; r0 += BLOCK) subs.push_back({r0, std::min<int64_t>(BLOCK, T * K - r0)});
    FrontierPlan plan;
    std::string err;
    const int32_t cap = (int32_t) (T * K);   // room enough; the peak is reported
    if (!plan_frontier(slot.data(), T, K, subs, cap, plan, err)) { std::printf("plan: %s\n", err.c_str()); return 1; }
    float *g = nullptr, *pool = nullptr;
    cudaMalloc((void**) &g, (size_t) (BLOCK * N) * 4);
    cudaMalloc((void**) &pool, (size_t) cap * N * 4);
    auto* dplan = (int32_t*) up(plan.data);
    for (size_t b = 0; b < subs.size(); ++b) {
        cudaMemcpy(g, dDm + subs[b].first * N, (size_t) (subs[b].second * N) * 4, cudaMemcpyDeviceToDevice);
        frontier_run(plan, b, dplan, g, pool, dw, acc, nullptr);
    }
    cudaDeviceSynchronize();
    std::vector<float> a((size_t) (T * N)), r((size_t) (T * N));
    cudaMemcpy(a.data(), acc, a.size() * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(r.data(), ref, r.size() * 4, cudaMemcpyDeviceToHost);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { std::printf("%s\n", cudaGetErrorString(e)); return 1; }
    size_t diff = 0;
    for (size_t i = 0; i < a.size(); ++i) diff += std::memcmp(&a[i], &r[i], 4) != 0;
    std::printf("frontier vs moe_routed_sum: %zu of %zu values differ; pool peak %d rows of %lld (%.1f %%)\n%s\n", diff,
                a.size(), plan.peak, (long long) (T * K), 100.0 * plan.peak / (double) (T * K), diff == 0 ? "PASS" : "FAIL");
    return diff == 0 ? 0 : 1;
}
