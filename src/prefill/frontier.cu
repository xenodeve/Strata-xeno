// #41: the Dm frontier's host planner and its two kernels (see include/strata/prefill/frontier.hpp).
#include "strata/prefill/frontier.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>

namespace strata::prefill {
namespace {
constexpr int64_t N = 2560;   // n_embd
constexpr int KMAX = 10;      // the routed sum's K (moe_routed_sum's fmaf chain)

// one block column range of one token's run of ranks: s starts at 0 for rank 0, else at the running sum
__global__ void frontier_commit_kernel(const int32_t* __restrict__ groups, const int32_t* __restrict__ plan,
                                       const float* __restrict__ g, const float* __restrict__ pool,
                                       const float* __restrict__ w, float* __restrict__ acc) {
    const int32_t* gr = groups + 4 * (int64_t) blockIdx.x;
    const int64_t t = gr[0];
    const int k0 = gr[1], n = gr[2];
    const int32_t* src = plan + gr[3];
    const int64_t d = (int64_t) blockIdx.y * blockDim.x + threadIdx.x;
    if (d >= N) return;
    float s = k0 == 0 ? 0.0f : acc[t * N + d];
    for (int j = 0; j < n; ++j) {
        const int32_t r = src[j];
        const float* row = r >= 0 ? g + (int64_t) r * N : pool + (int64_t) (-1 - r) * N;
        s = fmaf(w[t * KMAX + k0 + j], row[d], s);
    }
    acc[t * N + d] = s;
}
__global__ void frontier_copy_kernel(const int32_t* __restrict__ copies, const float* __restrict__ g,
                                     float* __restrict__ pool) {
    const int32_t* c = copies + 2 * (int64_t) blockIdx.x;
    const int64_t d = (int64_t) blockIdx.y * blockDim.x + threadIdx.x;
    if (d >= N) return;
    pool[(int64_t) c[1] * N + d] = g[(int64_t) c[0] * N + d];
}
}  // namespace

bool plan_frontier(const int32_t* slot, int64_t T, int64_t K, const std::vector<std::pair<int64_t, int64_t>>& subs,
                   int32_t pool_cap, FrontierPlan& plan, std::string& err) {
    if (K != KMAX) { err = "frontier: K must be 10"; return false; }
    const int64_t R = T * K;
    std::vector<int32_t> tk_of_row((size_t) R, -1);
    for (int64_t i = 0; i < R; ++i) {
        if (slot[i] < 0 || slot[i] >= R) { err = "frontier: slot out of range"; return false; }
        tk_of_row[(size_t) slot[i]] = (int32_t) i;
    }
    // where each (t, k) row is right now: -1 not arrived, >= 0 in the pool (slot index), -2 committed
    std::vector<int32_t> where((size_t) R, -1);
    std::vector<int32_t> next((size_t) T, 0);
    std::vector<int32_t> free_slots;
    free_slots.reserve((size_t) pool_cap);
    for (int32_t s = pool_cap - 1; s >= 0; --s) free_slots.push_back(s);
    std::vector<int32_t> in_sub((size_t) R, -1);   // (t, k) -> its row within the current sub-product, or -1
    plan.data.clear();
    plan.subs.assign(subs.size(), {});
    plan.peak = 0;
    int32_t held = 0;
    std::vector<int32_t> touched, grp, src;
    for (size_t b = 0; b < subs.size(); ++b) {
        const int64_t r0 = subs[b].first, nr = subs[b].second;
        touched.clear();
        for (int64_t r = r0; r < r0 + nr; ++r) {
            const int32_t i = tk_of_row[(size_t) r];
            if (i < 0) { err = "frontier: a row with no (t, k)"; return false; }
            in_sub[(size_t) i] = (int32_t) (r - r0);
            touched.push_back(i / (int32_t) K);
        }
        std::sort(touched.begin(), touched.end());
        touched.erase(std::unique(touched.begin(), touched.end()), touched.end());
        // commits: each touched token's run of ranks that are now all present, in k order
        grp.clear();
        src.clear();
        for (int32_t t : touched) {
            const int32_t k0 = next[(size_t) t];
            int32_t n = 0;
            const int32_t soff = (int32_t) src.size();
            for (int32_t k = k0; k < K; ++k) {
                const size_t i = (size_t) t * K + k;
                if (in_sub[i] >= 0) src.push_back(in_sub[i]);
                else if (where[i] >= 0) { src.push_back(-1 - where[i]); free_slots.push_back(where[i]); --held; }
                else break;
                where[i] = -2;
                ++n;
            }
            if (n == 0) continue;
            next[(size_t) t] = k0 + n;
            grp.insert(grp.end(), {t, k0, n, soff});
        }
        // copies: this sub-product's rows that must wait (after the commits, which may have freed slots)
        std::vector<int32_t> cp;
        for (int64_t r = r0; r < r0 + nr; ++r) {
            const size_t i = (size_t) tk_of_row[(size_t) r];
            if (where[i] == -2) continue;
            if (free_slots.empty()) {
                err = "frontier: the pool needs more than " + std::to_string(pool_cap) + " rows";
                return false;
            }
            const int32_t s = free_slots.back();
            free_slots.pop_back();
            where[i] = s;
            ++held;
            cp.insert(cp.end(), {(int32_t) (r - r0), s});
        }
        plan.peak = std::max(plan.peak, held);
        for (int64_t r = r0; r < r0 + nr; ++r) in_sub[(size_t) tk_of_row[(size_t) r]] = -1;
        // lay the sub-product's arrays out: sources (their offsets become absolute), groups, copies
        FrontierPlan::Sub& sb = plan.subs[b];
        const int32_t base_src = (int32_t) plan.data.size();
        plan.data.insert(plan.data.end(), src.begin(), src.end());
        for (size_t j = 3; j < grp.size(); j += 4) grp[j] += base_src;
        sb.groups = (int32_t) plan.data.size();
        sb.n_groups = (int32_t) (grp.size() / 4);
        plan.data.insert(plan.data.end(), grp.begin(), grp.end());
        sb.copies = (int32_t) plan.data.size();
        sb.n_copies = (int32_t) (cp.size() / 2);
        plan.data.insert(plan.data.end(), cp.begin(), cp.end());
    }
    for (int64_t t = 0; t < T; ++t)
        if (next[(size_t) t] != K) { err = "frontier: a token's ranks did not all arrive"; return false; }
    return true;
}

void frontier_run(const FrontierPlan& plan, size_t sub, const int32_t* dplan, const float* g, float* pool, const float* w,
                  float* acc, void* stream) {
    const FrontierPlan::Sub& sb = plan.subs[sub];
    const unsigned cols = (unsigned) ((N + 255) / 256);
    if (sb.n_groups > 0)
        frontier_commit_kernel<<<dim3((unsigned) sb.n_groups, cols), 256, 0, (cudaStream_t) stream>>>(
            dplan + sb.groups, dplan, g, pool, w, acc);
    if (sb.n_copies > 0)
        frontier_copy_kernel<<<dim3((unsigned) sb.n_copies, cols), 256, 0, (cudaStream_t) stream>>>(dplan + sb.copies, g,
                                                                                                      pool);
}

}  // namespace strata::prefill
