#pragma once
// #41: a Dm frontier.  The routed sum adds a token's K expert rows in rank order k = 0..K-1 (an fmaf chain byte identity
// depends on).  Instead of holding every (token, k) row of a layer (Dm, T*K rows), rows arrive a sub-product at a time
// and are added to the token's running sum as soon as every lower rank has been added; a row that arrives early waits
// in a pool slot.  The host knows the layer's routing before the experts run, so plan_frontier decides everything up
// front and the kernels only follow the plan - the same fmaf chain, the same bytes (xeno_frontier_combine_parity).
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace strata::prefill {

struct FrontierPlan {
    /// every sub-product's arrays back to back, uploaded once per layer:
    ///   groups: n_groups x {token, k0, n, src offset}; sources: n per group (>= 0: the sub-product's row, relative to
    ///   its first row; < 0: pool slot -1-s); copies: n_copies x {sub-product row, pool slot}
    std::vector<int32_t> data;
    struct Sub { int32_t n_groups = 0, groups = 0, n_copies = 0, copies = 0; };
    std::vector<Sub> subs;
    int32_t peak = 0;   ///< the most rows waiting in the pool at once
};

/// slot: (t, k) -> the row it lands in (rows laid out by expert); subs: each sub-product's rows [first, first + count),
/// in the order they are produced.  false (and err) when the pool would need more than pool_cap slots.
bool plan_frontier(const int32_t* slot, int64_t T, int64_t K, const std::vector<std::pair<int64_t, int64_t>>& subs,
                   int32_t pool_cap, FrontierPlan& plan, std::string& err);

/// sub-product `sub` of the plan: its rows are in g (row i = the sub-product's i-th row, n_embd 2560 f32); commits the
/// planned ranks into acc (T x 2560, the routed sum; rank 0 starts from zero) with weights w (T x K), then moves the rows
/// that must wait into the pool.  dplan: plan.data on the device.
void frontier_run(const FrontierPlan& plan, size_t sub, const int32_t* dplan, const float* g, float* pool, const float* w,
                  float* acc, void* stream);

}  // namespace strata::prefill
