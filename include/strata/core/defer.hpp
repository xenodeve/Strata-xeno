#pragma once
// #254 (opt-in STRATA_DEFER_RANK=R): expert deferral in the verify window (KTransformers, SOSP '25 §4.1).  A CPU-missed
// routed expert that every token of the window routes at rank R or lower is computed after the layer's other experts
// and its weighted output is added one layer later, so the GPU runs the next layer while the CPU pool still works.
// R = 0 defers every CPU-missed expert; unset (or negative) is off and the window is byte-identical.
#include <cstdint>
#include <cstdlib>

namespace strata::core {

/// What the verify window's host loop hands the CPU dispatch of one step (thread-local: the window's host thread).
struct DeferHook {
    bool active = false;                   // this step may defer (not the first or the last layer)
    int rank = -1;                         // defer an expert whose best rank in the window is >= rank
    volatile uint32_t* flag = nullptr;     // the GPU's CPU-share flag: raised once the immediate rows are in place
    uint32_t want = 0;
    const float* w = nullptr;              // the window rows' router weights, [t][k] (mapped)
    float* out = nullptr;                  // the deferred experts' weighted sum, [t][n_embd] (mapped)
    int64_t deferred_entries = 0;          // counted over the hook's life (diagnostics)
};

DeferHook& defer_hook();

/// #265 (opt-in STRATA_ROWS_F16=1): the verify window's CPU / 4070 rows also as fp16, which the 5060 reads over its
/// x4 link instead of the fp32 rows (half the bytes).  Set per step by the verify window's host loop (thread-local).
struct RowsF16Hook {
    uint16_t* out16 = nullptr;             // [n_tok * k][n_embd] fp16 (mapped); null: off for this step
};
RowsF16Hook& rows_f16_hook();
/// STRATA_ROWS_F16, read once (inline: the 4070 runner's library reads it too)
inline bool rows_f16_env() {
    static const bool on = [] { const char* v = std::getenv("STRATA_ROWS_F16"); return v != nullptr && v[0] == '1'; }();
    return on;
}
/// STRATA_DEFER_RANK, read once: -1 when unset or invalid
int defer_rank_env();

}  // namespace strata::core
