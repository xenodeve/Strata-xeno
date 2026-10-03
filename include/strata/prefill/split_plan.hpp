// strata/prefill/split_plan.hpp - #113: which layers of a prompt chunk the expert split (#35,
// STRATA_PREFILL_EXPERT_SPLIT=1) sends to the 4070.  The split runs a layer's routed experts there through MMQ, so a
// layer whose expert formats MMQ has no tile for (Swift 1.5 IQ2_XS: IQ1_M gate/up in layers 8, 13 and 37) stays on
// CUDA0's one-card path inside the same chunk.  Before #113 one such layer turned the whole split off, silently.
#pragma once

#include <cstddef>
#include <vector>

namespace strata::prefill {

struct SplitPlan {
    bool usable = false;           // a native pack with at least one MMQ layer: the split can run
    bool full = false;             // every layer is on MMQ: the one-card buffers may stay short-chunk sized
    std::vector<int> one_card;     // the layers that stay on CUDA0 (ascending)
};

/// `native`: the experts come from a native pack; `mmq_layer[l]`: layer l's experts are on MMQ.
SplitPlan split_plan(bool native, const std::vector<char>& mmq_layer);

/// #115: the wave (two lanes overlapping the cards) may run on this plan: wherever the split can run, since a
/// one-card layer of a split chunk publishes its own hand-off to the other lane.
bool wave_ok(const SplitPlan& p);

}  // namespace strata::prefill
