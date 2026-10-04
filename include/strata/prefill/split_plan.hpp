// strata/prefill/split_plan.hpp - #113: which layers of a prompt chunk the expert split (#35,
// STRATA_PREFILL_EXPERT_SPLIT=1) sends to the 4070.  The split runs a layer's routed experts there through MMQ, so a
// layer whose expert formats MMQ has no tile for (Swift 1.5 IQ2_XS: IQ1_M gate/up in layers 8, 13 and 37) stays on
// CUDA0's one-card path inside the same chunk.  Before #113 one such layer turned the whole split off, silently.
#pragma once

#include <cstddef>
#include <atomic>
#include <cstdint>
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

/// #119: the old constant STREAM_ALL_MIN - the split's default chunk floor, and the smallest lane the wave runs.
inline constexpr int64_t kSplitMinDefault = 2048;

/// #119: the smallest prompt chunk the split runs on (and that streams every expert under the split layout):
/// STRATA_PREFILL_SPLIT_MIN, 256-2048 tokens (above 2048 nothing was measured, and a floor above the wave's lane
/// runs both lanes unsplit); unset or anything else is the old constant, 2048.
int64_t split_min_from(const char* env);

/// #119: a prompt part waves (two lanes of half the chunk) only where each lane still reads 2048+ tokens, whatever
/// the split's floor: lanes of ~900-1,100 tokens read 1,768-2,248-token parts in 5.1-5.6 s against 3.2 s on one
/// lane (tl119-*.json).
bool wave_lane_ok(int64_t lane_chunk, int64_t split_min);

/// #133: a split layer's routed rows, grouped by expert: the experts the 4070 computes first, then the ones CUDA0
/// computes from its own cache (`local[e]`), each block in `at_order` (the layer's expert walk order), so each card's
/// experts are one contiguous row range.  `off[e]` is expert e's first row and `off[n]` the total; returns the row the
/// local block starts at - a (token, k) pair is CUDA0's exactly when its row is at or past it.  `cnt[e]`: expert e's rows.
int64_t split_row_layout(const std::vector<int32_t>& cnt, const std::vector<int32_t>& at_order,
                         const std::vector<char>& local, std::vector<int32_t>& off);


/// #142: a split chunk copies only the host experts its routing reaches when it is shorter than this - the streamed
/// walk (every host expert of a split layer, routed or not) pays only on a chunk that routes to nearly all of them.
/// A Claude Code part of 408-1,837 tokens copied 13.3-13.6K blobs to the 4070 for 5.4-7.6K routed ones (#142).
inline constexpr int64_t kSplitRoutedMaxDefault = 2048;

/// #142: STRATA_SPLIT_ROUTED_MAX, a token count (0 = off: every chunk streams the walk); unset or not a non-negative
/// number is the default.
int64_t split_routed_max_from(const char* env);

/// #142: whether a split chunk of `T` tokens copies only its routed host experts.  Never in a wave: both lanes read
/// one plan, and each lane routes its own chunk.
bool split_routed_only(int64_t T, bool wave, int64_t routed_max);

/// #142: a plan entry's state once its layer is routed (kNeedUnknown before): the stager and the issuer wait for it.
inline constexpr uint8_t kNeedUnknown = 0, kNeedCopy = 1, kNeedSkip = 2;

/// #142: a layer's plan entries (`entry_e[i]`: the expert of entry i) from its routing (`cnt[e]`: rows of expert e):
/// kNeedCopy where a token routes to the expert, else kNeedSkip; each store is a release and wakes its waiters.
void split_mark_routed(const int32_t* entry_e, size_t n, const int32_t* cnt, std::atomic<uint8_t>* need);

}  // namespace strata::prefill
