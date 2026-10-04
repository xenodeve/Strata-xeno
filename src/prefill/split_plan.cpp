// src/prefill/split_plan.cpp - #113 (see strata/prefill/split_plan.hpp)
#include "strata/prefill/split_plan.hpp"

#include <algorithm>
#include <cstdlib>

namespace strata::prefill {

SplitPlan split_plan(bool native, const std::vector<char>& mmq_layer) {
    SplitPlan p;
    if (!native) return p;
    for (size_t l = 0; l < mmq_layer.size(); ++l) {
        if (mmq_layer[l]) p.usable = true;
        else p.one_card.push_back((int) l);
    }
    p.full = p.usable && p.one_card.empty();
    return p;
}

bool wave_ok(const SplitPlan& p) { return p.usable; }

int64_t split_min_from(const char* env) {
    if (env == nullptr || *env == '\0') return kSplitMinDefault;
    char* end = nullptr;
    const long long v = std::strtoll(env, &end, 10);
    return (end != nullptr && *end == '\0' && v >= 256 && v <= kSplitMinDefault) ? (int64_t) v : kSplitMinDefault;
}

int64_t split_row_layout(const std::vector<int32_t>& cnt, const std::vector<int32_t>& at_order,
                         const std::vector<char>& local, std::vector<int32_t>& off) {
    off.assign(cnt.size() + 1, 0);
    int32_t row = 0;
    int64_t local_first = 0;
    for (int pass = 0; pass < 2; ++pass) {   // the 4070's experts, then CUDA0's
        if (pass == 1) local_first = row;
        for (const int32_t e : at_order) {
            if ((local[(size_t) e] != 0) != (pass == 1)) continue;
            off[(size_t) e] = row;
            row += cnt[(size_t) e];
        }
    }
    off[cnt.size()] = row;
    return local_first;
}

bool wave_lane_ok(int64_t lane_chunk, int64_t split_min) { return lane_chunk >= std::max(split_min, kSplitMinDefault); }


int64_t split_routed_max_from(const char* env) {
    if (env == nullptr || *env == '\0') return kSplitRoutedMaxDefault;
    char* end = nullptr;
    const long long v = std::strtoll(env, &end, 10);
    return (end != nullptr && *end == '\0' && v >= 0) ? (int64_t) v : kSplitRoutedMaxDefault;
}

bool split_routed_only(int64_t T, bool wave, int64_t routed_max) { return !wave && routed_max > 0 && T < routed_max; }

void split_mark_routed(const int32_t* entry_e, size_t n, const int32_t* cnt, std::atomic<uint8_t>* need) {
    for (size_t i = 0; i < n; ++i) {
        need[i].store(cnt[entry_e[i]] > 0 ? kNeedCopy : kNeedSkip, std::memory_order_release);
        need[i].notify_all();
    }
}

}  // namespace strata::prefill
