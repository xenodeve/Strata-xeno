// src/prefill/split_plan.cpp - #113 (see strata/prefill/split_plan.hpp)
#include "strata/prefill/split_plan.hpp"

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
    if (env == nullptr || *env == '\0') return 2048;
    char* end = nullptr;
    const long long v = std::strtoll(env, &end, 10);
    return (end != nullptr && *end == '\0' && v >= 256 && v <= 65536) ? (int64_t) v : 2048;
}

bool wave_lane_ok(int64_t lane_chunk, int64_t split_min) { return lane_chunk >= (split_min > 2048 ? split_min : 2048); }

}  // namespace strata::prefill
