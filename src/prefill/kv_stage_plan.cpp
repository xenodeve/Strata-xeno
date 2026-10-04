// src/prefill/kv_stage_plan.cpp - see include/strata/prefill/kv_stage_plan.hpp.
#include "strata/prefill/kv_stage_plan.hpp"

#include <algorithm>
#include <bit>

namespace strata::prefill {

int64_t kv_stage_pages(int64_t kv_end, int64_t n_pages, int64_t page_size) {
    if (kv_end <= 0 || page_size <= 0 || n_pages <= 0) return n_pages;
    // the next power of two of the end, clamped to the context first (bit_ceil has no result past 2^63)
    const int64_t cells = (int64_t) std::bit_ceil((uint64_t) std::min(kv_end, n_pages * page_size));
    return std::min(n_pages, (cells + page_size - 1) / page_size);
}

}  // namespace strata::prefill
