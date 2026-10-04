// src/prefill/kv_stage_plan.cpp - see include/strata/prefill/kv_stage_plan.hpp.
#include "strata/prefill/kv_stage_plan.hpp"

#include <algorithm>

namespace strata::prefill {

int64_t kv_stage_pages(int64_t kv_end, int64_t floor_cells, int64_t n_pages, int64_t page_size) {
    if (kv_end <= 0 || page_size <= 0 || n_pages <= 0) return n_pages;
    int64_t cells = 1;
    while (cells < kv_end && cells <= n_pages * page_size) cells <<= 1;   // the next power of two (or past the cap)
    cells = std::max(cells, floor_cells);
    return std::min(n_pages, (cells + page_size - 1) / page_size);
}

}  // namespace strata::prefill
