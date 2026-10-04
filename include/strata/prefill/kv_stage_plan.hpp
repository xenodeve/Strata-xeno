// strata/prefill/kv_stage_plan.hpp - #122: how many pages of one QSA layer the prompt path's KV-streaming staging pool
// holds.  With --kv-resident the prompt path stages a layer's cells [0, p0) from the host copy into an identity-layout
// pool and appends the chunk's cells after them, so the pool must hold the request's end position - not, as before
// #122, every page of the context limit (264 MiB per wave lane at 262K, carved from borrowed expert slots: +395 slots
// lent per request and 0.2-0.5 s per Claude Code turn, #106).
#pragma once

#include <cstdint>

namespace strata::prefill {

/// The pages the staging pool holds for a request whose prompt ends at `kv_end` cells: the pages of the next power of
/// two >= `kv_end` cells, at most `n_pages`.  The power of two keeps a growing conversation's layouts to a few sizes.
/// `kv_end` <= 0: the worst case, `n_pages` (boot-time sizing: the chunk choice and the lendable tail).  No floor: one
/// at the resident window (65,536 cells) read every Claude Code part 20-50 ms slower (#122).
int64_t kv_stage_pages(int64_t kv_end, int64_t n_pages, int64_t page_size);

}  // namespace strata::prefill
