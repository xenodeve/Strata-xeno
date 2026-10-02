// include/strata/core/lend_rows.hpp - #102: which residency rows a prompt-path loan takes.
#pragma once

#include <cstdint>
#include <utility>
#include <vector>

namespace strata::core {

/// A participant's loan (CUDA0 or one layer-split stage): every row of ITS layers [lb, le) whose slot in ITS cache is
/// >= `first` is appended to `lent` as (row index, slot) and marked not resident in `host_res`.  Slot numbers are per
/// cache, so the layer range is what keeps one participant from lending another's rows.  Giving the loan back is
/// host_res[row] = slot for each pair (refill_lent does it once the slot holds its expert again).
inline void lend_rows(std::vector<int32_t>& host_res, int64_t n_expert, int64_t lb, int64_t le, int32_t first,
                      std::vector<std::pair<int32_t, int32_t>>& lent) {
    for (int64_t i = lb * n_expert; i < le * n_expert && i < (int64_t) host_res.size(); ++i)
        if (host_res[(size_t) i] >= first) {   // kNotResident (-1) is never >= a slot number
            lent.emplace_back((int32_t) i, host_res[(size_t) i]);
            host_res[(size_t) i] = -1;          // strata::core::kNotResident (expert_cache.hpp)
        }
}

}  // namespace strata::core
