#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace strata::core {

struct SecondaryCandidate {
    int32_t layer = -1;
    int32_t expert = -1;
    uint64_t bytes = 0;
};

// Keep the profile's rank order, excluding the primary GPU's resident pairs.
inline bool secondary_candidates(const std::vector<std::pair<int32_t, int32_t>>& ranked,
                                 const std::vector<int32_t>& primary_residency,
                                 const std::vector<uint64_t>& layer_blob_bytes, int64_t n_expert,
                                 std::vector<SecondaryCandidate>& out, std::string& err) {
    out.clear();
    if (n_expert <= 0 || layer_blob_bytes.empty() ||
        (uint64_t) n_expert > std::numeric_limits<size_t>::max() / layer_blob_bytes.size() ||
        primary_residency.size() != layer_blob_bytes.size() * (size_t) n_expert) {
        err = "secondary profile geometry does not match primary residency";
        return false;
    }
    std::vector<uint8_t> seen(primary_residency.size(), 0);
    std::vector<SecondaryCandidate> selected;
    selected.reserve(ranked.size());
    for (const auto& [layer, expert] : ranked) {
        if (layer < 0 || (size_t) layer >= layer_blob_bytes.size() ||
            expert < 0 || expert >= n_expert) {
            err = "secondary profile contains an out-of-range pair";
            return false;
        }
        const size_t index = (size_t) layer * (size_t) n_expert + (size_t) expert;
        if (seen[index] != 0 || layer_blob_bytes[(size_t) layer] == 0) {
            err = "secondary profile contains a duplicate pair or zero-size expert";
            return false;
        }
        seen[index] = 1;
        if (primary_residency[index] < 0)
            selected.push_back({layer, expert, layer_blob_bytes[(size_t) layer]});
    }
    out.swap(selected);
    err.clear();
    return true;
}

} // namespace strata::core
