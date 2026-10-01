// include/strata/core/routing_trace.hpp - the --dump-routing file format (#84: #85, #86).
//
// Every record has one shape: int32 layer, int32 k, then k int32 and k float32.  A route record (0 <= layer <
// n_layer) carries a position's routed expert ids and their router weights.  A NEGATIVE layer is a tag: its k int32
// are the tag's values and its floats are zero padding.  A reader that keeps only 0 <= layer < n_layer and advances
// by the record's own k (tools/make_profile.py does) skips every tag without knowing it.
//
// Order inside a verify window: for each layer, positions 0..n-1 (layer-major), then one commit tag.  Position 0 is
// the committed input token, positions 1..n_accepted the accepted drafts, the rest were rejected.  Only windows write
// route records: the batched prompt path writes none, so phase 0 means "prompt tokens read through windows" (serve
// prompt parts of at most --short-read fresh tokens).
//
// The owned and boot tags are a snapshot taken when the file opens.  Adaptive swaps change both later and are not
// traced: record with --adapt-swaps 0 when a replay must match the run.
#pragma once

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace strata::core::routing_trace {

/// 2: the commit tag carries the phase (version 1 had a separate phase tag, -5, now unused).
inline constexpr int32_t kFormatVersion = 2;

enum Tag : int32_t {
    kTagCommit = -1,    ///< [window_id, n_positions, n_accepted, phase (0 prompt, 1 decode)], after a window's routes
    kTagFormat = -2,    ///< [version], the first record of a file
    kTagRequest = -3,   ///< [request_id], a served request starts
    kTagOwned = -4,     ///< flattened layer * n_expert + expert ids the GPU tiers own when the file opens
    kTagBoot = -6,      ///< flattened ids resident in the host tier after the boot fill (capacity mode)
};

/// One route record, in one write.  `weights` may be null: the record then carries zeros (the fused verify dispatch
/// does not surface the router weights; readers take zeros as unit weights).
inline void write_route(std::FILE* f, int32_t layer, int32_t k, const int32_t* ids, const float* weights) {
    if (k <= 64) {
        uint32_t buf[2 + 128] = {};
        buf[0] = (uint32_t) layer;
        buf[1] = (uint32_t) k;
        std::memcpy(buf + 2, ids, (size_t) k * sizeof(int32_t));
        if (weights != nullptr) std::memcpy(buf + 2 + k, weights, (size_t) k * sizeof(float));
        std::fwrite(buf, sizeof(uint32_t), (size_t) (2 + 2 * k), f);
        return;
    }
    const int32_t rec[2] = {layer, k};
    std::fwrite(rec, sizeof rec, 1, f);
    std::fwrite(ids, sizeof(int32_t), (size_t) k, f);
    if (weights != nullptr) {
        std::fwrite(weights, sizeof(float), (size_t) k, f);
    } else {
        static const float zeros[64] = {};
        for (int32_t left = k; left > 0; left -= 64) std::fwrite(zeros, sizeof(float), (size_t) (left < 64 ? left : 64), f);
    }
}

/// One tag record: `tag` must be negative.
inline void write_tag(std::FILE* f, int32_t tag, const int32_t* values, int32_t n) {
    write_route(f, tag, n, values, nullptr);
}

inline void write_ids(std::FILE* f, int32_t tag, const std::vector<int32_t>& ids) {
    write_tag(f, tag, ids.data(), (int32_t) ids.size());
}

inline void write_format(std::FILE* f) { write_tag(f, kTagFormat, &kFormatVersion, 1); }

inline void write_commit(std::FILE* f, int32_t window_id, int32_t n_positions, int32_t n_accepted, int32_t phase) {
    const int32_t v[4] = {window_id, n_positions, n_accepted, phase};
    write_tag(f, kTagCommit, v, 4);
}

inline void write_request(std::FILE* f, int32_t request_id) { write_tag(f, kTagRequest, &request_id, 1); }

}  // namespace strata::core::routing_trace
