// include/strata/core/routing_trace.hpp - the --dump-routing file format (#84 / #85).
//
// Every record has one shape: int32 layer, int32 k, then k int32 and k float32.  A route record (0 <= layer <
// n_layer) carries a position's routed expert ids and their router weights.  A NEGATIVE layer is a tag: its k int32
// are the tag's values and its floats are zero padding.  A reader that keeps only 0 <= layer < n_layer and advances
// by the record's own k (tools/make_profile.py does) skips every tag without knowing it.
//
// Order inside a verify window: for each layer, positions 0..n-1 (layer-major), then one commit tag.  Position 0 is
// the committed input token, positions 1..n_accepted the accepted drafts, the rest were rejected.
#pragma once

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace strata::core::routing_trace {

inline constexpr int32_t kFormatVersion = 1;

enum Tag : int32_t {
    kTagCommit = -1,    ///< [window_id, n_positions, n_accepted], after a verify window's route records
    kTagFormat = -2,    ///< [version], the first record of a file
    kTagRequest = -3,   ///< [request_id], a served request starts
    kTagOwned = -4,     ///< flattened layer * n_expert + expert ids the GPU tiers own at boot
    kTagPhase = -5,     ///< [0 = prefill, 1 = decode], the phase of the route records that follow
    kTagBoot = -6,      ///< flattened ids resident in the host tier after the boot fill (capacity mode)
};

/// One route record.  `weights` may be null: the record then carries zeros (the fused verify dispatch does not
/// surface the router weights; readers take zeros as unit weights).
inline void write_route(std::FILE* f, int32_t layer, int32_t k, const int32_t* ids, const float* weights) {
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

inline void write_format(std::FILE* f) { write_tag(f, kTagFormat, &kFormatVersion, 1); }

inline void write_commit(std::FILE* f, int32_t window_id, int32_t n_positions, int32_t n_accepted) {
    const int32_t v[3] = {window_id, n_positions, n_accepted};
    write_tag(f, kTagCommit, v, 3);
}

inline void write_request(std::FILE* f, int32_t request_id) { write_tag(f, kTagRequest, &request_id, 1); }
inline void write_phase(std::FILE* f, int32_t phase) { write_tag(f, kTagPhase, &phase, 1); }
inline void write_owned(std::FILE* f, const std::vector<int32_t>& flat_ids) {
    write_tag(f, kTagOwned, flat_ids.data(), (int32_t) flat_ids.size());
}
inline void write_boot(std::FILE* f, const std::vector<int32_t>& flat_ids) {
    write_tag(f, kTagBoot, flat_ids.data(), (int32_t) flat_ids.size());
}

struct Record {
    int32_t layer = 0;
    std::vector<int32_t> ids;
    std::vector<float> weights;
};

/// Every record of a trace file, in order; false on a truncated record or an unreadable file.
inline bool read_all(const std::string& path, std::vector<Record>& out) {
    out.clear();
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (f == nullptr) return false;
    std::fseek(f, 0, SEEK_END);
    const long size = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    long used = 0;
    bool ok = true;
    int32_t rec[2];
    while (std::fread(rec, sizeof rec, 1, f) == 1) {
        if (rec[1] < 0 || rec[1] > (1 << 20)) { ok = false; break; }
        Record r;
        r.layer = rec[0];
        r.ids.resize((size_t) rec[1]);
        r.weights.resize((size_t) rec[1]);
        if (std::fread(r.ids.data(), sizeof(int32_t), r.ids.size(), f) != r.ids.size() ||
            std::fread(r.weights.data(), sizeof(float), r.weights.size(), f) != r.weights.size()) {
            ok = false;
            break;
        }
        used += (long) (sizeof rec + 8 * r.ids.size());
        out.push_back(std::move(r));
    }
    std::fclose(f);
    return ok && used == size;   // a 1-7 byte tail (a partial header) is a truncation too
}

}  // namespace strata::core::routing_trace
