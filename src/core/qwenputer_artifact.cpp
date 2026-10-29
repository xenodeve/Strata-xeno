#include "strata/core/qwenputer_artifact.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <map>
#include <new>
#include <set>
#include <type_traits>

namespace strata::qwenputer {
static_assert(sizeof(Binding) == 192 && sizeof(Dimension) == 8 && sizeof(Feature) == 16);
static_assert(sizeof(Expression) == 8 && sizeof(Term) == 8 && sizeof(Channel) == 20);
static_assert(sizeof(Mapping) == 8 && sizeof(FfnTerm) == 12 && sizeof(Ffn) == 24);
static_assert(sizeof(Head) == 88 && sizeof(Output) == 12 && sizeof(InitialRow) == 520);
namespace {
struct Hash {
    std::array<std::uint32_t, 8> state{0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                                       0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
    std::array<std::uint8_t, 64> block{};
    std::uint64_t size = 0;
    std::size_t used = 0;
    void transform() {
        constexpr std::uint32_t k[]{
            0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
            0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
            0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
            0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
            0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
            0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
            0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
            0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};
        std::uint32_t w[64]{};
        for (int i = 0; i < 16; ++i)
            for (int j = 0; j < 4; ++j) w[i] = (w[i] << 8) | block[4 * i + j];
        for (int i = 16; i < 64; ++i) {
            const auto x = w[i - 15], y = w[i - 2];
            w[i] = w[i - 16] + (std::rotr(x,7) ^ std::rotr(x,18) ^ (x >> 3)) + w[i - 7]
                 + (std::rotr(y,17) ^ std::rotr(y,19) ^ (y >> 10));
        }
        auto [a,b,c,d,e,f,g,h] = state;
        for (int i = 0; i < 64; ++i) {
            const auto t1 = h + (std::rotr(e,6) ^ std::rotr(e,11) ^ std::rotr(e,25))
                          + ((e & f) ^ (~e & g)) + k[i] + w[i];
            const auto t2 = (std::rotr(a,2) ^ std::rotr(a,13) ^ std::rotr(a,22))
                          + ((a & b) ^ (a & c) ^ (b & c));
            h=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
        }
        state[0]+=a; state[1]+=b; state[2]+=c; state[3]+=d;
        state[4]+=e; state[5]+=f; state[6]+=g; state[7]+=h;
    }
    void update(std::span<const std::uint8_t> bytes) {
        size += bytes.size();
        while (!bytes.empty()) {
            const auto n = std::min(bytes.size(), block.size() - used);
            std::memcpy(block.data() + used, bytes.data(), n);
            used += n;
            bytes = bytes.subspan(n);
            if (used == 64) { transform(); used = 0; }
        }
    }
    Digest finish() {
        const auto bits = size * 8;
        const std::uint8_t one = 0x80, zero = 0;
        update({&one, 1});
        while (used != 56) update({&zero, 1});
        std::array<std::uint8_t, 8> tail{};
        for (int i = 0; i < 8; ++i) tail[7-i] = static_cast<std::uint8_t>(bits >> (8*i));
        update(tail);
        Digest result{};
        for (int i = 0; i < 8; ++i)
            for (int j = 0; j < 4; ++j) result[4*i+j] = static_cast<std::uint8_t>(state[i] >> (24-8*j));
        return result;
    }
};

template<class T> T scalar(std::span<const std::uint8_t> data, std::size_t offset) {
    T value;
    std::memcpy(&value, data.data() + offset, sizeof(T));
    return value;
}
bool range(std::int32_t start, std::int32_t count, std::size_t total) {
    return start >= 0 && count >= 0 && static_cast<std::size_t>(start) <= total
        && static_cast<std::size_t>(count) <= total - static_cast<std::size_t>(start);
}
bool index(std::int32_t i, std::size_t total) { return range(i, 1, total); }
bool boolean(std::int32_t i) { return i == 0 || i == 1; }
struct Section { std::span<const std::uint8_t> data; std::uint32_t stride; };
template<class T> std::vector<T> copy(const Section& section) {
    static_assert(std::is_trivially_copyable_v<T>);
    std::vector<T> result(section.data.size() / sizeof(T));
    if (!result.empty()) std::memcpy(result.data(), section.data.data(), section.data.size());
    return result;
}
constexpr std::uint32_t family_sizes[]{32768,32768,32768,4,65536,65536,512,8,16,16,128,256,
                                       1,1,1,1,1,1,1,1};
bool valid_feature(const Feature& f) {
    if (f.opcode < 0 || f.opcode > 11) return false;
    switch (f.opcode) {
    case 2: return f.p0 >= 0 && f.p0 < 20 && f.p1 == 0;
    case 3: {
        constexpr int widths[]{15,15,15,2,16,16,9,0,0,0,7,8};
        return f.p0 >= 0 && f.p0 < 12 && f.p1 >= 0 && f.p1 < widths[f.p0];
    }
    case 4: return f.p0 >= 0 && f.p0 < 8 && f.p1 == 0;
    case 5: return f.p0 >= 0 && f.p0 < 21 && f.p1 == 0;
    case 11: return f.p0 >= 0 && f.p0 < 16 && f.p1 >= 0 && f.p1 < 8;
    default: return f.p0 == 0 && f.p1 == 0;
    }
}
} // namespace

Digest sha256(std::span<const std::uint8_t> bytes) { Hash hash; hash.update(bytes); return hash.finish(); }
const Digest& arithmetic_abi() {
    static const Digest value{0x47,0xac,0xf4,0x8b,0x7f,0x07,0x43,0xf9,
        0x5a,0xa2,0x72,0xb4,0xf6,0x5a,0x10,0x27,0x63,0xe3,0x9c,0x25,
        0x05,0x04,0x6b,0xb2,0x94,0xe6,0x0a,0xc7,0xfd,0x99,0xad,0x0a};
    return value;
}
bool parse_artifact(std::span<const std::uint8_t> bytes, const Binding& binding, const LoadLimits& limits,
                    Artifact& output, MemoryPlan& memory, std::string& error) try {
    const auto fail = [&](const char* reason) { error = reason; return false; };
    error.clear();
    if constexpr (std::endian::native != std::endian::little) return fail("host_endian");
    if (bytes.size() < 64 || bytes.size() > std::min<std::uint64_t>(limits.file_bytes, 256ull << 20)) return fail("artifact_size");
    if (std::memcmp(bytes.data(), "STQPUT01", 8)) return fail("artifact_magic");
    if (scalar<std::uint16_t>(bytes, 8) != 1 || scalar<std::uint16_t>(bytes, 10) != 0) return fail("artifact_version");
    if (scalar<std::uint32_t>(bytes, 12) != 0x01020304) return fail("artifact_endian");
    const auto count = scalar<std::uint32_t>(bytes, 16), hbytes = scalar<std::uint32_t>(bytes, 20);
    if (scalar<std::uint64_t>(bytes, 24) != bytes.size() || bytes.size() % 64) return fail("artifact_size");
    if (count < 1 || count > 32 || hbytes != ((64 + count * 80 + 63) & ~63u)) return fail("section_count");
    if (hbytes > bytes.size()) return fail("section_bounds");
    const auto body_digest = sha256(bytes.subspan(64));
    if (std::memcmp(body_digest.data(), bytes.data() + 32, 32)) return fail("artifact_digest");
    for (std::size_t i = 64 + count * 80; i < hbytes; ++i) if (bytes[i]) return fail("section_padding");
    std::map<std::string, Section> sections;
    std::size_t end = hbytes;
    std::string previous;
    for (std::uint32_t i = 0; i < count; ++i) {
        const auto record = bytes.subspan(64 + i * 80, 80);
        std::size_t n = 0;
        while (n < 16 && record[n]) ++n;
        if (n == 0 || n == 16) return fail("section_name");
        std::string name(reinterpret_cast<const char*>(record.data()), n);
        for (std::size_t j = 0; j < 16; ++j) {
            const auto c = record[j];
            if (j >= n ? c != 0 : !((c >= 'a' && c <= 'z') || (j && ((c >= '0' && c <= '9') || c == '_'))))
                return fail("section_name");
        }
        if (name <= previous) return fail("section_name");
        const auto start = scalar<std::uint64_t>(record, 16), size = scalar<std::uint64_t>(record, 24);
        const auto stride = scalar<std::uint32_t>(record, 32);
        if (start != end || start % 64 || start > bytes.size() || size > bytes.size() - start) return fail("section_bounds");
        if (stride < 1 || stride > 4096 || size % stride) return fail("section_stride");
        if (scalar<std::uint32_t>(record, 36)) return fail("section_flags");
        for (std::size_t j = 72; j < 80; ++j) if (record[j]) return fail("section_reserved");
        const auto data = bytes.subspan(static_cast<std::size_t>(start), static_cast<std::size_t>(size));
        const auto digest = sha256(data);
        if (std::memcmp(digest.data(), record.data() + 40, 32)) return fail("section_digest");
        end = (static_cast<std::size_t>(start + size) + 63) & ~std::size_t{63};
        if (end > bytes.size()) return fail("section_bounds");
        for (std::size_t j = static_cast<std::size_t>(start + size); j < end; ++j) if (bytes[j]) return fail("section_padding");
        sections.emplace(name, Section{data, stride});
        previous = name;
    }
    if (end != bytes.size()) return fail("section_bounds");
    const std::map<std::string, std::uint32_t> schema{
        {"chans",20},{"config",64},{"dims",8},{"exprs",8},{"features",16},
        {"ffn_terms",12},{"ffns",24},{"head_levels",1},{"heads",88},{"identity",32},
        {"init",520},{"keep",4},{"maps",8},{"meta",1},{"outputs",12},
        {"printable",4},{"terms",8},{"text_lens",4},{"text_rows",16},{"text_types",1}};
    if (sections.size() != schema.size()) return fail("artifact_sections");
    for (const auto& [name, stride] : schema) {
        const auto it = sections.find(name);
        if (it == sections.end()) return fail("artifact_sections");
        if (it->second.stride != stride) return fail("artifact_stride");
    }
    if (sections.at("config").data.size() != 64 || sections.at("identity").data.size() != 192) return fail("artifact_identity_shape");
    if (std::memcmp(sections.at("identity").data.data(), binding.digests.data(), 192)) return fail("artifact_binding");
    if (binding.digests[0] != arithmetic_abi()) return fail("numeric_abi");
    std::uint64_t payload_bytes = 0;
    for (const auto& [name, section] : sections) if (name != "meta") payload_bytes += section.data.size();
    Artifact a;
    std::memcpy(a.config.data(), sections.at("config").data.data(), 64);
    const auto& c = a.config;
    if (c[0] < 24 || c[0] > 16384 || c[1] != 230324 || c[2] < 260 || c[2] > 1048576
        || c[3] < 1 || c[3] > 1024 || c[4] < 1 || c[4] > 16384 || c[5] < 1 || c[5] > 128
        || c[6] != c[3] * 16384 || c[7] >= c[2] || c[8] >= c[2] || c[9] >= c[2]
        || c[7] == c[8] || c[10] != 80 || c[11] != 16 || c[12] != 255 || c[13] != 128
        || c[14] != 3 || c[15] >= c[0]) return fail("circuit_shape");
    MemoryPlan plan;
    // Include conservative temporary ordered-map/set storage as well as input,
    // typed arrays and the derived text index. A caller retaining an older
    // artifact must budget that existing allocation separately.
    const auto validation_records = sections.at("features").data.size() / 16
        + sections.at("ffns").data.size() / 24 + sections.at("heads").data.size() / 88 + c[0] + 256ull;
    plan.host_peak = bytes.size() + payload_bytes + static_cast<std::uint64_t>(c[2]) * 4
                   + validation_records * 128 + 8192;
    if (plan.host_peak > limits.host_bytes) return fail("artifact_host_budget");
    try {
        a.dims = copy<Dimension>(sections.at("dims"));
        a.features = copy<Feature>(sections.at("features"));
        a.expressions = copy<Expression>(sections.at("exprs"));
        a.terms = copy<Term>(sections.at("terms"));
        a.channels = copy<Channel>(sections.at("chans"));
        a.maps = copy<Mapping>(sections.at("maps"));
        a.ffn_terms = copy<FfnTerm>(sections.at("ffn_terms"));
        a.ffns = copy<Ffn>(sections.at("ffns"));
        a.heads = copy<Head>(sections.at("heads"));
        a.head_levels = copy<std::uint8_t>(sections.at("head_levels"));
        a.outputs = copy<Output>(sections.at("outputs"));
        a.initial = copy<InitialRow>(sections.at("init"));
        a.keep = copy<std::int32_t>(sections.at("keep"));
        a.text_rows = copy<std::array<std::uint8_t,16>>(sections.at("text_rows"));
        a.text_lengths = copy<std::uint32_t>(sections.at("text_lens"));
        a.text_types = copy<std::uint8_t>(sections.at("text_types"));
        a.printable = copy<std::int32_t>(sections.at("printable"));
        a.printable_index.assign(c[2], -1);
    } catch (const std::bad_alloc&) { return fail("artifact_allocation"); }
    if (a.dims.size() != c[0] || a.heads.size() != c[3] || a.ffns.size() != c[4]
        || a.head_levels.size() != c[3]) return fail("circuit_counts");
    if (a.text_rows.size() != c[2] || a.text_lengths.size() != c[2] || a.text_types.size() != c[2]) return fail("text_geometry");
    if (a.dims[0].kind != 0 || a.dims[0].level != 0) return fail("dimension_kind_level");
    for (const auto& d : a.dims) if (d.kind < 0 || d.kind > 3 || d.level < 0 || d.level > 128) return fail("dimension_kind_level");
    std::map<std::int32_t, Feature> features;
    for (const auto& f : a.features) {
        if (!index(f.dim, a.dims.size()) || !valid_feature(f) || !features.emplace(f.dim, f).second) return fail("feature_descriptor");
    }
    if (!features.contains(0) || features.at(0).opcode != 1) return fail("feature_one");
    if (a.keep.size() != 23) return fail("retained_dimensions");
    std::set<std::int32_t> retained;
    for (std::size_t i = 0; i < a.keep.size(); ++i) {
        const auto d = a.keep[i];
        if (!index(d, a.dims.size()) || !retained.insert(d).second || !features.contains(d)) return fail("retained_dimensions");
        const auto& f = features.at(d);
        if (i < 21 ? f.opcode != 5 || f.p0 != static_cast<std::int32_t>(i) : f.opcode != (i == 21 ? 8 : 7))
            return fail("retained_dimensions");
    }
    std::size_t expr_end = 0;
    for (const auto& e : a.expressions) {
        if (!range(e.start, e.count, a.terms.size()) || static_cast<std::size_t>(e.start) != expr_end) return fail("expression_range");
        expr_end += static_cast<std::size_t>(e.count);
    }
    if (expr_end != a.terms.size()) return fail("expression_range");
    for (const auto& t : a.terms) {
        if (!index(t.dim, a.dims.size())) return fail("term_dim");
        if (!std::isfinite(t.coefficient)) return fail("coefficient");
    }
    for (const auto& ch : a.channels) {
        if (!index(ch.expression, a.expressions.size())) return fail("channel_expression");
        for (float tap : ch.taps) if (!std::isfinite(tap)) return fail("coefficient");
    }
    for (const auto& t : a.ffn_terms) {
        if (!index(t.a, a.expressions.size()) || !index(t.b, a.expressions.size())) return fail("ffn_expression");
        if (!std::isfinite(t.coefficient)) return fail("coefficient");
    }
    for (const auto& f : a.ffns) {
        if (f.node < 0 || f.node >= 32768 || f.level < 2 || f.level > static_cast<std::int32_t>(c[5]) || f.level % 2
            || !boolean(f.into) || !index(f.out, a.dims.size())) return fail("ffn_node");
        if (!range(f.start, f.count, a.ffn_terms.size())) return fail("ffn_range");
    }
    for (const auto& o : a.outputs) if (o.column < 0 || o.column >= 128 || !index(o.dim, a.dims.size())
                                      || !std::isfinite(o.weight)) return fail("head_output");
    std::set<std::int32_t> head_sublayers;
    auto valid_map = [&](std::int32_t start, std::int32_t n, bool expr) {
        if (!range(start, n, a.maps.size())) return false;
        std::set<std::int32_t> lanes;
        for (std::int32_t i = 0; i < n; ++i) {
            const auto& m = a.maps[static_cast<std::size_t>(start) + i];
            if (m.lane < 0 || m.lane >= 128 || !lanes.insert(m.lane).second
                || !index(m.index, expr ? a.expressions.size() : a.channels.size())) return false;
        }
        return true;
    };
    std::size_t initial_end = 0;
    for (std::size_t hi = 0; hi < a.heads.size(); ++hi) {
        const auto& h = a.heads[hi];
        if (h.node < 0 || h.node >= 32768 || h.level < 1 || h.level > static_cast<std::int32_t>(c[5]) || !(h.level % 2)
            || h.value_index < 0 || h.value_index > 1 || !boolean(h.into) || h.rows != 128 || h.cols != 128
            || h.state_offset != static_cast<std::int32_t>(hi * 16384) || a.head_levels[hi] > 2
            || !std::isfinite(h.tiny) || h.tiny < 0) return fail("head_geometry");
        if (!valid_map(h.qstart, h.qcount, false) || !valid_map(h.kstart, h.kcount, false)
            || !valid_map(h.vstart, h.vcount, false) || !valid_map(h.zstart, h.zcount, true)) return fail("head_map_range");
        if (!index(h.decay, a.expressions.size()) || (h.beta != -1 && !index(h.beta, a.expressions.size()))) return fail("head_expression");
        if (!range(h.ostart, h.ocount, a.outputs.size())) return fail("head_output_range");
        if (!range(h.istart, h.icount, a.initial.size()) || static_cast<std::size_t>(h.istart) != initial_end) return fail("head_init_range");
        int previous_row = -1;
        for (std::int32_t i = 0; i < h.icount; ++i) {
            const auto& row = a.initial[static_cast<std::size_t>(h.istart) + i];
            if (row.head != static_cast<std::int32_t>(hi) || row.row <= previous_row || row.row >= 128) return fail("head_init_row");
            previous_row = row.row;
            for (int col = 0; col < 128; ++col) {
                const auto value = row.values[col];
                if (!std::isfinite(value)) return fail("head_init_levels");
                if (col < 2) {
                    if (value != 0 && value != 16777216.0f) return fail("head_init_levels");
                } else if (a.head_levels[hi] == 0) {
                    if (value != 0 && value != 128) return fail("head_init_levels");
                } else if (a.head_levels[hi] == 1) {
                    if (value < 0 || value > 255 || value != std::floor(value)) return fail("head_init_levels");
                } else if (value != 0) {
                    int exponent = 0;
                    if (value < 0 || std::frexp(value / 128, &exponent) != 0.5f || exponent < -125 || exponent > 1)
                        return fail("head_init_levels");
                }
            }
            if (row.values[0] != row.values[1]) return fail("head_init_ballast");
        }
        initial_end += static_cast<std::size_t>(h.icount);
        head_sublayers.insert(h.level);
    }
    if (initial_end != a.initial.size()) return fail("head_init_range");
    std::set<std::int32_t> nodes;
    int previous_ffn = -1;
    for (const auto& f : a.ffns) {
        if (f.node <= previous_ffn || !nodes.insert(f.node).second) return fail("node_identity");
        previous_ffn = f.node;
    }
    int previous_head = -1;
    for (std::size_t hi = 0; hi < a.heads.size(); ++hi) {
        const auto& h = a.heads[hi];
        if (h.value_index == 0) {
            if (h.node <= previous_head || !nodes.insert(h.node).second) return fail("node_identity");
            previous_head = h.node;
        } else {
            if (hi == 0) return fail("node_identity");
            const auto& first = a.heads[hi - 1];
            if (first.value_index != 0 || h.node != first.node || h.level != first.level
                || h.qstart != first.qstart || h.qcount != first.qcount
                || h.kstart != first.kstart || h.kcount != first.kcount
                || h.tiny != first.tiny || a.head_levels[hi] != a.head_levels[hi - 1])
                return fail("node_identity");
        }
    }
    if (nodes.empty() || nodes.size() > 16384 || *nodes.begin() != 0
        || *nodes.rbegin() != static_cast<std::int32_t>(nodes.size() - 1)) return fail("node_identity");
    if (a.printable.size() != 260 || a.text_types[c[7]] != 4 || a.text_types[c[8]] != 4) return fail("printable_mapping");
    for (std::size_t q = 0; q < a.text_rows.size(); ++q) {
        const auto n = a.text_lengths[q];
        if (n < 1 || n > 67108864 || a.text_types[q] < 1 || a.text_types[q] > 6) return fail("text_geometry");
        for (std::size_t j = std::min<std::size_t>(n,16); j < 16; ++j) if (a.text_rows[q][j]) return fail("text_padding");
    }
    for (std::size_t p = 0; p < a.printable.size(); ++p) {
        const auto q = a.printable[p];
        if (!index(q, a.text_rows.size()) || a.printable_index[q] != -1) return fail("printable_mapping");
        if (p < 256 && (a.text_lengths[q] != 1 || a.text_rows[q][0] != p || a.text_types[q] != 1)) return fail("printable_mapping");
        if (p >= 256 && a.text_types[q] != 4) return fail("printable_mapping");
        a.printable_index[q] = static_cast<std::int32_t>(p);
    }
    if (a.printable[10] != static_cast<std::int32_t>(c[9])) return fail("printable_mapping");
    const auto meta = sections.at("meta").data;
    if (meta.empty() || meta.size() > 4 * 1024 * 1024) return fail("metadata_size");
    for (auto byte : meta) if (byte >= 127 || (byte < 32 && byte != 9 && byte != 10 && byte != 13)) return fail("metadata_ascii");
    plan.weights = payload_bytes + static_cast<std::uint64_t>(c[2]) * 4;
    plan.state_each = static_cast<std::uint64_t>(c[6]) * 4;
    plan.lag_each = head_sublayers.size() * 3ull * c[0] * 4;
    plan.scratch = static_cast<std::uint64_t>(c[0]) * 8 + a.expressions.size() * 4ull
                 + a.channels.size() * 16ull + a.heads.size() * 128ull * 16;
    plan.fifo = 4096 * 4 + 128 * 16 + 256;
    plan.device_total = plan.weights + 3 * plan.state_each + 3 * plan.lag_each + plan.scratch + plan.fifo;
    if (plan.device_total > limits.device_bytes) return fail("artifact_device_budget");
    a.binding = binding;
    a.digest = sha256(bytes);
    output = std::move(a);
    memory = plan;
    return true;
} catch (const std::bad_alloc&) {
    error = "artifact_allocation";
    return false;
}
bool load_artifact(const std::filesystem::path& path, const Binding& binding, const LoadLimits& limits,
                   Artifact& output, MemoryPlan& memory, std::string& error) {
    std::error_code ec;
    const auto size = std::filesystem::file_size(path, ec);
    if (ec) { error = "artifact_io"; return false; }
    if (size < 64 || size > std::min<std::uint64_t>(limits.file_bytes, 256ull << 20)) { error = "artifact_size"; return false; }
    try {
        std::vector<std::uint8_t> bytes(static_cast<std::size_t>(size));
        std::ifstream file(path, std::ios::binary);
        if (!file.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()))
            || file.peek() != std::char_traits<char>::eof()) { error = "artifact_io"; return false; }
        return parse_artifact(bytes, binding, limits, output, memory, error);
    } catch (const std::bad_alloc&) { error = "artifact_allocation"; return false; }
}
bool feature_value(const Artifact& a, const Feature& f, TokenKind kind, std::uint32_t id,
                   float& output) {
    if (!valid_feature(f) || (kind != TokenKind::Trace && kind != TokenKind::Text)
        || id >= (kind == TokenKind::Trace ? a.config[1] : a.config[2])) return false;
    const bool text = kind == TokenKind::Text;
    float value = 0;
    std::uint32_t family = 0, payload = id;
    // Family decoding is token embedding logic, never instruction emulation.
    if (!text && (f.opcode == 2 || f.opcode == 3 || f.opcode == 4 || f.opcode == 5)) {
        while (payload >= family_sizes[family]) { payload -= family_sizes[family]; ++family; }
    }
    switch (f.opcode) {
    case 0: break;
    case 1: value = 1; break;
    case 2: value = !text && family == static_cast<std::uint32_t>(f.p0) ? 1.0f : 0.0f; break;
    case 3: if (!text && family == static_cast<std::uint32_t>(f.p0)) value = static_cast<float>((payload >> f.p1) & 1); break;
    case 4:
        if (!text) value = ((family == 9 && payload == static_cast<std::uint32_t>(f.p0))
                 || (family == 10 && static_cast<std::int32_t>(payload >> 4) == f.p0 - 1)) ? 1.0f : 0.0f;
        break;
    case 5: {
        if (text) {
            const auto p = a.printable_index[id];
            if (p < 0) break;
            family = 20; payload = static_cast<std::uint32_t>(p);
        }
        const auto code = family | (payload << 5);
        value = ((code >> f.p0) & 1) ? 128.0f : -128.0f;
        break;
    }
    case 6: value = text ? 1.0f : 0.0f; break;
    case 7: value = text ? 0.0f : 1.0f; break;
    case 8: value = text ? 0x1p-40f : -1.0f; break;
    case 9: value = text && id == a.config[7] ? 1.0f : 0.0f; break;
    case 10: value = text && id == a.config[8] ? 1.0f : 0.0f; break;
    case 11: if (text) value = static_cast<float>((a.text_rows[id][f.p0] >> f.p1) & 1); break;
    default: return false;
    }
    output = value;
    return true;
}
bool feature_digest(const Artifact& a, Digest& output, std::string& error) {
    Hash hash;
    std::array<float, 8192> values{};
    for (const auto& f : a.features) {
        for (std::uint32_t kind = 0; kind < 2; ++kind) {
            const auto count = a.config[kind == 0 ? 1 : 2];
            for (std::uint32_t base = 0; base < count;) {
                const auto n = std::min<std::uint32_t>(static_cast<std::uint32_t>(values.size()), count - base);
                for (std::uint32_t i = 0; i < n; ++i) {
                    if (!feature_value(a, f, static_cast<TokenKind>(kind), base + i, values[i])) {
                        error = "feature_domain"; return false;
                    }
                }
                hash.update({reinterpret_cast<const std::uint8_t*>(values.data()), n * sizeof(float)});
                base += n;
            }
        }
    }
    output = hash.finish();
    error.clear();
    return true;
}
} // namespace strata::qwenputer
