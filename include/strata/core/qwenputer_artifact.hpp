#pragma once

#include <array>
#include <cstdint>
#include <filesystem>
#include <iosfwd>
#include <span>
#include <string>
#include <vector>

namespace strata::qwenputer {

using Digest = std::array<std::uint8_t, 32>;

struct Binding {
    // ABI, source, firmware, boot, model, tokenizer, in container order.
    std::array<Digest, 6> digests{};
};

struct LoadLimits {
    std::uint64_t file_bytes = 256ull << 20;
    std::uint64_t host_bytes = 512ull << 20;
    std::uint64_t device_bytes = 256ull << 20;
};

struct MemoryPlan {
    std::uint64_t weights = 0;
    std::uint64_t state_each = 0;
    std::uint64_t lag_each = 0;
    std::uint64_t scratch = 0;
    std::uint64_t fifo = 0;
    std::uint64_t device_total = 0;
    std::uint64_t host_peak = 0;
    // Buffer estimates only until P4 instantiates/measures the actual graph.
    std::uint64_t graph_bytes = 0;
    bool graph_measured = false;
};

struct Dimension { std::int32_t kind, level; };
struct Feature { std::int32_t dim, opcode, p0, p1; };
struct Expression { std::int32_t start, count; };
struct Term { std::int32_t dim; float coefficient; };
struct Channel { std::int32_t expression; float taps[4]; };
struct Mapping { std::int32_t lane, index; };
struct FfnTerm { float coefficient; std::int32_t a, b; };
struct Ffn { std::int32_t node, level, out, into, start, count; };
struct Head {
    std::int32_t node, level, value_index, into, rows, cols;
    std::int32_t qstart, qcount, kstart, kcount, vstart, vcount, zstart, zcount;
    std::int32_t ostart, ocount, decay, beta, istart, icount, state_offset;
    float tiny;
};
struct Output { std::int32_t column, dim; float weight; };
struct InitialRow { std::int32_t head, row; float values[128]; };

struct Artifact {
    std::array<std::uint32_t, 16> config{};
    Binding binding;
    Digest digest{};
    std::vector<Dimension> dims;
    std::vector<Feature> features;
    std::vector<Expression> expressions;
    std::vector<Term> terms;
    std::vector<Channel> channels;
    std::vector<Mapping> maps;
    std::vector<FfnTerm> ffn_terms;
    std::vector<Ffn> ffns;
    std::vector<Head> heads;
    std::vector<std::uint8_t> head_levels;
    std::vector<Output> outputs;
    std::vector<InitialRow> initial;
    std::vector<std::int32_t> keep;
    std::vector<std::array<std::uint8_t, 16>> text_rows;
    std::vector<std::uint32_t> text_lengths;
    // v1.1 CPU-only lexicon. These arrays are never uploaded by the executor.
    std::vector<std::uint32_t> text_offsets;
    std::vector<std::uint8_t> text_bytes;
    std::vector<std::uint8_t> text_types;
    std::vector<std::int32_t> printable;
    std::vector<std::int32_t> printable_index; // derived, never trusted from the file
    bool has_full_text() const { return text_offsets.size() == text_lengths.size() + 1; }
    std::span<const std::uint8_t> token_bytes(std::uint32_t id) const {
        if (!has_full_text() || id >= text_lengths.size()) return {};
        return std::span(text_bytes).subspan(text_offsets[id], text_lengths[id]);
    }
};

Digest sha256(std::span<const std::uint8_t> bytes);
bool sha256_stream(std::istream&, std::uint64_t max_bytes, Digest& output,
                   std::uint64_t& bytes, std::string& error);
// This loader implements the arithmetic ABI only; a matching caller binding
// cannot authorize an otherwise unsupported numerical contract.
const Digest& arithmetic_abi();
bool parse_artifact(std::span<const std::uint8_t> bytes, const Binding& binding,
                    const LoadLimits& limits, Artifact& output, MemoryPlan& memory,
                    std::string& error);
bool load_artifact(const std::filesystem::path& path, const Binding& binding,
                   const LoadLimits& limits, Artifact& output, MemoryPlan& memory,
                   std::string& error);
// Token kinds are tagged at the interface, never inferred from an unchecked ID.
enum class TokenKind : std::uint32_t { Trace = 0, Text = 1 };
bool feature_value(const Artifact&, const Feature&, TokenKind, std::uint32_t id,
                   float& output);
bool feature_digest(const Artifact&, Digest& output, std::string& error);

} // namespace strata::qwenputer
