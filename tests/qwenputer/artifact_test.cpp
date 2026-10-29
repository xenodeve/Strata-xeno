#include "strata/core/qwenputer_artifact.hpp"

#include <fstream>
#include <iostream>
#include <iterator>
#include <string_view>
#include <sstream>

using namespace strata::qwenputer;

static int failures = 0;
static void check(bool condition, const char* name) {
    if (!condition) { std::cerr << "FAIL " << name << '\n'; ++failures; }
}
static std::string hex(const Digest& digest) {
    constexpr char alphabet[] = "0123456789abcdef";
    std::string result;
    for (auto byte : digest) { result += alphabet[byte >> 4]; result += alphabet[byte & 15]; }
    return result;
}
static std::vector<std::uint8_t> read(const char* path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) { std::cerr << "cannot read fixture\n"; return {}; }
    return {std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
}

int main(int argc, char** argv) {
    if (argc == 5 && std::string_view(argv[1]) == "--reject") {
        const auto bytes = read(argv[2]), identity = read(argv[3]);
        if (identity.size() != 192) return 2;
        Binding binding;
        for (int i = 0; i < 192; ++i) binding.digests[i / 32][i % 32] = identity[i];
        Artifact output;
        output.config[0] = 77;
        MemoryPlan memory;
        memory.weights = 123;
        std::string error;
        const auto accepted = parse_artifact(bytes, binding, {}, output, memory, error);
        check(!accepted, "semantic mutant refused");
        check(error == argv[4], "named refusal matches");
        check(output.config[0] == 77 && memory.weights == 123, "mutant leaves state untouched");
        std::cout << (failures ? "FAIL " : "PASS ") << "error=" << error << '\n';
        return failures ? 1 : 0;
    }
    check(hex(sha256({})) == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855", "SHA256 empty");
    std::string_view abc = "abc";
    check(hex(sha256({reinterpret_cast<const std::uint8_t*>(abc.data()), abc.size()})) ==
          "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad", "SHA256 abc");
    std::vector<std::uint8_t> million(1000000, 'a');
    check(hex(sha256(million)) == "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0", "SHA256 million a");
    {
        Digest result{}; result[0]=77; std::uint64_t bytes=99; std::string reason;
        std::istringstream input("abc");
        check(sha256_stream(input,3,result,bytes,reason) && bytes==3
              && hex(result)=="ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad", "stream SHA256 abc");
        const auto prior=result;
        std::istringstream oversized("abc"); bytes=99;
        check(!sha256_stream(oversized,2,result,bytes,reason) && reason=="binding_file_size"
              && result==prior && bytes==99,"stream cap preserves outputs");
    }
    Artifact output;
    output.config[0] = 77;
    MemoryPlan memory;
    memory.weights = 123;
    Binding binding;
    std::string error;
    check(!parse_artifact({}, binding, {}, output, memory, error) && error == "artifact_size", "empty rejected");
    check(output.config[0] == 77 && memory.weights == 123, "failure leaves outputs unchanged");
    if (argc == 3) {
        const auto bytes = read(argv[1]);
        const auto identity = read(argv[2]);
        check(identity.size() == 192, "binding fixture size");
        if (identity.size() == 192) {
            for (int i = 0; i < 192; ++i) binding.digests[i / 32][i % 32] = identity[i];
            check(parse_artifact(bytes, binding, {}, output, memory, error), "real artifact accepted");
            if (!error.empty()) std::cerr << "loader: " << error << '\n';
            check(output.config[0] == 2918 && output.config[2] == 248320, "real geometry");
            check(output.heads.size() == 196 && output.ffn_terms.size() == 42114, "real circuit counts");
            check(memory.state_each == 12845056, "derived state memory");
            check(memory.device_total >= 3 * memory.state_each, "all state copies budgeted");
            check(!memory.graph_measured && memory.graph_bytes == 0, "driver graph resources not invented");
            const bool full = bytes.size() >= 12 && bytes[10] == 1;
            check(output.has_full_text() == full, "version-specific CPU lexicon presence");
            check(output.token_bytes(output.config[2]).empty(), "out-of-range token bytes refused");
            if (full) {
                std::size_t long_tokens = 0;
                for (std::uint32_t q = 0; q < output.config[2]; ++q) {
                    const auto raw = output.token_bytes(q);
                    check(raw.size() == output.text_lengths[q], "full token byte count");
                    if (raw.size() > 16) ++long_tokens;
                }
                check(long_tokens == 6882, "all actual long Gyro tokens retained");
                check(output.token_bytes(63434).size() == 32, "32-byte tilde fence token retained");
            }
            const auto valid_memory = memory;
            const auto valid_digest = output.digest;
            auto bad = binding;
            bad.digests[5][0] ^= 1;
            check(!parse_artifact(bytes, bad, {}, output, memory, error) && error == "artifact_binding", "tokenizer mismatch rejected");
            LoadLimits small;
            small.file_bytes = 64;
            check(!parse_artifact(bytes, binding, small, output, memory, error) && error == "artifact_size", "file cap enforced");
            small = {};
            small.host_bytes = valid_memory.host_peak - 1;
            check(!parse_artifact(bytes, binding, small, output, memory, error)
                  && error == "artifact_host_budget", "host expansion cap enforced");
            small = {};
            small.device_bytes = valid_memory.device_total - 1;
            check(!parse_artifact(bytes, binding, small, output, memory, error)
                  && error == "artifact_device_budget", "all device buffers capped before upload");
            check(output.digest == valid_digest && memory.device_total == valid_memory.device_total,
                  "budget failure leaves previous loaded artifact unchanged");
            auto corrupt = bytes;
            if (corrupt.size() > 64) {
                corrupt.back() ^= 1;
                check(!parse_artifact(corrupt, binding, {}, output, memory, error) && error == "artifact_digest", "payload mutation rejected");
            }
            // Deterministic truncation/bit-flip controls exercise the untrusted
            // container before semantic decoding. Resigned records are tested
            // separately through --reject.
            std::uint32_t seed = 20261028;
            for (int i = 0; i < 32; ++i) {
                seed = seed * 1664525u + 1013904223u;
                const auto n = static_cast<std::size_t>(seed) % bytes.size();
                check(!parse_artifact(std::span(bytes).first(n), binding, {}, output, memory, error),
                      "seeded truncation refused");
                corrupt = bytes;
                corrupt[n] ^= 1;
                check(!parse_artifact(corrupt, binding, {}, output, memory, error), "seeded bit flip refused");
            }
            std::cout << "memory weights=" << valid_memory.weights << " state_each=" << valid_memory.state_each
                      << " lag_each=" << valid_memory.lag_each << " scratch=" << valid_memory.scratch
                      << " fifo=" << valid_memory.fifo << " device_buffers=" << valid_memory.device_total
                      << " host_peak_estimate=" << valid_memory.host_peak << '\n';
        }
    } else {
        check(false, "provide real artifact and independent identity fixtures");
    }
    std::cout << (failures ? "FAIL " : "PASS ") << failures << " failures\n";
    return failures ? 1 : 0;
}
