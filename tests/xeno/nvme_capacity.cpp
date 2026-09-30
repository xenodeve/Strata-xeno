// #11 N1: the bounded host tier. With a capacity, only the first experts of the given order are committed and read;
// the rest stay on NVMe until the CPU pool asks for one (materialize), which reads it from the pack and evicts the
// lowest-scored host expert of ANOTHER layer (the current layer's jobs may still point at its own experts).
// `read_into` serves the prompt path without admitting anything.
#include "strata/core/expert_source.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

int main() {
    namespace fs = std::filesystem;
    const fs::path dir = fs::temp_directory_path() /
        ("strata-nvme-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    fs::create_directory(dir);
    const uint64_t blob = strata::kernels::cpu::BLOB;
    const int layers = 2, experts = 3;
    auto mark = [](int l, int e) { return (uint8_t) (0x11 * (l * 3 + e) + 1); };
    {
        std::ofstream out(dir / "experts.bin", std::ios::binary);
        std::vector<uint8_t> b(blob);
        for (int l = 0; l < layers; ++l)
            for (int e = 0; e < experts; ++e) {
                for (uint64_t i = 0; i < blob; ++i) b[i] = (uint8_t) (mark(l, e) + i);
                out.write((const char*) b.data(), (std::streamsize) blob);
            }
    }
    std::string err;
    if (!strata::kernels::cpu::expert_layout_load(dir.string(), layers, experts, err)) {
        std::fprintf(stderr, "layout: %s\n", err.c_str()); return 1;
    }
    strata::core::ArenaExpertSource s;
    if (!s.open(dir.string(), layers, experts, 1, err, /*pin_for_cuda=*/false, /*defer_load=*/true)) {
        std::fprintf(stderr, "open: %s\n", err.c_str()); return 1;
    }
    // room for two experts; order: (0,0), (1,0), then the rest
    const std::vector<int32_t> order = {0, 3, 1, 2, 4, 5};
    s.set_capacity(2 * blob, order);
    if (!s.load_rest(1, err)) { std::fprintf(stderr, "load_rest: %s\n", err.c_str()); return 1; }
    int bad = 0;
    auto expect = [&](bool ok, const char* what) { if (!ok) { std::fprintf(stderr, "FAIL: %s\n", what); ++bad; } };
    expect(s.blob(0, 0) && s.blob(0, 0)[0] == mark(0, 0), "first in order is resident");
    expect(s.blob(1, 0) && s.blob(1, 0)[5] == (uint8_t) (mark(1, 0) + 5), "second in order is resident");
    expect(s.blob(0, 1) == nullptr && !s.resident(0, 1), "third is on NVMe");
    for (int i = 0; i < 5; ++i) (void) s.blob(0, 0);   // (0,0) is hot
    // a miss in layer 0 may not evict layer 0: the victim is (1,0)
    const uint8_t* m = s.materialize(0, 1, /*avoid_layer=*/0, err);
    expect(m && m[0] == mark(0, 1) && m[blob - 1] == (uint8_t) (mark(0, 1) + blob - 1), "materialize reads the pack");
    expect(s.resident(0, 0) && !s.resident(1, 0), "the victim is the other layer's expert");
    // a miss in layer 1 with only layer-0 experts resident and hot: over capacity rather than evicting in use
    const uint8_t* m2 = s.materialize(1, 2, /*avoid_layer=*/1, err);
    expect(m2 && m2[0] == mark(1, 2), "second materialize");
    expect(s.host_cache_bytes() <= 2 * blob, "capacity holds: a layer-0 expert was evicted for the layer-1 miss");
    // no other layer to evict from: the tier may run over and trim later
    (void) s.materialize(0, 2, /*avoid_layer=*/0, err);
    s.trim(/*avoid_layer=*/-1);
    expect(s.host_cache_bytes() <= 2 * blob, "trim restores the capacity");
    // the prompt path reads without admitting
    std::vector<uint8_t> buf(blob);
    const bool was = s.resident(1, 1);
    expect(s.read_into(1, 1, buf.data(), err) && buf[0] == mark(1, 1), "read_into reads the pack");
    expect(s.resident(1, 1) == was, "read_into admits nothing");
    // a failed read is an error, never stale bytes
    s.close();
    std::error_code ec;
    fs::remove_all(dir, ec);   // read_into keeps a per-thread reader open; the temp dir may outlive the test
    if (bad) { std::fprintf(stderr, "nvme_capacity: %d failures\n", bad); return 1; }
    std::printf("nvme_capacity: bounded host tier, eviction by layer and score, read_into without admission\n");
    return 0;
}
