// #62 crash (2026-10-01): a served engine read decommitted memory in parallel_copy (0xC0000005, "read of address").
// A paired swap's stage 2 gathers the newcomers' host pointers, materializing the ones on NVMe one by one, then copies
// them all.  materialize evicts the lowest-scored host expert of any layer (avoid_layer -1), which can be a newcomer
// gathered a moment earlier: its pointer then names decommitted pages.  hold() keeps an expert resident until
// release_hold(); the eviction skips it (the tier may run over and trim later, as it already may for avoid_layer).
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
        ("strata-hold-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
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
    s.set_capacity(2 * blob, {0, 3, 1, 2, 4, 5});   // (0,0) and (1,0) in RAM
    if (!s.load_rest(1, err)) { std::fprintf(stderr, "load_rest: %s\n", err.c_str()); return 1; }
    int bad = 0;
    auto expect = [&](bool ok, const char* what) { if (!ok) { std::fprintf(stderr, "FAIL: %s\n", what); ++bad; } };
    for (int i = 0; i < 5; ++i) (void) s.blob(0, 0);   // (0,0) is hot, (1,0) is the coldest host expert

    // stage 2's shape: a newcomer already gathered (cold), then two more materialized with avoid_layer -1
    const uint8_t* gathered = s.blob(1, 0);
    s.hold(1, 0);
    (void) s.materialize(0, 1, /*avoid_layer=*/-1, err);
    (void) s.materialize(0, 2, /*avoid_layer=*/-1, err);
    const bool kept = s.resident(1, 0);
    expect(kept, "a held expert is not evicted by a later materialize");
    if (kept)   // only read it when resident: an evicted one is decommitted and would crash the test itself
        expect(gathered[0] == mark(1, 0) && gathered[blob - 1] == (uint8_t) (mark(1, 0) + blob - 1),
               "the gathered pointer still holds its expert");
    s.release_hold(1, 0);
    s.trim(/*avoid_layer=*/-1);
    expect(s.host_cache_bytes() <= 2 * blob, "released, the tier trims back to its capacity");
    s.close();
    std::error_code ec;
    fs::remove_all(dir, ec);
    if (bad) { std::fprintf(stderr, "nvme_hold: %d failures\n", bad); return 1; }
    std::printf("nvme_hold: a held expert survives materialize's eviction, then trims\n");
    return 0;
}
