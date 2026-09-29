#include "strata/core/expert_source.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>

int main() {
    namespace fs = std::filesystem;
    const fs::path dir = fs::temp_directory_path() /
        ("strata-exclusive-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    fs::create_directory(dir);
    const fs::path file = dir / "experts.bin";
    const uint64_t blob = strata::kernels::cpu::BLOB;
    {
        std::ofstream out(file, std::ios::binary);
        out.seekp((std::streamoff) (2 * blob - 1));
        out.put('\0');
        out.seekp((std::streamoff) blob);
        out.put((char) 0x5a);
    }
    std::string err;
    if (!strata::kernels::cpu::expert_layout_load(dir.string(), 1, 2, err)) {
        std::fprintf(stderr, "layout: %s\n", err.c_str()); return 1;
    }
    strata::core::ArenaExpertSource source;
    if (!source.open(dir.string(), 1, 2, 1, err, /*pin_for_cuda=*/false)) {
        std::fprintf(stderr, "source: %s\n", err.c_str()); return 1;
    }
    if (source.blob(0, 0) == nullptr || source.blob(0, 1)[0] != 0x5a) return 1;
    if (!source.release_host_copy(0, 0, err)) {
        std::fprintf(stderr, "release: %s\n", err.c_str()); return 1;
    }
    if (source.blob(0, 0) != nullptr || source.blob(0, 1)[0] != 0x5a ||
        source.released_host_bytes() == 0 || source.released_host_bytes() >= blob ||
        source.release_host_copy(0, 0, err)) {
        std::fprintf(stderr, "exclusive owner/decommit contract failed\n"); return 1;
    }
    const uint64_t freed = source.released_host_bytes();
    // Phase 4 paired swap: re-commit the GPU-owned expert, copy its bytes home, publish, and release it again.
    std::string refused;
    if (source.recommit_host_copy(0, 1, refused) != nullptr) {   // a CPU-owned expert is not re-committed
        std::fprintf(stderr, "re-commit accepted a host-owned expert\n"); return 1;
    }
    uint8_t* home = source.recommit_host_copy(0, 0, err);
    if (home == nullptr || source.blob(0, 0) != nullptr || source.released_host_bytes() != 0) {
        std::fprintf(stderr, "re-commit: %s\n", err.c_str()); return 1;
    }
    for (uint64_t i = 0; i < blob; ++i) home[i] = (uint8_t) (i * 7u);   // the copy-home, as the D2H would write it
    source.publish_host_copy(0, 0);
    const uint8_t* back = source.blob(0, 0);
    if (back == nullptr || back[0] != 0 || back[blob / 2] != (uint8_t) ((blob / 2) * 7u) ||
        back[blob - 1] != (uint8_t) ((blob - 1) * 7u) || source.blob(0, 1)[0] != 0x5a) {
        std::fprintf(stderr, "paired swap round trip lost bytes\n"); return 1;
    }
    if (!source.release_host_copy(0, 0, err) || source.released_host_bytes() != freed) {
        std::fprintf(stderr, "second release: %s\n", err.c_str()); return 1;
    }
    source.close();
    strata::core::ArenaExpertSource pinned_source;
    if (!pinned_source.open(dir.string(), 1, 2, 1, err, /*pin_for_cuda=*/true) ||
        !pinned_source.pinned(0, 0) || pinned_source.release_host_copy(0, 0, err) ||
        pinned_source.blob(0, 0) == nullptr) {
        std::fprintf(stderr, "pinned backing was released or was unavailable: %s\n", err.c_str());
        return 1;
    }
    pinned_source.close();
    fs::remove(file);
    fs::remove(dir);
    std::printf("exclusive host page test: released %llu B; neighbor readable; duplicate refused\n",
                (unsigned long long) freed);
    return 0;
}
