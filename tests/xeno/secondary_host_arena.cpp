#include "strata/core/pinned.hpp"

#include <cstdio>
#include <vector>

int main() {
    constexpr uint64_t mib = 1ull << 20;
    const std::vector<uint64_t> layers{0, mib / 2, mib};
    strata::core::PinnedArena arena(mib, layers, /*pin_for_cuda=*/false);
    if (!arena.valid() || arena.registered_bytes != 0 || arena.locked_bytes != 0 ||
        arena.note.find("pageable host arena") == std::string::npos) {
        std::fprintf(stderr, "secondary host arena was CUDA-registered or locked\n");
        return 1;
    }
    std::puts("secondary host arena: pageable, no CUDA registration or OS lock");
    return 0;
}
