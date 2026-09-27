#include "strata/core/pinned.hpp"
#include <cuda_runtime.h>

#include <cstdio>
#include <vector>

int main() {
    constexpr uint64_t mib = 1ull << 20;
    const std::vector<uint64_t> layers{0, mib / 2, mib};
    (void) cudaGetLastError();
    {
        strata::core::PinnedArena arena(mib, layers, /*pin_for_cuda=*/false);
        if (!arena.valid() || arena.registered_bytes != 0 || arena.locked_bytes != 0 ||
            arena.backing == strata::core::PageBacking::LargePages ||
            arena.note.find("large pages skipped") == std::string::npos) {
            std::fprintf(stderr, "secondary host arena was CUDA-registered or OS-locked\n");
            return 1;
        }
    }
    if (cudaPeekAtLastError() != cudaSuccess) {
        std::fprintf(stderr, "secondary host arena teardown left a CUDA error\n");
        return 1;
    }
    std::puts("secondary host arena: pageable, no CUDA registration or OS lock");
    return 0;
}
