#pragma once

#include <cstdint>
#include <limits>
#include <vector>

namespace strata::core {

// The display GPU's minimum free memory. 2560 MiB is more conservative than decimal 2.5 GB.
inline constexpr uint64_t kSecondaryReserveBytes = 2560ull << 20;

// CUDA's context estimate and NVML's physical framebuffer figure can differ on WDDM.
// A failed query is represented by zero and must disable placement.
inline uint64_t secondary_effective_free(uint64_t cuda_free, uint64_t nvml_free) {
    return cuda_free < nvml_free ? cuda_free : nvml_free;
}

struct SecondaryBudget {
    uint64_t bytes = 0;
    uint64_t slots = 0;
};

// Profile order is significant: stop at the first pair that does not fit.
// Every expert slot is 256-byte aligned for the grouped kernel's vector loads.
inline SecondaryBudget secondary_budget(uint64_t free_bytes, const std::vector<uint64_t>& ranked_blob_bytes) {
    SecondaryBudget result;
    if (free_bytes <= kSecondaryReserveBytes) return result;
    const uint64_t usable = free_bytes - kSecondaryReserveBytes;
    for (const uint64_t raw : ranked_blob_bytes) {
        if (raw == 0 || raw > std::numeric_limits<uint64_t>::max() - 255) break;
        const uint64_t aligned = (raw + 255) & ~uint64_t{255};
        if (aligned > usable - result.bytes) break;
        result.bytes += aligned;
        ++result.slots;
    }
    return result;
}

} // namespace strata::core
