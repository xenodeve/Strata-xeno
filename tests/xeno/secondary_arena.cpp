#include "strata/core/secondary_arena.hpp"
#include "strata/core/secondary_budget.hpp"

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

struct InjectedFree {
    uint64_t values[4];
    int next = 0;
};

static bool injected_free(int, uint64_t& lower, std::string& err, void* context) {
    auto& readings = *static_cast<InjectedFree*>(context);
    if (readings.next >= 4) { err = "injected readings exhausted"; return false; }
    lower = readings.values[readings.next++];
    return true;
}

#define CHECK(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "failed: %s\n", #condition); return 1; } \
} while (false)

int main() {
    constexpr uint64_t mib = 1ull << 20;
    std::vector<uint64_t> ranked(64, mib);
    std::string err;
    strata::core::SecondaryArena arena;
    if (!arena.open(1, ranked, 64 * mib, err)) {
        std::fprintf(stderr, "secondary arena: %s\n", err.c_str());
        return 1;
    }
    CHECK(arena.slots() == 64);
    CHECK(arena.bytes() == 64 * mib);
    CHECK(arena.lower_free_after() >= strata::core::kSecondaryReserveBytes);
    CHECK(arena.slot_ptr(0) != nullptr);
    CHECK(arena.slot_ptr(63) == arena.slot_ptr(0) + 63 * mib);
    CHECK(cudaSetDevice(1) == cudaSuccess);
    uint8_t last = 255;
    CHECK(cudaMemcpy(&last, arena.slot_ptr(63) + mib - 1, 1, cudaMemcpyDeviceToHost) == cudaSuccess);
    CHECK(last == 0); // the last page was touched, not just virtually reserved
    std::vector<uint8_t> blob((size_t) mib, 0x5a);
    CHECK(!arena.fill_slot(0, blob.data(), mib + 1, err));
    CHECK(arena.fill_slot(0, blob.data(), mib, err));
    CHECK(arena.verify_slot(0, blob.data(), mib, err));
    blob[0] ^= 1;
    CHECK(!arena.verify_slot(0, blob.data(), mib, err));
    CHECK(arena.close());
    // The free floor can be lower when existing desktop processes already
    // consume part of the shared 2.5 GiB headroom.
    constexpr uint64_t shared_floor = 512 * mib;
    InjectedFree shared{{shared_floor + 128 * mib, shared_floor + 64 * mib}};
    CHECK(arena.open(1, ranked, 64 * mib, err, injected_free, &shared, shared_floor));
    CHECK(arena.slots() == 64 && arena.lower_free_after() == shared_floor + 64 * mib);
    CHECK(arena.close());
    CHECK(arena.slots() == 0 && arena.bytes() == 0);
    InjectedFree readings{{strata::core::kSecondaryReserveBytes + 256 * mib,
                           strata::core::kSecondaryReserveBytes - 1,
                           strata::core::kSecondaryReserveBytes + 256 * mib,
                           strata::core::kSecondaryReserveBytes + 100 * mib}};
    std::vector<uint64_t> retry_ranked(128, mib);
    CHECK(arena.open(1, retry_ranked, 128 * mib, err, injected_free, &readings));
    CHECK(readings.next == 4);
    CHECK(arena.slots() == 63 && arena.bytes() == 63 * mib);
    CHECK(arena.close());
    std::puts("secondary arena: 64 MiB touched, reserve held, allocation released");
    return 0;
}
