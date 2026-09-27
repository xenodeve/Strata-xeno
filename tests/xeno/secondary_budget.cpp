#include "strata/core/secondary_budget.hpp"
#include <cstdint>
#include <cstdio>
#include <limits>
#include <vector>

#define CHECK(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "failed: %s\n", #condition); return 1; } \
} while (false)

int main() {
    using strata::core::secondary_budget;
    using strata::core::secondary_effective_free;
    constexpr uint64_t mib = 1ull << 20;
    constexpr uint64_t gib = 1ull << 30;
    constexpr uint64_t reserve = 2560ull * mib;

    CHECK(secondary_budget(reserve, {mib}).slots == 0);
    CHECK(secondary_budget(reserve - 1, {mib}).slots == 0);
    CHECK(secondary_effective_free(11070 * mib, 9812 * mib) == 9812 * mib);
    CHECK(secondary_effective_free(9000 * mib, 9812 * mib) == 9000 * mib);
    CHECK(secondary_effective_free(9000 * mib, 0) == 0);
    const auto exact = secondary_budget(4 * gib, {gib, 512 * mib, mib});
    CHECK(exact.slots == 2 && exact.bytes == gib + 512 * mib);

    const auto aligned = secondary_budget(reserve + 768, {257, 257, 257});
    CHECK(aligned.slots == 1 && aligned.bytes == 512);
    const auto ranked = secondary_budget(reserve + 1024, {768, 2048, 256});
    CHECK(ranked.slots == 1 && ranked.bytes == 768); // no skipping a higher-ranked pair
    const auto overflow = secondary_budget(std::numeric_limits<uint64_t>::max(),
                                           {std::numeric_limits<uint64_t>::max(), 256});
    CHECK(overflow.slots == 0 && overflow.bytes == 0);

    std::puts("secondary budget: reserve, alignment, rank order and overflow passed");
    return 0;
}
