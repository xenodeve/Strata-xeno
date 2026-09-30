#include "strata/core/secondary_profile.hpp"

#include <cstdint>
#include <cstdio>
#include <string>
#include <utility>
#include <vector>

#define CHECK(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "failed: %s\n", #condition); return 1; } \
} while (false)

int main() {
    std::vector<std::pair<int32_t, int32_t>> ranked{{0, 7}, {1, 13}, {0, 8}};
    std::vector<int32_t> primary(2 * 16, -1);
    primary[7] = 0;
    std::vector<uint64_t> sizes{100, 200};
    std::vector<strata::core::SecondaryCandidate> out;
    std::string err;
    CHECK(strata::core::secondary_candidates(ranked, primary, sizes, 16, out, err));
    CHECK(out.size() == 2);
    CHECK(out[0].layer == 1 && out[0].expert == 13 && out[0].bytes == 200);
    CHECK(out[1].layer == 0 && out[1].expert == 8 && out[1].bytes == 100);

    ranked.push_back({1, 13});
    CHECK(!strata::core::secondary_candidates(ranked, primary, sizes, 16, out, err));
    CHECK(out.empty());
    ranked.back() = {2, 0};
    CHECK(!strata::core::secondary_candidates(ranked, primary, sizes, 16, out, err));
    CHECK(out.empty());
    std::puts("secondary profile: ranked, disjoint, duplicate and bounds checks passed");
    return 0;
}
