// #11: Swift 1.5 IQ3_XXS (i-quant gate/up, IQ4_NL / Q2_0 down) was refused by --exclusive-primary-experts
// ("requires native Q2_0"), so --ram-cache-gib could never reach its NVMe tier.  An explicit request is allowed on
// any pack the native kernels compute; the automatic default and the 4070 tier stay Q2_0-only.  And
// --ram-cache-gib alone was silently ignored without placement-first (generate.cpp `place_first && ram_cache_gib`),
// leaving a 40 GiB resident arena on a 47.7 GB PC: it now asks for exclusive ownership itself.
#include "strata/core/placement_formats.hpp"

#include <cstdio>
#include <vector>

#define CHECK(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "failed: %s\n", #condition); return 1; } \
} while (false)

namespace {
struct Fmt { int gu_type, d_type; };
// a stand-in for the engine's iq_supported: the formats the cases below use, and not Q4_K (12)
bool supported(int t) { return t == 16 || t == 17 || t == 18 || t == 20 || t == 21 || t == 22 || t == 42; }
}  // namespace

int main() {
    using strata::core::exclusive_primary_formats_ok;
    using strata::core::exclusive_requested;
    using strata::core::kGgmlQ2_0;
    using strata::core::placement_formats;
    // the 10 gate/up x down pairs of Swift 1.5 IQ3_XXS's 48 layers (native_experts.txt)
    const std::vector<Fmt> swift = {{16, 20}, {16, 42}, {17, 20}, {17, 42}, {18, 20},
                                    {18, 42}, {21, 20}, {21, 42}, {22, 20}, {22, 42}};
    const std::vector<Fmt> ista_q2(48, Fmt{kGgmlQ2_0, kGgmlQ2_0});
    const std::vector<Fmt> unknown = {{16, 20}, {12, 42}};   // Q4_K gate/up: no native kernel

    const auto s = placement_formats(true, swift, supported);
    CHECK(!s.all_q2 && s.all_native);
    CHECK(exclusive_primary_formats_ok(s, /*requested=*/true));    // explicit: allowed
    CHECK(!exclusive_primary_formats_ok(s, /*requested=*/false));  // automatic default: Q2_0 only

    const auto q = placement_formats(true, ista_q2, supported);
    CHECK(q.all_q2 && q.all_native);
    CHECK(exclusive_primary_formats_ok(q, false) && exclusive_primary_formats_ok(q, true));

    const auto u = placement_formats(true, unknown, supported);
    CHECK(!u.all_q2 && !u.all_native);
    CHECK(!exclusive_primary_formats_ok(u, true));

    const auto canonical = placement_formats(false, ista_q2, supported);   // not a native pack
    CHECK(!canonical.all_q2 && !canonical.all_native);
    CHECK(!exclusive_primary_formats_ok(canonical, true));

    // exclusive_mode: 1 --exclusive-primary-experts, 0 --no-exclusive-primary-experts, -1 neither
    CHECK(exclusive_requested(1, 0.0));
    CHECK(!exclusive_requested(-1, 0.0));
    CHECK(exclusive_requested(-1, 8.0));    // --ram-cache-gib asks for placement-first
    CHECK(!exclusive_requested(0, 8.0));    // an explicit opt-out wins (the engine then refuses the cache)
    std::printf("placement_formats: ok\n");
    return 0;
}
