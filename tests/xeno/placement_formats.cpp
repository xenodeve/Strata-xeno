// #11: Swift 1.5 IQ3_XXS (i-quant gate/up, IQ4_NL / Q2_0 down) was refused by --exclusive-primary-experts
// ("requires native Q2_0"), so --ram-cache-gib could never reach its NVMe tier.  An explicit request is allowed on
// any pack the native kernels compute; the automatic default and the 4070 tier stay Q2_0-only.
#include "strata/core/placement_formats.hpp"

#include <cstdio>
#include <vector>

#define CHECK(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "failed: %s\n", #condition); return 1; } \
} while (false)

namespace {
struct Fmt { int gu_type, d_type; };
// the engine's iq_supported (src/kernels/cuda/iq_kernels.cu is_iq)
bool supported(int t) { return t == 16 || t == 17 || t == 18 || t == 20 || t == 21 || t == 22 || t == 23 ||
                               t == 29 || t == 42 || t == 11; }
}  // namespace

int main() {
    using strata::core::exclusive_primary_formats_ok;
    using strata::core::placement_formats;
    // the 10 gate/up x down pairs of Swift 1.5 IQ3_XXS's 48 layers (native_experts.txt)
    const std::vector<Fmt> swift = {{16, 20}, {16, 42}, {17, 20}, {17, 42}, {18, 20},
                                    {18, 42}, {21, 20}, {21, 42}, {22, 20}, {22, 42}};
    const std::vector<Fmt> ista_q2(48, Fmt{42, 42});
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
    std::printf("placement_formats: ok\n");
    return 0;
}
