// tests/xeno/secondary_pair.cpp - #117: the secondary tier pairs any two distinct cards the build can run.  Before
// #117 only a 5060 Ti then a 4070 SUPER sm_89 (by name), so "the 4070 SUPER as primary with the 5060 Ti as its tier"
// could not even start (#112).
#include "strata/core/secondary_arena.hpp"

#include <cstdio>
#include <string>

using strata::core::secondary_pair_ok;

int main() {
    int bad = 0;
    auto expect = [&](bool ok, const char* what) { if (!ok) { std::fprintf(stderr, "FAIL: %s\n", what); ++bad; } };
    std::string why;

    expect(secondary_pair_ok(120, 89, true, why), "5060 Ti primary, 4070 SUPER tier (the served D2x)");
    expect(secondary_pair_ok(89, 120, true, why), "4070 SUPER primary, 5060 Ti tier");
    expect(secondary_pair_ok(86, 89, true, why), "any two sm_80+ cards");
    why.clear();
    expect(!secondary_pair_ok(120, 120, false, why) && !why.empty(), "the same card twice: refused, with a reason");
    why.clear();
    expect(!secondary_pair_ok(120, 75, true, why) && !why.empty(), "a tier older than sm_80: refused, with a reason");

    if (bad == 0) std::printf("secondary_pair: all cases pass\n");
    return bad == 0 ? 0 : 1;
}
