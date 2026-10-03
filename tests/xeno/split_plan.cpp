// tests/xeno/split_plan.cpp - #113: the expert split is decided per layer.  Swift 1.5 IQ2_XS has three IQ1_M layers
// (8, 13, 37) that MMQ cannot run; before #113 they turned the whole split off and the wave ran on one card, streaming
// every expert twice (#112: 66 GB per 8K prompt, ~525 tok/s instead of ~870).
#include "strata/prefill/split_plan.hpp"

#include <cstdio>
#include <vector>

using strata::prefill::split_plan;

int main() {
    int bad = 0;
    auto expect = [&](bool ok, const char* what) { if (!ok) { std::fprintf(stderr, "FAIL: %s\n", what); ++bad; } };

    std::vector<char> all(48, 1);
    auto p = split_plan(true, all);
    expect(p.usable && p.full && p.one_card.empty(), "every layer on MMQ: usable and full");

    std::vector<char> swift = all;
    swift[8] = swift[13] = swift[37] = 0;
    p = split_plan(true, swift);
    expect(p.usable, "three IQ1_M layers: still usable");
    expect(!p.full, "three IQ1_M layers: not full (the wave stays off, the one-card buffers hold the chunk)");
    expect(p.one_card == std::vector<int>({8, 13, 37}), "the IQ1_M layers stay on CUDA0, in order");

    p = split_plan(true, std::vector<char>(48, 0));
    expect(!p.usable && !p.full, "no MMQ layer: not usable");

    p = split_plan(false, all);
    expect(!p.usable && !p.full, "not a native pack: not usable");

    p = split_plan(true, {});
    expect(!p.usable && !p.full, "no layers: not usable");

    if (bad == 0) std::printf("split_plan: all cases pass\n");
    return bad == 0 ? 0 : 1;
}
