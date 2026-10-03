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

    // #115: the wave runs wherever the split can run - a one-card layer publishes its hand-off itself
    expect(strata::prefill::wave_ok(split_plan(true, all)), "every layer on MMQ: the wave may run");
    expect(strata::prefill::wave_ok(split_plan(true, swift)), "three IQ1_M layers: the wave may run");
    expect(!strata::prefill::wave_ok(split_plan(true, std::vector<char>(48, 0))), "no MMQ layer: no wave");
    expect(!strata::prefill::wave_ok(split_plan(false, all)), "not a native pack: no wave");

    // #119: the split's chunk floor - opt-in, the old constant otherwise
    using strata::prefill::split_min_from;
    expect(split_min_from(nullptr) == 2048, "unset: 2048, as before");
    expect(split_min_from("1024") == 1024, "1024: the split from 1,024-token chunks");
    expect(split_min_from("512") == 512, "512");
    expect(split_min_from("abc") == 2048 && split_min_from("") == 2048, "not a number: 2048");
    expect(split_min_from("100") == 2048 && split_min_from("99999999") == 2048, "out of 256-65536: 2048");

    // #119: the wave keeps its 2048-token lanes when the split's floor is lowered (lanes of ~1K were 5.1-5.6 s against
    // 3.2 s on one lane for 1,768-2,248-token parts)
    using strata::prefill::wave_lane_ok;
    expect(wave_lane_ok(2048, 2048) && wave_lane_ok(4096, 1024), "lanes of 2048+: wave, as before");
    expect(!wave_lane_ok(1024, 1024), "floor 1024, lanes of 1024: no wave (one lane runs split)");
    expect(!wave_lane_ok(1024, 512) && !wave_lane_ok(1500, 2048), "lanes below 2048: no wave");

    if (bad == 0) std::printf("split_plan: all cases pass\n");
    return bad == 0 ? 0 : 1;
}
