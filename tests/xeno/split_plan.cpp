// tests/xeno/split_plan.cpp - #113: the expert split is decided per layer.  Swift 1.5 IQ2_XS has three IQ1_M layers
// (8, 13, 37) that MMQ cannot run; before #113 they turned the whole split off and the wave ran on one card, streaming
// every expert twice (#112: 66 GB per 8K prompt, ~525 tok/s instead of ~870).
#include "strata/prefill/split_plan.hpp"

#include <atomic>
#include <cstdint>
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
    expect(split_min_from("256") == 256 && split_min_from("2048") == 2048, "the bounds are inclusive");
    // scrutiny of 7673c02: above 2048 nothing was measured, and a floor above the wave's lane makes both lanes run
    // unsplit (the slow pattern of tl119-*.json) while a SplitTier sits unused on the 4070
    expect(split_min_from("255") == 2048 && split_min_from("4096") == 2048 && split_min_from("65536") == 2048,
           "outside 256-2048: 2048");
    expect(split_min_from("512x") == 2048 && split_min_from("-512") == 2048, "trailing garbage or negative: 2048");

    // #119: the wave keeps its 2048-token lanes when the split's floor is lowered (lanes of ~1K were 5.1-5.6 s against
    // 3.2 s on one lane for 1,768-2,248-token parts)
    using strata::prefill::wave_lane_ok;
    expect(wave_lane_ok(2048, 2048) && wave_lane_ok(4096, 1024), "lanes of 2048+: wave, as before");
    expect(!wave_lane_ok(1024, 1024), "floor 1024, lanes of 1024: no wave (one lane runs split)");
    expect(!wave_lane_ok(1024, 512) && !wave_lane_ok(1500, 2048), "lanes below 2048: no wave");

    // #133: each card's experts are one contiguous row block - the 4070's first, CUDA0's last, walk order kept inside
    {
        using strata::prefill::split_row_layout;
        const std::vector<int32_t> cnt = {3, 0, 2, 5, 1, 4};        // rows per expert
        const std::vector<int32_t> at = {5, 4, 3, 2, 1, 0};         // a reversed walk order
        const std::vector<char> local = {1, 0, 0, 1, 1, 0};         // CUDA0 holds 0, 3, 4
        std::vector<int32_t> off;
        const int64_t lf = split_row_layout(cnt, at, local, off);
        // remote in walk order: 5 (4 rows), 2 (2), 1 (0) -> 0, 4, 6; local: 4 (1), 3 (5), 0 (3) -> 6, 7, 12
        expect(off == std::vector<int32_t>({12, 6, 4, 7, 6, 0, 15}), "row offsets: remote block, then local block");
        expect(lf == 6, "the local block starts after the 4070's rows");
        const std::vector<char> none(6, 0), all(6, 1);
        expect(split_row_layout(cnt, at, none, off) == 15 && off[6] == 15, "nothing local: the block starts at the end");
        expect(split_row_layout(cnt, at, all, off) == 0 && off[5] == 0, "all local: the block starts at row 0");
    }

    // #142: a short split chunk copies only the host experts its routing reaches (a plan entry is copied or skipped
    // once its layer is routed); 13.3K copies per Claude Code part against 5.4-7.6K routed host experts
    {
        using strata::prefill::split_routed_max_from;
        using strata::prefill::split_routed_only;
        using strata::prefill::kSplitRoutedMaxDefault;
        expect(split_routed_max_from(nullptr) == kSplitRoutedMaxDefault && split_routed_max_from("") == kSplitRoutedMaxDefault,
               "unset: the default");
        expect(split_routed_max_from("0") == 0, "0: off (every chunk streams the walk, as before)");
        expect(split_routed_max_from("4096") == 4096, "4096: chunks below 4,096 tokens");
        expect(split_routed_max_from("abc") == kSplitRoutedMaxDefault && split_routed_max_from("-5") == kSplitRoutedMaxDefault &&
                   split_routed_max_from("12x") == kSplitRoutedMaxDefault,
               "not a non-negative number: the default");
        expect(split_routed_only(399, false, 2048) && split_routed_only(2047, false, 2048), "below the limit: routed only");
        expect(!split_routed_only(2048, false, 2048), "at the limit: the walk");
        expect(!split_routed_only(399, true, 2048), "a wave: the walk (both lanes read one plan)");
        expect(!split_routed_only(399, false, 0), "off: the walk");

        using strata::prefill::split_mark_routed;
        struct Entry { int32_t l, e; };   // the plan's entries carry the expert in `e`
        const std::vector<int32_t> cnt = {0, 0, 0, 0, 0, 2, 0, 0, 0, 1};
        const Entry entry[4] = {{3, 5}, {3, 7}, {3, 9}, {3, 0}};   // a layer's plan entries, in plan order
        std::atomic<bool> skip[4];
        for (auto& f : skip) f.store(false);
        split_mark_routed(entry, 4, cnt.data(), skip);
        expect(!skip[0].load() && skip[1].load() && !skip[2].load() && skip[3].load(),
               "routed entries copy, the others skip");
    }

    if (bad == 0) std::printf("split_plan: all cases pass\n");
    return bad == 0 ? 0 : 1;
}
