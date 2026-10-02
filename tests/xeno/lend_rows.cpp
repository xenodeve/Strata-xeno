// tests/xeno/lend_rows.cpp - #102: on a layer split every participant (CUDA0 and each stage) lends prompt-path slots out
// of its OWN expert cache, and must mark only its OWN layers' rows as lent.  Slot numbers are per cache, so a stage's
// rows can hold slot numbers >= CUDA0's first lent slot: the fork's CUDA0 lend used to mark every row of the global
// residency table with `host_res[i] >= first`, which on a split would also take the stage's rows - the stage would then
// keep serving prompt buffers as experts (wrong tokens, no error).  lend_rows marks rows by layer range and slot.
#include "strata/core/expert_cache.hpp"
#include "strata/core/lend_rows.hpp"

#include <cstdint>
#include <cstdio>
#include <utility>
#include <vector>

int main() {
    int bad = 0;
    auto expect = [&](bool ok, const char* what) { if (!ok) { std::fprintf(stderr, "FAIL: %s\n", what); ++bad; } };
    const int64_t n_expert = 4;
    // 4 layers: CUDA0 holds layers 0-1, the stage layers 2-3; both caches have 8 slots, numbered 0..7 each
    std::vector<int32_t> res = {
        0, 1, 2, strata::core::kNotResident,   // layer 0 (CUDA0)
        5, 6, 7, 3,                            // layer 1 (CUDA0)
        0, 6, 1, 7,                            // layer 2 (stage)
        5, strata::core::kNotResident, 2, 3,   // layer 3 (stage)
    };
    const std::vector<int32_t> before = res;

    // CUDA0 lends its slots from 5 up: rows (1,0) slot 5, (1,1) slot 6, (1,2) slot 7 - and NOT the stage's 6, 7, 7, 5
    std::vector<std::pair<int32_t, int32_t>> lent0;
    strata::core::lend_rows(res, n_expert, /*lb=*/0, /*le=*/2, /*first=*/5, lent0);
    expect(lent0.size() == 3, "CUDA0 lends exactly its own 3 rows at slot >= 5");
    expect(lent0.size() == 3 && lent0[0] == std::make_pair(4, 5) && lent0[1] == std::make_pair(5, 6) &&
               lent0[2] == std::make_pair(6, 7), "(row index, slot) pairs, in row order");
    expect(res[4] == strata::core::kNotResident && res[5] == strata::core::kNotResident &&
               res[6] == strata::core::kNotResident, "CUDA0's lent rows are no longer resident");
    expect(res[9] == 6 && res[11] == 7 && res[12] == 5, "the stage's rows with slot >= 5 are untouched");
    expect(res[7] == 3 && res[0] == 0, "CUDA0's rows below the first lent slot stay resident");

    // the stage lends its slots from 6 up: rows (2,1) slot 6, (2,3) slot 7
    std::vector<std::pair<int32_t, int32_t>> lent1;
    strata::core::lend_rows(res, n_expert, 2, 4, 6, lent1);
    expect(lent1.size() == 2 && lent1[0] == std::make_pair(9, 6) && lent1[1] == std::make_pair(11, 7),
           "the stage lends only its own rows at slot >= 6");
    expect(res[12] == 5, "the stage's row below its first lent slot stays resident");

    // a row already not resident is never lent, and giving every loan back restores the table exactly
    for (const auto& [i, slot] : lent0) res[(size_t) i] = slot;
    for (const auto& [i, slot] : lent1) res[(size_t) i] = slot;
    expect(res == before, "giving the loans back restores the residency table");

    // one card (no split): lb 0, le n_layers - the same rows the old global loop marked
    std::vector<std::pair<int32_t, int32_t>> all;
    strata::core::lend_rows(res, n_expert, 0, 4, 6, all);
    size_t old = 0;
    for (int32_t s : before) old += s >= 6;
    expect(all.size() == old, "one card: lb..le covers every layer, as the old loop did");

    if (bad == 0) std::printf("xeno_lend_rows: PASS\n");
    return bad == 0 ? 0 : 1;
}
