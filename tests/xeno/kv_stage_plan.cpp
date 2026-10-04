// tests/xeno/kv_stage_plan.cpp - #122: the KV-streaming staging pool is sized by the request's end position.  Before
// #122 it always held every page of the context limit: 2 x 264 MiB of borrowed expert slots per request at 262K.
#include "strata/prefill/kv_stage_plan.hpp"

#include <cstdio>

using strata::prefill::kv_stage_pages;

int main() {
    int bad = 0;
    auto expect = [&](long long got, long long want, const char* what) {
        if (got != want) { std::fprintf(stderr, "FAIL: %s: got %lld, want %lld\n", what, got, want); ++bad; }
    };
    // the D2x shapes: 4-cell pages, 262,144 cells (65,536 pages), --kv-resident 65536 as the floor
    const long long P = 4, N = 65536, F = 65536;
    expect(kv_stage_pages(0, F, N, P), N, "kv_end 0: the worst case");
    expect(kv_stage_pages(-5, F, N, P), N, "kv_end negative: the worst case");
    expect(kv_stage_pages(10000, F, N, P), 16384, "a 10K turn: the floor's 65,536 cells");
    expect(kv_stage_pages(65536, F, N, P), 16384, "exactly the floor");
    expect(kv_stage_pages(65537, F, N, P), 32768, "one cell past the floor: 131,072 cells");
    expect(kv_stage_pages(131072, F, N, P), 32768, "a power of two stays");
    expect(kv_stage_pages(200000, F, N, P), 65536, "200K: 262,144 cells");
    expect(kv_stage_pages(300000, F, N, P), N, "past the context limit: capped");
    expect(kv_stage_pages(10000, 0, N, P), 4096, "no floor: 16,384 cells");
    expect(kv_stage_pages(5, 0, N, P), 2, "no floor, 5 cells: 8 cells, 2 pages");
    expect(kv_stage_pages(3, 0, N, P), 1, "no floor, 3 cells: 4 cells, 1 page");
    expect(kv_stage_pages(300, 0, 100, P), 100, "a cap that is not a power of two");
    expect(kv_stage_pages(1, 20480, N, P), 5120, "the engine's minimum resident (20,480): 5,120 pages");
    expect(kv_stage_pages(5, 0, N, 3), 3, "pages round up: 8 cells of 3-cell pages");
    if (bad == 0) std::printf("kv_stage_plan: all cases pass\n");
    return bad == 0 ? 0 : 1;
}
