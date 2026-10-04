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
    // the D2x shapes: 4-cell pages, 262,144 cells (65,536 pages).  No floor: a floor at the resident window (65,536 cells)
    // read every part 20-50 ms slower and lent ~500 more slots per run (#122, merge-137-record/f122-run.txt)
    const long long P = 4, N = 65536;
    expect(kv_stage_pages(0, N, P), N, "kv_end 0: the worst case");
    expect(kv_stage_pages(-5, N, P), N, "kv_end negative: the worst case");
    expect(kv_stage_pages(10000, N, P), 4096, "a 10K turn: 16,384 cells");
    expect(kv_stage_pages(65536, N, P), 16384, "a power of two stays");
    expect(kv_stage_pages(65537, N, P), 32768, "one cell past it: 131,072 cells");
    expect(kv_stage_pages(200000, N, P), 65536, "200K: 262,144 cells");
    expect(kv_stage_pages(300000, N, P), N, "past the context limit: capped");
    expect(kv_stage_pages(5, N, P), 2, "5 cells: 8 cells, 2 pages");
    expect(kv_stage_pages(3, N, P), 1, "3 cells: 4 cells, 1 page");
    expect(kv_stage_pages(300, 100, P), 100, "a cap that is not a power of two");
    expect(kv_stage_pages(5, N, 3), 3, "pages round up: 8 cells of 3-cell pages");
    if (bad == 0) std::printf("kv_stage_plan: all cases pass\n");
    return bad == 0 ? 0 : 1;
}
