// #220 A5 STRATA_POOL_ECORE_SKIP_TAIL=K: an E-core worker claims no task among the last K of a batch, so the P-cores
// (and the host) take the batch's tail instead of waiting for an E-core that runs a task ~2.4x slower.  The rule is
// pure: the task rows are unchanged, so the outputs are too (xeno_pool_fused_parity runs the real batches with it).
//
//     xeno_pool_ecore_tail
#include "strata/kernels/cpu/pool.hpp"

#include <cstdio>
#include <cstdlib>

using strata::kernels::cpu::ExpertPool;

static int failures = 0;
static void expect(bool ok, const char* what) {
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", what); ++failures; }
}

int main() {
    // keep = 0: every thread claims every task
    expect(ExpertPool::may_claim(83, 84, false, 6, 6), "a P-core claims the last task");
    expect(ExpertPool::may_claim(83, 84, true, 0, 6), "K = 0: an E-core claims the last task");
    // an E-core stops at n - K; a P-core does not
    expect(ExpertPool::may_claim(77, 84, true, 6, 6), "an E-core claims task n - K - 1");
    expect(!ExpertPool::may_claim(78, 84, true, 6, 6), "an E-core does not claim task n - K");
    expect(!ExpertPool::may_claim(83, 84, true, 6, 6), "an E-core does not claim the last task");
    expect(ExpertPool::may_claim(78, 84, false, 6, 6), "a P-core claims task n - K");
    // a batch smaller than twice the fast threads is left alone (the P-cores alone would be the slower side)
    expect(ExpertPool::may_claim(10, 11, true, 6, 6), "a small batch: an E-core claims its tail");
    expect(!ExpertPool::may_claim(11, 12, true, 6, 6), "a batch of 2 x fast threads: the rule holds");
    // no fast thread to take the tail (fast = 0): never skip
    expect(ExpertPool::may_claim(83, 84, true, 6, 0), "no fast thread: an E-core claims the tail");
    // an exhausted batch is never claimable
    expect(!ExpertPool::may_claim(84, 84, false, 0, 6), "past the end");

    // the switch is read at construction; an unpinned pool knows no core classes and keeps it off
    _putenv("STRATA_POOL_ECORE_SKIP_TAIL=6");
    ExpertPool unpinned(4, false, true);
    expect(unpinned.ecore_skip_tail() == 0, "an unpinned pool does not skip (no core classes)");
    ExpertPool pinned(13, true, true);
    std::printf("pinned pool: %d workers, %d fast threads (host included), E-core skip tail %d\n", pinned.workers(),
                pinned.fast_threads(), pinned.ecore_skip_tail());
    expect(pinned.e_cores() == 0 || pinned.ecore_skip_tail() == 6, "a pinned hybrid pool takes K from the switch");
    _putenv("STRATA_POOL_ECORE_SKIP_TAIL=");
    ExpertPool off(13, true, true);
    expect(off.ecore_skip_tail() == 0, "unset: off");
    std::printf("%s (%d failures)\n", failures ? "FAILED" : "ok", failures);
    return failures ? 1 : 0;
}
