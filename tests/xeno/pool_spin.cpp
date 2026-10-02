// tests/xeno/pool_spin.cpp - #64: the pool's park spin is a constructor choice.  Capacity mode (an NVMe tier) builds
// the pool with no spin: with the SMT siblings busy, 13 HIGHEST workers spinning for 20 ms leave no logical CPU for
// the NVMe path and decode drops to a third.  The default stays kSpinBeforeSleep, and STRATA_POOL_SPIN_US still
// overrides either (the A/B knob).
#include "strata/kernels/cpu/pool.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>

using strata::kernels::cpu::ExpertPool;

int main() {
    int bad = 0;
    auto expect = [&](bool ok, const char* what) { if (!ok) { std::fprintf(stderr, "FAIL: %s\n", what); ++bad; } };
    _putenv_s("STRATA_POOL_SPIN_US", "");
    {
        ExpertPool p(1, false, true);
        expect(p.spin_before_sleep() == std::chrono::microseconds(ExpertPool::kSpinBeforeSleep), "the default spin");
    }
    {
        ExpertPool p(1, false, true, 0);
        expect(p.spin_before_sleep() == std::chrono::microseconds(0), "spin_us 0: no spin");
    }
    _putenv_s("STRATA_POOL_SPIN_US", "123");
    {
        ExpertPool p(1, false, true, 0);
        expect(p.spin_before_sleep() == std::chrono::microseconds(123), "the env knob overrides the argument");
    }
    _putenv_s("STRATA_POOL_SPIN_US", "");
    if (bad == 0) std::printf("xeno_pool_spin: PASS\n");
    return bad == 0 ? 0 : 1;
}
