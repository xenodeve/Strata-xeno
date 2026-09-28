// A parked worker spins for kSpinBeforeSleep (20 ms) before it sleeps. Between verify windows that spin held every
// pinned core at HIGHEST priority, so with 13 workers the adapt thread could not run: `join wait` rose from 0.9 to
// 19 ms/round (strata-claude-workers thai-3-C). `rest()` sends parked workers to sleep at once; the next publish
// wakes them as before.
#include "strata/kernels/cpu/pool.hpp"

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <chrono>
#include <cstdio>
#include <thread>

static unsigned long long cpu_ms() {
    FILETIME created, exited, kernel, user;
    if (!GetProcessTimes(GetCurrentProcess(), &created, &exited, &kernel, &user)) return 0;
    ULARGE_INTEGER k, u;
    k.LowPart = kernel.dwLowDateTime; k.HighPart = kernel.dwHighDateTime;
    u.LowPart = user.dwLowDateTime; u.HighPart = user.dwHighDateTime;
    return (k.QuadPart + u.QuadPart) / 10000;
}

int main() {
    // Four fresh pools: each one's two workers start parked and spinning. Without rest() that is about
    // 4 x 2 x 20 = 160 ms of CPU time; with it the workers sleep at once.
    const auto start = cpu_ms();
    for (int i = 0; i < 4; ++i) {
        strata::kernels::cpu::ExpertPool pool(2, false, true);
        pool.rest();
        std::this_thread::sleep_for(std::chrono::milliseconds(30));
    }
    const auto spent = cpu_ms() - start;
    std::printf("rest(): process CPU %llu ms over four 30 ms idle pools\n", spent);
    return spent < 60 ? 0 : 1;
}
