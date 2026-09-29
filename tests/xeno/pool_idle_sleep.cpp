// The resident server must not burn P/E cores while no expert job is pending.
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
    strata::kernels::cpu::ExpertPool pool(2, false, true);
    const auto start = cpu_ms();
    std::this_thread::sleep_for(std::chrono::milliseconds(150));
    const auto spent = cpu_ms() - start;
    std::printf("two idle pool workers: process CPU %llu ms over 150 ms wall\n", spent);
    return spent < 140 ? 0 : 1;
}
