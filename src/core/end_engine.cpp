// src/core/end_engine.cpp - see include/strata/core/end_engine.hpp.
#include "strata/core/end_engine.hpp"

#include "strata/core/progress.hpp"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#if defined(_WIN32)
#include <windows.h>
#endif

namespace strata::core {
namespace {
std::atomic<void (*)()> g_arm{nullptr};
}  // namespace

void set_end_engine_arm(void (*arm)()) { g_arm.store(arm); }

void end_engine(int code, const char* why) {
    if (why != nullptr) {
        std::fprintf(stderr, "%s\n", why);
        std::fflush(stderr);
    }
    if (auto arm = g_arm.load()) arm();     // whatever blocks below, the deadline ends the process
    release_gpu_waits(stderr);              // #267: no spin kernel keeps the GPU (and the exit) waiting
    std::fflush(stderr);
#if defined(_WIN32)
    TerminateProcess(GetCurrentProcess(), (UINT) code);   // no DLL detach: nothing waits on the loader lock (#185)
#endif
    std::_Exit(code);
}

}  // namespace strata::core
