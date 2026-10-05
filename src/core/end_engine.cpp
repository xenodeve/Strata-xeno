// src/core/end_engine.cpp - see include/strata/core/end_engine.hpp.  CUDA-free: the CPU pool uses it too.
#include "strata/core/end_engine.hpp"

#include "strata/core/progress.hpp"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <thread>
#if defined(_WIN32)
#include <windows.h>
#endif

namespace strata::core {
namespace {
std::atomic<bool> g_started{false}, g_armed{false};
std::atomic<int> g_code{3};
std::mutex g_mu;
std::condition_variable g_cv;

[[noreturn]] void terminate_now(int code) {
#if defined(_WIN32)
    TerminateProcess(GetCurrentProcess(), (UINT) code);   // no DLL detach: nothing waits on the loader lock (#185)
#endif
    std::_Exit(code);
}
}  // namespace

void start_end_engine_deadline(int ms) {
    bool expected = false;
    if (!g_started.compare_exchange_strong(expected, true)) return;
    std::thread([ms] {
        {
            std::unique_lock<std::mutex> lk(g_mu);
            g_cv.wait(lk, [] { return g_armed.load(); });
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(ms));
        terminate_now(g_code.load());
    }).detach();
}

void arm_end_engine() {
    {
        std::lock_guard<std::mutex> lk(g_mu);
        g_armed.store(true);
    }
    g_cv.notify_all();
}

void end_engine(int code, const char* why) {
    if (why != nullptr) {
        std::fprintf(stderr, "%s\n", why);
        std::fflush(stderr);
    }
    g_code.store(code);
    arm_end_engine();                       // whatever blocks below, the deadline ends the process
    release_gpu_waits(stderr);              // #267: no spin kernel keeps the GPU (and the exit) waiting
    if (why != nullptr) std::fprintf(stderr, "%s\n", why);   // last: the cause the server shows
    std::fflush(stderr);
    terminate_now(code);
}

}  // namespace strata::core
