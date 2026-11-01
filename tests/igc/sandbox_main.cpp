// Runs one program through the #279 sandbox: its output to stdout, one status line to stderr.
// sandbox_main file.py [--steps N] [--heap N] [--output N] [--recursion N] [--parser N] [--digits N] [--cancel-ms N]
#include "strata/core/igc_sandbox.hpp"
#include <atomic>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <string>
#include <string_view>
#include <thread>
#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#endif
using namespace strata::igc::sandbox;

static const char* name(Status status) {
    switch (status) {
        case Status::Ok: return "Ok";
        case Status::Raised: return "Raised";
        case Status::Syntax: return "Syntax";
        case Status::Unsupported: return "Unsupported";
        case Status::StepLimit: return "StepLimit";
        case Status::MemoryLimit: return "MemoryLimit";
        case Status::RecursionLimit: return "RecursionLimit";
        case Status::Cancelled: return "Cancelled";
    }
    return "?";
}

int main(int argc, char** argv) {
    if (argc < 2) return 2;
    Limits limits;
    long long cancel_ms = -1;
    for (int i = 2; i + 1 < argc; i += 2) {
        const std::string_view flag = argv[i];
        const unsigned long long value = std::stoull(argv[i + 1]);
        if (flag == "--steps") limits.steps = value;
        else if (flag == "--heap") limits.heap_bytes = (std::size_t) value;
        else if (flag == "--output") limits.output_bytes = (std::size_t) value;
        else if (flag == "--recursion") limits.recursion_depth = (int) value;
        else if (flag == "--parser") limits.parser_depth = (int) value;
        else if (flag == "--digits") limits.integer_digits = (int) value;
        else if (flag == "--cancel-ms") cancel_ms = (long long) value;
        else return 2;
    }
    std::ifstream file(argv[1], std::ios::binary);
    if (!file) return 2;
    const std::string source((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    std::atomic<bool> cancel{false};
    std::thread canceller;
    if (cancel_ms >= 0) canceller = std::thread([&] {
        std::this_thread::sleep_for(std::chrono::milliseconds(cancel_ms));
        cancel = true;
    });
    const auto start = std::chrono::steady_clock::now();
    const Result result = run(source, limits, cancel_ms >= 0 ? &cancel : nullptr);
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    if (canceller.joinable()) canceller.join();
#ifdef _WIN32
    _setmode(_fileno(stdout), _O_BINARY);
#endif
    std::fwrite(result.output.data(), 1, result.output.size(), stdout);
    std::fflush(stdout);
    std::fprintf(stderr, "status=%s steps=%llu peak=%llu truncated=%d ms=%.3f", name(result.status),
                 (unsigned long long) result.steps, (unsigned long long) result.peak_bytes, result.truncated ? 1 : 0, ms);
    if (!result.error.empty()) std::fprintf(stderr, " error=%s", result.error.c_str());
    std::fprintf(stderr, "\n");
    return 0;
}
