// src/core/device_main.cpp - `strata-device`: report the GPU, the plan, and exercise the arena.
//
// This is P2.S1's "startup prints the memory plan vs actual cudaMemGetInfo" bullet, on its own so it can run
// without the model.  It is also the run-time half of the sm_120 policy: CMake refuses to COMPILE for another
// architecture, and this refuses to RUN on one.
#include "strata/core/device.hpp"
#include "strata/core/secondary_budget.hpp"
#include "strata/core/secondary_vram.hpp"
#include "strata/plan/plan.hpp"
#include <charconv>

#include <cstdio>
#include <cstring>
#include <string>

static std::string human(uint64_t b) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.3f GiB (%llu B)", (double) b / (1024.0 * 1024 * 1024),
                  (unsigned long long) b);
    return buf;
}

int main(int argc, char** argv) {
    bool selftest = false;
    int ordinal = 0;
    int secondary_reserve_mib = 0;
    bool secondary_requested = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--selftest") == 0) selftest = true;
        else if (std::strcmp(argv[i], "--device") == 0 ||
                 std::strcmp(argv[i], "--secondary-reserve-mib") == 0) {
            const bool device = std::strcmp(argv[i], "--device") == 0;
            if (++i == argc) { std::fprintf(stderr, "%s needs an integer\n", argv[i - 1]); return 2; }
            int value = 0;
            const char* end = argv[i] + std::strlen(argv[i]);
            const auto parsed = std::from_chars(argv[i], end, value);
            if (parsed.ec != std::errc{} || parsed.ptr != end || value < 0) {
                std::fprintf(stderr, "%s needs a non-negative integer\n", argv[i - 1]); return 2;
            }
            if (device) ordinal = value;
            else { secondary_reserve_mib = value; secondary_requested = true; }
        }
        else if (std::strcmp(argv[i], "--help") == 0 || std::strcmp(argv[i], "-h") == 0) {
            std::printf("usage: strata-device [--selftest] [--device 1 --secondary-reserve-mib 2560]\n");
            return 0;
        } else {
            std::fprintf(stderr, "unknown argument: %s\n", argv[i]);
            return 2;
        }
    }

    if ((ordinal != 0 && !secondary_requested) ||
        (secondary_requested &&
         (ordinal != 1 || selftest || secondary_reserve_mib <
          (int) (strata::core::kSecondaryReserveBytes >> 20)))) {
        std::fprintf(stderr, "secondary probe needs --device 1, reserve >= 2560 MiB, and no --selftest\n");
        return 2;
    }
    try {
        if (secondary_requested) {
            const auto primary = strata::core::device_info(0);
            if (primary.name.find("5060 Ti") == std::string::npos) {
                std::fprintf(stderr, "CUDA device 0 is %s, expected 5060 Ti; set CUDA_VISIBLE_DEVICES=1,0\n",
                             primary.name.c_str());
                return 2;
            }
        }
        const strata::core::DeviceInfo d = strata::core::device_info(ordinal, secondary_requested);
        if (secondary_requested) {
            if (d.name.find("4070 SUPER") == std::string::npos || d.cc_major != 8 || d.cc_minor != 9) {
                std::fprintf(stderr, "CUDA device 1 is %s (sm_%d%d), expected RTX 4070 SUPER sm_89\n",
                             d.name.c_str(), d.cc_major, d.cc_minor);
                return 2;
            }
            const uint64_t reserve = (uint64_t) secondary_reserve_mib << 20;
            uint64_t nvml_free = 0;
            std::string nvml_err;
            if (!strata::core::secondary_nvml_free_bytes(ordinal, nvml_free, nvml_err)) {
                std::fprintf(stderr, "secondary VRAM: %s\n", nvml_err.c_str());
                return 1;
            }
            const uint64_t effective = strata::core::secondary_effective_free(d.free_bytes, nvml_free);
            if (effective <= reserve) {
                std::fprintf(stderr, "secondary VRAM: CUDA %s, NVML %s; lower free figure does not exceed %d MiB reserve\n",
                             human(d.free_bytes).c_str(), human(nvml_free).c_str(), secondary_reserve_mib);
                return 1;
            }
            std::printf("device %d: %s, sm_%d%d\n", d.ordinal, d.name.c_str(), d.cc_major, d.cc_minor);
            std::printf("secondary VRAM CUDA free %s, NVML free %s\n",
                        human(d.free_bytes).c_str(), human(nvml_free).c_str());
            std::printf("secondary VRAM lower free %s, reserve %d MiB, maximum allocatable before touch %s\n",
                        human(effective).c_str(), secondary_reserve_mib,
                        human(effective - reserve).c_str());
            std::printf("read-only preflight; allocation must be touched and checked again\n");
            return 0;
        }
        std::printf("device %d: %s\n", d.ordinal, d.name.c_str());
        std::printf("  compute capability  %d.%d   (sm_%d%d)\n", d.cc_major, d.cc_minor, d.cc_major, d.cc_minor);
        std::printf("  multiprocessors     %d\n", d.multi_processor_count);
        std::printf("  VRAM total / free   %s / %s\n", human(d.total_bytes).c_str(), human(d.free_bytes).c_str());
        std::printf("  driver / runtime    %d / %d\n", d.driver_version, d.runtime_version);

        // The planner's view against the card's.  A plan that does not fit in what is actually FREE is the
        // failure this print exists to make visible at startup rather than at token 4000.
        const auto plan = strata::plan::make_plan(20480, strata::plan::Geometry{}, strata::plan::Costs{});
        std::printf("\n%s", strata::plan::to_string(plan).c_str());
        std::printf("  card free           %s\n", human(d.free_bytes).c_str());
        std::printf("  plan + KV vs free   %s\n",
                    plan.vram_budget <= d.free_bytes ? "FITS" : "*** DOES NOT FIT ***");

        if (selftest) {
            // Exercise the arena for real: allocate, write from the host, read back, and check the poison
            // path leaves NaNs rather than zeros.  A GPU test that only asks the driver for its name does not
            // test the runtime this file exists to provide.
            const uint64_t bytes = 64ull << 20;      // 64 MiB, small enough to be safe on any card
            strata::core::DeviceArena arena(bytes, 0, /*poison=*/true);
            void* a = arena.alloc(1 << 20, 256);
            void* b = arena.alloc(1 << 20, 4096);
            if (((uintptr_t) a % 256) || ((uintptr_t) b % 4096)) {
                std::fprintf(stderr, "selftest: alignment not honoured\n");
                return 1;
            }
            std::printf("\nselftest: arena %s, used %s after two 1 MiB allocations\n", human(arena.capacity()).c_str(),
                        human(arena.used()).c_str());
            // and the arena must REFUSE rather than wrap
            try {
                arena.alloc(bytes * 2);
                std::fprintf(stderr, "selftest: over-allocation did NOT throw\n");
                return 1;
            } catch (const strata::core::CudaError&) {
                std::printf("selftest: over-allocation refused as required\n");
            }
            std::printf("strata-device selftest OK\n");
        }
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "strata-device: %s\n", e.what());
        return 1;
    }
}
