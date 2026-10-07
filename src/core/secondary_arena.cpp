#include "strata/core/secondary_arena.hpp"
#include "strata/core/secondary_budget.hpp"
#include "strata/core/secondary_vram.hpp"

#include <cuda_runtime.h>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <utility>

namespace strata::core {
namespace {

constexpr uint64_t kAllocationCushion = 64ull << 20;

struct RestoreDevice {
    int previous;
    ~RestoreDevice() { cudaSetDevice(previous); }
};

bool free_snapshot(int ordinal, uint64_t& lower, std::string& err) {
    size_t cuda_free = 0, cuda_total = 0;
    int current = -1;
    (void) cudaGetDevice(&current);
    const cudaError_t pending = cudaPeekAtLastError();
    const cudaError_t rc = cudaMemGetInfo(&cuda_free, &cuda_total);
    if (rc != cudaSuccess) {
        uint64_t nvml_free = 0;
        std::string nvml_err;
        const bool nvml_ok = secondary_nvml_free_bytes(ordinal, nvml_free, nvml_err);
        err = std::string("secondary cudaMemGetInfo on CUDA device ") + std::to_string(current) +
              ": " + cudaGetErrorString(rc) + "; pending before query: " + cudaGetErrorString(pending) +
              (nvml_ok ? "; NVML free: " + std::to_string(nvml_free) + " B" : "; " + nvml_err);
        return false;
    }
    uint64_t nvml_free = 0;
    if (!secondary_nvml_free_bytes(ordinal, nvml_free, err)) return false;
    lower = secondary_effective_free((uint64_t) cuda_free, nvml_free);
    if (lower == 0) { err = "secondary free-memory query returned zero"; return false; }
    return true;
}

} // namespace

SecondaryArena::~SecondaryArena() {
    std::string err;
    if (!close(&err)) {
        std::fprintf(stderr, "SecondaryArena: %s; terminating to release display VRAM\n", err.c_str());
        std::fflush(stderr);
        std::abort();
    }
}

bool SecondaryArena::open(int ordinal, const std::vector<uint64_t>& ranked_blob_bytes,
                          uint64_t max_bytes, std::string& err, FreeReader reader,
                          void* reader_context, uint64_t free_floor_bytes) {
    if (base_ != nullptr) { err = "secondary arena is already open"; return false; }
    if (ordinal != 1 || ranked_blob_bytes.empty() || max_bytes == 0 || free_floor_bytes == 0) {
        err = "secondary arena needs CUDA device 1, ranked slots and a positive byte cap";
        return false;
    }
    int previous = -1;
    if (cudaGetDevice(&previous) != cudaSuccess || cudaSetDevice(ordinal) != cudaSuccess) {
        err = "secondary arena cannot select CUDA device 1";
        return false;
    }
    const RestoreDevice restore{previous};
    cudaDeviceProp primary{}, display{};
    if (cudaGetDeviceProperties(&primary, 0) != cudaSuccess ||
        cudaGetDeviceProperties(&display, ordinal) != cudaSuccess ||
        std::strstr(primary.name, "5060 Ti") == nullptr ||
        std::strstr(display.name, "4070 SUPER") == nullptr ||
        display.major != 8 || display.minor != 9) {
        err = "secondary arena requires CUDA_VISIBLE_DEVICES=1,0 (5060 Ti then 4070 SUPER sm_89)";
        return false;
    }
    auto snapshot = [&](uint64_t& lower) {
        return reader ? reader(ordinal, lower, err, reader_context)
                      : free_snapshot(ordinal, lower, err);
    };
    auto release_candidate = [&](uint8_t* candidate) {
        const cudaError_t released = cudaFree(candidate);
        if (released == cudaSuccess) return true;
        // Keep ownership but publish no usable slots. A later close() can retry the release.
        base_ = candidate;
        ordinal_ = ordinal;
        offsets_.clear();
        lower_free_after_ = 0;
        err += std::string("; release failed: ") + cudaGetErrorString(released);
        return false;
    };
    uint64_t limit_slots = ranked_blob_bytes.size();
    for (int attempt = 0; attempt < 8 && limit_slots > 0; ++attempt) {
        uint64_t before = 0;
        if (!snapshot(before)) return false;
        if (before <= free_floor_bytes + kAllocationCushion) {
            err = "display GPU has no space above the free floor and allocation cushion";
            return false;
        }
        const std::vector<uint64_t> prefix(ranked_blob_bytes.begin(), ranked_blob_bytes.begin() + limit_slots);
        const SecondaryBudget plan = secondary_budget(before - kAllocationCushion, prefix, max_bytes,
                                                       free_floor_bytes);
        if (plan.slots == 0 || plan.bytes > std::numeric_limits<size_t>::max()) {
            err = "no ranked secondary expert slot fits while preserving the display reserve";
            return false;
        }
        std::vector<uint64_t> offsets((size_t) plan.slots + 1, 0);
        for (uint64_t i = 0; i < plan.slots; ++i)
            offsets[(size_t) i + 1] = offsets[(size_t) i] +
                ((ranked_blob_bytes[(size_t) i] + 255) & ~uint64_t{255});
        uint8_t* candidate = nullptr;
        const cudaError_t allocated = cudaMalloc((void**) &candidate, (size_t) plan.bytes);
        if (allocated != cudaSuccess) {
            err = std::string("secondary cudaMalloc: ") + cudaGetErrorString(allocated);
            if (allocated != cudaErrorMemoryAllocation) return false;
            limit_slots = plan.slots / 2;
            continue;
        }
        const cudaError_t touched = cudaMemset(candidate, 0, (size_t) plan.bytes);
        const cudaError_t synced = touched == cudaSuccess ? cudaDeviceSynchronize() : touched;
        if (synced != cudaSuccess) {
            err = std::string("secondary touch: ") + cudaGetErrorString(synced);
            release_candidate(candidate);
            return false;
        }
        uint64_t after = 0;
        if (!snapshot(after)) {
            release_candidate(candidate);
            return false;
        }
        if (after >= free_floor_bytes) {
            base_ = candidate;
            ordinal_ = ordinal;
            offsets_ = std::move(offsets);
            lower_free_after_ = after;
            free_floor_bytes_ = free_floor_bytes;
            err.clear();
            return true;
        }
        const uint64_t shortage = free_floor_bytes - after + kAllocationCushion;
        err = "secondary allocation breached display reserve after touch";
        if (!release_candidate(candidate)) return false;
        limit_slots = secondary_retry_slots(ranked_blob_bytes, plan.slots, shortage);
        err += "; retrying with fewer slots";
    }
    if (limit_slots == 0) err = "secondary allocation cannot preserve the display free floor";
    return false;
}

bool SecondaryArena::close(std::string* err) {
    if (base_ != nullptr) {
        int previous = -1;
        const cudaError_t current = cudaGetDevice(&previous);
        if (current != cudaSuccess) {
            if (err) *err = std::string("cudaGetDevice before release: ") + cudaGetErrorString(current);
            return false;
        }
        const cudaError_t selected = cudaSetDevice(ordinal_);
        if (selected != cudaSuccess) {
            if (err) *err = std::string("cudaSetDevice before release: ") + cudaGetErrorString(selected);
            return false;
        }
        const cudaError_t released = cudaFree(base_);
        const cudaError_t restored = cudaSetDevice(previous);
        if (released != cudaSuccess) {
            if (err) *err = std::string("cudaFree secondary arena: ") + cudaGetErrorString(released);
            if (err && restored != cudaSuccess)
                *err += std::string("; restore: ") + cudaGetErrorString(restored);
            return false;
        }
        base_ = nullptr;
        if (restored != cudaSuccess) {
            ordinal_ = -1;
            offsets_.clear();
            lower_free_after_ = 0;
            if (err) *err = std::string("cudaSetDevice after release: ") + cudaGetErrorString(restored);
            return false;
        }
    }
    base_ = nullptr;
    ordinal_ = -1;
    offsets_.clear();
    lower_free_after_ = 0;
    return true;
}

uint8_t* SecondaryArena::slot_ptr(uint64_t slot) const {
    return base_ != nullptr && slot < slots() ? base_ + offsets_[(size_t) slot] : nullptr;
}

bool SecondaryArena::fill_slot(uint64_t slot, const uint8_t* blob, uint64_t bytes, std::string& err) {
    if (blob == nullptr || !fits(slot, bytes)) {
        err = "secondary fill needs an open slot and a blob fitting that slot";
        return false;
    }
    int previous = -1;
    if (cudaGetDevice(&previous) != cudaSuccess || cudaSetDevice(ordinal_) != cudaSuccess) {
        err = "secondary fill cannot select its CUDA device";
        return false;
    }
    const RestoreDevice restore{previous};
    const cudaError_t copied = cudaMemcpy(slot_ptr(slot), blob, (size_t) bytes, cudaMemcpyHostToDevice);
    if (copied != cudaSuccess) {
        err = std::string("secondary fill copy: ") + cudaGetErrorString(copied);
        return false;
    }
    // Filling a slot allocates nothing (the arena was allocated whole at open), so the display card's free memory
    // can only move with other processes. Sampling it after every slot cost ~15 ms per expert (104 s for 6,602
    // experts, strata-claude-memtrace-E); every 256th slot and the last one keep the floor check.
    if (slot % 256 != 0 && slot + 1 != slots()) { err.clear(); return true; }
    uint64_t lower = 0;
    if (!free_snapshot(ordinal_, lower, err)) return false;
    if (lower < free_floor_bytes_) {
        err = "secondary fill crossed the display free floor";
        return false;
    }
    lower_free_after_ = lower;
    err.clear();
    return true;
}

bool SecondaryArena::check_free_floor(std::string& err) {
    // #208: free_snapshot's cudaMemGetInfo reads the CURRENT device; the caller (after the fill) has device 0 current
    int previous = -1;
    if (cudaGetDevice(&previous) != cudaSuccess || cudaSetDevice(ordinal_) != cudaSuccess) {
        err = "secondary free-floor check cannot select its CUDA device";
        return false;
    }
    const RestoreDevice restore{previous};
    uint64_t lower = 0;
    if (!free_snapshot(ordinal_, lower, err)) return false;
    if (lower < free_floor_bytes_) {
        err = "secondary fill crossed the display free floor";
        return false;
    }
    lower_free_after_ = lower;
    err.clear();
    return true;
}

bool SecondaryArena::verify_slot(uint64_t slot, const uint8_t* blob, uint64_t bytes, std::string& err) {
    if (blob == nullptr || !fits(slot, bytes)) {
        err = "secondary verify needs an open slot and a blob fitting that slot";
        return false;
    }
    int previous = -1;
    if (cudaGetDevice(&previous) != cudaSuccess || cudaSetDevice(ordinal_) != cudaSuccess) {
        err = "secondary verify cannot select its CUDA device";
        return false;
    }
    const RestoreDevice restore{previous};
    static thread_local std::vector<uint8_t> actual;   // reused: one allocation, not one per slot
    actual.resize((size_t) bytes);
    const cudaError_t copied = cudaMemcpy(actual.data(), slot_ptr(slot), (size_t) bytes, cudaMemcpyDeviceToHost);
    if (copied != cudaSuccess) {
        err = std::string("secondary verify copy: ") + cudaGetErrorString(copied);
        return false;
    }
    if (std::memcmp(actual.data(), blob, (size_t) bytes) != 0) {
        uint64_t i = 0;
        while (i < bytes && actual[(size_t) i] == blob[i]) ++i;
        err = "secondary slot " + std::to_string(slot) + " differs at byte " + std::to_string(i);
        return false;
    }
    err.clear();
    return true;
}

} // namespace strata::core
