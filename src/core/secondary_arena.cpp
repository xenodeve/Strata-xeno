#include "strata/core/secondary_arena.hpp"
#include "strata/core/secondary_budget.hpp"
#include "strata/core/secondary_vram.hpp"

#include <cuda_runtime.h>
#include <cstddef>
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
    const cudaError_t rc = cudaMemGetInfo(&cuda_free, &cuda_total);
    if (rc != cudaSuccess) {
        err = std::string("secondary cudaMemGetInfo: ") + cudaGetErrorString(rc);
        return false;
    }
    uint64_t nvml_free = 0;
    if (!secondary_nvml_free_bytes(ordinal, nvml_free, err)) return false;
    lower = secondary_effective_free((uint64_t) cuda_free, nvml_free);
    if (lower == 0) { err = "secondary free-memory query returned zero"; return false; }
    return true;
}

} // namespace

SecondaryArena::~SecondaryArena() { close(); }

bool SecondaryArena::open(int ordinal, const std::vector<uint64_t>& ranked_blob_bytes,
                          uint64_t max_bytes, std::string& err, FreeReader reader,
                          void* reader_context) {
    if (base_ != nullptr) { err = "secondary arena is already open"; return false; }
    if (ordinal != 1 || ranked_blob_bytes.empty() || max_bytes == 0) {
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
    uint64_t limit_slots = ranked_blob_bytes.size();
    for (int attempt = 0; attempt < 8 && limit_slots > 0; ++attempt) {
        uint64_t before = 0;
        if (!snapshot(before)) return false;
        if (before <= kSecondaryReserveBytes + kAllocationCushion) {
            err = "display GPU has no space above the 2560 MiB reserve and allocation cushion";
            return false;
        }
        const std::vector<uint64_t> prefix(ranked_blob_bytes.begin(), ranked_blob_bytes.begin() + limit_slots);
        const SecondaryBudget plan = secondary_budget(before - kAllocationCushion, prefix, max_bytes);
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
            const cudaError_t released = cudaFree(candidate);
            err = std::string("secondary touch: ") + cudaGetErrorString(synced);
            if (released != cudaSuccess) err += std::string("; release: ") + cudaGetErrorString(released);
            return false;
        }
        uint64_t after = 0;
        if (!snapshot(after)) {
            const cudaError_t released = cudaFree(candidate);
            if (released != cudaSuccess) err += std::string("; release: ") + cudaGetErrorString(released);
            return false;
        }
        if (after >= kSecondaryReserveBytes) {
            base_ = candidate;
            ordinal_ = ordinal;
            offsets_ = std::move(offsets);
            lower_free_after_ = after;
            return true;
        }
        const uint64_t shortage = kSecondaryReserveBytes - after + kAllocationCushion;
        const cudaError_t released = cudaFree(candidate);
        if (released != cudaSuccess) {
            err = std::string("secondary reserve breached and allocation release failed: ") + cudaGetErrorString(released);
            return false;
        }
        limit_slots = secondary_retry_slots(ranked_blob_bytes, plan.slots, shortage);
        err = "secondary allocation breached display reserve after touch; retrying with fewer slots";
    }
    if (limit_slots == 0) err = "secondary allocation cannot preserve the 2560 MiB display reserve";
    return false;
}

void SecondaryArena::close() {
    if (base_ != nullptr) {
        int previous = -1;
        if (cudaGetDevice(&previous) == cudaSuccess && cudaSetDevice(ordinal_) == cudaSuccess) {
            cudaFree(base_);
            cudaSetDevice(previous);
        }
    }
    base_ = nullptr;
    ordinal_ = -1;
    offsets_.clear();
    lower_free_after_ = 0;
}

uint8_t* SecondaryArena::slot_ptr(uint64_t slot) const {
    return base_ != nullptr && slot < slots() ? base_ + offsets_[(size_t) slot] : nullptr;
}

} // namespace strata::core
