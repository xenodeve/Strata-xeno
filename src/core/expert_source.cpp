// src/core/expert_source.cpp - the adapter.  See the header for the three clauses of the contract.
#include "strata/core/expert_source.hpp"
#include "strata/core/remote_experts.hpp"
#include "strata/timeline.hpp"
#include "strata/platform/direct_file.hpp"
#include "strata/core/secondary_runner.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"

#include "strata/core/pinned.hpp"
#include "strata/platform/memory.hpp"
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/quantize_act.hpp"
#include "strata/kernels/s2_expert_grouped.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <filesystem>
#include <sstream>
#include <limits>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX   // std::numeric_limits<T>::max() below
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace strata::core {

namespace detail {

bool cgroup_available_bytes(uint64_t limit, const CgroupMemoryStat& stat, uint64_t& bytes) {
    bytes = 0;
    if (!stat.valid) return false;

    // memory.stat's inactive_file can race memory.current, so bound it to charged usage first.
    uint64_t reclaimable = std::min(stat.inactive_file, stat.current);
    reclaimable = stat.file_dirty >= reclaimable ? 0 : reclaimable - stat.file_dirty;
    reclaimable = stat.file_writeback >= reclaimable ? 0 : reclaimable - stat.file_writeback;

    // Reclaiming clean file pages reduces usage; saturating subtraction also handles a transient over-limit read.
    const uint64_t usage_after_reclaim = stat.current - reclaimable;
    bytes = usage_after_reclaim < limit ? limit - usage_after_reclaim : 0;
    return true;
}

bool make_cache_complement_plan(
    int64_t n_layers, int64_t n_expert, const std::vector<uint64_t>& layer_blob_bytes,
    const std::vector<std::pair<int32_t, int32_t>>& primary_gpu_pairs,
    const std::vector<std::pair<int32_t, int32_t>>& additional_gpu_pairs,
    std::vector<uint64_t>& offsets, uint64_t& bytes, std::string& err) {
    offsets.clear();
    bytes = 0;
    err.clear();
    if (n_layers <= 0 || n_expert <= 0 || layer_blob_bytes.size() != (size_t) n_layers) {
        err = "FileExpertSource: invalid geometry for the cache complement plan";
        return false;
    }
    if ((uint64_t) n_layers > (uint64_t) std::numeric_limits<size_t>::max() / (uint64_t) n_expert) {
        err = "FileExpertSource: cache complement index table is too large";
        return false;
    }
    const size_t count = (size_t) n_layers * (size_t) n_expert;
    std::vector<uint8_t> omitted(count, 0);
    auto mark_pairs = [&](const std::vector<std::pair<int32_t, int32_t>>& pairs, uint8_t bit,
                          const char* label) -> bool {
        for (const auto& pair : pairs) {
            if (pair.first < 0 || pair.second < 0 || pair.first >= n_layers || pair.second >= n_expert) {
                err = std::string("FileExpertSource: ") + label + " pair is outside the expert geometry";
                return false;
            }
            const size_t index = (size_t) pair.first * (size_t) n_expert + (size_t) pair.second;
            if ((omitted[index] & bit) != 0) {
                err = std::string("FileExpertSource: duplicate ") + label + " pair in the cache complement plan";
                return false;
            }
            if (bit == 2 && (omitted[index] & 1) != 0) {
                err = "FileExpertSource: the primary and additional GPU expert tiers overlap";
                return false;
            }
            omitted[index] |= bit;
        }
        return true;
    };
    if (!mark_pairs(primary_gpu_pairs, 1, "primary GPU") ||
        !mark_pairs(additional_gpu_pairs, 2, "additional GPU")) return false;
    for (uint64_t blob_bytes : layer_blob_bytes) {
        if (blob_bytes == 0) {
            err = "FileExpertSource: cache complement layer has zero-sized expert blobs";
            return false;
        }
    }

    offsets.assign(count, kNoCacheComplement);
    for (int64_t layer = 0; layer < n_layers; ++layer) {
        const uint64_t blob_bytes = layer_blob_bytes[(size_t) layer];
        for (int64_t expert = 0; expert < n_expert; ++expert) {
            const size_t index = (size_t) layer * (size_t) n_expert + (size_t) expert;
            if (omitted[index] != 0) continue;
            if (bytes > std::numeric_limits<uint64_t>::max() - blob_bytes) {
                offsets.clear();
                bytes = 0;
                err = "FileExpertSource: cache complement size overflows";
                return false;
            }
            offsets[index] = bytes;
            bytes += blob_bytes;
        }
    }
    if (bytes > (uint64_t) std::numeric_limits<size_t>::max()) {
        offsets.clear();
        bytes = 0;
        err = "FileExpertSource: cache complement exceeds the host address space";
        return false;
    }
    return true;
}

const uint8_t* cache_complement_blob_or_fallback(
    size_t index, const std::vector<uint64_t>& offsets, const uint8_t* complement_host,
    const uint8_t* mapped_fallback) {
    if (complement_host != nullptr && index < offsets.size() && offsets[index] != kNoCacheComplement)
        return complement_host + (size_t) offsets[index];
    return mapped_fallback;
}

int64_t choose_resident_keep_from(const std::vector<uint64_t>& slot_bytes, uint64_t base_bytes, uint64_t budget,
                                  int64_t lend_from) {
    if (base_bytes > budget) return -1;
    const int64_t slots = (int64_t) slot_bytes.size();
    if (lend_from < 0 || lend_from > slots) lend_from = slots;   // no lend region: only the experts no slot holds
    int64_t keep = slots;
    uint64_t bytes = base_bytes;
    while (keep > lend_from) {
        const uint64_t b = slot_bytes[(size_t) keep - 1];
        if (b > budget - bytes) break;
        bytes += b;
        --keep;
    }
    return keep;
}

bool exchange_cache_complement(std::vector<uint64_t>& offsets, size_t in, size_t out) {
    if (in == out || in >= offsets.size() || out >= offsets.size() || offsets[in] == kNoCacheComplement ||
        offsets[out] != kNoCacheComplement) return false;
    offsets[out] = offsets[in];
    offsets[in] = kNoCacheComplement;
    return true;
}

}  // namespace detail

namespace {

#if defined(__linux__)
bool read_cgroup_memory_stat(const std::filesystem::path& path, uint64_t current,
                             detail::CgroupMemoryStat& stat) {
    std::ifstream input(path / "memory.stat");
    if (!input) return false;

    bool inactive_file = false, file_dirty = false, file_writeback = false;
    std::string line;
    while (std::getline(input, line)) {
        std::istringstream fields(line);
        std::string key;
        uint64_t value = 0;
        if (!(fields >> key >> value)) return false;
        fields >> std::ws;
        if (!fields.eof()) return false;

        if (key == "inactive_file") {
            if (inactive_file) return false;
            inactive_file = true;
            stat.inactive_file = value;
        } else if (key == "file_dirty") {
            if (file_dirty) return false;
            file_dirty = true;
            stat.file_dirty = value;
        } else if (key == "file_writeback") {
            if (file_writeback) return false;
            file_writeback = true;
            stat.file_writeback = value;
        }
    }
    if (!input.eof() || !inactive_file || !file_dirty || !file_writeback) return false;
    stat.current = current;
    stat.valid = true;
    return true;
}
#endif

bool available_memory_bytes(uint64_t& bytes) {
#if defined(_WIN32)
    MEMORYSTATUSEX status{};
    status.dwLength = sizeof(status);
    if (!GlobalMemoryStatusEx(&status)) return false;
    bytes = (uint64_t) status.ullAvailPhys;
    return bytes > 0;
#elif defined(__linux__)
    // MemAvailable includes reclaimable page cache, unlike _SC_AVPHYS_PAGES.
    std::ifstream info("/proc/meminfo");
    std::string line;
    bytes = 0;
    while (std::getline(info, line)) {
        std::istringstream fields(line);
        std::string key, unit;
        uint64_t value = 0;
        if (fields >> key >> value >> unit && key == "MemAvailable:" && unit == "kB" &&
            value <= std::numeric_limits<uint64_t>::max() / 1024) bytes = value * 1024;
    }
    if (bytes == 0) return false;
    // Account for the tightest cgroup-v2 ancestor limit when its normal mount is visible.
    // This is a point-in-time guard, not a reservation against concurrent allocations.
    std::ifstream groups("/proc/self/cgroup");
    if (!groups) return false;
    bool resolved_v2 = false;
    while (std::getline(groups, line)) {
        if (line.rfind("0::/", 0) != 0) continue;
        const std::filesystem::path root("/sys/fs/cgroup");
        auto path = (root / line.substr(4)).lexically_normal();
        if (path.string().rfind(root.string(), 0) != 0 || !std::filesystem::is_directory(path)) return false;
        resolved_v2 = true;
        while (path.string().rfind(root.string(), 0) == 0) {
            std::ifstream limit_file(path / "memory.max"), current_file(path / "memory.current");
            std::string limit;
            uint64_t current = 0;
            const bool readable = bool(limit_file >> limit) && bool(current_file >> current);
            // The host's root cgroup has no memory.max; ordinary child groups must expose their limits.
            if (!readable && !(path == root && !std::filesystem::exists(path / "memory.max") &&
                               std::filesystem::exists(path / "cgroup.controllers"))) return false;
            if (readable && limit != "max") {
                try {
                    size_t consumed = 0;
                    const uint64_t cap = std::stoull(limit, &consumed);
                    if (consumed != limit.size()) return false;
                    detail::CgroupMemoryStat stat;
                    if (!read_cgroup_memory_stat(path, current, stat)) return false;
                    uint64_t cgroup_available = 0;
                    if (!detail::cgroup_available_bytes(cap, stat, cgroup_available)) return false;
                    bytes = std::min(bytes, cgroup_available);
                } catch (...) { return false; }
            }
            if (path == root) break;
            path = path.parent_path();
        }
    }
    return resolved_v2;
#else
    const long pages = sysconf(_SC_AVPHYS_PAGES);
    const long page_bytes = sysconf(_SC_PAGESIZE);
    if (pages <= 0 || page_bytes <= 0 ||
        (uint64_t) pages > std::numeric_limits<uint64_t>::max() / (uint64_t) page_bytes) return false;
    bytes = (uint64_t) pages * (uint64_t) page_bytes;
    return bytes > 0;
#endif
}

}  // namespace

// ================================ THE FILE-BACKED SOURCE ================================

FileExpertSource::~FileExpertSource() { close(); }

bool FileExpertSource::open(const std::string& pack_dir, int64_t n_layers, int64_t n_expert, std::string& err) {
    close();
    if (n_layers <= 0 || n_expert <= 0) { err = "FileExpertSource: the geometry is empty"; return false; }
    const auto& layout = strata::kernels::cpu::expert_layout();
    if (layout.n_layers != n_layers || layout.n_expert != n_expert) {
        err = "FileExpertSource: the requested geometry does not match the loaded expert layout";
        return false;
    }
    if ((uint64_t) n_layers > (uint64_t) std::numeric_limits<int64_t>::max() / (uint64_t) n_expert) {
        err = "FileExpertSource: the expert count overflows";
        return false;
    }
    const uint64_t blob_count = (uint64_t) n_layers * (uint64_t) n_expert;
    if (blob_count > (uint64_t) std::numeric_limits<int64_t>::max() ||
        (uint64_t) n_layers > (uint64_t) std::numeric_limits<size_t>::max()) {
        err = "FileExpertSource: the expert count overflows";
        return false;
    }

    std::vector<uint64_t> layer_offsets((size_t) n_layers), layer_blob_bytes((size_t) n_layers);
    const uint64_t want = layout.total;
    if (want == 0 || want > (uint64_t) std::numeric_limits<size_t>::max()) {
        err = "FileExpertSource: the loaded expert layout has an invalid size";
        return false;
    }
    if (!layout.native) {
        if (blob_count > std::numeric_limits<uint64_t>::max() / (uint64_t) strata::kernels::cpu::BLOB) {
            err = "FileExpertSource: the canonical expert size overflows";
            return false;
        }
        const uint64_t canonical_size = blob_count * (uint64_t) strata::kernels::cpu::BLOB;
        if (want != canonical_size) {
            err = "FileExpertSource: the canonical expert layout has an inconsistent size";
            return false;
        }
        const uint64_t bytes = (uint64_t) strata::kernels::cpu::BLOB;
        const uint64_t layer_bytes = (uint64_t) n_expert * bytes;
        for (int64_t layer = 0; layer < n_layers; ++layer) {
            layer_offsets[(size_t) layer] = (uint64_t) layer * layer_bytes;
            layer_blob_bytes[(size_t) layer] = bytes;
        }
    } else {
        if (layout.offset.size() != (size_t) n_layers || layout.bytes.size() != (size_t) n_layers ||
            layout.fmt.size() != (size_t) n_layers) {
            err = "FileExpertSource: the native expert layout is incomplete";
            return false;
        }
        uint64_t at = 0;
        for (int64_t layer = 0; layer < n_layers; ++layer) {
            const size_t i = (size_t) layer;
            const uint64_t bytes = (uint64_t) layout.fmt[i].bytes;
            if (layout.offset[i] != at || bytes == 0 || layout.bytes[i] != bytes ||
                bytes > std::numeric_limits<uint64_t>::max() / (uint64_t) n_expert) {
                err = "FileExpertSource: the native expert layout is invalid at layer " + std::to_string(layer);
                return false;
            }
            const uint64_t layer_bytes = bytes * (uint64_t) n_expert;
            if (at > want || layer_bytes > want - at) {
                err = "FileExpertSource: the native expert layout exceeds its declared size at layer " +
                      std::to_string(layer);
                return false;
            }
            layer_offsets[i] = layout.offset[i];
            layer_blob_bytes[i] = bytes;
            at += layer_bytes;
        }
        if (at != want) {
            err = "FileExpertSource: the native expert layout has an inconsistent size";
            return false;
        }
    }
    const std::string path = pack_dir + "/experts.bin";

#if defined(_WIN32)
    // UTF-8 -> UTF-16: the pack may live under a path with non-ASCII characters, and `CreateFileA` would
    // silently mangle it into a file-not-found.
    const int wide = MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, nullptr, 0);
    std::vector<wchar_t> wpath((size_t) (wide > 0 ? wide : 1));
    if (wide > 0) MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, wpath.data(), wide);
    // **`FILE_FLAG_RANDOM_ACCESS` WAS HERE AND IT COST 14x.**
    //
    // The design depends on the OS page cache holding the whole 34 GB expert set, because this machine has
    // 64 GB of DDR5 and `L9` measured the CPU path at 44.14 GB/s from DRAM.  `FILE_FLAG_RANDOM_ACCESS` tells
    // the cache manager the opposite: it disables read-ahead AND it lets the manager drop the pages again
    // quickly, on the assumption that a large randomly-accessed file will not be re-read.  Measured, on
    // `strata generate --max-new 24`: **1.93 GB/s** - disk speed, 344 ms/token, and it never warmed up over 25
    // tokens, because the pages were being evicted as fast as they were faulted in.
    //
    // The correct flag is NO flag.  The access pattern IS random (10 of 512 experts per layer, a different 10
    // each layer), but every byte read is read again on the next token, so retention is the whole game.
    HANDLE f = CreateFileW(wpath.data(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                           FILE_ATTRIBUTE_NORMAL, nullptr);
    if (f == INVALID_HANDLE_VALUE) {
        err = "FileExpertSource: cannot open " + path;
        return false;
    }
    LARGE_INTEGER sz{};
    if (!GetFileSizeEx(f, &sz)) {
        CloseHandle(f);
        err = "FileExpertSource: cannot size " + path;
        return false;
    }
    if ((uint64_t) sz.QuadPart != want) {
        char buf[400];
        std::snprintf(buf, sizeof buf,
                      "FileExpertSource: %s is %llu B but the loaded expert layout requires %llu B - this is not "
                      "the pack this geometry came from",
                      path.c_str(), (unsigned long long) sz.QuadPart, (unsigned long long) want);
        CloseHandle(f);
        err = buf;
        return false;
    }
    HANDLE m = CreateFileMappingW(f, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (m == nullptr) {
        CloseHandle(f);
        err = "FileExpertSource: CreateFileMapping failed on " + path;
        return false;
    }
    void* view = MapViewOfFile(m, FILE_MAP_READ, 0, 0, 0);
    if (view == nullptr) {
        CloseHandle(m);
        CloseHandle(f);
        err = "FileExpertSource: MapViewOfFile failed on " + path;
        return false;
    }
    file_ = f;
    mapping_ = m;
    base_ = (const uint8_t*) view;
#else
    const int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) { err = "FileExpertSource: cannot open " + path; return false; }
    struct stat st{};
    if (fstat(fd, &st) != 0) { ::close(fd); err = "FileExpertSource: cannot stat " + path; return false; }
    if (st.st_size < 0 || (uint64_t) st.st_size != want) {
        char buf[400];
        std::snprintf(buf, sizeof buf,
                      "FileExpertSource: %s is %llu B but the loaded expert layout requires %llu B - this is not "
                      "the pack this geometry came from",
                      path.c_str(), (unsigned long long) (st.st_size < 0 ? 0 : st.st_size),
                      (unsigned long long) want);
        ::close(fd);
        err = buf;
        return false;
    }
    void* view = mmap(nullptr, (size_t) want, PROT_READ, MAP_SHARED, fd, 0);
    if (view == MAP_FAILED) { ::close(fd); err = "FileExpertSource: mmap failed on " + path; return false; }
    fd_ = fd;
    base_ = (const uint8_t*) view;
#endif
    blobs_ = (int64_t) blob_count;
    n_layers_ = n_layers;
    n_expert_ = n_expert;
    mapped_bytes_ = want;
    layer_offsets_ = std::move(layer_offsets);
    layer_blob_bytes_ = std::move(layer_blob_bytes);
    return true;
}

void FileExpertSource::close() {
    if (complement_arena_ != nullptr) {
        if (complement_pinned_) (void) cudaFreeHost(complement_arena_);
        else {
            if (complement_locked_ > 0) strata::platform::unlock_resident(complement_arena_, complement_locked_);
            std::free(complement_arena_);
        }
    }
    if (xstage_ != nullptr) {
        if (xstage_pinned_) (void) cudaFreeHost(xstage_);
        else std::free(xstage_);
    }
    xstage_ = nullptr;
    xstage_pinned_ = false;
    xstage_cap_ = 0;
    xstage_blob_ = 0;
    override_.clear();
    staged_.clear();
    exchanges_ = 0;
    file_reads_.store(0);
    complement_arena_ = nullptr;
    complement_host_ = nullptr;
    complement_device_ = nullptr;
    complement_bytes_ = 0;
    complement_offsets_.clear();
    complement_pinned_ = false;
    complement_ready_ = false;
    complement_locked_ = 0;
    complement_lent_slots_ = 0;
#if defined(_WIN32)
    if (base_ != nullptr) UnmapViewOfFile((LPCVOID) base_);
    if (mapping_ != nullptr) CloseHandle((HANDLE) mapping_);
    if (file_ != nullptr) CloseHandle((HANDLE) file_);
    mapping_ = nullptr;
    file_ = nullptr;
#else
    if (base_ != nullptr) munmap((void*) base_, (size_t) mapped_bytes_);
    if (fd_ >= 0) ::close(fd_);
    fd_ = -1;
#endif
    base_ = nullptr;
    blobs_ = 0;
    n_layers_ = 0;
    n_expert_ = 0;
    mapped_bytes_ = 0;
    layer_offsets_.clear();
    layer_blob_bytes_.clear();
    reads_ = 0;
}

const uint8_t* FileExpertSource::mapped_blob(int64_t layer, int64_t expert) const {
    if (base_ == nullptr || layer < 0 || expert < 0 || layer >= n_layers_ || expert >= n_expert_) return nullptr;
    const size_t i = (size_t) layer;
    if (i >= layer_offsets_.size() || i >= layer_blob_bytes_.size()) return nullptr;
    const uint64_t blob_bytes = layer_blob_bytes_[i];
    if (blob_bytes == 0 || (uint64_t) expert > std::numeric_limits<uint64_t>::max() / blob_bytes) return nullptr;
    const uint64_t expert_offset = (uint64_t) expert * blob_bytes;
    const uint64_t layer_offset = layer_offsets_[i];
    if (layer_offset > mapped_bytes_ || expert_offset > mapped_bytes_ - layer_offset) return nullptr;
    const uint64_t offset = layer_offset + expert_offset;
    if (blob_bytes > mapped_bytes_ - offset) return nullptr;
    return base_ + (size_t) offset;
}

bool FileExpertSource::pin_cache_complement(
    const ExpertCache& cache, std::string& err, bool pin,
    const std::vector<std::pair<int32_t, int32_t>>& additional_gpu_pairs, int64_t lend_from_slot,
    uint64_t headroom_bytes) {
    err.clear();
    if (base_ == nullptr) { err = "FileExpertSource: open the mapped experts before pinning a complement"; return false; }
    if (complement_ready_) { err = "FileExpertSource: the cache complement is already pinned"; return false; }
    if (!cache.valid()) { err = "FileExpertSource: the GPU expert cache is not open"; return false; }
    if (cache.fills() != cache.resident()) {
        err = "FileExpertSource: the GPU expert cache is not fully filled";
        return false;
    }
    const cudaError_t sync = cudaDeviceSynchronize();
    if (sync != cudaSuccess) {
        err = std::string("FileExpertSource: GPU expert cache is not ready: ") + cudaGetErrorString(sync);
        (void) cudaGetLastError();
        return false;
    }

    // The GPU cache's experts, and the bytes each slot's expert takes here (for the lend region below).
    const int64_t n_slots = cache.slots();
    std::vector<std::pair<int32_t, int32_t>> primary_gpu_pairs;
    std::vector<int32_t> pair_slot;
    std::vector<uint64_t> slot_bytes((size_t) std::max<int64_t>(n_slots, 0), 0);
    primary_gpu_pairs.reserve((size_t) cache.resident());
    pair_slot.reserve((size_t) cache.resident());
    for (int64_t layer = 0; layer < n_layers_; ++layer) {
        for (int64_t expert = 0; expert < n_expert_; ++expert) {
            const int32_t slot = cache.slot_of(layer, expert);
            if (slot == kNotResident) continue;
            primary_gpu_pairs.emplace_back((int32_t) layer, (int32_t) expert);
            pair_slot.push_back(slot);
            if (slot >= 0 && slot < n_slots) slot_bytes[(size_t) slot] = layer_blob_bytes_[(size_t) layer];
        }
    }
    std::vector<uint64_t> offsets;
    uint64_t bytes = 0;
    if (!detail::make_cache_complement_plan(n_layers_, n_expert_, layer_blob_bytes_, primary_gpu_pairs,
                                            additional_gpu_pairs, offsets, bytes, err)) return false;

    const bool lend = lend_from_slot >= 0 && lend_from_slot < n_slots && additional_gpu_pairs.empty();
    uint64_t budget = std::numeric_limits<uint64_t>::max();
    if (bytes > 0 || lend) {
        uint64_t physical = 0;
        if (!available_memory_bytes(physical)) {
            err = "FileExpertSource: cannot determine available RAM for the resident-memory safety check";
            return false;
        }
        budget = physical > headroom_bytes ? physical - headroom_bytes : 0;
        if (bytes > budget) {
            char message[320];
            std::snprintf(message, sizeof message,
                          "FileExpertSource: resident complement %.2f GiB exceeds available RAM (%.2f GiB) minus the "
                          "%.0f GiB safety headroom",
                          (double) bytes / 1073741824.0, (double) physical / 1073741824.0,
                          (double) headroom_bytes / 1073741824.0);
            err = message;
            return false;
        }
    }
    // The prompt path's lend region: its slots' experts are streamed from here during a prompt and copied back into
    // their slots after it, so the ones that fit are kept here too (from the last slot down: a short prompt lends
    // only the last few).  The rest keep the mapped-file fallback.
    int64_t keep_from = n_slots;
    if (lend) {
        keep_from = detail::choose_resident_keep_from(slot_bytes, bytes, budget, lend_from_slot);
        if (keep_from < 0) keep_from = n_slots;
        if (keep_from < n_slots) {
            std::vector<std::pair<int32_t, int32_t>> core;
            core.reserve(primary_gpu_pairs.size());
            for (size_t i = 0; i < primary_gpu_pairs.size(); ++i)
                if (pair_slot[i] < keep_from) core.push_back(primary_gpu_pairs[i]);
            if (!detail::make_cache_complement_plan(n_layers_, n_expert_, layer_blob_bytes_, core,
                                                    additional_gpu_pairs, offsets, bytes, err)) return false;
        }
    }

    void* arena = nullptr;
    const uint8_t* host = nullptr;
    const uint8_t* device = nullptr;
    bool pinned_ok = false;
    uint64_t locked = 0;
    std::string note;
    auto release = [&]() {
        if (arena == nullptr) return;
        if (pinned_ok) (void) cudaFreeHost(arena);
        else {
            if (locked > 0) strata::platform::unlock_resident(arena, locked);
            std::free(arena);
        }
        arena = nullptr;
    };
    if (bytes > 0) {
        std::fprintf(stderr, "FileExpertSource: allocating %.2f GiB %s cache complement\n",
                     (double) bytes / 1073741824.0, pin ? "page-locked" : "pageable resident");
        std::fflush(stderr);
        if (pin) {
            const cudaError_t allocated = cudaHostAlloc(&arena, (size_t) bytes,
                                                         cudaHostAllocMapped | cudaHostAllocPortable);
            if (allocated == cudaSuccess) {
                void* alias = nullptr;
                const cudaError_t aliased = cudaHostGetDevicePointer(&alias, arena, 0);
                if (aliased == cudaSuccess && alias != nullptr) {
                    device = (const uint8_t*) alias;
                    pinned_ok = true;
                    note = "page-locked and mapped";
                } else {
                    note = std::string("no device alias (") + cudaGetErrorString(aliased) + ")";
                    (void) cudaGetLastError();
                    (void) cudaFreeHost(arena);
                    arena = nullptr;
                }
            } else {
                // Refused (the driver's page-locked limit): the same bytes in ordinary memory, locked in the working
                // set instead, as the arena does - resident either way, only copied by the CPU instead of by DMA.
                note = std::string("page-locking refused (") + cudaGetErrorString(allocated) + ")";
                (void) cudaGetLastError();
                arena = nullptr;
            }
        }
        if (arena == nullptr) {
            arena = std::malloc((size_t) bytes);
            if (arena == nullptr) {
                err = "FileExpertSource: pageable resident complement allocation failed";
                return false;
            }
            if (pin) {
                const strata::platform::LockResult lr = strata::platform::lock_resident(arena, bytes);
                locked = lr.locked_bytes;
                note += (note.empty() ? "" : "; ") + lr.note;
            }
        }
        host = (const uint8_t*) arena;
    }

    // Copied layer by layer on a few threads: the page faults of the mapped file are the cost, and they overlap.
#if !defined(_WIN32)
    const long page_size = sysconf(_SC_PAGESIZE);
    if (page_size <= 0) {
        err = "FileExpertSource: cannot determine page size for mapped-page release";
        release();
        return false;
    }
#endif
    std::atomic<int64_t> next_layer{0}, layers_done{0};
    std::atomic<uint64_t> copied{0};
    std::atomic<bool> failed{false};
    std::mutex fail_mu;
    std::string fail_msg;
    auto fail = [&](const std::string& m) {
        std::lock_guard<std::mutex> lock(fail_mu);
        if (fail_msg.empty()) fail_msg = m;
        failed.store(true);
    };
    auto worker = [&]() {
        for (;;) {
            const int64_t layer = next_layer.fetch_add(1);
            if (layer >= n_layers_ || failed.load()) return;
            const uint64_t blob_bytes = layer_blob_bytes_[(size_t) layer];
            for (int64_t expert = 0; expert < n_expert_; ++expert) {
                const size_t index = (size_t) layer * (size_t) n_expert_ + (size_t) expert;
                const uint64_t offset = offsets[index];
                if (offset == kNoComplement) continue;
                const uint8_t* source = mapped_blob(layer, expert);
                if (source == nullptr || offset > bytes || blob_bytes > bytes - offset) {
                    fail("FileExpertSource: invalid blob bounds while building the cache complement");
                    return;
                }
                std::memcpy((uint8_t*) host + (size_t) offset, source, (size_t) blob_bytes);
                copied.fetch_add(blob_bytes);
            }
#if !defined(_WIN32)
            const uint64_t layer_offset = layer_offsets_[(size_t) layer];
            const uint64_t layer_bytes = blob_bytes * (uint64_t) n_expert_;
            const uint64_t layer_end = layer_offset + layer_bytes;
            const uint64_t page = (uint64_t) page_size;
            const uint64_t advice_start = layer_offset - layer_offset % page;
            const uint64_t end_remainder = layer_end % page;
            const uint64_t extra = end_remainder == 0 ? 0 : page - end_remainder;
            const uint64_t advice_end = extra > mapped_bytes_ - layer_end ? mapped_bytes_ : layer_end + extra;
            if (advice_end > advice_start &&
                madvise((void*) (base_ + (size_t) advice_start), (size_t) (advice_end - advice_start), MADV_DONTNEED) != 0) {
                fail("FileExpertSource: madvise could not release mapped expert layer " + std::to_string(layer));
                return;
            }
            if (posix_fadvise(fd_, (off_t) layer_offset, (off_t) layer_bytes, POSIX_FADV_DONTNEED) != 0) {
                fail("FileExpertSource: posix_fadvise could not release expert layer " + std::to_string(layer));
                return;
            }
#endif
            const int64_t done = layers_done.fetch_add(1) + 1;
            if (done % 8 == 0 || done == n_layers_)
                std::fprintf(stderr, "FileExpertSource: copied cache complement through layer %lld/%lld (%.2f GiB)\n",
                             (long long) done, (long long) n_layers_, (double) copied.load() / 1073741824.0);
        }
    };
    {
        const int threads = (int) std::max<int64_t>(1, std::min<int64_t>(6, n_layers_));
        std::vector<std::thread> pool;
        for (int i = 1; i < threads; ++i) pool.emplace_back(worker);
        worker();
        for (auto& t : pool) t.join();
    }
    std::fflush(stderr);
    if (failed.load()) {
        err = fail_msg;
        release();
        return false;
    }
#if defined(_WIN32)
    // The mapped pages this process touched (the GPU cache's fill and this copy) leave its working set for the
    // standby list: VirtualUnlock on pages that are not locked does exactly that (it then reports ERROR_NOT_LOCKED).
    (void) VirtualUnlock((LPVOID) base_, (SIZE_T) mapped_bytes_);
#endif

    complement_arena_ = arena;
    complement_host_ = host;
    complement_device_ = device;
    complement_bytes_ = bytes;
    complement_offsets_ = std::move(offsets);
    complement_pinned_ = pinned_ok && bytes > 0;
    complement_locked_ = locked;
    complement_lent_slots_ = lend ? n_slots - keep_from : 0;
    complement_ready_ = true;
    std::fprintf(stderr, "FileExpertSource: %s cache complement ready: resident %.2f GiB, pinned %.2f GiB%s%s\n",
                 complement_pinned_ ? "mapped pinned" : pin ? "locked resident" : "pageable resident",
                 (double) resident_bytes() / 1073741824.0, (double) pinned_bytes() / 1073741824.0,
                 note.empty() ? "" : "; ", note.c_str());
    if (lend)
        std::fprintf(stderr, "FileExpertSource: %lld of the prompt path's %lld lendable slots keep their experts in RAM "
                             "too%s\n", (long long) complement_lent_slots_, (long long) (n_slots - lend_from_slot),
                     complement_lent_slots_ < n_slots - lend_from_slot
                         ? " (the others are read from the file when lent: not enough RAM for them)" : "");
    if (!additional_gpu_pairs.empty()) {
        std::fprintf(stderr, "FileExpertSource: %zu verified additional-GPU experts remain on the mmap fallback\n",
                     additional_gpu_pairs.size());
    }
    std::fflush(stderr);
    return true;
}

bool FileExpertSource::has_resident(int64_t layer, int64_t expert) const {
    if (!complement_ready_ || layer < 0 || expert < 0 || layer >= n_layers_ || expert >= n_expert_) return false;
    const size_t index = (size_t) layer * (size_t) n_expert_ + (size_t) expert;
    return index < complement_offsets_.size() && complement_offsets_[index] != kNoComplement;
}

bool FileExpertSource::reserve_exchanges(int64_t n, std::string& err) {
    err.clear();
    if (n <= xstage_cap_) return true;
    if (!staged_.empty()) { err = "FileExpertSource: exchange buffers are in use"; return false; }
    uint64_t blob = 0;
    for (const uint64_t b : layer_blob_bytes_) blob = std::max(blob, b);
    if (blob == 0 || n <= 0) { err = "FileExpertSource: no expert geometry for the exchange buffers"; return false; }
    if (xstage_ != nullptr) {
        if (xstage_pinned_) (void) cudaFreeHost(xstage_);
        else std::free(xstage_);
        xstage_ = nullptr;
        xstage_cap_ = 0;
    }
    const size_t total = (size_t) n * (size_t) blob;
    void* p = nullptr;
    if (cudaHostAlloc(&p, total, cudaHostAllocDefault) == cudaSuccess && p != nullptr) {
        xstage_pinned_ = true;
    } else {
        (void) cudaGetLastError();
        p = std::malloc(total);
        xstage_pinned_ = false;
        if (p == nullptr) { err = "FileExpertSource: cannot allocate the exchange buffers"; return false; }
    }
    xstage_ = (uint8_t*) p;
    xstage_cap_ = n;
    xstage_blob_ = blob;
    return true;
}

uint8_t* FileExpertSource::exchange_buffer(int64_t q) const {
    if (xstage_ == nullptr || q < 0 || q >= xstage_cap_) return nullptr;
    return xstage_ + (size_t) q * (size_t) xstage_blob_;
}

bool FileExpertSource::stage_exchange(int64_t layer, int64_t in, int64_t out, int64_t q) {
    if (!has_resident(layer, in) || has_resident(layer, out) || exchange_buffer(q) == nullptr) return false;
    const size_t i_in = (size_t) layer * (size_t) n_expert_ + (size_t) in;
    const size_t i_out = (size_t) layer * (size_t) n_expert_ + (size_t) out;
    if (override_.empty()) override_.assign((size_t) blobs_, nullptr);
    if (override_[i_out] != nullptr) return false;
    for (const Exchange& x : staged_)
        if (x.in == i_in || x.q == q) return false;
    override_[i_out] = exchange_buffer(q);
    staged_.push_back({i_in, i_out, q, layer_blob_bytes_[(size_t) layer]});
    return true;
}

int64_t FileExpertSource::commit_exchanges() {
    int64_t n = 0;
    for (const Exchange& x : staged_) {
        const uint8_t* src = override_[x.out];
        const uint64_t at = complement_offsets_[x.in];
        if (src != nullptr && at != kNoComplement && at <= complement_bytes_ && x.bytes <= complement_bytes_ - at &&
            complement_host_ != nullptr) {
            std::memcpy((uint8_t*) complement_host_ + (size_t) at, src, (size_t) x.bytes);
            if (detail::exchange_cache_complement(complement_offsets_, x.in, x.out)) ++n;
        }
        override_[x.out] = nullptr;
    }
    staged_.clear();
    exchanges_ += n;
    return n;
}

const uint8_t* FileExpertSource::blob(int64_t layer, int64_t expert) {
    if (base_ == nullptr || layer < 0 || expert < 0 || layer >= n_layers_ || expert >= n_expert_) return nullptr;
    const size_t index = (size_t) layer * (size_t) n_expert_ + (size_t) expert;
    const uint8_t* mapped_fallback = mapped_blob(layer, expert);
    const uint8_t* result = mapped_fallback;
    if (complement_ready_) {
        result = detail::cache_complement_blob_or_fallback(index, complement_offsets_, complement_host_, nullptr);
        if (result == nullptr) {
            result = !override_.empty() && override_[index] != nullptr ? override_[index] : mapped_fallback;
            if (result == mapped_fallback && result != nullptr) file_reads_.fetch_add(1, std::memory_order_relaxed);
        }
    }
    if (result != nullptr) ++reads_;
    return result;
}

bool FileExpertSource::pinned(int64_t layer, int64_t expert) const {
    if (!complement_ready_ || !complement_pinned_ || complement_host_ == nullptr || layer < 0 || expert < 0 ||
        layer >= n_layers_ || expert >= n_expert_) return false;
    const size_t index = (size_t) layer * (size_t) n_expert_ + (size_t) expert;
    return index < complement_offsets_.size() && complement_offsets_[index] != kNoComplement;
}

const uint8_t* FileExpertSource::device_alias(int64_t layer, int64_t expert) const {
    if (!pinned(layer, expert) || complement_device_ == nullptr) return nullptr;
    const size_t index = (size_t) layer * (size_t) n_expert_ + (size_t) expert;
    return complement_device_ + (size_t) complement_offsets_[index];
}

bool FileExpertSource::pcie_layer(int64_t layer) const {
    if (complement_ready_ && complement_pinned_ && complement_device_ != nullptr)
        return layer >= 0 && layer < n_layers_;
    return device_alias(layer, 0) != nullptr;
}

// ================================ THE ADAPTER ================================

void expert_pool_dispatch(void* user, const float* x_f, const int32_t* ids, const float* weights, int64_t n_embd,
                          int64_t k, float* out) {
    (void) weights;   // clause 2: `moe_combine` applies it on the device.  Not an oversight.
    ExpertDispatch& d = *(ExpertDispatch*) user;
    if (d.failed) return;   // a previous layer already failed; do not make it worse

    using namespace strata::kernels::cpu;
    // Clause 3: the blob's internal offsets are compile-time constants, so a mismatched geometry does not
    // produce a wrong answer - it produces a walk off the end of the blob into the next expert's bytes, which
    // is finite and plausible.  Refuse, name the number, and let the driver report it.
    if (n_embd != H) {
        d.failed = true;
        d.fail = "the expert kernel is compiled for a 2560-wide activation";
        d.fail_layer = d.layers;
        return;
    }
    if (expert_layout().native) {
        // plan v0.3 P6: a native pack runs its experts in verify windows only (the driver guarantees it)
        d.failed = true;
        d.fail = "the single-token expert path does not take a native (IQ) pack";
        d.fail_layer = d.layers;
        return;
    }
    if (k > (int64_t) d.jobs.size()) d.jobs.resize((size_t) k);

    d.src->begin_layer(d.layers, ids, k);

    // Clause 1: rebuilt from `x_f` on EVERY call.  `x_f` is mapped pinned memory whose address never changes,
    // so anything cached against it would be layer 0's activation reused 48 times.
    act_quant_q8_1(x_f, H, d.act);

    // ---- R4.2c: THE POOL'S HALF OF THE SPLIT.  **IT DOES NOT DECIDE ANYTHING - `Launch` ALREADY DID.**
    //
    // The decision has to be made on THIS layer's ids, and `Launch` is the only callback that runs before the
    // pool while the ids are known (the doorbell publishes them when the ring fires).  So `expert_hit_run`
    // decides, and this consumes `d.is_hit`.  The first version decided here instead, which meant `Launch`
    // computed the PREVIOUS layer's experts into this layer's rows: C1 went from mean KL 9.69e-02 to 1.03e+00.
    //
    // `njobs` indexes the JOB ARRAY and `i` indexes the OUTPUT - they are the same only when nothing is a hit.
    const bool graph_hits = d.host_res != nullptr;
    const bool use_hits = graph_hits || (d.hits_ready() && d.decided);
    int64_t njobs = 0;

    if (d.remote_count > 0) {
        int32_t kind[32];
        if (k > 32) {
            d.failed = true; d.fail = "remote experts: routing width exceeds 32"; return;
        }
        for (int64_t i = 0; i < k; ++i)
            kind[i] = use_hits && ids[i] >= 0 && ids[i] < d.n_expert && (graph_hits
                ? d.host_res[(size_t) d.layers * (size_t) d.n_expert + (size_t) ids[i]] >= 0
                : d.is_hit[(size_t) i] != 0) ? 0 : -1;
        static thread_local std::string remote_error;
        for (int r = 0; r < d.remote_count; ++r)
            if (!d.remote[r]->begin(d.layers, x_f, ids, 1, k, kind, d.host_res, remote_error)) {
                d.failed = true; d.fail = remote_error.c_str(); d.fail_layer = d.layers; return;
            }
    }

    for (int64_t i = 0; i < k; ++i) {
        const int64_t e = ids[i];
        if (e < 0 || e >= d.n_expert) {
            d.failed = true;
            d.fail = "a routed expert id is out of range";
            d.fail_layer = d.layers;
            d.fail_expert = e;
            return;
        }
        const uint8_t* b = d.src->blob(d.layers, e);
        if (b == nullptr) { std::string ne; b = d.src->materialize(d.layers, e, d.layers, ne); }   // #11 NVMe tier
        if (b == nullptr) {
            // The one failure the loop cannot see.  Leaving `out` at its previous contents would feed the NEXT
            // layer a stale expert vector, which `moe_combine` would weight and add - the token would still be
            // finite and would still be wrong, 48 layers deep.
            d.failed = true;
            d.fail = "the expert source could not produce a blob";
            d.fail_layer = d.layers;
            d.fail_expert = e;
            ++d.missing;
            return;
        }
        // A hit's row was zeroed by `Launch` and belongs to the GPU; the pool must not touch it.
        if (use_hits && (graph_hits ? d.host_res[(size_t) d.layers * (size_t) d.n_expert + (size_t) e] >= 0
                                    : d.is_hit[(size_t) i] != 0)) {
            if (graph_hits) ++d.cache_hits;
            // The GPU owns this row and `hit_out` is zeroed, so the CPU's contribution is zero - but
            // `y_miss` is a REUSED pinned buffer, so the row must be written, not merely skipped.
            std::memset(out + (size_t) i * (size_t) n_embd, 0, (size_t) n_embd * sizeof(float));
            continue;
        }
        if (graph_hits) ++d.cache_refused;   // token graph: a miss (nothing is admitted during a token)

        bool remote_owns = false;
        for (int r = 0; r < d.remote_count; ++r) remote_owns |= d.remote[r]->owns(i);
        if (remote_owns) {
            std::memset(out + (size_t) i * (size_t) n_embd, 0, (size_t) n_embd * sizeof(float));
            continue;
        }

        // `njobs` indexes the JOB ARRAY and `i` indexes the OUTPUT - they are the same only when nothing is a
        // hit, and using one for the other is how a hit's row would get two experts summed into it.
        ExpertJob& j = d.jobs[(size_t) njobs++];
        j.blob = b;
        j.act = &d.act;             // SHARED across the batch: one conversion serves all ten experts
        j.out = out + (size_t) i * (size_t) n_embd;
        j.weight = 1.0f;            // clause 2: a diagnostic field, NOT the router weight
        j.slot = (int) i;
    }

    // Plan v0.3 P4: rows of every expert across all threads (bitwise the same as `run`).
    if (d.split_rows) d.pool->run_split(d.jobs.data(), (int) njobs);
    else d.pool->run(d.jobs.data(), (int) njobs);
    if (d.remote_count > 0) {
        static thread_local std::string remote_error;
        for (int r = 0; r < d.remote_count; ++r)
            if (!d.remote[r]->finish(out, remote_error)) {
                d.failed = true; d.fail = remote_error.c_str(); d.fail_layer = d.layers; return;
            }
    }
    ++d.layers;
    d.experts += k;
}

namespace {
// the verify window's per-entry tables in `expert_pool_dispatch_multi` (`kind`, `distinct`, `first_of`)
// are fixed arrays of this many entries: MAXT tokens of the model's 10 routed experts must fit, and a larger k is
// refused at run time rather than written past them.
constexpr int64_t kMaxWindowEntries = 128;
static_assert(strata::kernels::cpu::MAXT * 10 <= kMaxWindowEntries, "a verify window's entries overflow the tables");
}  // namespace

void expert_pool_dispatch_multi(ExpertDispatch& d, const float* x_f, const int32_t* ids, int64_t n_tok, int64_t k,
                                float* out) {
    using namespace strata::kernels::cpu;
    if (d.failed) return;
    if (n_tok < 1 || n_tok > MAXT) {
        d.failed = true;
        d.fail = "a verify window has more tokens than the multi-token expert kernel takes";
        d.fail_layer = d.layers;
        return;
    }
    if (k < 1 || n_tok * k > kMaxWindowEntries) {
        d.failed = true;
        d.fail = "a verify window routes more entries than the expert pool's window tables hold";
        d.fail_layer = d.layers;
        return;
    }
    if ((int64_t) d.act_multi.size() < n_tok) d.act_multi.resize((size_t) MAXT);
    const ExpertLayout& lay = expert_layout();
    const bool native = lay.native;
    if (native && d.nact_multi.size() < (size_t) MAXT * kNativeActBytes) d.nact_multi.resize((size_t) MAXT * kNativeActBytes);
    if (d.job_of.size() != (size_t) d.n_expert) d.job_of.assign((size_t) d.n_expert, (int16_t) -1);
    if (d.jobs_multi.size() < (size_t) (n_tok * k)) d.jobs_multi.resize((size_t) (MAXT * k));
    static const bool ptrace = std::getenv("STRATA_POOL_TRACE") != nullptr;
    auto pt = [&](const char* what, long long a = -1) {
        if (ptrace) { std::fprintf(stderr, "pool trace: layer %lld %s %lld\n", (long long) d.layers, what, a); std::fflush(stderr); }
    };
    const auto c0 = std::chrono::steady_clock::now();
    pt("begin");
    d.src->begin_layer(d.layers, ids, n_tok * k);
    pt("begun");
    if (d.route_trace != nullptr) {
        int16_t rec[3 + 128];
        const int64_t n = n_tok * k < 128 ? n_tok * k : 128;
        rec[0] = (int16_t) d.layers; rec[1] = (int16_t) n_tok; rec[2] = (int16_t) k;
        for (int64_t i = 0; i < n; ++i) rec[3 + i] = (int16_t) ids[i];
        std::fwrite(rec, sizeof(int16_t), (size_t) (3 + n), d.route_trace);
    }
    if (!d.usage.empty())
        for (int64_t i = 0; i < n_tok * k; ++i)
            if (ids[i] >= 0 && ids[i] < d.n_expert) d.usage[(size_t) d.layers * (size_t) d.n_expert + (size_t) ids[i]] += 1.0f;
    // ---- plan v0.3 P6: the GPU's share, decided and published FIRST so the GPU starts while the CPU works.
    // Distinct experts in routing order; resident ones and the last pcie_num/256 of the missed ones go to the GPU.
    const int64_t n = n_tok * k;
    int32_t kind[kMaxWindowEntries];       // per entry: -1 CPU, 0 VRAM, 1 PCIe
    if (d.plan != nullptr && n <= kMaxWindowEntries && n <= d.plan->cap) {
        int64_t distinct[kMaxWindowEntries], first_of[kMaxWindowEntries];
        int nd = 0, nmiss = 0;
        for (int64_t i = 0; i < n; ++i) {
            first_of[i] = i;
            for (int64_t j = 0; j < i; ++j)
                if (ids[j] == ids[i]) { first_of[i] = first_of[j]; break; }
            if (first_of[i] == i) {
                distinct[nd++] = i;
                const int32_t e = ids[i];
                if (e >= 0 && e < d.n_expert && d.host_res[(size_t) d.layers * (size_t) d.n_expert + (size_t) e] < 0) ++nmiss;
            }
        }
        const bool pcie_ok = d.pcie_num > 0 && d.src->pcie_layer(d.layers);
        const int m = pcie_ok ? (nmiss * d.pcie_num) >> 8 : 0;
        int miss_rank = 0, groups = 0, entries = 0, fetches = 0;
        GpuPlanSink& P = *d.plan;
        const uint8_t* dma_src[64];
        int64_t pcie_i0[64];
        for (int q = 0; q < nd; ++q) {
            const int64_t i0 = distinct[q];
            const int32_t e = ids[i0];
            int kd = -1;
            unsigned long long ptr = 0;
            if (e >= 0 && e < d.n_expert) {
                const int32_t slot = d.host_res[(size_t) d.layers * (size_t) d.n_expert + (size_t) e];
                if (slot >= 0) {
                    kd = 0;
                    ptr = (unsigned long long) (d.cache_base + (d.cache_slot_off ? (size_t) d.cache_slot_off[slot]
                                                                                 : (size_t) slot * (size_t) d.cache_blob));
                } else {
                    if (miss_rank >= nmiss - m && fetches < P.staging_cap && fetches < 64) {
                        const uint8_t* src = d.src->blob(d.layers, e);
                        if (src != nullptr && d.src->pinned(d.layers, e)) {
                            kd = 1;
                            dma_src[fetches] = src;
                            pcie_i0[fetches] = i0;
                            ++fetches;
                        }
                    }
                    ++miss_rank;
                }
            }
            for (int64_t i = i0; i < n; ++i)
                if (first_of[i] == i0) kind[i] = kd;
            if (kd != 0) continue;                 // the VRAM groups first; the PCIe groups below
            P.ptr[groups] = ptr;
            P.start[groups] = entries;
            for (int64_t i = i0; i < n; ++i)
                if (first_of[i] == i0) {
                    P.dst[entries] = (int32_t) i;
                    P.tok[entries] = (int32_t) (i / k);
                    ++entries;
                }
            ++groups;
        }
        P.start[groups] = entries;
        const uint64_t bb = lay.blob_bytes(d.layers);
        for (int q = 0; q < fetches; ++q) {       // the PCIe groups: staging slot q, entries after the VRAM ones
            const int64_t i0 = pcie_i0[q];
            P.ptr2[q] = P.pcie_mode != 0 ? (unsigned long long) d.src->device_alias(d.layers, ids[i0])
                                 : P.staging + (unsigned long long) q * (unsigned long long) bb;
            P.start2[q] = entries;
            for (int64_t i = i0; i < n; ++i)
                if (first_of[i] == i0) {
                    P.dst[entries] = (int32_t) i;
                    P.tok[entries] = (int32_t) (i / k);
                    ++entries;
                }
            ++d.pcie_experts;
        }
        P.start2[fetches] = entries;
        P.counts[0] = groups;
        P.counts[1] = entries;
        P.counts[2] = fetches;
        std::atomic_thread_fence(std::memory_order_seq_cst);
        pt("publish", fetches);
        if (P.publish) P.publish(P.ctx);
        pt("fetch", fetches);
        if (P.fetch) P.fetch(P.ctx, dma_src, P.pcie_mode != 0 ? 0 : fetches, (size_t) bb);   // the copy engine, beside the CPU's work
    } else {
        for (int64_t i = 0; i < n; ++i) {
            const int32_t e = ids[i];
            kind[i] = (e >= 0 && e < d.n_expert && d.host_res != nullptr &&
                       d.host_res[(size_t) d.layers * (size_t) d.n_expert + (size_t) e] >= 0) ? 0 : -1;
        }
    }
    if (d.remote_count > 0) {
        static thread_local std::string remote_error;
        for (int r = 0; r < d.remote_count; ++r) {
            if (!d.remote[r]->begin(d.layers, x_f, ids, n_tok, k, kind, d.host_res, remote_error)) {
                d.failed = true; d.fail = remote_error.c_str(); d.fail_layer = d.layers; return;
            }
            for (int64_t i = 0; i < n; ++i) if (d.remote[r]->owns(i)) kind[i] = 2;
        }
    }
    bool secondary_claims = false;
    int32_t secondary_slots[128];
    if (d.secondary_runner != nullptr) {
        if (!native || d.secondary_weights == nullptr || d.secondary_res == nullptr || n > 128) {
            d.failed = true;
            d.fail = "secondary expert dispatch is not configured for this verify window";
            d.fail_layer = d.layers;
            return;
        }
        for (int64_t i = 0; i < n; ++i) {
            secondary_slots[i] = -1;
            const int32_t e = ids[i];
            if (kind[i] < 0 && e >= 0 && e < d.n_expert) {
                const int32_t slot = d.secondary_res[(size_t) d.layers * (size_t) d.n_expert + (size_t) e];
                if (slot >= 0) {
                    secondary_slots[i] = slot;
                    kind[i] = 2;
                    secondary_claims = true;
                }
            }
        }
        if (secondary_claims) {
            const auto& f = lay.fmt[(size_t) d.layers];
            const auto L = strata::kernels::native_expert_layout(f.gu_type, f.d_type, f.n_embd, f.n_ff);
            std::string secondary_err;
            if (!d.secondary_runner->launch(L, *d.secondary_weights, x_f, secondary_slots,
                                             (int) n_tok, (int) k, secondary_err)) {
                d.failed = true;
                d.secondary_fail = "secondary expert launch: " + secondary_err;
                d.fail = d.secondary_fail.c_str();
                d.fail_layer = d.layers;
                return;
            }
        }
    }
    const auto c1 = std::chrono::steady_clock::now();
    if (native && lay.fmt[(size_t) d.layers].gu_type == 42)   // a native Q2_0 pack: the Q2_0 kernels' activations
        for (int64_t t = 0; t < n_tok; ++t) act_quant_any(x_f + (size_t) t * H, H, d.act_multi[(size_t) t]);
    else if (native)
        for (int64_t t = 0; t < n_tok; ++t)
            native_quant_act(lay.fmt[(size_t) d.layers], x_f + (size_t) t * H, d.nact_multi.data() + (size_t) t * kNativeActBytes);
    else
        for (int64_t t = 0; t < n_tok; ++t) act_quant_q8_1(x_f + (size_t) t * H, H, d.act_multi[(size_t) t]);
    const auto c2 = std::chrono::steady_clock::now();
    {   // #11 NVMe tier: this layer's host misses are read together (overlapped), one wait for the layer
        int32_t miss[128];
        int nm = 0;
        for (int64_t i = 0; i < n_tok * k && nm < 128; ++i) {
            const int64_t e = ids[i];
            if (kind[i] >= 0 || e < 0 || e >= d.n_expert || d.src->resident(d.layers, e)) continue;
            bool dup = false;
            for (int q = 0; q < nm; ++q) dup |= miss[q] == (int32_t) e;
            if (!dup) miss[nm++] = (int32_t) e;
        }
        if (nm > 0) {
            std::string ne;
            if (!d.src->materialize_batch(d.layers, miss, nm, ne)) {
                d.failed = true;
                d.fail = "an NVMe-tier expert could not be read";
                d.fail_layer = d.layers;
                return;
            }
        }
    }
    int njobs = 0;
    for (int64_t t = 0; t < n_tok; ++t)
        for (int64_t j = 0; j < k; ++j) {
            const int64_t i = t * k + j;
            const int64_t e = ids[i];
            float* row = out + (size_t) i * H;
            if (e < 0 || e >= d.n_expert) {
                d.failed = true;
                d.fail = "a routed expert id is out of range";
                d.fail_layer = d.layers;
                d.fail_expert = e;
                return;
            }
            if (kind[i] >= 0) {             // CUDA0, PCIe, or a remote result staged into this row below
                if (kind[i] == 0) ++d.cache_hits;
                ++d.tier_entries[kind[i] == 0 ? 0 : kind[i] == 2 ? 1 : 2];
                std::memset(row, 0, (size_t) H * sizeof(float));
                continue;
            }
            ++d.cache_refused;
            int16_t& jo = d.job_of[(size_t) e];
            if (jo < 0) {
                const uint8_t* b = d.src->blob(d.layers, e);
        if (b == nullptr) { std::string ne; b = d.src->materialize(d.layers, e, d.layers, ne); }   // #11 NVMe tier
                if (b == nullptr) {
                    d.failed = true;
                    d.fail = "the expert source could not produce a blob";
                    d.fail_layer = d.layers;
                    d.fail_expert = e;
                    ++d.missing;
                    return;
                }
                jo = (int16_t) njobs++;
                ExpertJobMulti& nj = d.jobs_multi[(size_t) jo];
                nj.blob = b;
                nj.nt = 0;
            }
            ExpertJobMulti& jb = d.jobs_multi[(size_t) jo];
            jb.act[jb.nt] = &d.act_multi[(size_t) t];
            jb.nact[jb.nt] = native ? d.nact_multi.data() + (size_t) t * kNativeActBytes : nullptr;
            jb.out[jb.nt] = row;
            ++jb.nt;
            ++d.multi_entries;
            ++d.tier_entries[3];
        }
    const auto c3 = std::chrono::steady_clock::now();
    pt("run", njobs);
    const auto pool_start = std::chrono::steady_clock::now();
    if (native) d.pool->run_split_multi_native(lay.fmt[(size_t) d.layers], d.jobs_multi.data(), njobs);
    else d.pool->run_split_multi(d.jobs_multi.data(), njobs);
    const auto pool_end = std::chrono::steady_clock::now();
    if (d.remote_count > 0) {
        static thread_local std::string remote_error;
        for (int r = 0; r < d.remote_count; ++r)
            if (!d.remote[r]->finish(out, remote_error)) {
                d.failed = true; d.fail = remote_error.c_str(); d.fail_layer = d.layers; return;
            }
    }
    double secondary_finish_ms = 0;
    if (secondary_claims) {
        std::string secondary_err;
        const auto finish_start = std::chrono::steady_clock::now();
        const bool finished = d.secondary_runner->finish(out, secondary_err);
        secondary_finish_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - finish_start).count();
        if (!finished) {
            d.failed = true;
            d.secondary_fail = "secondary expert completion: " + secondary_err;
            d.fail = d.secondary_fail.c_str();
            d.fail_layer = d.layers;
            return;
        }
        d.secondary_entries = (int64_t) d.secondary_runner->served_entries();
        d.secondary_groups = (int64_t) d.secondary_runner->served_groups();
    }
    const auto c4 = std::chrono::steady_clock::now();
    pt("ran");
    auto ms = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
    d.ms_plan += ms(c0, c1);
    d.ms_actq += ms(c1, c2);
    d.ms_jobs += ms(c2, c3);
    d.ms_run += ms(c3, c4);
    d.ms_cpu_pool += ms(pool_start, pool_end);
    d.ms_secondary_finish += secondary_finish_ms;
    if (timeline::enabled()) {   // #33: the layer's dispatch stages (inside the verify window's "cpu experts")
        timeline::complete("dispatch plan", c0, c1, d.layers, n_tok * k);
        timeline::complete("act quantize", c1, c2, d.layers, n_tok);
        timeline::complete("dispatch jobs", c2, c3, d.layers, njobs);
        timeline::complete("cpu pool", pool_start, pool_end, d.layers, njobs);
        if (secondary_claims)
            timeline::complete("4070 finish", pool_end, c4, d.layers, (int64_t) d.secondary_runner->served_entries());
    }
    for (int64_t i = 0; i < n_tok * k; ++i) {
        const int64_t e = ids[i];
        if (e >= 0 && e < d.n_expert) d.job_of[(size_t) e] = -1;
    }
    d.multi_misses += njobs;
    ++d.layers;
    d.experts += n_tok * k;
}

void expert_hit_run(void* user, void* stream, HitPhase phase, const int32_t* ids, int64_t k) {
    ExpertDispatch& d = *(ExpertDispatch*) user;
    if (d.failed) return;
    cudaStream_t cs = (cudaStream_t) stream;

    if (phase == HitPhase::Launch) {
        d.decided = false;
        d.hit_pending = false;
        if (!d.hits_ready() || ids == nullptr || k <= 0) return;
        if ((int64_t) d.is_hit.size() < k) d.is_hit.resize((size_t) k);

        // ================================ THE DECISION, ONCE, ON THIS LAYER'S IDS ================================
        //
        // Every routed expert is asked of the cache.  Resident -> the GPU computes it.  Not resident -> it is
        // admitted and filled if there is room (which makes it a hit on THIS call, because the fill and the
        // kernel are on one stream in that order), and otherwise it stays a miss for the CPU.
        d.n_hits = 0;
        for (int64_t i = 0; i < k; ++i) {
            const int64_t e = ids[i];
            d.is_hit[(size_t) i] = 0;
            if (e < 0 || e >= d.n_expert) continue;   // out of range: the pool refuses it, with a message
            int32_t slot = d.cache->slot_of(d.layers, e);
            if (slot == kNotResident) {
                const int32_t cand = d.cache->admit(d.layers, e);
                if (cand == kNotResident) {
                    ++d.cache_refused;
                    continue;
                }
                // `blob` is asked ONLY for an expert about to be filled, so the source's read counter stays a
                // count of distinct experts moved rather than of looks.
                const uint8_t* b = d.src->blob(d.layers, e);
        if (b == nullptr) { std::string ne; b = d.src->materialize(d.layers, e, d.layers, ne); }   // #11 NVMe tier
                std::string ferr;
                if (b == nullptr || !d.cache->fill_slot(cand, b, cs, ferr, (int64_t) strata::kernels::cpu::expert_layout().blob_bytes(d.layers))) {
                    d.failed = true;
                    d.fail = "the expert cache could not fill a slot";
                    d.fail_layer = d.layers;
                    d.fail_expert = e;
                    return;
                }
                ++d.cache_admitted;
                slot = cand;
            } else {
                ++d.cache_hits;
            }
            d.is_hit[(size_t) i] = 1;
            d.h_slot[(size_t) d.n_hits] = slot;
            d.h_dst[(size_t) d.n_hits] = (int32_t) i;
            ++d.n_hits;
        }
        d.decided = true;
        if (d.n_hits <= 0) return;   // nothing resident yet: no GPU work, and nothing for `Combine` to add

        const size_t list_bytes = (size_t) d.n_hits * sizeof(int32_t);
        // `hit_out` is ZEROED rather than overwritten: the kernel writes only the rows this layer's hits own,
        // so a row that was a hit last layer and a miss this one would still hold last layer's expert and
        // `add_inplace` would sum it in.  Finite, plausible, wrong.
        if (cudaMemsetAsync(d.hit_out, 0, (size_t) d.parts_elems * sizeof(float), cs) != cudaSuccess ||
            cudaMemcpyAsync(d.d_slot, d.h_slot.data(), list_bytes, cudaMemcpyHostToDevice, cs) != cudaSuccess ||
            cudaMemcpyAsync(d.d_dst, d.h_dst.data(), list_bytes, cudaMemcpyHostToDevice, cs) != cudaSuccess) {
            d.hit_fail = "the hit list could not be staged";
            d.failed = true;
            d.fail = d.hit_fail;
            return;
        }
        // The activation is quantized HERE rather than reused from `s.moe.x_q8_0`, which `post[l-1]` wrote from
        // the PREVIOUS layer's `mixed`.  `pre[l]` has since overwritten `mixed`, so that buffer is a layer stale
        // - and a stale activation produces a perfectly finite expert for the wrong input.
        // **R4.2h: THE SCALED QUANTIZER, SO A HIT REPRODUCES A MISS.**  The CPU pool quantizes this same
        // activation with `act_quant_q8_1` and multiplies by the fp32 `ActQ::scale`; `quantize_q8_0` writes
        // an fp16 `d` instead, and `bench/micro/act_quant_parity.cu` measured **80 of 80 chunks differing by
        // up to 4.761e-04 relative**.  `quantize_q8_0_scaled` adopts the CPU's rule and scale, and the kernel
        // takes the fp32 array.  Falling back to the old path would silently reintroduce the divergence, so
        // the scales are required here rather than optional.
        if (d.x_q8_0_hit_scale == nullptr) {
            d.failed = true;
            d.fail = "the hit path has no fp32 activation scales (R4.2h)";
            return;
        }
        strata::kernels::quantize_q8_0_scaled(d.mixed, d.x_q8_0_hit, d.x_q8_0_hit_scale, strata::kernels::cpu::H,
                                              cs);
        if (d.hit_cpu_order)
            strata::kernels::moe_hit_grouped_s2_cpu_order(d.cache_base, d.d_slot, d.d_dst, d.n_hits,
                d.cache_blob, d.x_q8_0_hit, d.hit_scratch, d.hit_out, cs, d.x_q8_0_hit_scale);
        else
            strata::kernels::moe_hit_grouped_s2(d.cache_base, d.d_slot, d.d_dst, d.n_hits, d.cache_blob,
                d.x_q8_0_hit, d.hit_scratch, d.hit_out, cs, d.x_q8_0_hit_scale);
        d.hit_pending = true;
        if (d.hit_done != nullptr) cudaEventRecord((cudaEvent_t) d.hit_done, cs);
        // The A/B arm: ONE driver entry here, and nothing else changes.  If the work was waiting for the host
        // to enter the driver, this is what lets it start while the pool runs.
        if (d.hit_poke && d.hit_done != nullptr) (void) cudaEventQuery((cudaEvent_t) d.hit_done);
        return;
    }

    // Combine: `parts += hit_out`, stream-ordered after the misses were copied into `parts`.
    if (!d.hit_pending) return;
    d.hit_pending = false;
    // Did the GPU get the hit work done while the CPU was in the pool?  This query is itself a driver entry,
    // so it is the LAST chance to observe a late start: a NOT-READY here means the work had not finished by the
    // time the pool returned, and with no poke in front of it that can only be because it began after.
    if (d.hit_done != nullptr) {
        if (cudaEventQuery((cudaEvent_t) d.hit_done) == cudaSuccess) ++d.hit_ready;
        else ++d.hit_late;
    }
    strata::kernels::add_inplace(d.parts_out, d.hit_out, d.parts_elems, cs);
}

// ================================ THE RESIDENT ARENA (R2.1) ================================

namespace {

void hash_u64(uint64_t& h, uint64_t v) {
    h = fnv1a64((const uint8_t*) &v, sizeof v, h);
}

void hash_text(uint64_t& h, const std::string& s) {
    h = fnv1a64((const uint8_t*) s.data(), (uint64_t) s.size(), h);
}

bool hash_small_file(const std::filesystem::path& path, uint64_t& h, std::string& err) {
    hash_text(h, path.filename().string());
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        hash_u64(h, 0);
        return true;
    }
    hash_u64(h, 1);
    std::vector<uint8_t> buf(64u << 10);
    for (;;) {
        f.read((char*) buf.data(), (std::streamsize) buf.size());
        const std::streamsize n = f.gcount();
        if (n > 0) h = fnv1a64(buf.data(), (uint64_t) n, h);
        if (f.eof()) break;
        if (!f) {
            err = "ArenaExpertSource: cannot hash pack metadata " + path.string();
            return false;
        }
    }
    return true;
}

bool hash_sampled_file(const std::filesystem::path& path, uint64_t& h, std::string& err) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) {
        err = "ArenaExpertSource: cannot sample pack source " + path.string();
        return false;
    }
    const std::streamoff end = f.tellg();
    if (end < 0) {
        err = "ArenaExpertSource: cannot size pack source " + path.string();
        return false;
    }
    const uint64_t bytes = (uint64_t) end;
    hash_text(h, path.filename().string());
    hash_u64(h, bytes);
    constexpr uint64_t sample = 64u << 10;
    const uint64_t starts[3] = {0, bytes / 2, bytes > sample ? bytes - sample : 0};
    std::vector<uint8_t> buf((size_t) std::min<uint64_t>(sample, bytes));
    for (uint64_t off : starts) {
        if (buf.empty()) break;
        const uint64_t at = std::min<uint64_t>(off, bytes - (uint64_t) buf.size());
        f.clear();
        f.seekg((std::streamoff) at);
        f.read((char*) buf.data(), (std::streamsize) buf.size());
        if ((size_t) f.gcount() != buf.size()) {
            err = "ArenaExpertSource: short read while hashing pack source " + path.string();
            return false;
        }
        hash_u64(h, at);
        h = fnv1a64(buf.data(), (uint64_t) buf.size(), h);
    }
    return true;
}

bool shared_arena_pack_hash(const std::string& pack_dir, const std::string& experts_path,
                            const std::string& gguf, const strata::kernels::cpu::ExpertLayout& lay,
                            uint64_t& out, std::string& err) {
    uint64_t h = 1469598103934665603ull;
    hash_text(h, "strata-shared-expert-arena-pack-v1");
    hash_u64(h, (uint64_t) lay.n_layers);
    hash_u64(h, (uint64_t) lay.n_expert);
    hash_u64(h, lay.total);
    hash_u64(h, lay.max_blob);
    hash_u64(h, lay.native ? 1 : 0);

    const std::filesystem::path pack(pack_dir);
    for (const char* name : {"manifest.json", "index.txt", "native_experts.txt"}) {
        if (!hash_small_file(pack / name, h, err)) return false;
    }

    if (std::filesystem::exists(experts_path)) {
        if (!hash_sampled_file(experts_path, h, err)) return false;
    } else if (!gguf.empty()) {
        // Native packs may read experts straight from one or more GGUF shards.  Sample every distinct source
        // file named by native_experts.txt; this keeps the fingerprint cheap while still tying it to the model
        // bytes rather than only to an equal-size layout.
        const std::filesystem::path first(gguf);
        std::vector<std::filesystem::path> sources{first};
        for (const std::string& name : lay.gguf_file) {
            if (name.empty()) continue;
            const std::filesystem::path p = first.parent_path() / name;
            if (std::find(sources.begin(), sources.end(), p) == sources.end()) sources.push_back(p);
        }
        for (const auto& p : sources) {
            if (!hash_sampled_file(p, h, err)) return false;
        }
    }

    out = h == 0 ? 1 : h;
    return true;
}

}  // namespace

// Plan v0.3 P6: the arena from the model's shard 1.  Each layer's gate, up and down tensors hold the 512 experts
// one after another; they are read in chunks and each expert's slice lands at its place in the blob
// [gate rows | up rows | down rows] - the layout tools/iq_pack.py would have written to experts.bin.
LoadStats load_experts_gguf(const std::string& gguf, uint8_t* dst, const strata::kernels::cpu::ExpertLayout& lay,
                            int threads, const uint8_t* skip = nullptr) {
    LoadStats st;
    st.layers = (uint64_t) lay.n_layers;
    const auto t0 = std::chrono::steady_clock::now();
    std::atomic<int64_t> next{0};
    std::atomic<bool> bad{false};
    // a layer's experts may sit in another shard of the model (native_experts.txt v3): a name beside `gguf`
    const size_t cut = gguf.find_last_of("/\\");
    const std::string dir = cut == std::string::npos ? std::string() : gguf.substr(0, cut + 1);
    auto file_of = [&](int64_t l) -> std::string {
        if (lay.gguf_file.empty() || lay.gguf_file[(size_t) l].empty()) return gguf;
        return dir + lay.gguf_file[(size_t) l];
    };
    auto worker = [&]() {
        std::ifstream f;
        std::string open_name;
        std::vector<uint8_t> buf;
        for (;;) {
            const int64_t l = next.fetch_add(1);
            if (l >= lay.n_layers || bad) break;
            const std::string name = file_of(l);
            if (name != open_name) {
                f.close();
                f.clear();
                f.open(name, std::ios::binary);
                if (!f) { bad = true; return; }
                open_name = name;
            }
            const auto& fm = lay.fmt[(size_t) l];
            const uint64_t blob = lay.bytes[(size_t) l];
            const uint64_t per[3] = {fm.up_off, fm.up_off, blob - fm.down_off};
            const uint64_t at[3] = {0, fm.up_off, fm.down_off};
            for (int r = 0; r < 3; ++r) {
                const uint64_t src = lay.gguf_off[(size_t) (3 * l + r)];
                const uint64_t total = per[r] * (uint64_t) lay.n_expert;
                const uint64_t chunk = per[r] * 16;           // 16 experts per read
                buf.resize((size_t) chunk);
                for (uint64_t done = 0; done < total; done += chunk) {
                    const uint64_t n = std::min<uint64_t>(chunk, total - done);
                    f.seekg((std::streamoff) (src + done));
                    f.read((char*) buf.data(), (std::streamsize) n);
                    if ((uint64_t) f.gcount() != n) { bad = true; return; }
                    for (uint64_t k = 0; k < n / per[r]; ++k) {
                        const uint64_t e = done / per[r] + k;
                        // a GPU-owned expert's pages stay untouched (placement-first cold start)
                        if (skip != nullptr && skip[(size_t) (l * lay.n_expert + (int64_t) e)]) continue;
                        std::memcpy(dst + lay.blob_offset(l, (int64_t) e) + at[r], buf.data() + k * per[r], (size_t) per[r]);
                    }
                }
            }
        }
    };
    std::vector<std::thread> pool;
    for (int i = 1; i < threads; ++i) pool.emplace_back(worker);
    worker();
    for (auto& t : pool) t.join();
    if (bad) {
        st.seconds = -1.0;
        st.ok = false;
        st.error = "short read or unreadable shard while reading the experts from the GGUF";
        return st;
    }
    st.bytes = lay.total;
    st.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return st;
}

ArenaExpertSource::~ArenaExpertSource() { close(); }

bool ArenaExpertSource::open(const std::string& pack_dir, int64_t n_layers, int64_t n_expert, int threads,
                             std::string& err, bool pin_for_cuda, bool defer_load, uint64_t max_pinned_bytes,
                             const std::string& shared_arena_file) {
    close();
    const std::string path = pack_dir + "/experts.bin";
    // plan v0.3 P6: the layout (canonical, or a native pack's per-layer blobs) was loaded by the driver
    const strata::kernels::cpu::ExpertLayout& lay = strata::kernels::cpu::expert_layout();
    if (lay.n_layers != n_layers || lay.n_expert != n_expert) {
        err = "ArenaExpertSource: the expert layout was loaded for a different geometry";
        return false;
    }
    const int64_t blob = (int64_t) lay.max_blob;
    const uint64_t want = lay.total;

    // plan v0.3 P6: no experts.bin in a native pack -> the experts come straight from the GGUF
    const bool from_gguf = !std::ifstream(path, std::ios::binary) && lay.native && !lay.gguf_off.empty() && !gguf_.empty();
    // SIZE CHECK BEFORE THE ALLOCATION, not after.  A wrong pack should name the two numbers rather than spend
    // 34 GB and a minute of loading first.
    if (!from_gguf) {
        std::ifstream f(path, std::ios::binary | std::ios::ate);
        if (!f) { err = "ArenaExpertSource: cannot open " + path; return false; }
        const uint64_t got = (uint64_t) f.tellg();
        if (got != want) {
            char buf[400];
            std::snprintf(buf, sizeof buf,
                          "ArenaExpertSource: %s is %llu B but %lld layers x %lld experts (blobs up to %lld B) "
                          "make %llu B - this is not the pack this geometry came from",
                          path.c_str(), (unsigned long long) got, (long long) n_layers, (long long) n_expert,
                          (long long) blob, (unsigned long long) want);
            err = buf;
            return false;
        }
    }

    uint64_t pack_hash = 0;
    if (!shared_arena_file.empty() &&
        !shared_arena_pack_hash(pack_dir, path, gguf_, lay, pack_hash, err)) return false;

    // one layer per registration slice, so no expert straddles two registrations.  The arena is one blob
    // longer than the file: a copy of a whole VRAM slot (the largest blob) may then start at any expert.
    std::vector<uint64_t> bounds, loff, lbytes;
    for (int64_t l = 0; l < n_layers; ++l) {
        bounds.push_back(lay.layer_offset(l));
        loff.push_back(lay.layer_offset(l));
        lbytes.push_back(lay.blob_bytes(l) * (uint64_t) n_expert);
    }
    bounds.push_back(want);
    PinnedArena* a = defer_load ? PinnedArena::reserve_only(want + (uint64_t) blob)
                                : new PinnedArena(want + (uint64_t) blob, bounds, pin_for_cuda, max_pinned_bytes,
                                                  shared_arena_file, pack_hash);
    if (!a->valid()) {
        const std::string why = a->note;
        delete a;
        err = "ArenaExpertSource: the arena could not be reserved (" +
              std::to_string(want + (uint64_t) blob) + " B)" +
              (why.empty() ? std::string{} : ": " + why);
        return false;
    }
    path_ = path;
    from_gguf_ = from_gguf;
    deferred_ = defer_load;
    LoadStats st;
    if (defer_load) st.bytes = want;   // nothing read yet: load_rest reads what the host keeps
    else st = from_gguf ? load_experts_gguf(gguf_, a->data(), lay, threads)
                        : load_experts_ranges(path, a->data(), loff, lbytes, threads, /*chunk=*/8u << 20);
    if (!st.ok) {
        delete a;
        err = "ArenaExpertSource: the expert load was refused: " + (st.error.empty() ? std::string("unknown") : st.error);
        return false;
    }
    if (st.bytes != want) {
        delete a;
        err = "ArenaExpertSource: the load read " + std::to_string(st.bytes) + " B of " + std::to_string(want);
        return false;
    }
    arena_ = a;
    base_ = a->data();
    exclusive_.assign((size_t) (n_layers * n_expert), 0);
    released_host_bytes_ = 0;
    pinned_bytes_ = a->registered_bytes;
    // plan v0.3 P6: device aliases of the mapped registration, for the PCIe share of the misses
    dev_slice_.clear();
    slice_bytes_ = a->slice_bytes;
    if (a->registered_bytes > 0) {
        std::vector<uint64_t> starts = a->slice_bytes > 0 ? a->slice_starts : std::vector<uint64_t>{0};
        for (uint64_t off : starts) {
            void* d = nullptr;
            if (cudaHostGetDevicePointer(&d, (void*) (base_ + off), 0) != cudaSuccess) {
                (void) cudaGetLastError();
                dev_slice_.clear();
                break;
            }
            dev_slice_.push_back((const uint8_t*) d);
        }
    }
    blobs_ = n_layers * n_expert;
    n_expert_ = n_expert;
    reads_ = 0;
    note_ = a->note;
    gib_per_s_ = st.gib_per_second();
    load_seconds_ = st.seconds;
    load_read_s_ = st.read_seconds;
    load_copy_s_ = st.copy_seconds;
    return true;
}

bool ArenaExpertSource::read_expert(int64_t layer, int64_t expert, uint8_t* dst, std::string& err) {
    const auto& lay = strata::kernels::cpu::expert_layout();
    if (layer < 0 || layer >= lay.n_layers || expert < 0 || expert >= lay.n_expert) {
        err = "read_expert: expert out of range";
        return false;
    }
    std::string name = path_;
    if (from_gguf_) {
        name = gguf_;
        if (!lay.gguf_file.empty() && !lay.gguf_file[(size_t) layer].empty()) {
            const size_t cut = gguf_.find_last_of("/\\");
            name = (cut == std::string::npos ? std::string() : gguf_.substr(0, cut + 1)) + lay.gguf_file[(size_t) layer];
        }
    }
    if (name != rf_name_ || !rf_.is_open()) {
        rf_.close();
        rf_.clear();
        rf_.open(name, std::ios::binary);
        if (!rf_) { err = "read_expert: cannot open " + name; rf_name_.clear(); return false; }
        rf_name_ = name;
    }
    auto read_at = [&](uint64_t off, uint8_t* to, uint64_t n) {
        rf_.clear();
        rf_.seekg((std::streamoff) off);
        rf_.read((char*) to, (std::streamsize) n);
        return (uint64_t) rf_.gcount() == n;
    };
    if (!from_gguf_) {
        if (!read_at(lay.blob_offset(layer, expert), dst, lay.blob_bytes(layer))) { err = "read_expert: short read"; return false; }
        return true;
    }
    // the GGUF holds each role's 512 experts one after another: gate rows | up rows | down rows
    const auto& fm = lay.fmt[(size_t) layer];
    const uint64_t blob = lay.bytes[(size_t) layer];
    const uint64_t per[3] = {fm.up_off, fm.up_off, blob - fm.down_off};
    const uint64_t at[3] = {0, fm.up_off, fm.down_off};
    for (int r = 0; r < 3; ++r)
        if (!read_at(lay.gguf_off[(size_t) (3 * layer + r)] + (uint64_t) expert * per[r], dst + at[r], per[r])) {
            err = "read_expert: short read";
            return false;
        }
    return true;
}

namespace {
std::string expert_file(const std::string& gguf, const std::string& path, bool from_gguf,
                        const strata::kernels::cpu::ExpertLayout& lay, int64_t layer) {
    if (!from_gguf) return path;
    if (lay.gguf_file.empty() || lay.gguf_file[(size_t) layer].empty()) return gguf;
    const size_t cut = gguf.find_last_of("/\\");
    return (cut == std::string::npos ? std::string() : gguf.substr(0, cut + 1)) + lay.gguf_file[(size_t) layer];
}
// The file ranges one expert occupies: three role slices from a GGUF (gate | up | down), or one blob.
int expert_ranges(const strata::kernels::cpu::ExpertLayout& lay, bool from_gguf, int64_t layer, int64_t expert,
                  uint64_t off[3], uint64_t len[3], uint64_t at[3]) {
    if (!from_gguf) {
        off[0] = lay.blob_offset(layer, expert); len[0] = lay.blob_bytes(layer); at[0] = 0;
        return 1;
    }
    const auto& fm = lay.fmt[(size_t) layer];
    const uint64_t blob = lay.bytes[(size_t) layer];
    const uint64_t per[3] = {fm.up_off, fm.up_off, blob - fm.down_off};
    const uint64_t a[3] = {0, fm.up_off, fm.down_off};
    for (int r = 0; r < 3; ++r) {
        off[r] = lay.gguf_off[(size_t) (3 * layer + r)] + (uint64_t) expert * per[r];
        len[r] = per[r];
        at[r] = a[r];
    }
    return 3;
}
}  // namespace

bool ArenaExpertSource::read_experts(const int32_t* layers, const int32_t* experts, int n, uint8_t* dst, size_t stride,
                                     std::string& err) {
    std::vector<uint8_t*> dsts((size_t) n);
    for (int i = 0; i < n; ++i) dsts[(size_t) i] = dst + (size_t) i * stride;
    return read_experts_to(layers, experts, n, dsts.data(), err);
}

bool ArenaExpertSource::read_experts_to(const int32_t* layers, const int32_t* experts, int n, uint8_t* const* dsts,
                                        std::string& err) {
    using strata::platform::DirectFile;
    const auto& lay = strata::kernels::cpu::expert_layout();
    constexpr uint64_t A = DirectFile::alignment();
    const size_t slot_bytes = ((size_t) lay.max_blob + 2 * A + A - 1) / A * A;   // one aligned range, with its edges
    const size_t need = (size_t) n * 3 * slot_bytes;
    if (need > dscratch_bytes_) {
        if (dscratch_ != nullptr) DirectFile::free_aligned(dscratch_);
        dscratch_ = DirectFile::alloc_aligned(need);
        dscratch_bytes_ = dscratch_ ? need : 0;
        if (dscratch_ == nullptr) { err = "read_experts: bounce buffer"; return false; }
    }
    auto file_for = [&](const std::string& name) -> DirectFile* {
        for (auto& [nm, f] : dfiles_) if (nm == name) return (DirectFile*) f;
        auto* f = new DirectFile();
        if (!f->open(name, err)) { delete f; return nullptr; }
        dfiles_.emplace_back(name, f);
        return f;
    };
    struct Req { DirectFile* f; uint64_t skip, len; uint8_t* to; };
    std::vector<Req> reqs;
    reqs.reserve((size_t) n * 3);
    for (int i = 0; i < n; ++i) {
        uint64_t off[3], len[3], at[3];
        const int nr = expert_ranges(lay, from_gguf_, layers[i], experts[i], off, len, at);
        DirectFile* f = file_for(expert_file(gguf_, path_, from_gguf_, lay, layers[i]));
        if (f == nullptr) return false;
        for (int r = 0; r < nr; ++r) {
            const uint64_t a0 = off[r] & ~(A - 1), a1 = (off[r] + len[r] + A - 1) & ~(A - 1);
            uint8_t* bounce = (uint8_t*) dscratch_ + reqs.size() * slot_bytes;
            if (!f->submit(a0, bounce, (uint32_t) (a1 - a0), (uint64_t) reqs.size(), err)) return false;
            reqs.push_back({f, off[r] - a0, len[r], dsts[i] + at[r]});
        }
    }
    // collect every completion (each file's port reports its own requests)
    std::vector<char> done(reqs.size(), 0);
    size_t got = 0;
    for (auto& [nm, fp] : dfiles_) {
        DirectFile* f = (DirectFile*) fp;
        size_t mine = 0;
        for (const Req& q : reqs) mine += q.f == f;
        strata::platform::Completion c[64];
        while (mine > 0) {
            const int k = f->wait(c, 64, -1);
            for (int j = 0; j < k; ++j) {
                if (c[j].tag == DirectFile::WAKE_TAG) continue;
                const Req& q = reqs[(size_t) c[j].tag];
                if (!c[j].ok || c[j].bytes < q.skip + q.len) { err = "read_experts: short read"; return false; }
                std::memcpy(q.to, (uint8_t*) dscratch_ + (size_t) c[j].tag * slot_bytes + q.skip, (size_t) q.len);
                done[(size_t) c[j].tag] = 1;
                ++got;
                --mine;
            }
        }
    }
    if (got != reqs.size()) { err = "read_experts: missing completions"; return false; }
    return true;
}

// load_rest's reader: each worker takes a layer and reads each role's 512 slices in aligned chunks of 32 experts
// with its own unbuffered file, copying only the host-owned experts into the arena.
static LoadStats load_experts_gguf_direct(const std::string& gguf, uint8_t* dst,
                                          const strata::kernels::cpu::ExpertLayout& lay, int threads,
                                          const uint8_t* skip) {
    using strata::platform::DirectFile;
    LoadStats st;
    const auto t0 = std::chrono::steady_clock::now();
    std::atomic<int64_t> next{0};
    std::atomic<bool> bad{false};
    constexpr uint64_t A = DirectFile::alignment();
    auto worker = [&]() {
        std::string open_name, e;
        DirectFile f;
        void* buf = nullptr;
        size_t buf_bytes = 0;
        for (;;) {
            const int64_t l = next.fetch_add(1);
            if (l >= lay.n_layers || bad) break;
            const std::string name = expert_file(gguf, std::string(), true, lay, l);
            if (name != open_name) {
                f.close();
                if (!f.open(name, e)) { bad = true; break; }
                open_name = name;
            }
            const auto& fm = lay.fmt[(size_t) l];
            const uint64_t blob = lay.bytes[(size_t) l];
            const uint64_t per[3] = {fm.up_off, fm.up_off, blob - fm.down_off};
            const uint64_t at[3] = {0, fm.up_off, fm.down_off};
            for (int r = 0; r < 3 && !bad; ++r) {
                const uint64_t src = lay.gguf_off[(size_t) (3 * l + r)];
                for (int64_t e0 = 0; e0 < lay.n_expert && !bad; e0 += 32) {
                    const int64_t ne = std::min<int64_t>(32, lay.n_expert - e0);
                    bool any = false;
                    for (int64_t x = e0; x < e0 + ne; ++x) any |= !skip[(size_t) (l * lay.n_expert + x)];
                    if (!any) continue;   // a GPU-owned run: not even read
                    const uint64_t off = src + (uint64_t) e0 * per[r], n = (uint64_t) ne * per[r];
                    const uint64_t a0 = off & ~(A - 1), a1 = (off + n + A - 1) & ~(A - 1);
                    if (a1 - a0 > buf_bytes) {
                        if (buf) DirectFile::free_aligned(buf);
                        buf = DirectFile::alloc_aligned((size_t) (a1 - a0));
                        buf_bytes = buf ? (size_t) (a1 - a0) : 0;
                        if (!buf) { bad = true; break; }
                    }
                    strata::platform::Completion c;
                    if (!f.submit(a0, buf, (uint32_t) (a1 - a0), 0, e) || f.wait(&c, 1, -1) != 1 || !c.ok ||
                        c.bytes < (off - a0) + n) { bad = true; break; }
                    for (int64_t x = e0; x < e0 + ne; ++x) {
                        if (skip[(size_t) (l * lay.n_expert + x)]) continue;
                        std::memcpy(dst + lay.blob_offset(l, x) + at[r],
                                    (uint8_t*) buf + (off - a0) + (uint64_t) (x - e0) * per[r], (size_t) per[r]);
                    }
                }
            }
        }
        if (buf) DirectFile::free_aligned(buf);
    };
    std::vector<std::thread> pool;
    for (int i = 1; i < threads; ++i) pool.emplace_back(worker);
    worker();
    for (auto& t : pool) t.join();
    st.seconds = bad ? -1.0 : std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    st.bytes = lay.total;
    return st;
}

bool ArenaExpertSource::load_rest(int threads, std::string& err) {
    if (!deferred_) return true;
    const auto& lay = strata::kernels::cpu::expert_layout();
    const auto t0 = std::chrono::steady_clock::now();
    uint64_t kept = 0;
    std::vector<uint8_t> skip(exclusive_);
    for (size_t i = 0; i < nvme_.size(); ++i) skip[i] |= nvme_[i];
    for (size_t i = 0; i < exclusive_.size(); ++i) {
        if (skip[i]) continue;
        const int64_t l = (int64_t) i / lay.n_expert, x = (int64_t) i % lay.n_expert;
        uint64_t c = 0;
        if (!((PinnedArena*) arena_)->commit_interior(lay.blob_offset(l, x), lay.blob_bytes(l), c, err)) return false;
        kept += lay.blob_bytes(l);
    }
    if (from_gguf_) {
        const LoadStats st = load_experts_gguf_direct(gguf_, const_cast<uint8_t*>(base_), lay, threads, skip.data());
        if (st.seconds < 0) { err = "load_rest: the GGUF read failed"; return false; }
    } else {
        std::string e;
        for (int64_t l = 0; l < lay.n_layers; ++l)
            for (int64_t x = 0; x < lay.n_expert; ++x)
                if (!skip[(size_t) (l * lay.n_expert + x)] &&
                    !read_expert(l, x, const_cast<uint8_t*>(base_) + lay.blob_offset(l, x), e)) { err = e; return false; }
    }
    deferred_ = false;
    cache_used_ = kept;
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    gib_per_s_ = s > 0 ? (double) kept / 1073741824.0 / s : 0.0;
    return true;
}

void ArenaExpertSource::set_capacity(uint64_t bytes, const std::vector<int32_t>& order) {
    const auto& lay = strata::kernels::cpu::expert_layout();
    cache_cap_ = bytes;
    nvme_.assign(exclusive_.size(), 0);
    score_.assign(exclusive_.size(), 0.0f);
    if (bytes == 0) { nvme_.clear(); score_.clear(); return; }
    // everything host-owned starts on NVMe; the first `bytes` of the order come up at load_rest
    for (size_t i = 0; i < exclusive_.size(); ++i) nvme_[i] = exclusive_[i] ? 0 : 1;
    uint64_t used = 0;
    std::vector<uint8_t> seen(exclusive_.size(), 0);
    auto take = [&](size_t i) {
        if (i >= exclusive_.size() || exclusive_[i] || seen[i]) return;
        seen[i] = 1;
        const uint64_t b = lay.blob_bytes((int64_t) i / lay.n_expert);
        if (used + b > bytes) return;
        used += b;
        nvme_[i] = 0;
        score_[i] = 1.0f;   // an admitted expert starts ahead of one never used
    };
    for (int32_t i : order) take((size_t) i);
}

bool ArenaExpertSource::resident(int64_t layer, int64_t expert) const {
    const size_t i = (size_t) (layer * n_expert_ + expert);
    if (i >= exclusive_.size() || exclusive_[i] || deferred_) return false;
    return nvme_.empty() || !nvme_[i];
}

bool ArenaExpertSource::evict_one(int64_t avoid_layer) {
    const auto& lay = strata::kernels::cpu::expert_layout();
    size_t victim = SIZE_MAX;
    float best = 0.0f;
    for (size_t i = 0; i < nvme_.size(); ++i) {
        if (nvme_[i] || exclusive_[i] || (int64_t) i / lay.n_expert == avoid_layer) continue;
        if (victim == SIZE_MAX || score_[i] < best) { victim = i; best = score_[i]; }
    }
    if (victim == SIZE_MAX) return false;
    const int64_t l = (int64_t) victim / lay.n_expert, x = (int64_t) victim % lay.n_expert;
    uint64_t released = 0;
    std::string e;
    if (!((PinnedArena*) arena_)->decommit_interior(lay.blob_offset(l, x), lay.blob_bytes(l), released, e)) return false;
    nvme_[victim] = 1;
    cache_used_ -= lay.blob_bytes(l);
    return true;
}

const uint8_t* ArenaExpertSource::materialize(int64_t layer, int64_t expert, int64_t avoid_layer, std::string& err) {
    std::lock_guard<std::mutex> lk(host_mu_);
    const auto& lay = strata::kernels::cpu::expert_layout();
    const size_t i = (size_t) (layer * n_expert_ + expert);
    if (base_ == nullptr || i >= exclusive_.size() || exclusive_[i]) { err = "materialize: not a host-tier expert"; return nullptr; }
    uint8_t* at = const_cast<uint8_t*>(base_) + lay.blob_offset(layer, expert);
    if (nvme_.empty() || !nvme_[i]) return at;
    const auto t0 = std::chrono::steady_clock::now();
    const uint64_t b = lay.blob_bytes(layer);
    while (cache_cap_ != 0 && cache_used_ + b > cache_cap_ && evict_one(avoid_layer)) {}
    uint64_t c = 0;
    if (!((PinnedArena*) arena_)->commit_interior(lay.blob_offset(layer, expert), b, c, err)) return nullptr;
    // the read lands before the tier says so: a failed read leaves the expert on NVMe and returns null
    if (!read_expert(layer, expert, at, err)) return nullptr;
    nvme_[i] = 0;
    score_[i] += 1.0f;
    cache_used_ += b;
    ++nvme_loads_;
    nvme_ms_ += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    return at;
}

bool ArenaExpertSource::materialize_batch(int64_t layer, const int32_t* experts, int n, std::string& err) {
    std::lock_guard<std::mutex> lk(host_mu_);
    const auto& lay = strata::kernels::cpu::expert_layout();
    if (nvme_.empty() || n <= 0) return true;
    const auto t0 = std::chrono::steady_clock::now();
    std::vector<int32_t> ls, es;
    std::vector<uint8_t*> dsts;
    const uint64_t b = lay.blob_bytes(layer);
    for (int i = 0; i < n; ++i) {
        const size_t idx = (size_t) (layer * n_expert_ + experts[i]);
        if (idx >= nvme_.size() || exclusive_[idx] || !nvme_[idx]) continue;
        while (cache_cap_ != 0 && cache_used_ + b > cache_cap_ && evict_one(layer)) {}
        uint64_t c = 0;
        if (!((PinnedArena*) arena_)->commit_interior(lay.blob_offset(layer, experts[i]), b, c, err)) return false;
        cache_used_ += b;   // accounted now so the next eviction sees it; published below once the bytes landed
        ls.push_back((int32_t) layer);
        es.push_back(experts[i]);
        dsts.push_back(const_cast<uint8_t*>(base_) + lay.blob_offset(layer, experts[i]));
    }
    if (es.empty()) return true;
    if (!read_experts_to(ls.data(), es.data(), (int) es.size(), dsts.data(), err)) {
        cache_used_ -= b * es.size();   // nothing published: they stay on NVMe, and the caller fails loudly
        return false;
    }
    for (int32_t x : es) {
        const size_t idx = (size_t) (layer * n_expert_ + x);
        nvme_[idx] = 0;
        score_[idx] += 1.0f;
    }
    nvme_loads_ += (int64_t) es.size();
    nvme_ms_ += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    return true;
}

void ArenaExpertSource::trim(int64_t avoid_layer) {
    std::lock_guard<std::mutex> lk(host_mu_);
    while (cache_cap_ != 0 && cache_used_ > cache_cap_ && evict_one(avoid_layer)) {}
}

void ArenaExpertSource::decay_scores(float f) {
    for (float& v : score_) v *= f;
}

void ArenaExpertSource::admit_home(int64_t layer, int64_t expert) {
    if (nvme_.empty()) return;
    std::lock_guard<std::mutex> lk(host_mu_);
    const auto& lay = strata::kernels::cpu::expert_layout();
    const size_t i = (size_t) (layer * n_expert_ + expert);
    if (i < nvme_.size() && nvme_[i]) { nvme_[i] = 0; cache_used_ += lay.blob_bytes(layer); }
    score_[i] += 1.0f;
    while (cache_cap_ != 0 && cache_used_ > cache_cap_ && evict_one(-1)) {}
}

bool ArenaExpertSource::read_into(int64_t layer, int64_t expert, uint8_t* dst, std::string& err) {
    if (tail_reader_ && tail_reader_(layer, expert, dst)) return true;   // #34: the tail file
    // the prompt path calls this from several stager threads: each thread keeps its own reader
    thread_local std::ifstream f;
    thread_local std::string name;
    const auto& lay = strata::kernels::cpu::expert_layout();
    const std::string want = expert_file(gguf_, path_, from_gguf_, lay, layer);
    if (want != name || !f.is_open()) {
        f.close(); f.clear();
        f.open(want, std::ios::binary);
        if (!f) { err = "read_into: cannot open " + want; name.clear(); return false; }
        name = want;
    }
    uint64_t off[3], len[3], at[3];
    const int nr = expert_ranges(lay, from_gguf_, layer, expert, off, len, at);
    for (int r = 0; r < nr; ++r) {
        f.clear();
        f.seekg((std::streamoff) off[r]);
        f.read((char*) dst + at[r], (std::streamsize) len[r]);
        if ((uint64_t) f.gcount() != len[r]) { err = "read_into: short read"; return false; }
    }
    return true;
}

void ArenaExpertSource::close() {
    if (rf_.is_open()) rf_.close();
    rf_name_.clear();
    for (auto& [nm, f] : dfiles_) delete (strata::platform::DirectFile*) f;
    dfiles_.clear();
    if (dscratch_ != nullptr) strata::platform::DirectFile::free_aligned(dscratch_);
    dscratch_ = nullptr;
    dscratch_bytes_ = 0;
    if (arena_ != nullptr) {
        delete (PinnedArena*) arena_;
        arena_ = nullptr;
    }
    base_ = nullptr;
    exclusive_.clear();
    released_host_bytes_ = 0;
    blobs_ = 0;
    n_expert_ = 0;
}

uint8_t* ArenaExpertSource::recommit_host_copy(int64_t layer, int64_t expert, std::string& err) {
    std::lock_guard<std::mutex> lk(host_mu_);
    if (base_ == nullptr || arena_ == nullptr || layer < 0 || expert < 0 ||
        expert >= n_expert_ || layer >= blobs_ / n_expert_) {
        err = "exclusive host re-commit needs a loaded, in-range expert";
        return nullptr;
    }
    const size_t index = (size_t) (layer * n_expert_ + expert);
    if (index >= exclusive_.size() || !exclusive_[index]) {
        err = "exclusive host re-commit needs a GPU-owned expert";
        return nullptr;
    }
    const auto& layout = strata::kernels::cpu::expert_layout();
    uint64_t committed = 0;
    if (!((PinnedArena*) arena_)->commit_interior(layout.blob_offset(layer, expert), layout.blob_bytes(layer),
                                                  committed, err)) return nullptr;
    released_host_bytes_ -= committed < released_host_bytes_ ? committed : released_host_bytes_;
    err.clear();
    return const_cast<uint8_t*>(base_) + layout.blob_offset(layer, expert);   // GPU-owned until publish_host_copy
}

void ArenaExpertSource::publish_host_copy(int64_t layer, int64_t expert) {
    std::lock_guard<std::mutex> lk(host_mu_);
    const size_t index = (size_t) (layer * n_expert_ + expert);
    if (index < exclusive_.size()) exclusive_[index] = 0;
}

bool ArenaExpertSource::release_host_copy(int64_t layer, int64_t expert, std::string& err) {
    std::lock_guard<std::mutex> lk(host_mu_);
    if (base_ == nullptr || arena_ == nullptr || layer < 0 || expert < 0 ||
        expert >= n_expert_ || layer >= blobs_ / n_expert_) {
        err = "exclusive host release needs a loaded, in-range expert";
        return false;
    }
    const size_t index = (size_t) (layer * n_expert_ + expert);
    if (index >= exclusive_.size() || exclusive_[index]) {
        err = "exclusive host expert is already GPU-owned";
        return false;
    }
    const auto& layout = strata::kernels::cpu::expert_layout();
    uint64_t decommitted = 0;
    if (!((PinnedArena*) arena_)->decommit_interior(layout.blob_offset(layer, expert),
                                                    layout.blob_bytes(layer), decommitted, err)) return false;
    exclusive_[index] = 1; // publish ownership only after the Windows decommit succeeds
    released_host_bytes_ += decommitted;
    err.clear();
    return true;
}

bool ArenaExpertSource::pinned(int64_t layer, int64_t expert) const {
    if (base_ == nullptr || layer < 0 || expert < 0 || expert >= n_expert_) return false;
    if ((size_t) (layer * n_expert_ + expert) < exclusive_.size() &&
        exclusive_[(size_t) (layer * n_expert_ + expert)]) return false;
    const auto& lay = strata::kernels::cpu::expert_layout();
    return lay.blob_offset(layer, expert) + lay.blob_bytes(layer) <= pinned_bytes_;
}

const uint8_t* ArenaExpertSource::device_alias(int64_t layer, int64_t expert) const {
    if (dev_slice_.empty() || !pinned(layer, expert)) return nullptr;
    const auto& lay = strata::kernels::cpu::expert_layout();
    if (slice_bytes_ == 0) return dev_slice_[0] + lay.blob_offset(layer, expert);
    // one registration slice per layer
    if ((size_t) layer >= dev_slice_.size()) return nullptr;
    return dev_slice_[(size_t) layer] + (uint64_t) expert * lay.blob_bytes(layer);
}

const uint8_t* ArenaExpertSource::blob(int64_t layer, int64_t expert) {
    if (base_ == nullptr) return nullptr;
    if (layer < 0 || expert < 0 || expert >= n_expert_) return nullptr;
    const int64_t idx = layer * n_expert_ + expert;
    if (idx < 0 || idx >= blobs_) return nullptr;
    if ((size_t) idx < exclusive_.size() && exclusive_[(size_t) idx]) return nullptr;
    if (deferred_) return nullptr;   // placement-first: not loaded yet (its pages may not even be committed)
    if (!nvme_.empty()) {
        if (nvme_[(size_t) idx]) return nullptr;   // #11: on NVMe; the caller materializes it
        score_[(size_t) idx] += 1.0f;
    }
    ++reads_;
    // Pointer arithmetic into resident memory.  No fault, no copy, no mapping - which is the entire point of
    // this class over `FileExpertSource`.
    return base_ + strata::kernels::cpu::expert_layout().blob_offset(layer, expert);
}

}  // namespace strata::core
