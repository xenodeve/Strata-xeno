// src/core/expert_source.cpp - the adapter.  See the header for the three clauses of the contract.
#include "strata/core/expert_source.hpp"
#include "strata/core/remote_experts.hpp"
#include "strata/timeline.hpp"
#include "strata/platform/direct_file.hpp"
#include "strata/core/secondary_runner.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/artifact/gguf_reader.hpp"

#include "strata/core/pinned.hpp"
#include "strata/platform/memory.hpp"
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/quantize_act.hpp"
#include "strata/kernels/s2_expert_grouped.hpp"
#include "strata/kernels/cpu/kq_avx2.hpp"

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
#include <map>
#include <memory>
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

// a 64-bit seek (as in pinned.cu): the 32-bit `fseek` wraps past 4 GiB, and the spelling differs per platform
#ifndef STRATA_FSEEK64
#ifdef _WIN32
#define STRATA_FSEEK64(f, o) _fseeki64((f), (long long) (o), SEEK_SET)
#else
#define STRATA_FSEEK64(f, o) fseeko((f), (off_t) (o), SEEK_SET)
#endif
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
    if (layout.native && !gguf_.empty() && !std::filesystem::exists(path)) {
        // CS-T: no experts.bin - the model's GGUF shards, read in place
        blobs_ = (int64_t) blob_count;
        n_layers_ = n_layers;
        n_expert_ = n_expert;
        layer_offsets_ = std::move(layer_offsets);
        layer_blob_bytes_ = std::move(layer_blob_bytes);
        if (!open_gguf(err)) { close(); return false; }
        return true;
    }

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
        if (complement_pinned_ && !complement_partial_) (void) cudaFreeHost(complement_arena_);
        else {
            if (complement_partial_) (void) cudaHostUnregister(complement_arena_);
            if (complement_locked_ > 0)
                strata::platform::unlock_resident((uint8_t*) complement_arena_ + complement_lock_off_, complement_locked_);
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
    complement_partial_ = false;
    complement_pin_limit_ = 0;
    complement_lock_off_ = 0;
    complement_ready_ = false;
    complement_locked_ = 0;
    complement_lent_slots_ = 0;
    if (!maps_.empty()) {
        for (Map& m : maps_) {
#if defined(_WIN32)
            if (m.base != nullptr) UnmapViewOfFile((LPCVOID) m.base);
            if (m.mapping != nullptr) CloseHandle((HANDLE) m.mapping);
            if (m.file != nullptr) CloseHandle((HANDLE) m.file);
#else
            if (m.base != nullptr) munmap((void*) m.base, (size_t) m.bytes);
            if (m.fd >= 0) ::close(m.fd);
#endif
        }
        maps_.clear();
        base_ = nullptr;   // one of the views above
    }
    role_ptr_.clear();
    role_bytes_.clear();
    {
        std::lock_guard<std::mutex> lk(stage_mu_);
        stage_buf_.clear();
        stage_key_.clear();
        stage_epoch_.clear();
        stage_used_.clear();
        stage_busy_.clear();
        stage_of_.clear();
        stage_blob_ = 0;
        stage_seq_ = 0;
        epoch_ = 0;
        last_layer_ = -1;
        stage_grew_ = false;
    }
    ram_reads_.store(0);
    warm_stamp_.reset();
    warm_hits_.store(0);
    warm_count_.store(0);
    file_read_bytes_.store(0);
    file_blob_bytes_.store(0);
    file_us_.store(0);
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


bool ExpertSource::copy_blob(int64_t layer, int64_t expert, uint8_t* dst) {
    const uint8_t* b = blob(layer, expert);
    if (b == nullptr || dst == nullptr) return false;
    std::memcpy(dst, b, (size_t) strata::kernels::cpu::expert_layout().blob_bytes(layer));
    return true;
}

// ================================ CS-T: THE GGUF SHARDS IN PLACE ================================
//
// A native pack without experts.bin: every file native_experts.txt names is mapped (MapViewOfFile / mmap, no
// flag - the same retention argument as experts.bin above), and an expert's blob [gate rows | up rows | down rows]
// is three slices of three tensors, possibly in two shards (UD-Q4_K_XL's layer 11).  Nothing is read at open; a
// blob is assembled when it is asked for (`blob`, into a small pool of buffers) or copied where it is needed
// (`copy_blob`: the RAM copy, the prompt path's pinned stager buffers).
bool FileExpertSource::open_gguf(std::string& err) {
    const auto& lay = strata::kernels::cpu::expert_layout();
    if (!check_experts_gguf(gguf_, lay, err)) { err = "FileExpertSource: " + err; return false; }
    const size_t cut = gguf_.find_last_of("/\\");
    const std::string dir = cut == std::string::npos ? std::string() : gguf_.substr(0, cut + 1);
    std::map<std::string, size_t> index;
    role_ptr_.assign((size_t) (3 * n_layers_), nullptr);
    role_bytes_.assign((size_t) (3 * n_layers_), 0);
    for (int64_t l = 0; l < n_layers_; ++l) {
        const auto& fm = lay.fmt[(size_t) l];
        const uint64_t per[3] = {fm.up_off, fm.up_off, lay.bytes[(size_t) l] - fm.down_off};
        for (int r = 0; r < 3; ++r) {
            const size_t i = (size_t) (3 * l + r);
            const std::string path = lay.gguf_file.size() > i && !lay.gguf_file[i].empty() ? dir + lay.gguf_file[i]
                                                                                            : gguf_;
            auto it = index.find(path);
            if (it == index.end()) {
                Map m;
#if defined(_WIN32)
                const int wide = MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, nullptr, 0);
                std::vector<wchar_t> wpath((size_t) (wide > 0 ? wide : 1));
                if (wide > 0) MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, wpath.data(), wide);
                HANDLE f = CreateFileW(wpath.data(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                                       FILE_ATTRIBUTE_NORMAL, nullptr);
                LARGE_INTEGER sz{};
                if (f == INVALID_HANDLE_VALUE || !GetFileSizeEx(f, &sz)) {
                    if (f != INVALID_HANDLE_VALUE) CloseHandle(f);
                    err = "FileExpertSource: cannot open " + path;
                    return false;
                }
                HANDLE mh = CreateFileMappingW(f, nullptr, PAGE_READONLY, 0, 0, nullptr);
                void* view = mh != nullptr ? MapViewOfFile(mh, FILE_MAP_READ, 0, 0, 0) : nullptr;
                if (view == nullptr) {
                    if (mh != nullptr) CloseHandle(mh);
                    CloseHandle(f);
                    err = "FileExpertSource: cannot map " + path;
                    return false;
                }
                m.file = f;
                m.mapping = mh;
                m.bytes = (uint64_t) sz.QuadPart;
                m.base = (const uint8_t*) view;
#else
                const int fd = ::open(path.c_str(), O_RDONLY);
                struct stat st{};
                if (fd < 0 || fstat(fd, &st) != 0 || st.st_size <= 0) {
                    if (fd >= 0) ::close(fd);
                    err = "FileExpertSource: cannot open " + path;
                    return false;
                }
                void* view = mmap(nullptr, (size_t) st.st_size, PROT_READ, MAP_SHARED, fd, 0);
                if (view == MAP_FAILED) { ::close(fd); err = "FileExpertSource: cannot map " + path; return false; }
                m.fd = fd;
                m.bytes = (uint64_t) st.st_size;
                m.base = (const uint8_t*) view;
#endif
                maps_.push_back(m);
                it = index.emplace(path, maps_.size() - 1).first;
            }
            const Map& m = maps_[it->second];
            const uint64_t at = lay.gguf_off[i], bytes = per[r] * (uint64_t) n_expert_;
            if (at > m.bytes || bytes > m.bytes - at) {   // check_experts_gguf proved it; the mapping must agree
                err = "FileExpertSource: an expert span runs past the end of " + path;
                return false;
            }
            role_ptr_[i] = m.base + (size_t) at;
            role_bytes_[i] = per[r];
        }
    }
    base_ = maps_.front().base;       // "opened"; mapped_blob answers nullptr in this mode
    warm_stamp_.reset(new std::atomic<uint32_t>[(size_t) (n_layers_ * n_expert_)]());
    for (uint64_t b : layer_blob_bytes_) stage_blob_ = std::max(stage_blob_, b);
    return true;
}

bool FileExpertSource::copy_from_files(int64_t layer, int64_t expert, uint8_t* dst) const {
    if (dst == nullptr || layer < 0 || expert < 0 || layer >= n_layers_ || expert >= n_expert_) return false;
    if (!role_ptr_.empty()) {
        uint64_t at = 0;
        for (int r = 0; r < 3; ++r) {
            const size_t i = (size_t) (3 * layer + r);
            const uint64_t per = role_bytes_[i];
            std::memcpy(dst + at, role_ptr_[i] + (size_t) ((uint64_t) expert * per), (size_t) per);
            at += per;
        }
        return true;
    }
    const uint8_t* b = mapped_blob(layer, expert);
    if (b == nullptr) return false;
    std::memcpy(dst, b, (size_t) layer_blob_bytes_[(size_t) layer]);
    return true;
}

// A blob assembled from the three role slices.  The buffer of a (layer, expert) is reused for another only once
// its blob has not been asked for during `kStageAge` layers (begin_layer) or 256 assemblies, whichever comes
// first, and never while it is being filled: the pool computes a layer's misses before it starts the next, and a
// fill (the GPU cache at startup, an adaptive swap, a helper GPU) copies the blob right away.
//
// `claim_stage` finds or reserves the buffer of `key` (stage_mu_ held): true when the blob is already there (or
// being filled by another thread - the caller then waits), false when the caller must fill buffer `v`.
bool FileExpertSource::claim_stage(int64_t key, size_t& v, bool& fill) {
    constexpr uint64_t kStageSeq = 256;
    const uint64_t seq = ++stage_seq_;
    auto it = stage_of_.find(key);
    fill = false;
    if (it != stage_of_.end()) {
        v = it->second;
        stage_epoch_[v] = epoch_;
        stage_used_[v] = seq;
        return true;
    }
    v = stage_buf_.size();
    uint64_t oldest = std::numeric_limits<uint64_t>::max();
    for (size_t i = 0; i < stage_buf_.size(); ++i)
        if (!stage_busy_[i] && (stage_epoch_[i] + kStageAge <= epoch_ || stage_used_[i] + kStageSeq <= seq) &&
            stage_used_[i] < oldest) {
            oldest = stage_used_[i];
            v = i;
        }
    if (v == stage_buf_.size()) {
        stage_buf_.emplace_back(new (std::nothrow) uint8_t[(size_t) stage_blob_]);
        if (!stage_buf_.back()) { stage_buf_.pop_back(); return false; }
        stage_key_.push_back(-1);
        stage_epoch_.push_back(0);
        stage_used_.push_back(0);
        stage_busy_.push_back(0);
        if (stage_buf_.size() == 512 && !stage_grew_) {
            stage_grew_ = true;
            std::fprintf(stderr, "FileExpertSource: %zu blobs assembled from the GGUF are in use at once (%.2f GiB)\n",
                         stage_buf_.size(), (double) stage_buf_.size() * (double) stage_blob_ / 1073741824.0);
        }
    } else {
        stage_of_.erase(stage_key_[v]);
    }
    stage_key_[v] = key;
    stage_epoch_[v] = epoch_;
    stage_used_[v] = seq;
    stage_busy_[v] = 1;
    stage_of_[key] = v;
    fill = true;
    return false;
}

// Fills buffer `v` (reserved by claim_stage) outside the lock, then publishes it.
bool FileExpertSource::fill_stage(size_t v, int64_t layer, int64_t expert, uint8_t* dst) {
    const auto t0 = std::chrono::steady_clock::now();
    const bool ok = copy_from_files(layer, expert, dst);
    const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
    file_us_.fetch_add((uint64_t) us, std::memory_order_relaxed);
    if (ok) {
        file_read_bytes_.fetch_add(layer_blob_bytes_[(size_t) layer], std::memory_order_relaxed);
        file_blob_bytes_.fetch_add(layer_blob_bytes_[(size_t) layer], std::memory_order_relaxed);
    }
    {
        std::lock_guard<std::mutex> lk(stage_mu_);
        stage_busy_[v] = 0;
        if (!ok) {
            stage_of_.erase(stage_key_[v]);
            stage_key_[v] = -1;
        }
    }
    stage_cv_.notify_all();
    return ok;
}

const uint8_t* FileExpertSource::staged_blob(int64_t layer, int64_t expert) {
    const int64_t key = layer * n_expert_ + expert;
    size_t v = 0;
    bool fill = false;
    uint8_t* dst = nullptr;
    {
        std::unique_lock<std::mutex> lk(stage_mu_);
        const bool have = claim_stage(key, v, fill);
        if (!have && !fill) return nullptr;
        dst = stage_buf_[v].get();
        if (have) {
            // another thread (a prefetch, the adaptive tier) is filling it: wait for that
            stage_cv_.wait(lk, [&] { return !stage_busy_[v] || stage_key_[v] != key; });
            if (stage_key_[v] != key) return nullptr;   // its fill failed
            return dst;
        }
    }
    return fill_stage(v, layer, expert, dst) ? dst : nullptr;
}

void FileExpertSource::prefetch(int64_t layer, const int64_t* experts, int64_t n) {
    if (role_ptr_.empty() || n <= 0 || layer < 0 || layer >= n_layers_) return;
    struct Fill { size_t v; int64_t e; uint8_t* dst; };
    std::vector<Fill> todo;
    {
        std::lock_guard<std::mutex> lk(stage_mu_);
        for (int64_t i = 0; i < n; ++i) {
            const int64_t e = experts[i];
            if (e < 0 || e >= n_expert_) continue;
            const size_t index = (size_t) layer * (size_t) n_expert_ + (size_t) e;
            if (complement_ready_ && index < complement_offsets_.size() && complement_offsets_[index] != kNoComplement)
                continue;                                     // in the RAM copy
            if (!override_.empty() && override_[index] != nullptr) continue;
            size_t v = 0;
            bool fill = false;
            if (!claim_stage(layer * n_expert_ + e, v, fill) && fill) {
                todo.push_back({v, e, stage_buf_[v].get()});
                if (warm_stamp_) {
                    const uint32_t s = warm_stamp_[index].load(std::memory_order_relaxed);
                    if (s != 0 && (uint64_t) s + 3 >= epoch_ + 1) warm_hits_.fetch_add(1, std::memory_order_relaxed);
                }
            }
        }
    }
    if (todo.empty()) return;
#if defined(_WIN32)
    // One PrefetchVirtualMemory call for every slice about to be copied: the memory manager reads them in large
    // requests, all queued at once, where the copies' page faults would read a few clusters each.  The copies below
    // then find the pages resident (or in flight).  STRATA_FETCH_PVM=0 is the A/B arm.
    {
        using Pvm = BOOL(WINAPI*)(HANDLE, ULONG_PTR, PWIN32_MEMORY_RANGE_ENTRY, ULONG);
        static const Pvm pvm = [] {
            const char* v = std::getenv("STRATA_FETCH_PVM");
            if (v != nullptr && std::atoi(v) == 0) return (Pvm) nullptr;
            return (Pvm) (void*) GetProcAddress(GetModuleHandleW(L"kernel32.dll"), "PrefetchVirtualMemory");
        }();
        if (pvm != nullptr) {
            std::vector<WIN32_MEMORY_RANGE_ENTRY> ranges;
            ranges.reserve(todo.size() * 3);
            for (const Fill& f : todo)
                for (int r = 0; r < 3; ++r) {
                    const size_t i = (size_t) (3 * layer + r);
                    ranges.push_back({(PVOID) (role_ptr_[i] + (size_t) ((uint64_t) f.e * role_bytes_[i])),
                                      (SIZE_T) role_bytes_[i]});
                }
            (void) pvm(GetCurrentProcess(), (ULONG_PTR) ranges.size(), ranges.data(), 0);
        }
    }
#endif
    // the page faults of a mapped read are one outstanding request each: several threads keep the SSD's queue full
    std::atomic<size_t> next{0};
    auto work = [&] {
        for (size_t i; (i = next.fetch_add(1)) < todo.size();)
            (void) fill_stage(todo[i].v, layer, todo[i].e, todo[i].dst);
    };
    const size_t nt = std::min<size_t>(todo.size(), (size_t) fetch_threads_);
    std::vector<std::thread> th;
    for (size_t t = 1; t < nt; ++t) th.emplace_back(work);
    work();
    for (auto& t : th) t.join();
}


void FileExpertSource::warm(int64_t layer, const int64_t* experts, int64_t n) {
    if (role_ptr_.empty() || n <= 0 || layer < 0 || layer >= n_layers_) return;
    uint32_t stamp;
    {
        std::lock_guard<std::mutex> lk(stage_mu_);
        stamp = (uint32_t) epoch_ + 1;
    }
#if defined(_WIN32)
    using Pvm = BOOL(WINAPI*)(HANDLE, ULONG_PTR, PWIN32_MEMORY_RANGE_ENTRY, ULONG);
    static const Pvm pvm = (Pvm) (void*) GetProcAddress(GetModuleHandleW(L"kernel32.dll"), "PrefetchVirtualMemory");
    std::vector<WIN32_MEMORY_RANGE_ENTRY> ranges;
#endif
    for (int64_t j = 0; j < n; ++j) {
        const int64_t e = experts[j];
        if (e < 0 || e >= n_expert_) continue;
        const size_t index = (size_t) layer * (size_t) n_expert_ + (size_t) e;
        if (complement_ready_ && index < complement_offsets_.size() && complement_offsets_[index] != kNoComplement)
            continue;                                                    // in the RAM copy
        if (warm_stamp_) warm_stamp_[index].store(stamp, std::memory_order_relaxed);
        warm_count_.fetch_add(1, std::memory_order_relaxed);
        for (int r = 0; r < 3; ++r) {
            const size_t i = (size_t) (3 * layer + r);
            const uint8_t* p = role_ptr_[i] + (size_t) ((uint64_t) e * role_bytes_[i]);
#if defined(_WIN32)
            ranges.push_back({(PVOID) p, (SIZE_T) role_bytes_[i]});
#else
            const uintptr_t pg = 4096, a = (uintptr_t) p & ~(pg - 1);
            (void) madvise((void*) a, (size_t) ((uintptr_t) p + role_bytes_[i] - a), MADV_WILLNEED);
#endif
        }
    }
#if defined(_WIN32)
    if (pvm != nullptr && !ranges.empty()) (void) pvm(GetCurrentProcess(), (ULONG_PTR) ranges.size(), ranges.data(), 0);
#endif
}

RouterLookahead::~RouterLookahead() {
    {
        std::lock_guard<std::mutex> lk(mu_);
        quit_ = true;
    }
    cv_.notify_all();
    if (thread_.joinable()) thread_.join();
}

bool RouterLookahead::start(std::vector<std::vector<uint16_t>> routers, int64_t n_embd, int64_t n_expert, int k,
                            ExpertSource* src, std::string& err) {
    if (src == nullptr || !src->warms()) { err = "RouterLookahead: the expert source does not warm"; return false; }
    if (n_embd % 8 != 0) { err = "RouterLookahead: n_embd is not a multiple of 8"; return false; }
    for (const auto& r : routers)
        if (r.size() != (size_t) (n_embd * n_expert)) { err = "RouterLookahead: a router of another shape"; return false; }
    routers_ = std::move(routers);
    n_embd_ = n_embd;
    n_expert_ = n_expert;
    k_ = k < 1 ? 1 : k > (int) n_expert ? (int) n_expert : k;
    src_ = src;
    x_.assign((size_t) (8 * n_embd), 0.f);
    thread_ = std::thread([this] { run(); });
    return true;
}

void RouterLookahead::submit(int64_t layer, const float* x, int64_t n_tok, const int32_t* host_res) {
    if (layer + 1 >= (int64_t) routers_.size() || n_tok <= 0 || x == nullptr) return;
    {
        std::lock_guard<std::mutex> lk(mu_);
        if (busy_ || pending_) { skipped_.fetch_add(1, std::memory_order_relaxed); return; }
        n_tok_ = std::min<int64_t>(n_tok, 8);
        std::memcpy(x_.data(), x, (size_t) (n_tok_ * n_embd_) * sizeof(float));
        layer_ = layer + 1;
        host_res_ = host_res;
        pending_ = true;
    }
    cv_.notify_one();
}

void RouterLookahead::run() {
    std::vector<float> logits((size_t) (8 * n_expert_));
    std::vector<int32_t> order((size_t) n_expert_);
    std::vector<int64_t> want;
    for (;;) {
        int64_t layer, nt;
        const int32_t* host_res;
        {
            std::unique_lock<std::mutex> lk(mu_);
            cv_.wait(lk, [&] { return quit_ || pending_; });
            if (quit_) return;
            pending_ = false;
            busy_ = true;
            layer = layer_;
            nt = n_tok_;
            host_res = host_res_;
        }
        const auto t0 = std::chrono::steady_clock::now();
        want.clear();
        strata::kernels::cpu::bf16_rows_dot_multi(routers_[(size_t) layer].data(), (int) n_expert_, (int) n_embd_,
                                                  x_.data(), (int) nt, logits.data());
        for (int64_t t = 0; t < nt; ++t) {
            const float* lt = logits.data() + (size_t) (t * n_expert_);
            for (int64_t e = 0; e < n_expert_; ++e) order[(size_t) e] = (int32_t) e;
            std::partial_sort(order.begin(), order.begin() + k_, order.end(),
                              [&](int32_t a, int32_t b) { return lt[(size_t) a] > lt[(size_t) b]; });
            for (int j = 0; j < k_; ++j) {
                const int64_t e = order[(size_t) j];
                if (host_res != nullptr && host_res[(size_t) (layer * n_expert_ + e)] >= 0) continue;   // on the GPU
                if (std::find(want.begin(), want.end(), e) == want.end()) want.push_back(e);
            }
        }
        src_->warm(layer, want.data(), (int64_t) want.size());
        predicted_.fetch_add((int64_t) want.size(), std::memory_order_relaxed);
        busy_us_.fetch_add((uint64_t) std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count(),
                           std::memory_order_relaxed);
        {
            std::lock_guard<std::mutex> lk(mu_);
            busy_ = false;
        }
    }
}

void FileExpertSource::begin_layer(int64_t layer, const int32_t* ids, int64_t k) {
    (void) ids;
    (void) k;
    if (role_ptr_.empty()) return;
    std::lock_guard<std::mutex> lk(stage_mu_);
    if (layer != last_layer_) {
        ++epoch_;
        last_layer_ = layer;
    }
}

bool FileExpertSource::transient(int64_t layer, int64_t expert) const {
    if (role_ptr_.empty() || layer < 0 || expert < 0 || layer >= n_layers_ || expert >= n_expert_) return false;
    const size_t index = (size_t) layer * (size_t) n_expert_ + (size_t) expert;
    if (complement_ready_ && index < complement_offsets_.size() && complement_offsets_[index] != kNoComplement)
        return false;
    return override_.empty() || override_[index] == nullptr;
}

bool FileExpertSource::copy_blob(int64_t layer, int64_t expert, uint8_t* dst) {
    if (base_ == nullptr || dst == nullptr || layer < 0 || expert < 0 || layer >= n_layers_ || expert >= n_expert_)
        return false;
    const size_t index = (size_t) layer * (size_t) n_expert_ + (size_t) expert;
    const uint64_t bytes = layer_blob_bytes_[(size_t) layer];
    if (complement_ready_) {
        const uint8_t* held =
            detail::cache_complement_blob_or_fallback(index, complement_offsets_, complement_host_, nullptr);
        if (held == nullptr && !override_.empty()) held = override_[index];
        if (held != nullptr) {
            std::memcpy(dst, held, (size_t) bytes);
            return true;
        }
    }
    const auto t0 = std::chrono::steady_clock::now();
    if (!copy_from_files(layer, expert, dst)) return false;
    file_us_.fetch_add((uint64_t) std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count(),
                       std::memory_order_relaxed);
    file_read_bytes_.fetch_add(bytes, std::memory_order_relaxed);
    return true;
}

const uint8_t* FileExpertSource::mapped_blob(int64_t layer, int64_t expert) const {
    if (!role_ptr_.empty()) return nullptr;   // the GGUF in place: no contiguous blob in any file
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
    uint64_t headroom_bytes, uint64_t budget_bytes, const std::vector<std::pair<int32_t, int32_t>>* rank) {
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
#if defined(_WIN32)
    // #467: the GPU cache's pre-fill touched its experts through the mapping (~19 GiB on a 24 GB card), and Windows
    // counts those file pages in this process's working set, not as available: a 32 GB PC read 0.44 GiB here
    // (20.7 GiB before the start).  Trimmed, they move to the standby list (still cached, counted as available).
    // Resident mode only: nothing else calls this function.  Locked/pinned pages stay; the rest fault back softly.
    {
        uint64_t before = 0, after = 0;
        const bool read_before = available_memory_bytes(before);
        (void) SetProcessWorkingSetSize(GetCurrentProcess(), (SIZE_T) -1, (SIZE_T) -1);
        if (read_before && available_memory_bytes(after))
            std::fprintf(stderr, "FileExpertSource: available RAM %.2f GiB, %.2f GiB after the mapped experts left the "
                                 "process working set (#467)\n",
                         (double) before / 1073741824.0, (double) after / 1073741824.0);
    }
#endif
    const bool what_fits = budget_bytes == kResidentWhatFits;   // #467: the soft mode's second try
    uint64_t budget_physical = 0;   // #403: the RAM reading a budget was sized from (0: no budget)
    if (budget_bytes > 0) {
        // CS-T: a RAM budget.  The complement's experts in `rank` order (the expert profile, hottest first) while
        // they fit, the rest left on the mapped files; clamped to what the RAM has room for.
        uint64_t physical = 0;
        if (!available_memory_bytes(physical)) {
            err = "FileExpertSource: cannot determine available RAM for --resident-budget-gib";
            return false;
        }
        budget_physical = physical;
        const uint64_t room = physical > headroom_bytes ? physical - headroom_bytes : 0;
        if (budget_bytes > room) {
            // #403: 256 MiB under the room, so the engine's own allocations after this reading still leave the
            // headroom (a budget clamped to exactly the room failed the safety check below on a reading a few MB
            // lower).  An unclamped budget is unchanged.
            const uint64_t margin = 256ull << 20;
            const uint64_t clamped = room > margin ? room - margin : 0;
            if (what_fits)
                std::fprintf(stderr, "FileExpertSource: RAM room for the complement: %.2f GiB (%.2f GiB available "
                                     "minus %.0f GiB headroom and a 0.25 GiB margin)\n",
                             (double) clamped / 1073741824.0, (double) physical / 1073741824.0,
                             (double) headroom_bytes / 1073741824.0);
            else
                std::fprintf(stderr, "FileExpertSource: --resident-budget-gib %.2f is more than the RAM has room for "
                                     "(%.2f GiB available minus %.0f GiB headroom and a 0.25 GiB margin): %.2f GiB\n",
                             (double) budget_bytes / 1073741824.0, (double) physical / 1073741824.0,
                             (double) headroom_bytes / 1073741824.0, (double) clamped / 1073741824.0);
            budget_bytes = clamped;
        }
        std::vector<uint64_t> ranked(offsets.size(), kNoComplement);
        uint64_t at = 0;
        int64_t held = 0;
        if (rank != nullptr)
            for (const auto& pr : *rank) {
                if (pr.first < 0 || pr.second < 0 || pr.first >= n_layers_ || pr.second >= n_expert_) continue;
                const size_t i = (size_t) pr.first * (size_t) n_expert_ + (size_t) pr.second;
                if (offsets[i] == kNoComplement || ranked[i] != kNoComplement) continue;   // on a GPU, or twice
                const uint64_t b = layer_blob_bytes_[(size_t) pr.first];
                if (b > budget_bytes - at) continue;
                ranked[i] = at;
                at += b;
                ++held;
            }
        if (what_fits && held == 0) {   // #467: nothing to keep - the caller's plain mmap fallback, not an empty copy
            err = "FileExpertSource: the RAM has no room for any expert of the complement";
            return false;
        }
        std::fprintf(stderr, "FileExpertSource: RAM budget %.2f GiB: %lld of the %.2f GiB of experts the GPU cache does "
                             "not hold, by profile rank; the rest are read from the files\n",
                     (double) budget_bytes / 1073741824.0, (long long) held, (double) bytes / 1073741824.0);
        offsets.swap(ranked);
        bytes = at;
        lend_from_slot = -1;
    }

    const bool lend = lend_from_slot >= 0 && lend_from_slot < n_slots && additional_gpu_pairs.empty();
    uint64_t budget = std::numeric_limits<uint64_t>::max();
    if (bytes > 0 || lend) {
        // #403: with a budget, the reading it was sized from - a second reading a few MB lower (the engine's own
        // allocations, the file cache) failed a budget the first one had clamped.  (A budget turns `lend` off.)
        uint64_t physical = budget_physical;
        if (physical == 0 && !available_memory_bytes(physical)) {
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
    uint64_t partial_pin = 0;   ///< CS-T: a registered prefix of a locked arena
    uint64_t lock_off = 0;      ///< where the working-set lock starts (after the registered prefix)
    std::string note;
    auto release = [&]() {
        if (arena == nullptr) return;
        if (pinned_ok) (void) cudaFreeHost(arena);
        else {
            if (partial_pin > 0) (void) cudaHostUnregister(arena);
            if (locked > 0) strata::platform::unlock_resident((uint8_t*) arena + lock_off, locked);
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
                // CS-T, a RAM budget: its bytes are in profile order, hottest first, so the driver is asked to
                // register the largest prefix it takes (from the cap down in 2 GiB steps).  Those experts can be
                // read by the GPU over PCIe (--pcie-frac) and copied by DMA; only the rest is locked in the working
                // set (the registered prefix is page-locked by the driver already - locking it twice made the next
                // device allocation fail).
                // opt-in (STRATA_PARTIAL_PIN=1): on the RTX 5070 PC the GPU's PCIe share of the misses measured no
                // faster than the CPU computing them (7.30 / 7.44 tok/s with 24 / 16 GiB registered against 7.05-7.74
                // unpinned at a 40 GiB budget), and registering adds startup time and driver memory pressure
                static const bool partial_on = [] {
                    const char* v = std::getenv("STRATA_PARTIAL_PIN");
                    return v != nullptr && std::atoi(v) != 0;
                }();
                // at most STRATA_PARTIAL_PIN_GIB (default 24): registering 30 GiB of a 40 GiB arena left the driver
                // unable to page-lock the prompt path's small buffers afterwards (RTX 5070, WDDM)
                static const uint64_t pin_cap = [] {
                    const char* v = std::getenv("STRATA_PARTIAL_PIN_GIB");
                    return (uint64_t) ((v != nullptr && std::atof(v) > 0 ? std::atof(v) : 24.0) * 1073741824.0);
                }();
                if (budget_bytes > 0 && partial_on) {
                    const uint64_t step = 2ull << 30;
                    for (uint64_t want = std::min(bytes, pin_cap); want >= step; want = want > step ? want - step : 0) {
                        // cut at an expert boundary: a blob that started inside the registered range and ran past it
                        // would be taken as page-locked by a cudaMemcpyAsync and refused ("adaptive refill failed")
                        uint64_t w = want;
                        for (size_t i = 0; i < offsets.size(); ++i) {
                            if (offsets[i] == kNoComplement) continue;
                            const uint64_t b = layer_blob_bytes_[i / (size_t) n_expert_];
                            if (offsets[i] < want && offsets[i] + b > want) { w = offsets[i]; break; }
                        }
                        if (w == 0) break;
                        if (cudaHostRegister(arena, (size_t) w, cudaHostRegisterMapped | cudaHostRegisterPortable) ==
                            cudaSuccess) {
                            void* alias = nullptr;
                            if (cudaHostGetDevicePointer(&alias, arena, 0) == cudaSuccess && alias != nullptr) {
                                device = (const uint8_t*) alias;
                                partial_pin = w;
                            } else {
                                (void) cudaGetLastError();
                                (void) cudaHostUnregister(arena);
                            }
                            break;
                        }
                        (void) cudaGetLastError();
                        if (want <= step) break;
                    }
                    char msg[160];
                    std::snprintf(msg, sizeof msg, "%.2f GiB of it registered for the GPU (the hottest)",
                                  (double) partial_pin / 1073741824.0);
                    note += std::string("; ") + msg;
                }
                lock_off = partial_pin;
                const strata::platform::LockResult lr =
                    strata::platform::lock_resident((uint8_t*) arena + lock_off, bytes - lock_off);
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
                if (offset > bytes || blob_bytes > bytes - offset ||
                    !copy_from_files(layer, expert, (uint8_t*) host + (size_t) offset)) {
                    fail("FileExpertSource: invalid blob bounds while building the cache complement");
                    return;
                }
                copied.fetch_add(blob_bytes);
            }
#if !defined(_WIN32)
            if (role_ptr_.empty()) {   // experts.bin; the GGUF in place leaves its pages to the OS
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
    if (maps_.empty()) (void) VirtualUnlock((LPVOID) base_, (SIZE_T) mapped_bytes_);
    for (const Map& m : maps_) (void) VirtualUnlock((LPVOID) m.base, (SIZE_T) m.bytes);
#endif

    complement_arena_ = arena;
    complement_host_ = host;
    complement_device_ = device;
    complement_bytes_ = bytes;
    complement_offsets_ = std::move(offsets);
    complement_pinned_ = (pinned_ok || partial_pin > 0) && bytes > 0;
    complement_pin_limit_ = pinned_ok ? bytes : partial_pin;
    complement_partial_ = !pinned_ok && partial_pin > 0;
    complement_locked_ = locked;
    complement_lock_off_ = lock_off;
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
    const uint8_t* result = nullptr;
    bool from_files = true;
    if (complement_ready_) {
        result = detail::cache_complement_blob_or_fallback(index, complement_offsets_, complement_host_, nullptr);
        if (result != nullptr) {
            ram_reads_.fetch_add(1, std::memory_order_relaxed);
            from_files = false;
        } else if (!override_.empty() && override_[index] != nullptr) {
            result = override_[index];
            from_files = false;
        }
    }
    if (from_files) {
        if (!role_ptr_.empty()) {
            result = staged_blob(layer, expert);          // counts its bytes
        } else {
            result = mapped_blob(layer, expert);
            if (result != nullptr)
                file_read_bytes_.fetch_add(layer_blob_bytes_[(size_t) layer], std::memory_order_relaxed);
        }
        if (complement_ready_ && result != nullptr) file_reads_.fetch_add(1, std::memory_order_relaxed);
    }
    if (result != nullptr) ++reads_;
    return result;
}

bool FileExpertSource::pinned(int64_t layer, int64_t expert) const {
    if (!complement_ready_ || !complement_pinned_ || complement_host_ == nullptr || layer < 0 || expert < 0 ||
        layer >= n_layers_ || expert >= n_expert_) return false;
    const size_t index = (size_t) layer * (size_t) n_expert_ + (size_t) expert;
    if (index >= complement_offsets_.size() || complement_offsets_[index] == kNoComplement) return false;
    // a partial pin (CS-T): only the registered prefix
    return !complement_partial_ ||
           complement_offsets_[index] + layer_blob_bytes_[(size_t) layer] <= complement_pin_limit_;
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
    if (d.lookahead != nullptr) d.lookahead->submit(d.layers, x_f, n_tok, d.host_res);   // CS-T: warm layer + 1
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
                        const uint8_t* src = d.src->pinned(d.layers, e) ? d.src->blob(d.layers, e) : nullptr;
                        if (src != nullptr) {
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
    {   // CS-T: the experts the CPU computes, fetched together (the GGUF in place reads them on several threads)
        static thread_local std::vector<int64_t> miss;
        miss.clear();
        for (int64_t i = 0; i < n_tok * k; ++i)
            if (kind[i] < 0 && ids[i] >= 0 && ids[i] < d.n_expert &&
                std::find(miss.begin(), miss.end(), (int64_t) ids[i]) == miss.end())
                miss.push_back(ids[i]);
        d.src->prefetch(d.layers, miss.data(), (int64_t) miss.size());
    }
    // #95: with the overlap on (default; STRATA_NVME_OVERLAP=0 is the A/B arm), the layer's misses are submitted here and
    // collected only after the pool has run the resident experts; the missed experts' jobs run in a second pool run.
    static const bool nvme_overlap = [] {
        const char* v = std::getenv("STRATA_NVME_OVERLAP");
        return v == nullptr || std::atoi(v) != 0;
    }();
    auto fail_nvme = [&](const std::string& ne) {
        d.failed = true;
        d.nvme_fail = "an NVMe-tier expert could not be read: " + ne;   // #62: e.g. a wrong mirror
        d.fail = d.nvme_fail.c_str();
        d.fail_layer = d.layers;
    };
    bool nvme_pending = false;
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
            if (!d.src->materialize_begin(d.layers, miss, nm, ne) || !(nvme_overlap || d.src->materialize_end(ne))) {
                fail_nvme(ne);
                return;
            }
            nvme_pending = nvme_overlap;
        }
    }
    // #95: a failure below returns with the batch in flight; ending it on the way out releases host_mu_ (else the next
    // hold()/materialize() waits forever and the failure becomes a hang).  A no-op once the batch has ended.
    struct EndBatch {
        ExpertSource* src;
        ~EndBatch() { std::string e; src->materialize_end(e); }
    } end_batch{d.src};
    int njobs = 0;
    // routed entry i onto its expert's job, made at the expert's first entry; false: the dispatch failed
    auto add_entry = [&](int64_t i) -> bool {
        const int64_t t = i / k, e = ids[i];
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
                return false;
            }
            jo = (int16_t) njobs++;
            ExpertJobMulti& nj = d.jobs_multi[(size_t) jo];
            nj.blob = b;
            nj.nt = 0;
        }
        ExpertJobMulti& jb = d.jobs_multi[(size_t) jo];
        jb.act[jb.nt] = &d.act_multi[(size_t) t];
        jb.nact[jb.nt] = native ? d.nact_multi.data() + (size_t) t * kNativeActBytes : nullptr;
        jb.out[jb.nt] = out + (size_t) i * H;
        ++jb.nt;
        ++d.multi_entries;
        ++d.tier_entries[3];
        return true;
    };
    auto run_jobs = [&](int first, int count) {
        if (native) d.pool->run_split_multi_native(lay.fmt[(size_t) d.layers], d.jobs_multi.data() + first, count);
        else d.pool->run_split_multi(d.jobs_multi.data() + first, count);
    };
    static thread_local std::vector<int32_t> deferred;   // #95: routed entries whose expert is still being read
    deferred.clear();
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
            // #95: still in flight (or past the batch's 128): after materialize_end, in the second pass - a
            // materialize here would wait on the host tier this thread holds.  resident(), not blob(): blob() counts a
            // use, and add_entry's lookup is the one use
            if (nvme_pending && d.job_of[(size_t) e] < 0 && !d.src->resident(d.layers, e)) {
                deferred.push_back((int32_t) i);
                continue;
            }
            if (!add_entry(i)) return;
        }
    const auto c3 = std::chrono::steady_clock::now();
    pt("run", njobs);
    const auto pool_start = std::chrono::steady_clock::now();
    run_jobs(0, njobs);
    const auto pool_end = std::chrono::steady_clock::now();
    if (nvme_pending) {   // #95: the reads have had the resident run to land; publish them, then run their jobs
        std::string ne;
        if (!d.src->materialize_end(ne)) { fail_nvme(ne); return; }
        const auto p1 = std::chrono::steady_clock::now();
        const int first = njobs;
        for (const int32_t i : deferred)
            if (!add_entry(i)) return;
        const auto p2 = std::chrono::steady_clock::now();
        if (njobs > first) run_jobs(first, njobs - first);
        const auto p3 = std::chrono::steady_clock::now();
        d.ms_cpu_pool += std::chrono::duration<double, std::milli>(p3 - p2).count();
        if (timeline::enabled()) {   // own names: decode_paths.py pairs one "cpu pool" per layer
            timeline::complete("nvme collect", pool_end, p1, d.layers, (int64_t) deferred.size());
            timeline::complete("cpu pool misses", p2, p3, d.layers, njobs - first);
        }
    }
    const auto cpu_end = std::chrono::steady_clock::now();
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
            timeline::complete("4070 finish", cpu_end, c4, d.layers, (int64_t) d.secondary_runner->served_entries());
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

namespace {
/// The GGUF file that holds role `r` (0 gate, 1 up, 2 down) of layer `l`: a name beside the --native shard
/// (native_experts.txt v3 per layer, v4 per role), or the --native shard itself.
std::string expert_gguf_file(const std::string& gguf, const strata::kernels::cpu::ExpertLayout& lay, int64_t l, int r) {
    const size_t i = (size_t) (3 * l + r);
    if (lay.gguf_file.size() <= i || lay.gguf_file[i].empty()) return gguf;
    const size_t cut = gguf.find_last_of("/\\");
    return (cut == std::string::npos ? std::string() : gguf.substr(0, cut + 1)) + lay.gguf_file[i];
}
}  // namespace

bool check_experts_gguf(const std::string& gguf, const strata::kernels::cpu::ExpertLayout& lay, std::string& err) {
    static const char* roles[3] = {"gate", "up", "down"};
    if (lay.gguf_off.size() != (size_t) (3 * lay.n_layers)) {
        err = "native_experts.txt has no GGUF offsets (a pack older than v2): repack it with tools/iq_pack.py";
        return false;
    }
    try {
        std::map<std::string, std::unique_ptr<strata::GgufFile>> files;
        for (int64_t l = 0; l < lay.n_layers; ++l) {
            const auto& fm = lay.fmt[(size_t) l];
            const uint64_t blob = lay.bytes[(size_t) l];
            const uint64_t per[3] = {fm.up_off, fm.up_off, blob - fm.down_off};
            for (int r = 0; r < 3; ++r) {
                const std::string path = expert_gguf_file(gguf, lay, l, r);
                auto& f = files[path];
                if (!f) f = std::make_unique<strata::GgufFile>(path);
                const std::string name = "blk." + std::to_string(l) + ".ffn_" + roles[r] + "_exps.weight";
                const strata::TensorInfo* t = f->find(name);
                const uint64_t want_type = (uint64_t) (r < 2 ? fm.gu_type : fm.d_type);
                // GGUF order: dim 0 is the row (the input), dim 1 the rows, dim 2 the experts
                const uint64_t cols = (uint64_t) (r < 2 ? fm.n_embd : fm.n_ff);
                const uint64_t rows = (uint64_t) (r < 2 ? fm.n_ff : fm.n_embd);
                const uint64_t bytes = per[r] * (uint64_t) lay.n_expert;
                const uint64_t payload = f->file_size() - f->data_start();
                std::string why;
                if (t == nullptr) why = "is not in it";
                else if (t->type != want_type)
                    why = std::string("is ") + t->type_name() + ", the pack says type " + std::to_string(want_type);
                else if (t->shape.size() != 3 || t->shape[0] != cols || t->shape[1] != rows ||
                         t->shape[2] != (uint64_t) lay.n_expert)
                    why = "is not [" + std::to_string(cols) + ", " + std::to_string(rows) + ", " +
                          std::to_string(lay.n_expert) + "]";
                else if (strata::tensor_payload_bytes(*t) != bytes)
                    why = "is not " + std::to_string(per[r]) + " B per expert";
                else if (f->data_start() + t->offset != lay.gguf_off[(size_t) (3 * l + r)])
                    why = "starts at byte " + std::to_string(f->data_start() + t->offset) + ", the pack says " +
                          std::to_string(lay.gguf_off[(size_t) (3 * l + r)]);
                else if (t->offset > payload || bytes > payload - t->offset)
                    why = "runs past the end of the file (a truncated shard?)";
                if (!why.empty()) {
                    err = "the pack's native_experts.txt does not match the model: " + name + " in " + path + " " +
                          why + " - repack with tools/iq_pack.py from this model's shards";
                    return false;
                }
            }
        }
        return true;
    } catch (const std::exception& e) {
        err = std::string("native experts from the GGUF: ") + e.what();
        return false;
    }
}

// Plan v0.3 P6: the arena from the model's GGUF shards.  Each layer's gate, up and down tensors hold the 512
// experts one after another; they are read in chunks and each expert's slice lands at its place in the blob
// [gate rows | up rows | down rows] - the layout tools/iq_pack.py would have written to experts.bin.  Each role
// is read from its own file (native_experts.txt v4: a shard boundary can fall inside a layer; per role as in
// #255, gopinath87607).  The caller checks the spans first (check_experts_gguf).
LoadStats load_experts_gguf(const std::string& gguf, uint8_t* dst, const strata::kernels::cpu::ExpertLayout& lay,
                            int threads, const uint8_t* skip) {
    LoadStats st;
    st.layers = (uint64_t) lay.n_layers;
    const auto t0 = std::chrono::steady_clock::now();
    std::atomic<int64_t> next{0};
    std::atomic<bool> bad{false};
    auto worker = [&]() {
        // #230: `fread` on a `FILE*`, as load_experts_ranges (#89): MSVC's `std::ifstream::read` splits a request
        // into 4095-byte freads, which took this path to 0.02 GiB/s on a Windows install without experts.bin.
        // The guard closes the handle on every return.
        struct Closer {
            FILE* f = nullptr;
            ~Closer() { if (f != nullptr) std::fclose(f); }
        } file;
        std::string open_name;
        std::vector<uint8_t> buf;
        for (;;) {
            const int64_t l = next.fetch_add(1);
            if (l >= lay.n_layers || bad) break;
            const auto& fm = lay.fmt[(size_t) l];
            const uint64_t blob = lay.bytes[(size_t) l];
            const uint64_t per[3] = {fm.up_off, fm.up_off, blob - fm.down_off};
            const uint64_t at[3] = {0, fm.up_off, fm.down_off};
            for (int r = 0; r < 3; ++r) {
                // the handle is kept while consecutive roles share a file (every layer of a v3 pack)
                const std::string name = expert_gguf_file(gguf, lay, l, r);
                if (name != open_name) {
                    if (file.f != nullptr) std::fclose(file.f);
                    file.f = std::fopen(name.c_str(), "rb");
                    if (file.f == nullptr) { bad = true; return; }
                    open_name = name;
                }
                FILE* const f = file.f;
                const uint64_t src = lay.gguf_off[(size_t) (3 * l + r)];
                const uint64_t total = per[r] * (uint64_t) lay.n_expert;
                const uint64_t chunk = per[r] * 16;           // 16 experts per read
                buf.resize((size_t) chunk);
                for (uint64_t done = 0; done < total; done += chunk) {
                    const uint64_t n = std::min<uint64_t>(chunk, total - done);
                    // 64-bit seek: a shard is tens of GB
                    if (STRATA_FSEEK64(f, src + done) != 0) { bad = true; return; }
                    if (std::fread(buf.data(), 1, (size_t) n, f) != (size_t) n) { bad = true; return; }
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
    if (from_gguf) {
        // every (file, offset) of native_experts.txt must be the tensor it claims, of the pack's type and
        // dimensions and inside its file - before the allocation, so a pack of another model or a truncated
        // shard is a message rather than an arena of plausible wrong experts
        if (!check_experts_gguf(gguf_, lay, err)) { err = "ArenaExpertSource: " + err; return false; }
    } else {
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
    // the file is reopened only when it changes: per role, since native_experts.txt v4 may put a layer's roles in
    // different shards (UD-Q4_K_XL's layer 11); every other layer keeps one handle for all three
    auto open_name = [&](const std::string& name) {
        if (name != rf_name_ || !rf_.is_open()) {
            rf_.close();
            rf_.clear();
            rf_.open(name, std::ios::binary);
            if (!rf_) { err = "read_expert: cannot open " + name; rf_name_.clear(); return false; }
            rf_name_ = name;
        }
        return true;
    };
    auto read_at = [&](uint64_t off, uint8_t* to, uint64_t n) {
        rf_.clear();
        rf_.seekg((std::streamoff) off);
        rf_.read((char*) to, (std::streamsize) n);
        return (uint64_t) rf_.gcount() == n;
    };
    if (!from_gguf_) {
        if (!open_name(path_)) return false;
        if (!read_at(lay.blob_offset(layer, expert), dst, lay.blob_bytes(layer))) { err = "read_expert: short read"; return false; }
        return true;
    }
    // the GGUF holds each role's 512 experts one after another: gate rows | up rows | down rows
    const auto& fm = lay.fmt[(size_t) layer];
    const uint64_t blob = lay.bytes[(size_t) layer];
    const uint64_t per[3] = {fm.up_off, fm.up_off, blob - fm.down_off};
    const uint64_t at[3] = {0, fm.up_off, fm.down_off};
    for (int r = 0; r < 3; ++r) {
        if (!open_name(expert_gguf_file(gguf_, lay, layer, r))) return false;
        if (!read_at(lay.gguf_off[(size_t) (3 * layer + r)] + (uint64_t) expert * per[r], dst + at[r], per[r])) {
            err = "read_expert: short read";
            return false;
        }
    }
    return true;
}

namespace {
// The file that holds range `r` of expert_ranges: experts.bin, or role r's GGUF shard (per role since v4).
std::string expert_file(const std::string& gguf, const std::string& path, bool from_gguf,
                        const strata::kernels::cpu::ExpertLayout& lay, int64_t layer, int r) {
    return from_gguf ? expert_gguf_file(gguf, lay, layer, r) : path;
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

// #62: the same bytes at the same offsets?  Sizes first, then the first and last MiB and eight 64 KiB pages spread
// over the file: a copy cut short, of another quant or another revision differs in its header or tensor data.
static bool same_file_sampled(const std::string& a, const std::string& b, std::string& why) {
    namespace fs = std::filesystem;
    std::error_code ea, eb;
    const uint64_t na = fs::file_size(a, ea), nb = fs::file_size(b, eb);
    if (ea || eb) { why = "cannot read its size"; return false; }
    if (na != nb) { why = "size " + std::to_string(nb) + " B, the source has " + std::to_string(na); return false; }
    std::ifstream fa(a, std::ios::binary), fb(b, std::ios::binary);
    if (!fa || !fb) { why = "cannot open it"; return false; }
    constexpr uint64_t MIB = 1ull << 20, PAGE = 64ull << 10;
    std::vector<std::pair<uint64_t, uint64_t>> at = {{0, std::min(MIB, na)}, {na > MIB ? na - MIB : 0, std::min(MIB, na)}};
    uint64_t x = na ^ 0x9e3779b97f4a7c15ull;
    for (int i = 0; i < 8 && na > PAGE; ++i) {
        x = x * 6364136223846793005ull + 1442695040888963407ull;
        at.emplace_back((x >> 11) % (na - PAGE) & ~(uint64_t) 4095, PAGE);
    }
    std::vector<char> ba, bb;
    for (const auto& [off, len] : at) {
        ba.resize((size_t) len);
        bb.resize((size_t) len);
        fa.seekg((std::streamoff) off);
        fb.seekg((std::streamoff) off);
        if (!fa.read(ba.data(), (std::streamsize) len) || !fb.read(bb.data(), (std::streamsize) len)) {
            why = "short read at " + std::to_string(off);
            return false;
        }
        if (std::memcmp(ba.data(), bb.data(), (size_t) len) != 0) { why = "bytes differ at " + std::to_string(off); return false; }
    }
    return true;
}

bool ArenaExpertSource::copies_of(const std::string& source, const std::vector<std::string>*& out, std::string& err) {
    for (const auto& [src, cs] : copies_) if (src == source) { out = &cs; return true; }
    std::vector<std::string> cs;
    const std::string name = std::filesystem::path(source).filename().string();
    std::error_code sz;
    const uint64_t na = std::filesystem::file_size(source, sz);
    const int regions = na > (64ull << 10) ? 10 : 2;   // same_file_sampled: first and last MiB, then 8 pages
    for (const std::string& dir : mirror_dirs_) {
        const std::string copy = (std::filesystem::path(dir) / name).string();
        std::error_code ec;
        if (!std::filesystem::exists(copy, ec)) continue;           // this drive holds no copy of this file
        if (std::filesystem::equivalent(copy, source, ec)) continue;
        std::string why;
        if (!same_file_sampled(source, copy, why)) {
            err = "expert mirror " + copy + " is not a copy of " + source + ": " + why;
            return false;
        }
        std::fprintf(stderr, "strata experts: mirror %s verified (size and %d sampled regions match %s)\n",
                     copy.c_str(), regions, source.c_str());
        cs.push_back(copy);
    }
    copies_.emplace_back(source, std::move(cs));
    out = &copies_.back().second;
    return true;
}

bool ArenaExpertSource::read_experts_to(const int32_t* layers, const int32_t* experts, int n, uint8_t* const* dsts,
                                        std::string& err) {
    return submit_reads(layers, experts, n, dsts, /*into_slots=*/false, err) && collect_reads(err);
}

// #81: the expert pack's alignment, which is also the slab's slot stride unit and the direct reader's sector: a pack
// expert then reads straight into its slot
constexpr uint64_t kPackAlign = 4096;
static uint64_t round_up(uint64_t x, uint64_t a) { return (x + a - 1) / a * a; }

bool ArenaExpertSource::submit_reads(const int32_t* layers, const int32_t* experts, int n, uint8_t* const* dsts,
                                     bool into_slots, std::string& err) {
    using strata::platform::DirectFile;
    const auto& lay = strata::kernels::cpu::expert_layout();
    constexpr uint64_t A = DirectFile::alignment();
    static_assert(A == kPackAlign, "an expert pack's alignment is the direct reader's");
    const size_t slot_bytes = ((size_t) lay.max_blob + 2 * A + A - 1) / A * A;   // one aligned range, with its edges
    const size_t need = (size_t) n * 3 * slot_bytes;
    if (need > dscratch_bytes_) {
        if (dscratch_ != nullptr) DirectFile::free_aligned(dscratch_);
        dscratch_ = DirectFile::alloc_aligned(need);
        dscratch_bytes_ = dscratch_ ? need : 0;
        if (dscratch_ == nullptr) { err = "read_experts: bounce buffer"; return false; }
    }
    auto open_file = [&](const std::string& name) -> int {   // the file's index in dfiles_, -1 on error
        for (size_t i = 0; i < dfiles_.size(); ++i) if (dfiles_[i].name == name) return (int) i;
        auto* f = new DirectFile();
        if (!f->open(name, err)) { delete f; return -1; }
        DFile d;
        d.name = name;
        d.file = f;
        d.st.path = name;
        dfiles_.push_back(std::move(d));
        return (int) dfiles_.size() - 1;
    };
    auto& reqs = pend_.reqs;
    reqs.clear();
    reqs.reserve((size_t) n * 3);
    bool used_mirror = false;
    pend_.slot_bytes = slot_bytes;
    std::vector<uint64_t> queued;   // #62: bytes queued per dfiles_ index in this batch
    auto queued_of = [&](int i) -> uint64_t& {
        if ((size_t) i >= queued.size()) queued.resize((size_t) i + 1, 0);
        return queued[(size_t) i];
    };
    const double t0 = strata::timeline::now_us();
    pend_.t0 = t0;
    for (int i = 0; i < n; ++i) {
        uint64_t off[3], len[3], at[3];
        const bool packed = !pack_path_.empty();   // #81: one contiguous, aligned range per expert
        int nr;
        if (packed) {
            off[0] = pack_off_[(size_t) layers[i]] + (uint64_t) experts[i] * pack_stride_[(size_t) layers[i]];
            len[0] = lay.blob_bytes(layers[i]);
            at[0] = 0;
            nr = 1;
        } else {
            nr = expert_ranges(lay, from_gguf_, layers[i], experts[i], off, len, at);
        }
        // each range's file: experts.bin or the pack (one range), else role r's GGUF shard - per role, since v4 can
        // put a shard boundary inside a layer.  Consecutive ranges in one file go to one copy together: for every
        // layer whose roles share a shard (all of a v3 pack) that is the whole expert, as before; an expert split
        // across shards counts one read in each file it touches.
        std::string srcs[3];
        for (int r = 0; r < nr; ++r)
            srcs[r] = packed ? pack_path_ : expert_file(gguf_, path_, from_gguf_, lay, layers[i], r);
        for (int r0 = 0, r1 = 0; r0 < nr; r0 = r1) {
            r1 = r0 + 1;
            while (r1 < nr && srcs[r1] == srcs[r0]) ++r1;
            const std::string& source = srcs[r0];
            int fi = open_file(source);
            if (fi < 0) return false;
            if (!mirror_dirs_.empty()) {   // #62: the whole expert (its ranges in this file) to the least-queued copy
                const std::vector<std::string>* cs = nullptr;
                if (!copies_of(source, cs, err)) return false;
                int cand[8], nc = 0;
                cand[nc++] = fi;
                for (const std::string& c : *cs) {
                    const int gi = open_file(c);
                    if (gi < 0) return false;
                    if (nc < 8) cand[nc++] = gi;
                }
                // #82: ties go to the copy this batch starts at, one further on each batch - a decode layer misses
                // ~0.3 experts, so most batches hold one, and "ties to the source" sent nearly every read to the source
                const int start = (int) (mirror_turn_ % (uint64_t) nc);
                fi = cand[start];
                for (int k = 1; k < nc; ++k) {
                    const int gi = cand[(start + k) % nc];
                    if (queued_of(gi) < queued_of(fi)) fi = gi;
                }
                used_mirror = true;
            }
            DFile& d = dfiles_[(size_t) fi];
            ++d.st.reads;
            for (int r = r0; r < r1; ++r) {
                const uint64_t a0 = off[r] & ~(A - 1), a1 = (off[r] + len[r] + A - 1) & ~(A - 1);
                // #81: a pack's expert into a slab slot directly: set_expert_pack checked its offset is aligned, a
                // slot is page-aligned and its stride is the read's length (the blob rounded up to kPackAlign).  Only
                // the caller knows its destinations are slots; anything else lands in the bounce buffer and is copied
                const bool direct = packed && into_slots && slab_;
                uint8_t* buf = direct ? dsts[i] : (uint8_t*) dscratch_ + reqs.size() * slot_bytes;
                if (!((DirectFile*) d.file)->submit(a0, buf, (uint32_t) (a1 - a0), (uint64_t) reqs.size(), err))
                    return false;
                reqs.push_back({fi, off[r] - a0, len[r], dsts[i] + at[r], direct});
                d.st.bytes += a1 - a0;
                queued_of(fi) += a1 - a0;
            }
        }
    }
    if (used_mirror && mirror_rotate_) ++mirror_turn_;
    return true;
}

bool ArenaExpertSource::set_expert_pack(const std::string& path, std::string& err) {
    const auto& lay = strata::kernels::cpu::expert_layout();
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) { err = "expert pack: cannot open " + path; return false; }
    const uint64_t size = (uint64_t) f.tellg();
    std::vector<uint8_t> head(kPackAlign);
    f.seekg(0);
    if (size < head.size() || !f.read((char*) head.data(), (std::streamsize) head.size())) {
        err = "expert pack: " + path + " is shorter than its header";
        return false;
    }
    uint32_t h32[4];
    std::memcpy(h32, head.data() + 8, sizeof h32);
    if (std::memcmp(head.data(), "STRAPACK", 8) != 0 || h32[0] != 1) { err = "expert pack: " + path + " is not a v1 pack"; return false; }
    if (h32[3] != kPackAlign) {
        err = "expert pack: " + path + " has alignment " + std::to_string(h32[3]) + ", this engine reads " +
              std::to_string(kPackAlign);
        return false;
    }
    if ((int64_t) h32[1] != lay.n_layers || (int64_t) h32[2] != n_expert_ || 24 + 24 * (uint64_t) h32[1] > head.size()) {
        err = "expert pack: " + path + " holds " + std::to_string(h32[1]) + " layers x " + std::to_string(h32[2]) +
              " experts, the model " + std::to_string(lay.n_layers) + " x " + std::to_string(n_expert_);
        return false;
    }
    std::vector<uint64_t> off((size_t) lay.n_layers), stride((size_t) lay.n_layers);
    for (int64_t l = 0; l < lay.n_layers; ++l) {
        uint64_t row[3];
        std::memcpy(row, head.data() + 24 + 24 * l, sizeof row);
        if (row[2] != lay.blob_bytes(l) || row[1] < row[2] || row[0] % kPackAlign || row[1] % kPackAlign ||
            row[0] + (uint64_t) n_expert_ * row[1] > size) {
            err = "expert pack: " + path + " layer " + std::to_string(l) + " does not match the model's blobs";
            return false;
        }
        off[(size_t) l] = row[0];
        stride[(size_t) l] = row[1];
    }
    pack_path_ = path;
    pack_off_ = std::move(off);
    pack_stride_ = std::move(stride);
    return true;
}

bool ArenaExpertSource::write_expert_pack(const std::string& path, std::string& err) {
    const auto& lay = strata::kernels::cpu::expert_layout();
    constexpr uint64_t A = kPackAlign;
    std::vector<uint8_t> head(A, 0);
    std::memcpy(head.data(), "STRAPACK", 8);
    const uint32_t h32[4] = {1, (uint32_t) lay.n_layers, (uint32_t) n_expert_, (uint32_t) A};
    std::memcpy(head.data() + 8, h32, sizeof h32);
    if (24 + 24 * (uint64_t) lay.n_layers > A) { err = "expert pack: too many layers for the header"; return false; }
    uint64_t at = A, max_stride = 0;
    std::vector<uint64_t> strides((size_t) lay.n_layers);
    for (int64_t l = 0; l < lay.n_layers; ++l) {
        const uint64_t blob = lay.blob_bytes(l), stride = round_up(blob, A);
        const uint64_t row[3] = {at, stride, blob};
        std::memcpy(head.data() + 24 + 24 * l, row, sizeof row);
        at += (uint64_t) n_expert_ * stride;
        max_stride = std::max(max_stride, stride);
        strides[(size_t) l] = stride;
    }
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) { err = "expert pack: cannot create " + path; return false; }
    out.write((const char*) head.data(), (std::streamsize) A);
    std::vector<uint8_t> e((size_t) max_stride);
    const auto t0 = std::chrono::steady_clock::now();
    for (int64_t l = 0; l < lay.n_layers; ++l) {
        const uint64_t stride = strides[(size_t) l];
        for (int64_t x = 0; x < n_expert_; ++x) {
            std::fill(e.begin(), e.begin() + (std::ptrdiff_t) stride, (uint8_t) 0);
            if (!read_expert(l, x, e.data(), err)) return false;
            out.write((const char*) e.data(), (std::streamsize) stride);
        }
        if (!out) { err = "expert pack: writing " + path + " failed (disk full?)"; return false; }
        std::fprintf(stderr, "strata expert pack: layer %lld of %lld written (%.1f GiB, %.0f s)\n", (long long) l + 1,
                     (long long) lay.n_layers, (double) out.tellp() / 1073741824.0,
                     std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    }
    out.close();
    if (!out) { err = "expert pack: closing " + path + " failed"; return false; }
    return true;
}

bool ArenaExpertSource::collect_reads(std::string& err) {
    using strata::platform::DirectFile;
    using Req = PendingReads::Req;
    const auto& reqs = pend_.reqs;
    const size_t slot_bytes = pend_.slot_bytes;
    const double t0 = pend_.t0;
    // Collect every completion.  Each file's port reports its own requests; the ports are polled in turn, so each
    // copy's reads are timed when they land rather than after the copy drained before it (#62: a second drive
    // must be able to show that it was faster).  With nothing ready, one port is waited on for 1 ms.
    const size_t nf = dfiles_.size();
    std::vector<size_t> left(nf, 0), sent(nf, 0);
    for (const Req& q : reqs) { ++left[(size_t) q.file]; ++sent[(size_t) q.file]; }
    std::vector<double> last(nf, t0);
    size_t got = 0;
    auto take = [&](size_t fi, int timeout_ms) -> bool {   // false on a failed read
        strata::platform::Completion c[64];
        const auto tw = std::chrono::steady_clock::now();
        const int k = ((DirectFile*) dfiles_[fi].file)->wait(c, 64, timeout_ms);
        const auto tk = std::chrono::steady_clock::now();
        stages_.wait_ms += std::chrono::duration<double, std::milli>(tk - tw).count();
        const double t = strata::timeline::now_us();
        bool copied = false;
        for (int j = 0; j < k; ++j) {
            if (c[j].tag == DirectFile::WAKE_TAG) continue;
            const Req& q = reqs[(size_t) c[j].tag];
            if (!c[j].ok || c[j].bytes < q.skip + q.len) { err = "read_experts: short read"; return false; }
            if (!q.direct) {   // #81: a pack read straight into its slot has nothing to copy
                std::memcpy(q.to, (uint8_t*) dscratch_ + (size_t) c[j].tag * slot_bytes + q.skip, (size_t) q.len);
                copied = true;
            }
            ++got;
            --left[fi];
            last[fi] = t;
            dfiles_[fi].lat_us[dfiles_[fi].lat_next++ % DFile::kLat] = (float) (t - t0);
            dfiles_[fi].lat_n = std::min<uint64_t>(dfiles_[fi].lat_n + 1, DFile::kLat);
        }
        if (copied) stages_.copy_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tk).count();
        return true;
    };
    while (got < reqs.size()) {
        const size_t before = got;
        for (size_t fi = 0; fi < nf; ++fi)
            if (left[fi] > 0 && !take(fi, 0)) return false;
        if (got == before)
            for (size_t fi = 0; fi < nf; ++fi)
                if (left[fi] > 0) { if (!take(fi, 1)) return false; break; }
    }
    for (size_t fi = 0; fi < nf; ++fi) {
        if (sent[fi] == 0) continue;
        dfiles_[fi].st.ms += (last[fi] - t0) / 1000.0;
        ++dfiles_[fi].st.batches;
        if (strata::timeline::enabled())   // #62: one span per copy and batch: submit to its last completion
            strata::timeline::complete("nvme copy read", t0, last[fi], (int64_t) fi, (int64_t) sent[fi]);
    }
    return true;
}

std::vector<ArenaExpertSource::NvmeFileStat> ArenaExpertSource::nvme_file_stats() const {
    std::vector<NvmeFileStat> out;
    for (const DFile& d : dfiles_) {
        NvmeFileStat s = d.st;
        std::vector<float> v(d.lat_us.begin(), d.lat_us.begin() + (std::ptrdiff_t) d.lat_n);
        if (!v.empty()) {
            auto at = [&](double q) {
                const size_t k = std::min(v.size() - 1, (size_t) (q * (double) (v.size() - 1) + 0.5));
                std::nth_element(v.begin(), v.begin() + (std::ptrdiff_t) k, v.end());
                return (double) v[k];
            };
            s.p50_us = at(0.50);
            s.p99_us = at(0.99);
            s.max_us = *std::max_element(v.begin(), v.end());
        }
        out.push_back(std::move(s));
    }
    return out;
}

// load_rest's reader: each worker takes a layer and reads each role's 512 slices in aligned chunks of 32 experts
// with its own unbuffered file, copying only the host-owned experts into the arena.
static LoadStats load_experts_gguf_direct(const std::string& gguf, uint8_t* dst,
                                          const strata::kernels::cpu::ExpertLayout& lay, int threads,
                                          const uint8_t* skip, uint8_t* const* dst_of = nullptr) {
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
            const auto& fm = lay.fmt[(size_t) l];
            const uint64_t blob = lay.bytes[(size_t) l];
            const uint64_t per[3] = {fm.up_off, fm.up_off, blob - fm.down_off};
            const uint64_t at[3] = {0, fm.up_off, fm.down_off};
            for (int r = 0; r < 3 && !bad; ++r) {
                // per role (v4: a shard boundary can fall inside a layer); the handle is kept while roles share a file
                const std::string name = expert_gguf_file(gguf, lay, l, r);
                if (name != open_name) {
                    f.close();
                    if (!f.open(name, e)) { bad = true; break; }
                    open_name = name;
                }
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
                        uint8_t* to = dst_of != nullptr ? dst_of[(size_t) (l * lay.n_expert + x)]
                                                        : dst + lay.blob_offset(l, x);
                        std::memcpy(to + at[r], (uint8_t*) buf + (off - a0) + (uint64_t) (x - e0) * per[r],
                                    (size_t) per[r]);
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
    std::vector<uint8_t*> dst_of;
    if (slab_) dst_of.assign(exclusive_.size(), nullptr);
    for (size_t i = 0; i < exclusive_.size(); ++i) {
        if (skip[i]) continue;
        uint64_t held = 0;
        if (!host_commit(i, held, err)) return false;
        if (slab_) dst_of[i] = host_at(i);
        kept += lay.blob_bytes((int64_t) i / lay.n_expert);
    }
    if (from_gguf_) {
        const LoadStats st = load_experts_gguf_direct(gguf_, const_cast<uint8_t*>(base_), lay, threads, skip.data(),
                                                      dst_of.empty() ? nullptr : dst_of.data());
        if (st.seconds < 0) { err = "load_rest: the GGUF read failed"; return false; }
    } else {
        std::string e;
        for (int64_t l = 0; l < lay.n_layers; ++l)
            for (int64_t x = 0; x < lay.n_expert; ++x)
                if (!skip[(size_t) (l * lay.n_expert + x)] &&
                    !read_expert(l, x, host_at((size_t) (l * lay.n_expert + x)), e)) { err = e; return false; }
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
    slab_release();
    // the slab: capacity mode on a deferred (placement-first), unpinned arena (Windows).  STRATA_NVME_SLOTS=0 keeps
    // the fixed addresses (the A/B arm).  One reserved region per blob size, room for every expert of its layers;
    // the byte cap still decides residency and eviction, the slab only stores.
    const char* sv = std::getenv("STRATA_NVME_SLOTS");
#if defined(_WIN32)
    if (bytes > 0 && deferred_ && pinned_bytes_ == 0 && !(sv != nullptr && *sv && std::atoi(sv) == 0)) {
        const char* sk = std::getenv("STRATA_NVME_SLACK_MIB");
        slack_bytes_ = (uint64_t) (sk != nullptr && *sk ? std::max(0, std::atoi(sk)) : 512) << 20;
        class_of_layer_.assign((size_t) lay.n_layers, -1);
        for (int64_t l = 0; l < lay.n_layers; ++l) {
            const size_t stride = (size_t) round_up(lay.blob_bytes(l), kPackAlign);
            int16_t c = -1;
            for (size_t k = 0; k < classes_.size(); ++k) if (classes_[k].stride == stride) c = (int16_t) k;
            if (c < 0) { classes_.push_back(SlabClass{}); c = (int16_t) (classes_.size() - 1); classes_.back().stride = stride; }
            class_of_layer_[(size_t) l] = c;
            classes_[(size_t) c].cap += (int32_t) lay.n_expert;
        }
        slab_ = true;
        for (SlabClass& k : classes_) {
            k.base = (uint8_t*) VirtualAlloc(nullptr, (size_t) k.cap * k.stride, MEM_RESERVE, PAGE_READWRITE);
            if (k.base == nullptr) slab_ = false;
        }
        if (!slab_) slab_release();
        else slot_of_.assign(exclusive_.size(), -1);
    }
#else
    (void) sv;
#endif
    nvme_.assign(exclusive_.size(), 0);
    score_.assign(exclusive_.size(), 0.0f);
    held_.assign(exclusive_.size(), 0);
    {
        const char* es = std::getenv("STRATA_NVME_EVICT_SCAN");
        evict_scan_ = es != nullptr && *es && std::atoi(es) != 0;
        const size_t nl = n_expert_ > 0 ? exclusive_.size() / (size_t) n_expert_ : 0;
        lmin_.assign(nl, 0.0f);
        larg_.assign(nl, -1);
        ldirty_.assign(nl, 1);
    }
    if (bytes == 0) { nvme_.clear(); score_.clear(); held_.clear(); return; }
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

void ArenaExpertSource::hold(int64_t layer, int64_t expert) {
    std::lock_guard<std::mutex> lk(host_mu_);
    const size_t i = (size_t) (layer * n_expert_ + expert);
    if (i < held_.size() && held_[i] < 255) { ++held_[i]; touch_up(i); }
}

void ArenaExpertSource::release_hold(int64_t layer, int64_t expert) {
    std::lock_guard<std::mutex> lk(host_mu_);
    const size_t i = (size_t) (layer * n_expert_ + expert);
    if (i < held_.size() && held_[i] > 0) { --held_[i]; touch(i); }
}

bool ArenaExpertSource::evict_one(int64_t avoid_layer) {
    const auto& lay = strata::kernels::cpu::expert_layout();
    size_t victim = SIZE_MAX;
    float best = 0.0f;
    if (evict_scan_ || ldirty_.empty()) {   // the reference: every expert
        for (size_t i = 0; i < nvme_.size(); ++i) {
            if (nvme_[i] || exclusive_[i] || held_[i] || (int64_t) i / lay.n_expert == avoid_layer) continue;
            if (victim == SIZE_MAX || score_[i] < best) { victim = i; best = score_[i]; }
        }
    } else {   // #97: the layers' cached minima, a dirty layer rescanned; layer order keeps the lowest-index tie
        const size_t ne = (size_t) lay.n_expert;
        for (size_t l = 0; l < ldirty_.size(); ++l) {
            if ((int64_t) l == avoid_layer) continue;
            if (ldirty_[l]) {
                int32_t arg = -1;
                float m = 0.0f;
                for (size_t i = l * ne; i < (l + 1) * ne; ++i) {
                    if (nvme_[i] || exclusive_[i] || held_[i]) continue;
                    if (arg < 0 || score_[i] < m) { arg = (int32_t) i; m = score_[i]; }
                }
                larg_[l] = arg;
                lmin_[l] = m;
                ldirty_[l] = 0;
            }
            if (larg_[l] >= 0 && (victim == SIZE_MAX || lmin_[l] < best)) { victim = (size_t) larg_[l]; best = lmin_[l]; }
        }
    }
    if (victim == SIZE_MAX) return false;
    touch(victim);
    uint64_t given = 0;
    std::string e;
    if (!host_release(victim, given, e)) return false;   // a slot goes idle: the next load overwrites it
    nvme_[victim] = 1;
    cache_used_ -= lay.blob_bytes((int64_t) victim / lay.n_expert);
    return true;
}

const uint8_t* ArenaExpertSource::materialize(int64_t layer, int64_t expert, int64_t avoid_layer, std::string& err) {
    std::lock_guard<std::mutex> lk(host_mu_);
    return materialize_locked(layer, expert, avoid_layer, err);
}

const uint8_t* ArenaExpertSource::hold_resident(int64_t layer, int64_t expert) {
    std::lock_guard<std::mutex> lk(host_mu_);
    if (!resident(layer, expert)) return nullptr;
    const size_t i = (size_t) (layer * n_expert_ + expert);
    if (i < held_.size() && held_[i] < 255) ++held_[i];
    if (i < score_.size()) score_[i] += 1.0f;
    touch_up(i);
    ++reads_;
    return host_at(i);
}

void ArenaExpertSource::add_mirror(const std::string& dir) {
    mirror_dirs_.push_back(dir);
    const char* r = std::getenv("STRATA_MIRROR_ROTATE");   // #82's A/B arm: 0 = every tie to the source, as before
    mirror_rotate_ = !(r != nullptr && *r && std::atoi(r) == 0);
}

const uint8_t* ArenaExpertSource::acquire(int64_t layer, int64_t expert, std::string& err) {
    std::lock_guard<std::mutex> lk(host_mu_);
    const size_t i = (size_t) (layer * n_expert_ + expert);
    if (base_ == nullptr || deferred_ || i >= exclusive_.size() || exclusive_[i]) { err = "acquire: not a host-tier expert"; return nullptr; }
    const uint8_t* p = materialize_locked(layer, expert, -1, err);
    if (p != nullptr) {
        if (i < held_.size() && held_[i] < 255) ++held_[i];
        if (i < score_.size()) score_[i] += 1.0f;   // a use, as blob() counts one
        touch_up(i);   // a load inside materialize_locked already touched it
    }
    return p;
}

const uint8_t* ArenaExpertSource::materialize_locked(int64_t layer, int64_t expert, int64_t avoid_layer,
                                                     std::string& err) {
    const auto& lay = strata::kernels::cpu::expert_layout();
    const size_t i = (size_t) (layer * n_expert_ + expert);
    if (base_ == nullptr || i >= exclusive_.size() || exclusive_[i]) { err = "materialize: not a host-tier expert"; return nullptr; }
    if (nvme_.empty() || !nvme_[i]) return host_at(i);
    const auto t0 = std::chrono::steady_clock::now();
    const uint64_t b = lay.blob_bytes(layer);
    while (cache_cap_ != 0 && cache_used_ + b > cache_cap_ && evict_one(avoid_layer)) {}
    uint64_t held = 0;
    if (!host_commit(i, held, err)) return nullptr;
    uint8_t* at = host_at(i);
    // the read lands before the tier says so: a failed read leaves the expert on NVMe and returns null
    if (!read_expert(layer, expert, at, err)) {
        std::string e;
        host_release(i, held, e);
        return nullptr;
    }
    nvme_[i] = 0;
    score_[i] += 1.0f;
    touch(i);
    cache_used_ += b;
    ++nvme_loads_;
    nvme_ms_ += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    return at;
}

bool ArenaExpertSource::materialize_batch(int64_t layer, const int32_t* experts, int n, std::string& err) {
    return materialize_begin(layer, experts, n, err) && materialize_end(err);
}

bool ArenaExpertSource::materialize_begin(int64_t layer, const int32_t* experts, int n, std::string& err) {
    // one batch at a time, from one thread (the verify window's dispatch): mat_lock_ is that thread's, so reading it
    // here without host_mu_ is safe; another thread that wants the host tier waits on host_mu_ instead
    if (mat_lock_.owns_lock()) { err = "materialize_begin: a batch is already in flight"; return false; }
    std::unique_lock<std::mutex> lk(host_mu_);
    const auto& lay = strata::kernels::cpu::expert_layout();
    if (nvme_.empty() || n <= 0) return true;
    const auto t0 = std::chrono::steady_clock::now();
    std::vector<int32_t> ls, es;
    std::vector<uint8_t*> dsts;
    const uint64_t b = lay.blob_bytes(layer);
    for (int i = 0; i < n; ++i) {
        const size_t idx = (size_t) (layer * n_expert_ + experts[i]);
        if (idx >= nvme_.size() || exclusive_[idx] || !nvme_[idx]) continue;
        const auto te = std::chrono::steady_clock::now();
        while (cache_cap_ != 0 && cache_used_ + b > cache_cap_ && evict_one(layer)) {}
        stages_.evict_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - te).count();
        uint64_t held = 0;
        if (!host_commit(idx, held, err)) {   // commit time is counted where pages are committed
            unwind_batch(layer, es);
            return false;
        }
        cache_used_ += b;   // accounted now so the next eviction sees it; published below once the bytes landed
        ls.push_back((int32_t) layer);
        es.push_back(experts[i]);
        dsts.push_back(host_at(idx));
    }
    if (es.empty()) return true;
    const auto ts = std::chrono::steady_clock::now();
    const bool submitted = submit_reads(ls.data(), es.data(), (int) es.size(), dsts.data(), /*into_slots=*/true, err);
    stages_.submit_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - ts).count();
    if (!submitted) {   // the caller fails loudly
        unwind_batch(layer, es);
        return false;
    }
    // #95: the reads are in flight; host_mu_ stays held (the bounce buffer, the ports and the tier are this batch's)
    // until materialize_end publishes them.  The batch's experts are still on NVMe, so blob() refuses them.
    mat_es_ = std::move(es);
    mat_layer_ = layer;
    mat_ms_ = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    mat_lock_ = std::move(lk);
    return true;
}

bool ArenaExpertSource::materialize_end(std::string& err) {
    if (!mat_lock_.owns_lock()) return true;
    const auto t1 = std::chrono::steady_clock::now();
    const bool ok = collect_reads(err);
    if (!ok) {   // the caller fails loudly
        unwind_batch(mat_layer_, mat_es_);
    } else {
        for (int32_t x : mat_es_) {
            const size_t idx = (size_t) (mat_layer_ * n_expert_ + x);
            nvme_[idx] = 0;
            score_[idx] += 1.0f;
            touch(idx);
        }
        nvme_loads_ += (int64_t) mat_es_.size();
    }
    // the exposed time only: the begin half plus the wait here (an overlapped pool run between them is not counted)
    nvme_ms_ += mat_ms_ + std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t1).count();
    mat_es_.clear();
    mat_lock_.unlock();
    return ok;
}

void ArenaExpertSource::trim(int64_t avoid_layer) {
    std::lock_guard<std::mutex> lk(host_mu_);
    while (cache_cap_ != 0 && cache_used_ > cache_cap_ && evict_one(avoid_layer)) {}
}

void ArenaExpertSource::decay_scores(float f) {
    for (float& v : score_) v *= f;
    for (float& v : lmin_) v *= f;   // the same multiply as each expert's score: the cached minima stay exact
}

void ArenaExpertSource::admit_home(int64_t layer, int64_t expert) {
    if (nvme_.empty()) return;
    std::lock_guard<std::mutex> lk(host_mu_);
    const auto& lay = strata::kernels::cpu::expert_layout();
    const size_t i = (size_t) (layer * n_expert_ + expert);
    if (i < nvme_.size() && nvme_[i]) { nvme_[i] = 0; cache_used_ += lay.blob_bytes(layer); }
    score_[i] += 1.0f;
    touch(i);
    while (cache_cap_ != 0 && cache_used_ > cache_cap_ && evict_one(-1)) {}
}

bool ArenaExpertSource::read_into(int64_t layer, int64_t expert, uint8_t* dst, std::string& err) {
    if (tail_reader_ && tail_reader_(layer, expert, dst)) return true;   // #34: the tail file
    // the prompt path calls this from several stager threads: each thread keeps its own reader
    thread_local std::ifstream f;
    thread_local std::string name;
    const auto& lay = strata::kernels::cpu::expert_layout();
    uint64_t off[3], len[3], at[3];
    const int nr = expert_ranges(lay, from_gguf_, layer, expert, off, len, at);
    for (int r = 0; r < nr; ++r) {
        // per range: a GGUF role may sit in another shard (v4); the handle is kept while the file is the same
        const std::string want = expert_file(gguf_, path_, from_gguf_, lay, layer, r);
        if (want != name || !f.is_open()) {
            f.close(); f.clear();
            f.open(want, std::ios::binary);
            if (!f) { err = "read_into: cannot open " + want; name.clear(); return false; }
            name = want;
        }
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
    for (auto& d : dfiles_) delete (strata::platform::DirectFile*) d.file;
    dfiles_.clear();
    if (dscratch_ != nullptr) strata::platform::DirectFile::free_aligned(dscratch_);
    dscratch_ = nullptr;
    dscratch_bytes_ = 0;
    slab_release();
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
    uint64_t held = 0;   // capacity mode: the copy comes home into a slot
    if (!host_commit(index, held, err)) return nullptr;
    released_host_bytes_ -= held < released_host_bytes_ ? held : released_host_bytes_;
    err.clear();
    return host_at(index);   // GPU-owned until publish_host_copy
}

void ArenaExpertSource::publish_host_copy(int64_t layer, int64_t expert) {
    std::lock_guard<std::mutex> lk(host_mu_);
    const size_t index = (size_t) (layer * n_expert_ + expert);
    if (index < exclusive_.size()) { exclusive_[index] = 0; touch(index); }
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
    uint64_t decommitted = 0;
    if (!host_release(index, decommitted, err)) return false;
    exclusive_[index] = 1; // publish ownership only after the Windows decommit succeeds
    touch(index);
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
        touch_up((size_t) idx);   // k x layers per token: rescan a layer only when its minimum moved
    }
    ++reads_;
    // Pointer arithmetic into resident memory.  No fault, no copy, no mapping - which is the entire point of
    // this class over `FileExpertSource`.
    return host_at((size_t) idx);
}

uint8_t* ArenaExpertSource::host_at(size_t idx) const {
    const auto& lay = strata::kernels::cpu::expert_layout();
    if (slab_) {
        if (slot_of_[idx] < 0) return nullptr;
        const SlabClass& k = slab_class(idx);
        return k.base + (size_t) slot_of_[idx] * k.stride;
    }
    return const_cast<uint8_t*>(base_) + lay.blob_offset((int64_t) idx / lay.n_expert, (int64_t) idx % lay.n_expert);
}

bool ArenaExpertSource::host_commit(size_t idx, uint64_t& held, std::string& err) {
    held = 0;
    const auto& lay = strata::kernels::cpu::expert_layout();
    const int64_t l = (int64_t) idx / lay.n_expert, x = (int64_t) idx % lay.n_expert;
    if (slab_) {
        if (slot_of_[idx] >= 0) return true;
        if (!take_slot(idx, err)) return false;
        held = lay.blob_bytes(l);
        return true;
    }
    const auto t0 = std::chrono::steady_clock::now();
    if (!((PinnedArena*) arena_)->commit_interior(lay.blob_offset(l, x), lay.blob_bytes(l), held, err)) return false;
    stages_.commit_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    return true;
}

bool ArenaExpertSource::host_release(size_t idx, uint64_t& given, std::string& err) {
    given = 0;
    const auto& lay = strata::kernels::cpu::expert_layout();
    const int64_t l = (int64_t) idx / lay.n_expert, x = (int64_t) idx % lay.n_expert;
    if (slab_) {
        if (slot_of_[idx] >= 0) { free_slot(idx); given = lay.blob_bytes(l); }
        return true;
    }
    return ((PinnedArena*) arena_)->decommit_interior(lay.blob_offset(l, x), lay.blob_bytes(l), given, err);
}

void ArenaExpertSource::unwind_batch(int64_t layer, const std::vector<int32_t>& es) {
    cache_used_ -= strata::kernels::cpu::expert_layout().blob_bytes(layer) * es.size();
    uint64_t given = 0;
    std::string e;
    for (int32_t x : es) host_release((size_t) (layer * n_expert_ + x), given, e);
}

bool ArenaExpertSource::take_slot(size_t idx, std::string& err) {
    // the byte cap evicts; a slot is always available (the region holds every expert)
    if (slot_of_[idx] >= 0) return true;
    SlabClass& k = slab_class(idx);
    if (!k.idle.empty()) {   // the common case: an evicted expert's slot, committed and already faulted in
        slot_of_[idx] = k.idle.back();
        k.idle.pop_back();
        idle_bytes_ -= k.stride;
        return true;
    }
    int32_t s;
    if (!k.cold.empty()) { s = k.cold.back(); k.cold.pop_back(); }
    else if (k.next < k.cap) s = k.next++;
    else { err = "capacity mode: the slab region is full"; return false; }
#if defined(_WIN32)
    const auto t0 = std::chrono::steady_clock::now();
    if (VirtualAlloc(k.base + (size_t) s * k.stride, k.stride, MEM_COMMIT, PAGE_READWRITE) == nullptr) {
        k.cold.push_back(s);
        err = "capacity mode: a slab slot could not be committed";
        return false;
    }
    stages_.commit_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
#endif
    ++slab_committed_;
    slot_of_[idx] = s;
    return true;
}

void ArenaExpertSource::free_slot(size_t idx) {
    if (slot_of_[idx] < 0) return;
    SlabClass& k = slab_class(idx);
    k.idle.push_back(slot_of_[idx]);
    idle_bytes_ += k.stride;
    slot_of_[idx] = -1;
    while (idle_bytes_ > slack_bytes_) {   // past the slack: decommit an idle slot of the size holding the most
        SlabClass* most = nullptr;
        for (SlabClass& c : classes_)
            if (!c.idle.empty() && (most == nullptr || c.idle.size() * c.stride > most->idle.size() * most->stride)) most = &c;
        if (most == nullptr) break;
        const int32_t s = most->idle.back();
        most->idle.pop_back();
#if defined(_WIN32)
        if (!VirtualFree(most->base + (size_t) s * most->stride, most->stride, MEM_DECOMMIT)) {
            most->idle.push_back(s);   // still committed: it stays idle (over the slack) rather than lost
            break;
        }
#endif
        most->cold.push_back(s);
        idle_bytes_ -= most->stride;
        --slab_committed_;
    }
}

void ArenaExpertSource::slab_release() {
#if defined(_WIN32)
    for (SlabClass& k : classes_) if (k.base != nullptr) VirtualFree(k.base, 0, MEM_RELEASE);
#endif
    classes_.clear();
    class_of_layer_.clear();
    slot_of_.clear();
    slab_ = false;
    idle_bytes_ = 0;
    slab_committed_ = 0;
}

}  // namespace strata::core
