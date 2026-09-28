#include <chrono>
#include "strata/core/secondary_runner.hpp"
#include "strata/core/secondary_budget.hpp"
#include "strata/core/secondary_vram.hpp"
#include "strata/kernels/iq_kernels.hpp"

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>

namespace strata::core {
namespace {

struct RestoreDevice {
    int previous;
    double* restore_ms = nullptr;
    ~RestoreDevice() {
        if (restore_ms == nullptr) { cudaSetDevice(previous); return; }
        const auto start = std::chrono::steady_clock::now();
        cudaSetDevice(previous);
        *restore_ms += std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - start).count();
    }
};

uint64_t aligned(uint64_t bytes) { return (bytes + 255) & ~uint64_t{255}; }

bool cuda_ok(cudaError_t result, const char* operation, std::string& err) {
    if (result == cudaSuccess) return true;
    err = std::string(operation) + ": " + cudaGetErrorString(result);
    return false;
}

bool check_display_free(uint64_t free_floor_bytes, std::string& err, uint64_t* lower_out = nullptr) {
    size_t cuda_free = 0, total = 0;
    if (!cuda_ok(cudaMemGetInfo(&cuda_free, &total), "secondary runner cudaMemGetInfo", err)) return false;
    uint64_t nvml_free = 0;
    if (!secondary_nvml_free_bytes(1, nvml_free, err)) return false;
    const uint64_t lower = secondary_effective_free((uint64_t) cuda_free, nvml_free);
    if (lower_out) *lower_out = lower;
    if (lower < free_floor_bytes) {
        err = "secondary runner: display VRAM below configured free floor";
        return false;
    }
    return true;
}

} // namespace

SecondaryRunner::~SecondaryRunner() { close(); }

void SecondaryRunner::record_free(uint64_t bytes) {
    uint64_t old = min_free_bytes_.load();
    while (bytes < old && !min_free_bytes_.compare_exchange_weak(old, bytes)) {}
}

bool SecondaryRunner::init(int max_tokens, int max_entries, int n_embd, int n_ff,
                           std::string& err, uint64_t free_floor_bytes, bool profile_timing) {
    if (max_tokens <= 0 || max_tokens > 16 || max_entries <= 0 || max_entries > 128 ||
        n_embd <= 0 || n_embd % 32 != 0 || n_ff <= 0 || n_ff % 32 != 0) {
        err = "secondary runner: invalid verify-window geometry";
        return false;
    }
    if (workspace_.slots() != 0 || stream_ != nullptr) {
        err = "secondary runner is already initialized";
        return false;
    }
    free_floor_bytes_ = free_floor_bytes;
    profile_timing_ = profile_timing;
    timing_ = {};
    max_tokens_ = max_tokens;
    max_entries_ = max_entries;
    n_embd_ = n_embd;
    n_ff_ = n_ff;
    uint64_t cursor = 0;
    auto carve = [&](uint64_t bytes) { const uint64_t at = cursor; cursor += aligned(bytes); return at; };
    const uint64_t x_at = carve((uint64_t) max_tokens * n_embd * sizeof(float));
    const uint64_t xq_at = carve((uint64_t) max_tokens * (n_embd / 32) * 36);
    const uint64_t scales_at = carve((uint64_t) max_tokens * (n_embd / 32) * sizeof(float));
    const uint64_t out_at = carve((uint64_t) max_entries * n_embd * sizeof(float));
    const uint64_t scratch_at = carve(kernels::native_expert_scratch_bytes(max_entries, n_ff));
    const uint64_t ptr_at = carve((uint64_t) max_entries * sizeof(unsigned long long));
    const uint64_t start_at = carve((uint64_t) (max_entries + 1) * sizeof(int32_t));
    const uint64_t count_at = carve(sizeof(int32_t));
    const uint64_t dst_at = carve((uint64_t) max_entries * sizeof(int32_t));
    const uint64_t tok_at = carve((uint64_t) max_entries * sizeof(int32_t));
    meta_start_off_ = (size_t) (start_at - ptr_at);
    meta_count_off_ = (size_t) (count_at - ptr_at);
    meta_dst_off_ = (size_t) (dst_at - ptr_at);
    meta_tok_off_ = (size_t) (tok_at - ptr_at);
    meta_bytes_ = (size_t) (cursor - ptr_at);
    if (!workspace_.open(1, {cursor}, cursor, err)) return false;
    uint8_t* base = workspace_.slot_ptr(0);
    device_x_ = (float*) (base + x_at);
    device_xq_ = base + xq_at;
    device_scales_ = (float*) (base + scales_at);
    device_out_ = (float*) (base + out_at);
    device_scratch_ = base + scratch_at;
    device_ptr_ = (unsigned long long*) (base + ptr_at);
    device_start_ = (int32_t*) (base + start_at);
    device_count_ = (int32_t*) (base + count_at);
    device_dst_ = (int32_t*) (base + dst_at);
    device_tok_ = (int32_t*) (base + tok_at);

    int previous = -1;
    if (!cuda_ok(cudaGetDevice(&previous), "runner current device", err) ||
        !cuda_ok(cudaSetDevice(1), "runner select device", err)) return false;
    const RestoreDevice restore{previous};
    cudaStream_t stream = nullptr;
    cudaEvent_t done = nullptr;
    if (!cuda_ok(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "runner stream", err) ||
        !cuda_ok(cudaEventCreateWithFlags(&done, cudaEventDisableTiming), "runner event", err) ||
        !cuda_ok(cudaHostAlloc((void**) &host_x_, (size_t) max_tokens * n_embd * sizeof(float),
                               cudaHostAllocPortable), "runner pinned input", err) ||
        !cuda_ok(cudaHostAlloc((void**) &host_out_, (size_t) max_entries * n_embd * sizeof(float),
                               cudaHostAllocPortable), "runner pinned output", err) ||
        !cuda_ok(cudaHostAlloc((void**) &host_meta_, meta_bytes_, cudaHostAllocPortable),
                 "runner pinned metadata", err)) {
        stream_ = stream;
        done_ = done;
        return false;
    }
    stream_ = stream;
    done_ = done;
    std::memset(host_meta_, 0, meta_bytes_);
    if (profile_timing_) {
        for (void*& raw : timing_events_) {
            cudaEvent_t event = nullptr;
            if (!cuda_ok(cudaEventCreate(&event), "runner timing event", err)) return false;
            raw = event;
        }
    }
    ptr_.reserve((size_t) max_entries);
    start_.reserve((size_t) max_entries + 1);
    dst_.reserve((size_t) max_entries);
    tok_.reserve((size_t) max_entries);
    selected_rows_.reserve((size_t) max_entries);
    group_slots_.reserve((size_t) max_entries);
    uint64_t lower = 0;
    if (!check_display_free(free_floor_bytes_, err, &lower)) return false;
    record_free(lower);
    ++free_checks_;
    last_free_check_ = std::chrono::steady_clock::now();
    err.clear();
    return true;
}

bool SecondaryRunner::launch(const kernels::NativeExpertLayout& layout, const SecondaryArena& weights,
                             const float* x, const int32_t* selected_slots, int n_tokens, int k,
                             std::string& err) {
    if (stream_ == nullptr || pending_ || failed_ || x == nullptr || selected_slots == nullptr ||
        n_tokens <= 0 || n_tokens > max_tokens_ || k <= 0 || k > max_entries_ / n_tokens ||
        layout.n_embd != n_embd_ || layout.n_ff != n_ff_ || layout.gu_type != 42 || layout.d_type != 42) {
        err = "secondary runner: invalid launch geometry or pending work";
        return false;
    }
    const auto launch_t0 = std::chrono::steady_clock::now();
    ptr_.clear(); start_.clear(); dst_.clear(); tok_.clear(); selected_rows_.clear(); group_slots_.clear();
    const int entries = n_tokens * k;
    for (int i = 0; i < entries; ++i) {
        const int32_t slot = selected_slots[i];
        if (slot < 0) continue;
        if (weights.slot_ptr((uint64_t) slot) == nullptr) {
            err = "secondary runner: selected slot is not resident";
            return false;
        }
        bool seen = false;
        for (const int32_t prior : group_slots_) if (prior == slot) { seen = true; break; }
        if (!seen) group_slots_.push_back(slot);
    }
    if (group_slots_.empty()) { err.clear(); return true; }
    for (const int32_t slot : group_slots_) {
        ptr_.push_back((unsigned long long) weights.slot_ptr((uint64_t) slot));
        start_.push_back((int32_t) dst_.size());
        for (int i = 0; i < entries; ++i) if (selected_slots[i] == slot) {
            dst_.push_back(i);
            tok_.push_back(i / k);
            selected_rows_.push_back(i);
        }
    }
    start_.push_back((int32_t) dst_.size());
    std::memcpy(host_x_, x, (size_t) n_tokens * n_embd_ * sizeof(float));
    host_count_ = (int32_t) group_slots_.size();
    std::memcpy(host_meta_, ptr_.data(), ptr_.size() * sizeof(ptr_[0]));
    std::memcpy(host_meta_ + meta_start_off_, start_.data(), start_.size() * sizeof(start_[0]));
    std::memcpy(host_meta_ + meta_count_off_, &host_count_, sizeof host_count_);
    std::memcpy(host_meta_ + meta_dst_off_, dst_.data(), dst_.size() * sizeof(dst_[0]));
    std::memcpy(host_meta_ + meta_tok_off_, tok_.data(), tok_.size() * sizeof(tok_[0]));
    const auto switch_t0 = std::chrono::steady_clock::now();
    if (profile_timing_)
        timing_.host_plan_ms += std::chrono::duration<double, std::milli>(switch_t0 - launch_t0).count();
    int previous = -1;
    if (!cuda_ok(cudaGetDevice(&previous), "runner current device", err) ||
        !cuda_ok(cudaSetDevice(1), "runner select device", err)) return false;
    const RestoreDevice restore{previous, profile_timing_ ? &timing_.host_switch_ms : nullptr};
    if (profile_timing_)
        timing_.host_switch_ms += std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - switch_t0).count();
    const auto now = std::chrono::steady_clock::now();
    if (!monitor_running_.load() && now - last_free_check_ >= std::chrono::milliseconds(250)) {
        uint64_t lower = 0;
        if (!check_display_free(free_floor_bytes_, err, &lower)) return false;
        record_free(lower);
        ++free_checks_;
        last_free_check_ = now;
    }
    cudaStream_t stream = (cudaStream_t) stream_;
    auto fail_enqueued = [&]() {
        failed_ = true;
        const cudaError_t drained = cudaStreamSynchronize(stream);
        if (drained != cudaSuccess)
            err += std::string("; stream drain: ") + cudaGetErrorString(drained);
        return false;
    };
    auto mark = [&](size_t index) {
        return !profile_timing_ || cuda_ok(cudaEventRecord((cudaEvent_t) timing_events_[index], stream),
                                           "runner timing marker", err);
    };
    const auto enqueue_t0 = std::chrono::steady_clock::now();
    if (!mark(0) ||
        !cuda_ok(cudaMemcpyAsync(device_ptr_, host_meta_, meta_bytes_, cudaMemcpyHostToDevice, stream),
                 "runner packed metadata H2D", err) ||
        !cuda_ok(cudaMemcpyAsync(device_x_, host_x_, (size_t) n_tokens * n_embd_ * sizeof(float),
                                 cudaMemcpyHostToDevice, stream), "runner activation H2D", err) ||
        !mark(1) ||
        !cuda_ok(cudaMemsetAsync(device_out_, 0, (size_t) entries * n_embd_ * sizeof(float), stream),
                 "runner clear partials", err) || !mark(2)) return fail_enqueued();
    kernels::quantize_q8_1_rows_scaled(device_x_, n_tokens, n_embd_, device_xq_, device_scales_, stream);
    if (!mark(3)) return fail_enqueued();
    kernels::native_expert_grouped(layout, device_ptr_, device_start_, device_count_, device_dst_, device_tok_,
                                   host_count_, (int) dst_.size(), device_xq_, device_scratch_, device_out_, stream,
                                   device_scales_);
    if (!mark(4)) return fail_enqueued();
    if (!cuda_ok(cudaMemcpyAsync(host_out_, device_out_, (size_t) entries * n_embd_ * sizeof(float),
                                 cudaMemcpyDeviceToHost, stream), "runner partial D2H", err) ||
        !mark(5) ||
        !cuda_ok(cudaEventRecord((cudaEvent_t) done_, stream), "runner completion event", err))
        return fail_enqueued();
    pending_ = true;
    if (profile_timing_)
        timing_.host_enqueue_ms += std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - enqueue_t0).count();
    ms_launch_ += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - launch_t0).count();
    err.clear();
    return true;
}

bool SecondaryRunner::finish(float* output, std::string& err) {
    if (!pending_) { err.clear(); return true; }
    if (output == nullptr) { err = "secondary runner: output is null"; return false; }
    const auto switch_t0 = std::chrono::steady_clock::now();
    int previous = -1;
    if (!cuda_ok(cudaGetDevice(&previous), "runner current device", err) ||
        !cuda_ok(cudaSetDevice(1), "runner select device", err)) return false;
    const RestoreDevice restore{previous, profile_timing_ ? &timing_.host_switch_ms : nullptr};
    if (profile_timing_)
        timing_.host_switch_ms += std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - switch_t0).count();
    const auto wait_t0 = std::chrono::steady_clock::now();
    if (!cuda_ok(cudaEventSynchronize((cudaEvent_t) done_), "runner wait partials", err)) {
        failed_ = true;
        return false;
    }
    ms_wait_ += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - wait_t0).count();
    if (profile_timing_) {
        const auto query_t0 = std::chrono::steady_clock::now();
        double* stages[] = {&timing_.h2d_ms, &timing_.clear_ms, &timing_.quantize_ms,
                            &timing_.expert_ms, &timing_.d2h_ms};
        for (size_t i = 0; i < 5; ++i) {
            float elapsed = 0.f;
            if (!cuda_ok(cudaEventElapsedTime(&elapsed, (cudaEvent_t) timing_events_[i],
                                              (cudaEvent_t) timing_events_[i + 1]),
                         "runner stage elapsed", err)) {
                failed_ = true;
                return false;
            }
            *stages[i] += elapsed;
        }
        timing_.host_query_ms += std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - query_t0).count();
        ++timing_.launches;
    }
    const auto copy_t0 = std::chrono::steady_clock::now();
    for (const int32_t row : selected_rows_)
        std::memcpy(output + (size_t) row * n_embd_, host_out_ + (size_t) row * n_embd_,
                    (size_t) n_embd_ * sizeof(float));
    if (profile_timing_)
        timing_.host_copyout_ms += std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - copy_t0).count();
    served_entries_ += selected_rows_.size();
    served_groups_ += group_slots_.size();
    pending_ = false;
    err.clear();
    return true;
}

bool SecondaryRunner::start_monitor(int interval_ms, std::string& err,
                                    SecondaryArena::FreeReader reader,
                                    BreachHandler on_breach, void* context,
                                    uint64_t free_floor_bytes) {
    if (stream_ == nullptr || monitor_.joinable() || interval_ms < 10 || interval_ms > 1000) {
        err = "secondary monitor needs an initialized runner and 10..1000 ms interval";
        return false;
    }
    int previous = -1;
    if (!cuda_ok(cudaGetDevice(&previous), "monitor current device", err) ||
        !cuda_ok(cudaSetDevice(1), "monitor select device", err)) return false;
    uint64_t lower = 0;
    free_floor_bytes_ = free_floor_bytes;
    const bool safe = check_display_free(free_floor_bytes_, err, &lower);
    const cudaError_t restored = cudaSetDevice(previous);
    if (!safe || !cuda_ok(restored, "monitor restore device", err)) return false;
    record_free(lower);
    ++free_checks_;
    monitor_stop_.store(false);
    monitor_running_.store(true);
    try {
        monitor_ = std::thread([this, interval_ms, reader, on_breach, context] {
            const cudaError_t selected = cudaSetDevice(1);
            if (selected != cudaSuccess) {
                std::fprintf(stderr, "secondary monitor cannot select display GPU: %s\n",
                             cudaGetErrorString(selected));
                std::fflush(stderr);
                std::_Exit(3);
            }
            while (!monitor_stop_.load()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(interval_ms));
                if (monitor_stop_.load()) break;
                std::string problem;
                uint64_t lower = 0;
                bool safe = false;
                if (reader != nullptr) {
                    safe = reader(1, lower, problem, context) && lower >= free_floor_bytes_;
                    if (!safe && problem.empty())
                        problem = "injected display free " + std::to_string(lower) + " B below configured floor";
                } else {
                    safe = check_display_free(free_floor_bytes_, problem, &lower);
                }
                record_free(lower);
                if (!safe) {
                    if (on_breach != nullptr) {
                        on_breach(problem.c_str(), context);
                        monitor_stop_.store(true);
                        break;
                    }
                    std::fprintf(stderr, "secondary display VRAM reserve failed: %s; terminating to release tier\n",
                                 problem.c_str());
                    std::fflush(stderr);
                    std::_Exit(3);
                }
                ++free_checks_;
            }
        });
    } catch (const std::exception& e) {
        monitor_running_.store(false);
        err = std::string("secondary monitor thread: ") + e.what();
        return false;
    }
    err.clear();
    return true;
}

void SecondaryRunner::stop_monitor() {
    monitor_stop_.store(true);
    if (monitor_.joinable()) monitor_.join();
    monitor_running_.store(false);
}

void SecondaryRunner::close() {
    stop_monitor();
    if (workspace_.slots() == 0 && stream_ == nullptr && done_ == nullptr &&
        host_x_ == nullptr && host_out_ == nullptr && host_meta_ == nullptr) return;
    int previous = -1;
    const cudaError_t current = cudaGetDevice(&previous);
    const cudaError_t selected = current == cudaSuccess ? cudaSetDevice(1) : current;
    if (selected != cudaSuccess) {
        std::fprintf(stderr, "secondary runner cleanup cannot select display device: %s\n",
                     cudaGetErrorString(selected));
        std::abort();
    }
    if (stream_ != nullptr && cudaStreamSynchronize((cudaStream_t) stream_) != cudaSuccess) {
        std::fprintf(stderr, "secondary runner cleanup found a failed GPU stream\n");
        std::abort();
    }
    for (void*& raw : timing_events_) {
        if (raw != nullptr && cudaEventDestroy((cudaEvent_t) raw) != cudaSuccess) std::abort();
        raw = nullptr;
    }
    if (done_ != nullptr && cudaEventDestroy((cudaEvent_t) done_) != cudaSuccess) std::abort();
    if (stream_ != nullptr && cudaStreamDestroy((cudaStream_t) stream_) != cudaSuccess) std::abort();
    if (host_out_ != nullptr && cudaFreeHost(host_out_) != cudaSuccess) std::abort();
    if (host_meta_ != nullptr && cudaFreeHost(host_meta_) != cudaSuccess) std::abort();
    if (host_x_ != nullptr && cudaFreeHost(host_x_) != cudaSuccess) std::abort();
    std::string err;
    if (!workspace_.close(&err)) {
        std::fprintf(stderr, "secondary runner: %s; terminating to release display VRAM\n", err.c_str());
        std::fflush(stderr);
        std::abort();
    }
    cudaSetDevice(previous);
}

} // namespace strata::core
