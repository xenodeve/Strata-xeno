#pragma once

#include "strata/core/secondary_arena.hpp"
#include <array>

#include <chrono>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <string>
#include <thread>
#include <vector>

namespace strata::kernels { struct NativeExpertLayout; }

namespace strata::core {

// Opt-in timing totals. CUDA event intervals are stream elapsed time and can
// include gaps while the host enqueues work; they are not pure kernel timings.
struct SecondaryTiming {
    uint64_t launches = 0;
    double host_plan_ms = 0, host_switch_ms = 0, host_enqueue_ms = 0;
    double host_query_ms = 0, host_copyout_ms = 0;
    double h2d_ms = 0, clear_ms = 0, quantize_ms = 0, expert_ms = 0, d2h_ms = 0;
};

// Device-1 grouped native Q2_0 partials. launch() queues pinned H2D, compute,
// pinned D2H; finish() waits and copies only claimed router rows into host output.
class SecondaryRunner {
public:
    using BreachHandler = void (*)(const char* reason, void* context);
    SecondaryRunner() = default;
    ~SecondaryRunner();
    SecondaryRunner(const SecondaryRunner&) = delete;
    SecondaryRunner& operator=(const SecondaryRunner&) = delete;

    bool init(int max_tokens, int max_entries, int n_embd, int n_ff, std::string& err,
              uint64_t free_floor_bytes = 2560ull << 20, bool profile_timing = false);
    bool launch(const kernels::NativeExpertLayout& layout, const SecondaryArena& weights,
                const float* x, const int32_t* selected_slots, int n_tokens, int k,
                std::string& err);
    bool finish(float* output, std::string& err);
    bool start_monitor(int interval_ms, std::string& err,
                       SecondaryArena::FreeReader reader = nullptr,
                       BreachHandler on_breach = nullptr, void* context = nullptr,
                       uint64_t free_floor_bytes = 2560ull << 20);
    void stop_monitor();
    uint64_t served_entries() const { return served_entries_; }
    uint64_t served_groups() const { return served_groups_; }
    /// Host time spent inside launch() enqueueing, and blocked in finish() waiting for the 4070 SUPER's partials.
    /// A non-zero wait means the secondary tier, not the CPU pool, was the slower side of that layer.
    double ms_launch() const { return ms_launch_; }
    double ms_wait() const { return ms_wait_; }
    const SecondaryTiming& timing() const { return timing_; }
    uint64_t free_checks() const { return free_checks_.load(); }
    uint64_t min_free_bytes() const { return min_free_bytes_.load(); }

private:
    void close();
    void record_free(uint64_t bytes);
    SecondaryArena workspace_;
    int max_tokens_ = 0, max_entries_ = 0, n_embd_ = 0, n_ff_ = 0;
    void* stream_ = nullptr;
    void* done_ = nullptr;
    std::array<void*, 6> timing_events_{};
    bool profile_timing_ = false;
    SecondaryTiming timing_{};
    float* host_x_ = nullptr;
    float* host_out_ = nullptr;
    uint8_t* host_meta_ = nullptr;
    size_t meta_start_off_ = 0, meta_count_off_ = 0, meta_dst_off_ = 0, meta_tok_off_ = 0;
    size_t meta_bytes_ = 0;
    float* device_x_ = nullptr;
    uint8_t* device_xq_ = nullptr;
    float* device_scales_ = nullptr;
    float* device_out_ = nullptr;
    void* device_scratch_ = nullptr;
    unsigned long long* device_ptr_ = nullptr;
    int32_t *device_start_ = nullptr, *device_count_ = nullptr;
    int32_t *device_dst_ = nullptr, *device_tok_ = nullptr;
    std::vector<unsigned long long> ptr_;
    std::vector<int32_t> start_, dst_, tok_, selected_rows_, group_slots_;
    int32_t host_count_ = 0; ///< copied into portable pinned metadata before enqueue
    bool pending_ = false;
    bool failed_ = false; ///< async enqueue failure poisons the runner until teardown
    uint64_t served_entries_ = 0, served_groups_ = 0;
    double ms_launch_ = 0, ms_wait_ = 0;
    std::atomic<uint64_t> free_checks_{0};
    std::atomic<uint64_t> min_free_bytes_{~uint64_t{0}};
    uint64_t free_floor_bytes_ = 2560ull << 20;
    std::atomic<bool> monitor_stop_{false}, monitor_running_{false};
    std::thread monitor_;
    std::chrono::steady_clock::time_point last_free_check_{};
};

} // namespace strata::core
