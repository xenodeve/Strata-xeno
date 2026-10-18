#pragma once

#include "strata/core/secondary_arena.hpp"
#include <array>

#include <chrono>
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <cstddef>
#include <cstdint>
#include <string>
#include <thread>
#include <vector>

#include "strata/kernels/iq_kernels.hpp"

namespace strata::core {

// Opt-in timing totals. CUDA event intervals are stream elapsed time and can
// include gaps while the host enqueues work; they are not pure kernel timings.
struct SecondaryTiming {
    uint64_t launches = 0;
    uint64_t d2h_full_bytes = 0, d2h_requested_bytes = 0;
    double host_plan_ms = 0, host_switch_ms = 0, host_enqueue_ms = 0;
    double host_query_ms = 0, host_copyout_ms = 0;
    double h2d_ms = 0, clear_ms = 0, quantize_ms = 0, expert_ms = 0, d2h_ms = 0;
};

// Device-1 grouped native partials (Q2_0, and the i-quant formats since #11). launch() queues pinned H2D, compute,
// pinned D2H; finish() waits and copies only claimed router rows into host output.
class SecondaryRunner {
public:
    using BreachHandler = void (*)(const char* reason, void* context);
    SecondaryRunner() = default;
    ~SecondaryRunner();
    SecondaryRunner(const SecondaryRunner&) = delete;
    SecondaryRunner& operator=(const SecondaryRunner&) = delete;

    /// #228: `stage_slots` (0..kMaxStageSlots) device-1 slots of `stage_blob_bytes` each, carved in the workspace, for
    /// experts that are not resident on the 4070: launch() copies them in from pinned host memory first.
    static constexpr int kMaxStageSlots = 16;
    bool init(int max_tokens, int max_entries, int n_embd, int n_ff, std::string& err,
              uint64_t free_floor_bytes = 2560ull << 20, bool profile_timing = false,
              int stage_slots = 0, uint64_t stage_blob_bytes = 0);
    /// `selected_slots[i]` is a resident slot (>= 0), -1 (not on the 4070), or #228 a staged blob -(s + 2): staging slot
    /// s, filled from `staged[s]` (`staged_bytes` each, pinned host memory for an asynchronous copy) before the kernels.
    bool launch(const kernels::NativeExpertLayout& layout, const SecondaryArena& weights,
                const float* x, const int32_t* selected_slots, int n_tokens, int k,
                std::string& err, const uint8_t* const* staged = nullptr, int n_staged = 0,
                size_t staged_bytes = 0);
    /// #228: staging slots launch() and launch_layer() take (0 when none were carved, or once the async launcher is
    /// on: refused there).  In the window graph (v2) the layer's segment fetches them itself from mapped host memory.
    int stage_slots() const { return async_ ? 0 : stage_slots_; }

    /// #227 STRATA_SECONDARY_WINDOW_GRAPH=1: one CUDA graph per verify window instead of one launch per layer.  The
    /// graph (one per token count, captured here for 1..max_tokens) holds a segment per layer: wait until the layer's
    /// go flag (mapped host memory) is raised, copy its plan and activations from that layer's pinned buffers, run the
    /// experts, copy the claimed rows to mapped host memory and raise its done flag.  launch_layer() then makes no CUDA
    /// call except at layer 0 (the graph launch); finish() spins on the done flag.  `layouts` has one entry per layer
    /// (its formats and folding, as the dispatch builds them).  Call once, after init(), before any launch.
    bool enable_window(const std::vector<kernels::NativeExpertLayout>& layouts, std::string& err);
    /// The window graph needs every layer dispatched once per window, in order.  The reason it cannot run with these
    /// options (a split verify window dispatches each layer twice; a layer split, pipelined windows and batch groups
    /// interleave two windows' layers), or null when it can.
    static const char* window_refusal(bool spec_split, bool layer_split, int pipeline_windows, int batch_groups,
                                      bool batch_groups_auto);
    bool window_on() const { return window_; }
    /// The window mode's per-layer publish: every layer of every window, claimed or not (`selected_slots` all -1 when
    /// not; a layer that is not published is filled with an empty plan at the next layer 0 or end_window).
    /// #228 v2: `staged_dev[s]` (s < n_staged) is the device pointer of a page-locked, mapped host blob of
    /// `staged_bytes` (16-byte aligned) for slot code -(s + 2); the segment fetches it into staging slot s first.
    bool launch_layer(int layer, const kernels::NativeExpertLayout& layout, const SecondaryArena& weights,
                      const float* x, const int32_t* selected_slots, int n_tokens, int k, std::string& err,
                      const uint8_t* const* staged_dev = nullptr, int n_staged = 0, size_t staged_bytes = 0);
    /// Publish an empty plan for every layer of the open window not published yet, so its graph runs to the end and
    /// no segment keeps spinning on the display card (between requests, before teardown).
    void end_window();
    /// #267: raise every go flag past any count, so no segment of a window graph spins after the engine ends.
    void release_window();
    /// #245: open the next window for `n_tokens` before its layer 0 is dispatched: close the last window and hand the
    /// graph launch to a helper thread on device 1, so the ~0.6 ms cudaGraphLaunch overlaps the host's wait for layer
    /// 0's router instead of sitting between layer 0's plan and its CPU pool.  launch_layer(0) then publishes into the
    /// open window without launching.  On by default with the window graph; STRATA_SECONDARY_WINDOW_EARLY=0 keeps the
    /// launch at layer 0 (for A/B).  A failed launch is reported by the next launch_layer or finish.
    void begin_window(int n_tokens);
    uint64_t windows() const { return win_seq_; }
    /// #228 v3: the window segment's cost on the 4070, fitted from its own %globaltimer stamps over the claimed layers
    /// (us = base + group * resident groups + copy * staged blobs, exponentially weighted); false until enough
    /// layers were seen (or STRATA_SECONDARY_STAGE_LIVE=0).
    bool window_cost(double& base, double& group, double& copy) const;
    uint64_t staged_experts() const { return staged_experts_; }
    bool finish(float* output, std::string& err);
    /// Opt-in: from now on launch() hands its enqueue to a helper thread that stays on device 1 (pinned to
    /// `core` when >= 0), so the host thread goes straight on to the CPU pool; finish() first waits for the
    /// helper. Arithmetic and outputs are unchanged. Call once, after init().
    bool start_async_launch(int core, std::string& err);
    /// Issue #25: enqueue each launch as one CUDA graph per token count (captured on first use, sized for every
    /// entry; the kernels read the real group count from the metadata) instead of ~12 API calls. Outputs are
    /// unchanged. Ignored while profile timing is on (its event markers are per call).
    bool set_graph(bool on, std::string& err);
    /// Helper-thread time spent enqueueing (async mode), and host time finish() waited for that enqueue.
    double ms_async_enqueue() const { return ms_async_enqueue_; }
    double ms_async_wait() const { return ms_async_wait_; }
    bool start_monitor(int interval_ms, std::string& err,
                       SecondaryArena::FreeReader reader = nullptr,
                       BreachHandler on_breach = nullptr, void* context = nullptr,
                       uint64_t free_floor_bytes = 2560ull << 20);
    void stop_monitor();
    /// xeno (#208, developer 2026-10-13): the display card's rule is "other processes + free = the reserve; the server
    /// may use the rest". When others grow into free memory the server has not broken that rule, so with warn the
    /// monitor logs the breach (at most once a minute) and the engine keeps serving instead of ending (#203).
    /// Set before start_monitor.
    void set_breach_warn(bool on) { breach_warn_ = on; }
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
    bool launch_now(const kernels::NativeExpertLayout& layout, const SecondaryArena& weights,
                    const float* x, const int32_t* selected_slots, int n_tokens, int k, std::string& err,
                    const uint8_t* const* staged = nullptr, int n_staged = 0, size_t staged_bytes = 0);
    void launcher_loop(int core);
    void stop_launcher();
    void close();
    void record_free(uint64_t bytes);
    bool capture_window(int n_tokens, std::string& err);
    bool publish_empty(int layer, std::string& err);
    bool wait_consumed(int layer, std::string& err);
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
    size_t meta_stage_n_off_ = 0, meta_stage_src_off_ = 0, meta_stage_bytes_off_ = 0;   ///< #228 v2 (window plans)
    // #262: the plan's groups split for the fetch overlap - the resident ones' count (they come first), and the staged
    // ones' own pointer / start table and count
    size_t meta_count_a_off_ = 0, meta_ptr_b_off_ = 0, meta_start_b_off_ = 0, meta_count_b_off_ = 0;
    int32_t* device_count_a_ = nullptr;
    unsigned long long* device_ptr_b_ = nullptr;
    int32_t *device_start_b_ = nullptr, *device_count_b_ = nullptr;
    bool fetch_overlap_ = false;     ///< #262 STRATA_SECONDARY_FETCH_OVERLAP (default on with staging slots)
    void* fetch_stream_ = nullptr;   ///< the window graph's side branch for the staged blobs' fetch (capture only)
    void* fetch_fork_ = nullptr;
    void* fetch_join_ = nullptr;
    uint8_t* dev_wmeta_ = nullptr;   ///< #262: the plans and activations, mapped (the fetch overlap copies them by kernel)
    float* dev_wx_ = nullptr;
    size_t meta_bytes_ = 0;
    float* device_x_ = nullptr;
    uint8_t* device_stage_ = nullptr;   ///< #228: stage_slots_ x stage_stride_ bytes
    int32_t* device_stage_n_ = nullptr;                ///< #228 v2: the layer plan's staged count, sources and bytes
    unsigned long long* device_stage_src_ = nullptr;
    long long* device_stage_bytes_ = nullptr;
    int stage_slots_ = 0;
    uint64_t stage_stride_ = 0;
    uint64_t staged_experts_ = 0;
    // #227 window graph
    bool window_ = false;
    int win_layers_ = 0, win_next_ = 0, win_tokens_ = 0, win_layer_ = -1;
    uint32_t win_seq_ = 0;
    std::atomic<bool> win_released_{false};   ///< release_window ran: nothing is launched or published after it
    // #241 / #228 v3: per layer the segment's %globaltimer start and end (mapped), and the fit of its cost
    volatile unsigned long long* host_wstamp_ = nullptr;
    unsigned long long* dev_wstamp_ = nullptr;
    int win_groups_ = 0, win_staged_ = 0;     ///< the published layer's resident groups and staged blobs
    double fit_[3][4] = {};                   ///< decayed normal equations of [1, groups, staged] -> us (+ the rhs)
    uint64_t fit_n_ = 0;
    bool fit_live_ = true;
    double fit_cost_[3] = {0, 0, 0};          ///< solved every 256 layers
    bool fit_ok_ = false;
    long long tl_off_ns_ = 0;                 ///< steady_clock ns - %globaltimer ns (the smallest seen): the lane's clock
    bool tl_off_set_ = false;
    void window_sample(int layer, std::chrono::steady_clock::time_point seen);
    // #245 begin_window: a helper thread on device 1 launches each window's graph, in the order the windows open
    bool win_early_ = false;                  ///< STRATA_SECONDARY_WINDOW_EARLY (default on with the window graph)
    uint32_t win_opened_ = 0;                 ///< win_seq_ of the window begin_window opened and layer 0 has not taken
    std::thread win_launcher_;
    std::mutex win_mu_;
    std::condition_variable win_cv_;
    int win_req_tokens_ = 0;                  ///< (win_mu_) the token count of a launch not yet taken, 0 when none
    uint32_t win_req_seq_ = 0;                ///< (win_mu_) its window
    bool win_stop_ = false;                   ///< (win_mu_)
    uint32_t win_requested_ = 0;              ///< host thread: the last window handed to the helper
    std::atomic<uint32_t> win_launched_{0};   ///< the last window the helper launched (or failed to)
    std::atomic<bool> win_launch_failed_{false};
    std::string win_launch_err_;              ///< written before win_launch_failed_ is raised
    void win_launcher_loop();
    void stop_win_launcher();
    bool wait_win_launched(std::string& err);
    std::vector<kernels::NativeExpertLayout> win_layout_;
    uint8_t* host_wmeta_ = nullptr;      ///< win_layers_ x meta_bytes_, pinned
    float* host_wx_ = nullptr;           ///< win_layers_ x max_tokens_ x n_embd_, pinned
    float* host_wout_ = nullptr;         ///< max_entries_ x n_embd_, mapped
    float* dev_wout_ = nullptr;
    volatile uint32_t* host_go_ = nullptr;     ///< win_layers_, mapped
    volatile uint32_t* host_done_ = nullptr;   ///< win_layers_, mapped
    uint32_t *dev_go_ = nullptr, *dev_done_ = nullptr;
    uint32_t* dev_count_ = nullptr;      ///< win_layers_, device: how often each segment has run
    std::vector<void*> win_graph_;       ///< per token count
    // APR (prism.hadamard): this device's copies of the gate/up (n_embd) and down (n_ff) sign vectors, nullptr for
    // identity signs or a pack without folded experts (hadamard_device_signs owns them)
    const float* device_sx_ = nullptr;
    const float* device_sh_ = nullptr;
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
    bool breach_warn_ = false;   ///< --secondary-breach warn (set_breach_warn)
    std::atomic<bool> monitor_stop_{false}, monitor_running_{false};
    std::thread monitor_;
    // async launch: one request in flight at a time (launch -> finish per layer)
    std::thread launcher_;
    std::mutex launch_mu_;
    std::condition_variable launch_cv_;
    bool launch_req_ = false, launch_stop_ = false;
    std::atomic<bool> launch_done_{true};
    bool async_ = false, async_ok_ = true;
    bool graph_ = false;
    std::vector<void*> graph_exec_;   // per (token count, claimed rows): the D2H copies only the claimed rows
    std::vector<int> graph_k_;
    std::vector<int> graph_fmts_;     // #11: the (gate/up << 8 | down) formats of each table of graph_exec_
    std::string async_err_;
    kernels::NativeExpertLayout job_layout_{};
    const SecondaryArena* job_weights_ = nullptr;
    const float* job_x_ = nullptr;
    std::array<int32_t, 128> job_slots_{};
    int job_tokens_ = 0, job_k_ = 0;
    double ms_async_enqueue_ = 0, ms_async_wait_ = 0;
    std::chrono::steady_clock::time_point last_free_check_{};
};

} // namespace strata::core
