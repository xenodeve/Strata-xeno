#include <chrono>
#include "strata/core/secondary_runner.hpp"
#include "strata/core/end_engine.hpp"
#include "strata/core/secondary_budget.hpp"
#include "strata/core/secondary_vram.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/cpu/hadamard.hpp"
#include "strata/kernels/cpu/pool.hpp"
#include "strata/timeline.hpp"
#include "strata/timeline_gpu.hpp"

#include "strata/core/progress.hpp"
#include "strata/kernels/verify_kernels.hpp"

#include <cuda_runtime.h>
#include <immintrin.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>

namespace strata::core {
namespace {

// #33 STRATA_TIMELINE: the 4070's share of a layer on its own GPU lane (one runner per process; launch_now and
// finish never overlap - finish waits for the launch first)
timeline::GpuClock& tl_clock() {
    static timeline::GpuClock c;
    return c;
}
bool tl_anchored = false;

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
                           std::string& err, uint64_t free_floor_bytes, bool profile_timing,
                           int stage_slots, uint64_t stage_blob_bytes) {
    if (max_tokens <= 0 || max_tokens > 16 || max_entries <= 0 || max_entries > 128 ||
        n_embd <= 0 || n_embd % 32 != 0 || n_ff <= 0 || n_ff % 32 != 0) {
        err = "secondary runner: invalid verify-window geometry";
        return false;
    }
    if (stage_slots < 0 || stage_slots > kMaxStageSlots || (stage_slots > 0 && stage_blob_bytes == 0)) {
        err = "secondary runner: staging slots must be 0.." + std::to_string(kMaxStageSlots) + " of a non-empty blob";
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
    const uint64_t scratch_at = carve(kernels::native_expert_scratch_bytes(max_entries, n_ff, n_embd));
    const uint64_t ptr_at = carve((uint64_t) max_entries * sizeof(unsigned long long));
    const uint64_t start_at = carve((uint64_t) (max_entries + 1) * sizeof(int32_t));
    const uint64_t count_at = carve(sizeof(int32_t));
    const uint64_t dst_at = carve((uint64_t) max_entries * sizeof(int32_t));
    const uint64_t tok_at = carve((uint64_t) max_entries * sizeof(int32_t));
    const uint64_t meta_end = cursor;
    stage_slots_ = stage_slots;
    stage_stride_ = stage_slots > 0 ? aligned(stage_blob_bytes) : 0;
    const uint64_t stage_at = carve((uint64_t) stage_slots * stage_stride_);   // after the metadata (not in its H2D)
    meta_start_off_ = (size_t) (start_at - ptr_at);
    meta_count_off_ = (size_t) (count_at - ptr_at);
    meta_dst_off_ = (size_t) (dst_at - ptr_at);
    meta_tok_off_ = (size_t) (tok_at - ptr_at);
    meta_bytes_ = (size_t) (meta_end - ptr_at);
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
    device_stage_ = stage_slots > 0 ? base + stage_at : nullptr;

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
    if (kernels::cpu::hadamard_spec().any()) {   // APR: device 1's copies of the sign vectors (device 1 is current)
        device_sx_ = kernels::hadamard_device_signs(kernels::cpu::hadamard_signs(n_embd), n_embd);
        device_sh_ = kernels::hadamard_device_signs(kernels::cpu::hadamard_signs(n_ff), n_ff);
    }
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
                             std::string& err, const uint8_t* const* staged, int n_staged, size_t staged_bytes) {
    if (!async_) return launch_now(layout, weights, x, selected_slots, n_tokens, k, err, staged, n_staged, staged_bytes);
    if (n_staged > 0) {
        err = "secondary runner: staged experts are not taken by the async launcher";
        return false;
    }
    const auto t0 = std::chrono::steady_clock::now();
    const int entries = n_tokens * k;
    if (!launch_done_.load(std::memory_order_acquire) || x == nullptr || selected_slots == nullptr ||
        n_tokens <= 0 || k <= 0 || entries > (int) job_slots_.size()) {
        err = "secondary runner: invalid async launch or a launch still in flight";
        return false;
    }
    {
        std::lock_guard<std::mutex> lk(launch_mu_);
        job_layout_ = layout; job_weights_ = &weights; job_x_ = x;
        std::memcpy(job_slots_.data(), selected_slots, (size_t) entries * sizeof(int32_t));
        job_tokens_ = n_tokens; job_k_ = k;
        launch_done_.store(false, std::memory_order_relaxed);
        launch_req_ = true;
    }
    launch_cv_.notify_one();
    ms_launch_ += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    err.clear();
    return true;
}

bool SecondaryRunner::set_graph(bool on, std::string& err) {
    if (stream_ == nullptr) { err = "secondary runner: set_graph needs an initialised runner"; return false; }
    graph_ = on;
    graph_exec_.assign((size_t) (max_tokens_ + 1) * (size_t) (max_entries_ + 1), nullptr);
    graph_k_.assign(graph_exec_.size(), 0);
    graph_fmts_.clear();
    err.clear();
    return true;
}

bool SecondaryRunner::start_async_launch(int core, std::string& err) {
    if (stream_ == nullptr || launcher_.joinable()) {
        err = "secondary runner: async launch needs an initialised runner, once";
        return false;
    }
    launch_stop_ = false;
    launcher_ = std::thread([this, core] { launcher_loop(core); });
    async_ = true;
    err.clear();
    return true;
}

void SecondaryRunner::launcher_loop(int core) {
    kernels::cpu::pin_current_thread(core);
    cudaSetDevice(1);   // stays on the display device, so launch_now's select/restore is a no-op here
    timeline::name_thread("4070 launcher");
    for (;;) {
        std::unique_lock<std::mutex> lk(launch_mu_);
        launch_cv_.wait(lk, [this] { return launch_req_ || launch_stop_; });
        if (launch_stop_) return;
        launch_req_ = false;
        lk.unlock();
        const auto t0 = std::chrono::steady_clock::now();
        std::string e;
        timeline::Span enqueue_span("4070 enqueue", job_tokens_, job_k_);
        async_ok_ = launch_now(job_layout_, *job_weights_, job_x_, job_slots_.data(), job_tokens_, job_k_, e);
        async_err_ = e;
        ms_async_enqueue_ += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        launch_done_.store(true, std::memory_order_release);
    }
}

void SecondaryRunner::stop_launcher() {
    if (!launcher_.joinable()) return;
    {
        std::lock_guard<std::mutex> lk(launch_mu_);
        launch_stop_ = true;
    }
    launch_cv_.notify_one();
    launcher_.join();
    async_ = false;
}

bool SecondaryRunner::launch_now(const kernels::NativeExpertLayout& layout_in, const SecondaryArena& weights,
                                 const float* x, const int32_t* selected_slots, int n_tokens, int k,
                                 std::string& err, const uint8_t* const* staged, int n_staged, size_t staged_bytes) {
    // APR (prism.hadamard): a folded down projection rotates h with this device's sign vector; a folded gate/up
    // takes H (s * x), rotated below in device_x_ (this runner's private copy) before it is quantized
    kernels::NativeExpertLayout layout = layout_in;
    if (layout.had_block > 0) layout.d_signs = device_sh_;
    const int gu_block = layout.had_gu_block;
    if (stream_ == nullptr || pending_ || failed_ || x == nullptr || selected_slots == nullptr ||
        n_tokens <= 0 || n_tokens > max_tokens_ || k <= 0 || k > max_entries_ / n_tokens ||
        layout.n_embd != n_embd_ || layout.n_ff != n_ff_ || !kernels::iq_supported(layout.gu_type) ||
        !kernels::iq_supported(layout.d_type)) {
        err = "secondary runner: invalid launch geometry or pending work";
        return false;
    }
    // #11: activations as the 5060's verify path quantizes them (verify.cpp): CPU-order rounding plus fp32 scales
    // for a Q2_0 layer, plain q8_1 for any other, so a layer gives the same bytes on either card
    const bool q2 = layout.gu_type == 42 && layout.d_type == 42;
    const auto launch_t0 = std::chrono::steady_clock::now();
    ptr_.clear(); start_.clear(); dst_.clear(); tok_.clear(); selected_rows_.clear(); group_slots_.clear();
    const int entries = n_tokens * k;
    if (n_staged < 0 || n_staged > stage_slots_ || (n_staged > 0 && (staged == nullptr || staged_bytes == 0 ||
                                                                     staged_bytes > stage_stride_))) {
        err = "secondary runner: staged blobs do not fit the staging slots";
        return false;
    }
    int n_staged_used = 0;
    for (int i = 0; i < entries; ++i) {
        const int32_t slot = selected_slots[i];
        if (slot == -1) continue;
        if (slot < -1) {   // #228: staging slot -(slot + 2)
            const int s = -(slot + 2);
            if (s >= n_staged || staged[s] == nullptr) {
                err = "secondary runner: selected staging slot has no staged blob";
                return false;
            }
        } else if (weights.slot_ptr((uint64_t) slot) == nullptr) {
            err = "secondary runner: selected slot is not resident";
            return false;
        }
        bool seen = false;
        for (const int32_t prior : group_slots_) if (prior == slot) { seen = true; break; }
        if (!seen) { group_slots_.push_back(slot); n_staged_used += slot < -1; }
    }
    if (group_slots_.empty()) { err.clear(); return true; }
    for (const int32_t slot : group_slots_) {
        ptr_.push_back(slot < -1 ? (unsigned long long) (device_stage_ + (uint64_t) -(slot + 2) * stage_stride_)
                                 : (unsigned long long) weights.slot_ptr((uint64_t) slot));
        start_.push_back((int32_t) dst_.size());
        for (int i = 0; i < entries; ++i) if (selected_slots[i] == slot) {
            dst_.push_back((int32_t) selected_rows_.size());
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
    if (timeline::enabled() && !tl_anchored) { tl_clock().anchor(stream); tl_anchored = true; }
    cudaEvent_t tl_e0 = tl_clock().record(stream);
    auto tl_close = [&]() {
        static const int tl_lane = timeline::lane("gpu1 4070 experts");
        tl_clock().span(tl_lane, "4070 experts", tl_e0, tl_clock().record(stream), n_tokens, (int64_t) group_slots_.size());
    };
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
    if (n_staged_used > 0) {   // #228: the staged blobs first; stream order holds the kernels until they have landed
        cudaEvent_t st0 = tl_clock().record(stream);
        for (const int32_t slot : group_slots_) {
            if (slot >= -1) continue;
            const int s = -(slot + 2);
            if (!cuda_ok(cudaMemcpyAsync(device_stage_ + (uint64_t) s * stage_stride_, staged[s], staged_bytes,
                                         cudaMemcpyHostToDevice, stream), "runner staged blob H2D", err))
                return fail_enqueued();
        }
        static const int tl_stage_lane = timeline::lane("gpu1 4070 experts");
        tl_clock().span(tl_stage_lane, "4070 stage h2d", st0, tl_clock().record(stream), n_tokens, n_staged_used);
        staged_experts_ += (uint64_t) n_staged_used;
    }
    if (graph_ && !profile_timing_) {
        const size_t rows = selected_rows_.size();
        // a captured graph bakes in the layer's formats: one table of graphs per (gate/up, down) pair
        // (APR: and whether the layer is folded, which can differ between layers of the same formats)
        const int fmt_key = layout.gu_type << 8 | layout.d_type | (gu_block > 0 ? 1 << 20 : 0) |
                            (layout.had_block > 0 ? 1 << 21 : 0);
        const size_t per = (size_t) (max_tokens_ + 1) * (size_t) (max_entries_ + 1);
        size_t fi = 0;
        while (fi < graph_fmts_.size() && graph_fmts_[fi] != fmt_key) ++fi;
        if (fi == graph_fmts_.size()) {
            graph_fmts_.push_back(fmt_key);
            graph_exec_.resize((fi + 1) * per, nullptr);
            graph_k_.resize(graph_exec_.size(), 0);
        }
        const size_t gi = fi * per + (size_t) n_tokens * (size_t) (max_entries_ + 1) + rows;
        void*& ge = graph_exec_[gi];
        if (ge != nullptr && graph_k_[gi] != k) {
            cudaGraphExecDestroy((cudaGraphExec_t) ge);
            ge = nullptr;
        }
        if (ge == nullptr) {
            if (!cuda_ok(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal), "runner graph capture", err))
                return false;
            cudaMemcpyAsync(device_ptr_, host_meta_, meta_bytes_, cudaMemcpyHostToDevice, stream);
            cudaMemcpyAsync(device_x_, host_x_, (size_t) n_tokens * n_embd_ * sizeof(float), cudaMemcpyHostToDevice,
                            stream);
            if (gu_block > 0) kernels::hadamard_rows(device_x_, device_x_, n_tokens, n_embd_, gu_block, device_sx_, stream);
            if (q2) kernels::quantize_q8_1_rows_scaled(device_x_, n_tokens, n_embd_, device_xq_, device_scales_, stream);
            else kernels::quantize_q8_1_rows(device_x_, n_tokens, n_embd_, device_xq_, stream);
            kernels::native_expert_grouped(layout, device_ptr_, device_start_, device_count_, device_dst_, device_tok_,
                                           entries, entries, device_xq_, device_scratch_, device_out_, stream,
                                           q2 ? device_scales_ : nullptr);
            cudaMemcpyAsync(host_out_, device_out_, rows * (size_t) n_embd_ * sizeof(float), cudaMemcpyDeviceToHost,
                            stream);
            cudaGraph_t graph = nullptr;
            const cudaError_t ce = cudaStreamEndCapture(stream, &graph);
            const cudaError_t ie = ce == cudaSuccess
                ? cudaGraphInstantiate((cudaGraphExec_t*) &ge, graph, 0) : ce;
            if (graph != nullptr) cudaGraphDestroy(graph);
            if (ie != cudaSuccess) {
                ge = nullptr;
                err = std::string("secondary runner: graph capture: ") + cudaGetErrorString(ie);
                return false;
            }
            graph_k_[gi] = k;
        }
        const auto gl0 = std::chrono::steady_clock::now();
        if (!cuda_ok(cudaGraphLaunch((cudaGraphExec_t) ge, stream), "runner graph launch", err)) return fail_enqueued();
        const auto gl1 = std::chrono::steady_clock::now();
        tl_close();
        if (!cuda_ok(cudaEventRecord((cudaEvent_t) done_, stream), "runner completion event", err))
            return fail_enqueued();
        if (timeline::enabled()) {   // #220: where the launch's host time goes (the device restore follows, unstamped)
            const auto gl2 = std::chrono::steady_clock::now();
            timeline::complete("4070 l plan", launch_t0, switch_t0, n_tokens, (int64_t) rows);
            timeline::complete("4070 l device", switch_t0, now, n_tokens);
            timeline::complete("4070 l lookup", now, gl0, n_tokens);
            timeline::complete("4070 l graph launch", gl0, gl1, n_tokens);
            timeline::complete("4070 l event", gl1, gl2, n_tokens);
        }
        // #10 item 1 STRATA_SECONDARY_POKE=1: hand the batch to the 4070 now (WDDM may hold it until finish() asks)
        static const bool poke = [] { const char* v = std::getenv("STRATA_SECONDARY_POKE"); return v && v[0] == '1'; }();
        if (poke) (void) cudaStreamQuery(stream);
        pending_ = true;
        if (!async_)
            ms_launch_ += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - launch_t0).count();
        err.clear();
        return true;
    }
    const auto enqueue_t0 = std::chrono::steady_clock::now();
    if (!mark(0) ||
        !cuda_ok(cudaMemcpyAsync(device_ptr_, host_meta_, meta_bytes_, cudaMemcpyHostToDevice, stream),
                 "runner packed metadata H2D", err) ||
        !cuda_ok(cudaMemcpyAsync(device_x_, host_x_, (size_t) n_tokens * n_embd_ * sizeof(float),
                                 cudaMemcpyHostToDevice, stream), "runner activation H2D", err) ||
        !mark(1) || !mark(2)) return fail_enqueued();   // no clear: the down kernel writes every claimed row
    if (gu_block > 0) kernels::hadamard_rows(device_x_, device_x_, n_tokens, n_embd_, gu_block, device_sx_, stream);
    if (q2) kernels::quantize_q8_1_rows_scaled(device_x_, n_tokens, n_embd_, device_xq_, device_scales_, stream);
    else kernels::quantize_q8_1_rows(device_x_, n_tokens, n_embd_, device_xq_, stream);
    if (!mark(3)) return fail_enqueued();
    kernels::native_expert_grouped(layout, device_ptr_, device_start_, device_count_, device_dst_, device_tok_,
                                   host_count_, (int) dst_.size(), device_xq_, device_scratch_, device_out_, stream,
                                   q2 ? device_scales_ : nullptr);
    if (!mark(4)) return fail_enqueued();
    const size_t full_d2h_bytes = (size_t) entries * n_embd_ * sizeof(float);
    const size_t requested_d2h_bytes = selected_rows_.size() * (size_t) n_embd_ * sizeof(float);
    if (!cuda_ok(cudaMemcpyAsync(host_out_, device_out_, requested_d2h_bytes,
                                 cudaMemcpyDeviceToHost, stream), "runner partial D2H", err) ||
        !mark(5))
        return fail_enqueued();
    tl_close();
    if (!cuda_ok(cudaEventRecord((cudaEvent_t) done_, stream), "runner completion event", err))
        return fail_enqueued();
    static const bool poke = [] { const char* v = std::getenv("STRATA_SECONDARY_POKE"); return v && v[0] == '1'; }();
    if (poke) (void) cudaStreamQuery(stream);   // #10 item 1 (see the graph path above)
    pending_ = true;
    if (profile_timing_) {
        timing_.d2h_full_bytes += full_d2h_bytes;
        timing_.d2h_requested_bytes += requested_d2h_bytes;
    }
    if (profile_timing_)
        timing_.host_enqueue_ms += std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - enqueue_t0).count();
    if (!async_)   // async: launch() counts the host's own time, the helper's goes to ms_async_enqueue_
        ms_launch_ += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - launch_t0).count();
    err.clear();
    return true;
}

namespace {
// #227/#267: the runner whose window graph end_engine releases (one runner per process)
std::atomic<SecondaryRunner*> g_window_runner{nullptr};
void release_window_runner(std::FILE* f) {
    if (SecondaryRunner* r = g_window_runner.load()) {
        r->release_window();
        if (f != nullptr) std::fprintf(f, "strata: released the 4070 window graph's waits (#227, #267)\n");
    }
}
}  // namespace

const char* SecondaryRunner::window_refusal(bool spec_split, bool layer_split, int pipeline_windows, int batch_groups,
                                            bool batch_groups_auto) {
    // launch_layer takes layer 0 as a new window: a second dispatch of layer 0 in the same window, or layer 0 of the
    // next window before this one's last layer, closes the window early and the next layer fails as out of order
    if (spec_split) return "--spec-split dispatches each layer twice per window";
    if (layer_split) return "--layer-split runs two windows' layers interleaved";
    if (pipeline_windows > 0) return "--pipeline-windows runs two windows' layers interleaved";
    if (batch_groups > 1 || batch_groups_auto) return "--batch-groups runs two groups' layers interleaved";
    return nullptr;
}

// #227: before layer l's plan and activations are rewritten for window win_seq_, the previous window's segment of
// that layer must have read them (its done flag).  A layer that had claims was waited for by finish(); one without
// claims was not, and with the 4070 a window behind (WDDM preemption) its copy could read a half-written plan.
bool SecondaryRunner::wait_consumed(int layer, std::string& err) {
    if (win_seq_ < 2 || host_done_[layer] + 1 >= win_seq_) return true;
    const auto t0 = std::chrono::steady_clock::now();
    uint32_t spins = 0;
    while (host_done_[layer] + 1 < win_seq_) {
        _mm_pause();
        if ((++spins & 4095u) == 0 && std::chrono::steady_clock::now() - t0 > std::chrono::seconds(30)) {
            failed_ = true;
            err = "secondary runner: the 4070's window graph did not reach layer " + std::to_string(layer) +
                  " of the previous window in 30 s";
            return false;
        }
    }
    std::atomic_thread_fence(std::memory_order_acquire);
    return true;
}

bool SecondaryRunner::enable_window(const std::vector<kernels::NativeExpertLayout>& layouts, std::string& err) {
    if (stream_ == nullptr || window_ || async_ || profile_timing_ || layouts.empty() || layouts.size() > 256) {
        err = "secondary runner: the window graph needs an initialised runner without the async launcher or profile "
              "timing, once";
        return false;
    }
    for (const auto& L : layouts)
        if (L.n_embd != n_embd_ || L.n_ff != n_ff_ || !kernels::iq_supported(L.gu_type) || !kernels::iq_supported(L.d_type)) {
            err = "secondary runner: a layer's layout does not fit the window graph";
            return false;
        }
    int previous = -1;
    if (!cuda_ok(cudaGetDevice(&previous), "window current device", err) ||
        !cuda_ok(cudaSetDevice(1), "window select device", err)) return false;
    const RestoreDevice restore{previous};
    win_layers_ = (int) layouts.size();
    win_layout_ = layouts;
    const size_t xs = (size_t) max_tokens_ * n_embd_;
    void* go = nullptr;
    void* done = nullptr;
    if (!cuda_ok(cudaHostAlloc((void**) &host_wmeta_, (size_t) win_layers_ * meta_bytes_, cudaHostAllocPortable),
                 "window plans", err) ||
        !cuda_ok(cudaHostAlloc((void**) &host_wx_, (size_t) win_layers_ * xs * sizeof(float), cudaHostAllocPortable),
                 "window activations", err) ||
        !cuda_ok(cudaHostAlloc((void**) &host_wout_, (size_t) max_entries_ * n_embd_ * sizeof(float),
                               cudaHostAllocPortable | cudaHostAllocMapped), "window rows", err) ||
        !cuda_ok(cudaHostAlloc(&go, (size_t) win_layers_ * sizeof(uint32_t), cudaHostAllocPortable | cudaHostAllocMapped),
                 "window go flags", err) ||
        !cuda_ok(cudaHostAlloc(&done, (size_t) win_layers_ * sizeof(uint32_t), cudaHostAllocPortable | cudaHostAllocMapped),
                 "window done flags", err))
        return false;
    host_go_ = (volatile uint32_t*) go;
    host_done_ = (volatile uint32_t*) done;
    std::memset(host_wmeta_, 0, (size_t) win_layers_ * meta_bytes_);
    for (int l = 0; l < win_layers_; ++l) { host_go_[l] = 0; host_done_[l] = 0; }
    if (!cuda_ok(cudaHostGetDevicePointer((void**) &dev_wout_, host_wout_, 0), "window rows pointer", err) ||
        !cuda_ok(cudaHostGetDevicePointer((void**) &dev_go_, go, 0), "window go pointer", err) ||
        !cuda_ok(cudaHostGetDevicePointer((void**) &dev_done_, done, 0), "window done pointer", err) ||
        !cuda_ok(cudaMalloc((void**) &dev_count_, (size_t) win_layers_ * sizeof(uint32_t)), "window counts", err) ||
        !cuda_ok(cudaMemset(dev_count_, 0, (size_t) win_layers_ * sizeof(uint32_t)), "window counts clear", err))
        return false;
    win_graph_.assign((size_t) max_tokens_ + 1, nullptr);
    for (int t = 1; t <= max_tokens_; ++t)
        if (!capture_window(t, err)) return false;
    window_ = true;
    g_window_runner.store(this);
    core::release_secondary_fn().store(&release_window_runner);
    err.clear();
    return true;
}

bool SecondaryRunner::capture_window(int n_tokens, std::string& err) {
    cudaStream_t stream = (cudaStream_t) stream_;
    const int entries = n_tokens * (max_entries_ / max_tokens_);
    const size_t xs = (size_t) max_tokens_ * n_embd_;
    if (!cuda_ok(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal), "window graph capture", err))
        return false;
    for (int l = 0; l < win_layers_; ++l) {
        kernels::NativeExpertLayout layout = win_layout_[(size_t) l];
        if (layout.had_block > 0) layout.d_signs = device_sh_;
        const bool q2 = layout.gu_type == 42 && layout.d_type == 42;
        kernels::window_wait(dev_go_ + l, dev_count_ + l, stream);
        cudaMemcpyAsync(device_ptr_, host_wmeta_ + (size_t) l * meta_bytes_, meta_bytes_, cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(device_x_, host_wx_ + (size_t) l * xs, (size_t) n_tokens * n_embd_ * sizeof(float),
                        cudaMemcpyHostToDevice, stream);
        if (layout.had_gu_block > 0)
            kernels::hadamard_rows(device_x_, device_x_, n_tokens, n_embd_, layout.had_gu_block, device_sx_, stream);
        if (q2) kernels::quantize_q8_1_rows_scaled(device_x_, n_tokens, n_embd_, device_xq_, device_scales_, stream);
        else kernels::quantize_q8_1_rows(device_x_, n_tokens, n_embd_, device_xq_, stream);
        kernels::native_expert_grouped(layout, device_ptr_, device_start_, device_count_, device_dst_, device_tok_,
                                       entries, entries, device_xq_, device_scratch_, device_out_, stream,
                                       q2 ? device_scales_ : nullptr);
        kernels::window_rows_to_host(device_out_, dev_wout_, device_start_, device_count_, n_embd_, max_entries_, stream);
        kernels::window_signal(dev_done_ + l, dev_count_ + l, stream);
    }
    cudaGraph_t graph = nullptr;
    const cudaError_t ce = cudaStreamEndCapture(stream, &graph);
    void*& ge = win_graph_[(size_t) n_tokens];
    const cudaError_t ie = ce == cudaSuccess ? cudaGraphInstantiate((cudaGraphExec_t*) &ge, graph, 0) : ce;
    if (graph != nullptr) cudaGraphDestroy(graph);
    if (ie != cudaSuccess) {
        ge = nullptr;
        err = std::string("secondary runner: window graph capture: ") + cudaGetErrorString(ie);
        return false;
    }
    return true;
}

bool SecondaryRunner::publish_empty(int layer, std::string& err) {
    if (win_released_.load(std::memory_order_acquire)) return false;   // the flags stay raised past any count
    if (!wait_consumed(layer, err)) return false;
    int32_t zero = 0;
    std::memcpy(host_wmeta_ + (size_t) layer * meta_bytes_ + meta_count_off_, &zero, sizeof zero);
    std::atomic_thread_fence(std::memory_order_release);
    _mm_sfence();
    host_go_[layer] = win_seq_;
    return true;
}

void SecondaryRunner::end_window() {
    if (!window_ || win_seq_ == 0 || failed_) return;
    std::string err;
    for (int l = win_next_; l < win_layers_; ++l)
        if (!publish_empty(l, err)) break;
    win_next_ = win_layers_;
}

void SecondaryRunner::release_window() {
    if (host_go_ == nullptr) return;
    win_released_.store(true, std::memory_order_release);   // launch_layer and publish_empty refuse from now on
    for (int l = 0; l < win_layers_; ++l) host_go_[l] = UINT32_MAX;
    std::atomic_thread_fence(std::memory_order_seq_cst);
    _mm_sfence();
}

bool SecondaryRunner::launch_layer(int layer, const kernels::NativeExpertLayout& layout_in, const SecondaryArena& weights,
                                   const float* x, const int32_t* selected_slots, int n_tokens, int k, std::string& err) {
    const auto t0 = std::chrono::steady_clock::now();
    if (!window_ || pending_ || failed_ || layer < 0 || layer >= win_layers_ || x == nullptr ||
        selected_slots == nullptr || n_tokens <= 0 || n_tokens > max_tokens_ || k <= 0 || k > max_entries_ / n_tokens ||
        k != max_entries_ / max_tokens_ || layout_in.gu_type != win_layout_[(size_t) layer].gu_type ||
        layout_in.d_type != win_layout_[(size_t) layer].d_type) {
        err = "secondary runner: invalid window launch";
        return false;
    }
    if (win_released_.load(std::memory_order_acquire)) {
        // only end_engine (#203/#267) and close() release, and nothing launches after close(): the process is being
        // terminated with end_engine's code.  Wait for that rather than fail this layer, whose error would race it to
        // the server as an invalid request and an exit code of its own (the server then restarts the engine at once,
        // where a breach's code 3 keeps the tier released: measured 2026-10-17, gyro-record/breach227-win3b)
        std::this_thread::sleep_for(std::chrono::seconds(10));
        err = "secondary runner: the window graph was released (the engine is ending)";
        return false;
    }
    for (int i = 0; i < n_tokens * k; ++i)
        if (selected_slots[i] < -1) {   // a staged slot: this mode has no staging slots (stage_slots() is 0)
            err = "secondary runner: a staged slot in window mode";
            return false;
        }
    if (layer == 0) {   // a new window: close the last one, then the window's one launch
        const auto l0 = std::chrono::steady_clock::now();
        int previous = -1;
        if (!cuda_ok(cudaGetDevice(&previous), "runner current device", err) ||
            !cuda_ok(cudaSetDevice(1), "runner select device", err)) {
            failed_ = true;
            return false;
        }
        const RestoreDevice restore{previous};
        end_window();
        if (failed_) { err = "secondary runner: the window graph's last window did not end"; return false; }
        ++win_seq_;   // from here every exit without the graph launched poisons the runner (its count moved on)
        win_next_ = 0;
        win_tokens_ = n_tokens;
        cudaStream_t stream = (cudaStream_t) stream_;
        if (!cuda_ok(cudaGraphLaunch((cudaGraphExec_t) win_graph_[(size_t) n_tokens], stream), "window graph launch", err)) {
            failed_ = true;
            return false;
        }
        (void) cudaStreamQuery(stream);   // WDDM: hand the graph to the 4070 now
        if (timeline::enabled()) timeline::complete("4070 window launch", l0, std::chrono::steady_clock::now(), n_tokens);
    } else if (n_tokens != win_tokens_ || layer < win_next_) {
        err = "secondary runner: a window layer out of order";
        return false;
    }
    for (int l = win_next_; l < layer; ++l)   // layers the dispatch did not publish
        if (!publish_empty(l, err)) { failed_ = true; return false; }
    if (!wait_consumed(layer, err)) return false;
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
    for (const int32_t slot : group_slots_) {
        ptr_.push_back((unsigned long long) weights.slot_ptr((uint64_t) slot));
        start_.push_back((int32_t) dst_.size());
        for (int i = 0; i < entries; ++i) if (selected_slots[i] == slot) {
            dst_.push_back((int32_t) selected_rows_.size());
            tok_.push_back(i / k);
            selected_rows_.push_back(i);
        }
    }
    start_.push_back((int32_t) dst_.size());
    uint8_t* meta = host_wmeta_ + (size_t) layer * meta_bytes_;
    host_count_ = (int32_t) group_slots_.size();
    if (host_count_ > 0) {
        std::memcpy(host_wx_ + (size_t) layer * max_tokens_ * n_embd_, x, (size_t) n_tokens * n_embd_ * sizeof(float));
        std::memcpy(meta, ptr_.data(), ptr_.size() * sizeof(ptr_[0]));
        std::memcpy(meta + meta_start_off_, start_.data(), start_.size() * sizeof(start_[0]));
        std::memcpy(meta + meta_dst_off_, dst_.data(), dst_.size() * sizeof(dst_[0]));
        std::memcpy(meta + meta_tok_off_, tok_.data(), tok_.size() * sizeof(tok_[0]));
    }
    std::memcpy(meta + meta_count_off_, &host_count_, sizeof host_count_);
    std::atomic_thread_fence(std::memory_order_release);
    _mm_sfence();
    host_go_[layer] = win_seq_;
    win_next_ = layer + 1;
    win_layer_ = layer;
    pending_ = host_count_ > 0;
    ms_launch_ += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    err.clear();
    return true;
}

bool SecondaryRunner::finish(float* output, std::string& err) {
    if (window_) {   // #227: the layer's done flag, then its rows from mapped memory - no CUDA call
        if (!pending_) { err.clear(); return true; }
        if (output == nullptr) { err = "secondary runner: output is null"; return false; }
        const auto wait_t0 = std::chrono::steady_clock::now();
        uint32_t spins = 0;
        while (host_done_[win_layer_] < win_seq_) {
            _mm_pause();
            if ((++spins & 4095u) == 0 &&
                std::chrono::steady_clock::now() - wait_t0 > std::chrono::seconds(30)) {
                failed_ = true;
                err = "secondary runner: the 4070's window graph did not finish layer " + std::to_string(win_layer_) +
                      " in 30 s";
                return false;
            }
        }
        std::atomic_thread_fence(std::memory_order_acquire);   // the rows are read after the done flag
        const auto w1 = std::chrono::steady_clock::now();
        ms_wait_ += std::chrono::duration<double, std::milli>(w1 - wait_t0).count();
        if (timeline::enabled()) timeline::complete("4070 wait", wait_t0, w1, win_layer_);
        for (size_t dense = 0; dense < selected_rows_.size(); ++dense)
            std::memcpy(output + (size_t) selected_rows_[dense] * n_embd_, host_wout_ + dense * n_embd_,
                        (size_t) n_embd_ * sizeof(float));
        served_entries_ += selected_rows_.size();
        served_groups_ += group_slots_.size();
        pending_ = false;
        err.clear();
        return true;
    }
    if (async_) {
        const auto t0 = std::chrono::steady_clock::now();
        while (!launch_done_.load(std::memory_order_acquire)) std::this_thread::yield();
        ms_async_wait_ += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        if (timeline::enabled()) timeline::complete("4070 wait launcher", t0, std::chrono::steady_clock::now());
        if (!async_ok_) { err = async_err_; return false; }
    }
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
    if (timeline::enabled()) {
        timeline::complete("4070 wait", wait_t0, std::chrono::steady_clock::now());
        tl_clock().resolve(false);
    }
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
    for (size_t dense = 0; dense < selected_rows_.size(); ++dense)
        std::memcpy(output + (size_t) selected_rows_[dense] * n_embd_,
                    host_out_ + dense * n_embd_,
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
                const std::string why = std::string("strata: the secondary monitor cannot select the display GPU (") +
                                        cudaGetErrorString(selected) + ") - ending the engine (#203)";
                strata::core::end_engine(3, why.c_str());
            }
            // #203 fault injection: the monitor sees a breach this many seconds after it starts (the engine must
            // then end at once, not hang in its exit)
            static const double inject_s = [] {
                const char* v = std::getenv("STRATA_TEST_SECONDARY_BREACH_S");
                return v != nullptr ? std::atof(v) : -1.0;
            }();
            const auto started = std::chrono::steady_clock::now();
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
                    if (safe && inject_s >= 0 && std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                                                               started).count() >= inject_s) {
                        safe = false;
                        problem = "injected breach (STRATA_TEST_SECONDARY_BREACH_S)";
                    }
                }
                record_free(lower);
                if (!safe && breach_warn_) {   // xeno #208: others grew into free memory; keep serving
                    static auto last = std::chrono::steady_clock::time_point{};
                    const auto now = std::chrono::steady_clock::now();
                    if (last == std::chrono::steady_clock::time_point{} || now - last >= std::chrono::seconds(60)) {
                        last = now;
                        std::fprintf(stderr, "strata: secondary display VRAM below the free floor (%s; %.0f MiB free) - "
                                             "--secondary-breach warn: the engine keeps serving (#208)\n",
                                     problem.c_str(), (double) lower / 1048576.0);
                    }
                    ++free_checks_;
                    continue;
                }
                if (!safe) {
                    if (on_breach != nullptr) {
                        on_breach(problem.c_str(), context);
                        monitor_stop_.store(true);
                        break;
                    }
                    const std::string why = "strata: secondary display VRAM reserve failed: " + problem +
                                            " - ending the engine to release the tier (#203)";
                    strata::core::end_engine(3, why.c_str());
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
    stop_launcher();
    stop_monitor();
    if (window_) {   // #227: no segment may still wait when the stream is drained below
        release_window();
        SecondaryRunner* self = this;
        g_window_runner.compare_exchange_strong(self, nullptr);
    }
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
    for (void*& ge : graph_exec_)
        if (ge != nullptr) { cudaGraphExecDestroy((cudaGraphExec_t) ge); ge = nullptr; }
    for (void*& ge : win_graph_)
        if (ge != nullptr) { cudaGraphExecDestroy((cudaGraphExec_t) ge); ge = nullptr; }
    if (dev_count_ != nullptr) cudaFree(dev_count_);
    for (void* p : {(void*) host_wmeta_, (void*) host_wx_, (void*) host_wout_, (void*) host_go_, (void*) host_done_})
        if (p != nullptr) cudaFreeHost(p);
    dev_count_ = nullptr; host_wmeta_ = nullptr; host_wx_ = nullptr; host_wout_ = nullptr;
    host_go_ = nullptr; host_done_ = nullptr; window_ = false;
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
