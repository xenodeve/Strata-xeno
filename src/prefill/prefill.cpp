// src/prefill/prefill.cpp - see include/strata/prefill/prefill.hpp.
#include "strata/prefill/prefill.hpp"
#include "strata/core/mtp.hpp"
#include "strata/core/progress.hpp"
#include "strata/core/on_device.hpp"

#include "strata/core/layout.hpp"
#include "strata/core/native_head.hpp"
#include "strata/kernels/cpu/expert.hpp"
#include "strata/kernels/native_qsa_indexer.hpp"
#include "strata/kernels/ngram.hpp"
#include "strata/kernels/ple.hpp"
#include "strata/kernels/native_ple_postops.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/kv_stream.hpp"
#include "strata/kernels/cvec.hpp"
#include "strata/kernels/kv_q4.hpp"
#include "strata/core/layer.hpp"
#include "strata/core/native_head.hpp"
#include "strata/kernels/verify_kernels.hpp"
#include "strata/kernels/qsa_decode_attn.hpp"
#include "strata/kernels/qsa_prompt_attn.hpp"
#include "strata/kernels/qsa_select.hpp"
#include "strata/prefill/gemm.hpp"
#include "strata/prefill/moe_mmq.hpp"
#include "strata/prefill/frontier.hpp"
#include "strata/prefill/kernels.hpp"
#include "strata/timeline.hpp"
#include "strata/timeline_gpu.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cmath>
#include <cstdio>
#include <atomic>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#ifndef STRATA_PREFILL_MMQ
// A build without the llama.cpp sources (no STRATA_NATIVE_EXPERTS): no MMQ, the FP16 expert path everywhere.
namespace strata::prefill::mmq {
bool built() { return false; }
bool supported(int) { return false; }
size_t matrix_bytes(int, int64_t, int64_t) { return 0; }
size_t q8_bytes(int64_t, int64_t) { return 0; }
void quantize(const float*, const int32_t*, void*, int, int64_t, int64_t, int64_t, void*) {}
void warm() {}
Context::Context() {}
Context::~Context() {}
void Context::run(const Product&, void*) {}
void gather_native(const void*, const void*, size_t, const void*, size_t, void*, void*, void*) {}
void gather_strata_q2(const uint8_t*, void*, void*, void*) {}
void swiglu(const float*, float*, int64_t, int64_t, bool, void*) {}
void iota(int32_t*, int64_t, void*) {}
void expert_rows(Context&, const ExpertRows&, void*) {}
void expert_rows_gate_up(Context&, const ExpertRows&, void*) {}
void expert_rows_down(Context&, const ExpertRows&, void*) {}
}  // namespace strata::prefill::mmq
#endif

namespace strata::prefill {
namespace {

using Clock = std::chrono::steady_clock;
constexpr float EPS = 1e-6f;
constexpr int64_t N = 2560, HC = 4, D = N * HC, LR = 320, K = 10, NE = 512;
constexpr int64_t C = 10240, ZV = 6144, HV = 48;
// plan v0.3 P6: staging holds the largest blob of the pack (a native pack's blobs differ per layer)
inline int64_t MAXBLOB() { return (int64_t) strata::kernels::cpu::expert_layout().max_blob; }
constexpr int STAGE = 8;           // host->device expert staging ring (chunks below stream_all_min())
// Step 3: from this chunk size on, every non-resident expert of every layer streams in a fixed order through a
// RING_MAX-slot ring (nearly all 512 are routed at such a chunk), so the copy engine keeps working through the
// attention halves instead of waiting for each layer's routing.
constexpr int RING_MAX = 512;           // the arrays; the ring itself is ring_slots()
constexpr int64_t STREAM_ALL_MIN = 2048;
constexpr int MMQ_GROUP = 16;
static_assert(MMQ_GROUP <= mmq::kGatherGroupMax, "one gather_native_group launch must hold an MMQ group");                  // experts per MMQ launch (the gather is per expert, as blobs arrive)
// MMQ reads up to one 256-value tile past a matrix's last row when the row length is not a multiple of it (the down
// product: 640 values).  Those bytes meet zero activations, which is harmless only if they decode to finite numbers -
// llama.cpp zero-pads after every tensor, and so does a group buffer: this many zeroed bytes follow its last expert.
constexpr size_t MMQ_TAIL = 4096;
// The chunk size from which every expert streams: 1024 since 0.1.30 (was 2048).  Measured on the 5070, Q2_0 / IQ2_XS,
// fixed cache: 1,500-token prompts 621 -> 785 / 612 -> 735 tok/s, 2,000 727 -> 934 / 712 -> 892, 4,000 (its last
// chunk) 779 -> 912 / 766 -> 844, the same output.  Below ~1,000 tokens the output changed on Q2_0 (a smaller chunk
// takes other kernels), so 1024 is the floor.  STRATA_PREFILL_STREAM_MIN overrides (A/B).
inline int64_t stream_all_min() {
    static const int64_t v = [] { const char* e = std::getenv("STRATA_PREFILL_STREAM_MIN"); return e ? (int64_t) std::atoll(e) : (int64_t) 1024; }();
    return v;
}
double g_pinned_share = 1.0;
bool g_split_layout = false;   // #35 D6 (Prefill::set_split_layout)
// The streamed ring: 384 slots when (nearly) every streamed expert is DMA'd from pinned RAM - measured on Q2_0,
// 8192-token chunks: 96 slots 1153 tok/s, 384 1294 (the next layer's experts arrive during its attention half) -
// and 96 when a large share goes through host copies (IQ3_S on 64 GB, a third unpinned: 96 slots 1216, 256 1070 -
// the host copies are the limit and the bigger ring only takes cache slots).  STRATA_PREFILL_RING overrides.
// #41: the order a layer's experts run in - the row layout, the MMQ groups and both stream plans follow it.  A row's
// bytes do not depend on its group (xeno_moe_layer_cross_arch), so the output does not change (gate 2).
//   STRATA_EXPERT_ORDER=reverse   reverse id order (the gate-2 check)
//   STRATA_EXPERT_ORDER=<file>    a static per-layer order (tests/xeno/perf/make_expert_order.py: mean router rank
//                                 first), which keeps a Dm frontier small (#41: 29.7 % of T*K against 48.9 % in id order)
//   unset                         id order
struct ExpertOrder {
    bool reverse = false;
    int64_t layers = 0, experts = 0;
    std::vector<int32_t> at, pos;   // per layer: position -> expert, expert -> position
};
inline const ExpertOrder& expert_order() {
    static const ExpertOrder o = [] {
        ExpertOrder r;
        const char* e = std::getenv("STRATA_EXPERT_ORDER");
        if (e == nullptr || *e == 0) return r;
        if (std::strcmp(e, "reverse") == 0) { r.reverse = true; return r; }
        std::FILE* f = std::fopen(e, "rb");
        int32_t hdr[3] = {0, 0, 0};
        if (f == nullptr || std::fread(hdr, 4, 3, f) != 3 || hdr[0] != 0x4F585053 || hdr[1] <= 0 || hdr[2] <= 0) {
            std::fprintf(stderr, "strata prefill: STRATA_EXPERT_ORDER=%s is not an expert-order file; id order\n", e);
            if (f) std::fclose(f);
            return r;
        }
        r.layers = hdr[1];
        r.experts = hdr[2];
        // the header must match the file's size before anything is sized from it (a corrupt one would throw here)
        std::fseek(f, 0, SEEK_END);
        const long long bytes = (long long) std::ftell(f);
        std::fseek(f, 12, SEEK_SET);
        if (r.layers > 4096 || r.experts > 65536 || bytes != 12 + 4LL * r.layers * r.experts) {
            std::fprintf(stderr, "strata prefill: STRATA_EXPERT_ORDER=%s: header and size disagree; id order\n", e);
            std::fclose(f);
            return ExpertOrder{};
        }
        r.at.resize((size_t) (r.layers * r.experts));
        r.pos.assign(r.at.size(), -1);
        const bool ok = std::fread(r.at.data(), 4, r.at.size(), f) == r.at.size();
        std::fclose(f);
        for (int64_t l = 0; ok && l < r.layers; ++l)
            for (int64_t i = 0; i < r.experts; ++i) {
                const int32_t x = r.at[(size_t) (l * r.experts + i)];
                if (x >= 0 && x < r.experts) r.pos[(size_t) (l * r.experts + x)] = (int32_t) i;
            }
        for (int32_t v : r.pos)
            if (!ok || v < 0) {   // not a permutation of every layer: never run a partial order
                std::fprintf(stderr, "strata prefill: STRATA_EXPERT_ORDER=%s is not a permutation; id order\n", e);
                return ExpertOrder{};
            }
        std::fprintf(stderr, "strata prefill: expert order from %s (%lld layers x %lld experts)\n", e,
                     (long long) r.layers, (long long) r.experts);
        return r;
    }();
    return o;
}
inline int32_t expert_at(int64_t l, int32_t i, int64_t n_expert) {
    const ExpertOrder& o = expert_order();
    if (o.reverse) return (int32_t) (n_expert - 1 - i);
    if (!o.at.empty() && (l >= o.layers || n_expert != o.experts)) {   // say once that this layer runs in id order
        static std::atomic<bool> warned{false};
        if (!warned.exchange(true))
            std::fprintf(stderr, "strata prefill: the expert order covers %lld layers x %lld experts; layer %lld "
                                 "(%lld experts) and any like it run in id order\n", (long long) o.layers,
                         (long long) o.experts, (long long) l, (long long) n_expert);
    }
    if (!o.at.empty() && l < o.layers && n_expert == o.experts) return o.at[(size_t) (l * o.experts + i)];
    return i;
}
// an expert's position in that order (n_expert itself: past every expert)
inline int64_t expert_pos(int64_t l, int64_t e, int64_t n_expert) {
    if (e >= n_expert) return e;
    const ExpertOrder& o = expert_order();
    if (o.reverse) return n_expert - 1 - e;
    if (!o.pos.empty() && l < o.layers && n_expert == o.experts) return o.pos[(size_t) (l * o.experts + e)];
    return e;
}
inline int ring_slots(size_t T) {
    const char* v = std::getenv("STRATA_PREFILL_RING");
    const int r = v ? std::atoi(v) : (g_pinned_share >= 0.9 ? 384 : 96);
    if (v && r == STAGE) return STAGE; // Explicit opt-in to routed-only staging, including large chunks.
    const int big = r < 16 ? 16 : r > RING_MAX ? RING_MAX : r;
    return (int64_t) T >= stream_all_min() ? big : STAGE;
}
constexpr int DQ = 2;              // dequantized-expert ring (FP16 gate/up + down)
// F-1 (upstream 882bb6d, #42): STRATA_GR_UNFUSED=1 keeps the FP32 copy of the normalized rows (gr_norm + gr_mix, and
// no F-2 fusion), the A/B arm
inline bool gr_unfused() {
    static const bool v = [] { const char* e = std::getenv("STRATA_GR_UNFUSED"); return e && e[0] == '1'; }();
    return v;
}
// #35 D6: the tokens the one-card MoE buffers hold: every chunk, or in the split layout only the chunks below
// STREAM_ALL_MIN (a bigger one runs its routed experts on the peer card and never touches them)
bool split_layout_usable();   // below: the split's static conditions (a native pack, every layer on MMQ)
inline size_t moe_cap(size_t T) {
    return g_split_layout && split_layout_usable() ? std::min(T, (size_t) STREAM_ALL_MIN - 1) : T;
}


double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }

// Either cudaMalloc (owned, freed with the object) or a bump allocation from a borrowed region; with no base and
// no region it only counts, which is how `bytes_needed` sizes the region.
struct Alloc {
    uint8_t* base = nullptr;
    uint64_t cap = 0, used = 0;
    bool count_only = false;
    std::vector<void*>* owned = nullptr;
    template <typename T> T* take(size_t n, bool& ok) {
        const uint64_t bytes = ((uint64_t) n * sizeof(T) + 256 + 255) & ~255ull;
        if (count_only) { used += bytes; return nullptr; }
        if (base != nullptr) {
            if (used + bytes > cap) { ok = false; return nullptr; }
            T* p = (T*) (base + used);
            used += bytes;
            return p;
        }
        void* p = nullptr;
        if (cudaMalloc(&p, bytes) != cudaSuccess) { ok = false; return nullptr; }
        owned->push_back(p);
        used += bytes;
        return (T*) p;
    }
};

}  // namespace

// Step 4 of the prompt-speed plan: the experts the arena could not pin (a third of the streamed ones on IQ3_S) are
// copied into pinned buffers by these threads, ahead of the launches.  Copied in line by the launching thread they
// left the GPU without queued work while each ~2 MB memcpy ran (~15 s of a 32K prompt on IQ3_S).  Job j - a layer's
// j-th unpinned expert, in launch order - lands in host buffer j % kRing, which is free again once the DMA of job
// j - kRing (recorded by the launching thread, `issued`) is done.
struct Stager {
    // D-5: the pinned ring's depth (STRATA_STAGER_RING, default 16) - how far the host copies can run ahead of the
    // DMAs of the unpinned experts' blobs.  (xeno #31: 128 and 384 measured no faster at 8K and cost pinned RAM.)
    int kRing = 16;
    struct Job { const uint8_t* src; size_t bytes; int32_t l = -1, e = -1; };   // src null: read (l, e) from the pack
    core::ExpertSource* xsrc = nullptr;   // #11: the NVMe tier's experts are read, never admitted
    std::vector<uint8_t*> buf;
    std::vector<char> pinned;
    std::vector<std::vector<uint8_t>> pageable;   // the fallback when no more RAM can be pinned
    std::vector<cudaEvent_t> dma_done;
    std::vector<Job> jobs;
    std::unique_ptr<std::atomic<int>[]> ready;
    size_t ready_cap = 0;
    // gen << 32 | n << 16 | next index: a claim is a CAS on the generation it woke for (a thread late from the
    // previous layer can never take a job of this one - the expert pool's issue #29 lesson)
    std::atomic<uint64_t> head{0};
    std::atomic<int> issued{0}, active{0};
    uint32_t gen = 0;
    bool quit = false;
    std::mutex mu;
    std::condition_variable cv;
    std::vector<std::thread> threads;
    int device = 0;

    bool init(size_t blob_bytes, int nthreads) {
        if (const char* v = std::getenv("STRATA_STAGER_RING")) kRing = std::clamp(std::atoi(v), 2, 256);
        buf.assign((size_t) kRing, nullptr);
        pinned.assign((size_t) kRing, 0);
        dma_done.assign((size_t) kRing, nullptr);
        pageable.resize(kRing);
        for (int i = 0; i < kRing; ++i) {
            pinned[i] = cudaHostAlloc((void**) &buf[i], blob_bytes, cudaHostAllocDefault) == cudaSuccess;
            if (!pinned[i]) {
                cudaGetLastError();
                pageable[(size_t) i].resize(blob_bytes);
                buf[i] = pageable[(size_t) i].data();
            }
            // #35 D4: a thread that waits for a DMA sleeps instead of spinning (the host thread needs the core)
            if (cudaEventCreateWithFlags(&dma_done[i], cudaEventDisableTiming | cudaEventBlockingSync) != cudaSuccess)
                return false;
        }
        cudaGetDevice(&device);
        int n_pinned = 0;
        for (int i = 0; i < kRing; ++i) n_pinned += pinned[i] ? 1 : 0;
        // a pageable staging buffer makes every DMA from it synchronous on the launching thread: say so
        std::fprintf(stderr, "strata prefill: stager %d threads, %d of %d staging buffers pinned\n", nthreads, n_pinned,
                     kRing);
        for (int t = 0; t < nthreads; ++t) threads.emplace_back([this, t] { work(t); });
        return true;
    }
    ~Stager() {
        const bool tr = std::getenv("STRATA_EXIT_TRACE") != nullptr;   // #45 exit-hang probe
        if (tr) { std::fprintf(stderr, "exit trace: ~Stager start, active %d\n", active.load()); std::fflush(stderr); }
        finish();
        if (tr) { std::fprintf(stderr, "exit trace: ~Stager finished\n"); std::fflush(stderr); }
        { std::lock_guard<std::mutex> lk(mu); quit = true; }
        cv.notify_all();
        if (tr) { std::fprintf(stderr, "exit trace: ~Stager joining %zu threads, quit %d gen %u\n", threads.size(), (int) quit, gen); std::fflush(stderr); }
        for (size_t ti = 0; ti < threads.size(); ++ti) {
            threads[ti].join();
            if (tr) { std::fprintf(stderr, "exit trace: ~Stager joined %zu\n", ti); std::fflush(stderr); }
        }
        if (tr) { std::fprintf(stderr, "exit trace: ~Stager joined\n"); std::fflush(stderr); }
        for (int i = 0; i < kRing; ++i) {
            if (dma_done[i]) cudaEventDestroy(dma_done[i]);
            if (buf[i] && pinned[i]) cudaFreeHost(buf[i]);
        }
    }
    void work(int id) {
        cudaSetDevice(device);
        if (timeline::enabled()) {
            char nm[40];
            std::snprintf(nm, sizeof nm, "prefill stager %d", id);
            timeline::name_thread(nm);
        }
        uint32_t seen = 0;
        for (;;) {
            {
                std::unique_lock<std::mutex> lk(mu);
                cv.wait(lk, [&] { return quit || gen != seen; });
                if (quit) {
                    if (std::getenv("STRATA_EXIT_TRACE")) { std::fprintf(stderr, "exit trace: stager %d returns\n", id); std::fflush(stderr); }
                    return;
                }
                seen = gen;
            }
            for (;;) {
                active.fetch_add(1, std::memory_order_acq_rel);
                const int j = claim(seen);
                if (j < 0) { active.fetch_sub(1, std::memory_order_acq_rel); break; }
                const int b = j % kRing;
                if (j >= kRing) {
                    timeline::Span wait_span("stager wait buffer", j, j - kRing);
                    for (int v; (v = issued.load(std::memory_order_acquire)) <= j - kRing;) issued.wait(v);
                    cudaEventSynchronize(dma_done[b]);
                }
                const Job& jb = jobs[(size_t) j];
                timeline::Span stage_span(jb.src ? "stager memcpy" : "stager nvme read", j, jb.l);
                if (jb.src) {
                    std::memcpy(buf[b], jb.src, jb.bytes);
                } else {
                    std::string re;
                    if (!xsrc || !xsrc->read_into(jb.l, jb.e, buf[b], re)) {
                        std::fprintf(stderr, "prefill: NVMe read of (%d,%d) failed: %s\n", jb.l, jb.e, re.c_str());
                        std::abort();   // never stale expert data
                    }
                }
                ready[(size_t) j].store(1, std::memory_order_release);
                ready[(size_t) j].notify_all();
                active.fetch_sub(1, std::memory_order_acq_rel);
            }
        }
    }
    int claim(uint32_t g) {
        uint64_t cur = head.load(std::memory_order_acquire);
        for (;;) {
            if ((uint32_t) (cur >> 32) != g) return -1;
            const int n = (int) ((cur >> 16) & 0xffff), j = (int) (cur & 0xffff);
            if (j >= n) return -1;
            if (head.compare_exchange_weak(cur, cur + 1, std::memory_order_acq_rel, std::memory_order_acquire)) return j;
        }
    }
    /// A layer's jobs; the previous layer's are finished (finish()).
    void start(std::vector<Job>&& js) {
        if (js.empty()) return;
        std::lock_guard<std::mutex> lk(mu);
        jobs = std::move(js);
        if (ready_cap < jobs.size()) {
            ready_cap = jobs.size() * 2;
            ready.reset(new std::atomic<int>[ready_cap]);
        }
        for (size_t i = 0; i < jobs.size(); ++i) ready[i].store(0, std::memory_order_relaxed);
        issued.store(0);
        ++gen;
        head.store((uint64_t) gen << 32 | (uint64_t) jobs.size() << 16, std::memory_order_release);
        cv.notify_all();
    }
    /// Job j's bytes, in a pinned buffer (waits for the copy).
    const uint8_t* wait(int j) {
        while (!ready[(size_t) j].load(std::memory_order_acquire)) ready[(size_t) j].wait(0);
        return buf[j % kRing];
    }
    /// The launching thread queued job j's DMA on `copy`: its buffer is free once that is done.
    void issued_one(int j, cudaStream_t copy) {
        cudaEventRecord(dma_done[j % kRing], copy);
        issued.store(j + 1, std::memory_order_release);
        issued.notify_all();
    }
    /// No job is running after this (the end of a layer, or an early return in the middle of one).
    void finish() {
        head.store((uint64_t) gen << 32, std::memory_order_release);   // n = 0: nothing more to claim
        issued.store(1 << 30, std::memory_order_release);
        issued.notify_all();
        while (active.load(std::memory_order_acquire) != 0) std::this_thread::yield();
    }
};

// #32 S4 expert_split (STRATA_PREFILL_EXPERT_SPLIT=1), the simulator's whole_4070 policy (it wins every layer of
// the real 8K dual-GPU trace, 2.2 vs 4.0 s lower bound): the 4070 runs every routed expert of an MoE layer and their
// routed sum.  The 5060 keeps the trunk, the router and the shared expert, and finishes bo with the shared term (#35
// D1: moe_routed_sum + moe_shared_finish == moe_combine, xeno_combine_split_parity).  Per layer, over the 5060's x4
// link: the per-token q8 activations and the gates go down (5060 -> pinned host -> 4070), the weights of the experts
// only the 5060 holds go down too, and the routed sum comes back.  Every other expert reaches the 4070 over its own
// x16: the ones it owns in place, the host ones through a pinned stager and a ring.  The products are
// mmq::expert_rows, byte-identical across the cards (xeno_moe_layer_cross_arch).
// #35 D7: in a wave, the lane whose chunk is ahead (lane 0: chunk c-1 of every pair) must hand its layer to the 4070
// before the other lane's trunk fills this card, or both lanes' MoE land on the 4070 together (tlD7: lockstep).
inline int wave_priority(bool high) {
    int least = 0, greatest = 0;
    cudaDeviceGetStreamPriorityRange(&least, &greatest);
    return high ? greatest : least;
}
inline bool make_stream(cudaStream_t* st, int prio) {
    return cudaStreamCreateWithPriority(st, cudaStreamNonBlocking, prio) == cudaSuccess;
}
struct SplitTier {
    static constexpr int64_t R = 4096;   // rows per sub-product (an expert with more rows spans several)
    // 4070 expert staging slots.  96, 192, 320, 480 and 512 measured the same (8.85-9.1 s at 8K) once the 5060's
    // experts came down on their own thread: the smallest keeps the VRAM
    int RING = 96;   // #35 D4: 512 (a whole layer, +~500 MB of VRAM) measured 1.6 % vs 1.0 % faster than D1: not worth it
    static constexpr int RES = 32;       // pinned slots for the 5060's own experts on their way down
    int dev = -1, home = 0;
    int64_t T = 0;                       // the token capacity
    size_t blob = 0;                     // the largest expert blob, 16-byte aligned
    // on the 4070: compute, host expert copies, the 5060's experts' copies (they wait on the x4: a stream of their
    // own, so the host copies queued behind them in one FIFO do not), late inputs
    cudaStream_t s = nullptr, c = nullptr, cr = nullptr, u = nullptr;
    cudaStream_t res_stream = nullptr;                      // on the 5060: its own experts on their way down
    std::unique_ptr<mmq::Context> ctx;       // on the 4070
    std::unique_ptr<Stager> stager;          // on the 4070: host experts into pinned buffers
    void *xtok = nullptr, *xq = nullptr, *hq = nullptr;
    float *gu = nullptr, *h = nullptr, *dm = nullptr, *w = nullptr, *bo = nullptr;   // bo: the routed sum
    // #41 STRATA_DM_FRONTIER=1: no Dm; a sub-product's rows land in fg, the ones that must wait in fpool (pool_cap rows,
    // STRATA_DM_FRONTIER_FRAC of T*K, default 0.6), and a host plan (hplan -> dplan) drives the k-order commits into bo
    bool frontier = false;
    float *fg = nullptr, *fpool = nullptr;
    int32_t *dplan = nullptr, *hplan = nullptr;
    size_t plan_cap = 0;
    int32_t pool_cap = 0;
    cudaEvent_t ev_plan = nullptr;
    FrontierPlan fplan;
    int32_t *rows = nullptr, *slot = nullptr, *ids = nullptr, *bounds = nullptr;
    uint8_t *grp_gu = nullptr, *grp_d = nullptr, *ring = nullptr;
    size_t grp_gu_bytes = 0, grp_d_bytes = 0;
    // pinned relay buffers (portable) and the host side of the uploads
    uint8_t *hx = nullptr, *hres = nullptr;
    float *hw = nullptr, *hbo = nullptr;
    int32_t *hrows = nullptr, *hslot = nullptr, *hbounds = nullptr;
    int64_t hbounds_cap = 0;
    // on the 5060: the activations in host, the gates in host, the output uploaded, a resident
    // expert's blob in host
    cudaEvent_t ev_x = nullptr, ev_gates = nullptr, ev_done = nullptr, ev_res[RES] = {};
    cudaEvent_t ev_q = nullptr;   // on the 5060: the relay stream waits for the compute stream's quantize
    // on the 4070: the inputs read, the uploads read, the output in host, a pinned slot / ring slot free, copied
    cudaEvent_t ev_xread = nullptr, ev_gatesread = nullptr, ev_meta = nullptr, ev_bo = nullptr, ev_resread[RES] = {};
    std::vector<cudaEvent_t> used, copied;   // per ring slot
    // the chunk's stream plan: every expert of every layer the 4070 does not own, in (layer, id) order; entry k lands
    // in ring slot k % RING and is issued (by the issuer thread) once entry k - RING is consumed
    struct Entry { int32_t l, e; int job; uint8_t kind; };   // kind: 1 the 5060's cache, 2 pinned host, 3 staged
    std::vector<Entry> seq;
    std::vector<size_t> seq_start;                           // per layer, and one past the last
    std::atomic<size_t> consumed{0}, issued{0};
    std::atomic<bool> stop{false};
    std::thread issuer;
    // #35 D7: the expert stream above (ring, plan, issuer, stager, resident thread) belongs to one SplitTier, `xs`.
    // In a wave both lanes gather from it, so a layer's experts cross the x16 once for the pair of chunks: lane 2's
    // gathers record used1 and its progress is consumed1; a slot is reused once both lanes have gathered it.  The
    // plan is per unit (a chunk, or a wave's chunk pair): seq_start holds units * (layers + 1) entries.
    SplitTier* xs = this;
    bool stream_owner = true;              // false: lane 2's SplitTier (no ring, stager or plan of its own)
    static constexpr size_t kNever = (size_t) 1 << 60;   // a counter past any entry (a lane that no longer holds)
    std::vector<cudaEvent_t> used1;        // lane 2's gathers, per ring slot
    std::atomic<size_t> consumed1{kNever};
    std::atomic<int> lanes_done{0};
    int lanes_expected = 1;                // the lanes that consume this plan (2: a wave whose lane 2 has a chunk)
    std::atomic<bool> aborted{false};      // a wave lane failed: every waiter below gives up
    bool plan_live = false;
    int64_t layers = 0;
    std::vector<std::vector<Stager::Job>> unit_jobs;
    std::vector<size_t> unit_first;        // per unit: its first entry, and one past the last unit
    // the 5060's own experts go down on a thread of their own, bound to the 5060 (the issuer switching devices for
    // each one cost ~1.5 ms per expert on WDDM): res_ready counts the ones in hres, res_uploaded the ones the issuer
    // has sent up (their hres slot is free once that upload is done)
    std::vector<size_t> res_list;   // the stream plan's entries of kind 1, in order
    std::atomic<size_t> res_ready{0}, res_uploaded{0};
    std::thread res_thread;
    // #35 D4: every wait below blocks on the atomic it watches (a spinning helper took the host thread's core: with a
    // whole-layer ring, host grouping rose from 3.6 to 14.5 ms per layer); join sets them past any wait's target
    template <class A> static void publish(A& a, size_t v) { a.store(v, std::memory_order_release); a.notify_all(); }
    template <class A> static bool wait_above(A& a, size_t k, const std::atomic<bool>& stop) {   // until a > k
        for (size_t v; (v = a.load(std::memory_order_acquire)) <= k;) {
            if (stop.load(std::memory_order_acquire)) return false;
            a.wait(v);
        }
        return !stop.load(std::memory_order_acquire);
    }
    void join_issuer() {
        stop.store(true);
        for (auto* a : {&consumed, &consumed1, &issued, &res_ready, &res_uploaded}) publish(*a, kNever);
        if (issuer.joinable()) issuer.join();
        if (res_thread.joinable()) res_thread.join();
        if (tl_res >= 0) clk_res.resolve(true);
        if (stager) stager->finish();
        if (dev >= 0) { Dev g(dev); clk_c.resolve(true); }
        stop.store(false);
        plan_live = false;
    }
    // per chunk: experts by source (4070 slots, 5060 cache, host)
    int64_t last_l = -1, n_own = 0, n_res = 0, n_host = 0;
    // #33 STRATA_TIMELINE: the 4070's compute and copy streams and the 5060's relay stream, each on its own clock
    timeline::GpuClock clk_s, clk_c, clk_relay, clk_res;   // clk_res: res_thread's (the 5060's resident copies)
    int tl_s = -1, tl_c = -1, tl_relay = -1, tl_res = -1, tl_in = -1;
    void resolve(bool wait) {   // clk_c is the issuer's: join_issuer resolves it
        if (!timeline::enabled()) return;
        { Dev g(dev); clk_s.resolve(wait); }
        clk_relay.resolve(wait);
    }
    std::vector<void*> dev_bufs;
    struct Dev {   // run the enclosed calls on device d
        int prev = 0;
        explicit Dev(int d) { cudaGetDevice(&prev); cudaSetDevice(d); }
        ~Dev() { cudaSetDevice(prev); }
    };
    void report() {
        if (n_own + n_res + n_host == 0) return;
        std::fprintf(stderr, "strata prefill: expert_split: %lld experts from the 4070's slots, %lld from the 5060's "
                             "cache, %lld from host\n", (long long) n_own, (long long) n_res, (long long) n_host);
        n_own = n_res = n_host = 0;
    }
    ~SplitTier() {
        if (dev < 0) return;
        join_issuer();
        report();
        resolve(true);
        stager.reset();
        { Dev g(dev); if (s) cudaStreamSynchronize(s); if (c) cudaStreamSynchronize(c); if (cr) cudaStreamSynchronize(cr);
          for (void* b : dev_bufs) cudaFree(b);
          for (cudaEvent_t e : {ev_xread, ev_gatesread, ev_meta, ev_bo, ev_plan}) if (e) cudaEventDestroy(e);
          for (cudaEvent_t e : ev_resread) if (e) cudaEventDestroy(e);
          for (size_t i = 0; i < used.size(); ++i) { if (used[i]) cudaEventDestroy(used[i]); if (copied[i]) cudaEventDestroy(copied[i]); }
          for (cudaEvent_t e : used1) if (e) cudaEventDestroy(e);
          if (s) cudaStreamDestroy(s); if (c) cudaStreamDestroy(c); if (cr) cudaStreamDestroy(cr);
          if (u) cudaStreamDestroy(u); ctx.reset(); }
        if (res_stream) { cudaStreamSynchronize(res_stream); cudaStreamDestroy(res_stream); }
        for (cudaEvent_t e : {ev_x, ev_gates, ev_done, ev_q}) if (e) cudaEventDestroy(e);
        for (cudaEvent_t e : ev_res) if (e) cudaEventDestroy(e);
        for (void* b : {(void*) hx, (void*) hres, (void*) hw, (void*) hbo, (void*) hrows,
                        (void*) hslot, (void*) hbounds, (void*) hplan}) if (b) cudaFreeHost(b);
    }
    /// allocate for chunks of up to T tokens; false (and err) when the 4070 cannot hold it
    bool high = false;   // #35 D7: this lane's streams run at the cards' highest priority
    std::string lane_tag;   // #35 D7: ", lane 2" for the wave's second lane (the timeline keeps them apart)
    bool init(int peer, int64_t tokens, int64_t n_expert, size_t gub, size_t db, size_t max_blob,
              core::ExpertSource* xsrc, std::string& err) {
        cudaGetDevice(&home);
        dev = peer;
        T = tokens;
        blob = (max_blob + 255) / 256 * 256;
        grp_gu_bytes = gub;
        grp_d_bytes = db;
        bool ok = true;
        auto h_alloc = [&](size_t n) -> void* {
            void* q = nullptr;
            if (cudaHostAlloc(&q, n, cudaHostAllocPortable) != cudaSuccess) { ok = false; cudaGetLastError(); return nullptr; }
            return q;
        };
        // #35 D4: the cross-card events are waited on by the host (blocking), never by a stream of the other card:
        // such a wait sat in the copy engine's queue and held back every copy behind it (58 ms stalls, tlD4/tlD4b)
        for (cudaEvent_t* e : {&ev_x, &ev_gates, &ev_done})
            ok = ok && cudaEventCreateWithFlags(e, cudaEventDisableTiming | cudaEventBlockingSync) == cudaSuccess;
        ok = ok && cudaEventCreateWithFlags(&ev_q, cudaEventDisableTiming) == cudaSuccess;   // a stream wait: no host sync
        ok = ok && make_stream(&res_stream, wave_priority(high));
        for (cudaEvent_t& e : ev_res)   // waited on by the host (blocking), never by a stream of the other card
            ok = ok && cudaEventCreateWithFlags(&e, cudaEventDisableTiming | cudaEventBlockingSync) == cudaSuccess;
        const int64_t TK = T * 10;
        hbounds_cap = 2 * n_expert + 2 * (TK / R + 2) + 8;
        hx = (uint8_t*) h_alloc(mmq::q8_bytes(T, N));
        hbo = (float*) h_alloc((size_t) (T * N) * 4);
        hw = (float*) h_alloc((size_t) TK * 4);
        hrows = (int32_t*) h_alloc((size_t) TK * 4);
        hslot = (int32_t*) h_alloc((size_t) TK * 4);
        hbounds = (int32_t*) h_alloc((size_t) hbounds_cap * 4);
        hres = (uint8_t*) h_alloc((size_t) RES * blob);
        if (const char* fv = std::getenv("STRATA_DM_FRONTIER"); fv != nullptr && fv[0] == '1') {
            frontier = true;
            const char* fr = std::getenv("STRATA_DM_FRONTIER_FRAC");
            const double frac = fr ? std::atof(fr) : 0.6;
            pool_cap = (int32_t) std::max<int64_t>(1, (int64_t) ((double) TK * std::min(1.0, std::max(0.05, frac))));
            plan_cap = (size_t) (7 * TK + 64);   // worst case: T*K sources, 4 ints per (sub, token) group, 2 per copy
            hplan = (int32_t*) h_alloc(plan_cap * 4);
        }
        Dev g(dev);
        size_t f0 = 0, tot = 0;
        cudaMemGetInfo(&f0, &tot);
        const int prio = wave_priority(high);
        for (cudaStream_t* st : {&s, &c, &u, &cr}) ok = ok && make_stream(st, prio);
        for (cudaEvent_t* e : {&ev_xread, &ev_gatesread, &ev_meta, &ev_bo, &ev_plan})
            ok = ok && cudaEventCreateWithFlags(e, cudaEventDisableTiming | cudaEventBlockingSync) == cudaSuccess;
        for (cudaEvent_t& e : ev_resread)
            ok = ok && cudaEventCreateWithFlags(&e, cudaEventDisableTiming | cudaEventBlockingSync) == cudaSuccess;
        if (!stream_owner) RING = 0;   // lane 2 gathers from lane 1's ring
        used.assign((size_t) RING, nullptr);
        copied.assign((size_t) RING, nullptr);
        used1.assign((size_t) RING, nullptr);
        for (int i = 0; i < RING; ++i) {
            ok = ok && cudaEventCreateWithFlags(&used1[i], cudaEventDisableTiming) == cudaSuccess;
            ok = ok && cudaEventCreateWithFlags(&used[i], cudaEventDisableTiming) == cudaSuccess;
            ok = ok && cudaEventCreateWithFlags(&copied[i], cudaEventDisableTiming) == cudaSuccess;
        }
        auto d_alloc = [&](size_t n) -> void* {
            void* q = nullptr;
            if (cudaMalloc(&q, n) != cudaSuccess) { ok = false; cudaGetLastError(); return nullptr; }
            dev_bufs.push_back(q);
            return q;
        };
        xtok = d_alloc(mmq::q8_bytes(T, N));
        xq = d_alloc(mmq::q8_bytes(R, N));
        hq = d_alloc(mmq::q8_bytes(R, 640));
        gu = (float*) d_alloc((size_t) (R * 1280) * 4);
        h = (float*) d_alloc((size_t) (R * 640) * 4);
        if (frontier) {   // #41: a sub-product's rows and the pool instead of Dm
            fg = (float*) d_alloc((size_t) (R * N) * 4);
            fpool = (float*) d_alloc((size_t) pool_cap * N * 4);
            dplan = (int32_t*) d_alloc(plan_cap * 4);
        } else {
            dm = (float*) d_alloc((size_t) (TK * N) * 4);
        }
        bo = (float*) d_alloc((size_t) (T * N) * 4);
        w = (float*) d_alloc((size_t) TK * 4);
        rows = (int32_t*) d_alloc((size_t) TK * 4);
        slot = (int32_t*) d_alloc((size_t) TK * 4);
        ids = (int32_t*) d_alloc((size_t) R * 4);
        bounds = (int32_t*) d_alloc((size_t) hbounds_cap * 4);
        grp_gu = (uint8_t*) d_alloc(MMQ_GROUP * gub + MMQ_TAIL);
        grp_d = (uint8_t*) d_alloc(MMQ_GROUP * db + MMQ_TAIL);
        if (RING > 0) ring = (uint8_t*) d_alloc((size_t) RING * blob);
        if (ok) {
            ctx = std::make_unique<mmq::Context>();
            mmq::iota(ids, R, s);
            cudaStreamSynchronize(s);
            if (timeline::enabled()) {
                tl_s = timeline::lane((std::string("gpu1 compute (prefill split") + lane_tag + ")").c_str());
                tl_in = timeline::lane((std::string("gpu1 inputs (prefill split") + lane_tag + ")").c_str());   // #35 D2
                tl_c = timeline::lane((std::string("gpu1 copy engine (prefill split") + lane_tag + ")").c_str());
                clk_s.anchor(s);
                clk_c.anchor(c);
            }
            if (stream_owner) {
            stager = std::make_unique<Stager>();   // on this device: its events and threads
            stager->xsrc = xsrc;
            ok = stager->init(max_blob, 4);   // 8, 12 and 16 threads measured no faster
            }
        }
        size_t f1 = 0;
        cudaMemGetInfo(&f1, &tot);
        std::fprintf(stderr, "strata prefill: expert_split (whole layer) on device %d: %.0f MiB of VRAM (%.0f MiB left "
                             "free)\n", dev, (f0 - f1) / 1048576.0, f1 / 1048576.0);
        if (ok && timeline::enabled()) {
            tl_relay = timeline::lane((std::string("gpu0 relay (prefill split") + lane_tag + ")").c_str());
            tl_res = timeline::lane("gpu0 resident copies (prefill split)");
            Dev hm(home);                 // init runs on the 4070 from `Dev g` on; res_stream is the 5060's
            clk_res.anchor(res_stream);
        }
        if (!ok) err = "prefill: expert_split could not allocate its 4070 buffers";
        return ok;
    }
};

struct Prefill::Impl {
    const core::WeightTable* wt = nullptr;
    const core::ModelGeometry* g = nullptr;
    core::SessionState* ss = nullptr;
    core::ExpertSource* src = nullptr;
    const core::ExpertCache* cache = nullptr;
    const int32_t* host_res = nullptr;
    int64_t T = 0, T_max = 0;
    bool borrowed = false;
    cudaStream_t cs = nullptr, copy = nullptr;
    Gemm gemm;
    std::vector<void*> owned;
    // chunk buffers
    float *emb = nullptr, *R = nullptr, *xn = nullptr, *lo = nullptr, *gated = nullptr, *inj = nullptr;
    float* grs = nullptr;                    // F-1: the hyper-connection read's row scales (T x 4)
    uint16_t *xn16 = nullptr, *lo16 = nullptr;
    float* mixed = nullptr;
    uint16_t *mixed_bf = nullptr, *mixed_h = nullptr;
    float* bo = nullptr;
    // GDN
    float *qkv = nullptr, *z = nullptr, *ab = nullptr, *gate = nullptr, *beta = nullptr, *hbuf = nullptr, *y = nullptr;
    uint16_t* y_h = nullptr;
    // QSA
    float *Kc = nullptr, *Vc = nullptr, *Qf = nullptr, *q = nullptr, *idx_raw = nullptr, *q_idx = nullptr, *attn = nullptr;
    uint16_t* attn_h = nullptr;
    int32_t* steps_dev = nullptr;
    std::vector<int32_t> steps_host;
    int32_t* sel_ids = nullptr;
    float* sel_scores = nullptr;          // [sel_batch, max_blocks]
    int64_t sel_batch = 256, max_blocks = 0;
    float* attn_scratch = nullptr;
    int64_t attn_batch = 32, cap = 0;
    // MoE
    float *logits = nullptr, *w = nullptr, *GU = nullptr, *Dm = nullptr, *sgate = nullptr, *sup = nullptr,
          *shared = nullptr, *sg = nullptr;
    int32_t *ids = nullptr, *slot_dev = nullptr, *src_dev = nullptr;
    uint16_t *Xs = nullptr, *Hh = nullptr, *sh_h = nullptr;
    // step 2b (MMQ): the activations quantized per layer, H in FP32 and its group's quantized rows, the identity
    // row map, the group bounds, the group buffers of gathered experts
    void *Xq = nullptr, *Hq = nullptr;
    void* Xtok = nullptr;                    // #34: the chunk's activations, q8_1, one row per token
    float* H = nullptr;
    int64_t mmq_rows = 0;                    // #34: the rows GU / H / Xq / Hq hold (a sub-product's, mmq_rows_cap)
    int64_t moe_tokens = 0;                  // #35 D6: the tokens the one-card MoE buffers hold (moe_cap)
    int32_t *ids_identity = nullptr, *bounds_dev = nullptr;
    uint8_t *grp_gu = nullptr, *grp_d = nullptr;
    std::vector<int32_t> bounds_host;
    // #34: an MMQ group's products run as sub-products of at most mmq_rows rows: group positions [q0, q1), their
    // first row r0 in the layer's expert order, nr rows, the largest expert's rows, and a 0-based bounds block
    struct MmqSub { size_t q0, q1; int64_t r0, nr, maxr; int32_t off; };
    std::vector<MmqSub> mmq_subs;
    std::vector<size_t> mmq_sub_first;       // per group: its first sub-product (and one past the last group)
    std::unique_ptr<mmq::Context> mmq_ctx;
    std::vector<int32_t> ids_host, slot_host, src_host, cnt, off;
    // #42 (upstream fe609ce): the grouping tables in mapped pinned memory, [ids | slot | src] of T_max * K each, then
    // the MMQ bounds.  Kernels (copy_i32) read and write them in place: a cudaMemcpyAsync of them queued behind the
    // expert blobs the copy stream already held, and the GPU idled meanwhile.  STRATA_GROUP_COPY=1: the copies (A/B).
    int32_t* grp_host = nullptr;
    int32_t* grp_dev = nullptr;          // its device alias
    size_t grp_n = 0, grp_tk = 0;        // int32s allocated; T_max * K (the offset of slot, and of src past it)
    int32_t* ids_h() { return grp_host ? grp_host : ids_host.data(); }
    int32_t* slot_h() { return grp_host ? grp_host + grp_tk : slot_host.data(); }
    int32_t* src_h() { return grp_host ? grp_host + 2 * grp_tk : src_host.data(); }
    uint16_t* dq_gu[DQ] = {};
    uint16_t* dq_d[DQ] = {};
    uint8_t* stage_dev[RING_MAX] = {};
    int ring = STAGE;                        // the slots of this layout's ring (ring_slots)
    std::unique_ptr<Stager> stager;          // the unpinned experts' host copies (step 4)
    cudaEvent_t copied[RING_MAX] = {}, used[RING_MAX] = {};
    bool stage_live[RING_MAX] = {};
    const int32_t* peer_res = nullptr;       // set_peer_tier: experts owned by another device (#4)
    std::unique_ptr<SplitTier> split;        // #32 S4: the 4070's share of the routed experts
    std::shared_ptr<Prefill::WaveLink> wave; // #35 D7: set_wave (null: one lane reads every chunk)
    int wave_lane = 0;
    bool in_wave = false;                    // run_wave is running this lane (run() alone reads every chunk)
    cudaStream_t relay = nullptr;            // #32 S4: the 5060's side of the split's copies
    std::function<const void*(int32_t)> peer_ptr;
    int peer_dev = -1;
    // PLE
    float* ple_emb = nullptr;
    std::vector<float> ple_pageable[2];      // the fallback when no more RAM can be pinned
    float* ple_emb_host[2] = {};             // pinned, double-buffered: the next chunk's rows are read while this
    cudaEvent_t ple_copied[2] = {};          // one runs; the event marks that buffer's upload done
    std::vector<uint32_t> ple_rows[2];
    float* ple_norm = nullptr;
    uint8_t* region = nullptr;               // the attention/MoE scratch region (idle while the PLE block runs)
    uint64_t region_bytes = 0;
    PrefillStats* stats = nullptr;
    // KV streaming: one layer's whole K/V, staged from the host copy per layer and chunk (identity layout)
    strata::kernels::KvHostPools stage;
    int32_t* ident_table = nullptr;
    // layer split: the device, and the hand-off to the next stage (two pinned chunk buffers, used in turn)
    int device = -1;
    float* hand[2] = {};
    // C-4: the chunk's token ids on the device, for one batched embedding gather
    int32_t* tok_dev = nullptr;
    std::vector<int32_t> tok_host;
};

namespace {
// the staging pool of a streamed session: every page of one layer (same sequence in init and bytes_needed)
// STRATA_KV_STAGE_OWN (A/B only): the staging pool gets its own allocation instead of borrowed expert slots, so a
// streamed run lends the prompt path exactly the slots a resident one does (a lent expert runs on the CPU, which
// rounds differently: without this an A/B compares two expert placements as well as two KV placements)
bool stage_own() { static const bool v = std::getenv("STRATA_KV_STAGE_OWN") != nullptr; return v; }
void take_stage(Alloc& o_borrowed, const core::SessionState& ss, const strata::kernels::QsaShapes& s,
                strata::kernels::KvHostPools& st, bool& ok) {
    const core::QsaState& q0 = ss.qsa_states[ss.qsa_primary()];
    if (q0.kv_mode != 1) return;
    if (stage_own() && o_borrowed.count_only) return;
    Alloc own;
    own.owned = o_borrowed.owned;
    Alloc& o = stage_own() ? own : o_borrowed;
    const size_t rows = (size_t) q0.n_pages * s.n_head_kv * s.page_size;
    if (q0.kv_q4) {
        st.k_q4 = o.take<uint8_t>(rows * strata::kernels::kv_q4_bytes_per_head((int) s.head_dim), ok);
        st.v_q4 = o.take<uint8_t>(rows * strata::kernels::kv_q4_bytes_per_head((int) s.head_dim), ok);
    } else if (q0.kv_int8) {
        st.k_q = o.take<int8_t>(rows * s.head_dim, ok);
        st.v_q = o.take<int8_t>(rows * s.head_dim, ok);
        st.k_scale = o.take<uint16_t>(rows * (s.head_dim / 64), ok);
        st.v_scale = o.take<uint16_t>(rows * (s.head_dim / 64), ok);
    } else {
        st.k_pool = o.take<uint16_t>(rows * s.head_dim, ok);
        st.v_pool = o.take<uint16_t>(rows * s.head_dim, ok);
    }
}
strata::kernels::QsaAttnPools pools_of(const strata::kernels::KvHostPools& h, const int32_t* table) {
    strata::kernels::QsaAttnPools p;
    p.k_pool = h.k_pool; p.v_pool = h.v_pool; p.k_q = h.k_q; p.v_q = h.v_q; p.k_scale = h.k_scale; p.v_scale = h.v_scale;
    p.k_q4 = h.k_q4; p.v_q4 = h.v_q4;
    p.page_table = table;
    return p;
}
}  // namespace

// #35 D7: the wavefront's hand-off between the two lanes (Prefill::set_wave).  Per chunk: how many layers' attention
// halves are queued (each with an event on the lane's stream) and whether its on_chunk ran.  A lane that fails wakes
// the other.
struct Prefill::WaveLink {
    std::mutex mu;
    std::condition_variable cv;
    int64_t layers = 0;
    std::vector<int64_t> attn;   // per chunk
    std::vector<char> tail;      // per chunk: on_chunk done
    std::vector<cudaEvent_t> ev; // chunk * layers + l
    bool failed = false;
    SplitTier* split_owner = nullptr;   // lane 1's SplitTier once its plan is live (lane 2 consumes its stream)
    SplitTier* wait_split_owner() {
        std::unique_lock<std::mutex> lk(mu);
        cv.wait(lk, [&] { return failed || split_owner != nullptr; });
        return failed ? nullptr : split_owner;
    }
    SplitTier* owner_now() { std::lock_guard<std::mutex> lk(mu); return split_owner; }
    static void release(SplitTier* o) {   // a failed wave: the stream's waiters give up (the output is discarded)
        o->aborted.store(true, std::memory_order_release);
        for (auto* a : {&o->consumed, &o->consumed1, &o->issued}) SplitTier::publish(*a, SplitTier::kNever);
    }
    void set_split_owner(SplitTier* o) {
        {
            std::lock_guard<std::mutex> lk(mu);
            split_owner = o;
            if (failed) release(o);
        }
        cv.notify_all();
    }
    ~WaveLink() { for (cudaEvent_t e : ev) if (e) cudaEventDestroy(e); }
    bool wait_attn(int64_t c, int64_t l, cudaStream_t s) {
        {
            std::unique_lock<std::mutex> lk(mu);
            cv.wait(lk, [&] { return failed || attn[(size_t) c] > l; });
            if (failed) return false;
        }
        cudaStreamWaitEvent(s, ev[(size_t) (c * layers + l)], 0);   // the same card's stream: no cross-card wait
        return true;
    }
    void publish_attn(int64_t c, int64_t l, cudaStream_t s) {
        cudaEventRecord(ev[(size_t) (c * layers + l)], s);
        { std::lock_guard<std::mutex> lk(mu); attn[(size_t) c] = l + 1; }
        cv.notify_all();
    }
    bool wait_tail(int64_t c) {
        std::unique_lock<std::mutex> lk(mu);
        cv.wait(lk, [&] { return failed || tail[(size_t) c] != 0; });
        return !failed;
    }
    void publish_tail(int64_t c) {
        { std::lock_guard<std::mutex> lk(mu); tail[(size_t) c] = 1; }
        cv.notify_all();
    }
    void fail() {
        {
            std::lock_guard<std::mutex> lk(mu);
            failed = true;
            if (split_owner) release(split_owner);
        }
        cv.notify_all();
    }
};
std::shared_ptr<Prefill::WaveLink> Prefill::make_wave_link() { return std::make_shared<WaveLink>(); }
bool Prefill::wave_lane_splits(int64_t chunk) { return wave_lane_chunk(chunk) >= STREAM_ALL_MIN; }
bool Prefill::run_wave(Prefill& a, Prefill& b, WaveLink& link, const int64_t* tokens, int64_t n, int64_t pos0,
                       int64_t n_layers, std::string& err) {
    wave_reset(link, (n + a.chunk() - 1) / a.chunk(), n_layers);
    int dev = 0;
    cudaGetDevice(&dev);
    std::string err2;
    struct InWave {   // run() uses the link only inside run_wave
        Impl* a; Impl* b;
        ~InWave() { a->in_wave = b->in_wave = false; }
    } in_wave{a.impl_.get(), b.impl_.get()};
    a.impl_->in_wave = b.impl_->in_wave = true;
    auto lane2 = std::async(std::launch::async, [&, dev] {
        cudaSetDevice(dev);
        timeline::name_thread("prompt wave lane 2");
        return b.run(tokens, n, pos0, err2);
    });
    const bool ok1 = a.run(tokens, n, pos0, err);
    const bool ok2 = lane2.get();
    if (!ok2 && (ok1 || err.find("other wave lane") != std::string::npos)) err = err2;
    cudaStreamSynchronize(b.impl_->cs);
    return ok1 && ok2;
}
void Prefill::wave_reset(WaveLink& w, int64_t n_chunks, int64_t n_layers) {
    std::lock_guard<std::mutex> lk(w.mu);
    w.layers = n_layers;
    const size_t need = (size_t) (n_chunks * n_layers);
    while (w.ev.size() < need) {
        cudaEvent_t e = nullptr;
        cudaEventCreateWithFlags(&e, cudaEventDisableTiming);
        w.ev.push_back(e);
    }
    w.attn.assign((size_t) n_chunks, 0);
    w.tail.assign((size_t) n_chunks, 0);
    w.failed = false;
    w.split_owner = nullptr;
}
void Prefill::set_wave(std::shared_ptr<WaveLink> link, int lane) {
    impl_->wave = std::move(link);
    impl_->wave_lane = lane;
}

Prefill::Prefill() : impl_(new Impl) {}
Prefill::~Prefill() {
    const bool tr = std::getenv("STRATA_EXIT_TRACE") != nullptr;   // #45 exit-hang probe
    auto T_ = [&](const char* w) { if (tr) { std::fprintf(stderr, "exit trace: ~Prefill %s\n", w); std::fflush(stderr); } };
    if (!impl_) return;
    if (impl_->cs) cudaStreamSynchronize(impl_->cs);
    T_("cs synced");
    if (impl_->copy) cudaStreamSynchronize(impl_->copy);
    T_("copy synced");
    for (int i = 0; i < RING_MAX; ++i) {
        if (impl_->copied[i]) cudaEventDestroy(impl_->copied[i]);
        if (impl_->used[i]) cudaEventDestroy(impl_->used[i]);
    }
    for (int b = 0; b < 2; ++b) {
        if (impl_->hand[b]) cudaFreeHost(impl_->hand[b]);
        if (impl_->ple_copied[b]) cudaEventDestroy(impl_->ple_copied[b]);
        if (impl_->ple_emb_host[b] && impl_->ple_pageable[b].empty()) cudaFreeHost(impl_->ple_emb_host[b]);
    }
    if (impl_->copy) cudaStreamDestroy(impl_->copy);
    if (impl_->grp_host) cudaFreeHost(impl_->grp_host);
    impl_->split.reset();
    T_("split reset");
    if (impl_->relay) { cudaStreamSynchronize(impl_->relay); cudaStreamDestroy(impl_->relay); }
    T_("relay done");
    for (void* p : impl_->owned) cudaFree(p);
    T_("owned freed; members next");
}

namespace {
/// #32 S4: this MoE layer's routed experts and combine on the 4070 (SplitTier).  Everything is issued
/// asynchronously except the host experts, which the 4070's stager copies to pinned buffers as the loop reaches them.
/// On return the 5060's compute stream only has to wait for ev_done before it reads bo.
template <class Impl>   // Prefill::Impl (private: deduced, not named)
bool split_experts(Impl& m, int64_t l, int64_t T, size_t unit, int64_t chunk_i, const std::vector<int32_t>& order,
                   int mmq_gt, int mmq_dt, size_t gub, size_t db, std::string& err) {
    SplitTier& sp = *m.split;
    SplitTier& xs = *sp.xs;     // #35 D7: the expert stream (this lane's own, or lane 1's in a wave)
    const bool lane2 = m.wave && m.wave_lane == 1;
    auto& my_used = lane2 ? xs.used1 : xs.used;
    auto& my_consumed = lane2 ? xs.consumed1 : xs.consumed;
    const size_t us = unit * (size_t) (xs.layers + 1);
    if (l <= sp.last_l) sp.report();   // a new chunk
    sp.last_l = l;
    const int64_t TK = T * K;
    const auto& lay = strata::kernels::cpu::expert_layout();
    const auto& f = lay.fmt[(size_t) l];
    const int64_t NE = m.g->n_expert;
    // 1. the 5060: this layer's inputs to host once the compute stream has made them (and the 4070 has read the
    //    previous layer's)
    const size_t xbytes = mmq::q8_bytes(T, N);
    cudaEventRecord(sp.ev_q, m.cs);
    cudaStreamWaitEvent(m.relay, sp.ev_q, 0);
    {
        timeline::Span ws("split: sync prev reads", l);
        cudaEventSynchronize(sp.ev_xread);    // the 4070 has read the previous layer's activations and gates
        cudaEventSynchronize(sp.ev_gatesread);
    }
    sp.resolve(false);
    cudaEvent_t tr0 = sp.clk_relay.record(m.relay);
    cudaMemcpyAsync(sp.hx, m.Xtok, xbytes, cudaMemcpyDeviceToHost, m.relay);
    cudaEventRecord(sp.ev_x, m.relay);
    if (m.wave) m.wave->publish_attn(chunk_i, l, m.cs);   // #35 D7: handed off: the next chunk may take this card
    cudaEvent_t tr1 = sp.clk_relay.record(m.relay);
    sp.clk_relay.span(sp.tl_relay, "activations down", tr0, tr1, l);
    // the routed sum's gates: needed only at the end, so they cross while the experts run (#35 D1: the shared
    // output and its gate stay on the 5060, which finishes bo itself)
    cudaMemcpyAsync(sp.hw, m.w, (size_t) TK * 4, cudaMemcpyDeviceToHost, m.relay);
    cudaEventRecord(sp.ev_gates, m.relay);
    sp.clk_relay.span(sp.tl_relay, "gates down", tr1, sp.clk_relay.record(m.relay), l);
    // 2. the host tables (rows by expert, (t, k) -> row) and the sub-products: row ranges cut at R within a group
    struct Sub { int grp; int64_t r0, nr, maxr; int q0, n; int64_t boff; };
    std::vector<Sub> subs;
    {
        timeline::Span ws("split: sync prev meta", l);
        cudaEventSynchronize(sp.ev_meta);   // the previous layer's uploads have read hrows / hslot / hbounds
    }
    std::memcpy(sp.hrows, m.src_h(), (size_t) TK * 4);
    std::memcpy(sp.hslot, m.slot_h(), (size_t) TK * 4);
    int64_t nb = 0;
    const int ng = (int) ((order.size() + MMQ_GROUP - 1) / MMQ_GROUP);
    for (int gi = 0; gi < ng; ++gi) {
        const size_t a = (size_t) gi * MMQ_GROUP, e_end = std::min(order.size(), a + MMQ_GROUP);
        const int64_t g0 = m.off[(size_t) order[a]], g1 = m.off[(size_t) order[e_end - 1]] + m.cnt[(size_t) order[e_end - 1]];
        for (int64_t r0 = g0; r0 < g1; r0 += SplitTier::R) {
            const int64_t r1 = std::min(g1, r0 + SplitTier::R);
            Sub sb{gi, r0, r1 - r0, 0, -1, 0, nb};
            for (size_t i = a; i < e_end; ++i) {
                const int64_t e0 = std::max<int64_t>(m.off[(size_t) order[i]], r0);
                const int64_t e1 = std::min<int64_t>(m.off[(size_t) order[i]] + m.cnt[(size_t) order[i]], r1);
                if (e1 <= e0) continue;
                if (sb.q0 < 0) sb.q0 = (int) (i - a);
                if (nb + 2 > sp.hbounds_cap) { err = "prefill: expert_split bounds overflow"; return false; }
                sp.hbounds[nb++] = (int32_t) (e0 - r0);
                sb.maxr = std::max(sb.maxr, e1 - e0);
                ++sb.n;
            }
            sp.hbounds[nb++] = (int32_t) sb.nr;
            subs.push_back(sb);
        }
    }
    // this layer's entries in the chunk's stream plan: expert -> entry (-1: the 4070 owns it)
    std::vector<int64_t> entry_of((size_t) NE, -1);
    for (size_t k = xs.seq_start[us + (size_t) l]; k < xs.seq_start[us + (size_t) l + 1]; ++k) entry_of[(size_t) xs.seq[k].e] = (int64_t) k;
    for (int32_t e : order) {
        if (entry_of[(size_t) e] < 0) ++sp.n_own;
        else if (xs.seq[(size_t) entry_of[(size_t) e]].kind == 1) ++sp.n_res;
        else ++sp.n_host;
    }
    SplitTier::Dev g(sp.dev);
    // 3. the 4070: the inputs up
    sp.clk_s.mark(sp.tl_s, "wait inputs", sp.s, l);
    {
        timeline::Span ws("split wait activations", l);
        cudaEventSynchronize(sp.ev_x);   // the 5060's quantize and D2H (host-side: see ev_x's creation)
    }
    sp.clk_s.mark(sp.tl_s, "inputs up", sp.s, l);
    cudaEvent_t u0 = sp.clk_s.record(sp.s);
    cudaMemcpyAsync(sp.xtok, sp.hx, xbytes, cudaMemcpyHostToDevice, sp.s);
    cudaEvent_t u1 = sp.clk_s.record(sp.s);
    sp.clk_s.span(sp.tl_in, "activations up", u0, u1, l);
    cudaEventRecord(sp.ev_xread, sp.s);
    // the gates on their own stream (the previous layer's routed sum has read them: u waits for s's last record)
    cudaStreamWaitEvent(sp.u, sp.ev_bo, 0);   // same card: the previous routed sum has read w
    cudaEventSynchronize(sp.ev_gates);
    cudaMemcpyAsync(sp.w, sp.hw, (size_t) TK * 4, cudaMemcpyHostToDevice, sp.u);
    cudaEventRecord(sp.ev_gatesread, sp.u);
    cudaEvent_t t0 = sp.clk_s.record(sp.s);
    cudaMemcpyAsync(sp.rows, sp.hrows, (size_t) TK * 4, cudaMemcpyHostToDevice, sp.s);
    cudaMemcpyAsync(sp.slot, sp.hslot, (size_t) TK * 4, cudaMemcpyHostToDevice, sp.s);
    cudaMemcpyAsync(sp.bounds, sp.hbounds, (size_t) nb * 4, cudaMemcpyHostToDevice, sp.s);
    sp.clk_s.span(sp.tl_in, "tables up", t0, sp.clk_s.record(sp.s), l);
    cudaEventRecord(sp.ev_meta, sp.s);
    // 4. the experts: each group gathers from the 4070's slots and the ring; a ring entry is released (consumed)
    //    once its group's gather is recorded, and a group whose entries span more than the ring holds gathers what it
    //    has first
    auto wait_issued = [&](size_t k) -> bool {   // false: a failed wave (it publishes `issued` past everything)
        if (xs.issued.load(std::memory_order_acquire) > k) return !xs.aborted.load(std::memory_order_acquire);
        timeline::Span ws("split wait issuer", (int64_t) k, l);
        return SplitTier::wait_above(xs.issued, k, xs.aborted);
    };
    if (sp.frontier) {   // #41: the layer's commit plan, then the gates (every commit reads w)
        std::vector<std::pair<int64_t, int64_t>> ranges;
        ranges.reserve(subs.size());
        for (const Sub& sb : subs) ranges.push_back({sb.r0, sb.nr});
        if (!plan_frontier(sp.hslot, T, K, ranges, sp.pool_cap, sp.fplan, err)) { err = "prefill: " + err; return false; }
        if (sp.fplan.data.size() > sp.plan_cap) { err = "prefill: the frontier plan outgrew its buffer"; return false; }
        cudaEventSynchronize(sp.ev_plan);   // the previous layer's plan upload has read hplan
        std::memcpy(sp.hplan, sp.fplan.data.data(), sp.fplan.data.size() * 4);
        cudaMemcpyAsync(sp.dplan, sp.hplan, sp.fplan.data.size() * 4, cudaMemcpyHostToDevice, sp.s);
        cudaEventRecord(sp.ev_plan, sp.s);
        cudaStreamWaitEvent(sp.s, sp.ev_gatesread, 0);
    }
    int cur_grp = -1;
    size_t sub_i = 0;
    for (const Sub& sb : subs) {
        if (sb.grp != cur_grp) {   // gather the group's experts into the group buffers
            cur_grp = sb.grp;
            const size_t a = (size_t) sb.grp * MMQ_GROUP, e_end = std::min(order.size(), a + MMQ_GROUP);
            const uint8_t* blobs[MMQ_GROUP];
            int64_t ents[MMQ_GROUP];
            size_t p0 = a, pn = 0;   // pending gathers: group positions [p0, p0 + pn)
            int64_t k_first = -1;    // the first ring entry a pending gather reads
            // the pending gathers' last ring entry per copy stream (cr: the 5060's experts, c: the rest).  Each copy
            // stream runs its entries in order, so waiting for its last one covers every earlier one
            int64_t k_wait[2] = {-1, -1};
            auto flush = [&]() {
                if (pn == 0) return;
                for (int64_t& kw : k_wait)
                    if (kw >= 0) { cudaStreamWaitEvent(sp.s, xs.copied[kw % xs.RING], 0); kw = -1; }   // same card
                sp.clk_s.mark(sp.tl_s, "gather", sp.s, l, sb.grp);
                mmq::gather_native_group(blobs, (int) pn, f.up_off, f.down_off, gub / 2, db, sp.grp_gu + (p0 - a) * gub,
                                         gub, sp.grp_d + (p0 - a) * db, db, sp.s);
                int64_t k_last = -1;
                for (size_t i = 0; i < pn; ++i)
                    if (ents[i] >= 0) { cudaEventRecord(my_used[ents[i] % xs.RING], sp.s); k_last = ents[i]; }
                if (k_last >= 0) SplitTier::publish(my_consumed, (size_t) k_last + 1);
                p0 += pn;
                pn = 0;
                k_first = -1;
            };
            sp.clk_s.mark(sp.tl_s, "wait copies", sp.s, l, sb.grp);
            for (size_t i = a; i < e_end; ++i) {
                const int64_t k = entry_of[(size_t) order[i]];
                if (k >= 0 && k_first >= 0 && k - k_first + 2 > xs.RING) flush();
                if (k < 0) {
                    blobs[pn] = (const uint8_t*) m.peer_ptr(m.peer_res[(size_t) l * NE + order[i]]);
                } else {
                    if (k_first < 0) k_first = k;
                    // the entries before k that this layer does not route are released now
                    if (pn == 0 || k_first == k) {
                        const size_t c = my_consumed.load(std::memory_order_acquire);
                        if ((size_t) k > c) SplitTier::publish(my_consumed, (size_t) k);
                    }
                    if (!wait_issued((size_t) k)) { err = "prefill: the other wave lane failed"; return false; }
                    k_wait[xs.seq[(size_t) k].kind == 1 ? 0 : 1] = k;   // the copy stream it came on (the issuer's cs4)
                    blobs[pn] = xs.ring + (size_t) (k % xs.RING) * xs.blob;
                }
                ents[pn] = k;
                ++pn;
            }
            flush();
            const int gn = (int) (e_end - a);
            cudaMemsetAsync(sp.grp_gu + (size_t) gn * gub, 0, MMQ_TAIL, sp.s);
            cudaMemsetAsync(sp.grp_d + (size_t) gn * db, 0, MMQ_TAIL, sp.s);
        }
        mmq::ExpertRows a;
        a.xtok = sp.xtok; a.xtok_rows = T; a.rows = sp.rows + sb.r0; a.nr = sb.nr; a.max_rows = sb.maxr;
        a.n = sb.n; a.gu = sp.grp_gu + (size_t) sb.q0 * gub; a.gu_type = mmq_gt; a.gu_bytes = gub;
        a.down = sp.grp_d + (size_t) sb.q0 * db; a.down_type = mmq_dt; a.down_bytes = db;
        a.bounds = sp.bounds + sb.boff; a.ids = sp.ids; a.n_embd = N; a.n_ff = 640; a.interleaved = false;
        a.xq = sp.xq; a.gu_out = sp.gu; a.h = sp.h; a.hq = sp.hq; a.dst = sp.frontier ? sp.fg : sp.dm + sb.r0 * N;
        sp.clk_s.mark(sp.tl_s, "products", sp.s, l, sb.grp);
        mmq::expert_rows(*sp.ctx, a, sp.s);
        if (sp.frontier) {
            sp.clk_s.mark(sp.tl_s, "frontier", sp.s, l, sb.grp);
            frontier_run(sp.fplan, sub_i, sp.dplan, sp.fg, sp.fpool, sp.w, sp.bo, sp.s);
        }
        ++sub_i;
    }
    sp.clk_s.mark(sp.tl_s, "wait gates", sp.s, l);
    cudaStreamWaitEvent(sp.s, sp.ev_gatesread, 0);
    sp.clk_s.mark(sp.tl_s, "routed sum", sp.s, l);
    {
        const size_t end = xs.seq_start[us + (size_t) l + 1];
        if (my_consumed.load(std::memory_order_acquire) < end) SplitTier::publish(my_consumed, end);
    }
    // 5. combine here (the frontier has already summed into bo); the output down to host and up into the 5060's bo
    if (!sp.frontier) moe_routed_sum(sp.dm, sp.slot, sp.w, sp.bo, T, sp.s);
    cudaEventSynchronize(sp.ev_done);   // the 5060 has uploaded the previous layer's output from hbo
    sp.clk_s.mark(sp.tl_s, "output down", sp.s, l);
    cudaMemcpyAsync(sp.hbo, sp.bo, (size_t) (T * N) * 4, cudaMemcpyDeviceToHost, sp.s);
    cudaEventRecord(sp.ev_bo, sp.s);
    (void) cudaStreamQuery(sp.s);
    sp.clk_s.mark(sp.tl_s, nullptr, sp.s);
    const cudaError_t ce = cudaGetLastError();
    if (ce != cudaSuccess) { err = std::string("prefill: expert_split: ") + cudaGetErrorString(ce); return false; }
    return true;
}

/// #32 S4 / #35 D4: the layer's routed sum up into the 5060's bo, queued once the host has seen the 4070's D2H
/// complete (a 5060 stream waiting on the 4070's event held back the 5060's copy engine).  The caller has queued the
/// shared expert on the 5060 first, so it runs meanwhile; afterwards ev_done marks bo complete.
template <class Impl>
bool split_output_up(Impl& m, int64_t l, int64_t T, std::string& err) {
    SplitTier& sp = *m.split;
    {
        timeline::Span ws("split wait output", l);
        cudaEventSynchronize(sp.ev_bo);
    }
    cudaEvent_t o0 = sp.clk_relay.record(m.relay);
    cudaMemcpyAsync(m.bo, sp.hbo, (size_t) (T * N) * 4, cudaMemcpyHostToDevice, m.relay);
    cudaEventRecord(sp.ev_done, m.relay);
    sp.clk_relay.span(sp.tl_relay, "output up", o0, sp.clk_relay.record(m.relay), l);
    const cudaError_t ce = cudaGetLastError();
    if (ce != cudaSuccess) { err = std::string("prefill: expert_split output: ") + cudaGetErrorString(ce); return false; }
    return true;
}

constexpr int64_t GEMM_SCRATCH = 32ll << 20;        // FP16 elements for the largest dequantized dense weight
constexpr size_t GEMM_WS = 32u << 20;               // cuBLAS workspace

// THE ATTENTION HALF AND THE MoE HALF SHARE THEIR BUFFERS.  A layer runs its attention (GDN or QSA), writes it back
// into the residual, and only then its MoE, so the three sets of scratch are never live at once: one region the size
// of the largest holds them all.  That is ~260 KB of the ~680 KB a prompt token cost - which is what lets a chunk
// grow (every expert is streamed once per chunk, so a bigger chunk streams fewer bytes per token).  The sizes are
// counted with the same `take` sequence `init` uses; a mismatch makes `init` fail with "do not fit", never overlap.
uint64_t gdn_set_bytes(size_t T) {
    Alloc a; a.count_only = true; bool ok = true;
    a.take<float>(T * C, ok); a.take<float>(T * ZV, ok); a.take<float>(T * 2 * HV, ok); a.take<float>(T * HV, ok);
    a.take<float>(T * HV, ok); a.take<float>(T * C, ok); a.take<float>(T * ZV, ok); a.take<uint16_t>(T * ZV, ok);
    return a.used;
}
uint64_t qsa_set_bytes(size_t T, int64_t cap, int64_t max_blocks, int64_t sel_batch, int64_t attn_batch,
                       const strata::kernels::QsaShapes& s) {
    Alloc a; a.count_only = true; bool ok = true;
    a.take<float>(T * 512, ok); a.take<float>(T * 512, ok); a.take<float>(T * 12288, ok); a.take<float>(T * ZV, ok);
    a.take<float>(T * 128, ok); a.take<float>(T * 512, ok); a.take<float>(T * ZV, ok); a.take<uint16_t>(T * ZV, ok);
    a.take<int32_t>(T * (size_t) cap, ok);
    a.take<float>((size_t) sel_batch * (size_t) max_blocks, ok);
    a.take<float>((size_t) attn_batch * strata::kernels::qsa_decode_attn_scratch_floats(cap, s), ok);
    return a.used;
}
// Step 2b: which layers' experts go through MMQ (both weight types covered; the Strata Q2_0 pack always - its blob
// is converted to GGUF Q2_0 blocks on the gather), whether any layer keeps the FP16 path (IQ1_M), and the largest
// gate/up and down matrices a group buffer slot holds.  STRATA_PREFILL_MMQ=0: the FP16 path everywhere (the A/B).
// #34: the MMQ products run one group at a time, so their scratch (the group's q8 activations, gate/up, SwiGLU and its
// q8) needs only a group's rows, not the layer's T*K (0.9 GB at 8K).  A token routes to an expert at most once, so an
// expert has at most T rows; a group with more rows than the cap runs as sub-products that each fit (a row's MMQ result
// does not depend on the rows beside it, xeno_mmq_cross_arch (d)).  The 8K prompt's largest group has 10,426 rows.
inline int64_t mmq_rows_cap(int64_t T) { return std::min<int64_t>(T * K, 2 * T); }
struct MmqPlan {
    bool any = false, fallback = true;
    std::vector<char> layer;                   // per layer: MMQ
    size_t gu_max = 0, d_max = 0;
};
const MmqPlan& mmq_plan() {
    static const MmqPlan plan = [] {
        MmqPlan p;
        const auto& lay = strata::kernels::cpu::expert_layout();
        const char* env = std::getenv("STRATA_PREFILL_MMQ");
        const bool on = mmq::built() && (env == nullptr || std::atoi(env) != 0);
        const int64_t layers = lay.native ? (int64_t) lay.fmt.size() : lay.n_layers;
        p.layer.assign((size_t) std::max<int64_t>(layers, 0), 0);
        p.fallback = !on || layers <= 0;
        for (int64_t l = 0; on && l < layers; ++l) {
            const int gt = lay.native ? lay.fmt[(size_t) l].gu_type : 42, dt = lay.native ? lay.fmt[(size_t) l].d_type : 42;
            if (!mmq::supported(gt) || !mmq::supported(dt)) { p.fallback = true; continue; }
            p.layer[(size_t) l] = 1;
            p.any = true;
            p.gu_max = std::max(p.gu_max, mmq::matrix_bytes(gt, 1280, N));
            p.d_max = std::max(p.d_max, mmq::matrix_bytes(dt, N, 640));
        }
        return p;
    }();
    return plan;
}
// #35 D6 (review): the split layout applies only where every big chunk can run split, or such a chunk has no
// one-card buffers to fall back to
bool split_layout_usable() {
    const auto& lay = strata::kernels::cpu::expert_layout();
    if (!lay.native || !mmq_plan().any) return false;
    for (char c : mmq_plan().layer) if (!c) return false;
    return true;
}
uint64_t moe_set_bytes(size_t T, int64_t n_expert) {
    const MmqPlan& mp = mmq_plan();
    Alloc a; a.count_only = true; bool ok = true;
    const size_t Tm = moe_cap(T);
    a.take<float>(T * n_expert, ok); a.take<float>(T * K, ok); a.take<int32_t>(T * K, ok); a.take<int32_t>(T * K, ok);
    a.take<int32_t>(T * K, ok);
    if (mp.fallback) a.take<uint16_t>(Tm * K * N, ok);
    const size_t R = (size_t) mmq_rows_cap((int64_t) Tm);
    a.take<float>((mp.fallback ? Tm * K : R) * 1280, ok);   // the FP16 path (a fallback layer) uses the layer's rows
    if (mp.fallback) a.take<uint16_t>(Tm * K * 640, ok);
    a.take<float>(Tm * K * N, ok); a.take<float>(T * 640, ok);
    a.take<float>(T * 640, ok); a.take<uint16_t>(T * 640, ok); a.take<float>(T * N, ok); a.take<float>(T, ok);
    if (mp.any) {
        a.take<uint8_t>(mmq::q8_bytes((int64_t) T, N), ok);
        a.take<uint8_t>(mmq::q8_bytes((int64_t) R, N), ok);
        a.take<float>(R * 640, ok);
        a.take<uint8_t>(mmq::q8_bytes((int64_t) R, 640), ok);
    }
    return a.used;
}
}

bool Prefill::init(const core::WeightTable& wt, const core::ModelGeometry& g, core::SessionState& ss,
                   core::ExpertSource* src, const core::ExpertCache* cache, const int32_t* host_res, int64_t chunk,
                   void* stream, std::string& err, void* borrow, uint64_t borrow_bytes) {
    Impl& m = *impl_;
    m.wt = &wt; m.g = &g; m.ss = &ss; m.src = src; m.cache = cache; m.host_res = host_res;
    m.T = chunk; m.cs = (cudaStream_t) stream; m.stats = &stats_;
    if (g.n_embd != N || g.hc != HC || g.hc_lr != LR || g.n_expert < 1 || ss.k != K) {
        err = "prefill: geometry differs from the artifact's"; return false;
    }
    cudaGetDevice(&m.device);
    if (stage_le_ < 0) stage_le_ = g.n_layers;
    if (stage_lb_ < 0 || stage_lb_ >= stage_le_ || stage_le_ > g.n_layers || (stage_le_ < g.n_layers) != (next_ != nullptr)) {
        err = "prefill: the stage's layer range is wrong";
        return false;
    }
    for (int b = 0; next_ != nullptr && b < 2; ++b)
        if (!m.hand[b] && cudaHostAlloc((void**) &m.hand[b], (size_t) chunk * D * 4, cudaHostAllocPortable) != cudaSuccess) {
            err = "prefill: the layer split's hand-off buffers";
            return false;
        }
    if (m.tok_dev == nullptr) {
        if (cudaMalloc((void**) &m.tok_dev, (size_t) chunk * sizeof(int32_t)) != cudaSuccess) {
            err = "prefill: the token id buffer";
            return false;
        }
        m.owned.push_back(m.tok_dev);
        m.tok_host.resize((size_t) chunk);
    }
    if (cudaStreamCreateWithFlags(&m.copy, cudaStreamNonBlocking) != cudaSuccess) { err = "prefill: copy stream"; return false; }
    const size_t T = (size_t) chunk;
    m.T_max = chunk;
    m.borrowed = borrow != nullptr;
    bool ok = true;
    // one-time: events, the stager, the host buffers (for the largest chunk), the identity page table
    for (int i = 0; i < RING_MAX; ++i) {
        if (cudaEventCreateWithFlags(&m.copied[i], cudaEventDisableTiming) != cudaSuccess) ok = false;
        if (cudaEventCreateWithFlags(&m.used[i], cudaEventDisableTiming) != cudaSuccess) ok = false;
    }
    if (!m.stager) {
        m.stager = std::make_unique<Stager>();
        m.stager->xsrc = m.src;   // #11: NVMe-tier experts are read from the pack
        const int hw = (int) std::thread::hardware_concurrency();
        const char* stv = std::getenv("STRATA_STAGER_THREADS");   // D-5: the host copy threads of unpinned blobs
        if (!m.stager->init((size_t) MAXBLOB(), stv ? std::clamp(std::atoi(stv), 1, 32) : std::max(2, std::min(4, hw / 4))))
            ok = false;
    }
    m.steps_host.resize(T * strata::kernels::kStepCount);
    m.ids_host.resize(T * K); m.slot_host.resize(T * K); m.src_host.resize(T * K); m.cnt.resize(m.g->n_expert); m.off.resize(m.g->n_expert + 1);
    {   // #42: the mapped grouping tables (bounds: bounds_dev's size)
        const size_t need = 3 * T * K + (size_t) (4 * (m.g->n_expert + m.g->n_expert / MMQ_GROUP + 2));
        const char* gc = std::getenv("STRATA_GROUP_COPY");
        if (m.grp_n < need && !(gc && gc[0] == '1')) {
            if (m.grp_host) cudaFreeHost(m.grp_host);
            m.grp_host = m.grp_dev = nullptr;
            m.grp_n = m.grp_tk = 0;
            void *h = nullptr, *d = nullptr;
            if (cudaHostAlloc(&h, need * 4, cudaHostAllocMapped) == cudaSuccess &&
                cudaHostGetDevicePointer(&d, h, 0) == cudaSuccess) {
                m.grp_host = (int32_t*) h;
                m.grp_dev = (int32_t*) d;
                m.grp_n = need;
                m.grp_tk = T * K;
            } else {                              // the copies, as before
                if (h) cudaFreeHost(h);
                cudaGetLastError();
            }
        }
    }
    for (int b = 0; b < 2; ++b) {
        if (!m.ple_emb_host[b] &&
            cudaHostAlloc((void**) &m.ple_emb_host[b], (size_t) T * N * 4, cudaHostAllocDefault) != cudaSuccess) {
            cudaGetLastError();
            m.ple_pageable[b].resize(T * N);          // pageable: the upload is staged before it returns
            m.ple_emb_host[b] = m.ple_pageable[b].data();
        }
        if (!m.ple_copied[b] && cudaEventCreateWithFlags(&m.ple_copied[b], cudaEventDisableTiming) != cudaSuccess)
            ok = false;
        m.ple_rows[b].resize(T * strata::kernels::PLE_N_HEADS);
    }
    if (ss.qsa_states[ss.qsa_primary()].kv_mode == 1) {   // KV streaming: the staging pool's identity page table
        const int64_t pages = ss.qsa_states[ss.qsa_primary()].n_pages;
        std::vector<int32_t> ident((size_t) pages);
        for (int64_t i = 0; i < pages; ++i) ident[(size_t) i] = (int32_t) i;
        if (cudaMalloc((void**) &m.ident_table, ident.size() * 4) != cudaSuccess ||
            cudaMemcpy(m.ident_table, ident.data(), ident.size() * 4, cudaMemcpyHostToDevice) != cudaSuccess)
            ok = false;
        else
            m.owned.push_back(m.ident_table);
    }
    if (!ok) { err = "prefill: host buffers or events for a chunk of " + std::to_string(chunk) + " tokens"; return false; }
    Alloc o;
    o.base = (uint8_t*) borrow;
    o.cap = borrow_bytes;
    o.owned = &m.owned;
    const auto t_alloc = Clock::now();
    if (borrow == nullptr) {
        // its own buffers: one allocation carved like a borrowed region.  One cudaMalloc per buffer (hundreds of
        // stream-all ring slots) took ~2.5 s of TTFT under WDDM (#30, strata-claude-merge-decode ts_B)
        const uint64_t need = bytes_needed(g, ss, chunk);
        void* block = nullptr;
        if (cudaMalloc(&block, (size_t) need) != cudaSuccess) {
            err = "prefill: device buffers for a chunk of " + std::to_string(chunk) + " tokens do not fit";
            return false;
        }
        m.owned.push_back(block);
        o.base = (uint8_t*) block;
        o.cap = need;
    }
    {
        uint16_t* gs = o.take<uint16_t>((size_t) GEMM_SCRATCH, ok);
        void* ws = o.take<uint8_t>(GEMM_WS, ok);
        if (!ok) { err = "prefill: GEMM scratch does not fit"; return false; }
        if (!m.gemm.init_external(stream, gs, GEMM_SCRATCH, ws, GEMM_WS, err)) return false;
    }
    if (!carve(T, &o)) {
        err = "prefill: device buffers for a chunk of " + std::to_string(chunk) + " tokens do not fit";
        return false;
    }
    std::fprintf(stderr, "strata prefill: chunk %lld, %.0f MiB of device buffers (%s), %d ring slots, laid out in %.0f ms\n",
                 (long long) chunk, (double) o.used / 1048576.0, borrow ? "borrowed" : "own", m.ring, ms_since(t_alloc));
    return true;
}

// Every device buffer of a chunk of T tokens, from the Alloc `alloc` (after the GEMM scratch and workspace): `init`
// once, and `relayout` for a request's own chunk.  The order is `bytes_needed`'s.
bool Prefill::carve(size_t T, void* alloc) {
    Impl& m = *impl_;
    Alloc& o = *static_cast<Alloc*>(alloc);
    const core::ModelGeometry& g = *m.g;
    core::SessionState& ss = *m.ss;
    bool ok = true;
    m.emb = o.take<float>(T * N, ok); m.R = o.take<float>(T * D, ok);
    m.xn = gr_unfused() ? o.take<float>(T * D, ok) : nullptr;   // F-1: not needed (gr_mix_r reads R)
    m.grs = o.take<float>(T * HC, ok);
    m.xn16 = o.take<uint16_t>(T * D, ok); m.lo = o.take<float>(T * LR, ok); m.lo16 = o.take<uint16_t>(T * LR, ok);
    m.gated = o.take<float>(T * D, ok); m.inj = o.take<float>(T * HC, ok);
    m.mixed = o.take<float>(T * N, ok); m.mixed_bf = o.take<uint16_t>(T * N, ok);
    m.mixed_h = o.take<uint16_t>(T * N, ok); m.bo = o.take<float>(T * N, ok);
    m.steps_dev = o.take<int32_t>(T * strata::kernels::kStepCount, ok);
    strata::kernels::QsaShapes s = strata::kernels::qsa_real_shapes();
    s.n_head = g.n_head; s.n_head_kv = g.n_head_kv; s.head_dim = g.head_dim; s.idx_n_head = g.idx_q_heads;
    s.idx_dim = g.idx_key_dim;
    m.cap = strata::kernels::qsa_selection_width(strata::kernels::kTopkMaxCells, s);
    m.max_blocks = ss.qsa_states[ss.qsa_primary()].max_cells / s.idx_block + 2;
    {
        // one region for the attention half's and the MoE half's scratch (see gdn_set_bytes)
        const uint64_t region = std::max({gdn_set_bytes(T), qsa_set_bytes(T, m.cap, m.max_blocks, m.sel_batch,
                                                                           m.attn_batch, s), moe_set_bytes(T, m.g->n_expert)});
        uint8_t* base = o.take<uint8_t>((size_t) region, ok);
        m.region = base;
        m.region_bytes = region;
        Alloc a;
        a.base = base; a.cap = region; a.owned = &m.owned;
        m.qkv = a.take<float>(T * C, ok); m.z = a.take<float>(T * ZV, ok); m.ab = a.take<float>(T * 2 * HV, ok);
        m.gate = a.take<float>(T * HV, ok); m.beta = a.take<float>(T * HV, ok); m.hbuf = a.take<float>(T * C, ok);
        m.y = a.take<float>(T * ZV, ok); m.y_h = a.take<uint16_t>(T * ZV, ok);
        Alloc b;
        b.base = base; b.cap = region; b.owned = &m.owned;
        m.Kc = b.take<float>(T * 512, ok); m.Vc = b.take<float>(T * 512, ok); m.Qf = b.take<float>(T * 12288, ok);
        m.q = b.take<float>(T * ZV, ok); m.idx_raw = b.take<float>(T * 128, ok); m.q_idx = b.take<float>(T * 512, ok);
        m.attn = b.take<float>(T * ZV, ok); m.attn_h = b.take<uint16_t>(T * ZV, ok);
        m.sel_ids = b.take<int32_t>(T * (size_t) m.cap, ok);
        m.sel_scores = b.take<float>((size_t) m.sel_batch * (size_t) m.max_blocks, ok);
        m.attn_scratch = b.take<float>((size_t) m.attn_batch * strata::kernels::qsa_decode_attn_scratch_floats(m.cap, s), ok);
        Alloc c;
        c.base = base; c.cap = region; c.owned = &m.owned;
        m.logits = c.take<float>(T * m.g->n_expert, ok); m.w = c.take<float>(T * K, ok); m.ids = c.take<int32_t>(T * K, ok);
        m.slot_dev = c.take<int32_t>(T * K, ok); m.src_dev = c.take<int32_t>(T * K, ok);
        const MmqPlan& mp = mmq_plan();
        const size_t Tm = moe_cap(T);
        m.moe_tokens = (int64_t) Tm;
        m.Xs = mp.fallback ? c.take<uint16_t>(Tm * K * N, ok) : nullptr;
        m.mmq_rows = mmq_rows_cap((int64_t) Tm);
        m.GU = c.take<float>((mp.fallback ? Tm * K : (size_t) m.mmq_rows) * 1280, ok);
        m.Hh = mp.fallback ? c.take<uint16_t>(Tm * K * 640, ok) : nullptr;
        m.Dm = c.take<float>(Tm * K * N, ok);
        m.sgate = c.take<float>(T * 640, ok); m.sup = c.take<float>(T * 640, ok); m.sh_h = c.take<uint16_t>(T * 640, ok);
        m.shared = c.take<float>(T * N, ok); m.sg = c.take<float>(T, ok);
        if (mp.any) {
            m.Xtok = c.take<uint8_t>(mmq::q8_bytes((int64_t) T, N), ok);
            m.Xq = c.take<uint8_t>(mmq::q8_bytes(m.mmq_rows, N), ok);
            m.H = c.take<float>((size_t) m.mmq_rows * 640, ok);
            m.Hq = c.take<uint8_t>(mmq::q8_bytes(m.mmq_rows, 640), ok);
        }
        if (base == nullptr) ok = false;
    }
    for (int i = 0; i < DQ; ++i) { m.dq_gu[i] = o.take<uint16_t>(1280 * 2560, ok); m.dq_d[i] = o.take<uint16_t>(2560 * 640, ok); }
    if (mmq_plan().any) {
        const MmqPlan& mp = mmq_plan();
        m.ids_identity = o.take<int32_t>(T * K, ok);
        m.bounds_dev = o.take<int32_t>((size_t) (4 * (m.g->n_expert + m.g->n_expert / MMQ_GROUP + 2)), ok);
        m.grp_gu = o.take<uint8_t>(MMQ_GROUP * mp.gu_max + MMQ_TAIL, ok);
        m.grp_d = o.take<uint8_t>(MMQ_GROUP * mp.d_max + MMQ_TAIL, ok);
        // (written at every run's start, not here: when serving, these are live expert-cache slots until a request
        // lends them - a write now would corrupt a resident expert)
        if (!m.mmq_ctx) m.mmq_ctx = std::make_unique<mmq::Context>();
    }
    m.ring = ring_slots(moe_cap(T));
    for (int i = 0; i < m.ring; ++i) {
        m.stage_dev[i] = o.take<uint8_t>((size_t) MAXBLOB(), ok);
        m.stage_live[i] = false;                        // a new buffer: nothing of an earlier layout to wait for
    }
    m.ple_emb = o.take<float>(T * N, ok);
    m.ple_norm = o.take<float>((size_t) strata::kernels::NG_HC_DIM, ok);
    take_stage(o, ss, s, m.stage, ok);
    m.T = (int64_t) T;
    return ok;
}

void Prefill::set_peer_tier(const int32_t* res, std::function<const void*(int32_t)> slot_ptr, int device) {
    impl_->peer_res = res;
    impl_->peer_ptr = std::move(slot_ptr);
    impl_->peer_dev = device;
}

bool Prefill::relayout(int64_t chunk, void* borrow, uint64_t borrow_bytes, std::string& err) {
    Impl& m = *impl_;
    if (!m.borrowed || borrow == nullptr || chunk <= 0 || chunk > m.T_max) {
        err = "prefill: relayout needs borrowed buffers and a chunk of at most " + std::to_string(m.T_max);
        return false;
    }
    if (cudaStreamSynchronize(m.cs) != cudaSuccess || cudaStreamSynchronize(m.copy) != cudaSuccess) {
        err = "prefill: relayout: the stream failed";
        return false;
    }
    bool ok = true;
    Alloc o;
    o.base = (uint8_t*) borrow;
    o.cap = borrow_bytes;
    o.owned = &m.owned;
    uint16_t* gs = o.take<uint16_t>((size_t) GEMM_SCRATCH, ok);
    void* ws = o.take<uint8_t>(GEMM_WS, ok);
    if (ok) m.gemm.rebind(gs, GEMM_SCRATCH, ws, GEMM_WS);
    if (!ok || !carve((size_t) chunk, &o)) {
        err = "prefill: device buffers for a chunk of " + std::to_string(chunk) + " tokens do not fit";
        return false;
    }
    return true;
}

int64_t Prefill::chunk() const { return impl_->T; }

bool Prefill::draft_kv(core::MtpDrafter& mtp, const float* R_rows, const int32_t* next_tokens, int64_t n, int64_t cell0,
                       std::string& err) {
    Impl& m = *impl_;
    static const bool off = [] { const char* v = std::getenv("STRATA_MTP_BATCH"); return v != nullptr && v[0] == '0'; }();
    core::QsaState& st = mtp.kv_state_rw();
    if (off || n <= 0 || m.g == nullptr || m.region == nullptr || st.kv_mode != 0 || st.kv_hybrid ||
        mtp.device() != m.device)
        return false;
    const auto t0 = Clock::now();
    const core::ModelGeometry& g = *m.g;
    const int64_t Nn = g.n_embd, HCN = g.hc * g.n_embd, KV = g.n_head_kv * g.head_dim;
    constexpr int kQ8_0 = 8;   // GGML_TYPE_Q8_0
    const float* w_ne = mtp.tensor_f32("pre_fc_norm_embedding.weight");
    const void* w_fe = mtp.tensor_q8("fc_embedding.weight");
    const float* w_nh = mtp.tensor_f32("pre_fc_norm_hidden.weight");
    const void* w_fh = mtp.tensor_q8("fc_hidden.weight");
    const float* w_hn = mtp.tensor_f32("attn_hyper_connection.hc_norm.weight");
    const uint16_t* w_dn = mtp.tensor_bf16("attn_hyper_connection.input_mix_weight_down.weight");
    const uint16_t* w_up = mtp.tensor_bf16("attn_hyper_connection.input_mix_weight_up.weight");
    const void* w_k = mtp.tensor_q8("self_attn.k_proj.weight");
    const void* w_v = mtp.tensor_q8("self_attn.v_proj.weight");
    const float* w_kn = mtp.tensor_f32("self_attn.k_norm.weight");
    if (!w_ne || !w_fe || !w_nh || !w_fh || !w_hn || !w_dn || !w_up || !w_k || !w_v || !w_kn) return false;
    const core::NativeEmbed* nemb = core::native_embed();
    const core::WeightRef* wemb = nemb ? nullptr : m.wt->find("token_embd.weight");
    if (!nemb && (wemb == nullptr || wemb->codebook_iq4nl || wemb->ne0 != g.n_embd || wemb->group_elems <= 0 ||
                  (wemb->code_bits != 2 && wemb->code_bits != 4 && wemb->code_bits != 8)))
        return false;
    // the cells the drafter's window can still reach
    const int64_t r0 = std::max<int64_t>(0, mtp.first_needed() - cell0);
    if (r0 >= n) return true;
    // per row: emb/e2 (N), en16 (N half), hn/h2/Rm/gated (HCN), hn16/xn16 (HCN half), lo (LR) + lo16, grs, mixed (N) +
    // mixed_h, K and V (KV each), the token id
    // E-9: the drafter's Q8_0 matrices through Q8_1 x Q8_0 MMQ - its own pass's integer dot products (mmvq), so
    // its K/V stay close to what the drafter computes itself; STRATA_MTP_BATCH_F16=1: FP16 GEMMs (the A/B)
    static const bool f16_only = [] { const char* v = std::getenv("STRATA_MTP_BATCH_F16"); return v && v[0] == '1'; }();
    const bool q8 = !f16_only && mmq::built() && mmq::supported(kQ8_0);
    const uint64_t per_row = 4 * (2 * Nn + 4 * HCN + LR + HC + Nn + 2 * KV + 1) + 2 * (Nn + 2 * HCN + LR + Nn) + 64 +
                             (q8 ? (uint64_t) mmq::q8_bytes(g.hc, Nn) + 4 * g.hc : 0);
    const int64_t B = std::min<int64_t>(n - r0, (int64_t) (m.region_bytes / per_row) & ~(int64_t) 63);
    if (B < 64) return false;
    uint8_t* q = m.region;
    auto carve = [&](size_t bytes) { void* p = q; q += (bytes + 255) & ~(size_t) 255; return p; };
    float* emb = (float*) carve((size_t) B * Nn * 4);
    float* e2 = (float*) carve((size_t) B * Nn * 4);
    uint16_t* en16 = (uint16_t*) carve((size_t) B * Nn * 2);
    float* hn = (float*) carve((size_t) B * HCN * 4);
    float* h2 = (float*) carve((size_t) B * HCN * 4);
    float* Rm = (float*) carve((size_t) B * HCN * 4);
    float* gated = (float*) carve((size_t) B * HCN * 4);
    uint16_t* hn16 = (uint16_t*) carve((size_t) B * HCN * 2);
    uint16_t* xn16 = (uint16_t*) carve((size_t) B * HCN * 2);
    float* lo = (float*) carve((size_t) B * LR * 4);
    uint16_t* lo16 = (uint16_t*) carve((size_t) B * LR * 2);
    float* grs = (float*) carve((size_t) B * HC * 4);
    float* mixed = (float*) carve((size_t) B * Nn * 4);
    uint16_t* mixed_h = (uint16_t*) carve((size_t) B * Nn * 2);
    float* Kc = (float*) carve((size_t) B * KV * 4);
    float* Vc = (float*) carve((size_t) B * KV * 4);
    int32_t* tok = (int32_t*) carve((size_t) B * 4);
    void* xq = q8 ? carve(mmq::q8_bytes(B * g.hc, Nn)) : nullptr;
    int32_t* ident = q8 ? (int32_t*) carve((size_t) B * g.hc * 4) : nullptr;
    int32_t* bnd = q8 ? (int32_t*) carve(16) : nullptr;
    if ((uint64_t) (q - m.region) > m.region_bytes) return false;
    if (!mtp.idle(err)) return false;   // the drafter's own stream (its graph uploads) before this writes its K/V
    strata::kernels::QsaShapes s = strata::kernels::qsa_real_shapes();
    s.n_head = g.n_head; s.n_head_kv = g.n_head_kv; s.head_dim = g.head_dim; s.idx_n_head = g.idx_q_heads;
    s.idx_dim = g.idx_key_dim;
    std::vector<int32_t> tk((size_t) B);
    if (q8) {
        if (!m.mmq_ctx) m.mmq_ctx = std::make_unique<mmq::Context>();
        mmq::iota(ident, B * g.hc, m.cs);
    }
    // y[rows, n_out] = x[rows, k] . w^T for a Q8_0 matrix: MMQ from the FP32 rows (bounds slot 0: rows, 1: rows*hc)
    auto proj = [&](const float* x, const uint16_t* x16, const void* w, float* y, int64_t rows, int64_t n_out,
                    int64_t k, int slot) {
        if (!q8) { m.gemm.native(x16, kQ8_0, w, y, rows, n_out, k); return; }
        mmq::quantize(x, nullptr, xq, kQ8_0, k, k, rows, m.cs);
        mmq::Product p;
        p.w = w; p.type = kQ8_0; p.w_rows = n_out; p.w_cols = k; p.expert_bytes = mmq::matrix_bytes(kQ8_0, n_out, k);
        p.n = 1; p.xq = xq; p.bounds = bnd + 2 * slot; p.ids = ident; p.total_rows = rows; p.max_rows = rows;
        p.dst = y; p.ld_dst = n_out;
        m.mmq_ctx->run(p, m.cs);
    };
    for (int64_t b0 = r0; b0 < n; b0 += B) {
        const int64_t nb = std::min(B, n - b0), c0 = cell0 + b0;
        for (int64_t i = 0; i < nb; ++i) tk[(size_t) i] = next_tokens[b0 + i];
        const int32_t bh[4] = {0, (int32_t) nb, 0, (int32_t) (nb * g.hc)};
        if (q8 && cudaMemcpyAsync(bnd, bh, sizeof bh, cudaMemcpyHostToDevice, m.cs) != cudaSuccess) {
            err = "prefill: the draft bounds' upload failed";
            return false;
        }
        if (cudaMemcpyAsync(tok, tk.data(), (size_t) nb * 4, cudaMemcpyHostToDevice, m.cs) != cudaSuccess) {
            err = "prefill: the draft tokens' upload failed";
            return false;
        }
        // the input branches: the next token's embedding, and this cell's final residual rows
        if (nemb) {
            nemb->gather_dev(tok, nb, emb, m.cs);
        } else {
            const auto* codes = (const uint8_t*) wemb->data;
            const auto* scales = (const float*) (codes + wemb->codes_bytes);
            const auto* offsets = wemb->has_offset ? (const float*) (codes + wemb->codes_bytes + wemb->scales_bytes)
                                                   : nullptr;
            strata::kernels::embedding_gather_dev(codes, scales, offsets, tok, (int) nb, wemb->ne0, wemb->code_bits,
                                                  wemb->code_bias, wemb->group_elems,
                                                  (uint64_t) (wemb->ne0 / (8 / wemb->code_bits)),
                                                  (uint64_t) (wemb->ne0 / wemb->group_elems), emb, m.cs);
        }
        rms_rows(emb, w_ne, nb, Nn, Nn, EPS, m.cs);
        if (!q8) to_f16(emb, en16, nb * Nn, m.cs);
        proj(emb, en16, w_fe, e2, nb, Nn, Nn, 0);
        cudaMemcpyAsync(hn, R_rows + (size_t) b0 * HCN, (size_t) nb * HCN * 4, cudaMemcpyDeviceToDevice, m.cs);
        rms_rows(hn, w_nh, nb, HCN, HCN, EPS, m.cs);
        if (!q8) to_f16(hn, hn16, nb * HCN, m.cs);
        proj(hn, hn16, w_fh, h2, nb * g.hc, Nn, Nn, 1);   // every stream through fc_hidden
        strata::kernels::add_streams_broadcast(h2, e2, Rm, Nn, (int) g.hc, (int) nb, m.cs);
        // the attention hyper-connection's read (its mixed input only: this pass writes nothing back)
        gr_norm_rs(Rm, w_hn, EPS, grs, xn16, nb, m.cs);
        m.gemm.bf16(xn16, w_dn, lo, nb, LR, HCN);
        gr_silu(lo, lo16, nb, m.cs);
        m.gemm.bf16(lo16, w_up, gated, nb, HCN, LR);
        gr_mix_r(Rm, grs, w_hn, gated, mixed, nullptr, nb, m.cs, mixed_h);
        // K and V into the drafter's cache, as the prompt path's QSA layers append theirs
        proj(mixed, mixed_h, w_k, Kc, nb, KV, Nn, 0);
        proj(mixed, mixed_h, w_v, Vc, nb, KV, Nn, 0);
        rms_rows(Kc, w_kn, nb * g.n_head_kv, g.head_dim, g.head_dim, EPS, m.cs);
        rope(Kc, nb, g.n_head_kv, g.head_dim, KV, c0, strata::kernels::rope_scaling(), m.cs);
        if (st.kv_q4) {
            strata::kernels::fwht256_inplace_cuda(Kc, nb * g.n_head_kv, m.cs);
            strata::kernels::fwht256_inplace_cuda(Vc, nb * g.n_head_kv, m.cs);
            strata::kernels::kv_append_q4(st.k_q4, st.v_q4, st.page_table, c0, nb, Kc, Vc, s, m.cs, &st.host);
        } else {
            kv_append(Kc, Vc, nb, c0, st.page_table, s.page_size, st.kv_int8 ? nullptr : st.k_pool,
                      st.kv_int8 ? nullptr : st.v_pool, st.k_q, st.v_q, st.k_scale, st.v_scale, m.cs, &st.host);
        }
    }
    if (cudaStreamSynchronize(m.cs) != cudaSuccess) {
        err = std::string("prefill: the draft layer's K/V: ") + cudaGetErrorString(cudaGetLastError());
        return false;
    }
    mtp.ms_prefill += ms_since(t0);
    return true;
}
void Prefill::set_pinned_share(double share) { g_pinned_share = share; }
void Prefill::set_split_layout(bool on) { g_split_layout = on; }
double Prefill::pinned_share() { return g_pinned_share; }

uint64_t Prefill::bytes_needed(const core::ModelGeometry& g, const core::SessionState& ss, int64_t chunk) {
    // the same allocation sequence as `init`, counted
    const size_t T = (size_t) chunk;
    bool ok = true;
    Alloc o;
    o.count_only = true;
    o.take<uint16_t>((size_t) GEMM_SCRATCH, ok);
    o.take<uint8_t>(GEMM_WS, ok);
    auto f = [&](size_t n) { o.take<float>(n, ok); };
    f(T * N); f(T * D);
    if (gr_unfused()) f(T * D);   // F-1: xn only in the unfused arm (carve's order)
    f(T * HC);
    o.take<uint16_t>(T * D, ok); f(T * LR); o.take<uint16_t>(T * LR, ok);
    f(T * D); f(T * HC); f(T * N); o.take<uint16_t>(T * N, ok); o.take<uint16_t>(T * N, ok); f(T * N);
    o.take<int32_t>(T * strata::kernels::kStepCount, ok);
    strata::kernels::QsaShapes s = strata::kernels::qsa_real_shapes();
    s.n_head = g.n_head; s.n_head_kv = g.n_head_kv; s.head_dim = g.head_dim; s.idx_n_head = g.idx_q_heads;
    s.idx_dim = g.idx_key_dim;
    const int64_t cap = strata::kernels::qsa_selection_width(strata::kernels::kTopkMaxCells, s);
    const int64_t max_blocks = ss.qsa_states[ss.qsa_primary()].max_cells / s.idx_block + 2;
    const uint64_t trunk = o.used, gdn = gdn_set_bytes(T), qsa = qsa_set_bytes(T, cap, max_blocks, 256, 32, s),
                   moe = moe_set_bytes(T, g.n_expert);
    o.take<uint8_t>((size_t) std::max({gdn, qsa, moe}), ok);
    // #38 STRATA_PREFILL_BUFFERS=1: where the prompt path's borrowed bytes go (the shared set is the largest of three)
    static const bool breakdown = [] { const char* v = std::getenv("STRATA_PREFILL_BUFFERS"); return v && std::atoi(v); }();
    struct Report {
        bool on; size_t T; uint64_t trunk, gdn, qsa, moe; const Alloc& o;
        ~Report() {
            if (!on) return;
            auto mib = [](uint64_t b) { return b / 1048576.0; };
            std::fprintf(stderr, "strata prefill: buffers for %zu tokens: trunk %.0f MiB, shared set %.0f MiB (gdn %.0f, "
                                 "qsa %.0f, moe %.0f), the rest %.0f MiB\n", T, mib(trunk), mib(std::max({gdn, qsa, moe})),
                         mib(gdn), mib(qsa), mib(moe), mib(o.used - trunk - std::max({gdn, qsa, moe})));
        }
    } report{breakdown, T, trunk, gdn, qsa, moe, o};
    for (int i = 0; i < DQ; ++i) { o.take<uint16_t>(1280 * 2560, ok); o.take<uint16_t>(2560 * 640, ok); }
    if (mmq_plan().any) {
        const MmqPlan& mp = mmq_plan();
        o.take<int32_t>(T * K, ok);
        o.take<int32_t>((size_t) (4 * (g.n_expert + g.n_expert / MMQ_GROUP + 2)), ok);
        o.take<uint8_t>(MMQ_GROUP * mp.gu_max + MMQ_TAIL, ok);
        o.take<uint8_t>(MMQ_GROUP * mp.d_max + MMQ_TAIL, ok);
    }
    for (int i = 0; i < ring_slots(moe_cap(T)); ++i) o.take<uint8_t>((size_t) MAXBLOB(), ok);
    f(T * N);
    f((size_t) strata::kernels::NG_HC_DIM);
    strata::kernels::KvHostPools stage;
    take_stage(o, ss, s, stage, ok);
    return o.used + (8u << 20);   // alignment slack
}

namespace {

const core::WeightRef* need(const core::LayerView& v, const char* suffix, std::string& err) {
    const core::WeightRef* r = v.get(suffix);
    if (!r) err = v.name(suffix) + " is missing";
    return r;
}
bool native_proj(Gemm& gm, const core::WeightRef* w, const uint16_t* X, float* Y, int64_t T, const std::string& name,
                 std::string& err, int64_t ldy = 0) {
    if (!w->native_data) { err = "prefill: " + name + " has no native GGUF blocks (run with --native)"; return false; }
    gm.native(X, w->native_type, w->native_data, Y, T, w->ne1, w->ne0, ldy);
    return true;
}
bool bf16_proj(Gemm& gm, const core::WeightRef* w, const uint16_t* X, float* Y, int64_t T, const std::string& name,
               std::string& err, int64_t ldy = 0) {
    if (w->kind != core::WeightKind::Bf16InF32 || !w->data) { err = "prefill: " + name + " is not a resident BF16 tensor"; return false; }
    gm.bf16(X, (const uint16_t*) w->data, Y, T, w->ne1 > 0 ? w->ne1 : 1, w->ne0, ldy);
    return true;
}

}  // namespace

namespace {
// STRATA_PREFILL_TIMING=1: the prompt path's GPU time by phase.  Events are recorded on the compute stream in order;
// the time between two consecutive marks is charged to the phase of the first, so a gap where the GPU waits (for the
// host's expert grouping, or for an expert's copy) lands on the phase that was waiting.  Events are reused: the marks
// are folded at every MoE layer's host sync, after which all of them have completed.
enum PfPhase { kPfStart, kPfHc, kPfGdn, kPfQsa, kPfQsaIdx, kPfQsaSel, kPfQsaAttn, kPfRouter, kPfHostGroup, kPfGather,
               kPfWaitCopy, kPfDequant, kPfGemmGU, kPfGemmD, kPfCombine, kPfPle, kPfKvStage, kPfGdnConv, kPfGdnRec, kPfGdnOut,
               kPfWaitHost, kPfCount };
const char* const kPfNames[kPfCount] = {"embed+steps", "hc read", "gdn", "qsa proj", "qsa indexer", "qsa select",
                                        "qsa attn", "router+shared", "host grouping", "gather", "wait copy", "dequant",
                                        "gemm gate/up", "gemm down", "combine", "ple", "kv stage", "gdn conv+gates",
                                        "gdn recurrence", "gdn out proj", "wait host"};
// "wait host" (#31): marked after the last kernel an expert enqueues, so the time until the next expert's work reaches
// the stream is the GPU idle while the launching thread is elsewhere (a stager or issuer wait, a copy issue); it used
// to land on "dequant", the phase before it.
struct PfTimer {
    bool on = std::getenv("STRATA_PREFILL_TIMING") != nullptr;
    std::vector<cudaEvent_t> ev;
    std::vector<int> ph;
    size_t used = 0;
    double ms[kPfCount] = {};
    timeline::GpuClock* clk = nullptr;   // #33: STRATA_TIMELINE - the same marks on the compute lane
    int lane = 0;
    int64_t layer = -1;
    void mark(int phase, cudaStream_t s) {
        if (clk) clk->mark(lane, kPfNames[phase], s, layer);
        if (!on) return;
        if (used == ev.size()) {
            cudaEvent_t e = nullptr;
            cudaEventCreate(&e);
            ev.push_back(e);
            ph.push_back(0);
        }
        ph[used] = phase;
        cudaEventRecord(ev[used], s);
        ++used;
    }
    // every recorded mark has completed (the stream was synchronized): charge the gaps, keep the last mark
    void fold() {
        if (clk) clk->resolve(false);
        if (!on || used < 2) return;
        for (size_t i = 0; i + 1 < used; ++i) {
            float t = 0.0f;
            if (cudaEventElapsedTime(&t, ev[i], ev[i + 1]) == cudaSuccess) ms[ph[i]] += t;
        }
        std::swap(ev[0], ev[used - 1]);
        std::swap(ph[0], ph[used - 1]);
        used = 1;
    }
    ~PfTimer() {
        for (cudaEvent_t e : ev) cudaEventDestroy(e);
    }
};
}  // namespace

bool Prefill::run(const int64_t* tokens, int64_t n, int64_t pos0, std::string& err) {
    err.clear();
    Impl& m = *impl_;
    const core::OnDevice on_device(m.device);
    // #35: a lane run on its own (serve: a part the lent wave layout holds but whose lane chunk does not run split)
    // reads every chunk itself.  With the link attached it read every other chunk and planned the 4070's stream for
    // a lane 2 that never ran: the issuer waited for it forever (serve_wave_mixed: 6,000 then 3,500 tokens)
    struct WaveDetach {
        Impl& m;
        std::shared_ptr<WaveLink> held;
        explicit WaveDetach(Impl& i) : m(i) { if (m.wave && !m.in_wave) held = std::move(m.wave); }
        ~WaveDetach() { if (held) m.wave = std::move(held); }
    } wave_detach{m};
    const core::ModelGeometry& g = *m.g;
    core::SessionState& ss = *m.ss;
    const auto t_start = Clock::now();
    const int64_t LB = stage_lb_, LE = stage_le_;
    // the next stage reads chunk c on a thread while this one reads chunk c + 1 (declared first: an early return
    // waits for it before anything it reads goes away)
    std::string next_err;
    std::future<bool> next_run;
    int hand_buf = 0;
    double host_sync_ms = 0, host_chunk_ms = 0, host_setup_ms = 0;   // STRATA_PREFILL_TIMING: the host's share
    strata::kernels::QsaShapes s = strata::kernels::qsa_real_shapes();
    s.n_head = g.n_head; s.n_head_kv = g.n_head_kv; s.head_dim = g.head_dim; s.idx_n_head = g.idx_q_heads;
    s.idx_dim = g.idx_key_dim;
    const uint64_t gdn_floats = (uint64_t) g.ssm_state_size * g.ssm_v_heads * g.ssm_state_size +
                                (uint64_t) g.ssm_conv_channels * (g.ssm_d_conv - 1);
    int32_t prev[2] = {ss.ple_prev[0], ss.ple_prev[1]};
    PfTimer pt;
    const cudaStream_t cs = (cudaStream_t) m.cs;
    // #33: STRATA_TIMELINE - the compute stream's phases (the marks above) and every expert copy on the copy stream,
    // each on its own clock (the copies are recorded by whichever thread issues them)
    timeline::GpuClock clk_compute, clk_copy;
    const bool lane2 = m.wave && m.wave_lane == 1;   // #35 D7: the wave's second lane on its own timeline lanes
    const int tl_compute = timeline::lane(lane2 ? "gpu0 compute (prefill, lane 2)" : "gpu0 compute (prefill)"),
              tl_copy = timeline::lane(lane2 ? "gpu0 copy engine (prefill, lane 2)" : "gpu0 copy engine (prefill)");
    if (timeline::enabled()) {
        clk_compute.anchor(cs);
        clk_copy.anchor((cudaStream_t) m.copy);
        pt.clk = &clk_compute;
        pt.lane = tl_compute;
    }
    timeline::Span run_span("prefill run", n, pos0);
    // the MMQ row table lives in the borrowed cache slots, which the refill after a prompt overwrites with experts:
    // write it again for every prompt (a layout is reused as long as the chunk and the slots are the same)
    if (m.ids_identity != nullptr) mmq::iota(m.ids_identity, m.T * K, m.cs);
    // The PLE rows of a chunk are read from the model file on the host (an SSD read per missed row): the chunk
    // after this one is read on a thread while the GPU runs this one, into the other of two buffers.  The rows
    // depend only on the tokens (the two before a position name its n-grams), so this is the same data.
    const bool ple_on = ss.ple.ready() && LB <= 1 && 1 < LE;
    // the PLE block batched over the chunk: the pinned postops and a BF16 or GGUF-native key (else token by token);
    // STRATA_PLE_BATCH=0 keeps the per-token block (the A/B)
    static const bool ple_batch_env = [] {
        const char* v = std::getenv("STRATA_PLE_BATCH");
        return v == nullptr || std::atoi(v) != 0;
    }();
    const bool ple_batch = ple_on && ple_batch_env && strata::kernels::ple_native_postops_enabled() &&
                           (ss.ple.w.key_bf16 != nullptr || ss.ple.w.key_native_data != nullptr) &&
                           m.region_bytes / ((uint64_t) (3 * strata::kernels::NG_HC_DIM + N + 4) * 4 + (uint64_t) N * 2 + 4096) >= 64;
    const int32_t prev0[2] = {prev[0], prev[1]};
    auto ple_gather = [&m, &ss, tokens, n, prev0](int64_t c0, int buf, std::string& e) -> bool {
        const int64_t T = std::min(m.T, n - c0);
        auto at = [&](int64_t i) { return i < 2 ? prev0[i] : (int32_t) tokens[i - 2]; };   // prev0, then the tokens
        int32_t pv[2] = {at(c0), at(c0 + 1)};
        for (int64_t t = 0; t < T; ++t) {
            const int32_t tok = (int32_t) tokens[c0 + t];
            strata::kernels::ngram_rows(&tok, pv, 1, ss.ple.consts,
                                        m.ple_rows[buf].data() + t * strata::kernels::PLE_N_HEADS);
            pv[0] = pv[1];
            pv[1] = tok;
        }
        return ss.ple.table->gather_batch(m.ple_rows[buf].data(), (size_t) T, m.ple_emb_host[buf], e);
    };
    std::string ple_next_err;
    std::future<bool> ple_next;             // declared after everything it reads: an early return waits for it
    int ple_buf = 0;

    // #35 D7: a wave lane reads every other chunk; a failure wakes the other lane
    WaveLink* const wave = m.wave.get();
    const int64_t stride = wave ? 2 * m.T : m.T;
    struct WaveFail { WaveLink* w; bool done = false; ~WaveFail() { if (w && !done) w->fail(); } } wave_fail{wave};
    // #35 D7: a wave lane that is done (or fails) no longer holds the shared stream back; the second to finish joins it
    struct WaveStreamDone {
        Impl& m;
        ~WaveStreamDone() {
            if (!m.wave) return;
            // lane 1 owns its plan; lane 2 asks the link for this run's owner (never a pointer from an earlier run)
            SplitTier* x = m.wave_lane == 0 ? m.split.get() : m.wave->owner_now();
            if (m.split) m.split->xs = m.split.get();
            if (!x || !x->plan_live || (m.wave_lane == 1 && x->lanes_expected < 2)) return;
            SplitTier::publish(m.wave_lane == 1 ? x->consumed1 : x->consumed, SplitTier::kNever);
            if (x->lanes_done.fetch_add(1) + 1 == x->lanes_expected) x->join_issuer();
        }
    } wave_stream_done{m};
    if (m.split) m.split->xs = m.split.get();   // #35 D7: a run starts on its own tier (lane 2 attaches below)
    for (int64_t c0 = wave ? m.wave_lane * m.T : 0; c0 < n; c0 += stride) {
        if (should_stop && should_stop()) { err = "cancelled"; return false; }
        const int64_t chunk_i = c0 / m.T;   // the chunk's index in the prompt (the wave hand-off's key)
        if (std::getenv("STRATA_TRACE")) { std::fprintf(stderr, "strata trace: prompt chunk %lld of %lld\n", (long long) c0, (long long) n); std::fflush(stderr); }
        const int64_t T = std::min(m.T, n - c0), p0 = pos0 + c0;
        ++stats_.chunks;
        clk_copy.resolve(false);   // the previous chunk's issuer has joined
        timeline::Span chunk_span("prompt chunk", c0, T);
        pt.layer = -1;
        pt.mark(kPfStart, cs);
        const auto tsetup = Clock::now();
        const double tl_embed = timeline::enabled() ? timeline::now_us() : 0;   // (xeno #33)
        // ---- embeddings, broadcast to the four streams - or, in a later stage of a layer split, the rows the
        // previous stage handed on
        if (hand_in_ != nullptr) {
            if (cudaMemcpyAsync(m.R, hand_in_ + (size_t) c0 * D, (size_t) T * D * 4, cudaMemcpyHostToDevice, m.cs) !=
                cudaSuccess) {
                err = "prefill: the layer split's hand-off upload failed";
                return false;
            }
        }
        // C-4: the whole chunk's rows in one gather (the same per-element arithmetic as the per-token path, so the
        // same bits); a chunk with picture rows, or a token outside the table, takes the per-token path
        bool batched = hand_in_ == nullptr && m.tok_dev != nullptr;
        const core::NativeEmbed* nemb = core::native_embed();
        const core::WeightRef* wemb = nemb ? nullptr : m.wt->find("token_embd.weight");
        if (batched && nemb == nullptr &&
            (wemb == nullptr || wemb->codebook_iq4nl || wemb->ne0 != g.n_embd || wemb->group_elems <= 0 ||
             (wemb->code_bits != 2 && wemb->code_bits != 4 && wemb->code_bits != 8)))
            batched = false;
        for (int64_t t = 0; batched && t < T; ++t) {
            const int64_t tok = tokens[c0 + t];
            if ((embd_rows && embd_rows[p0 + t]) || tok < 0 || (wemb && tok >= wemb->ne1)) batched = false;
            else m.tok_host[(size_t) t] = (int32_t) tok;
        }
        if (batched) {
            if (cudaMemcpyAsync(m.tok_dev, m.tok_host.data(), (size_t) T * sizeof(int32_t), cudaMemcpyHostToDevice,
                                m.cs) != cudaSuccess) {
                err = "prefill: the token id upload failed";
                return false;
            }
            if (nemb) {
                nemb->gather_dev(m.tok_dev, T, m.emb, m.cs);
            } else {
                const auto* codes = (const uint8_t*) wemb->data;
                const auto* scales = (const float*) (codes + wemb->codes_bytes);
                const auto* offsets = wemb->has_offset ? (const float*) (codes + wemb->codes_bytes + wemb->scales_bytes)
                                                       : nullptr;
                strata::kernels::embedding_gather_dev(codes, scales, offsets, m.tok_dev, (int) T, wemb->ne0,
                                                      wemb->code_bits, wemb->code_bias, wemb->group_elems,
                                                      (uint64_t) (wemb->ne0 / (8 / wemb->code_bits)),
                                                      (uint64_t) (wemb->ne0 / wemb->group_elems), m.emb, m.cs);
            }
        }
        for (int64_t t = 0; hand_in_ == nullptr && !batched && t < T; ++t) {
            const float* row = embd_rows ? embd_rows[p0 + t] : nullptr;
            if (row) {
                if (cudaMemcpyAsync(m.emb + t * N, row, (size_t) N * 4, cudaMemcpyHostToDevice, m.cs) != cudaSuccess) {
                    err = "prefill: the image embedding upload failed";
                    return false;
                }
            } else if (!core::embed_row(*m.wt, g, tokens[c0 + t], m.emb + t * N, m.cs, err)) {
                return false;
            }
        }
        if (hand_in_ == nullptr) gr_broadcast(m.emb, m.R, T, m.cs);   // (upstream) a later layer-split stage has R
        if (tl_embed > 0) timeline::complete("embed (host)", tl_embed, timeline::now_us(), c0, T);
        // ---- the PLE rows of the whole chunk, one batched SSD request, read on a thread (see ple_gather).  Only the
        // PLE block at layer 1 reads them, so the chunk waits for them there (ple_take below).  #31: the first chunk's
        // rows used to be read in line here, 672 ms of an 8K prompt with the GPU idle; now they are read while the
        // embeddings and layer 0 run.  Later chunks' rows are read a chunk ahead, as before.
        auto ple_read = [&ple_gather, &ple_next_err](int64_t c, int b) {
            return std::async(std::launch::async, [&ple_gather, &ple_next_err, c, b] {
                timeline::name_thread("ple read-ahead");
                timeline::Span sp("ple read-ahead", c);
                return ple_gather(c, b, ple_next_err);
            });
        };
        if (ple_on && !ple_next.valid()) ple_next = ple_read(c0, ple_buf);
        // STRATA_PREFILL_PLE_AHEAD=0 takes the rows here, before the layers (the A/B: the old in-line read)
        static const bool ple_ahead_on = [] {
            const char* v = std::getenv("STRATA_PREFILL_PLE_AHEAD");
            return v == nullptr || std::atoi(v) != 0;
        }();
        auto ple_take = [&]() -> bool {
            const auto tp = Clock::now();
            timeline::Span ple_span("ple rows (host)", c0);
            if (!ple_next.get()) {
                err = ple_next_err;
                return false;
            }
            cudaMemcpyAsync(m.ple_emb, m.ple_emb_host[ple_buf], (size_t) T * N * 4, cudaMemcpyHostToDevice, m.cs);
            cudaEventRecord(m.ple_copied[ple_buf], m.cs);
            if (c0 + stride < n) {
                cudaEventSynchronize(m.ple_copied[ple_buf ^ 1]);   // the other buffer's upload (a chunk ago) is done
                ple_next = ple_read(c0 + stride, ple_buf ^ 1);
            }
            ple_buf ^= 1;
            stats_.ms_ple += ms_since(tp);   // since 4935d9b: the wait for the rows, not the whole read
            return true;
        };
        if (ple_on && !ple_ahead_on && !ple_take()) return false;   // A/B: the rows before the layers
        if (wave && c0 > 0) {
            prev[0] = c0 >= 2 ? (int32_t) tokens[c0 - 2] : prev0[1];
            prev[1] = (int32_t) tokens[c0 - 1];
        }
        for (int64_t t = 0; t < T; ++t) { prev[0] = prev[1]; prev[1] = (int32_t) tokens[c0 + t]; }
        // ---- the QSA step records of every position in the chunk
        {
            timeline::Span steps_span("qsa steps (host)", c0, T);
            for (int64_t t = 0; t < T; ++t)
                strata::kernels::qsa_step_fill(m.steps_host.data() + t * strata::kernels::kStepCount, p0 + t, s);
            cudaMemcpyAsync(m.steps_dev, m.steps_host.data(), (size_t) T * strata::kernels::kStepCount * 4,
                            cudaMemcpyHostToDevice, m.cs);
        }

        int64_t qsa_index = 0, gdn_index = 0;
        for (int64_t l = 0; l < LB; ++l) (core::is_qsa_layer(g, l) ? qsa_index : gdn_index) += 1;
        // step 3: this chunk's stream - every non-resident expert of every layer, layer by layer in id order (entry
        // k lands in ring slot k % ring); a copy is issued once the entry `ring` before it is consumed (its slot's
        // `used` event recorded), so the copy stream never waits on an event that is not queued yet
        const strata::kernels::cpu::ExpertLayout& lay0 = strata::kernels::cpu::expert_layout();
        // xeno #56: upstream 0.1.30 streams every expert from 1024-token chunks; the split layout (#32/#35) keeps
        // 2048 (STREAM_ALL_MIN), which its one-card MoE buffers and wave lanes are sized for (not measured lower)
        const bool stream_all = m.ring > STAGE && T >= (g_split_layout ? STREAM_ALL_MIN : stream_all_min()) &&
                                m.src != nullptr;
        // #32 S4: expert_split for this chunk (MMQ layers of a native pack whose peer tier is set)
        static const bool split_env = [] { const char* v = std::getenv("STRATA_PREFILL_EXPERT_SPLIT"); return v && std::atoi(v) != 0; }();
        bool split_on = split_env && m.peer_res != nullptr && m.peer_dev >= 0 && m.src != nullptr && split_layout_usable();
        if (split_on && (!m.split || m.split->T < m.T)) {
            m.split.reset();
            std::string se;
            size_t gub = 0, db = 0;
            for (int64_t l = 0; l < g.n_layers; ++l) {   // every layer is on MMQ (split_layout_usable)
                gub = std::max(gub, mmq::matrix_bytes(lay0.fmt[(size_t) l].gu_type, 1280, N));
                db = std::max(db, mmq::matrix_bytes(lay0.fmt[(size_t) l].d_type, N, 640));
            }
            auto sp = std::make_unique<SplitTier>();
            if (!m.relay && !make_stream(&m.relay, wave_priority(m.wave && m.wave_lane == 0))) m.relay = nullptr;
            sp->high = m.wave && m.wave_lane == 0;
            if (m.wave && m.wave_lane == 1) { sp->lane_tag = ", lane 2"; sp->stream_owner = false; }
            if (m.wave && m.wave_lane == 0) sp->RING = 512;   // #35 D7: the stream feeds both lanes' layer
            if (m.relay && sp->init(m.peer_dev, m.T, m.g->n_expert, gub, db, (size_t) MAXBLOB(), m.src, se)) {
                if (timeline::enabled()) sp->clk_relay.anchor(m.relay);
                m.split = std::move(sp);
            }
            else std::fprintf(stderr, "strata prefill: expert_split off: %s\n", se.empty() ? "no relay stream" : se.c_str());
        }
        const bool split_base = split_on && m.split != nullptr;   // #35 D7: the wave's lane 1 plans with full chunks
        split_on = split_base && T >= STREAM_ALL_MIN;   // every candidate expert streams: big chunks only
        if (!split_on && (int64_t) T > m.moe_tokens) {
            err = "prefill: the split layout holds one-card MoE buffers for " + std::to_string(m.moe_tokens) +
                  " tokens, and this " + std::to_string(T) + "-token chunk cannot run split (see the expert_split "
                  "lines above); restart without STRATA_PREFILL_EXPERT_SPLIT";
            return false;
        }
        struct StreamEntry { int32_t l, e; const uint8_t* blob; int job; int32_t peer; };   // peer: a 4070 slot or -1
        std::vector<StreamEntry> seq;
        std::vector<size_t> seq_start;
        size_t issued = 0, consumed = 0;
        const double tl_plan = timeline::enabled() ? timeline::now_us() : 0;
        if (stream_all) {
            seq_start.assign((size_t) g.n_layers + 1, 0);
            std::vector<Stager::Job> js;
            for (int64_t l = LB; l < LE; ++l) {
                seq_start[(size_t) l] = seq.size();
                for (int32_t ei = 0; ei < m.g->n_expert; ++ei) {
                    const int32_t e = expert_at(l, ei, m.g->n_expert);   // #41 gate 2
                    if (m.host_res && m.cache && m.host_res[(size_t) l * m.g->n_expert + e] >= 0) continue;
                    if (split_on) break;   // #32 S4: every routed expert of the layer runs on the 4070
                    const int32_t ps = m.peer_res ? m.peer_res[(size_t) l * m.g->n_expert + e] : -1;
                    const uint8_t* b = ps >= 0 ? nullptr : m.src->blob(l, e);
                    const bool from_pack = ps < 0 && b == nullptr;   // #11: on NVMe - read, do not admit
                    const int tier = ps >= 0 ? 2 : from_pack ? 3 : m.src->pinned(l, e) ? 0 : 1;
                    ++stats_.src_n[tier];
                    stats_.src_bytes[tier] += lay0.blob_bytes(l);
                    int job = -1;
                    if (ps < 0 && (from_pack || !m.src->pinned(l, e))) {
                        job = (int) js.size();
                        js.push_back({b, (size_t) lay0.blob_bytes(l), (int32_t) l, e});
                    }
                    seq.push_back({(int32_t) l, e, b, job, ps});
                }
            }
            for (int64_t l = LE; l <= g.n_layers; ++l) seq_start[(size_t) l] = seq.size();
            m.stager->start(std::move(js));
        }
        if (tl_plan > 0) timeline::complete("stream plan (host)", tl_plan, timeline::now_us(), (int64_t) seq.size(), c0);
        // #32 S4: the 4070's stream plan, its stager and its issuer for this chunk (joined when the chunk ends)
        struct SplitJoin { SplitTier* sp; ~SplitJoin() { if (sp) sp->join_issuer(); } } split_join{nullptr};
        // #35 D7: in a wave, lane 1 plans the whole prompt at its first chunk (units of two chunks) and lane 2
        // consumes that stream; a lane whose chunk does not run split steps over its share of the unit
        const bool wave_split = wave != nullptr && m.split != nullptr;
        const size_t unit = wave ? (size_t) (chunk_i / 2) : 0;
        if (wave_split && m.wave_lane == 1 && split_base && m.T >= STREAM_ALL_MIN) {
            // lane 1's first chunk is full (this lane has a chunk), so it planned the stream: consume it
            SplitTier* owner = wave->wait_split_owner();
            if (owner == nullptr) { err = "prefill: the other wave lane failed"; return false; }
            m.split->xs = owner;
        }
        if (wave_split && !split_on && m.split->xs->plan_live) {   // this chunk runs on one card: pass the unit
            SplitTier& x = *m.split->xs;
            const size_t end = x.seq_start[(unit + 1) * (size_t) (x.layers + 1) - 1];
            auto& mine = m.wave_lane == 1 ? x.consumed1 : x.consumed;
            if (mine.load(std::memory_order_acquire) < end) SplitTier::publish(mine, end);
        }
        if (split_on && (!wave || (m.wave_lane == 0 && c0 == 0))) {
            SplitTier& sp = *m.split;
            if (sp.issuer.joinable() || sp.res_thread.joinable()) sp.join_issuer();   // a failed run's stream
            sp.xs = &sp;
            sp.layers = g.n_layers;
            // lane 2 consumes the plan when it has a chunk (it then attaches: same condition as this plan's)
            sp.lanes_expected = wave && n > m.T ? 2 : 1;
            sp.aborted.store(false);
            const int64_t n_units = wave ? ((n + m.T - 1) / m.T + 1) / 2 : 1;
            sp.seq.clear();
            sp.seq_start.assign((size_t) (n_units * (g.n_layers + 1)), 0);
            sp.unit_jobs.assign((size_t) n_units, {});
            sp.unit_first.assign((size_t) n_units + 1, 0);
            for (int64_t u = 0; u < n_units; ++u) {
                sp.unit_first[(size_t) u] = sp.seq.size();
                std::vector<Stager::Job>& js4 = sp.unit_jobs[(size_t) u];
                for (int64_t l = 0; l < g.n_layers; ++l) {
                    sp.seq_start[(size_t) (u * (g.n_layers + 1) + l)] = sp.seq.size();
                    for (int32_t ei = 0; ei < m.g->n_expert; ++ei) {
                        const int32_t e = expert_at(l, ei, m.g->n_expert);   // #41 gate 2
                        if (m.peer_res[(size_t) l * m.g->n_expert + e] >= 0) continue;   // the 4070's own
                        if (m.host_res && m.cache && m.host_res[(size_t) l * m.g->n_expert + e] >= 0) {
                            sp.seq.push_back({(int32_t) l, e, -1, 1});
                        } else if (m.src->pinned(l, e)) {
                            sp.seq.push_back({(int32_t) l, e, -1, 2});
                        } else {
                            sp.seq.push_back({(int32_t) l, e, (int) js4.size(), 3});
                            js4.push_back({m.src->blob(l, e), (size_t) lay0.blob_bytes(l), (int32_t) l, e});
                        }
                    }
                }
                sp.seq_start[(size_t) (u * (g.n_layers + 1) + g.n_layers)] = sp.seq.size();
            }
            sp.unit_first[(size_t) n_units] = sp.seq.size();
            sp.consumed.store(0);
            SplitTier::publish(sp.consumed1, sp.lanes_expected == 2 ? 0 : SplitTier::kNever);
            sp.lanes_done.store(0);
            sp.plan_live = true;
            sp.issued.store(0);
            sp.res_list.clear();
            for (size_t k = 0; k < sp.seq.size(); ++k) if (sp.seq[k].kind == 1) sp.res_list.push_back(k);
            sp.res_ready.store(0);
            sp.res_uploaded.store(0);
            sp.res_thread = std::thread([&m, &lay0] {
                SplitTier& t = *m.split;
                cudaSetDevice(t.home);
                timeline::name_thread("prefill split resident copies");
                // #35 D4: no stream of one card waits on an event of the other.  A 4070 copy that waited on this
                // card's D2H blocked the 4070's copy engine for everything queued behind it (the stager's DMAs
                // too), which stalled the issuer, which held back the D2H it waited for: 58 ms per layer (tlD4).
                // Now the D2Hs are queued ahead into free slots, and an entry is published only once its bytes
                // are in host memory; a slot is refilled only once the 4070's upload from it has completed.
                const size_t n = t.res_list.size();
                size_t enq = 0, done = 0;
                while (done < n) {
                    while (enq < n && enq < done + (size_t) SplitTier::RES) {
                        const int hs = (int) (enq % SplitTier::RES);
                        if (enq >= (size_t) SplitTier::RES) {   // the slot's previous upload was issued, and done
                            if (t.res_uploaded.load(std::memory_order_acquire) <= enq - (size_t) SplitTier::RES) break;
                            cudaEventSynchronize(t.ev_resread[hs]);
                        }
                        const SplitTier::Entry& en = t.seq[t.res_list[enq]];
                        cudaEvent_t r0 = t.clk_res.record(t.res_stream);
                        cudaMemcpyAsync(t.hres + (size_t) hs * t.blob,
                                        m.cache->device_slot(m.host_res[(size_t) en.l * m.g->n_expert + en.e]),
                                        (size_t) lay0.blob_bytes(en.l), cudaMemcpyDeviceToHost, t.res_stream);
                        cudaEventRecord(t.ev_res[hs], t.res_stream);
                        t.clk_res.span(t.tl_res, "resident down", r0, t.clk_res.record(t.res_stream), en.l, en.e);
                        ++enq;
                    }
                    (void) cudaStreamQuery(t.res_stream);
                    if (done < enq) {
                        cudaEventSynchronize(t.ev_res[done % SplitTier::RES]);
                        SplitTier::publish(t.res_ready, ++done);
                    } else if (!SplitTier::wait_above(t.res_uploaded, enq - (size_t) SplitTier::RES, t.stop)) {
                        return;   // every slot waits for an upload the issuer has not queued yet
                    }
                    if (t.stop.load(std::memory_order_acquire)) return;
                }
            });
            sp.stager->start(std::move(sp.unit_jobs[0]));
            if (!wave) split_join.sp = &sp;   // a wave's stream is joined by the lane that finishes last (run's end)
            sp.issuer = std::thread([&m, &lay0] {
                SplitTier& t = *m.split;
                cudaSetDevice(t.dev);
                timeline::name_thread("prefill split issuer");
                size_t ri = 0;   // the next kind-1 entry's index in res_list
                size_t u = 0;    // the unit being issued (its jobs are the stager's)
                for (size_t k = 0; k < t.seq.size(); ++k) {
                    while (k >= t.unit_first[u + 1]) {   // the next unit: its jobs to the stager
                        t.stager->finish();
                        ++u;
                        t.stager->start(std::move(t.unit_jobs[u]));
                    }
                    if (k >= (size_t) t.RING && (!SplitTier::wait_above(t.consumed, k - (size_t) t.RING, t.stop) ||
                                                 !SplitTier::wait_above(t.consumed1, k - (size_t) t.RING, t.stop)))
                        return;
                    if (t.stop.load(std::memory_order_relaxed) || t.aborted.load(std::memory_order_acquire)) return;
                    const SplitTier::Entry& en = t.seq[k];
                    timeline::Span issue_span(en.kind == 1 ? "split issue resident" : "split issue host", (int64_t) k, en.l);
                    const int rs = (int) (k % t.RING);
                    const size_t bb = (size_t) lay0.blob_bytes(en.l);
                    uint8_t* dst = t.ring + (size_t) rs * t.blob;
                    const cudaStream_t cs4 = en.kind == 1 ? t.cr : t.c;
                    cudaStreamWaitEvent(cs4, t.used[rs], 0);   // the group that read this slot has gathered it
                    if (t.lanes_expected == 2) cudaStreamWaitEvent(cs4, t.used1[rs], 0);   // and lane 2's (#35 D7)
                    cudaEvent_t c0 = nullptr;
                    if (en.kind == 1) {
                        // the 5060's own copy, brought down by res_thread into a pinned slot: up here
                        if (!SplitTier::wait_above(t.res_ready, ri, t.stop)) return;
                        const int hs = (int) (ri % SplitTier::RES);
                        c0 = t.clk_c.record(t.cr);
                        cudaMemcpyAsync(dst, t.hres + (size_t) hs * t.blob, bb, cudaMemcpyHostToDevice, t.cr);
                        cudaEventRecord(t.ev_resread[hs], t.cr);
                        SplitTier::publish(t.res_uploaded, ++ri);
                    } else if (en.kind == 2) {
                        c0 = t.clk_c.record(t.c);
                        cudaMemcpyAsync(dst, m.src->blob(en.l, en.e), bb, cudaMemcpyHostToDevice, t.c);
                    } else {
                        const uint8_t* hb = nullptr;
                        {
                            timeline::Span ws("split stager wait", en.job, en.l);
                            hb = t.stager->wait(en.job);
                        }
                        c0 = t.clk_c.record(t.c);
                        cudaMemcpyAsync(dst, hb, bb, cudaMemcpyHostToDevice, t.c);
                        t.stager->issued_one(en.job, t.c);
                    }
                    cudaEventRecord(t.copied[rs], cs4);
                    t.clk_c.span(t.tl_c, en.kind == 1 ? "copy resident" : en.kind == 2 ? "copy pinned" : "copy staged", c0,
                                 t.clk_c.record(cs4), en.l, en.e);
                    SplitTier::publish(t.issued, k + 1);
                }
            });
            if (wave) wave->set_split_owner(&sp);
        }
        struct StagerDone {
            Stager* st;
            ~StagerDone() { if (st) st->finish(); }
        } chunk_stager_done{stream_all ? m.stager.get() : nullptr};
        auto issue_one = [&](size_t i) {
                const StreamEntry& en = seq[i];
                const int sl = (int) (i % (size_t) m.ring);
                const auto th = Clock::now();
                timeline::Span issue_span("copy issue", (int64_t) i, en.l);
                const size_t bytes = (size_t) lay0.blob_bytes(en.l);
                if (m.stage_live[sl]) cudaStreamWaitEvent(m.copy, m.used[sl], 0);
                cudaEvent_t tl0 = nullptr;   // recorded right before the copy: never across a host wait
                const char* tl_name = en.peer >= 0 ? "copy peer" : en.job < 0 ? "copy pinned"
                                                           : en.blob == nullptr ? "copy nvme" : "copy staged";
                if (en.peer >= 0) {
                    // #4: its only copy is on the 4070 - a peer copy (staged through the host by the driver when
                    // the cards have no P2P path)
                    int self_dev = 0;
                    cudaGetDevice(&self_dev);
                    tl0 = clk_copy.record(m.copy);
                    cudaMemcpyPeerAsync(m.stage_dev[sl], self_dev, m.peer_ptr(en.peer), m.peer_dev, bytes, m.copy);
                    ++stats_.experts_dma;
                } else if (en.job < 0) {
                    tl0 = clk_copy.record(m.copy);
                    cudaMemcpyAsync(m.stage_dev[sl], en.blob, bytes, cudaMemcpyHostToDevice, m.copy);
                    ++stats_.experts_dma;
                } else {
                    const uint8_t* hb = nullptr;
                    {
                        timeline::Span wait_span("stager wait", en.job, (int64_t) i);
                        hb = m.stager->wait(en.job);
                    }
                    tl0 = clk_copy.record(m.copy);
                    cudaMemcpyAsync(m.stage_dev[sl], hb, bytes, cudaMemcpyHostToDevice, m.copy);
                    m.stager->issued_one(en.job, m.copy);
                }
                clk_copy.span(tl_copy, tl_name, tl0, clk_copy.record(m.copy), (int64_t) i, en.l);
                cudaEventRecord(m.copied[sl], m.copy);
                m.stage_live[sl] = true;
                stats_.ms_experts_host += ms_since(th);
                ++stats_.experts_streamed;
        };
        auto issue_until = [&](size_t limit) {
            limit = std::min(limit, seq.size());
            while (issued < limit) issue_one(issued++);
        };
        // #30, a x4 link: the copy queue fills and cudaMemcpyAsync blocks (243 us per expert, 5.0 s of an 8K prompt
        // on the launching thread), which then cannot launch the compute - the GPU idles while the link is busy.
        // STRATA_PREFILL_COPY_THREAD=1: a thread issues the stream's copies instead; entry i is issued once entry
        // i - ring is consumed (its `used` event recorded, published through a_consumed), and the compute thread
        // waits on entry k's `copied` event only once the issuer has recorded it (a_issued > k).
        // (upstream D-5) the issuer thread is on by default; STRATA_PREFILL_ISSUER=0 (or the older
        // STRATA_PREFILL_COPY_THREAD=0) issues in line
        static const bool copy_thread_on = [] {
            const char* v = std::getenv("STRATA_PREFILL_ISSUER");
            if (v == nullptr) v = std::getenv("STRATA_PREFILL_COPY_THREAD");
            return v == nullptr || std::atoi(v) != 0;
        }();
        const bool use_issuer = stream_all && copy_thread_on;
        // #29: STRATA_PREFILL_GROUP_GATHER=0 keeps one gather launch per expert (the A/B)
        static const bool group_gather_on = [] {
            const char* v = std::getenv("STRATA_PREFILL_GROUP_GATHER");
            return v == nullptr || std::atoi(v) != 0;
        }();
        std::atomic<size_t> a_consumed{0}, a_issued{0};
        std::atomic<bool> issuer_stop{false};
        std::thread issuer;
        struct IssuerJoin {
            std::atomic<bool>& stop;
            std::thread& t;
            ~IssuerJoin() { stop.store(true); if (t.joinable()) t.join(); }
        } issuer_join{issuer_stop, issuer};
        if (use_issuer) {
            int dev = 0;
            cudaGetDevice(&dev);
            issuer = std::thread([&, dev] {
                cudaSetDevice(dev);
                timeline::name_thread("prefill copy issuer");
                for (size_t i = 0; i < seq.size(); ++i) {
                    const double tw = timeline::enabled() ? timeline::now_us() : 0;
                    bool waited = false;
                    while (i >= a_consumed.load(std::memory_order_acquire) + (size_t) m.ring) {
                        if (issuer_stop.load(std::memory_order_relaxed)) return;
                        waited = true;
                        std::this_thread::yield();
                    }
                    if (waited && tw > 0) timeline::complete("issuer wait ring slot", tw, timeline::now_us(), (int64_t) i);
                    if (issuer_stop.load(std::memory_order_relaxed)) return;
                    issue_one(i);
                    a_issued.store(i + 1, std::memory_order_release);
                }
            });
        } else if (stream_all) {
            issue_until((size_t) m.ring);   // layer 0's first experts, behind the embedding and the PLE
        }
        const bool threaded_issue = use_issuer;   // upstream's name for the issuer thread being on
        // the consumer's side: entry k's copy is on the copy stream (the thread issued it), then k is given back
        auto wait_issued = [&](size_t k) {
            if (!threaded_issue) return;
            while (a_issued.load(std::memory_order_acquire) <= k) std::this_thread::yield();
        };
        auto give_back = [&](size_t upto) {
            if (threaded_issue) a_consumed.store(upto, std::memory_order_release);
            else issue_until(upto + (size_t) m.ring);
        };
        host_setup_ms += ms_since(tsetup);
        bool normed = false;   // F-2: the previous half's write already normed R for this half (grs, xn16)
        for (int64_t l = LB; l < LE; ++l) {
            core::progress_beat();   // the serve watchdog: a prompt chunk of 8192 tokens is still moving
            timeline::Span layer_span("layer (host)", l, c0);
            pt.layer = l;
            const core::LayerView v(*m.wt, l);
            if (wave && chunk_i > 0) {   // #35 D7: chunk c-1's layer-l state (KV, GDN, PLE history) is queued
                timeline::Span ws("wave wait", l, chunk_i);
                if (!wave->wait_attn(chunk_i - 1, l, m.cs)) { err = "prefill: the other wave lane failed"; return false; }
            }
            if (l == 1 && ple_on && ple_ahead_on && !ple_take()) return false;   // this chunk's PLE rows, uploaded
            // ---- the PLE block at layer 1, token by token (its conv reads the previous tokens' rows)
            if (l == 1 && ple_on && ple_batch) {
                // the whole chunk at once, in sub-batches carved from the idle scratch region: the key and value
                // projections as GEMMs (a token at a time they re-read ~52 MB of BF16 key per token on the IQ
                // files), the rest with the per-token kernels' arithmetic (native_ple_postops_batch)
                pt.mark(kPfPle, cs);
                const auto tp = Clock::now();
                const strata::kernels::PleWeights& pw = ss.ple.w;
                constexpr int64_t HD = strata::kernels::NG_HC_DIM;
                const uint64_t per_token = (uint64_t) (3 * HD + N + 4) * 4 + (uint64_t) N * 2 + 4096;
                const int64_t SB = std::min<int64_t>(T, (int64_t) (m.region_bytes / per_token));
                for (int64_t s0 = 0; s0 < T; s0 += SB) {
                    const int64_t nb = std::min(SB, T - s0);
                    uint8_t* q = m.region;
                    auto carve_f = [&](size_t n) { float* p = (float*) q; q += (n * 4 + 255) & ~(size_t) 255; return p; };
                    float* key = carve_f((size_t) nb * HD);
                    float* qn = carve_f((size_t) nb * HD);
                    float* gated = carve_f((size_t) nb * HD);
                    float* val = carve_f((size_t) nb * N);
                    float* gate = carve_f((size_t) nb * 4);
                    uint16_t* e16 = (uint16_t*) carve_f((size_t) nb * N / 2);
                    const float* emb = m.ple_emb + s0 * N;
                    if (pw.key_bf16 != nullptr) {
                        to_bf16(emb, e16, nb * N, m.cs);
                        m.gemm.bf16(e16, pw.key_bf16, key, nb, HD, N);
                    } else {
                        to_f16(emb, e16, nb * N, m.cs);
                        m.gemm.native(e16, pw.key_native_type, pw.key_native_data, key, nb, HD, N);
                        to_bf16(emb, e16, nb * N, m.cs);
                    }
                    m.gemm.bf16(e16, pw.value_bf16, val, nb, N, N);
                    try {
                        strata::kernels::native_ple_postops_batch(key, m.R + s0 * D, val, ss.ple.hist, pw, qn, gated,
                                                                  gate, (int) nb, m.cs);
                    } catch (const std::exception& e) { err = std::string("prefill PLE: ") + e.what(); return false; }
                }
                stats_.ms_ple += ms_since(tp);
            } else if (l == 1 && ple_on) {
                pt.mark(kPfPle, cs);
                const auto tp = Clock::now();
                for (int64_t t = 0; t < T; ++t) {
                    strata::kernels::PleOut po;
                    po.normalized = m.ple_norm;
                    po.result = m.R + t * D;
                    try {
                        strata::kernels::ple_block(m.ple_emb + t * N, m.R + t * D, ss.ple.hist, ss.ple.w, po,
                                                   ss.ple.scratch, m.cs);
                    } catch (const std::exception& e) { err = std::string("prefill PLE: ") + e.what(); return false; }
                    strata::kernels::ple_history_advance(ss.ple.hist, m.ple_norm, m.cs);
                }
                stats_.ms_ple += ms_since(tp);
            }
            for (int half = 0; half < 2; ++half) {
                // ---- the hyper-connection read of this half (F-2: already normed by the previous half's write)
                const char* pre = half == 0 ? "hc_attn_" : "hc_ffn_";
                const std::string sn = std::string(pre) + "norm.weight", sd = std::string(pre) + "down.weight",
                                  su = std::string(pre) + "up.weight", si = std::string(pre) + "inject.weight";
                const core::WeightRef *wn = need(v, sn.c_str(), err), *wd = need(v, sd.c_str(), err),
                                      *wu = need(v, su.c_str(), err), *wi = need(v, si.c_str(), err);
                if (!wn || !wd || !wu || !wi) return false;
                pt.mark(kPfHc, cs);
                if (gr_unfused()) gr_norm(m.R, (const float*) wn->data, EPS, m.xn, m.xn16, T, m.cs);
                else if (!normed) gr_norm_rs(m.R, (const float*) wn->data, EPS, m.grs, m.xn16, T, m.cs);
                normed = false;
                if (!bf16_proj(m.gemm, wd, m.xn16, m.lo, T, sd, err)) return false;
                gr_silu(m.lo, m.lo16, T, m.cs);
                if (!bf16_proj(m.gemm, wu, m.lo16, m.gated, T, su, err)) return false;
                if (!bf16_proj(m.gemm, wi, m.xn16, m.inj, T, si, err)) return false;
                if (gr_unfused()) gr_mix(m.xn, m.gated, m.mixed, m.mixed_bf, T, m.cs, m.mixed_h);
                else gr_mix_r(m.R, m.grs, (const float*) wn->data, m.gated, m.mixed, m.mixed_bf, T, m.cs, m.mixed_h);

                if (half == 0 && !core::is_qsa_layer(g, l)) {
                    // ======================= GDN =======================
                    const core::WeightRef *wqkv = need(v, "attn_qkv.weight", err), *wg = need(v, "attn_gate.weight", err),
                                          *wo = need(v, "ssm_out.weight", err), *wa = need(v, "ssm_alpha.weight", err),
                                          *wb = need(v, "ssm_beta.weight", err), *wc = need(v, "ssm_conv1d.weight", err),
                                          *wnm = need(v, "ssm_norm.weight", err), *wdt = need(v, "ssm_dt.bias", err),
                                          *wsa = need(v, "ssm_a", err);
                    if (!wqkv || !wg || !wo || !wa || !wb || !wc || !wnm || !wdt || !wsa) return false;
                    pt.mark(kPfGdn, cs);
                    float* state = ss.gdn_state + (size_t) (gdn_index - ss.gdn_ord0) * gdn_floats;
                    float* conv = state + (uint64_t) g.ssm_state_size * g.ssm_v_heads * g.ssm_state_size;
                    if (!native_proj(m.gemm, wqkv, m.mixed_h, m.qkv, T, v.name("attn_qkv.weight"), err)) return false;
                    if (!native_proj(m.gemm, wg, m.mixed_h, m.z, T, v.name("attn_gate.weight"), err)) return false;
                    if (!bf16_proj(m.gemm, wa, m.mixed_bf, m.ab, T, v.name("ssm_alpha.weight"), err, 2 * HV)) return false;
                    if (!bf16_proj(m.gemm, wb, m.mixed_bf, m.ab + HV, T, v.name("ssm_beta.weight"), err, 2 * HV)) return false;
                    pt.mark(kPfGdnConv, cs);   // "gdn" is the projections in; the rest on their own lines
                    gdn_gates(m.ab, (const float*) wdt->data, (const float*) wsa->data, m.gate, m.beta, T, m.cs);
                    gdn_conv(conv, m.qkv, (const float*) wc->data, m.hbuf, T, EPS, m.cs);
                    pt.mark(kPfGdnRec, cs);
                    gdn_recurrence(state, m.hbuf, m.gate, m.beta, m.z, (const float*) wnm->data, EPS, m.y, m.y_h, T, m.cs);
                    pt.mark(kPfGdnOut, cs);
                    if (!native_proj(m.gemm, wo, m.y_h, m.bo, T, v.name("ssm_out.weight"), err)) return false;
                    ++gdn_index;
                } else if (half == 0) {
                    // ======================= QSA =======================
                    const core::QsaState& st = ss.qsa_states[qsa_index];
                    const core::WeightRef *wq = need(v, "attn_q.weight", err), *wk = need(v, "attn_k.weight", err),
                                          *wv = need(v, "attn_v.weight", err), *wo = need(v, "attn_output.weight", err),
                                          *wik = need(v, "indexer.k_proj.weight", err),
                                          *wiq = need(v, "indexer.q_proj.weight", err),
                                          *wqn = need(v, "attn_q_norm.weight", err), *wkn = need(v, "attn_k_norm.weight", err),
                                          *wiqn = need(v, "indexer.q_norm.weight", err),
                                          *wikn = need(v, "indexer.k_norm.weight", err);
                    if (!wq || !wk || !wv || !wo || !wik || !wiq || !wqn || !wkn || !wiqn || !wikn) return false;
                    pt.mark(kPfQsa, cs);
                    if (!native_proj(m.gemm, wk, m.mixed_h, m.Kc, T, v.name("attn_k.weight"), err)) return false;
                    if (!native_proj(m.gemm, wv, m.mixed_h, m.Vc, T, v.name("attn_v.weight"), err)) return false;
                    if (!native_proj(m.gemm, wq, m.mixed_h, m.Qf, T, v.name("attn_q.weight"), err)) return false;
                    if (!bf16_proj(m.gemm, wik, m.mixed_bf, m.idx_raw, T, v.name("indexer.k_proj.weight"), err)) return false;
                    if (!bf16_proj(m.gemm, wiq, m.mixed_bf, m.q_idx, T, v.name("indexer.q_proj.weight"), err)) return false;
                    rms_rows(m.Kc, (const float*) wkn->data, T * 2, 256, 256, EPS, m.cs);
                    rope(m.Kc, T, 2, 256, 512, p0, strata::kernels::rope_scaling(), m.cs);
                    // KV streaming: this layer's cells [0, p0) come in from the host copy to the staging pool, and the
                    // chunk's cells go to the host copy, the staging pool, and the VRAM slots of resident blocks
                    const bool staged = st.kv_mode == 1;
                    if (staged) {
                        pt.mark(kPfKvStage, cs);
                        strata::kernels::kv_stage_from_host(pools_of(m.stage, m.ident_table), st.host,
                                                            core::qsa_kv_format(st),
                                                            (p0 + s.page_size - 1) / s.page_size, s, m.cs);
                        pt.mark(kPfQsa, cs);
                    }
                    if (st.kv_hybrid) {   // K8V4: K INT8 unrotated, V rotated Q4_0 (only V and the output rotate)
                        strata::kernels::fwht256_inplace_cuda(m.Vc, T * 2, m.cs);
                        kv_append(m.Kc, m.Kc, T, p0, st.page_table, s.page_size, nullptr, nullptr,
                                  st.k_q, st.k_q, st.k_scale, st.k_scale, m.cs, nullptr,   // mode 0: no host mirror
                                  staged ? &m.stage : nullptr);
                        strata::kernels::kv_append_q4(st.v_q4, st.v_q4, st.page_table, p0, T, m.Vc, m.Vc, s, m.cs,
                                                      nullptr, staged ? &m.stage : nullptr);
                    } else if (st.kv_q4) {   // Q4_0 KV (kv_q4.hpp): rotated K and V, the queries below too, the output back
                        strata::kernels::fwht256_inplace_cuda(m.Kc, T * 2, m.cs);
                        strata::kernels::fwht256_inplace_cuda(m.Vc, T * 2, m.cs);
                        strata::kernels::kv_append_q4(st.k_q4, st.v_q4, st.page_table, p0, T, m.Kc, m.Vc, s, m.cs,
                                                      &st.host, staged ? &m.stage : nullptr);
                    } else {
                        kv_append(m.Kc, m.Vc, T, p0, st.page_table, s.page_size, st.kv_int8 ? nullptr : st.k_pool,
                                  st.kv_int8 ? nullptr : st.v_pool, st.k_q, st.v_q, st.k_scale, st.v_scale, m.cs,
                                  &st.host, staged ? &m.stage : nullptr);
                    }
                    split_q(m.Qf, m.q, T, m.cs);
                    rms_rows(m.q, (const float*) wqn->data, T * 24, 256, 256, EPS, m.cs);
                    rope(m.q, T, 24, 256, 6144, p0, strata::kernels::rope_scaling(), m.cs);
                    if (st.kv_q4) strata::kernels::fwht256_inplace_cuda(m.q, T * 24, m.cs);
                    rms_rows(m.q_idx, (const float*) wiqn->data, T * 4, 128, 128, EPS, m.cs);
                    rope(m.q_idx, T, 4, 128, 512, p0, strata::kernels::rope_scaling(), m.cs);
                    // the indexer appends, token by token; then scores + selection for many queries at once:
                    // a query reads completed blocks (final once completed) and `dead` for its own tail block
                    const strata::kernels::QsaIndexerBuffers ib{st.idx_tail, st.idx_dead, st.idx_pooled, st.idx_block_pos};
                    pt.mark(kPfQsaIdx, cs);
                    // C-2: the chunk's appends in three launches instead of one per token (the same end state:
                    // the queries below read it only after the whole chunk is appended). STRATA_INDEXER_PER_TOKEN=1: the old
                    try {
                        static const bool per_token = std::getenv("STRATA_INDEXER_PER_TOKEN") != nullptr;
                        if (!per_token) {
                            strata::kernels::native_qsa_indexer_append_batch(m.idx_raw, T, p0, 0, (const float*) wikn->data,
                                                                             EPS, ib, s, st.max_cells,
                                                                             strata::kernels::rope_scaling(), m.cs);
                        }
                        for (int64_t t = 0; per_token && t < T; ++t) {
                            const int32_t* step_t = m.steps_dev + t * strata::kernels::kStepCount;
                            strata::kernels::native_qsa_indexer_append(m.idx_raw + t * 128, step_t + strata::kernels::kStepPos, 0,
                                                                       (const float*) wikn->data, EPS, ib, s, st.max_cells,
                                                                       strata::kernels::rope_scaling(), m.cs);
                        }
                    } catch (const std::exception& e) { err = std::string("prefill indexer: ") + e.what(); return false; }
                    pt.mark(kPfQsaSel, cs);
                    for (int64_t t0 = 0; t0 < T; t0 += m.sel_batch) {
                        const int64_t nb = std::min(m.sel_batch, T - t0);
                        const int32_t* steps0 = m.steps_dev + t0 * strata::kernels::kStepCount;
                        // C-1: the grid reaches the batch's last query's n_bid (they rise with the position)
                        const int64_t active = (int64_t) m.steps_host[(size_t) ((t0 + nb - 1) * strata::kernels::kStepCount +
                                                                                strata::kernels::kStepNBid)] + 1;
                        // the scores on tensor cores (3xTF32: FP32-level, not bitwise); STRATA_SELECT_OLD=1: the warp kernel
                        static const bool old_sel = std::getenv("STRATA_SELECT_OLD") != nullptr;
                        if (old_sel || !strata::kernels::qsa_block_scores_tc(st.idx_pooled, st.idx_dead, m.q_idx + t0 * 512,
                                                                             steps0, nb, m.max_blocks, s, m.sel_scores,
                                                                             m.cs, active))
                            strata::kernels::qsa_block_scores(st.idx_pooled, st.idx_dead, m.q_idx + t0 * 512, steps0, nb,
                                                              m.max_blocks, s, m.sel_scores, m.cs, active);
                        strata::kernels::qsa_block_topk(m.sel_scores, steps0, nb, m.max_blocks, m.cap, s,
                                                        m.sel_ids + t0 * m.cap, m.cs);
                    }
                    // STRATA_SEL_OVERLAP (debug, D-1's question): how much do neighbouring queries' selections share?
                    // Per tile of 16 queries: the union of their selected cells against the sum of their widths.
                    if (static const bool ovl = std::getenv("STRATA_SEL_OVERLAP") != nullptr; ovl && qsa_index == 0) {
                        std::vector<int32_t> ids((size_t) (T * m.cap));
                        cudaMemcpyAsync(ids.data(), m.sel_ids, ids.size() * 4, cudaMemcpyDeviceToHost, m.cs);
                        cudaStreamSynchronize(m.cs);
                        double sum_w = 0, sum_u = 0;
                        for (int64_t t0 = 0; t0 + 16 <= T; t0 += 16) {
                            std::vector<int32_t> u;
                            for (int64_t t = t0; t < t0 + 16; ++t) {
                                const int64_t w = m.steps_host[(size_t) (t * strata::kernels::kStepCount + strata::kernels::kStepWidth)];
                                sum_w += (double) w;
                                u.insert(u.end(), ids.begin() + t * m.cap, ids.begin() + t * m.cap + w);
                            }
                            std::sort(u.begin(), u.end());
                            sum_u += (double) (std::unique(u.begin(), u.end()) - u.begin());
                        }
                        std::fprintf(stderr, "strata prefill: selection overlap at %lld: 16-query tiles read %.1f%% of the "
                                             "cells one query at a time does\n", (long long) p0, sum_w > 0 ? 100.0 * sum_u / sum_w : 0.0);
                    }
                    // STRATA_IDX_FP16_CHECK: would FP16 pooled indexer keys select the same cells? (the KV-streaming
                    // design's last question). Every query is selected again from the pooled keys and `dead` rounded
                    // to fp16 (exactly what an fp16 store reads back); the agreement with the fp32 selection is
                    // printed cumulatively after each chunk's last QSA layer. Debug: syncs per layer.
                    if (static const bool f16chk = std::getenv("STRATA_IDX_FP16_CHECK") != nullptr; f16chk) {
                        static float *pooled16 = nullptr, *dead16 = nullptr;
                        static int32_t* ids16 = nullptr;
                        static double shared = 0, cells = 0;
                        static long long queries = 0, same = 0, sel_queries = 0;
                        const int64_t rows = st.idx_pooled_rows;
                        if (pooled16 == nullptr &&
                            (cudaMalloc((void**) &pooled16, (size_t) rows * s.idx_dim * 4) != cudaSuccess ||
                             cudaMalloc((void**) &dead16, (size_t) s.idx_dim * 4) != cudaSuccess ||
                             cudaMalloc((void**) &ids16, (size_t) (m.T * m.cap) * 4) != cudaSuccess)) {
                            err = "STRATA_IDX_FP16_CHECK: no room for its buffers";
                            return false;
                        }
                        round_f16(st.idx_pooled, pooled16, rows * s.idx_dim, m.cs);
                        round_f16(st.idx_dead, dead16, s.idx_dim, m.cs);
                        for (int64_t t0 = 0; t0 < T; t0 += m.sel_batch) {
                            const int64_t nb = std::min(m.sel_batch, T - t0);
                            const int32_t* steps0 = m.steps_dev + t0 * strata::kernels::kStepCount;
                            strata::kernels::qsa_block_scores(pooled16, dead16, m.q_idx + t0 * 512, steps0, nb,
                                                              m.max_blocks, s, m.sel_scores, m.cs);
                            strata::kernels::qsa_block_topk(m.sel_scores, steps0, nb, m.max_blocks, m.cap, s,
                                                            ids16 + t0 * m.cap, m.cs);
                        }
                        std::vector<int32_t> a((size_t) (T * m.cap)), b((size_t) (T * m.cap));
                        cudaMemcpyAsync(a.data(), m.sel_ids, a.size() * 4, cudaMemcpyDeviceToHost, m.cs);
                        cudaMemcpyAsync(b.data(), ids16, b.size() * 4, cudaMemcpyDeviceToHost, m.cs);
                        cudaStreamSynchronize(m.cs);
                        for (int64_t t = 0; t < T; ++t) {
                            const int64_t w = m.steps_host[(size_t) (t * strata::kernels::kStepCount + strata::kernels::kStepWidth)];
                            const int32_t *x = a.data() + t * m.cap, *y = b.data() + t * m.cap;
                            int64_t i = 0, j = 0, c = 0;
                            while (i < w && j < w) {
                                if (x[i] == y[j]) { ++c; ++i; ++j; } else if (x[i] < y[j]) ++i; else ++j;
                            }
                            ++queries;
                            same += c == w;
                            if (p0 + t + 1 > m.cap) { ++sel_queries; shared += (double) c; cells += (double) w; }
                        }
                        if (qsa_index + 1 == g.n_qsa_layers())
                            std::fprintf(stderr, "strata prefill: FP16 indexer keys: %lld of %lld selections identical; "
                                                 "where the selection is sparse, %.4f%% of cells shared (%lld queries)\n",
                                         same, queries, cells > 0 ? 100.0 * shared / cells : 100.0, sel_queries);
                    }
                    // STRATA_QSA_DUMP=<file>: append every QSA layer's selected cells for the prompt's last
                    // STRATA_QSA_DUMP_LAST (4096) positions - records of int32 {qsa layer, pos0, T, cap} + T*cap cells,
                    // for tools/qsa_locality.py (how local the sparse attention's reads are: the KV-streaming question)
                    if (static const char* dump = std::getenv("STRATA_QSA_DUMP"); dump != nullptr) {
                        static const long long last = std::getenv("STRATA_QSA_DUMP_LAST")
                                                          ? std::atoll(std::getenv("STRATA_QSA_DUMP_LAST")) : 4096;
                        if (p0 + T > pos0 + n - last) {
                            std::vector<int32_t> h((size_t) (T * m.cap));
                            cudaMemcpyAsync(h.data(), m.sel_ids, h.size() * 4, cudaMemcpyDeviceToHost, m.cs);
                            cudaStreamSynchronize(m.cs);
                            if (std::FILE* f = std::fopen(dump, "ab")) {
                                const int32_t hdr[4] = {(int32_t) qsa_index, (int32_t) p0, (int32_t) T, (int32_t) m.cap};
                                std::fwrite(hdr, 4, 4, f);
                                std::fwrite(h.data(), 4, h.size(), f);
                                std::fclose(f);
                            }
                        }
                    }
                    const strata::kernels::QsaAttnPools pools = staged ? pools_of(m.stage, m.ident_table)
                                                                       : core::qsa_attn_pools(st);
                    pt.mark(kPfQsaAttn, cs);
                    // perf-review D-1: the whole chunk on tensor cores, one block per (query, KV head), FP32-level
                    // accuracy but not bitwise (qsa_prompt_attn.hpp). Q4_0 KV, or STRATA_PROMPT_ATTN_OLD=1: the
                    // decode kernel, 32 queries at a time (K8V4 runs the tensor kernel's mode 3: INT8 K,
                    // V dequantized from its q4_0 blocks to fp16 at gather)
                    static const bool old_attn = std::getenv("STRATA_PROMPT_ATTN_OLD") != nullptr;
                    if (old_attn || !strata::kernels::qsa_prompt_attn_batch(m.q, pools, m.sel_ids, m.steps_dev, m.cap, s,
                                                                            m.attn, T, m.cs))
                        for (int64_t t0 = 0; t0 < T; t0 += m.attn_batch) {
                            const int64_t nb = std::min(m.attn_batch, T - t0);
                            strata::kernels::qsa_decode_attn_batch(m.q + t0 * ZV, pools, m.sel_ids + t0 * m.cap,
                                                                   m.steps_dev + t0 * strata::kernels::kStepCount, m.cap,
                                                                   s, m.attn_scratch, m.attn + t0 * ZV, nb, m.cs);
                        }
                    if (st.kv_q4 || st.kv_hybrid) strata::kernels::fwht256_inplace_cuda(m.attn, T * 24, m.cs);
                    pt.mark(kPfQsa, cs);
                    gate_attn(m.attn, m.Qf, m.attn_h, T, m.cs);
                    if (!native_proj(m.gemm, wo, m.attn_h, m.bo, T, v.name("attn_output.weight"), err)) return false;
                    ++qsa_index;
                } else {
                    // ======================= MoE =======================
                    if (split_on) cudaStreamWaitEvent(m.cs, m.split->ev_gates, 0);   // #32 S4: last layer's inputs are in host
                    const core::WeightRef *wr = need(v, "ffn_gate_inp.weight", err),
                                          *wgi = need(v, "ffn_gate_inp_shexp.weight", err),
                                          *wsg = need(v, "ffn_gate_shexp.weight", err),
                                          *wsu = need(v, "ffn_up_shexp.weight", err),
                                          *wsd = need(v, "ffn_down_shexp.weight", err);
                    if (!wr || !wgi || !wsg || !wsu || !wsd) return false;
                    pt.mark(kPfRouter, cs);
                    if (!bf16_proj(m.gemm, wr, m.mixed_bf, m.logits, T, v.name("ffn_gate_inp.weight"), err)) return false;
                    route(m.logits, m.ids, m.w, T, m.g->n_expert, m.cs);
                    // the shared expert and its scalar gate (#35 D1: in a split layer it runs after the routed experts
                    // are handed to the 4070, overlapping them; the 4070 never needs it)
                    auto shared_expert = [&]() -> bool {
                        if (!native_proj(m.gemm, wsg, m.mixed_h, m.sgate, T, v.name("ffn_gate_shexp.weight"), err)) return false;
                        if (!native_proj(m.gemm, wsu, m.mixed_h, m.sup, T, v.name("ffn_up_shexp.weight"), err)) return false;
                        swiglu_pair(m.sgate, m.sup, m.sh_h, T, m.cs);
                        if (!native_proj(m.gemm, wsd, m.sh_h, m.shared, T, v.name("ffn_down_shexp.weight"), err)) return false;
                        if (wgi->kind != core::WeightKind::Bf16InF32) { err = "prefill: shared gate is not BF16"; return false; }
                        m.gemm.bf16(m.mixed_bf, (const uint16_t*) wgi->data, m.sg, T, 1, N);
                        return true;
                    };
                    if (!split_on && !shared_expert()) return false;
                    // group the (token, k) pairs by expert on the host
                    pt.mark(kPfHostGroup, cs);
                    [[maybe_unused]] const bool grp_mapped = m.grp_host != nullptr;   // (upstream's name)
                    {
                        timeline::Span sync_span("router sync", l, (int64_t) consumed);
                        // (the sync also orders this layer's host writes of slot/src/bounds after the previous
                        // layer's kernels that read them)
                        if (m.grp_host) copy_i32(m.grp_dev, m.ids, T * K, m.cs);
                        else cudaMemcpyAsync(m.ids_host.data(), m.ids, (size_t) T * K * 4, cudaMemcpyDeviceToHost, m.cs);
                        cudaStreamSynchronize(m.cs);
                    }
                    const double tl_fold = timeline::enabled() ? timeline::now_us() : 0;
                    pt.fold();
                    if (tl_fold > 0) timeline::complete("group: timer fold", tl_fold, timeline::now_us(), l);
                    const double tl_group = timeline::enabled() ? timeline::now_us() : 0;
                    const int32_t* ids_h = m.ids_h();
                    int32_t* slot_h = m.slot_h();
                    int32_t* src_h = m.src_h();
                    std::fill(m.cnt.begin(), m.cnt.end(), 0);
                    for (int64_t i = 0; i < T * K; ++i) {
                        const int32_t e = ids_h[(size_t) i];
                        if (e < 0 || e >= m.g->n_expert) { err = "prefill: routed id out of range"; return false; }
                        ++m.cnt[(size_t) e];
                    }
                    // #32 S4: with expert_split, the 5060's experts' rows first and the 4070's (the peer tier's) after
                    const bool split_l = split_on;   // a split chunk runs every layer's routed experts there
                    {   // each expert's first row, experts laid out in expert_at order (#41 gate 2; id order by default)
                        int64_t at_row = 0;
                        for (int32_t ei = 0; ei < m.g->n_expert; ++ei) {
                            const int32_t e = expert_at(l, ei, m.g->n_expert);
                            m.off[(size_t) e] = (int32_t) at_row;
                            at_row += m.cnt[(size_t) e];
                        }
                        m.off[(size_t) m.g->n_expert] = (int32_t) at_row;
                    }
                    // #32 STRATA_PREFILL_ROUTE_TRACE=<file>: this layer's routing for the 2-GPU simulator
                    // (tests/xeno/perf/prefill_route_sim.py).  One record per MoE layer of a chunk, int32 little-endian:
                    // magic 0x52505453 ('STPR'), pos0 of the chunk, layer, T, K, n_expert, then n_expert rows each
                    // {rows, resident on this card (0/1), owned by the peer card (0/1)}, then the T*K routed ids.
                    if (static const char* rt = std::getenv("STRATA_PREFILL_ROUTE_TRACE"); rt != nullptr) {
                        if (std::FILE* f = std::fopen(rt, "ab")) {
                            const int32_t NE32 = (int32_t) m.g->n_expert;
                            const int32_t hdr[6] = {0x52505453, (int32_t) p0, (int32_t) l, (int32_t) T, (int32_t) K, NE32};
                            std::fwrite(hdr, 4, 6, f);
                            std::vector<int32_t> ex((size_t) NE32 * 3);
                            for (int32_t e = 0; e < NE32; ++e) {
                                ex[(size_t) e * 3] = m.cnt[(size_t) e];
                                ex[(size_t) e * 3 + 1] = m.host_res && m.cache && m.host_res[(size_t) l * NE32 + e] >= 0;
                                ex[(size_t) e * 3 + 2] = m.peer_res && m.peer_res[(size_t) l * NE32 + e] >= 0;
                            }
                            std::fwrite(ex.data(), 4, ex.size(), f);
                            std::fwrite(ids_h, 4, (size_t) T * K, f);
                            std::fclose(f);
                        }
                    }
                    if (tl_group > 0) timeline::complete("group: counts", tl_group, timeline::now_us(), l);
                    const double tl_fill = timeline::enabled() ? timeline::now_us() : 0;
                    std::vector<int32_t> fill(m.off.begin(), m.off.end() - 1);
                    for (int64_t i = 0; i < T * K; ++i) {
                        const int32_t e = ids_h[(size_t) i];
                        const int32_t p = fill[(size_t) e]++;
                        slot_h[(size_t) i] = p;
                        src_h[(size_t) p] = (int32_t) (i / K);
                    }
                    if (tl_fill > 0) timeline::complete("group: fill", tl_fill, timeline::now_us(), l);
                    // the 5060's own expert walk and combine read them; a split layer's go to the 4070 instead (#35 D5:
                    // two pageable uploads here held the 5060's queue ahead of the quantize the 4070 waits for)
                    if (!split_l && m.grp_host) {   // #42: kernels on the compute stream, not the copy engine
                        copy_i32(m.slot_dev, m.grp_dev + m.grp_tk, T * K, m.cs);
                        copy_i32(m.src_dev, m.grp_dev + 2 * m.grp_tk, T * K, m.cs);
                    } else if (!split_l) {
                        cudaMemcpyAsync(m.slot_dev, m.slot_host.data(), (size_t) T * K * 4, cudaMemcpyHostToDevice, m.cs);
                        cudaMemcpyAsync(m.src_dev, m.src_host.data(), (size_t) T * K * 4, cudaMemcpyHostToDevice, m.cs);
                    }
                    // the experts, in id order: resident ones from VRAM, the others through the staging ring
                    std::vector<int32_t> order;
                    std::vector<int32_t> order_4070;   // #32 S4
                    for (int32_t ei = 0; ei < m.g->n_expert; ++ei) {
                        const int32_t e = expert_at(l, ei, m.g->n_expert);   // #41 gate 2
                        if (m.cnt[(size_t) e] > 0) (split_l ? order_4070 : order).push_back(e);
                    }
                    const strata::kernels::cpu::ExpertLayout& lay = strata::kernels::cpu::expert_layout();
                    const bool use_mmq = mmq_plan().any && mmq_plan().layer[(size_t) l];
                    const int mmq_gt = lay.native ? lay.fmt[(size_t) l].gu_type : 42;
                    const int mmq_dt = lay.native ? lay.fmt[(size_t) l].d_type : 42;
                    const size_t mmq_gub = use_mmq ? mmq::matrix_bytes(mmq_gt, 1280, N) : 0;
                    const size_t mmq_db = use_mmq ? mmq::matrix_bytes(mmq_dt, N, 640) : 0;
                    const double tl_pre = timeline::enabled() ? timeline::now_us() : 0;
                    pt.mark(kPfGather, cs);
                    if (tl_pre > 0) timeline::complete("group: mark", tl_pre, timeline::now_us(), l);
                    if (use_mmq) {
                        // step 2b / #34: the chunk's activations as q8_1, once per token (a sub-product gathers its
                        // rows from them: byte-identical to quantizing the gathered rows, xeno_q8_row_gather)
                        {
                            timeline::Span qs("split: quantize enqueue", l);
                            mmq::quantize(m.mixed, nullptr, m.Xtok, mmq_gt, N, N, T, m.cs);
                        }
                        if (split_l && !split_experts(m, l, T, unit, chunk_i, order_4070, mmq_gt, mmq_dt, mmq_gub, mmq_db, err))
                            return false;
                        // the layer's rows per expert (absolute), then each group's sub-products: at most mmq_rows
                        // rows each, with a 0-based bounds block that gate/up and down both read
                        const size_t n = order.size(), ng = (n + MMQ_GROUP - 1) / MMQ_GROUP;
                        m.bounds_host.resize(n + 1);
                        for (size_t j = 0; j < n; ++j) m.bounds_host[j] = m.off[(size_t) order[j]];
                        m.bounds_host[n] = (int32_t) (T * K);
                        m.mmq_subs.clear();
                        m.mmq_sub_first.assign(ng + 1, 0);
                        for (size_t g = 0; g < ng; ++g) {
                            m.mmq_sub_first[g] = m.mmq_subs.size();
                            const size_t a = g * MMQ_GROUP, e = std::min(n, a + MMQ_GROUP);
                            for (size_t q0 = a; q0 < e;) {
                                size_t q1 = q0 + 1;   // one expert always fits: at most T rows, and mmq_rows >= T
                                while (q1 < e && m.bounds_host[q1 + 1] - m.bounds_host[q0] <= m.mmq_rows) ++q1;
                                int64_t maxr = 0;
                                for (size_t i = q0; i < q1; ++i) maxr = std::max<int64_t>(maxr, m.cnt[(size_t) order[i]]);
                                const Impl::MmqSub sub{q0 - a, q1 - a, m.bounds_host[q0],
                                                       m.bounds_host[q1] - m.bounds_host[q0], maxr,
                                                       (int32_t) m.bounds_host.size()};
                                for (size_t i = q0; i <= q1; ++i)
                                    m.bounds_host.push_back(m.bounds_host[i] - m.bounds_host[q0]);
                                m.mmq_subs.push_back(sub);
                                q0 = q1;
                            }
                        }
                        m.mmq_sub_first[ng] = m.mmq_subs.size();
                        if (m.grp_host && m.bounds_host.size() <= m.grp_n - 3 * m.grp_tk) {
                            std::memcpy(m.grp_host + 3 * m.grp_tk, m.bounds_host.data(), m.bounds_host.size() * 4);
                            copy_i32(m.bounds_dev, m.grp_dev + 3 * m.grp_tk, (int64_t) m.bounds_host.size(), m.cs);
                        } else {
                            cudaMemcpyAsync(m.bounds_dev, m.bounds_host.data(), m.bounds_host.size() * 4,
                                            cudaMemcpyHostToDevice, m.cs);
                        }
                    } else {
                        gather_rows16(m.mixed_h, m.src_dev, m.Xs, T * K, N, m.cs);
                    }
                    // Stage ahead: the copy stream moves blobs host -> device while the compute stream works.
                    int stage_next = 0;
                    std::vector<int> stage_of(order.size(), -1);
                    // the unpinned ones are copied to pinned buffers by the stager's threads, in this order
                    std::vector<int> job_of(order.size(), -1);
                    if (!stream_all) {
                        std::vector<Stager::Job> js;
                        for (size_t j = 0; j < order.size(); ++j) {
                            const int32_t e = order[j];
                            if (m.host_res && m.cache && m.host_res[(size_t) l * m.g->n_expert + e] >= 0) continue;
                            if (m.peer_res && m.peer_res[(size_t) l * m.g->n_expert + e] >= 0) continue;   // #4: peer copy
                            if (m.src->pinned(l, e)) continue;
                            const uint8_t* b = m.src->blob(l, e);   // null: on NVMe (#11), read from the pack
                            job_of[j] = (int) js.size();
                            js.push_back({b, (size_t) lay.blob_bytes(l), (int32_t) l, e});
                        }
                        m.stager->start(std::move(js));
                    }
                    if (tl_group > 0) timeline::complete("host grouping", tl_group, timeline::now_us(), l);
                    timeline::Span launch_span("expert launches", l, (int64_t) order.size());
                    StagerDone stager_done{stream_all ? nullptr : m.stager.get()};
                    auto stage_one = [&](size_t j) -> bool {
                        const int32_t e = order[j];
                        const bool resident = m.host_res && m.cache && m.host_res[(size_t) l * m.g->n_expert + e] >= 0;
                        if (resident) return true;
                        const int sl = stage_next;
                        stage_next = (stage_next + 1) % STAGE;
                        const auto th = Clock::now();
                        timeline::Span issue_span("copy issue", (int64_t) j, l);
                        cudaEvent_t tl0 = nullptr;
                        const char* tl_name = "copy pinned";
                        const int32_t peer_slot = m.peer_res ? m.peer_res[(size_t) l * NE + e] : -1;
                        const uint8_t* b = peer_slot >= 0 ? nullptr : m.src->blob(l, e);
                        const bool from_pack = peer_slot < 0 && b == nullptr;   // #11: on NVMe - read, do not admit
                        const int tier = peer_slot >= 0 ? 2 : from_pack ? 3 : m.src->pinned(l, e) ? 0 : 1;
                        ++stats_.src_n[tier];
                        stats_.src_bytes[tier] += lay.blob_bytes(l);
                        stats_.src_rows[tier] += m.cnt[(size_t) e];
                        if (peer_slot >= 0) {
                            // its only copy is on the 4070: a peer copy (staged through the host by the driver
                            // when the cards have no P2P path) into this slot
                            int self_dev = 0;
                            cudaGetDevice(&self_dev);
                            if (m.stage_live[sl]) cudaStreamWaitEvent(m.copy, m.used[sl], 0);
                            tl0 = clk_copy.record(m.copy);
                            tl_name = "copy peer";
                            cudaMemcpyPeerAsync(m.stage_dev[sl], self_dev, m.peer_ptr(peer_slot), m.peer_dev,
                                                (size_t) lay.blob_bytes(l), m.copy);
                            ++stats_.experts_dma;
                        } else if (m.src->pinned(l, e)) {
                            // DMA straight from the page-locked arena: the copy stream only waits for the slot
                            if (m.stage_live[sl]) cudaStreamWaitEvent(m.copy, m.used[sl], 0);
                            tl0 = clk_copy.record(m.copy);
                            cudaMemcpyAsync(m.stage_dev[sl], b, (size_t) lay.blob_bytes(l), cudaMemcpyHostToDevice, m.copy);
                            ++stats_.experts_dma;
                        } else {
                            // copied to a pinned buffer by the stager (waits only if it is behind), then DMA
                            const uint8_t* hb = nullptr;
                            {
                                timeline::Span wait_span("stager wait", job_of[j], (int64_t) j);
                                hb = m.stager->wait(job_of[j]);
                            }
                            if (m.stage_live[sl]) cudaStreamWaitEvent(m.copy, m.used[sl], 0);
                            tl0 = clk_copy.record(m.copy);
                            tl_name = b == nullptr ? "copy nvme" : "copy staged";
                            cudaMemcpyAsync(m.stage_dev[sl], hb, (size_t) lay.blob_bytes(l), cudaMemcpyHostToDevice, m.copy);
                            m.stager->issued_one(job_of[j], m.copy);
                        }
                        clk_copy.span(tl_copy, tl_name, tl0, clk_copy.record(m.copy), (int64_t) j, l);
                        cudaEventRecord(m.copied[sl], m.copy);   // every branch: the compute stream waits on it
                        m.stage_live[sl] = true;
                        stage_of[j] = sl;
                        stats_.ms_experts_host += ms_since(th);
                        ++stats_.experts_streamed;
                        return true;
                    };
                    // an MMQ group's products once its experts are gathered into the group slots; j = its last expert
                    auto mmq_products = [&](size_t j) {
                        const size_t q = j % MMQ_GROUP;
                        // the group's products, per sub-product (#34): its rows' q8 activations, gate/up, swiglu,
                        // H to q8_1, down into the layer's Dm rows
                        const size_t g = (j - q) / MMQ_GROUP;
                        const int ngx = (int) (q + 1);
                        pt.mark(kPfGemmGU, cs);
                        // the zeroed tail after the group's last expert (see MMQ_TAIL; a sub-product that ends earlier
                        // reads into the next expert's finite blocks)
                        cudaMemsetAsync(m.grp_gu + (size_t) ngx * mmq_gub, 0, MMQ_TAIL, m.cs);
                        cudaMemsetAsync(m.grp_d + (size_t) ngx * mmq_db, 0, MMQ_TAIL, m.cs);
                        for (size_t si = m.mmq_sub_first[g]; si < m.mmq_sub_first[g + 1]; ++si) {
                            const Impl::MmqSub& sb = m.mmq_subs[si];
                            mmq::ExpertRows a;
                            a.xtok = m.Xtok; a.xtok_rows = T; a.rows = m.src_dev + sb.r0; a.nr = sb.nr;
                            a.max_rows = sb.maxr; a.n = (int) (sb.q1 - sb.q0);
                            a.gu = m.grp_gu + sb.q0 * mmq_gub; a.gu_type = mmq_gt; a.gu_bytes = mmq_gub;
                            a.down = m.grp_d + sb.q0 * mmq_db; a.down_type = mmq_dt; a.down_bytes = mmq_db;
                            a.bounds = m.bounds_dev + sb.off; a.ids = m.ids_identity; a.n_embd = N; a.n_ff = 640;
                            a.interleaved = !lay.native;
                            a.xq = m.Xq; a.gu_out = m.GU; a.h = m.H; a.hq = m.Hq; a.dst = m.Dm + sb.r0 * N;
                            mmq::expert_rows_gate_up(*m.mmq_ctx, a, m.cs);
                            pt.mark(kPfGemmD, cs);
                            mmq::expert_rows_down(*m.mmq_ctx, a, m.cs);
                        }
                        pt.mark(kPfWaitHost, cs);
                    };
                    // one expert's products from its blob on the device; `slot` (a ring slot, or -1 for a resident
                    // expert) is released once the blob is read
                    auto compute = [&](size_t j, const uint8_t* blob_dev, int slot) -> bool {
                        const int32_t e = order[j];
                        pt.mark(kPfDequant, cs);
                        if (use_mmq) {
                            // gather the expert into its group slot (GGUF blocks, unchanged or converted)
                            const size_t q = j % MMQ_GROUP;
                            if (lay.native) {
                                const auto& f = lay.fmt[(size_t) l];
                                mmq::gather_native(blob_dev, blob_dev + f.up_off, mmq_gub / 2, blob_dev + f.down_off,
                                                   mmq_db, m.grp_gu + q * mmq_gub, m.grp_d + q * mmq_db, m.cs);
                            } else {
                                mmq::gather_strata_q2(blob_dev, m.grp_gu + q * mmq_gub, m.grp_d + q * mmq_db, m.cs);
                            }
                            if (slot >= 0) cudaEventRecord(m.used[slot], m.cs);
                            if (q + 1 < MMQ_GROUP && j + 1 < order.size()) {
                                pt.mark(kPfWaitHost, cs);
                                return true;
                            }
                            mmq_products(j);
                            return true;
                        }
                        const int q = (int) (j % DQ);
                        if (lay.native) {
                            // plan v0.3 P6: a native pack's layer, dequantized by llama.cpp's own formulas
                            const auto& f = lay.fmt[(size_t) l];
                            // #29: gate/up and down in one launch (byte-identical to the two it replaced)
                            strata::kernels::NativeExpertLayout L =
                                strata::kernels::native_expert_layout(f.gu_type, f.d_type, f.n_embd, f.n_ff);
                            L.up_off = f.up_off;
                            L.down_off = f.down_off;
                            strata::kernels::iq_dequant_expert_f16(L, blob_dev, m.dq_gu[q], m.dq_d[q], m.cs);
                        } else {
                            blob_dequant_f16(blob_dev, m.dq_gu[q], m.dq_d[q], m.cs);
                        }
                        if (slot >= 0) cudaEventRecord(m.used[slot], m.cs);
                        const int64_t o0 = m.off[(size_t) e], ne = m.cnt[(size_t) e];
                        pt.mark(kPfGemmGU, cs);
                        m.gemm.f16(m.Xs + o0 * N, m.dq_gu[q], m.GU + o0 * 1280, ne, 1280, N);
                        swiglu_interleaved(m.GU + o0 * 1280, m.Hh + o0 * 640, ne, m.cs);
                        pt.mark(kPfGemmD, cs);
                        m.gemm.f16(m.Hh + o0 * 640, m.dq_d[q], m.Dm + o0 * N, ne, N, 640);
                        pt.mark(kPfWaitHost, cs);
                        return true;
                    };
                    if (!stream_all) {
                        size_t staged = 0;
                        const size_t lookahead = STAGE - 1;
                        for (size_t j = 0; j < order.size(); ++j) {
                            while (staged < order.size() && staged <= j + lookahead) {
                                if (!stage_one(staged)) return false;
                                ++staged;
                            }
                            const int32_t e = order[j];
                            if (stage_of[j] < 0) {
                                ++stats_.experts_resident;
                                if (!compute(j, m.cache->device_slot(m.host_res[(size_t) l * m.g->n_expert + e]), -1)) return false;
                            } else {
                                pt.mark(kPfWaitCopy, cs);
                                cudaStreamWaitEvent(m.cs, m.copied[stage_of[j]], 0);
                                if (!compute(j, m.stage_dev[stage_of[j]], stage_of[j])) return false;
                            }
                        }
                    } else if (use_mmq && lay.native && group_gather_on) {
                        // #29, the streamed walk with one gather launch per MMQ group (was: a wait, a gather and a used
                        // record per expert, ~0.17 ms of host per expert).  A routed entry's gather is deferred until the
                        // group ends; its ring slot is released (used recorded, `consumed` published) only then, so the
                        // copy issuer can never overwrite a slot whose blob is not read yet.  The copy stream runs in
                        // entry order: waiting on the group's last streamed copy covers the ones before it.  A group that
                        // would span more entries than the ring holds gathers what it has first (the products still run
                        // per group: the gathers only fill slots).
                        size_t k = seq_start[(size_t) l];
                        const size_t kend = seq_start[(size_t) l + 1];
                        const auto& f = lay.fmt[(size_t) l];
                        const uint8_t* gsrc[MMQ_GROUP];
                        int gslot[MMQ_GROUP];
                        size_t g0 = 0, gn = 0;          // pending gathers: group positions [g0, g0 + gn)
                        size_t k_hold = SIZE_MAX;       // the first entry whose slot a pending gather still reads
                        size_t k_wait = SIZE_MAX;       // the last streamed entry of the pending gathers
                        auto publish = [&](size_t upto) {
                            consumed = upto;
                            if (use_issuer) a_consumed.store(consumed, std::memory_order_release);
                            else issue_until(consumed + (size_t) m.ring);
                        };
                        auto flush_gathers = [&]() {
                            if (gn == 0) return;
                            if (k_wait != SIZE_MAX) {
                                pt.mark(kPfWaitCopy, cs);
                                if (use_issuer && a_issued.load(std::memory_order_acquire) <= k_wait) {
                                    timeline::Span wait_span("wait issuer", (int64_t) k_wait, l);
                                    while (a_issued.load(std::memory_order_acquire) <= k_wait) std::this_thread::yield();
                                }
                                cudaStreamWaitEvent(m.cs, m.copied[k_wait % (size_t) m.ring], 0);
                            }
                            pt.mark(kPfDequant, cs);
                            mmq::gather_native_group(gsrc, (int) gn, f.up_off, f.down_off, mmq_gub / 2, mmq_db,
                                                     m.grp_gu + g0 * mmq_gub, mmq_gub, m.grp_d + g0 * mmq_db, mmq_db, m.cs);
                            for (size_t i = 0; i < gn; ++i)
                                if (gslot[i] >= 0) cudaEventRecord(m.used[gslot[i]], m.cs);
                            g0 += gn;
                            gn = 0;
                            k_hold = k_wait = SIZE_MAX;
                            publish(k);
                        };
                        auto release_to = [&](int32_t e_stop) {   // entries the routing did not pick: slot back at once
                            while (k < kend && expert_pos(l, seq[k].e, m.g->n_expert) < expert_pos(l, e_stop, m.g->n_expert)) {
                                cudaEventRecord(m.used[k % (size_t) m.ring], m.cs);
                                ++k;
                                if (k_hold == SIZE_MAX) publish(k);
                            }
                        };
                        for (size_t j = 0; j < order.size(); ++j) {
                            const int32_t e = order[j];
                            const size_t q = j % MMQ_GROUP;
                            if (q == 0) g0 = 0;
                            release_to(e);
                            if (k < kend && seq[k].e == e) {
                                // the pending gathers hold slots from k_hold on: the issuer can run ring entries past
                                // it, so this entry's copy exists only while it is within that reach
                                if (k_hold != SIZE_MAX && k + 2 > k_hold + (size_t) m.ring) flush_gathers();
                                if (k_hold == SIZE_MAX) k_hold = k;
                                gsrc[gn] = m.stage_dev[k % (size_t) m.ring];
                                gslot[gn] = (int) (k % (size_t) m.ring);
                                k_wait = k;
                                ++gn;
                                ++k;
                            } else {
                                ++stats_.experts_resident;
                                gsrc[gn] = m.cache->device_slot(m.host_res[(size_t) l * m.g->n_expert + e]);
                                gslot[gn] = -1;
                                ++gn;
                            }
                            if (q + 1 == MMQ_GROUP || j + 1 == order.size()) {
                                flush_gathers();
                                mmq_products(j);
                            }
                        }
                        release_to(m.g->n_expert);
                        if (k_hold == SIZE_MAX) publish(k);
                    } else {
                        // the streamed walk: this layer's entries [k, kend) in id order; an entry the routing did not
                        // pick only gives its slot back
                        size_t k = seq_start[(size_t) l];
                        const size_t kend = seq_start[(size_t) l + 1];
                        auto release_to = [&](int32_t e_stop) {
                            while (k < kend && expert_pos(l, seq[k].e, m.g->n_expert) < expert_pos(l, e_stop, m.g->n_expert)) {
                                cudaEventRecord(m.used[k % (size_t) m.ring], m.cs);
                                consumed = ++k;
                                if (use_issuer) a_consumed.store(consumed, std::memory_order_release);
                                else issue_until(consumed + (size_t) m.ring);
                            }
                        };
                        for (size_t j = 0; j < order.size(); ++j) {
                            const int32_t e = order[j];
                            release_to(e);
                            if (k < kend && seq[k].e == e) {
                                const int sl = (int) (k % (size_t) m.ring);
                                pt.mark(kPfWaitCopy, cs);
                                if (use_issuer && a_issued.load(std::memory_order_acquire) <= k) {
                                    timeline::Span wait_span("wait issuer", (int64_t) k, l);
                                    while (a_issued.load(std::memory_order_acquire) <= k) std::this_thread::yield();
                                }
                                cudaStreamWaitEvent(m.cs, m.copied[sl], 0);
                                if (!compute(j, m.stage_dev[sl], sl)) return false;
                                consumed = ++k;
                                if (use_issuer) a_consumed.store(consumed, std::memory_order_release);
                                else issue_until(consumed + (size_t) m.ring);
                            } else {
                                ++stats_.experts_resident;
                                if (!compute(j, m.cache->device_slot(m.host_res[(size_t) l * m.g->n_expert + e]), -1)) return false;
                            }
                        }
                        release_to(m.g->n_expert);
                    }
                    pt.mark(kPfCombine, cs);
                    if (split_l) {
                        if (!shared_expert()) return false;   // while the 4070 runs the routed experts
                        if (!split_output_up(m, l, T, err)) return false;
                        cudaStreamWaitEvent(m.cs, m.split->ev_done, 0);   // same card: the 4070's routed sum is in bo
                        moe_shared_finish(m.shared, m.sg, m.bo, T, m.cs);
                    } else {
                        moe_combine(m.Dm, m.slot_dev, m.w, m.shared, m.sg, m.bo, T, m.cs);
                    }
                    // debug: STRATA_DBG_NAN=1 reports the first layer of a chunk whose MoE produced non-finite values
                    if (static const bool dbg = std::getenv("STRATA_DBG_NAN") != nullptr; dbg) {
                        cudaStreamSynchronize(m.cs);
                        auto bad = [&](const float* d, int64_t n) {
                            std::vector<float> h((size_t) n);
                            cudaMemcpy(h.data(), d, (size_t) n * 4, cudaMemcpyDeviceToHost);
                            int64_t c = 0;
                            for (float v : h) c += !std::isfinite(v);
                            return c;
                        };
                        const int64_t gu_rows = use_mmq ? m.mmq_rows : T * K;
                        const int64_t bgu = bad(m.GU, gu_rows * 1280), bdm = bad(m.Dm, T * K * N), bbo = bad(m.bo, T * N);
                        const int64_t bh = m.H ? bad(m.H, m.mmq_rows * 640) : -1;
                        static int64_t reported = -1;
                        if ((bgu || bdm || bbo || bh > 0) && reported != stats_.chunks) {
                            reported = stats_.chunks;
                            std::fprintf(stderr, "strata dbg: layer %lld (mmq %d, types %d/%d, %zu experts): non-finite GU %lld "
                                         "H %lld Dm %lld bo %lld of T %lld\n", (long long) l, (int) use_mmq, mmq_gt, mmq_dt,
                                         order.size(), (long long) bgu, (long long) bh, (long long) bdm, (long long) bbo,
                                         (long long) T);
                        }
                    }
                }
                // ---- the hyper-connection write of this half; F-2: fused with the next half's norm when nothing else
                // touches R in between (not the stage's last half (upstream layer split), not before the PLE block of layer 1, not under a
                // control vector)
                const int64_t nl = half == 0 ? l : l + 1;
                const bool fuse = !gr_unfused() && nl < LE && !(half == 1 && nl == 1 && ple_on) &&
                                  !(half == 1 && strata::kernels::cvec().covers(l));
                const core::WeightRef* wnn = nullptr;
                if (fuse) {
                    const core::LayerView vn(*m.wt, nl);
                    wnn = need(vn, half == 0 ? "hc_ffn_norm.weight" : "hc_attn_norm.weight", err);
                    if (!wnn) return false;
                }
                if (wnn) {
                    gr_write_norm_rs(m.R, m.bo, m.inj, HC, (const float*) wnn->data, EPS, m.grs, m.xn16, T, m.cs);
                    normed = true;
                } else {
                    gr_write(m.R, m.bo, m.inj, HC, T, m.cs);
                }
                // #35 D7: chunk c+1 may start layer l - in a split layer only once this chunk's MoE is handed to the
                // 4070 (split_experts), or its trunk occupies this card and delays that hand-off (tlS: 37 ms a layer)
                if (wave && half == 0 && !split_on) wave->publish_attn(chunk_i, l, m.cs);
                if (half == 1 && strata::kernels::cvec().covers(l))   // --control-vector-scaled
                    strata::kernels::cvec_apply(m.R, l, T, D, nullptr, 0, nullptr, 0, false, m.cs);
            }
        }
        if (issuer.joinable()) {   // (xeno) issue_one counts the stream's stats itself, on whichever thread issues
            issuer_stop.store(true);
            issuer.join();
        }
        stats_.tokens += T;
        pt.layer = -1;
        pt.mark(kPfStart, cs);
        if (next_ != nullptr) {
            // the rows to the host buffer the next stage read two chunks ago (it has finished: waited below)
            float* h = m.hand[hand_buf];
            if (cudaMemcpyAsync(h, m.R, (size_t) T * D * 4, cudaMemcpyDeviceToHost, m.cs) != cudaSuccess ||
                cudaStreamSynchronize(m.cs) != cudaSuccess) {
                err = std::string("prefill: the layer split's hand-off: ") + cudaGetErrorString(cudaGetLastError());
                return false;
            }
            // this stage's state is at the chunk's end now (synced) and moves on with the next chunk below
            if (on_stage_chunk && !on_stage_chunk(p0 + T, err)) return false;
            if (next_run.valid() && !next_run.get()) { err = next_err; return false; }
            next_->hand_in_ = h;
            next_run = std::async(std::launch::async, [this, tokens, c0, T, p0, &next_err] {
                return next_->run(tokens + c0, T, p0, next_err);
            });
            hand_buf ^= 1;
            continue;   // the last stage reports the chunk (on_chunk)
        }
        if (const char* dump = std::getenv("STRATA_PREFILL_DUMP_R")) {   // debug: the final residuals, every 64th
            cudaStreamSynchronize(m.cs);                                  // position (A/B quality of this path)
            if (std::FILE* f = std::fopen(dump, c0 == 0 ? "wb" : "ab")) {
                std::vector<float> row((size_t) D);
                for (int64_t t = (64 - p0 % 64) % 64; t < T; t += 64) {
                    cudaMemcpy(row.data(), m.R + t * D, (size_t) D * 4, cudaMemcpyDeviceToHost);
                    const int64_t pos = p0 + t;
                    std::fwrite(&pos, sizeof pos, 1, f);
                    std::fwrite(row.data(), 4, row.size(), f);
                }
                std::fclose(f);
            }
        }
        if (on_chunk || on_stage_chunk) {
            const auto toc = Clock::now();
            {
                timeline::Span sync_span("chunk tail (sync)", c0, T);   // the GPU finishing the chunk's last layers
                if (cudaStreamSynchronize(m.cs) != cudaSuccess) {
                    err = std::string("prefill: ") + cudaGetErrorString(cudaGetLastError());
                    return false;
                }
            }
            const auto toc2 = Clock::now();
            if (wave && chunk_i > 0 && !wave->wait_tail(chunk_i - 1)) { err = "prefill: the other wave lane failed"; return false; }
            if (on_stage_chunk && !on_stage_chunk(p0 + T, err)) return false;
            timeline::Span on_chunk_span("on_chunk (drafter)", c0, T);
            if (on_chunk && !on_chunk(m.R, T, p0, err)) return false;
            host_sync_ms += std::chrono::duration<double, std::milli>(toc2 - toc).count();
            host_chunk_ms += ms_since(toc2);
        }
        if (wave) wave->publish_tail(chunk_i);
    }
    if (!wave || ((n - 1) / m.T) % 2 == m.wave_lane) {   // #35 D7: the lane that read the last chunk
        ss.ple_prev[0] = prev[0];
        ss.ple_prev[1] = prev[1];
    }
    if (next_run.valid() && !next_run.get()) { err = next_err; return false; }   // (upstream) the next layer-split stage
    if (std::getenv("STRATA_DBG_NAN") != nullptr) {   // debug: the state the prompt leaves for the token path
        cudaStreamSynchronize(m.cs);
        auto bad = [&](const float* d, int64_t n) {
            std::vector<float> h((size_t) n);
            cudaMemcpy(h.data(), d, (size_t) n * 4, cudaMemcpyDeviceToHost);
            int64_t c = 0;
            double mx = 0;
            for (float v : h) { c += !std::isfinite(v); if (std::isfinite(v)) mx = std::max(mx, (double) std::fabs(v)); }
            std::fprintf(stderr, " %lld non-finite (max |x| %.3g)", (long long) c, mx);
        };
        const int64_t last = (n - 1) % m.T;
        std::fprintf(stderr, "strata dbg: prompt end: last residual row");
        bad(m.R + last * D, D);
        if (ss.ple.ready()) { std::fprintf(stderr, "; PLE history"); bad(ss.ple.hist, (int64_t) strata::kernels::NG_HIST * strata::kernels::NG_HC_DIM); }
        std::fprintf(stderr, "; GDN state 0");
        bad(ss.gdn_state, 64 * 1024);
        std::fprintf(stderr, "\n");
    }
    if (cudaStreamSynchronize(m.cs) != cudaSuccess) {
        err = std::string("prefill: ") + cudaGetErrorString(cudaGetLastError());
        return false;
    }
    // (PR #121) an expert copy that failed on the copy stream surfaces here, not in the next request
    if (const cudaError_t cst = cudaStreamSynchronize(m.copy); cst != cudaSuccess) {
        err = std::string("prefill: expert copy stream: ") + cudaGetErrorString(cst);
        return false;
    }
    stats_.ms_total += ms_since(t_start);
    if (timeline::enabled()) {
        clk_compute.mark(tl_compute, nullptr, cs);
        cudaStreamSynchronize((cudaStream_t) m.copy);
        clk_compute.resolve(true);
        clk_copy.resolve(true);
    }
    if (pt.on) {
        pt.fold();
        double total = 0.0;
        for (double v : pt.ms) total += v;
        std::string line;
        char b[96];
        for (int i = 0; i < kPfCount; ++i) {
            if (pt.ms[i] <= 0.0) continue;
            std::snprintf(b, sizeof b, " %s %.0f (%.1f%%)", kPfNames[i], pt.ms[i], total > 0 ? 100.0 * pt.ms[i] / total : 0.0);
            line += b;
        }
        std::fprintf(stderr, "strata prefill timing: %lld tokens, GPU timeline %.0f ms, wall %.0f ms, host staging %.0f ms:%s\n",
                     (long long) n, total, ms_since(t_start), stats_.ms_experts_host, line.c_str());
        std::fprintf(stderr, "strata prefill timing: host: chunk setup (PLE rows, the expert stream plan) %.0f ms, "
                             "waiting for each chunk %.0f ms, after each chunk (the draft layer, progress) %.0f ms, "
                             "PLE %.0f ms\n", host_setup_ms, host_sync_ms, host_chunk_ms, stats_.ms_ple);
    }
    if (std::getenv("STRATA_STATE_HASH_GDN") != nullptr) {   // debug: the GDN states as the prompt path leaves them
        cudaStreamSynchronize(m.cs);
        std::vector<uint8_t> b((size_t) gdn_floats * 4);
        std::string line;
        char h[8];
        for (int64_t i = 0; i < ss.gdn_alloc; ++i) {
            cudaMemcpy(b.data(), ss.gdn_state + (size_t) i * gdn_floats, b.size(), cudaMemcpyDeviceToHost);
            uint64_t x = 1469598103934665603ull;
            for (uint8_t c : b) x = (x ^ c) * 1099511628211ull;
            std::snprintf(h, sizeof(h), "%04llx ", (unsigned long long) (x & 0xffff));
            line += h;
        }
        std::fprintf(stderr, "strata prefill: GDN_HASH %s\n", line.c_str());
    }
    wave_fail.done = true;
    return true;
}

}  // namespace strata::prefill
