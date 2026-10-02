// include/strata/core/expert_source.hpp - P2.S3/P2.S5: the expert pool's adapter to the host loop.
//
// `session_loop` publishes, per layer, the NORMED activation `x_f`, the ten routed expert ids and their router
// weights, and expects `k x n_embd` floats back.  This is the piece that turns those four things into an
// `ExpertPool::run` call.  Nothing here is clever, and that is the point: the pool, the kernel and the loop are
// each already verified, so this file has exactly one job - get the CONTRACT between them right.
//
// THE CONTRACT, and each clause is a way to be wrong:
//
//   1. **`x_f` IS THE SAME PINNED BUFFER EVERY LAYER.**  Its ADDRESS never changes, so the `ActQ` cannot be
//      cached by pointer - a cache keyed on `x_f` would quantize layer 0 and reuse it for all 47 remaining
//      layers, which is a finite, plausible, completely wrong token.  There is no cache here at all: the
//      conversion is one 2560-element pass against a 0.3 ms/layer budget, and a correct answer is worth more
//      than the microseconds.
//   2. **THE POOL DOES NOT APPLY THE ROUTER WEIGHT.**  `moe_combine` (`src/core/layer.cpp:479`) sums
//      `w[i] * parts[i]` on the device.  `ExpertJob::weight` is a diagnostic field; setting it here would
//      apply the weight twice, which is invisible in a single layer and compounds over 48.
//   3. **THE KERNEL'S GEOMETRY IS FIXED BY THE ARTIFACT** (`H = 2560`, `FF = 640`, `BLOB = 1,382,400`).  A
//      different `n_embd` or a different expert width is REFUSED rather than mis-indexed, because the blob's
//      internal offsets are compile-time constants and reading a 640-wide expert as a 2560-wide one walks off
//      the end into the next expert's bytes without faulting.
#pragma once

#include "strata/core/expert_cache.hpp"
#include "strata/core/hit_hook.hpp"
#include "strata/kernels/cpu/pool.hpp"

#include <atomic>
#include <cstdint>
#include <cstddef>
#include <utility>
#include <cstdio>
#include <fstream>
#include <mutex>
#include <functional>
#include <string>
#include <vector>

namespace strata::core {

class RemoteExperts;

namespace detail {

/// Sentinel used by the pure complement planner for a blob that remains in the mmap fallback.
inline constexpr uint64_t kNoCacheComplement = ~uint64_t{0};

/// Required cgroup-v2 usage counters for the conservative cache-reclaim allowance.
struct CgroupMemoryStat {
    uint64_t current = 0;
    uint64_t inactive_file = 0;
    uint64_t file_dirty = 0;
    uint64_t file_writeback = 0;
    bool valid = false;
};

/// Calculate additional bytes under a finite cgroup limit after reclaiming only clean inactive file cache.
/// Returns false when the required memory.stat counters were unavailable.
bool cgroup_available_bytes(uint64_t limit, const CgroupMemoryStat& stat, uint64_t& bytes);

/// Build compact offsets for experts absent from both the primary GPU cache and an optional second GPU tier.
/// Kept CPU-only so selection and byte accounting can be tested without initializing a GPU.
bool make_cache_complement_plan(
    int64_t n_layers, int64_t n_expert, const std::vector<uint64_t>& layer_blob_bytes,
    const std::vector<std::pair<int32_t, int32_t>>& primary_gpu_pairs,
    const std::vector<std::pair<int32_t, int32_t>>& additional_gpu_pairs,
    std::vector<uint64_t>& offsets, uint64_t& bytes, std::string& err);

/// Resolve one blob through the compact copy when present, otherwise preserve its exact mapped-file fallback.
const uint8_t* cache_complement_blob_or_fallback(
    size_t index, const std::vector<uint64_t>& offsets, const uint8_t* complement_host,
    const uint8_t* mapped_fallback);

/// The resident RAM mode: which GPU-cache slots' experts are kept in RAM too.  The prompt path lends the cache's
/// LAST slots (from `lend_from` on; a short prompt lends only the last few), and a lent slot's expert is streamed
/// from RAM during the prompt and copied back into its slot after it.  `base_bytes` (every expert no slot holds)
/// must fit `budget`; slots are then added from the end down to `lend_from` while they still fit.  Returns the
/// first slot kept in RAM (`slot_bytes.size()` = none), or -1 when `base_bytes` alone exceeds `budget`.
int64_t choose_resident_keep_from(const std::vector<uint64_t>& slot_bytes, uint64_t base_bytes, uint64_t budget,
                                  int64_t lend_from);

/// The adaptive tier swapped `in` into a GPU slot and `out` out of it: `out` takes `in`'s place in the compact copy
/// (the caller copies out's bytes there).  False, and nothing changed, unless `in` is in the copy and `out` is not.
bool exchange_cache_complement(std::vector<uint64_t>& offsets, size_t in, size_t out);

}  // namespace detail


class SecondaryArena;
class SecondaryRunner;

/// Where one routed expert's bytes come from.
///
/// Phase 2 has NO cache (`phase-2-correct-engine.md`: hit rate `h = 0`), so the only implementation is a
/// file-backed reader.  The interface exists anyway because Phase 3 replaces exactly this object with the VRAM
/// cache, and because a test can supply an in-memory source without a 34 GB artifact.
class ExpertSource {
public:
    virtual ~ExpertSource() = default;

    /// The expert-layout blob for `(layer, expert)`, or nullptr if it cannot be produced.
    ///
    /// The pointer only has to stay valid until the next `blob()` call: with `h = 0` every expert is computed
    /// immediately and nothing is retained.  A CACHING source must return pointers into the cache, not into a
    /// reused staging buffer - otherwise the pool would read the next expert's bytes while computing this one.
    virtual const uint8_t* blob(int64_t layer, int64_t expert) = 0;

    /// Blobs touched, for the driver to report.  A source that does not count returns 0.
    virtual int64_t reads() const { return 0; }

    /// Plan v0.3 P5: whether `blob(layer, expert)` lies in page-locked, CUDA-registered memory, so an
    /// asynchronous host-to-device copy can DMA it directly (no staging copy on the CPU).
    virtual bool pinned(int64_t layer, int64_t expert) const { (void) layer; (void) expert; return false; }

    /// Called once before the first expert of a layer.  A source that reads from disk wants to start the read
    /// here so it overlaps the quantisation, and a prefetching source in Phase 3 wants the ids.
    virtual void begin_layer(int64_t layer, const int32_t* ids, int64_t k) { (void) layer; (void) ids; (void) k; }
    /// Plan v0.3 P6: the DEVICE address of a pinned, mapped blob (the GPU can read it over PCIe), or null.
    virtual const uint8_t* device_alias(int64_t layer, int64_t expert) const { (void) layer; (void) expert; return nullptr; }
    /// #11 capacity mode: whether the host holds the expert now (a source without an NVMe tier always does).
    virtual bool resident(int64_t layer, int64_t expert) const { (void) layer; (void) expert; return true; }
    /// #11: bring an NVMe-tier expert into the host tier and return its bytes (null on a read failure: never stale
    /// data). Evicts the lowest-scored host expert outside `avoid_layer` when over capacity.
    virtual const uint8_t* materialize(int64_t layer, int64_t expert, int64_t avoid_layer, std::string& err) {
        (void) layer; (void) expert; (void) avoid_layer; err = "this expert source has no NVMe tier"; return nullptr;
    }
    /// #11: bring several NVMe-tier experts of one layer in with overlapped reads (one wait for the layer).
    virtual bool materialize_batch(int64_t layer, const int32_t* experts, int n, std::string& err) {
        for (int i = 0; i < n; ++i) if (!materialize(layer, experts[i], layer, err)) return false;
        return true;
    }
    /// #95: materialize_batch in two halves, so the caller can run its resident experts while the reads are in flight.
    /// A source with a real split (ArenaExpertSource): between them the batch's experts are not resident and the host
    /// tier is held; end publishes them.  This default does the whole batch in begin, and end has nothing to do.
    virtual bool materialize_begin(int64_t layer, const int32_t* experts, int n, std::string& err) {
        return materialize_batch(layer, experts, n, err);
    }
    virtual bool materialize_end(std::string& err) { (void) err; return true; }
    /// #11: the expert's bytes into `dst` straight from the pack, admitting nothing (the prompt path).
    virtual bool read_into(int64_t layer, int64_t expert, uint8_t* dst, std::string& err) {
        (void) layer; (void) expert; (void) dst; err = "this expert source cannot read the pack"; return false;
    }
    /// Whether the verify window may give the GPU a PCIe share of this layer's misses at all (each expert is still
    /// checked with `pinned`).  The arena answers per layer through its expert 0; the resident RAM mode's compact
    /// copy has no expert 0 when the GPU cache holds it, so it answers for the whole copy.
    virtual bool pcie_layer(int64_t layer) const { return device_alias(layer, 0) != nullptr; }
};

/// Plan v0.3 P6: what the GPU computes in a verify window's layer, written by the pool (mapped host memory) right
/// after the ring and published before the CPU starts its own share.  Groups of entries that share a blob; the
/// blob is a VRAM slot or a pinned host blob read over PCIe.
struct GpuPlanSink {
    int32_t* counts = nullptr;             ///< [0] groups, [1] entries
    int32_t* start = nullptr;              ///< cap + 1
    int32_t* dst = nullptr;                ///< cap: the entry's row of parts (token * k + j)
    int32_t* tok = nullptr;                ///< cap: the entry's token
    unsigned long long* ptr = nullptr;     ///< cap: the group's blob, device address
    /// Plan v0.3 P6 (DMA): the PCIe share is a second list of groups - `ptr2[q]` is staging slot q, `start2` indexes
    /// the same dst/tok entries - computed by the GPU after `fetch` has copied their blobs into staging with the
    /// copy engine.  counts[0] = VRAM groups, counts[1] = all entries, counts[2] = PCIe groups.
    unsigned long long* ptr2 = nullptr;
    int32_t* start2 = nullptr;
    unsigned long long staging = 0;
    int64_t staging_cap = 0;
    int64_t cap = 0;
    void (*publish)(void* ctx) = nullptr;
    /// Starts the DMA copies of `n` host blobs (pinned) into staging slots 0..n-1 and signals the GPU when they land.
    void (*fetch)(void* ctx, const uint8_t* const* src, int n, size_t bytes) = nullptr;
    void* ctx = nullptr;
    /// Plan v0.3 P6: how a PCIe group reaches the GPU.  0 = the copy engine stages it (DMA, `fetch`); 1 = the grouped
    /// kernel reads the mapped arena directly; 2 = a copy kernel stages it inside the graph.  For 1 and 2 `ptr2`
    /// holds the arena's device alias.
    int pcie_mode = 0;
};

/// The adapter's own state.  One per session, reused every layer so the token path allocates nothing (P2.T10).
struct ExpertDispatch {
    strata::kernels::cpu::ExpertPool* pool = nullptr;
    ExpertSource* src = nullptr;
    RemoteExperts* remote[3] = {}; ///< optional CUDA1..3 tiers for otherwise CPU-served rows
    int remote_count = 0;
    int64_t n_expert = strata::kernels::cpu::NE;

    /// Optional routing trace (--route-trace): per verify-window layer, int16 layer, n_tok, k, then n_tok*k
    /// int16 expert ids (-1 = none), little-endian, appended in dispatch order.
    std::FILE* route_trace = nullptr;

    /// Counters, for the driver to report rather than for control flow.
    int64_t layers = 0;
    int64_t experts = 0;
    int64_t missing = 0;

    // ================================ R4: THE VRAM TIER, MEASURED BEFORE IT IS USED ================================
    //
    // **THE CACHE IS CONSULTED AND FILLED HERE, AND NOTHING IS COMPUTED FROM IT YET.**  That is deliberate and
    // it is `R4-design-note.md` §7 step 2: the *dispatch* is measured on its own before any kernel is written,
    // because a hit rate measured after the kernel exists cannot say whether a disappointing result was the
    // policy, the split or the kernel.
    //
    // So every expert is still computed by the CPU, exactly as before, and the run is numerically identical
    // with the cache on or off.  What changes is that the engine now reports **h on its own routing, on a real
    // workload** - which is the number `R4.1` asks for and which until now existed only from offline traces.
    //
    // The policy is compulsory-miss: the first time `(layer, expert)` is routed, if a slot is free it is
    // admitted and filled from the arena.  No eviction, because eviction policy is the measured question
    // (`R4.1`'s LFU-decay vs LRU sweep) and a placeholder would set the hit rate everything is sized against.
    ExpertCache* cache = nullptr;
    int64_t cache_hits = 0;      ///< lookups already resident
    int64_t cache_admitted = 0;  ///< lookups that took a slot
    int64_t cache_refused = 0;   ///< lookups with no slot free (the cache is full)
    void* cache_stream = nullptr;
    const char* cache_fail = nullptr;

    // ================================ R4.2c: THE HITS GO TO THE GPU ================================
    //
    // **THE SPLIT IS BY ROUTER INDEX, AND THAT IS THE WHOLE TRICK.**  A layer routes ten experts; the resident
    // ones are computed on the GPU and the rest on the CPU, and both answers have to end up in `parts` at the
    // index the ROUTER gave them, because that is the order `moe_combine` weights against.  So the CPU's `out`
    // rows are zeroed for the hits, and each hit carries its routed index to the kernel as `dst`.
    //
    // Everything below is per-session state rather than per-call, so the token path allocates nothing (P2.T10).
    const uint8_t* cache_base = nullptr;   ///< the slot arena on the DEVICE
    int64_t cache_blob = 0;                ///< bytes per slot
    const uint64_t* cache_slot_off = nullptr;   ///< plan v0.3 P6: per-slot offsets when the slots differ in size
    void* hit_scratch = nullptr;           ///< `moe_hit_grouped_scratch_bytes(K, ...)`
    float* parts_out = nullptr;            ///< the graph's `parts` buffer, on the device
    /// Where the GPU's hits land, `K x n_embd`, DEVICE and separate from `parts_out` on purpose: see
    /// `HitPhase`.  Zeroed by the hit path each layer before the kernel writes it.
    float* hit_out = nullptr;
    int64_t parts_elems = 0;               ///< `K * n_embd`, the length of both buffers
    const float* mixed = nullptr;          ///< the layer's normed activation, for the hit kernel's Q8_0
    uint8_t* x_q8_0_hit = nullptr;         ///< `(n_embd/32) * 34` bytes, its own buffer
    /// **R4.2h: THE CPU's fp32 ACTIVATION SCALES, `(n_embd/32)` FLOATS.**  Without them the GPU's hits are
    /// computed with the `block_q8_0`'s fp16 `d` while the CPU's misses use the fp32 `ActQ::scale`
    /// (`cpu/expert.cpp:92`) - **4.761e-04 relative on 80 of 80 chunks**, measured with both real
    /// implementations linked in `bench/micro/act_quant_parity.cu`.  That disagreement is why turning the
    /// cache on changed the generated tokens.  Required whenever the hit path runs.
    float* x_q8_0_hit_scale = nullptr;
    int32_t* d_slot = nullptr;             ///< device, K entries
    int32_t* d_dst = nullptr;              ///< device, K entries
    std::vector<int32_t> h_slot, h_dst;    ///< host staging, sized at session setup
    /// **PER ROUTER INDEX, DECIDED IN `Launch` AND CONSUMED BY THE POOL.**  The two callbacks share it
    /// so the decision is made exactly once, on this layer's ids, and neither side can re-decide it.
    std::vector<uint8_t> is_hit;
    bool decided = false;
    /// Plan v0.3 P4 token graph: the STATIC residency table (`n_layers x n_expert`, slot or -1), the host's copy
    /// of what the device hit path reads.  When set, the pool leaves a resident expert's row at zero (the GPU
    /// computes it) without any `Launch` callback.
    const int32_t* host_res = nullptr;
    /// Plan v0.3 P4: split every expert by rows across the pool's threads (default on; A/B `--no-split-rows`).
    bool split_rows = true;

    // ================================ R4.2d: DID THE GPU ACTUALLY START? ================================
    //
    // **THE HIT PATH IS 0.91 ms SLOWER AND THE ONLY EXPLANATION LEFT IS THAT IT NEVER OVERLAPS.**  The kernel
    // is 159 GB/s against the CPU's 35, the CPU's half of the drain falls 6 ms, and none of it reaches the
    // token - which is what it would look like if the enqueued hit kernels did not BEGIN until the host next
    // entered the driver, i.e. after `pool()` returned.  That is the third sighting of this driver behaving
    // that way (rounds 195/287 on the doorbell, round 36 on the head and sampler) and it decides whether R4 can
    // pay at all, so it gets measured rather than argued.
    //
    // `hit_done` is recorded on the stream straight after the hit kernel.  `Combine` - which runs after the
    // pool - queries it: SUCCESS means the GPU finished while the CPU was working, NOT-READY means it had not.
    // Same shape as the doorbell's `rings_mid_graph`, and for the same reason.
    void* hit_done = nullptr;      ///< cudaEvent_t, created at session setup
    int64_t hit_ready = 0;         ///< layers where the hit work was DONE by the time the pool returned
    int64_t hit_late = 0;          ///< layers where it was not
    /// The candidate fix, as an A/B arm: enter the driver once right after the hit launch.  If submission is
    /// lazy, this starts the GPU work before the pool instead of after it.
    bool hit_poke = false;
    bool hit_cpu_order = false;    ///< experimental CPU-order GPU expert arithmetic; opt-in only
    int64_t n_hits = 0;                    ///< this layer's hits
    /// Set by `Launch` and consumed by `Combine`, so a `Combine` with no `Launch` in front of it cannot
    /// add a stale buffer into `parts`.
    bool hit_pending = false;
    const char* hit_fail = nullptr;

    /// Whether the hit path is wired up.  All of it or none of it: a half-configured hit path would compute
    /// some experts twice and others not at all, which is a wrong token rather than an error.
    bool hits_ready() const {
        return cache != nullptr && cache_base != nullptr && parts_out != nullptr && hit_out != nullptr &&
               mixed != nullptr &&
               hit_scratch != nullptr && x_q8_0_hit != nullptr && x_q8_0_hit_scale != nullptr && d_slot != nullptr && d_dst != nullptr;
    }

    std::vector<strata::kernels::cpu::ExpertJob> jobs;
    strata::kernels::cpu::ActQ act;
    /// Plan v0.3 P6 verify window: one quantized activation per token, the multi-token jobs, and the expert ->
    /// job map (reset after every layer).
    std::vector<strata::kernels::cpu::ActQ> act_multi;
    /// Plan v0.3 P6: a native pack's per-token activations (MAXT x kNativeActBytes).
    std::vector<uint8_t> nact_multi;
    std::vector<strata::kernels::cpu::ExpertJobMulti> jobs_multi;
    std::vector<int16_t> job_of;
    /// Plan v0.3 P6: the verify window's GPU plan (VRAM hits + the PCIe share of the misses); `pcie_num`/256 of
    /// each layer's distinct missed experts (the last ones in routing order) are read by the GPU over PCIe.
    GpuPlanSink* plan = nullptr;
    SecondaryRunner* secondary_runner = nullptr; ///< optional device-1 Q2_0 partials in native verify windows
    const SecondaryArena* secondary_weights = nullptr;
    const int32_t* secondary_res = nullptr; ///< n_layers x n_expert, -1 when not resident
    int64_t secondary_entries = 0, secondary_groups = 0;
    /// Verify-window routed entries by where they were computed: [0] primary VRAM hit, [1] secondary (4070 SUPER)
    /// tier, [2] GPU PCIe read, [3] CPU pool.  Their sum is every routed (token, expert) entry of those windows.
    int64_t tier_entries[4] = {0, 0, 0, 0};
    std::string secondary_fail; ///< owns dynamic device-1 error text while `fail` points to it
    std::string nvme_fail;      ///< #62: the NVMe reader's error text while `fail` points to it
    int pcie_num = 0;
    int64_t pcie_experts = 0;      ///< distinct experts the GPU read over PCIe in verify windows
    double ms_plan = 0, ms_actq = 0, ms_jobs = 0, ms_run = 0;   ///< verify-window dispatch sections
    double ms_cpu_pool = 0, ms_secondary_finish = 0; ///< non-overlapping parts of ms_run
    /// Plan v0.3 P6: decayed routing counts per (layer, expert) during decode (sized by the caller; empty = off),
    /// which the driver uses to swap the most-routed missing experts into the VRAM tier between rounds.
    std::vector<float> usage;
    int64_t multi_misses = 0;      ///< distinct (layer, expert) pairs the CPU computed in verify windows
    int64_t multi_entries = 0;     ///< routed (token, expert) entries the CPU served in verify windows
    /// Set when `dispatch` could not produce an answer.  The loop itself has no error channel, so this is
    /// where a source failure surfaces: the driver checks it after `session_loop` returns rather than the
    /// engine computing from a half-filled `parts`.
    ///
    /// **`failed` LATCHES AND `fail` IS ONLY THE MESSAGE.**  Clearing the message must not re-arm the adapter:
    /// a session that failed at layer 5 has already fed `moe_combine` whatever `parts` held, so every layer
    /// after it is built on a hole - and resuming into a plausible-looking token is the exact outcome this
    /// whole mechanism exists to prevent.  `failed` is what the short-circuit reads.
    bool failed = false;
    const char* fail = nullptr;
    int64_t fail_layer = -1;
    int64_t fail_expert = -1;
};

/// `strata::core::PoolFn`, exactly.  Silent on failure BY SIGNATURE - see `ExpertDispatch::fail`.
void expert_pool_dispatch(void* user, const float* x_f, const int32_t* ids, const float* weights, int64_t n_embd,
                          int64_t k, float* out);

/// Plan v0.3 P6: the pool for a verify window of `n_tok` tokens.  `x_f` is (n_tok, n_embd), `ids` (n_tok, k) and
/// `out` (n_tok * k, n_embd).  Each distinct missed expert is computed once for all the tokens routed to it;
/// resident experts' rows are zeroed (the GPU adds them).  Requires `host_res` (the token-graph residency).
void expert_pool_dispatch_multi(ExpertDispatch& d, const float* x_f, const int32_t* ids, int64_t n_tok, int64_t k,
                                float* out);

/// **THE HITS, LAUNCHED AFTER THE MISSES ARE STAGED AND BEFORE `post[l]`.**  Same shape as `PoolFn` and for the
/// same reason: `session_loop` owns the ORDER and this owns the work, so the loop needs to know nothing about
/// expert caches.  Returns immediately when there are no hits, which is every layer until the cache is warm.
///
/// It must run AFTER the host has copied the misses into `parts_dev` (it writes into the same buffer, on rows
/// the CPU zeroed) and BEFORE `post[l]` (which reads it).  Both are stream-ordered on the loop's own stream.
void expert_hit_run(void* user, void* stream, HitPhase phase, const int32_t* ids, int64_t k);
/// The pool half of the same decision; see `ExpertDispatch::is_hit`.

/// **PHASE 2'S ONLY SOURCE: `experts.bin`, memory-mapped, no cache.**
///
/// `experts.bin` is 33,973,862,400 B and `BLOB` is 1,382,400, so it holds exactly `48 x 512 = 24,576` blobs and
/// the index is `layer * 512 + expert` with NO padding.  A mapping is therefore the whole implementation: the
/// blob pointer is base plus a multiply, and the page fault that follows is the read.
///
/// **AND THAT IS NOT AS SLOW AS IT SOUNDS, WHICH IS THE POINT.**  The expert set is 34 GB and this machine has
/// 64 GB of DDR5, so a warm OS page cache holds ALL of it - the second token onward is a DRAM read at the
/// measured 44.14 GB/s, not a disk read.  The first token's 663.6 MB comes off the disk and is the cold-path
/// number; `benches` should report the two separately rather than blending them.
///
/// IT IS NOT AN LRU, AN LFU OR A PREFETCHER.  Phase 3 replaces this object with those.  Phase 2 is hit rate
/// `h = 0` on purpose, so that the CPU path and the host round trip are exercised on every layer of every
/// token - which is the only way they get debugged before speed matters.
class FileExpertSource : public ExpertSource {
public:
    FileExpertSource() = default;
    ~FileExpertSource() override;
    FileExpertSource(const FileExpertSource&) = delete;
    FileExpertSource& operator=(const FileExpertSource&) = delete;

    /// Maps `<pack_dir>/experts.bin` and checks its size against the loaded expert layout.  Canonical packs use
    /// `n_layers * n_expert * BLOB`; native packs use their variable per-layer blob sizes and offsets.
    ///
    /// The size check is not a formality: a short file would fault at the END of a long sequence, and an
    /// over-long one means the pack is not the one the geometry came from.  Refuses with the two numbers.
    bool open(const std::string& pack_dir, int64_t n_layers, int64_t n_expert, std::string& err);
    /// Pin a compact host mirror of experts absent from a fully filled static GPU cache. The mmap remains open
    /// as a fallback for later cache reloads. This is opt-in because the complement may still be a large allocation.
    ///
    /// The resident RAM mode (`--resident-experts`, `--resident-cpu-experts`):
    ///   - `pin`: page-locked and mapped (`cudaHostAlloc`), so the prompt path copies it by DMA and the verify
    ///     window may read a share of the misses over PCIe; when the driver refuses, ordinary memory locked in the
    ///     working set instead.  `pin = false` is ordinary pageable memory (the ROCm arm: large pinned allocations
    ///     can fail there, and it is what the HIP measurements used).
    ///   - `lend_from_slot` >= 0: the GPU-cache slots from there to the end are the prompt path's lend region; their
    ///     experts are kept in RAM too, from the last slot down, as far as `available RAM - headroom_bytes` allows
    ///     (a lent slot's expert is streamed during the prompt and copied back after it).
    ///   - the rest (the experts no slot holds) must fit that budget, or nothing is allocated and this returns false.
    bool pin_cache_complement(
        const ExpertCache& cache, std::string& err, bool pin = true,
        const std::vector<std::pair<int32_t, int32_t>>& additional_gpu_pairs = {}, int64_t lend_from_slot = -1,
        uint64_t headroom_bytes = 8ull << 30);
    void close();

    bool mapped() const { return base_ != nullptr; }
    int64_t blobs() const { return blobs_; }
    uint64_t pinned_bytes() const { return complement_pinned_ ? complement_bytes_ : 0; }
    uint64_t resident_bytes() const { return complement_bytes_; }
    bool complement_pinned() const { return complement_pinned_; }
    bool complement_ready() const { return complement_ready_; }
    uint64_t locked_bytes() const { return complement_locked_; }
    /// Lend-region slots whose experts the compact copy holds (the last ones of the cache).
    int64_t resident_lent_slots() const { return complement_lent_slots_; }

    // ---- the resident RAM mode and the adaptive tier.  A swap puts `in` (held here) into a GPU slot and evicts
    // `out` (held only by that slot).  Before the slot is overwritten the caller copies it back into an exchange
    // buffer and calls `stage_exchange`: `out` is then read from that buffer, and `in` still from here (the CPU
    // computes both until the swap lands).  Once the slot copy has landed, `commit_exchanges` moves `out` into
    // `in`'s place, so the copy keeps holding exactly the experts the GPU does not - with no read of the file.
    /// Whether the compact copy holds `(layer, expert)`.
    bool has_resident(int64_t layer, int64_t expert) const;
    /// Host room for `n` evicted blobs (page-locked when possible).  Idempotent for the same or a smaller `n`.
    bool reserve_exchanges(int64_t n, std::string& err);
    int64_t exchange_capacity() const { return xstage_cap_; }
    uint8_t* exchange_buffer(int64_t q) const;
    /// Requires `has_resident(layer, in)`, `!has_resident(layer, out)` and `exchange_buffer(q)` holding out's blob.
    bool stage_exchange(int64_t layer, int64_t in, int64_t out, int64_t q);
    /// After the GPU copies of every staged swap have landed.  Returns how many exchanges were applied.
    int64_t commit_exchanges();
    int64_t exchanges() const { return exchanges_; }
    /// With the compact copy ready: blobs read from the mapped file since (what the plain mmap mode may read from
    /// the SSD).  0 in a steady resident mode; lend-region experts that did not fit the RAM count here.
    int64_t file_reads() const { return file_reads_.load(std::memory_order_relaxed); }

    const uint8_t* blob(int64_t layer, int64_t expert) override;
    bool pinned(int64_t layer, int64_t expert) const override;
    const uint8_t* device_alias(int64_t layer, int64_t expert) const override;
    bool pcie_layer(int64_t layer) const override;

    /// Blobs touched, for the driver to report.  With `h = 0` this is `48 * k` per token and the number is only
    /// interesting once Phase 3 makes it not so.
    int64_t reads() const override { return reads_; }

private:
    const uint8_t* mapped_blob(int64_t layer, int64_t expert) const;
    static constexpr uint64_t kNoComplement = detail::kNoCacheComplement;
    const uint8_t* base_ = nullptr;
    int64_t blobs_ = 0;
    int64_t n_layers_ = 0;
    int64_t n_expert_ = 0;
    uint64_t mapped_bytes_ = 0;
    std::vector<uint64_t> layer_offsets_, layer_blob_bytes_;
    void* complement_arena_ = nullptr;
    const uint8_t* complement_host_ = nullptr;
    const uint8_t* complement_device_ = nullptr;
    uint64_t complement_bytes_ = 0;
    std::vector<uint64_t> complement_offsets_;
    bool complement_pinned_ = false;
    bool complement_ready_ = false;
    uint64_t complement_locked_ = 0;          ///< bytes held in the working set (pin refused)
    int64_t complement_lent_slots_ = 0;
    std::vector<const uint8_t*> override_;    ///< staged exchanges: an evicted expert read from its exchange buffer
    struct Exchange { size_t in, out; int64_t q; uint64_t bytes; };
    std::vector<Exchange> staged_;
    uint8_t* xstage_ = nullptr;               ///< exchange buffers, `xstage_cap_ x xstage_blob_`
    bool xstage_pinned_ = false;
    int64_t xstage_cap_ = 0;
    uint64_t xstage_blob_ = 0;
    int64_t exchanges_ = 0;
    std::atomic<int64_t> file_reads_{0};
    int64_t reads_ = 0;
#if defined(_WIN32)
    void* file_ = nullptr;
    void* mapping_ = nullptr;
#else
    int fd_ = -1;
#endif
};

// ================================ THE RESIDENT ARENA (R2.1) ================================
//
// **THE MMAP ABOVE IS THE REVIEW'S FINDING C1 AND IT IS WORTH 2.2x.**  Read the comment on `FileExpertSource`
// about the page cache holding all 34 GB: that is true of the expert file ALONE, and it is not what this engine
// does.  The PLE/n-gram shard is a 26.8 GB file that the engine also maps, so 34 GB of experts plus 26.8 GB of
// n-gram is 60.8 GB of mapped, file-backed pages on a 63 GB machine - and file-backed pages are exactly the ones
// the OS drops from the standby list when it wants memory.  The expert stream then re-faults from disk.
//
// Measured, this round: the pool runs at **~19 GB/s in the engine against 42.8 GB/s in `bench/micro/cpu_s2.cpp`
// on the same machine, reading the same 34 GB**.  The micro does `std::fread` into a heap arena and reads
// ordinary (anonymous, resident) memory; the engine reads `MapViewOfFile`.  That is the whole difference, and it
// is why this class exists.
//
// It reads `experts.bin` into a `PinnedArena` once at startup, so the expert stream comes from anonymous memory
// the OS has no cheaper reason to evict.  `PinnedArena` also tries `cudaHostRegister`, which the GPU needs for
// Phase 3's cache fills and the CPU/PCIe miss split - but registration is best-effort and reported, not assumed.
class ArenaExpertSource : public ExpertSource {
public:
    ArenaExpertSource() = default;
    ~ArenaExpertSource() override;
    ArenaExpertSource(const ArenaExpertSource&) = delete;
    ArenaExpertSource& operator=(const ArenaExpertSource&) = delete;

    /// Allocates and loads `<pack_dir>/experts.bin`.  Prints nothing; the caller reports `note()` and the load
    /// rate, because those are the two numbers that say whether the arena is the one that was asked for.
    bool open(const std::string& pack_dir, int64_t n_layers, int64_t n_expert, int threads,
              std::string& err, bool pin_for_cuda = true, bool defer_load = false,
              uint64_t max_pinned_bytes = 0, const std::string& shared_arena_file = {});
    /// xeno merge guard (#56): upstream's open() takes max_pinned_bytes where this one takes pin_for_cuda; a call
    /// written for upstream's order would convert the byte cap to a bool without a word. It does not compile.
    bool open(const std::string&, int64_t, int64_t, int, std::string&, uint64_t, const std::string& = {}) = delete;
    /// Placement-first cold start (#4, PRD "RAM-lean loading"): with `defer_load` the arena is reserved and
    /// committed but nothing is read into it. The caller fills its GPU tiers with `read_expert` (straight from the
    /// pack, not through the arena) and marks each GPU-owned expert with `release_host_copy` (its pages were never
    /// touched, so no physical RAM is used); `load_rest` then reads only the experts the host still owns.
    bool read_expert(int64_t layer, int64_t expert, uint8_t* dst, std::string& err);
    /// `n` experts into dst + i * stride, with unbuffered overlapped reads (FILE_FLAG_NO_BUFFERING: the pack never
    /// lands in the OS file cache) queued together. One caller thread at a time.
    bool read_experts(const int32_t* layers, const int32_t* experts, int n, uint8_t* dst, size_t stride,
                      std::string& err);
    /// The same, each expert to its own destination.
    bool read_experts_to(const int32_t* layers, const int32_t* experts, int n, uint8_t* const* dsts, std::string& err);
    bool materialize_batch(int64_t layer, const int32_t* experts, int n, std::string& err) override;
    bool materialize_begin(int64_t layer, const int32_t* experts, int n, std::string& err) override;
    bool materialize_end(std::string& err) override;
    /// #11 N1 capacity mode: before load_rest, keep at most `bytes` of host-owned experts, the first ones of
    /// `order` (layer * n_expert + expert, best first); the rest stay on NVMe. 0 = no limit.
    void set_capacity(uint64_t bytes, const std::vector<int32_t>& order);
    bool resident(int64_t layer, int64_t expert) const override;
    /// #86: a GPU tier owns this expert (exclusive placement: no host copy).  The routing trace's owned tag.
    bool owned_by_gpu(int64_t layer, int64_t expert) const {
        const size_t i = (size_t) (layer * n_expert_ + expert);
        return i < exclusive_.size() && exclusive_[i];
    }
    const uint8_t* materialize(int64_t layer, int64_t expert, int64_t avoid_layer, std::string& err) override;
    bool read_into(int64_t layer, int64_t expert, uint8_t* dst, std::string& err) override;
    /// #34: a faster copy of some experts (the lendable tail's contiguous file): read_into tries it first; it returns
    /// false for an expert it does not hold.  Called from the prompt path's stager threads.
    void set_tail_reader(std::function<bool(int64_t, int64_t, uint8_t*)> r) { tail_reader_ = std::move(r); }
    /// Evict down to the capacity (materialize may run over when every candidate sits in `avoid_layer`).
    void trim(int64_t avoid_layer);
    /// Decay the host experts' use scores (once per verify window).
    void decay_scores(float f = 0.97f);
    uint64_t host_cache_bytes() const { return cache_used_; }
    int64_t nvme_loads() const { return nvme_loads_; }
    /// The time the caller waited on NVMe-tier loads (#95: a pool run overlapped between begin and end not counted).
    double nvme_ms() const { return nvme_ms_; }
    /// Where the NVMe-tier loads' time went, summed over every load (the miss-cost breakdown): evicting (decommit),
    /// committing the slots, issuing the reads, waiting for completions, copying the bounce buffer into the slots.
    struct NvmeStages { double evict_ms = 0, commit_ms = 0, submit_ms = 0, wait_ms = 0, copy_ms = 0; };
    NvmeStages nvme_stages() const { return stages_; }
    /// #62: a directory holding byte-identical copies of the expert source files (a GGUF shard, experts.bin) on
    /// another drive; repeatable.  read_experts_to sends each expert, all its ranges, to the copy with the fewest
    /// bytes queued in that batch (ties: the source).  A copy is checked against its source when first used (size,
    /// then the first, last and sampled pages); a mismatch fails the read.  A directory without the file is ignored.
    void add_mirror(const std::string& dir) { mirror_dirs_.push_back(dir); }
    struct NvmeFileStat {
        std::string path;
        int64_t reads = 0;    ///< experts read from this copy
        uint64_t bytes = 0;   ///< aligned bytes requested
        double ms = 0;        ///< per batch: submit to this copy's last completion, summed
        int64_t batches = 0;
        double p50_us = 0, p99_us = 0, max_us = 0;   ///< per read: submit to completion (last 65,536)
    };
    /// One entry per file read_experts_to has opened, the source first.
    std::vector<NvmeFileStat> nvme_file_stats() const;
    /// #62 crash: keep a host expert resident while a caller holds its pointer across materialize calls (a paired
    /// swap's stage 2); the eviction skips it until release_hold.  No-op outside capacity mode.
    void hold(int64_t layer, int64_t expert);
    void release_hold(int64_t layer, int64_t expert);
    /// The expert's bytes, held (release with release_hold): a resident one is held at once, a miss is read in and
    /// held - one step under the host lock, so no eviction can take it between the lookup and the hold (with slots
    /// a reused slot would hand over another expert's bytes silently).  Null and `err` when it cannot be produced.
    const uint8_t* acquire(int64_t layer, int64_t expert, std::string& err);
    /// Capacity mode's slab: the host tier lives in committed slots, one region per blob size.  A load takes an idle
    /// slot of its size (no commit, no demand-zero faults); an eviction leaves its slot committed and idle, and only
    /// idle bytes past the slack (STRATA_NVME_SLACK_MIB, 512) are decommitted.  host_slots() = slots committed now
    /// (0 = the fixed arena addresses: STRATA_NVME_SLOTS=0, a pinned arena or no capacity); host_idle_bytes() = the
    /// committed bytes no expert uses.
    int64_t host_slots() const { return slab_committed_; }
    uint64_t host_idle_bytes() const { return idle_bytes_; }
    /// A GPU-owned expert that comes home (paired swap copy-home) joins the host tier: account it and trim.
    void admit_home(int64_t layer, int64_t expert);
    bool load_rest(int threads, std::string& err);
    bool deferred() const { return deferred_; }
    /// Caller must first fill and verify this pair in the primary GPU cache.
    bool release_host_copy(int64_t layer, int64_t expert, std::string& err);
    uint64_t released_host_bytes() const { return released_host_bytes_; }
    /// Phase 4 paired swap, for an expert the GPU owns exclusively: re-commit its host pages and return where its
    /// bytes go (the caller copies them home from the GPU), then publish_host_copy once that copy has landed.
    uint8_t* recommit_host_copy(int64_t layer, int64_t expert, std::string& err);
    void publish_host_copy(int64_t layer, int64_t expert);
    /// Plan v0.3 P6: a native pack without experts.bin takes its experts from the model's shard 1.
    void set_gguf(const std::string& shard1) { gguf_ = shard1; }
    void close();

    bool mapped() const { return base_ != nullptr; }
    int64_t blobs() const { return blobs_; }
    const uint8_t* blob(int64_t layer, int64_t expert) override;
    int64_t reads() const { return reads_; }
    bool pinned(int64_t layer, int64_t expert) const override;
    const uint8_t* device_alias(int64_t layer, int64_t expert) const override;

    /// What backing was obtained and why, for the startup print.  "The engine adapts to the machine it is on" is
    /// only true if the engine says what it got.
    const std::string& note() const { return note_; }
    double load_gib_per_second() const { return gib_per_s_; }
    // Loader fix: the load, split.  `load_seconds()` is the wall clock of the load loop; the other two are
    // sums over the reader threads (see LoadStats), so on their own they say how much of that wall was spent
    // waiting for the disk and how much in memcpy + FNV-1a.
    double load_seconds() const { return load_seconds_; }
    double load_read_seconds() const { return load_read_s_; }
    double load_copy_seconds() const { return load_copy_s_; }

private:
    std::function<bool(int64_t, int64_t, uint8_t*)> tail_reader_;
    void* arena_ = nullptr;          ///< the PinnedArena, owned
    std::vector<uint8_t> exclusive_;
    uint64_t released_host_bytes_ = 0;
    std::mutex host_mu_;   ///< release / recommit / publish run on the primary and the 4070 adapt threads
    std::vector<const uint8_t*> dev_slice_;   ///< device alias of each registered slice (or of the whole range)
    uint64_t slice_bytes_ = 0;
    const uint8_t* base_ = nullptr;
    int64_t blobs_ = 0;
    int64_t n_expert_ = 0;
    int64_t reads_ = 0;
    std::string note_;
    double gib_per_s_ = 0.0;
    double load_seconds_ = 0.0;
    double load_read_s_ = 0.0;
    double load_copy_s_ = 0.0;
    uint64_t pinned_bytes_ = 0;
    std::string gguf_;
    std::string path_;          ///< experts.bin, when the pack has one
    bool from_gguf_ = false;
    bool deferred_ = false;     ///< open(defer_load): experts not read yet (load_rest pending)
    std::ifstream rf_;          ///< read_expert's file, kept open across calls
    std::string rf_name_;
    void* dscratch_ = nullptr;  ///< read_experts' aligned bounce buffer
    size_t dscratch_bytes_ = 0;
    /// read_experts' open DirectFiles, by name, with #62's per-copy stats and its last read latencies
    struct DFile {
        static constexpr size_t kLat = 65536;
        std::string name;
        void* file = nullptr;
        NvmeFileStat st;
        std::vector<float> lat_us = std::vector<float>(kLat);   ///< submit to completion, a ring
        uint64_t lat_next = 0, lat_n = 0;
    };
    std::vector<DFile> dfiles_;
    /// #95: a submitted read batch (submit_reads), collected later (collect_reads); read_experts_to is both at once
    struct PendingReads {
        struct Req { int file; uint64_t skip, len; uint8_t* to; };
        std::vector<Req> reqs;
        size_t slot_bytes = 0;
        double t0 = 0;
    };
    PendingReads pend_;
    bool submit_reads(const int32_t* layers, const int32_t* experts, int n, uint8_t* const* dsts, std::string& err);
    bool collect_reads(std::string& err);
    std::unique_lock<std::mutex> mat_lock_;   ///< #95: host_mu_, held from materialize_begin to materialize_end
    std::vector<int32_t> mat_es_;
    int64_t mat_layer_ = -1;
    double mat_ms_ = 0;                        ///< the begin half's time (eviction, commit, submit)
    std::vector<std::string> mirror_dirs_;                 ///< #62: add_mirror's directories
    /// #62: per source file, its verified copies (resolved on first use)
    std::vector<std::pair<std::string, std::vector<std::string>>> copies_;
    bool copies_of(const std::string& source, const std::vector<std::string>*& out, std::string& err);
    // #11 N1 capacity mode
    uint64_t cache_cap_ = 0;         ///< 0 = no limit
    uint64_t cache_used_ = 0;        ///< host-owned expert bytes committed now
    std::vector<uint8_t> nvme_;      ///< 1 = on NVMe only (never committed, or evicted)
    std::vector<float> score_;       ///< decayed use count per expert
    std::vector<uint8_t> held_;      ///< #62 crash: hold() count per expert; evict_one skips these
    int64_t nvme_loads_ = 0;
    double nvme_ms_ = 0;
    NvmeStages stages_;
    bool evict_one(int64_t avoid_layer);
    // capacity mode's slab (set_capacity): each host-resident expert sits in a slot of its blob size's region
    struct SlabClass {
        uint8_t* base = nullptr;           ///< reserved for every expert of these layers; committed per slot
        size_t stride = 0;                 ///< the blob size rounded up to a page
        int32_t next = 0, cap = 0;         ///< slots ever committed / reserved
        std::vector<int32_t> idle;         ///< committed, no expert in it
        std::vector<int32_t> cold;         ///< decommitted again (reused before `next` grows)
    };
    bool slab_ = false;
    std::vector<SlabClass> classes_;
    std::vector<int16_t> class_of_layer_;
    std::vector<int32_t> slot_of_;         ///< per expert, -1 = no slot
    uint64_t idle_bytes_ = 0, slack_bytes_ = 0;
    int64_t slab_committed_ = 0;
    void slab_release();
    uint8_t* host_at(size_t idx) const;    ///< the expert's host bytes: its slot, or its fixed arena offset
    bool take_slot(size_t idx, int64_t avoid_layer, std::string& err);   ///< evicts until a slot is free
    void free_slot(size_t idx);
    const uint8_t* materialize_locked(int64_t layer, int64_t expert, int64_t avoid_layer, std::string& err);
};

}  // namespace strata::core
