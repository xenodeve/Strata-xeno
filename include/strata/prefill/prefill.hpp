// include/strata/prefill/prefill.hpp - plan v0.3 P5: batched prompt processing.
//
// The prompt's positions [pos0, pos0 + n) are processed in chunks of `chunk` tokens through all 48 layers, leaving
// the session state (GDN recurrence and conv state, QSA KV pools and indexer, PLE history) where the token path
// would have left it; the decode loop then continues with the next token.  Per layer: the projections are
// tensor-core GEMMs (quantized weights dequantized to FP16 on the fly, BF16 weights as they are), the recurrences
// walk the chunk inside one kernel, and the routed experts are grouped by expert: resident ones are read from the
// VRAM tier, the others streamed from the host arena through a pinned ring on a copy stream.
//
// Requires the native weights (`--native`): every quantized projection must carry its GGUF blocks.
#pragma once

#include "strata/core/expert_cache.hpp"
#include "strata/core/expert_source.hpp"
#include "strata/core/layer.hpp"
#include "strata/core/session.hpp"
#include "strata/core/weights.hpp"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace strata::core { class PeerExperts; }

namespace strata::prefill {

struct PrefillStats {
    int64_t tokens = 0;
    int64_t chunks = 0;
    double ms_total = 0;
    double ms_experts_host = 0;     ///< host time staging non-resident experts
    int64_t experts_streamed = 0;   ///< expert blobs copied host -> device
    int64_t experts_dma = 0;        ///< ...of which straight from the pinned arena (no CPU copy)
    int64_t experts_resident = 0;   ///< expert-layer groups served from the VRAM tier
    /// #5 P5a: the streamed experts by source (count, bytes): pinned arena, pageable arena, the 4070 tier, NVMe
    int64_t src_n[4] = {0, 0, 0, 0};
    uint64_t src_bytes[4] = {0, 0, 0, 0};
    int64_t src_rows[4] = {0, 0, 0, 0};   ///< routed token rows those experts served (small chunks; 0 on stream-all)
    double ms_ple = 0;
    /// #35 D7: another lane's stats of the same prompt: counts add up, the wall time is the longer lane's
    void merge_lane(const PrefillStats& o) {
        tokens += o.tokens; chunks += o.chunks; ms_total = ms_total > o.ms_total ? ms_total : o.ms_total;
        ms_experts_host += o.ms_experts_host; experts_streamed += o.experts_streamed; experts_dma += o.experts_dma;
        experts_resident += o.experts_resident; ms_ple += o.ms_ple;
        for (int i = 0; i < 4; ++i) { src_n[i] += o.src_n[i]; src_bytes[i] += o.src_bytes[i]; src_rows[i] += o.src_rows[i]; }
    }
};

}  // namespace strata::prefill
namespace strata::core { class MtpDrafter; }
namespace strata::prefill {

class Prefill {
public:
    /// E-9: the draft layer's K/V for prompt cells [cell0, cell0 + n) from their final residual rows `R_rows`
    /// (device) and `next_tokens` (host: the token at cell+1), in batches through this path's GEMMs and its idle
    /// scratch - call it from on_chunk.  false with `err` empty: not applicable here (a ring or hybrid K/V, another
    /// device, too little scratch; STRATA_MTP_BATCH=0), the caller runs the drafter's own pass.  Not bit-identical to
    /// that pass (FP16 GEMMs instead of Q8_1 activations): the drafts may differ, never the target's tokens' logits.
    bool draft_kv(core::MtpDrafter& mtp, const float* R_rows, const int32_t* next_tokens, int64_t n, int64_t cell0,
                  std::string& err);
    Prefill();
    ~Prefill();
    Prefill(const Prefill&) = delete;
    Prefill& operator=(const Prefill&) = delete;

    /// Frees every buffer, stream and event `init` made (as the destructor does) and starts over empty, so `init` can
    /// run again - with a smaller chunk when the first one did not fit.  The stage range (`set_stage`) and the
    /// callbacks stay.  The device `init` ran on must be current.
    void reset();

    /// `host_res`: the static residency table (n_layers x n_expert, slot or -1) or null; `cache` its slots.
    /// `borrow`/`borrow_bytes`: device memory to carve every buffer from (the top slots of the expert cache,
    /// lent for the prompt and refilled after it); null = allocate normally.
    bool init(const core::WeightTable& wt, const core::ModelGeometry& g, core::SessionState& ss,
              core::ExpertSource* src, const core::ExpertCache* cache, const int32_t* host_res, int64_t chunk,
              void* stream, std::string& err, void* borrow = nullptr, uint64_t borrow_bytes = 0);

    /// Exclusive 4070 tier (#4): experts whose only copy lives on another device. `res` is (n_layers x n_expert)
    /// slot or -1 and `slot_ptr(slot)` that slot's pointer on `device`; the prompt path stages such an expert with a
    /// peer copy instead of reading its (released) host pages.
    void set_peer_tier(const int32_t* res, std::function<const void*(int32_t)> slot_ptr, int device);
    /// With borrowed buffers: lay them out again for chunks of `chunk` tokens (at most `init`'s) in `borrow` - a
    /// request lends only the slots its prompt needs.  The stream must be idle (between prompts).
    /// #122: `kv_end` - the request's end position, which sizes the KV-streaming staging pool (kv_stage_plan.hpp);
    /// 0 = the worst case, every page of the context.
    bool relayout(int64_t chunk, void* borrow, uint64_t borrow_bytes, std::string& err, int64_t kv_end = 0);
    int64_t chunk() const;
    /// #122: the cells the KV-streaming staging pool of this layout holds (INT64_MAX when the session is not
    /// streamed): a prompt must end at or before them.
    int64_t stage_cells() const;

    /// The share of the streamed experts' bytes DMA-able straight from pinned RAM (1 = all).  Sizes the streamed
    /// ring (a big one only pays when the copy engine, not the host copies, is the limit); set before bytes_needed.
    static void set_pinned_share(double share);
    /// #35 D6: the split layout (STRATA_PREFILL_EXPERT_SPLIT with the peer tier): a chunk of split_min() tokens (#119:
    /// STRATA_PREFILL_SPLIT_MIN, default 2048) or
    /// more runs its routed experts on the peer card, so the one-card MoE buffers (expert rows, MMQ scratch, stream
    /// ring) are sized only for the shorter chunks that still run here.  Set before bytes_needed and init.
    static void set_split_layout(bool on);
    /// #113: the split runs per layer; `full` = every layer on MMQ, `one_card` = the layers a
    /// split chunk keeps on this card.  Valid once the expert layout is loaded (the MMQ plan reads it, once).
    static bool split_layout_full();
    static bool split_wave_ok();   // #115: the wave may run on this pack's split
    static std::vector<int> split_one_card_layers();
    static double pinned_share();
    /// #340: the streamed ring's slot count for chunks that stream every expert, instead of the pinned-share rule
    /// (0 = that rule). Set before any `bytes_needed`/`init` (both count the ring); STRATA_PREFILL_RING still wins.
    static void set_ring_override(int slots);

    /// Device bytes `init` needs for a chunk of `chunk` tokens (what a borrowed region must hold).  #122: `kv_end`
    /// as in relayout (0 = the worst case: boot-time sizing, the chunk choice and the lendable tail).
    static uint64_t bytes_needed(const core::ModelGeometry& g, const core::SessionState& ss, int64_t chunk,
                                 int64_t kv_end = 0);
    /// #35 D7: a wave lane's chunk for a prompt-path chunk of `chunk` tokens (half, rounded up to 256), and the device
    /// bytes both lanes need - the one sizing rule for the lend, the layouts and the relayouts
    static int64_t wave_lane_chunk(int64_t chunk) { return chunk / 2 < 256 ? 256 : (chunk / 2 + 255) / 256 * 256; }
    static uint64_t wave_bytes_needed(const core::ModelGeometry& g, const core::SessionState& ss, int64_t chunk,
                                      int64_t kv_end = 0) {
        return 2 * wave_lane_bytes(g, ss, wave_lane_chunk(chunk), kv_end);
    }
    /// one lane's share of a lent region for chunks of `lane_chunk` tokens (4 KiB aligned: lane 2's starts after it)
    static uint64_t wave_lane_bytes(const core::ModelGeometry& g, const core::SessionState& ss, int64_t lane_chunk,
                                    int64_t kv_end = 0) {
        return (bytes_needed(g, ss, lane_chunk, kv_end) + 4095) / 4096 * 4096;
    }
    /// whether a wave over prompt-path chunks of `chunk` tokens still runs each lane's chunk split
    static bool wave_lane_splits(int64_t chunk);

    /// #35 D7: the two-lane wavefront.  Two Prefill objects on one session read a prompt's chunks alternately
    /// (chunk c on lane c % 2, each lane on its own stream of the same GPU), and chunk c's layer l starts only once
    /// chunk c-1 has queued that layer's attention half (its KV, GDN and PLE state) - the order one lane keeps, so
    /// the output is the same bytes.  With expert_split, one lane's chunk runs its routed experts on the 4070 while
    /// the other lane's runs its trunk on this card.  Both lanes' run() get the whole prompt; reset() first.
    struct WaveLink;
    static std::shared_ptr<WaveLink> make_wave_link();
    static void wave_reset(WaveLink& link, int64_t n_chunks, int64_t n_layers);
    void set_wave(std::shared_ptr<WaveLink> link, int lane);
    /// Read [pos0, pos0 + n) through both lanes (`b` on its own thread; b's stream is synchronized before return).
    /// The first lane's error wins unless it only reports the other lane's failure.
    static bool run_wave(Prefill& a, Prefill& b, WaveLink& link, const int64_t* tokens, int64_t n, int64_t pos0,
                         int64_t n_layers, std::string& err);

    /// Positions [pos0, pos0 + n) holding `tokens`; `ss.ple_prev` must be the two tokens before pos0 (oldest
    /// first, -1 for none) and is advanced to the last two of these.
    bool run(const int64_t* tokens, int64_t n, int64_t pos0, std::string& err);

    const PrefillStats& stats() const { return stats_; }

    /// upstream's peer share of the prompt path (the peer's experts computed THERE for every chunk).  Not ported to
    /// this fork's split / wave walk (#129): it declines with a message, and the prompt path stays on the primary, as
    /// upstream's `--peer-prefill-rows 0`.
    bool set_peer(core::PeerExperts* peer, int64_t cap_rows, std::string& err);

    /// Plan v0.3 P6: called after every chunk with the chunk's final multi-stream residual rows (device,
    /// T x hc*n_embd, valid until the next chunk) and the chunk's first position; the MTP draft layer builds its
    /// K/V from them.  The prefill stream is synchronized before the call.
    std::function<bool(const float* R_rows, int64_t T, int64_t pos0, std::string& err)> on_chunk;

    /// Layer split: called by every stage when it has read a chunk, with the position reached, while its own state
    /// is still at that chunk's end (its stream synchronized; the last stage calls it just before `on_chunk`).  An
    /// earlier stage is a chunk or more ahead of the last one by the time `on_chunk` runs, so this is where a
    /// mid-prompt checkpoint takes each stage's part.  Runs on that stage's thread, with its device current.
    std::function<bool(int64_t done, std::string& err)> on_stage_chunk;

    /// Checked before every chunk: true stops the prompt early (`run` returns false with err "cancelled").
    std::function<bool()> should_stop;

    /// The vision path: HOST rows (n_embd floats) indexed by absolute position, read in place of the token
    /// embedding where non-null (an image's <|image_pad|> cells).  Null (default): every position embeds its token.
    const float* const* embd_rows = nullptr;

    /// LAYER SPLIT (multi-GPU): this prompt path runs layers [layer_begin, layer_end) (-1: to the last) on the
    /// device `init` runs on.  A stage that does not start at layer 0 reads each chunk's residual rows from the
    /// previous stage instead of embedding the tokens; a stage that does not end at the last layer copies its rows
    /// to pinned host buffers (two, allocated by `init`) and runs `next` on them - on a thread, so the next stage
    /// reads chunk c while this one reads chunk c + 1.  `on_chunk` belongs on the last stage.  Set before `init`.
    void set_stage(int64_t layer_begin, int64_t layer_end, Prefill* next) {
        stage_lb_ = layer_begin; stage_le_ = layer_end; next_ = next;
    }

private:
    int64_t stage_lb_ = 0, stage_le_ = -1;
    Prefill* next_ = nullptr;
    const float* hand_in_ = nullptr;    ///< the previous stage's rows of the chunk being read (host, pinned)
    bool carve(std::size_t T, void* alloc, int64_t kv_end);   // the device buffers of a chunk (prefill.cpp's Alloc)
    void release();                          // the destructor's cleanup (also `reset`'s)
    struct Impl;
    std::unique_ptr<Impl> impl_;
    PrefillStats stats_;
};

}  // namespace strata::prefill
