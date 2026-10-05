// include/strata/prefill/moe_mmq.hpp - prompt-speed plan step 2b: the prompt path's experts through llama.cpp's MMQ
// kernels (ggml-cuda mmq.cuh, MIT): the weights stay quantized and the activations are rounded to q8_1, the
// products run on int8 tensor cores.  The dequantize-to-FP16 + cuBLAS path wrote ~10 MB of FP16 per expert and
// multiplied in FP16; this reads the ~1.4-2 MB expert once.  A group of experts is gathered into one buffer
// (`gather_*`, one launch per expert as its blob arrives) and multiplied in one launch per product.
#pragma once

#include <cstddef>
#include <cstdint>

namespace strata::prefill::mmq {

/// This build has the MMQ path (the ggml sources were available to the build).
bool built();
/// llama.cpp's one-time CUDA init (device enumeration), ~2.4 s here: call it on a thread during load so the first
/// prompt does not pay it (#30).  Safe to call more than once.
void warm();
/// MMQ covers this ggml type (the i-quants and Q2_0 the packs use, Q8_0, and in a CUDA build with STRATA_MMQ_KQUANTS
/// the Q4_K / Q5_K / Q5_1 of Unsloth's UD-Q4_K_XL; IQ1_M with #169's tile unless STRATA_MMQ_IQ1M=0).
bool supported(int ggml_type);
/// #420: `supported`, and on every visible GPU llama.cpp's MMQ has a tile for this type and a weight matrix of
/// `w_rows` rows that fits the card's shared memory - the same test its tile choice makes, which aborts the process
/// ("J_best=0") when nothing fits.  false (said once per type) keeps that product on the non-MMQ path.
bool fits(int ggml_type, int64_t w_rows);
/// Bytes of one expert's gate+up ([2*n_ff, n_embd]) or down ([n_embd, n_ff]) weights in `ggml_type`.
size_t matrix_bytes(int ggml_type, int64_t rows, int64_t cols);
/// Bytes of `rows` activation rows of `cols` values quantized for MMQ (the row padded to 512 values).
size_t q8_bytes(int64_t rows, int64_t cols);

/// q8_1 activations for MMQ against weights of `ggml_type`: row i of the output is row ids[i] of x (or row i when
/// ids is null); `x` has `ld` floats per row.
void quantize(const float* x, const int32_t* ids, void* xq, int ggml_type, int64_t cols, int64_t ld, int64_t rows,
              void* stream);

/// One launch over n experts whose weights lie `expert_bytes` apart from `w`: for expert e, the activation rows
/// [bounds[e], bounds[e+1]) of `xq` (bounds on the device, n+1 entries) times its [w_rows, w_cols] matrix into
/// dst rows of the same indices (`ld_dst` floats apart, via `ids`: dst row = ids[row], an identity table works).
/// `total_rows`: the rows of xq; `max_rows`: the most rows one expert has (the launch grid).
struct Product {
    const void* w = nullptr;
    int type = -1;
    int64_t w_rows = 0, w_cols = 0;
    size_t expert_bytes = 0;
    int n = 0;
    const void* xq = nullptr;
    const int32_t* bounds = nullptr;
    const int32_t* ids = nullptr;
    int64_t total_rows = 0, max_rows = 0;
    float* dst = nullptr;
    int64_t ld_dst = 0;
};

/// The launch context (llama.cpp's MMQ keeps a small scratch pool for its stream-k fixup).  One per prompt path.
class Context {
public:
    Context();
    ~Context();
    Context(const Context&) = delete;
    Context& operator=(const Context&) = delete;
    void run(const Product& p, void* stream);

private:
    void* ctx_ = nullptr;
};

/// A GGUF-native expert (gate at `gate`, up at `up`, down at `down`, each its GGUF rows) into a group buffer's
/// slot: gate rows then up rows at `gu_dst`, down at `d_dst`.
void gather_native(const void* gate, const void* up, size_t gu_half_bytes, const void* down, size_t d_bytes,
                   void* gu_dst, void* d_dst, void* stream);
/// #32: one row's bytes in quantize()'s layout (q8_1 blocks of 128 values, 144 B, column-block major).
size_t q8_row_bytes(int64_t cols);
/// #32: rows `rows[0..n)` of `src` (src_rows rows quantized by quantize() without ids) into `dst`, in quantize()'s
/// layout: byte-identical to quantize() of the gathered float rows (xeno_q8_row_gather).
void gather_q8_rows(const void* src, int64_t src_rows, const int32_t* rows, int64_t n, int64_t cols, void* dst,
                    void* stream);
/// The most experts one group gather takes (both overloads below: the fork's #29 and upstream's #372).
constexpr int kGatherGroupMax = 16;
/// xeno #29: up to kGatherGroupMax GGUF-native expert blobs (gate at blob, up at blob + up_off, down at
/// blob + down_off) into consecutive group slots in ONE launch: expert i's gate then up rows at gu_dst + i * gu_stride,
/// its down at d_dst + i * d_stride.  Byte-identical to gather_native per expert (xeno_gather_group_parity); more than
/// kGatherGroupMax or an unaligned pointer falls back to gather_native per expert inside the call.
void gather_native_group(const uint8_t* const* blobs, int n, size_t up_off, size_t down_off, size_t gu_half_bytes,
                         size_t d_bytes, void* gu_dst, size_t gu_stride, void* d_dst, size_t d_stride, void* stream);
/// upstream #372: gather_native for an MMQ group's experts [first, n) in ONE launch: expert q's blob (`blob[q]`; gate
/// at +0, up at +up_off, down at +down_off) to gu_dst + q * gu_stride and d_dst + q * d_stride - the same bytes as one
/// gather_native each.  Every pointer, offset and size 16-byte aligned (false otherwise: nothing launched, gather one
/// at a time).  Note the argument order differs from the #29 overload (gu_half_bytes before down_off).
struct GatherGroup {
    const uint8_t* blob[kGatherGroupMax] = {};
    int first = 0, n = 0;
};
bool gather_native_group(const GatherGroup& g, size_t up_off, size_t gu_half_bytes, size_t down_off, size_t d_bytes,
                         void* gu_dst, size_t gu_stride, void* d_dst, size_t d_stride, void* stream);
/// A Strata-pack Q2_0 expert blob (codes and fp16 scales in separate planes, gate/up rows interleaved) into GGUF
/// Q2_0 blocks: gate/up [1280, 2560] at `gu_dst` (rows stay interleaved), down [2560, 640] at `d_dst`.  Same values.
void gather_strata_q2(const uint8_t* blob, void* gu_dst, void* d_dst, void* stream);

/// h[r, k] = silu(gate) * up of GU rows [2 n_ff wide]: interleaved (gate 2k, up 2k+1: the Strata pack) or split
/// (gate k, up n_ff + k: GGUF).  FP32 out (the down product's quantizer reads floats).
void swiglu(const float* gu, float* h, int64_t rows, int64_t n_ff, bool interleaved, void* stream);

/// #32: one sub-product of an MMQ group, the prompt path's routed-expert rows on whichever card runs it: the rows'
/// q8 activations gathered from the per-token ones, gate/up, swiglu, H to q8_1, down.  n experts' gate/up and down
/// lie gu_bytes / down_bytes apart (zeroed bytes after the last, see prefill's MMQ_TAIL); bounds: n + 1 row offsets
/// from the first row (device); dst: nr rows of n_embd floats.  xq, gu_out, h and hq hold nr rows (scratch).
struct ExpertRows {
    const void* xtok = nullptr;   // quantize() of the chunk's xtok_rows token rows (n_embd values each, no ids)
    int64_t xtok_rows = 0;
    const int32_t* rows = nullptr;   // device: the token of each of the nr rows
    int64_t nr = 0, max_rows = 0;
    int n = 0;
    const void* gu = nullptr;
    int gu_type = -1;
    size_t gu_bytes = 0;
    const void* down = nullptr;
    int down_type = -1;
    size_t down_bytes = 0;
    const int32_t* bounds = nullptr;
    const int32_t* ids = nullptr;   // an identity table of at least nr entries
    int64_t n_embd = 0, n_ff = 0;
    bool interleaved = false;       // gate/up rows interleaved (the Strata pack) or split (GGUF)
    void* xq = nullptr;
    float* gu_out = nullptr;
    float* h = nullptr;
    void* hq = nullptr;
    float* dst = nullptr;
};
void expert_rows(Context& ctx, const ExpertRows& a, void* stream);
/// expert_rows in its two halves, for a caller that times them apart: the gather, gate/up and swiglu (into h), then
/// H to q8_1 and down (into dst).
void expert_rows_gate_up(Context& ctx, const ExpertRows& a, void* stream);
void expert_rows_down(Context& ctx, const ExpertRows& a, void* stream);

/// dst[i] = i for i < n (the identity row map MMQ's MoE mode writes through).
void iota(int32_t* dst, int64_t n, void* stream);

}  // namespace strata::prefill::mmq
