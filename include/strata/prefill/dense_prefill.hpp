// include/strata/prefill/dense_prefill.hpp - #297 stage 3: the batched prompt path of a dense (qwen35dense) model.
//
// `Prefill` is built around Flash-Next: routed experts, the indexer, the PLE, the hyper-connection projections and
// fixed widths (HV 48, C 10240).  A qwen35dense model has none of them - its hyper-connection projections are zero by
// construction (tools/qwen35_to_qwen4exp.py), so every read is rms_norm(R) * (stored gamma / 2) and every write a
// plain add, and its FFN is the shared expert alone - so its prompt runs here, as a plain pre-norm transformer over
// one residual row per token, with the same weights and the same session state the verify windows use:
//
//   per layer   x = rms_norm(R) * hc_attn_norm/2 -> GDN or attention -> R += out
//               x = rms_norm(R) * hc_ffn_norm/2  -> down(silu(gate x) * up x) * sigmoid(x . gate_inp_shexp) -> R += it
//   GDN         qkv, z: GEMMs (Q8_0 blocks dequantized to FP16, cuBLAS); alpha/beta: a BF16 GEMM; the conv + silu +
//               L2 norm over all rows (gdn_conv_l2_rows, the history advanced); the recurrence + gated norm by
//               gdn_step_norm_multi, 8 rows at a time with the state committed (the model's gate, silu); out: a GEMM
//   attention   q|gate, k, v: GEMMs; per-head RMS norm + rope (native_qsa_rms_norm_rope, the rows' positions); K/V into
//               the session's cache in its format; every query over every cell up to its own (dense, causal) with the
//               decode kernel in batches of queries (qsa_decode_attn_batch, GQA 4); the sigmoid output gate; out GEMM
//   head        only for the rows a caller asks logits for (teacher-forced dumps): output_hc_norm/2, q8_1, the
//               native head's GEMV 8 rows at a time
//
// The activations enter the GEMMs as FP16 (the verify window's GEMVs read Q8_1 images), so the rows agree with the
// decode path to activation rounding, not bitwise; tools/test_qwen35_engine.py and the 4B's teacher-forced dumps
// measure it.  KV streaming (kv_mode 1) is refused.  STRATA_DENSE_PF_TIMING=1 prints the GPU time by phase; with
// STRATA_TIMELINE the phases are spans on the lane "gpu0 prompt".
#pragma once

#include "strata/core/layout.hpp"
#include "strata/core/native_head.hpp"
#include "strata/core/session.hpp"
#include "strata/prefill/gemm.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace strata::prefill {

class DensePrefill {
public:
    DensePrefill();
    ~DensePrefill();
    DensePrefill(const DensePrefill&) = delete;
    DensePrefill& operator=(const DensePrefill&) = delete;

    /// `chunk`: the most rows one run() reads (its buffers are sized for it).  The head may be null when no caller
    /// asks for logits.
    bool init(const core::WeightTable& wt, const core::ModelGeometry& g, core::SessionState& ss,
              const core::NativeHead* head, int64_t chunk, void* stream, std::string& err);
    /// Reads tokens[0, n) at positions pos0.. (n <= chunk), advancing the session's GDN state, conv history and K/V
    /// cache.  `logits` (host, n x n_vocab) gets every row's logits when non-null.  Synchronizes the stream.
    bool run(const int32_t* tokens, int64_t n, int64_t pos0, float* logits, std::string& err);
    int64_t n_vocab() const;
    /// The phase totals (ms) since init, when STRATA_DENSE_PF_TIMING=1; empty otherwise.
    std::string phase_report() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

/// The element-wise kernels (src/prefill/dense_kernels.cu).
namespace dense {
void rms_rows(const float* x, int64_t rows, int n, const float* w, float w_scale, float eps, uint16_t* y16,
              uint16_t* ybf, float* y32, void* stream);
/// R[r][i] += b[r][i] * (row_gate ? sigmoid(row_gate[r]) : 1)
void add_rows(float* R, const float* b, int64_t rows, int n, const float* row_gate, void* stream);
/// ab rows [alpha (hv) | beta (hv)]: gate = softplus(alpha + dt) * ssm_a, beta = sigmoid(beta) (gdn_ab_multi's)
void gdn_gates(const float* ab, int64_t rows, int hv, const float* dt, const float* ssm_a, float* gate, float* beta,
               void* stream);
void to_f16(const float* x, uint16_t* y, int64_t n, void* stream);
void swiglu_f16(const float* g, const float* u, uint16_t* h, int64_t n, void* stream);
/// The attention as GEMMs (an FP16 cache): cells [0, n_kv) into K [kvh][cell][hd] and V^T [kvh][hd][cell] (row
/// stride n_kv), cells from n_valid on as zeros (the GEMMs' padding)
void gather_kv_f16(const uint16_t* k_pool, const uint16_t* v_pool, const int32_t* page_table, int64_t n_kv,
                   int64_t n_valid, int n_head_kv, int head_dim, int page_size, uint16_t* K, uint16_t* Vt, void* stream);
/// n rows of K / V [t][kvh][hd] into an FP16 cache at the rows' steps (kv_append_step's stores, one launch)
void kv_append_f16_rows(uint16_t* k_pool, uint16_t* v_pool, const int32_t* page_table, const int32_t* steps,
                        int step_stride, const float* K, const float* V, int64_t n, int n_head_kv, int head_dim,
                        int page_size, void* stream);
/// q [t][head][hd] -> FP16 [head][t][hd], times `scale`
void q_heads_f16(const float* q, int64_t n, int n_head, int head_dim, float scale, uint16_t* out, void* stream);
/// rows x n_kv scores: row r's softmax over keys [0, pos0 + r], FP16, the rest 0
void causal_softmax_f16(const float* S, int64_t rows, int64_t n_kv, int64_t pos0, uint16_t* P, void* stream);
}  // namespace dense

}  // namespace strata::prefill
