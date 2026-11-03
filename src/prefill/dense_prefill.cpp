// src/prefill/dense_prefill.cpp - #297 stage 3: the dense (qwen35dense) prompt path.  See dense_prefill.hpp.
#include "strata/prefill/dense_prefill.hpp"

#include "strata/core/layer.hpp"
#include "strata/kernels/kv_q4.hpp"
#include "strata/kernels/kv_q8.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/native_qsa.hpp"
#include "strata/kernels/native_rope.hpp"
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/qsa_decode_attn.hpp"
#include "strata/kernels/rope_scaling.hpp"
#include "strata/kernels/verify_kernels.hpp"
#include "strata/timeline_gpu.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <tuple>
#include <stdexcept>

namespace strata::prefill {
namespace {

constexpr float EPS = 1e-6f;   // the model's rms_norm_eps (geometry_from_metadata refuses any other)

enum Phase { kEmbed, kNorm, kGdnProj, kGdnConv, kGdnRec, kGdnOut, kAttnProj, kAttnKv, kAttn, kAttnOut, kFfnGU, kFfnDown,
             kHead, kPhases };
const char* const kPhaseName[kPhases] = {"embed", "norms+adds", "gdn proj", "gdn conv+gates", "gdn recurrence",
                                         "gdn out proj", "attn proj+norm+rope", "attn kv append", "attn", "attn out proj",
                                         "ffn gate+up", "ffn down", "head"};

bool timing_on() {
    static const bool on = [] { const char* v = std::getenv("STRATA_DENSE_PF_TIMING"); return v && v[0] == '1'; }();
    return on;
}

}  // namespace

struct DensePrefill::Impl {
    const core::WeightTable* wt = nullptr;
    const core::ModelGeometry* g = nullptr;
    core::SessionState* ss = nullptr;
    const core::NativeHead* head = nullptr;
    cudaStream_t cs = nullptr;
    int64_t T = 0, n_vocab = 0, cap = 0, qb = 0;
    strata::kernels::QsaShapes s;
    Gemm gm;
    std::vector<void*> allocs;
    // device buffers
    int32_t *tok = nullptr, *steps = nullptr, *pos_q = nullptr, *pos_k = nullptr, *ids = nullptr, *n_dev = nullptr,
            *n_rows = nullptr;
    float *R = nullptr, *x32 = nullptr, *qkv = nullptr, *h = nullptr, *z = nullptr, *ab = nullptr, *gate = nullptr,
          *beta = nullptr, *y = nullptr, *Qf = nullptr, *Kc = nullptr, *Vc = nullptr, *qcur = nullptr, *attn = nullptr,
          *attn32 = nullptr, *scratch = nullptr, *sg = nullptr, *su = nullptr, *bo = nullptr, *gsc = nullptr,
          *lg = nullptr;
    uint16_t *x16 = nullptr, *xbf = nullptr, *y16 = nullptr, *sh16 = nullptr;
    uint8_t* xq = nullptr;
    // the attention as GEMMs (FP16 cache): K / V^T of the cells read so far, q per head, the scores and probabilities
    // of a block of queries; grown on demand (the context's cells, not its capacity)
    uint16_t *kg = nullptr, *vt = nullptr, *q16 = nullptr, *pb = nullptr;
    float* sb = nullptr;
    int64_t kg_n = 0, vt_n = 0, pb_n = 0, sb_n = 0;
    template <typename X>
    bool grow(X*& p, int64_t& have, int64_t want, std::string& err) {
        if (want <= have) return true;
        if (p) cudaFree(p);
        p = nullptr;
        have = 0;
        if (cudaMalloc((void**) &p, (size_t) want * sizeof(X)) != cudaSuccess) {
            (void) cudaGetLastError();
            err = "dense prefill: the attention's " + std::to_string(((size_t) want * sizeof(X)) >> 20) + " MiB do not fit";
            return false;
        }
        have = want;
        return true;
    }
    // host staging
    std::vector<int32_t> h_steps, h_pos_q, h_pos_k;
    // phase timing (STRATA_DENSE_PF_TIMING=1) and the timeline lane
    std::vector<std::pair<int, cudaEvent_t>> marks;
    std::vector<cudaEvent_t> free_events;
    double phase_ms[kPhases] = {};
    timeline::GpuClock* clock = nullptr;
    cudaEvent_t tl_prev = nullptr;
    int tl_phase = -1;

    template <typename X>
    bool alloc(X*& p, size_t n, std::string& err) {
        void* d = nullptr;
        if (cudaMalloc(&d, std::max<size_t>(n, 1) * sizeof(X)) != cudaSuccess) {
            (void) cudaGetLastError();
            err = "dense prefill: a " + std::to_string((n * sizeof(X)) >> 20) + " MiB buffer does not fit";
            return false;
        }
        allocs.push_back(d);
        p = (X*) d;
        return true;
    }
    // the GPU time between consecutive marks goes to the phase of the first
    void mark(int phase) {
        if (timing_on()) {
            cudaEvent_t e = nullptr;
            if (!free_events.empty()) { e = free_events.back(); free_events.pop_back(); }
            else cudaEventCreate(&e);
            cudaEventRecord(e, cs);
            marks.emplace_back(phase, e);
        }
        if (clock != nullptr) {
            static const int lane = timeline::lane("gpu0 prompt");
            cudaEvent_t e = clock->record(cs);
            if (tl_prev != nullptr && tl_phase >= 0) clock->span(lane, kPhaseName[tl_phase], tl_prev, e);
            tl_prev = e;
            tl_phase = phase;
        }
    }
    void fold() {   // after a sync
        for (size_t i = 0; i + 1 < marks.size(); ++i) {
            float ms = 0.0f;
            if (cudaEventElapsedTime(&ms, marks[i].second, marks[i + 1].second) == cudaSuccess)
                phase_ms[marks[i].first] += ms;
            else (void) cudaGetLastError();
        }
        for (auto& m : marks) free_events.push_back(m.second);
        marks.clear();
        if (clock != nullptr) clock->resolve(false);
        tl_prev = nullptr;
        tl_phase = -1;
    }
    ~Impl() {
        for (void* p : allocs) cudaFree(p);
        for (void* p : {(void*) kg, (void*) vt, (void*) pb, (void*) sb}) if (p) cudaFree(p);
        for (auto& m : marks) cudaEventDestroy(m.second);
        for (cudaEvent_t e : free_events) cudaEventDestroy(e);
    }
};

DensePrefill::DensePrefill() : impl_(std::make_unique<Impl>()) {}
DensePrefill::~DensePrefill() = default;
int64_t DensePrefill::n_vocab() const { return impl_->n_vocab; }

bool DensePrefill::init(const core::WeightTable& wt, const core::ModelGeometry& g, core::SessionState& ss,
                        const core::NativeHead* head, int64_t chunk, void* stream, std::string& err) {
    Impl& m = *impl_;
    if (!g.dense || chunk < 1 || chunk > 65535) { err = "dense prefill: a dense geometry and a chunk of 1..65535 rows"; return false; }
    m.wt = &wt; m.g = &g; m.ss = &ss; m.head = head; m.cs = (cudaStream_t) stream; m.T = chunk;
    for (int64_t j = 0; j < ss.qsa_alloc; ++j)
        if (ss.qsa_states[ss.qsa_ord0 + j].kv_mode == 1) { err = "dense prefill: KV streaming (--kv-resident) is not supported"; return false; }
    if (!strata::kernels::native_norm_rope_usable((int) g.head_dim, 64)) {
        err = "dense prefill: the fused RMS norm + rope kernel is not usable on this card";
        return false;
    }
    m.s = strata::kernels::qsa_real_shapes();
    m.s.n_head = g.n_head; m.s.n_head_kv = g.n_head_kv; m.s.head_dim = g.head_dim;
    m.s.idx_n_head = g.idx_q_heads; m.s.idx_dim = g.idx_key_dim; m.s.idx_top_k = strata::kernels::kDenseTopK;
    m.cap = ss.qsa_states[ss.qsa_primary()].max_cells;
    const core::WeightRef* wo = wt.find("output.weight");
    m.n_vocab = wo != nullptr ? wo->ne1 : 0;
    const int64_t T = chunk, N = g.n_embd, C = g.ssm_conv_channels, V = g.ssm_value_dim, HV = g.ssm_v_heads;
    const int64_t NH = g.n_head, NKV = g.n_head_kv, HD = g.head_dim, FF = g.n_ff;
    // the attention's queries go in batches whose split-K scratch stays near 256 MiB
    const uint64_t per = strata::kernels::qsa_decode_attn_scratch_floats(m.cap, m.s);
    m.qb = std::clamp<int64_t>((int64_t) ((256ull << 20) / (per * 4)), 1, 64);
    // the GEMM scratch: the largest projection, dequantized to FP16
    int64_t biggest = 0;
    for (int64_t l = 0; l < g.n_layers; ++l) {
        const core::LayerView v(wt, l);
        for (const char* sfx : {"attn_qkv.weight", "attn_gate.weight", "ssm_out.weight", "attn_q.weight", "attn_k.weight",
                                "attn_v.weight", "attn_output.weight", "ffn_gate_shexp.weight", "ffn_up_shexp.weight",
                                "ffn_down_shexp.weight"})
            if (const core::WeightRef* w = v.get(sfx)) {
                if (w->native_data == nullptr) { err = "dense prefill: " + v.name(sfx) + " has no native blocks"; return false; }
                biggest = std::max<int64_t>(biggest, w->ne0 * w->ne1);
            }
    }
    if (!m.gm.init(stream, biggest, err)) return false;
    const int64_t wide = std::max<int64_t>(V, NH * HD);
    bool ok = m.alloc(m.tok, T, err) && m.alloc(m.steps, T * strata::kernels::kStepCount, err) &&
              m.alloc(m.pos_q, T * NH, err) && m.alloc(m.pos_k, T * NKV, err) && m.alloc(m.ids, m.qb * m.cap, err) &&
              m.alloc(m.n_dev, 16, err) && m.alloc(m.n_rows, 1, err) && m.alloc(m.R, T * N, err) &&
              m.alloc(m.x16, T * N, err) && m.alloc(m.xbf, T * N, err) && m.alloc(m.qkv, T * C, err) &&
              m.alloc(m.h, T * C, err) && m.alloc(m.z, T * V, err) && m.alloc(m.ab, T * 2 * HV, err) &&
              m.alloc(m.gate, T * HV, err) && m.alloc(m.beta, T * HV, err) && m.alloc(m.y, T * V, err) &&
              m.alloc(m.y16, T * wide, err) && m.alloc(m.Qf, T * NH * 2 * HD, err) && m.alloc(m.Kc, T * NKV * HD, err) &&
              m.alloc(m.Vc, T * NKV * HD, err) && m.alloc(m.qcur, T * NH * HD, err) && m.alloc(m.attn, T * NH * HD, err) &&
              m.alloc(m.attn32, T * NH * HD, err) && m.alloc(m.scratch, (size_t) m.qb * per, err) &&
              m.alloc(m.sg, T * FF, err) && m.alloc(m.su, T * FF, err) && m.alloc(m.sh16, T * FF, err) &&
              m.alloc(m.bo, T * N, err) && m.alloc(m.gsc, T, err) && m.alloc(m.q16, T * NH * HD, err);
    if (ok && head != nullptr && head->loaded())
        ok = m.alloc(m.x32, T * N, err) && m.alloc(m.lg, (size_t) 8 * m.n_vocab, err) &&
             m.alloc(m.xq, strata::kernels::native_q8_1_bytes((int) N, 8), err);
    if (!ok) return false;
    {   // constants: the identity selection rows (dense attention reads cells 0..pos), and the counts 0..15
        std::vector<int32_t> ids((size_t) (m.qb * m.cap));
        for (int64_t q = 0; q < m.qb; ++q)
            for (int64_t i = 0; i < m.cap; ++i) ids[(size_t) (q * m.cap + i)] = (int32_t) i;
        int32_t counts[16];
        for (int i = 0; i < 16; ++i) counts[i] = i;
        if (cudaMemcpy(m.ids, ids.data(), ids.size() * 4, cudaMemcpyHostToDevice) != cudaSuccess ||
            cudaMemcpy(m.n_dev, counts, sizeof counts, cudaMemcpyHostToDevice) != cudaSuccess) {
            err = "dense prefill: the constant buffers could not be set";
            return false;
        }
    }
    {   // the first call of each GEMM shape loads its cuBLAS kernels (~0.35 s on the 5060 Ti for the first chunk, measured):
        // at boot, once, every projection shape at the chunk's row count (the outputs are discarded)
        const int64_t first_attn = g.qsa_interval - 1;
        const core::LayerView vg(wt, 0), va(wt, first_attn);
        for (const auto& [v, sfx, out] : std::initializer_list<std::tuple<const core::LayerView*, const char*, float*>>{
                 {&vg, "attn_qkv.weight", m.qkv}, {&vg, "attn_gate.weight", m.z}, {&va, "attn_q.weight", m.Qf},
                 {&va, "attn_k.weight", m.Kc}, {&vg, "ffn_gate_shexp.weight", m.sg}})
            if (const core::WeightRef* w = v->get(sfx)) m.gm.native(m.x16, w->native_type, w->native_data, out, T, w->ne1, w->ne0);
        if (const core::WeightRef* w = vg.get("ssm_out.weight")) m.gm.native(m.y16, w->native_type, w->native_data, m.bo, T, w->ne1, w->ne0);
        if (const core::WeightRef* w = va.get("attn_output.weight")) m.gm.native(m.y16, w->native_type, w->native_data, m.bo, T, w->ne1, w->ne0);
        if (const core::WeightRef* w = vg.get("ffn_down_shexp.weight")) m.gm.native(m.sh16, w->native_type, w->native_data, m.bo, T, w->ne1, w->ne0);
        if (const core::WeightRef* w = vg.get("ssm_alpha.weight")) m.gm.bf16(m.xbf, (const uint16_t*) w->data, m.ab, T, w->ne1, w->ne0, 2 * HV);
        if (const core::WeightRef* w = vg.get("ffn_gate_inp_shexp.weight")) m.gm.bf16(m.xbf, (const uint16_t*) w->data, m.gsc, T, 1, N);
        if (cudaStreamSynchronize(m.cs) != cudaSuccess) { err = "dense prefill: the GEMM warm-up failed"; return false; }
    }
    m.h_steps.resize((size_t) (T * strata::kernels::kStepCount));
    m.h_pos_q.resize((size_t) (T * NH));
    m.h_pos_k.resize((size_t) (T * NKV));
    if (timeline::enabled()) {
        static timeline::GpuClock clock;
        static bool anchored = false;
        if (!anchored) { clock.anchor(m.cs); anchored = true; }
        m.clock = &clock;
    }
    std::fprintf(stderr, "strata prefill: qwen35dense prompt path, chunks of up to %lld rows, attention in batches of %lld "
                         "queries over up to %lld cells, GEMM scratch %.0f MiB\n", (long long) T, (long long) m.qb,
                 (long long) m.cap, (double) biggest * 2.0 / 1048576.0);
    return true;
}

bool DensePrefill::run(const int32_t* tokens, int64_t n, int64_t pos0, float* logits, std::string& err) {
    using namespace strata::kernels;
    Impl& m = *impl_;
    const core::ModelGeometry& g = *m.g;
    const core::WeightTable& wt = *m.wt;
    core::SessionState& ss = *m.ss;
    if (n < 1 || n > m.T || pos0 < 0 || pos0 + n > m.cap) { err = "dense prefill: rows out of range"; return false; }
    if (logits != nullptr && m.lg == nullptr) { err = "dense prefill: logits asked for without the native head"; return false; }
    const core::NativeEmbed* embed = core::native_embed();
    if (embed == nullptr) { err = "dense prefill: no native token embedding"; return false; }
    const int64_t N = g.n_embd, C = g.ssm_conv_channels, V = g.ssm_value_dim, HV = g.ssm_v_heads, HK = g.ssm_k_heads;
    const int64_t NH = g.n_head, NKV = g.n_head_kv, HD = g.head_dim, FF = g.n_ff, S = g.ssm_state_size;
    const uint64_t gdn_floats = (uint64_t) S * HV * S + (uint64_t) C * (g.ssm_d_conv - 1);
    cudaStream_t cs = m.cs;
    Gemm& gm = m.gm;
    // the rows' steps and rope positions
    for (int64_t t = 0; t < n; ++t) {
        qsa_step_fill(m.h_steps.data() + t * kStepCount, pos0 + t, m.s);
        for (int64_t hh = 0; hh < NH; ++hh) m.h_pos_q[(size_t) (t * NH + hh)] = (int32_t) (pos0 + t);
        for (int64_t hh = 0; hh < NKV; ++hh) m.h_pos_k[(size_t) (t * NKV + hh)] = (int32_t) (pos0 + t);
    }
    const int32_t n32 = (int32_t) n;
    if (cudaMemcpyAsync(m.tok, tokens, (size_t) n * 4, cudaMemcpyHostToDevice, cs) != cudaSuccess ||
        cudaMemcpyAsync(m.steps, m.h_steps.data(), (size_t) n * kStepCount * 4, cudaMemcpyHostToDevice, cs) != cudaSuccess ||
        cudaMemcpyAsync(m.pos_q, m.h_pos_q.data(), (size_t) (n * NH) * 4, cudaMemcpyHostToDevice, cs) != cudaSuccess ||
        cudaMemcpyAsync(m.pos_k, m.h_pos_k.data(), (size_t) (n * NKV) * 4, cudaMemcpyHostToDevice, cs) != cudaSuccess ||
        cudaMemcpyAsync(m.n_rows, &n32, 4, cudaMemcpyHostToDevice, cs) != cudaSuccess) {
        err = "dense prefill: staging the rows failed";
        return false;
    }
    auto need = [&](const core::LayerView& v, const char* sfx) -> const core::WeightRef* {
        const core::WeightRef* w = v.get(sfx);
        if (w == nullptr && err.empty()) err = "dense prefill: " + v.name(sfx) + " is missing";
        return w;
    };
    auto native = [&](const uint16_t* X, const core::WeightRef* w, float* Y, int64_t ldy = 0) {
        gm.native(X, w->native_type, w->native_data, Y, n, w->ne1, w->ne0, ldy);
    };
    try {
        m.mark(kEmbed);
        embed->gather_dev(m.tok, n, m.R, cs);
        int64_t gi = 0, qi = 0;
        for (int64_t l = 0; l < g.n_layers; ++l) {
            const core::LayerView v(wt, l);
            const core::WeightRef *hn = need(v, "hc_attn_norm.weight"), *hf = need(v, "hc_ffn_norm.weight");
            if (!hn || !hf) return false;
            m.mark(kNorm);
            // the hyper-connection read with zero projections: rms_norm(R) * gamma / 2 (gamma = the stored 2 (1 + w))
            dense::rms_rows(m.R, n, (int) N, (const float*) hn->data, 0.5f, EPS, m.x16, m.xbf, nullptr, cs);
            if (!core::is_qsa_layer(g, l)) {
                const core::WeightRef *wqkv = need(v, "attn_qkv.weight"), *wg = need(v, "attn_gate.weight"),
                                      *wout = need(v, "ssm_out.weight"), *wa = need(v, "ssm_alpha.weight"),
                                      *wb = need(v, "ssm_beta.weight"), *wc = need(v, "ssm_conv1d.weight"),
                                      *wnm = need(v, "ssm_norm.weight"), *wdt = need(v, "ssm_dt.bias"),
                                      *wsa = need(v, "ssm_a");
                if (!wqkv || !wg || !wout || !wa || !wb || !wc || !wnm || !wdt || !wsa) return false;
                m.mark(kGdnProj);
                native(m.x16, wqkv, m.qkv);
                native(m.x16, wg, m.z);
                gm.bf16(m.xbf, (const uint16_t*) wa->data, m.ab, n, HV, N, 2 * HV);
                gm.bf16(m.xbf, (const uint16_t*) wb->data, m.ab + HV, n, HV, N, 2 * HV);
                m.mark(kGdnConv);
                dense::gdn_gates(m.ab, n, (int) HV, (const float*) wdt->data, (const float*) wsa->data, m.gate, m.beta, cs);
                float* state = ss.gdn_state + (size_t) (gi - ss.gdn_ord0) * gdn_floats;
                float* conv = state + (uint64_t) S * HV * S;
                gdn_conv_l2_rows(conv, m.qkv, (const float*) wc->data, m.h, (int) C, (int) (2 * HK), EPS, (int) n,
                                 m.n_rows, cs);
                m.mark(kGdnRec);
                // the recurrence over all the rows in one launch per layer, the state read once and committed once
                gdn_step_norm_rows(state, m.h, (int) C, m.gate, m.beta, m.z, (const float*) wnm->data, EPS, m.y,
                                   (int) HK, (int) HV, (int) n, m.n_rows, cs);
                m.mark(kGdnOut);
                dense::to_f16(m.y, m.y16, n * V, cs);
                native(m.y16, wout, m.bo);
                ++gi;
            } else {
                const core::WeightRef *wq = need(v, "attn_q.weight"), *wk = need(v, "attn_k.weight"),
                                      *wv = need(v, "attn_v.weight"), *wo = need(v, "attn_output.weight"),
                                      *wqn = need(v, "attn_q_norm.weight"), *wkn = need(v, "attn_k_norm.weight");
                if (!wq || !wk || !wv || !wo || !wqn || !wkn) return false;
                const core::QsaState& st = ss.qsa_states[qi];
                m.mark(kAttnProj);
                native(m.x16, wq, m.Qf);
                native(m.x16, wk, m.Kc);
                native(m.x16, wv, m.Vc);
                native_qsa_rms_norm_rope(m.Qf, (int) (2 * HD), (const float*) wqn->data, m.qcur, (int) (n * NH), (int) HD,
                                         64, EPS, rope_scaling(), m.pos_q, cs);
                native_qsa_rms_norm_rope(m.Kc, (int) HD, (const float*) wkn->data, m.Kc, (int) (n * NKV), (int) HD, 64, EPS,
                                         rope_scaling(), m.pos_k, cs);
                if (st.kv_rot) {   // the verify window's rotations (kv_q4.hpp): q, K and V; hybrid: V only
                    fwht256_inplace_cuda(m.qcur, n * NH, cs);
                    fwht256_inplace_cuda(m.Kc, n * NKV, cs);
                    fwht256_inplace_cuda(m.Vc, n * NKV, cs);
                } else if (st.kv_hybrid) {
                    fwht256_inplace_cuda(m.Vc, n * NKV, cs);
                }
                m.mark(kAttnKv);
                if (st.kv_hybrid) {   // K8V4: the unused half's lanes folded onto the used pool, as the verify window
                    kv_append_q8_steps(st.k_q, st.k_q, st.k_scale, st.k_scale, st.page_table, m.steps, kStepCount, m.Kc,
                                       m.Kc, (int) (NKV * HD), (int) n, m.s, cs, nullptr);
                    kv_append_q4_steps(st.v_q4, st.v_q4, st.page_table, m.steps, kStepCount, (int) n, m.Vc, m.Vc, m.s, cs,
                                       nullptr);
                } else if (st.kv_q4) {
                    kv_append_q4_steps(st.k_q4, st.v_q4, st.page_table, m.steps, kStepCount, (int) n, m.Kc, m.Vc, m.s, cs,
                                       nullptr);
                } else if (st.kv_int8) {
                    kv_append_q8_steps(st.k_q, st.v_q, st.k_scale, st.v_scale, st.page_table, m.steps, kStepCount, m.Kc,
                                       m.Vc, (int) (NKV * HD), (int) n, m.s, cs, nullptr);
                } else {   // FP16: the rows in one launch (kv_append_step's stores)
                    dense::kv_append_f16_rows(st.k_pool, st.v_pool, st.page_table, m.steps, kStepCount, m.Kc, m.Vc, n,
                                              (int) NKV, (int) HD, (int) m.s.page_size, cs);
                }
                m.mark(kAttn);
                // an FP16 cache: the attention as GEMMs - every query of the chunk over cells [0, pos0 + n), the
                // causal mask in the softmax (STRATA_DENSE_ATTN_DECODE=1: the decode kernel, one query at a time)
                static const bool attn_decode = [] { const char* e = std::getenv("STRATA_DENSE_ATTN_DECODE"); return e && e[0] == '1'; }();
                const bool gemm_attn = !attn_decode && st.k_pool != nullptr && st.v_pool != nullptr && !st.kv_int8 &&
                                       !st.kv_q4 && !st.kv_hybrid && !st.kv_rot;
                if (gemm_attn) {
                    const int64_t nv = pos0 + n, nkp = (nv + 63) / 64 * 64, grp = NH / NKV;
                    int64_t qs = std::max<int64_t>(64, ((128ll << 20) / (nkp * 6)) / 64 * 64);
                    qs = std::min<int64_t>(qs, n);
                    if (!m.grow(m.kg, m.kg_n, NKV * nkp * HD, err) || !m.grow(m.vt, m.vt_n, NKV * nkp * HD, err) ||
                        !m.grow(m.sb, m.sb_n, qs * nkp, err) || !m.grow(m.pb, m.pb_n, qs * nkp, err))
                        return false;
                    dense::gather_kv_f16(st.k_pool, st.v_pool, st.page_table, nkp, nv, (int) NKV, (int) HD,
                                         (int) m.s.page_size, m.kg, m.vt, cs);
                    dense::q_heads_f16(m.qcur, n, (int) NH, (int) HD, 1.0f / std::sqrt((float) HD), m.q16, cs);
                    for (int64_t hh = 0; hh < NH; ++hh) {
                        const int64_t kvh = hh / grp;
                        for (int64_t q0 = 0; q0 < n; q0 += qs) {
                            const int64_t mq = std::min<int64_t>(qs, n - q0);
                            gm.f16(m.q16 + (hh * n + q0) * HD, m.kg + kvh * nkp * HD, m.sb, mq, nkp, HD);
                            dense::causal_softmax_f16(m.sb, mq, nkp, pos0 + q0, m.pb, cs);
                            gm.f16(m.pb, m.vt + kvh * HD * nkp, m.attn + (q0 * NH + hh) * HD, mq, HD, nkp, NH * HD);
                        }
                    }
                } else {
                const QsaAttnPools pools = core::qsa_attn_pools(st);
                for (int64_t q0 = 0; q0 < n; q0 += m.qb) {
                    const int64_t nq = std::min<int64_t>(m.qb, n - q0);
                    qsa_decode_attn_batch(m.qcur + q0 * NH * HD, pools, m.ids, m.steps + q0 * kStepCount, m.cap, m.s,
                                          m.scratch, m.attn + q0 * NH * HD, nq, cs);
                }
                }
                if (st.kv_rot || st.kv_hybrid) fwht256_inplace_cuda(m.attn, n * NH, cs);   // back: H^-1 = H
                m.mark(kAttnOut);
                native_qsa_gate_apply(m.attn, m.Qf, m.attn32, (int) (n * NH), (int) HD, cs);
                dense::to_f16(m.attn32, m.y16, n * NH * HD, cs);
                native(m.y16, wo, m.bo);
                ++qi;
            }
            m.mark(kNorm);
            dense::add_rows(m.R, m.bo, n, (int) N, nullptr, cs);
            dense::rms_rows(m.R, n, (int) N, (const float*) hf->data, 0.5f, EPS, m.x16, m.xbf, nullptr, cs);
            const core::WeightRef *wgi = need(v, "ffn_gate_inp_shexp.weight"), *wsg = need(v, "ffn_gate_shexp.weight"),
                                  *wsu = need(v, "ffn_up_shexp.weight"), *wsd = need(v, "ffn_down_shexp.weight");
            if (!wgi || !wsg || !wsu || !wsd) return false;
            m.mark(kFfnGU);
            native(m.x16, wsg, m.sg);
            native(m.x16, wsu, m.su);
            dense::swiglu_f16(m.sg, m.su, m.sh16, n * FF, cs);
            m.mark(kFfnDown);
            native(m.sh16, wsd, m.bo);
            gm.bf16(m.xbf, (const uint16_t*) wgi->data, m.gsc, n, 1, N);   // the shared expert's scalar gate
            m.mark(kNorm);
            dense::add_rows(m.R, m.bo, n, (int) N, m.gsc, cs);
        }
        if (logits != nullptr) {
            m.mark(kHead);
            const core::WeightRef* hn = wt.find("output_hc_norm.weight");
            if (hn == nullptr) { err = "dense prefill: output_hc_norm.weight is missing"; return false; }
            dense::rms_rows(m.R, n, (int) N, (const float*) hn->data, 0.5f, EPS, nullptr, nullptr, m.x32, cs);
            for (int64_t r0 = 0; r0 < n; r0 += 8) {
                const int k = (int) std::min<int64_t>(8, n - r0);
                native_quantize_q8_1(m.x32 + r0 * N, m.xq, (int) N, k, cs);
                native_mmvq(m.head->type(), m.head->weights(), m.xq, m.lg, (int) N, (int) m.n_vocab, k, cs);
                if (cudaMemcpyAsync(logits + (size_t) r0 * m.n_vocab, m.lg, (size_t) k * m.n_vocab * 4,
                                    cudaMemcpyDeviceToHost, cs) != cudaSuccess) {
                    err = "dense prefill: the logits copy failed";
                    return false;
                }
            }
        }
        m.mark(kEmbed);   // the end mark (its own phase is never charged: nothing follows it)
    } catch (const std::exception& e) {
        err = std::string("dense prefill: ") + e.what();
        return false;
    }
    const cudaError_t se = cudaStreamSynchronize(cs);
    if (se != cudaSuccess) { err = std::string("dense prefill: ") + cudaGetErrorString(se); return false; }
    m.fold();
    return true;
}

std::string DensePrefill::phase_report() const {
    if (!timing_on()) return std::string();
    std::string out;
    char buf[96];
    for (int p = 0; p < kPhases; ++p) {
        if (impl_->phase_ms[p] <= 0.0) continue;
        std::snprintf(buf, sizeof buf, "%s%s %.1f", out.empty() ? "" : ", ", kPhaseName[p], impl_->phase_ms[p]);
        out += buf;
    }
    return out + " ms";
}

}  // namespace strata::prefill
