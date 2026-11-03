// src/kernels/qwen35dense_parity.cpp - #297 stage 2: the kernel switches the qwen35dense architecture adds, on tiny
// tensors, against the math of ref/qwen4exp_engine.py transcribed here in double precision.
//
//   1. the GDN output gate: gdn_step_norm_multi (the verify window's recurrence + gated norm) with
//      gdn_gate_silu_set(true) must give y = rms_norm(o) * gamma * silu(z), and with false (the default, Flash-Next)
//      sigmoid(z) - at the 4B's GDN geometry (16 key / 32 value heads x 128), a 3-token window, both the window
//      (state read, not written) and the commit (state written) forms, and the STRATA_GDN_SPLIT form.
//   2. GQA 4: qsa_decode_attn_batch with 16 query / 4 KV heads x 256 (qwen35dense) and, as a control, Flash-Next's
//      24 / 2 - softmax(q.k / 16) over the selected cells, query head h reading KV head h / group.  fp16 K/V pool,
//      4-cell pages through a shuffled page table, two queries with different cell counts, more than one 64-cell chunk.
//   3. dense selection: with a dense geometry's idx_top_k (kDenseTopK) the step's width is n_kv at every position
//      up to 262,144 (attention reads every cell), and the identity selection rows the verifier writes are read as
//      exactly those cells.
//
// Prints one line per check and exits non-zero on any failure.  `--selftest` is accepted (ctest).
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/gdn.hpp"
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/qsa_decode_attn.hpp"
#include "strata/kernels/verify_kernels.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

namespace {

using namespace strata::kernels;

int g_fail = 0;

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "qwen35dense_parity: %s: %s\n", what, cudaGetErrorString(e));
        std::exit(2);
    }
}
template <typename T>
T* up(const std::vector<T>& v) {
    T* d = nullptr;
    ck(cudaMalloc(&d, v.size() * sizeof(T)), "cudaMalloc");
    ck(cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice), "upload");
    return d;
}
template <typename T>
std::vector<T> down(const T* d, size_t n) {
    std::vector<T> v(n);
    ck(cudaMemcpy(v.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost), "download");
    return v;
}
void report(const char* name, double err, double tol) {
    const bool ok = err <= tol && std::isfinite(err);
    std::printf("%-58s max rel err %.3e (tol %.0e)  %s\n", name, err, tol, ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}
double max_rel(const std::vector<float>& got, const std::vector<double>& want) {
    double num = 0, den = 0;
    for (size_t i = 0; i < got.size(); ++i) {
        num = std::max(num, std::fabs((double) got[i] - want[i]));
        den = std::max(den, std::fabs(want[i]));
    }
    return den > 0 ? num / den : num;
}

// ---------------------------------------------------------------- 1. the GDN output gate
// The recurrence as gdn_step_norm_multi_kernel computes it (ref/qwen4exp_engine.py Engine.gdn after the conv):
// value head h reads key head h % h_k; state[(i * h_v + h) * S + j]; per token
//   g = exp(gate), kv_j = sum_i s_ij k_i, d_j = (v_j - g kv_j) beta, s_ij = g s_ij + k_i d_j, o_j = sum_i s_ij q_i,
//   oc = o / sqrt(S), y = oc * rsqrt(mean(oc^2) + eps) * gamma * act(z).
void gdn_reference(std::vector<double>& state, const std::vector<float>& h, int C, const std::vector<float>& gate,
                   const std::vector<float>& beta, const std::vector<float>& z, const std::vector<float>& gamma,
                   float eps, int h_k, int h_v, int T, bool silu, std::vector<double>& y) {
    constexpr int S = 128;
    const int qk = S * h_k, V = S * h_v;
    y.assign((size_t) T * V, 0.0);
    for (int t = 0; t < T; ++t) {
        const float* ht = h.data() + (size_t) t * C;
        for (int hh = 0; hh < h_v; ++hh) {
            const int kh = hh % h_k;
            const double g = std::exp((double) gate[(size_t) t * h_v + hh]);
            const double b = beta[(size_t) t * h_v + hh];
            std::vector<double> o(S, 0.0);
            for (int j = 0; j < S; ++j) {
                double kv = 0;
                for (int i = 0; i < S; ++i) kv += state[((size_t) i * h_v + hh) * S + j] * ht[qk + kh * S + i];
                const double d = (ht[2 * qk + hh * S + j] - g * kv) * b;
                double oj = 0;
                for (int i = 0; i < S; ++i) {
                    double& s = state[((size_t) i * h_v + hh) * S + j];
                    s = g * s + ht[qk + kh * S + i] * d;
                    oj += s * ht[kh * S + i];
                }
                o[j] = oj / std::sqrt((double) S);
            }
            double ss = 0;
            for (double v : o) ss += v * v;
            const double sc = 1.0 / std::sqrt(ss / S + eps);
            for (int j = 0; j < S; ++j) {
                const double zz = z[(size_t) t * V + hh * S + j];
                const double sg = 1.0 / (1.0 + std::exp(-zz));
                y[(size_t) t * V + hh * S + j] = o[j] * sc * gamma[j] * (silu ? zz * sg : sg);
            }
        }
    }
}

void test_gdn_gate() {
    constexpr int S = 128, HK = 16, HV = 32, T = 3;
    const int C = 2 * S * HK + S * HV, V = S * HV;
    std::mt19937 rng(297);
    std::normal_distribution<float> n01(0.0f, 1.0f);
    auto vec = [&](size_t n, float sc) { std::vector<float> v(n); for (auto& x : v) x = n01(rng) * sc; return v; };
    std::vector<float> h = vec((size_t) T * C, 1.0f);
    for (int t = 0; t < T; ++t)   // q and k L2-normalized per head, as the conv kernel hands them on
        for (int hh = 0; hh < 2 * HK; ++hh) {
            float* p = h.data() + (size_t) t * C + hh * S;
            double s = 0;
            for (int i = 0; i < S; ++i) s += (double) p[i] * p[i];
            for (int i = 0; i < S; ++i) p[i] = (float) (p[i] / std::sqrt(s + 1e-6));
        }
    std::vector<float> gate = vec((size_t) T * HV, 0.3f), beta((size_t) T * HV), z = vec((size_t) T * V, 1.5f),
                       gamma = vec(S, 0.1f), state0 = vec((size_t) S * HV * S, 0.05f);
    for (auto& g : gate) g = -std::fabs(g);                   // gate = softplus(..) * ssm_a < 0: a decay
    for (auto& b : beta) b = 0.2f + 0.6f * std::fabs(n01(rng)) / 3.0f;
    for (auto& g : gamma) g += 1.0f;
    const float eps = 1e-6f;
    float *d_h = up(h), *d_gate = up(gate), *d_beta = up(beta), *d_z = up(z), *d_gamma = up(gamma);
    float* d_state = up(state0);
    float* d_y = nullptr;
    ck(cudaMalloc(&d_y, (size_t) T * V * sizeof(float)), "cudaMalloc y");
    int32_t* d_keep = up(std::vector<int32_t>{T});
    cudaStream_t cs;
    ck(cudaStreamCreate(&cs), "stream");
    for (int silu = 0; silu < 2; ++silu) {
        std::vector<double> st(state0.begin(), state0.end()), y;
        gdn_reference(st, h, C, gate, beta, z, gamma, eps, HK, HV, T, silu != 0, y);
        gdn_gate_silu_set(silu != 0);
        // the window: the state read, not written
        ck(cudaMemcpy(d_state, state0.data(), state0.size() * 4, cudaMemcpyHostToDevice), "state");
        gdn_step_norm_multi(d_state, d_h, C, d_gate, d_beta, d_z, d_gamma, eps, d_y, HK, HV, T, nullptr, cs);
        ck(cudaStreamSynchronize(cs), "gdn window");
        const std::vector<float> yw = down(d_y, (size_t) T * V);
        report(silu ? "GDN gate silu: window (gdn_step_norm_multi, 16/32 heads)"
                    : "GDN gate sigmoid: window (gdn_step_norm_multi, 16/32 heads)", max_rel(yw, y), 1e-4);
        // the state it leaves untouched
        const std::vector<float> s_after = down(d_state, state0.size());
        report(silu ? "GDN gate silu: window leaves the state" : "GDN gate sigmoid: window leaves the state",
               max_rel(s_after, std::vector<double>(state0.begin(), state0.end())), 0.0);
        // the commit form (n_keep, outputs from t_out_begin 0): the same outputs, the state advanced
        gdn_step_norm_multi(d_state, d_h, C, d_gate, d_beta, d_z, d_gamma, eps, d_y, HK, HV, T, d_keep, cs, 0);
        ck(cudaStreamSynchronize(cs), "gdn commit");
        report(silu ? "GDN gate silu: commit outputs" : "GDN gate sigmoid: commit outputs",
               max_rel(down(d_y, (size_t) T * V), y), 1e-4);
        report(silu ? "GDN gate silu: commit state" : "GDN gate sigmoid: commit state",
               max_rel(down(d_state, state0.size()), st), 1e-4);
        // a gate that is NOT the model's must be far off: the switch is live, not a no-op
        std::vector<double> st2(state0.begin(), state0.end()), y_other;
        gdn_reference(st2, h, C, gate, beta, z, gamma, eps, HK, HV, T, silu == 0, y_other);
        const double off = max_rel(yw, y_other);
        std::printf("%-58s max rel err %.3e (must be > 1e-1)  %s\n",
                    silu ? "  ... the same window against the sigmoid reference" : "  ... against the silu reference",
                    off, off > 1e-1 ? "ok" : "FAIL");
        if (!(off > 1e-1)) ++g_fail;
    }
    gdn_gate_silu_set(false);   // the process default again
    cudaFree(d_h); cudaFree(d_gate); cudaFree(d_beta); cudaFree(d_z); cudaFree(d_gamma); cudaFree(d_state);
    cudaFree(d_y); cudaFree(d_keep);
    cudaStreamDestroy(cs);
}

// ---------------------------------------------------------------- 2. GQA attention over every selected cell
void test_attention(int n_head, int n_kv, const char* name) {
    constexpr int HD = 256, PAGE = 4;
    const int cells[2] = {150, 97};    // two queries: 3 and 2 chunks of 64 cells
    const int max_cells = 160, pages = max_cells / PAGE, nq = 2;
    QsaShapes s = qsa_real_shapes();
    s.n_head = n_head;
    s.n_head_kv = n_kv;
    s.idx_top_k = kDenseTopK;          // dense: width = n_kv
    std::mt19937 rng(1000 + n_head);
    std::normal_distribution<float> n01(0.0f, 1.0f);
    // the page table: a permutation, so a reader that ignores it reads the wrong cells
    std::vector<int32_t> table(pages);
    for (int i = 0; i < pages; ++i) table[i] = i;
    std::shuffle(table.begin(), table.end(), rng);
    std::vector<uint16_t> kp((size_t) pages * n_kv * PAGE * HD), vp(kp.size());
    std::vector<float> kf(kp.size()), vf(kp.size());   // the fp16 values, for the reference
    for (int cell = 0; cell < max_cells; ++cell)
        for (int kvh = 0; kvh < n_kv; ++kvh)
            for (int d = 0; d < HD; ++d) {
                const size_t row = ((size_t) table[cell / PAGE] * n_kv + kvh) * PAGE + cell % PAGE;
                kp[row * HD + d] = f16_from_f32(n01(rng));
                vp[row * HD + d] = f16_from_f32(n01(rng));
                const size_t logical = ((size_t) cell * n_kv + kvh) * HD + d;
                kf[logical] = f32_from_f16(kp[row * HD + d]);
                vf[logical] = f32_from_f16(vp[row * HD + d]);
            }
    std::vector<float> q((size_t) nq * n_head * HD);
    for (auto& x : q) x = n01(rng) * 0.6f;
    const int64_t cap = max_cells;
    std::vector<int32_t> ids((size_t) nq * cap), steps((size_t) nq * kStepCount);
    for (int r = 0; r < nq; ++r) {
        for (int64_t i = 0; i < cap; ++i) ids[(size_t) (r * cap + i)] = (int32_t) i;   // the verifier's identity rows
        qsa_step_fill(steps.data() + (size_t) r * kStepCount, cells[r] - 1, s);
    }
    QsaAttnPools pools;
    uint16_t *d_k = up(kp), *d_v = up(vp);
    int32_t* d_table = up(table);
    pools.k_pool = d_k;
    pools.v_pool = d_v;
    pools.page_table = d_table;
    float* d_q = up(q);
    int32_t *d_ids = up(ids), *d_steps = up(steps);
    const uint64_t per = qsa_decode_attn_scratch_floats(cap, s);
    float *d_scratch = nullptr, *d_out = nullptr;
    ck(cudaMalloc(&d_scratch, per * nq * sizeof(float)), "scratch");
    ck(cudaMalloc(&d_out, (size_t) nq * n_head * HD * sizeof(float)), "out");
    cudaStream_t cs;
    ck(cudaStreamCreate(&cs), "stream");
    qsa_decode_attn_batch(d_q, pools, d_ids, d_steps, cap, s, d_scratch, d_out, nq, cs);
    ck(cudaStreamSynchronize(cs), "attention");
    const std::vector<float> got = down(d_out, (size_t) nq * n_head * HD);
    std::vector<double> want(got.size());
    const int group = n_head / n_kv;
    for (int r = 0; r < nq; ++r)
        for (int hh = 0; hh < n_head; ++hh) {
            const int kvh = hh / group;   // the q -> kv map is q / group, NOT q % kv (qsa.hpp)
            const float* qr = q.data() + ((size_t) r * n_head + hh) * HD;
            std::vector<double> sc(cells[r]);
            double m = -1e300;
            for (int c = 0; c < cells[r]; ++c) {
                double dot = 0;
                for (int d = 0; d < HD; ++d) dot += (double) qr[d] * kf[((size_t) c * n_kv + kvh) * HD + d];
                sc[c] = dot / 16.0;
                m = std::max(m, sc[c]);
            }
            double l = 0;
            for (double& x : sc) { x = std::exp(x - m); l += x; }
            for (int d = 0; d < HD; ++d) {
                double acc = 0;
                for (int c = 0; c < cells[r]; ++c) acc += sc[c] * vf[((size_t) c * n_kv + kvh) * HD + d];
                want[((size_t) r * n_head + hh) * HD + d] = acc / l;
            }
        }
    report(name, max_rel(got, want), 1e-5);
    cudaFree(d_k); cudaFree(d_v); cudaFree(d_table); cudaFree(d_q); cudaFree(d_ids); cudaFree(d_steps);
    cudaFree(d_scratch); cudaFree(d_out);
    cudaStreamDestroy(cs);
}

// ---------------------------------------------------------------- 3. the dense selection width
void test_dense_width() {
    QsaShapes dense = qsa_real_shapes(), sparse = qsa_real_shapes();
    dense.idx_top_k = kDenseTopK;
    int bad = 0;
    int32_t st[kStepCount];
    for (int64_t pos : {int64_t(0), int64_t(1), int64_t(2050), int64_t(2051), int64_t(4096), int64_t(32767),
                        int64_t(65535), int64_t(262143)}) {
        qsa_step_fill(st, pos, dense);
        bad += st[kStepWidth] != pos + 1 || qsa_selection_width(pos + 1, dense) != pos + 1;
    }
    // the sparse geometry keeps its budget: 2048 + 3 cells once the context passes it
    qsa_step_fill(st, 9999, sparse);
    bad += st[kStepWidth] != 2051;
    std::printf("%-58s %s\n", "dense selection: width = n_kv to 262,144 cells; QSA keeps 2051", bad ? "FAIL" : "ok");
    if (bad) ++g_fail;
}

}  // namespace

int main(int, char**) {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n < 1) {
        std::printf("qwen35dense_parity: no CUDA device, skipped\n");
        return 0;
    }
    test_dense_width();
    test_gdn_gate();
    test_attention(16, 4, "attention GQA 4 (16 q / 4 kv x 256), dense, two queries");
    test_attention(24, 2, "attention GQA 12 (24 q / 2 kv, Flash-Next), two queries");
    std::printf("qwen35dense_parity: %s\n", g_fail ? "FAILED" : "all passed");
    return g_fail ? 1 : 0;
}
