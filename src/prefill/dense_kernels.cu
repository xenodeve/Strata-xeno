// src/prefill/dense_kernels.cu - #297 stage 3: the element-wise kernels of the dense (qwen35dense) prompt path; see
// include/strata/prefill/dense_prefill.hpp.
#include "strata/prefill/dense_prefill.hpp"
#include "strata/kernels/bf16_bits.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/qsa.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

namespace strata::prefill::dense {
namespace {

constexpr int THREADS = 256;

void check(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "dense prefill %s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

// one block per row: y = x * rsqrt(mean(x^2) + eps) * w * w_scale, as FP16 (the GEMMs' input), BF16 (the BF16
// projections') and FP32 (the head's q8_1 input) - each written when its pointer is given
__global__ void __launch_bounds__(THREADS) rms_rows_kernel(const float* __restrict__ x, int n, const float* __restrict__ w,
                                                           float w_scale, float eps, uint16_t* __restrict__ y16,
                                                           uint16_t* __restrict__ ybf, float* __restrict__ y32) {
    const float* xr = x + (size_t) blockIdx.x * n;
    float ss = 0.0f;
    for (int i = threadIdx.x; i < n; i += THREADS) ss = fmaf(xr[i], xr[i], ss);
    __shared__ float part[THREADS / 32];
    for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
    if ((threadIdx.x & 31) == 0) part[threadIdx.x >> 5] = ss;
    __syncthreads();
    float s = 0.0f;
#pragma unroll
    for (int i = 0; i < THREADS / 32; ++i) s += part[i];
    const float rs = rsqrtf(s / (float) n + eps);
    for (int i = threadIdx.x; i < n; i += THREADS) {
        const float v = xr[i] * rs * (w[i] * w_scale);
        const size_t o = (size_t) blockIdx.x * n + i;
        if (y16) y16[o] = __half_as_ushort(__float2half_rn(v));
        if (ybf) ybf[o] = strata::kernels::bf16_from_f32(v);
        if (y32) y32[o] = v;
    }
}

__global__ void add_rows_kernel(float* __restrict__ R, const float* __restrict__ b, int64_t n, int row,
                                const float* __restrict__ row_gate) {
    const int64_t i = (int64_t) blockIdx.x * THREADS + threadIdx.x;
    if (i >= n) return;
    float g = 1.0f;
    if (row_gate != nullptr) g = 1.0f / (1.0f + __expf(-row_gate[i / row]));
    R[i] = fmaf(b[i], g, R[i]);
}

__global__ void gdn_gates_kernel(const float* __restrict__ ab, int hv, int64_t n, const float* __restrict__ dt,
                                 const float* __restrict__ ssm_a, float* __restrict__ gate, float* __restrict__ beta) {
    const int64_t i = (int64_t) blockIdx.x * THREADS + threadIdx.x;   // t * hv + h
    if (i >= n) return;
    const int64_t t = i / hv;
    const int h = (int) (i - t * hv);
    const float a = ab[t * 2 * hv + h], b = ab[t * 2 * hv + hv + h];
    const float v = a + dt[h];
    const float sp = v > 20.0f ? v : log1pf(__expf(v));
    gate[i] = sp * ssm_a[h];
    beta[i] = 1.0f / (1.0f + __expf(-b));
}

__global__ void to_f16_kernel(const float* __restrict__ x, uint16_t* __restrict__ y, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * THREADS + threadIdx.x;
    if (i < n) y[i] = __half_as_ushort(__float2half_rn(x[i]));
}

__global__ void swiglu_f16_kernel(const float* __restrict__ g, const float* __restrict__ u, uint16_t* __restrict__ h,
                                  int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * THREADS + threadIdx.x;
    if (i >= n) return;
    const float a = g[i];
    h[i] = __half_as_ushort(__float2half_rn(a / (1.0f + __expf(-a)) * u[i]));
}

unsigned blocks(int64_t n) { return (unsigned) ((n + THREADS - 1) / THREADS); }

// the FP16 cache's cells [0, n_kv) of every KV head into contiguous K [kvh][cell][hd] and V^T [kvh][hd][cell]
__global__ void gather_kv_kernel(const uint16_t* __restrict__ k_pool, const uint16_t* __restrict__ v_pool,
                                 const int32_t* __restrict__ table, int n_kv, int n_valid, int nkv, int hd, int page,
                                 uint16_t* __restrict__ K, uint16_t* __restrict__ Vt) {
    const int cell = blockIdx.x, kvh = blockIdx.y;
    const bool valid = cell < n_valid;   // the padding cells are zeros (never NaN under a 0 weight)
    const long long row = valid ? ((long long) table[cell / page] * nkv + kvh) * page + cell % page : 0;
    for (int d = threadIdx.x; d < hd; d += blockDim.x) {
        K[((size_t) kvh * n_kv + cell) * hd + d] = valid ? k_pool[row * hd + d] : (uint16_t) 0;
        Vt[((size_t) kvh * hd + d) * n_kv + cell] = valid ? v_pool[row * hd + d] : (uint16_t) 0;
    }
}

// n rows of K/V [t][kvh][hd] into the FP16 cache at their steps' positions: kv_append_kernel's stores (f16_from_f32),
// one launch for the rows
__global__ void kv_append_rows_kernel(uint16_t* __restrict__ k_pool, uint16_t* __restrict__ v_pool,
                                      const int32_t* __restrict__ table, const int32_t* __restrict__ steps, int stride,
                                      const float* __restrict__ K, const float* __restrict__ V, int64_t n_elems,
                                      int kv_heads, int head_dim, int page_size) {
    const int64_t i = (int64_t) blockIdx.x * THREADS + threadIdx.x;
    if (i >= n_elems) return;
    const int per = kv_heads * head_dim;
    const int64_t t = i / per;
    const int j = (int) (i - t * per), h = j / head_dim, d = j - h * head_dim;
    const long long pos = (long long) steps[t * stride + strata::kernels::kStepPos];
    const long long page = (long long) table[pos / page_size];
    if (page < 0) return;
    const long long row = (page * kv_heads + h) * page_size + (pos % page_size);
    k_pool[row * head_dim + d] = strata::kernels::f16_from_f32(K[i]);
    v_pool[row * head_dim + d] = strata::kernels::f16_from_f32(V[i]);
}

// q [t][head][hd] f32 -> [head][t][hd] FP16, the scale folded in
__global__ void q_heads_kernel(const float* __restrict__ q, int n, int nh, int hd, float scale, uint16_t* __restrict__ out) {
    const int64_t i = (int64_t) blockIdx.x * THREADS + threadIdx.x;
    if (i >= (int64_t) n * nh * hd) return;
    const int d = (int) (i % hd);
    const int64_t th = i / hd;
    const int h = (int) (th % nh);
    const int64_t t = th / nh;
    out[((size_t) h * n + t) * hd + d] = __half_as_ushort(__float2half_rn(q[i] * scale));
}

// one block per query row: softmax over keys [0, pos0 + row] (causal), FP16 out, the masked keys 0
__global__ void __launch_bounds__(THREADS) causal_softmax_kernel(const float* __restrict__ S, int n_kv, int64_t pos0,
                                                                 uint16_t* __restrict__ P) {
    const int r = blockIdx.x;
    const float* s = S + (size_t) r * n_kv;
    uint16_t* p = P + (size_t) r * n_kv;
    const int last = (int) (pos0 + r);   // inclusive
    __shared__ float red[THREADS / 32];
    float m = -3.0e38f;
    for (int j = threadIdx.x; j <= last; j += THREADS) m = fmaxf(m, s[j]);
    for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = m;
    __syncthreads();
    m = red[0];
#pragma unroll
    for (int i = 1; i < THREADS / 32; ++i) m = fmaxf(m, red[i]);
    __syncthreads();
    float l = 0.0f;
    for (int j = threadIdx.x; j <= last; j += THREADS) l += __expf(s[j] - m);
    for (int o = 16; o > 0; o >>= 1) l += __shfl_xor_sync(0xffffffffu, l, o);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = l;
    __syncthreads();
    l = 0.0f;
#pragma unroll
    for (int i = 0; i < THREADS / 32; ++i) l += red[i];
    const float inv = 1.0f / l;
    for (int j = threadIdx.x; j < n_kv; j += THREADS)
        p[j] = __half_as_ushort(__float2half_rn(j <= last ? __expf(s[j] - m) * inv : 0.0f));
}

}  // namespace

void gather_kv_f16(const uint16_t* k_pool, const uint16_t* v_pool, const int32_t* page_table, int64_t n_kv,
                   int64_t n_valid, int n_head_kv, int head_dim, int page_size, uint16_t* K, uint16_t* Vt, void* stream) {
    gather_kv_kernel<<<dim3((unsigned) n_kv, (unsigned) n_head_kv), 128, 0, (cudaStream_t) stream>>>(
        k_pool, v_pool, page_table, (int) n_kv, (int) n_valid, n_head_kv, head_dim, page_size, K, Vt);
    check("gather_kv_f16");
}
void kv_append_f16_rows(uint16_t* k_pool, uint16_t* v_pool, const int32_t* page_table, const int32_t* steps,
                        int step_stride, const float* K, const float* V, int64_t n, int n_head_kv, int head_dim,
                        int page_size, void* stream) {
    const int64_t total = n * n_head_kv * head_dim;
    kv_append_rows_kernel<<<blocks(total), THREADS, 0, (cudaStream_t) stream>>>(k_pool, v_pool, page_table, steps,
                                                                                step_stride, K, V, total, n_head_kv,
                                                                                head_dim, page_size);
    check("kv_append_f16_rows");
}
void q_heads_f16(const float* q, int64_t n, int n_head, int head_dim, float scale, uint16_t* out, void* stream) {
    q_heads_kernel<<<blocks(n * n_head * head_dim), THREADS, 0, (cudaStream_t) stream>>>(q, (int) n, n_head, head_dim,
                                                                                         scale, out);
    check("q_heads_f16");
}
void causal_softmax_f16(const float* S, int64_t rows, int64_t n_kv, int64_t pos0, uint16_t* P, void* stream) {
    causal_softmax_kernel<<<(unsigned) rows, THREADS, 0, (cudaStream_t) stream>>>(S, (int) n_kv, pos0, P);
    check("causal_softmax_f16");
}

void rms_rows(const float* x, int64_t rows, int n, const float* w, float w_scale, float eps, uint16_t* y16,
              uint16_t* ybf, float* y32, void* stream) {
    rms_rows_kernel<<<(unsigned) rows, THREADS, 0, (cudaStream_t) stream>>>(x, n, w, w_scale, eps, y16, ybf, y32);
    check("rms_rows");
}
void add_rows(float* R, const float* b, int64_t rows, int n, const float* row_gate, void* stream) {
    const int64_t total = rows * n;
    add_rows_kernel<<<blocks(total), THREADS, 0, (cudaStream_t) stream>>>(R, b, total, n, row_gate);
    check("add_rows");
}
void gdn_gates(const float* ab, int64_t rows, int hv, const float* dt, const float* ssm_a, float* gate, float* beta,
               void* stream) {
    const int64_t n = rows * hv;
    gdn_gates_kernel<<<blocks(n), THREADS, 0, (cudaStream_t) stream>>>(ab, hv, n, dt, ssm_a, gate, beta);
    check("gdn_gates");
}
void to_f16(const float* x, uint16_t* y, int64_t n, void* stream) {
    to_f16_kernel<<<blocks(n), THREADS, 0, (cudaStream_t) stream>>>(x, y, n);
    check("to_f16");
}
void swiglu_f16(const float* g, const float* u, uint16_t* h, int64_t n, void* stream) {
    swiglu_f16_kernel<<<blocks(n), THREADS, 0, (cudaStream_t) stream>>>(g, u, h, n);
    check("swiglu_f16");
}

}  // namespace strata::prefill::dense
