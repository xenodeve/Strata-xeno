// src/prefill/moe_mmq.cu - see include/strata/prefill/moe_mmq.hpp.  llama.cpp's MMQ (ggml-cuda, MIT) is compiled
// from the pinned llama.cpp checkout the build already takes ggml from; src/prefill/ggml_cuda_host.cu supplies the
// few host symbols of ggml-cuda.cu it references.
#include "strata/prefill/moe_mmq.hpp"

#include "common.cuh"
#include "mmq.cuh"
#include "quantize.cuh"

#include <cstdio>
#include <cstdlib>

namespace strata::prefill::mmq {
namespace {

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "prefill mmq: %s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

int64_t pad512(int64_t n) { return (n + 511) / 512 * 512; }

__global__ void copy16_kernel(const uint4* __restrict__ a, int64_t na, const uint4* __restrict__ b, int64_t nb,
                              uint4* __restrict__ ab_dst, const uint4* __restrict__ c, int64_t nc, uint4* __restrict__ c_dst) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < na) ab_dst[i] = a[i];
    else if (i < na + nb) ab_dst[i] = b[i - na];
    else if (i < na + nb + nc) c_dst[i - na - nb] = c[i - na - nb];
}
// copy16_kernel for an MMQ group: blockIdx.y is the expert (first + y)
struct GroupArgs {
    const uint8_t* blob[kGatherGroupMax];
    int64_t up_off, down_off, gu_stride, d_stride;   // in uint4
};
__global__ void copy16_group_kernel(GroupArgs ga, int first, int64_t na, int64_t nc, uint4* __restrict__ gu_dst,
                                    uint4* __restrict__ d_dst) {
    const int q = first + (int) blockIdx.y;
    const uint4* src = (const uint4*) ga.blob[q];
    uint4* ab = gu_dst + (int64_t) q * ga.gu_stride;
    uint4* cd = d_dst + (int64_t) q * ga.d_stride;
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < na) ab[i] = src[i];
    else if (i < 2 * na) ab[i] = src[ga.up_off + (i - na)];
    else if (i < 2 * na + nc) cd[i - 2 * na] = src[ga.down_off + (i - 2 * na)];
}
__global__ void copy1_kernel(const uint8_t* __restrict__ a, int64_t na, const uint8_t* __restrict__ b, int64_t nb,
                             uint8_t* __restrict__ ab_dst, const uint8_t* __restrict__ c, int64_t nc, uint8_t* __restrict__ c_dst) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < na) ab_dst[i] = a[i];
    else if (i < na + nb) ab_dst[i] = b[i - na];
    else if (i < na + nb + nc) c_dst[i - na - nb] = c[i - na - nb];
}

// Strata blob: gate/up codes [1280][640 B], down codes [2560][160 B], gate/up scales [1280][40] f16, down scales
// [2560][10] f16 (the layout of prefill/kernels.cu's blob_dequant_kernel).  A GGUF Q2_0 block is {f16 d; 16 code
// bytes} with the same 2-bit codes in the same order, so a block is a scale and a 16-byte run of codes.
__global__ void strata_q2_kernel(const uint8_t* __restrict__ blob, uint16_t* __restrict__ gu, uint16_t* __restrict__ dn) {
    constexpr size_t O_D_CODES = (size_t) 1280 * 640, O_GU_SC = O_D_CODES + (size_t) 2560 * 160,
                     O_D_SC = O_GU_SC + (size_t) 1280 * 40 * 2;
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;   // one GGUF block
    const int64_t n_gu = 1280LL * 40, n_d = 2560LL * 10;
    const uint8_t* codes;
    const uint16_t* scale;
    uint16_t* out;
    if (i < n_gu) {
        const int64_t row = i / 40, b = i % 40;
        codes = blob + row * 640 + b * 16;
        scale = (const uint16_t*) (blob + O_GU_SC) + row * 40 + b;
        out = gu + i * 9;
    } else if (i < n_gu + n_d) {
        const int64_t j = i - n_gu, row = j / 10, b = j % 10;
        codes = blob + O_D_CODES + row * 160 + b * 16;
        scale = (const uint16_t*) (blob + O_D_SC) + row * 10 + b;
        out = dn + j * 9;
    } else {
        return;
    }
    const uint4 q = *(const uint4*) codes;
    const uint16_t* qh = (const uint16_t*) &q;
    out[0] = *scale;
#pragma unroll
    for (int k = 0; k < 8; ++k) out[1 + k] = qh[k];
}

__global__ void swiglu_kernel(const float* __restrict__ gu, float* __restrict__ h, int64_t rows, int64_t n_ff,
                              bool interleaved) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * n_ff) return;
    const int64_t r = i / n_ff, k = i % n_ff;
    const float* row = gu + r * 2 * n_ff;
    const float g = interleaved ? row[2 * k] : row[k], u = interleaved ? row[2 * k + 1] : row[n_ff + k];
    h[i] = g / (1.0f + __expf(-g)) * u;
}

__global__ void iota_kernel(int32_t* dst, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = (int32_t) i;
}

unsigned blocks(int64_t n) { return (unsigned) ((n + 255) / 256); }

}  // namespace

bool built() { return true; }
void warm() { (void) ggml_cuda_info(); }

bool supported(int t) {
    switch ((ggml_type) t) {
        case GGML_TYPE_Q2_0:
#ifdef STRATA_ORCA_Q4KS_MMQ
        case GGML_TYPE_Q5_0:   // #296 (Q4_K and Q5_1: STRATA_MMQ_KQUANTS)
#endif
        case GGML_TYPE_IQ2_XXS: case GGML_TYPE_IQ2_XS: case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ3_XXS: case GGML_TYPE_IQ3_S: case GGML_TYPE_IQ4_NL: case GGML_TYPE_IQ4_XS:
        case GGML_TYPE_Q8_0:   // the draft layer's dense matrices (E-9)
#ifdef STRATA_MMQ_KQUANTS
        case GGML_TYPE_Q4_K: case GGML_TYPE_Q5_K: case GGML_TYPE_Q5_1:   // Unsloth's UD-Q4_K_XL experts (CUDA)
#endif
            return true;
#ifdef STRATA_MMQ_IQ1M_TILE
        case GGML_TYPE_IQ1_M: {
            // #169: llama.cpp's MMQ with third_party/ggml/patches' IQ1_M tile. It rounds the IQ1_M layers' activations
            // to q8_1 as MMQ does everywhere else, so greedy outputs move (#61 class): on by default since the developer
            // accepted that (2026-10-05); STRATA_MMQ_IQ1M=0 keeps the FP16 path
            static const bool on = [] { const char* v = std::getenv("STRATA_MMQ_IQ1M"); return v == nullptr || v[0] != '0'; }();
            return on;
        }
#endif
        default:
            return false;
    }
}

bool fits(int t, int64_t w_rows) {
    if (!supported(t)) return false;
    // mul_mat_q_case's choice: the "fallback" configs when the rows are not a multiple of 128; then
    // mul_mat_q_switch_J's loop - a tile size whose config exists for this card and fits its shared memory
    const bool fallback = w_rows % 128 != 0;
    const ggml_cuda_device_info& info = ggml_cuda_info();
    for (int id = 0; id < info.device_count; ++id) {
        const int cc = info.devices[id].cc;
        const size_t smpbo = info.devices[id].smpbo;
        bool any = false;
        for (int J = 8; J <= 128 && !any; J += 8) {
            const ggml_cuda_mmq_config c = ggml_cuda_mmq_get_config((ggml_type) t, J, fallback, cc);
            any = c.type != GGML_TYPE_COUNT && mmq_get_nbytes_shared(c, cc) <= smpbo;
        }
        if (!any) {
            static bool said[GGML_TYPE_COUNT] = {};
            if (t >= 0 && t < GGML_TYPE_COUNT && !said[t]) {
                said[t] = true;
                std::fprintf(stderr, "strata: prompt kernels: llama.cpp's MMQ has no tile for %s (%lld rows) on GPU %d "
                                     "(cc %d, %zu bytes of shared memory per block): that product takes the non-MMQ path "
                                     "(#420)\n", ggml_type_name((ggml_type) t), (long long) w_rows, id, cc, smpbo);
            }
            return false;
        }
    }
    return true;
}

size_t matrix_bytes(int t, int64_t rows, int64_t cols) {
    return (size_t) rows * (size_t) (cols / ggml_blck_size((ggml_type) t)) * ggml_type_size((ggml_type) t);
}

size_t q8_bytes(int64_t rows, int64_t cols) {
    return (size_t) rows * (size_t) pad512(cols) * sizeof(block_q8_1_mmq) / (4 * QK8_1) + 128 * sizeof(block_q8_1_mmq);
}

size_t q8_row_bytes(int64_t cols) { return (size_t) pad512(cols) / (4 * QK8_1) * sizeof(block_q8_1_mmq); }

namespace {
static_assert(sizeof(block_q8_1_mmq) % 16 == 0, "a q8_1 MMQ block is copied as 16-byte words");
constexpr int kQ8Words = (int) (sizeof(block_q8_1_mmq) / 16);
// block (c, r) of the destination = block (c, rows[r]) of the source; one thread per 16-byte word
__global__ void gather_q8_rows_kernel(const uint4* __restrict__ src, int64_t src_rows, const int32_t* __restrict__ rows,
                                      int64_t n, int64_t nblk, uint4* __restrict__ dst) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nblk * n * kQ8Words) return;
    const int64_t w = i % kQ8Words, rc = i / kQ8Words, r = rc % n, c = rc / n;
    dst[(c * n + r) * kQ8Words + w] = src[(c * src_rows + rows[r]) * kQ8Words + w];
}
}  // namespace

void gather_q8_rows(const void* src, int64_t src_rows, const int32_t* rows, int64_t n, int64_t cols, void* dst,
                    void* stream) {
    if (n <= 0) return;
    const int64_t nblk = pad512(cols) / (4 * QK8_1);
    gather_q8_rows_kernel<<<blocks(nblk * n * kQ8Words), 256, 0, (cudaStream_t) stream>>>(
        (const uint4*) src, src_rows, rows, n, nblk, (uint4*) dst);
    ck(cudaGetLastError(), "gather_q8_rows");
}

void quantize(const float* x, const int32_t* ids, void* xq, int t, int64_t cols, int64_t ld, int64_t rows, void* stream) {
    if (rows <= 0) return;
    quantize_mmq_q8_1_cuda(x, ids, xq, (ggml_type) t, cols, ld, rows * ld, rows * ld, pad512(cols), rows, 1, 1,
                           (cudaStream_t) stream);
    ck(cudaGetLastError(), "quantize");
}

Context::Context() {
    int dev = 0;
    cudaGetDevice(&dev);
    ctx_ = new ggml_backend_cuda_context(dev);
    // #5: stream-k sizes its grid from the SM count (36 on the 5060 Ti, 56 on the 4070 SUPER) and splits a tile's K
    // range across blocks, summed in float by the fixup - so the result depends on the card. With nsm = 1 every tile
    // gets its own block over the whole K range in one fixed order, and the products are bit-identical on sm_89 and
    // sm_120 (xeno_mmq_cross_arch: 0 of 6.4M values differ; with stream-k 7 of 8 cases differed). Same speed and the
    // same 8K output on the 5060 Ti (strata-claude-stage K*), so it is the default. STRATA_MMQ_STREAM_K=1: stream-k.
    // ggml-cuda serves only these MMQ products in this process, so the device table is ours to set.
    if (std::getenv("STRATA_MMQ_STREAM_K") == nullptr)
        const_cast<ggml_cuda_device_info&>(ggml_cuda_info()).devices[dev].nsm = 1;
}
Context::~Context() { delete (ggml_backend_cuda_context*) ctx_; }

void Context::run(const Product& p, void* stream) {
    if (p.n <= 0 || p.max_rows <= 0) return;
    const ggml_type t = (ggml_type) p.type;
    const int64_t qk = ggml_blck_size(t), bpr = p.w_cols / qk;
    const mmq_args a = {(const char*) p.w, t, (const int*) p.xq, p.ids, p.bounds, p.dst, nullptr,
                        p.w_cols, p.w_rows, p.total_rows, bpr, p.total_rows, p.ld_dst,
                        p.n, p.n, (int64_t) (p.expert_bytes / ggml_type_size(t)), 0, 0,
                        1, 1, 0, 0, 0,
                        p.max_rows, p.max_rows};
    auto& ctx = *(ggml_backend_cuda_context*) ctx_;
    const cudaStream_t s = (cudaStream_t) stream;
    switch (t) {
#ifdef STRATA_ORCA_Q4KS_MMQ
        case GGML_TYPE_Q5_0: mul_mat_q_case<GGML_TYPE_Q5_0>(ctx, a, s); break;
#endif
        case GGML_TYPE_Q2_0: mul_mat_q_case<GGML_TYPE_Q2_0>(ctx, a, s); break;
        case GGML_TYPE_IQ2_XXS: mul_mat_q_case<GGML_TYPE_IQ2_XXS>(ctx, a, s); break;
        case GGML_TYPE_IQ2_XS: mul_mat_q_case<GGML_TYPE_IQ2_XS>(ctx, a, s); break;
        case GGML_TYPE_IQ2_S: mul_mat_q_case<GGML_TYPE_IQ2_S>(ctx, a, s); break;
        case GGML_TYPE_IQ3_XXS: mul_mat_q_case<GGML_TYPE_IQ3_XXS>(ctx, a, s); break;
        case GGML_TYPE_IQ3_S: mul_mat_q_case<GGML_TYPE_IQ3_S>(ctx, a, s); break;
        case GGML_TYPE_IQ4_NL: mul_mat_q_case<GGML_TYPE_IQ4_NL>(ctx, a, s); break;
        case GGML_TYPE_IQ4_XS: mul_mat_q_case<GGML_TYPE_IQ4_XS>(ctx, a, s); break;
        case GGML_TYPE_Q8_0: mul_mat_q_case<GGML_TYPE_Q8_0>(ctx, a, s); break;
#ifdef STRATA_MMQ_IQ1M_TILE
        case GGML_TYPE_IQ1_M: mul_mat_q_case<GGML_TYPE_IQ1_M>(ctx, a, s); break;
#endif
#ifdef STRATA_MMQ_KQUANTS
        case GGML_TYPE_Q4_K: mul_mat_q_case<GGML_TYPE_Q4_K>(ctx, a, s); break;
        case GGML_TYPE_Q5_K: mul_mat_q_case<GGML_TYPE_Q5_K>(ctx, a, s); break;
        case GGML_TYPE_Q5_1: mul_mat_q_case<GGML_TYPE_Q5_1>(ctx, a, s); break;
#endif
        default:
            std::fprintf(stderr, "prefill mmq: type %d is not covered\n", (int) t);
            std::exit(1);
    }
    ck(cudaGetLastError(), "mul_mat_q");
}

void expert_rows_gate_up(Context& ctx, const ExpertRows& a, void* stream) {
    gather_q8_rows(a.xtok, a.xtok_rows, a.rows, a.nr, a.n_embd, a.xq, stream);
    Product gu;
    gu.w = a.gu; gu.type = a.gu_type; gu.w_rows = 2 * a.n_ff; gu.w_cols = a.n_embd; gu.expert_bytes = a.gu_bytes;
    gu.n = a.n; gu.xq = a.xq; gu.bounds = a.bounds; gu.ids = a.ids; gu.total_rows = a.nr; gu.max_rows = a.max_rows;
    gu.dst = a.gu_out; gu.ld_dst = 2 * a.n_ff;
    ctx.run(gu, stream);
    swiglu(a.gu_out, a.h, a.nr, a.n_ff, a.interleaved, stream);
}

void expert_rows_down(Context& ctx, const ExpertRows& a, void* stream) {
    quantize(a.h, nullptr, a.hq, a.down_type, a.n_ff, a.n_ff, a.nr, stream);
    Product dn;
    dn.w = a.down; dn.type = a.down_type; dn.w_rows = a.n_embd; dn.w_cols = a.n_ff; dn.expert_bytes = a.down_bytes;
    dn.n = a.n; dn.xq = a.hq; dn.bounds = a.bounds; dn.ids = a.ids; dn.total_rows = a.nr; dn.max_rows = a.max_rows;
    dn.dst = a.dst; dn.ld_dst = a.n_embd;
    ctx.run(dn, stream);
}

void expert_rows(Context& ctx, const ExpertRows& a, void* stream) {
    expert_rows_gate_up(ctx, a, stream);
    expert_rows_down(ctx, a, stream);
}

void gather_native(const void* gate, const void* up, size_t gu_half_bytes, const void* down, size_t d_bytes,
                   void* gu_dst, void* d_dst, void* stream) {
    const cudaStream_t s = (cudaStream_t) stream;
    const bool a16 = ((uintptr_t) gate | (uintptr_t) up | (uintptr_t) down | (uintptr_t) gu_dst | (uintptr_t) d_dst |
                      gu_half_bytes | d_bytes) % 16 == 0;
    if (a16) {
        const int64_t na = (int64_t) gu_half_bytes / 16, nc = (int64_t) d_bytes / 16;
        copy16_kernel<<<blocks(2 * na + nc), 256, 0, s>>>((const uint4*) gate, na, (const uint4*) up, na, (uint4*) gu_dst,
                                                          (const uint4*) down, nc, (uint4*) d_dst);
    } else {
        const int64_t na = (int64_t) gu_half_bytes, nc = (int64_t) d_bytes;
        copy1_kernel<<<blocks(2 * na + nc), 256, 0, s>>>((const uint8_t*) gate, na, (const uint8_t*) up, na,
                                                         (uint8_t*) gu_dst, (const uint8_t*) down, nc, (uint8_t*) d_dst);
    }
    ck(cudaGetLastError(), "gather_native");
}

namespace {
struct XenoBlobSet {   // xeno #29 (upstream #372's GatherGroup / copy16_group_kernel are the other overload)
    const uint4* blob[kGatherGroupMax];
};
// blockIdx.y = the expert: the same element walk as copy16_kernel, from that expert's blob into its slot
__global__ void copy16_blobs_kernel(XenoBlobSet g, int64_t up16, int64_t down16, int64_t na, int64_t nc,
                                    uint4* __restrict__ gu, int64_t gu_stride16, uint4* __restrict__ dn, int64_t d_stride16) {
    const int e = blockIdx.y;
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const uint4* b = g.blob[e];
    if (i < na) gu[e * gu_stride16 + i] = b[i];
    else if (i < 2 * na) gu[e * gu_stride16 + i] = b[up16 + i - na];
    else if (i < 2 * na + nc) dn[e * d_stride16 + i - 2 * na] = b[down16 + i - 2 * na];
}
}  // namespace

void gather_native_group(const uint8_t* const* blobs, int n, size_t up_off, size_t down_off, size_t gu_half_bytes,
                         size_t d_bytes, void* gu_dst, size_t gu_stride, void* d_dst, size_t d_stride, void* stream) {
    if (n <= 0) return;
    uintptr_t align = (uintptr_t) gu_dst | (uintptr_t) d_dst | up_off | down_off | gu_half_bytes | d_bytes | gu_stride |
                      d_stride;
    for (int i = 0; i < n; ++i) align |= (uintptr_t) blobs[i];
    if (n > kGatherGroupMax || align % 16 != 0) {   // the per-expert path (its own unaligned fallback included)
        for (int i = 0; i < n; ++i)
            gather_native(blobs[i], blobs[i] + up_off, gu_half_bytes, blobs[i] + down_off, d_bytes,
                          (uint8_t*) gu_dst + (size_t) i * gu_stride, (uint8_t*) d_dst + (size_t) i * d_stride, stream);
        return;
    }
    XenoBlobSet g{};
    for (int i = 0; i < n; ++i) g.blob[i] = (const uint4*) blobs[i];
    const int64_t na = (int64_t) gu_half_bytes / 16, nc = (int64_t) d_bytes / 16;
    copy16_blobs_kernel<<<dim3(blocks(2 * na + nc), (unsigned) n), 256, 0, (cudaStream_t) stream>>>(
        g, (int64_t) up_off / 16, (int64_t) down_off / 16, na, nc, (uint4*) gu_dst, (int64_t) gu_stride / 16,
        (uint4*) d_dst, (int64_t) d_stride / 16);
    ck(cudaGetLastError(), "gather_native_group");
}

bool gather_native_group(const GatherGroup& g, size_t up_off, size_t gu_half_bytes, size_t down_off, size_t d_bytes,
                         void* gu_dst, size_t gu_stride, void* d_dst, size_t d_stride, void* stream) {
    if (g.first < 0 || g.n <= g.first || g.n > kGatherGroupMax) return false;
    uintptr_t a = (uintptr_t) gu_dst | (uintptr_t) d_dst | up_off | gu_half_bytes | down_off | d_bytes | gu_stride | d_stride;
    for (int q = g.first; q < g.n; ++q) a |= (uintptr_t) g.blob[q];
    if (a % 16 != 0) return false;
    GroupArgs ga{};
    for (int q = g.first; q < g.n; ++q) ga.blob[q] = g.blob[q];
    ga.up_off = (int64_t) up_off / 16;
    ga.down_off = (int64_t) down_off / 16;
    ga.gu_stride = (int64_t) gu_stride / 16;
    ga.d_stride = (int64_t) d_stride / 16;
    const int64_t na = (int64_t) gu_half_bytes / 16, nc = (int64_t) d_bytes / 16;
    copy16_group_kernel<<<dim3(blocks(2 * na + nc), (unsigned) (g.n - g.first)), 256, 0, (cudaStream_t) stream>>>(
        ga, g.first, na, nc, (uint4*) gu_dst, (uint4*) d_dst);
    ck(cudaGetLastError(), "gather_native_group");
    return true;
}

void gather_strata_q2(const uint8_t* blob, void* gu_dst, void* d_dst, void* stream) {
    strata_q2_kernel<<<blocks(1280LL * 40 + 2560LL * 10), 256, 0, (cudaStream_t) stream>>>(blob, (uint16_t*) gu_dst,
                                                                                         (uint16_t*) d_dst);
    ck(cudaGetLastError(), "gather_strata_q2");
}

void swiglu(const float* gu, float* h, int64_t rows, int64_t n_ff, bool interleaved, void* stream) {
    if (rows <= 0) return;
    swiglu_kernel<<<blocks(rows * n_ff), 256, 0, (cudaStream_t) stream>>>(gu, h, rows, n_ff, interleaved);
    ck(cudaGetLastError(), "swiglu");
}

void iota(int32_t* dst, int64_t n, void* stream) {
    if (n <= 0) return;
    iota_kernel<<<blocks(n), 256, 0, (cudaStream_t) stream>>>(dst, n);
    ck(cudaGetLastError(), "iota");
}

}  // namespace strata::prefill::mmq
