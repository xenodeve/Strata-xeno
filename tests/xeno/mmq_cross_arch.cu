// #5 / #12: which tensor-core paths give the same bits on the 5060 Ti (sm_120) and the 4070 SUPER (sm_89)?
// A floating mma.sync already differs in one k-step (gemm_cross_arch). This probe checks the two integer paths:
//  (a) mma.sync.m16n8k32 s8 x s8 -> s32, against an exact host reference and across the cards;
//  (b) the prompt path's own MMQ product (llama.cpp's int8 tensor-core kernels, Q2_0 weights x q8_1 activations,
//      several experts per launch as prefill runs it), across the cards.
// A probe, not a gate: it prints what differs and exits 0.  Needs CUDA_VISIBLE_DEVICES=1,0.
#include "strata/prefill/moe_mmq.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {
bool ok(cudaError_t e, const char* what) {
    if (e == cudaSuccess) return true;
    std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e));
    return false;
}

uint32_t rng = 2026u;
uint32_t next_u32() { rng = rng * 1664525u + 1013904223u; return rng; }
float next_f() { return (float) (next_u32() >> 8) / 16777216.0f - 0.5f; }

// Y[T, N] (s32) = X[T, K] (s8) . W[N, K]^T (s8), one warp per 16x8 tile, K % 32 == 0
__global__ void imma_gemm(const int8_t* X, const int8_t* W, int32_t* Y, int T, int N, int K) {
    const int lane = threadIdx.x, g = lane >> 2, t = lane & 3;
    const int m0 = blockIdx.y * 16, n0 = blockIdx.x * 8;
    auto pack_x = [&](int r, int c) -> uint32_t {
        uint32_t v = 0;
        for (int i = 0; i < 4; ++i) v |= (uint32_t) (uint8_t) (m0 + r < T ? X[(size_t) (m0 + r) * K + c + i] : 0) << (8 * i);
        return v;
    };
    auto pack_w = [&](int k, int n) -> uint32_t {
        uint32_t v = 0;
        for (int i = 0; i < 4; ++i) v |= (uint32_t) (uint8_t) (n0 + n < N ? W[(size_t) (n0 + n) * K + k + i] : 0) << (8 * i);
        return v;
    };
    int c0 = 0, c1 = 0, c2 = 0, c3 = 0;
    for (int k0 = 0; k0 < K; k0 += 32) {
        const uint32_t a0 = pack_x(g, k0 + t * 4), a1 = pack_x(g + 8, k0 + t * 4);
        const uint32_t a2 = pack_x(g, k0 + t * 4 + 16), a3 = pack_x(g + 8, k0 + t * 4 + 16);
        const uint32_t b0 = pack_w(k0 + t * 4, g), b1 = pack_w(k0 + t * 4 + 16, g);
        asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                     "{%0,%1,%2,%3};"
                     : "+r"(c0), "+r"(c1), "+r"(c2), "+r"(c3)
                     : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
    }
    const int r0 = m0 + g, r1 = m0 + g + 8, n = n0 + t * 2;
    if (r0 < T && n < N) Y[(size_t) r0 * N + n] = c0;
    if (r0 < T && n + 1 < N) Y[(size_t) r0 * N + n + 1] = c1;
    if (r1 < T && n < N) Y[(size_t) r1 * N + n] = c2;
    if (r1 < T && n + 1 < N) Y[(size_t) r1 * N + n + 1] = c3;
}

bool imma_on(int dev, const std::vector<int8_t>& X, const std::vector<int8_t>& W, int T, int N, int K,
             std::vector<int32_t>& Y) {
    cudaSetDevice(dev);
    int8_t *dx = nullptr, *dw = nullptr;
    int32_t* dy = nullptr;
    if (!ok(cudaMalloc((void**) &dx, X.size()), "x") || !ok(cudaMalloc((void**) &dw, W.size()), "w") ||
        !ok(cudaMalloc((void**) &dy, (size_t) T * N * 4), "y")) return false;
    cudaMemcpy(dx, X.data(), X.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(dw, W.data(), W.size(), cudaMemcpyHostToDevice);
    imma_gemm<<<dim3((unsigned) ((N + 7) / 8), (unsigned) ((T + 15) / 16)), 32>>>(dx, dw, dy, T, N, K);
    Y.assign((size_t) T * N, 0);
    const bool good = ok(cudaGetLastError(), "imma") && ok(cudaMemcpy(Y.data(), dy, Y.size() * 4, cudaMemcpyDeviceToHost), "y down");
    cudaFree(dx); cudaFree(dw); cudaFree(dy);
    return good;
}

// the prompt path's MMQ product: n experts of [w_rows, w_cols] Q2_0, rows[e] activation rows each
bool mmq_on(int dev, const std::vector<uint8_t>& W, int64_t w_rows, int64_t w_cols, const std::vector<float>& X,
            const std::vector<int32_t>& bounds, std::vector<float>& Y) {
    namespace mmq = strata::prefill::mmq;
    cudaSetDevice(dev);
    cudaStream_t s;
    if (!ok(cudaStreamCreate(&s), "stream")) return false;
    const int n = (int) bounds.size() - 1;
    const int64_t rows = bounds.back();
    int64_t maxr = 0;
    for (int e = 0; e < n; ++e) maxr = std::max<int64_t>(maxr, bounds[(size_t) e + 1] - bounds[(size_t) e]);
    void *dw = nullptr, *dxq = nullptr;
    float *dx = nullptr, *dy = nullptr;
    int32_t *db = nullptr, *dids = nullptr;
    const size_t xq_bytes = mmq::q8_bytes(rows, w_cols);
    if (!ok(cudaMalloc(&dw, W.size() + 4096), "w") || !ok(cudaMalloc((void**) &dx, X.size() * 4), "x") ||
        !ok(cudaMalloc(&dxq, xq_bytes), "xq") || !ok(cudaMalloc((void**) &dy, (size_t) rows * w_rows * 4), "y") ||
        !ok(cudaMalloc((void**) &db, bounds.size() * 4), "bounds") || !ok(cudaMalloc((void**) &dids, (size_t) rows * 4), "ids"))
        return false;
    cudaMemset(dw, 0, W.size() + 4096);   // the zeroed tail MMQ may read past the last expert
    cudaMemcpy(dw, W.data(), W.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(dx, X.data(), X.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(db, bounds.data(), bounds.size() * 4, cudaMemcpyHostToDevice);
    cudaMemset(dy, 0, (size_t) rows * w_rows * 4);
    mmq::iota(dids, rows, s);
    mmq::quantize(dx, nullptr, dxq, 42, w_cols, w_cols, rows, s);
    {
        mmq::Context ctx;
        mmq::Product p;
        p.w = dw; p.type = 42; p.w_rows = w_rows; p.w_cols = w_cols;
        p.expert_bytes = mmq::matrix_bytes(42, w_rows, w_cols);
        p.n = n; p.xq = dxq; p.bounds = db; p.ids = dids; p.total_rows = rows; p.max_rows = maxr;
        p.dst = dy; p.ld_dst = w_rows;
        ctx.run(p, s);
        if (!ok(cudaStreamSynchronize(s), "mmq")) return false;
    }
    Y.assign((size_t) rows * w_rows, 0.0f);
    const bool good = ok(cudaMemcpy(Y.data(), dy, Y.size() * 4, cudaMemcpyDeviceToHost), "y down");
    cudaFree(dw); cudaFree(dx); cudaFree(dxq); cudaFree(dy); cudaFree(db); cudaFree(dids);
    cudaStreamDestroy(s);
    return good;
}
}  // namespace

int main() {
    cudaDeviceProp p0{}, p1{};
    if (!ok(cudaGetDeviceProperties(&p0, 0), "dev 0") || !ok(cudaGetDeviceProperties(&p1, 1), "dev 1")) return 1;
    if (std::string(p0.name).find("5060 Ti") == std::string::npos ||
        std::string(p1.name).find("4070 SUPER") == std::string::npos) {
        std::fprintf(stderr, "expected CUDA_VISIBLE_DEVICES=1,0 (got %s / %s)\n", p0.name, p1.name);
        return 1;
    }
    std::printf("%s (%d SMs) vs %s (%d SMs)\n", p0.name, p0.multiProcessorCount, p1.name, p1.multiProcessorCount);

    // (a) integer mma.sync
    int ia_cases = 0, ia_exact_cross = 0, ia_exact_ref = 0;
    for (const int K : {32, 64, 640, 2560})
        for (const int T : {1, 7, 16, 100, 257}) {
            const int N = 1280;
            std::vector<int8_t> X((size_t) T * K), W((size_t) N * K);
            for (auto& v : X) v = (int8_t) ((int) (next_u32() % 255) - 127);
            for (auto& v : W) v = (int8_t) (next_u32() % 4);   // Q2_0 codes 0..3
            std::vector<int32_t> ya, yb, ref((size_t) T * N, 0);
            if (!imma_on(0, X, W, T, N, K, ya) || !imma_on(1, X, W, T, N, K, yb)) return 1;
            for (int r = 0; r < T; ++r)
                for (int c = 0; c < N; ++c) {
                    int32_t acc = 0;
                    for (int k = 0; k < K; ++k) acc += (int32_t) X[(size_t) r * K + k] * (int32_t) W[(size_t) c * K + k];
                    ref[(size_t) r * N + c] = acc;
                }
            ++ia_cases;
            ia_exact_cross += ya == yb;
            ia_exact_ref += ya == ref;
            std::printf("imma  K %5d T %4d  cross %s  vs exact reference %s\n", K, T, ya == yb ? "identical" : "DIFF",
                        ya == ref ? "exact" : "WRONG");
        }
    std::printf("imma: %d of %d cases identical across the cards, %d of %d equal to the exact reference\n", ia_exact_cross,
                ia_cases, ia_exact_ref, ia_cases);

    // (b) the prompt path's MMQ product: Q2_0 gate/up [1280, 2560] and down [2560, 640], several experts per launch
    int mq_cases = 0, mq_same = 0;
    struct Shape { int64_t rows, cols; } shapes[] = {{1280, 2560}, {2560, 640}};
    for (const Shape& sh : shapes)
        for (const int per : {1, 3, 40, 160}) {
            const int n = 8;
            const size_t eb = strata::prefill::mmq::matrix_bytes(42, sh.rows, sh.cols);
            std::vector<uint8_t> W(eb * n);
            for (size_t b = 0; b + 18 <= W.size(); b += 18) {   // block_q2_0: fp16 d, then 16 bytes of 2-bit codes
                const __half d = __float2half(0.01f + 0.02f * (next_f() + 0.5f));
                std::memcpy(&W[b], &d, 2);
                for (int i = 2; i < 18; ++i) W[b + i] = (uint8_t) next_u32();
            }
            std::vector<int32_t> bounds(n + 1, 0);
            for (int e = 0; e < n; ++e) bounds[(size_t) e + 1] = bounds[(size_t) e] + per + (e % 3);
            std::vector<float> X((size_t) bounds.back() * sh.cols);
            for (auto& v : X) v = next_f() * 4.0f;
            std::vector<float> ya, yb;
            if (!mmq_on(0, W, sh.rows, sh.cols, X, bounds, ya) || !mmq_on(1, W, sh.rows, sh.cols, X, bounds, yb)) return 1;
            size_t diff = 0;
            float worst = 0.0f;
            for (size_t i = 0; i < ya.size(); ++i)
                if (std::memcmp(&ya[i], &yb[i], 4) != 0) {
                    ++diff;
                    const float dl = ya[i] > yb[i] ? ya[i] - yb[i] : yb[i] - ya[i];
                    if (dl > worst) worst = dl;
                }
            ++mq_cases;
            mq_same += diff == 0;
            std::printf("mmq   [%lld x %lld] %d experts x ~%d rows: %s %zu of %zu differ, max |d| %.3g\n",
                        (long long) sh.rows, (long long) sh.cols, n, per, diff ? "DIFF " : "exact", diff, ya.size(), worst);
        }
    std::printf("mmq: %d of %d cases bit-identical across the cards\n", mq_same, mq_cases);
    return 0;
}
