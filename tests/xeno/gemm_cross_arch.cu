// #5: is the prompt path's expert GEMM bit-identical on the 5060 Ti (sm_120) and the 4070 SUPER (sm_89)?
// 4070-assisted prefill computes the 4070-owned experts there; if the kernel's result depends on the
// architecture, the prompt's KV differs from the 5060-only path and the output-parity gate cannot hold.
// Runs Gemm::f16 (the same handle setup as the prompt path: 32 MiB workspace, default math) on both cards
// with identical FP16 inputs over the expert shapes and a range of token counts, and compares every byte.
// Two fixed-order candidates run the same way: a tensor-core mma.sync m16n8k16 kernel (K walked in one order)
// and an FP32 FMA control. It is a probe, not a gate: it prints what differs and exits 0.
#include "strata/prefill/gemm.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {
bool check(cudaError_t e, const char* what) {
    if (e == cudaSuccess) return true;
    std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e));
    return false;
}

uint16_t half_bits(float f) {   // FP32 -> FP16 bits, round to nearest even, finite small values only
    uint32_t x;
    std::memcpy(&x, &f, 4);
    const uint32_t sign = (x >> 16) & 0x8000u;
    int32_t exp = (int32_t) ((x >> 23) & 0xff) - 127 + 15;
    uint32_t man = x & 0x7fffffu;
    if (exp <= 0) return (uint16_t) sign;
    uint32_t h = sign | ((uint32_t) exp << 10) | (man >> 13);
    const uint32_t rest = man & 0x1fffu;
    if (rest > 0x1000u || (rest == 0x1000u && (h & 1u))) ++h;
    return (uint16_t) h;
}

// Y[T, N] = X[T, K] . W[N, K]^T, one warp per 16x8 tile, K in steps of 16 in a fixed order (K % 16 == 0).
__global__ void mma_gemm(const uint16_t* X, const uint16_t* W, float* Y, int T, int N, int K) {
    const int lane = threadIdx.x, g = lane >> 2, t = lane & 3;
    const int m0 = blockIdx.y * 16, n0 = blockIdx.x * 8;
    auto xa = [&](int r, int c) -> uint32_t { return m0 + r < T ? X[(size_t) (m0 + r) * K + c] : 0u; };
    auto wb = [&](int k, int n) -> uint32_t { return n0 + n < N ? W[(size_t) (n0 + n) * K + k] : 0u; };
    float c0 = 0.f, c1 = 0.f, c2 = 0.f, c3 = 0.f;
    for (int k0 = 0; k0 < K; k0 += 16) {
        const int ca = k0 + t * 2;
        const uint32_t a0 = xa(g, ca) | xa(g, ca + 1) << 16, a1 = xa(g + 8, ca) | xa(g + 8, ca + 1) << 16;
        const uint32_t a2 = xa(g, ca + 8) | xa(g, ca + 9) << 16, a3 = xa(g + 8, ca + 8) | xa(g + 8, ca + 9) << 16;
        const uint32_t b0 = wb(ca, g) | wb(ca + 1, g) << 16, b1 = wb(ca + 8, g) | wb(ca + 9, g) << 16;
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                     "{%0,%1,%2,%3};"
                     : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                     : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
    }
    const int r0 = m0 + g, r1 = m0 + g + 8, n = n0 + t * 2;
    if (r0 < T && n < N) Y[(size_t) r0 * N + n] = c0;
    if (r0 < T && n + 1 < N) Y[(size_t) r0 * N + n + 1] = c1;
    if (r1 < T && n < N) Y[(size_t) r1 * N + n] = c2;
    if (r1 < T && n + 1 < N) Y[(size_t) r1 * N + n + 1] = c3;
}

__global__ void fma_gemm(const __half* X, const __half* W, float* Y, int T, int N, int K) {
    const int n = blockIdx.x * blockDim.x + threadIdx.x, m = blockIdx.y;
    if (n >= N) return;
    float acc = 0.f;
    for (int k = 0; k < K; ++k) acc = fmaf(__half2float(X[(size_t) m * K + k]), __half2float(W[(size_t) n * K + k]), acc);
    Y[(size_t) m * N + n] = acc;
}

enum Kind { CUBLAS, MMA, FMA };
const char* kind_name[3] = {"cublas", "mma", "fma"};

struct Device {
    int id = 0;
    cudaStream_t s = nullptr;
    strata::prefill::Gemm gemm;
    uint16_t *X = nullptr, *W = nullptr;
    float* Y = nullptr;
};

bool setup(Device& d, int id, size_t x_elems, size_t w_elems, size_t y_elems) {
    d.id = id;
    std::string err;
    if (!check(cudaSetDevice(id), "device") || !check(cudaStreamCreate(&d.s), "stream")) return false;
    if (!d.gemm.init(d.s, 0, err)) { std::fprintf(stderr, "gemm init: %s\n", err.c_str()); return false; }
    return check(cudaMalloc((void**) &d.X, x_elems * 2), "X") && check(cudaMalloc((void**) &d.W, w_elems * 2), "W") &&
           check(cudaMalloc((void**) &d.Y, y_elems * 4), "Y");
}

bool run(Device& d, Kind kind, const std::vector<uint16_t>& X, const std::vector<uint16_t>& W, int64_t T, int64_t N,
         int64_t K, std::vector<float>& Y) {
    cudaSetDevice(d.id);
    Y.assign((size_t) (T * N), 0.0f);
    if (!check(cudaMemcpyAsync(d.X, X.data(), (size_t) (T * K) * 2, cudaMemcpyHostToDevice, d.s), "X up") ||
        !check(cudaMemcpyAsync(d.W, W.data(), (size_t) (N * K) * 2, cudaMemcpyHostToDevice, d.s), "W up"))
        return false;
    if (kind == CUBLAS) {
        d.gemm.f16(d.X, d.W, d.Y, T, N, K);
    } else if (kind == MMA) {
        mma_gemm<<<dim3((unsigned) ((N + 7) / 8), (unsigned) ((T + 15) / 16)), 32, 0, d.s>>>(d.X, d.W, d.Y, (int) T,
                                                                                             (int) N, (int) K);
    } else {
        fma_gemm<<<dim3((unsigned) ((N + 127) / 128), (unsigned) T), 128, 0, d.s>>>(
            (const __half*) d.X, (const __half*) d.W, d.Y, (int) T, (int) N, (int) K);
    }
    return check(cudaGetLastError(), kind_name[kind]) &&
           check(cudaMemcpyAsync(Y.data(), d.Y, Y.size() * 4, cudaMemcpyDeviceToHost, d.s), "Y down") &&
           check(cudaStreamSynchronize(d.s), "sync");
}
}  // namespace

int main() {
    cudaDeviceProp p0{}, p1{};
    if (!check(cudaGetDeviceProperties(&p0, 0), "dev 0") || !check(cudaGetDeviceProperties(&p1, 1), "dev 1")) return 1;
    if (std::string(p0.name).find("5060 Ti") == std::string::npos ||
        std::string(p1.name).find("4070 SUPER") == std::string::npos) {
        std::fprintf(stderr, "expected CUDA_VISIBLE_DEVICES=1,0 (got %s / %s)\n", p0.name, p1.name);
        return 1;
    }
    // expert shapes: gate/up [T, n_embd] x [1280, n_embd]^T, down [T, 640] x [n_embd, 640]^T
    const int64_t embds[] = {16, 32, 2048, 2560, 4096};   // 16: one mma.sync k-step, no order to differ
    const int64_t toks[] = {1, 2, 3, 8, 17, 32, 64, 100, 128, 257, 512, 1024};
    const int64_t maxT = 1024, maxK = 4096, maxN = 4096;
    Device a, b;
    if (!setup(a, 0, (size_t) (maxT * maxK), (size_t) (maxN * maxK), (size_t) (maxT * maxN)) ||
        !setup(b, 1, (size_t) (maxT * maxK), (size_t) (maxN * maxK), (size_t) (maxT * maxN))) return 1;

    uint32_t rng = 12345u;
    auto rnd = [&] { rng = rng * 1664525u + 1013904223u; return ((float) (rng >> 8) / 16777216.0f - 0.5f); };
    int cases[3] = {0, 0, 0}, differ[3] = {0, 0, 0};
    float worst_vs_fma[3] = {0.f, 0.f, 0.f};
    for (int64_t embd : embds) {
        const int64_t shapes[2][2] = {{1280, embd}, {embd, 640}};   // {N, K}
        for (const auto& sh : shapes) {
            const int64_t N = sh[0], K = sh[1];
            std::vector<uint16_t> W((size_t) (N * K));
            for (auto& w : W) w = half_bits(rnd() * 0.2f);
            for (int64_t T : toks) {
                std::vector<uint16_t> X((size_t) (T * K));
                for (auto& x : X) x = half_bits(rnd() * 4.0f);
                std::vector<float> ref;
                for (int kind = FMA; kind >= CUBLAS; --kind) {
                    std::vector<float> ya, yb;
                    if (!run(a, (Kind) kind, X, W, T, N, K, ya) || !run(b, (Kind) kind, X, W, T, N, K, yb)) return 1;
                    if (kind == FMA) ref = ya;
                    size_t bad = 0;
                    float worst = 0.0f;
                    for (size_t i = 0; i < ya.size(); ++i) {
                        const float dr = ya[i] > ref[i] ? ya[i] - ref[i] : ref[i] - ya[i];
                        if (dr > worst_vs_fma[kind]) worst_vs_fma[kind] = dr;
                        if (std::memcmp(&ya[i], &yb[i], 4) != 0) {
                            ++bad;
                            const float dlt = ya[i] > yb[i] ? ya[i] - yb[i] : yb[i] - ya[i];
                            if (dlt > worst) worst = dlt;
                        }
                    }
                    ++cases[kind];
                    if (bad) ++differ[kind];
                    std::printf("%-6s N %5lld K %5lld T %5lld  %s  %zu of %zu differ, max |d| %.3g\n", kind_name[kind],
                                (long long) N, (long long) K, (long long) T, bad ? "DIFF " : "exact", bad, ya.size(),
                                worst);
                }
            }
        }
    }
    for (int kind = CUBLAS; kind <= FMA; ++kind)
        std::printf("%-6s %s vs %s: %d of %d cases bit-identical; max |d| vs the FMA control %.3g\n", kind_name[kind],
                    p0.name, p1.name, cases[kind] - differ[kind], cases[kind], worst_vs_fma[kind]);
    return 0;
}
