// tests/xeno/mmvq_type_bench.cpp - #256: native_mmvq's time per call for Gyro-M's dense shapes in Q8_0 (as the GGUF
// stores them) against the lower-bit types the GEMV also takes (Q6_K, Q5_K, Q4_K), at the verify window's widths.
//
//     build/xeno_mmvq_type_bench [iters=200]
//
// Weights are random bytes with sane fp16 scales (mmvq_multi_parity's rule); 8 copies are cycled so every call reads
// its matrix from DRAM, not the 5060 Ti's 32 MB L2.  Prints us per call and the weight bytes' GB/s.  Timing only: the
// outputs are not compared.  A K-quant at 2..4 columns also gets an "(il)" line: native_mmvq_il, which the verify
// graph calls there instead of native_mmvq.
#include "strata/kernels/dequant_bf16.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/native_mmvq.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <random>
#include <vector>

namespace {

struct Type {
    const char* name;
    int type, block_elems, block_bytes, scale_at[2];
};
const Type TYPES[] = {
    {"Q8_0", 8, 32, 34, {0, -1}},
    {"Q6_K", 14, 256, 210, {208, -1}},
    {"Q5_K", 13, 256, 176, {0, 2}},
    {"Q4_K", 12, 256, 144, {0, 2}},
    {"IQ4XS", 23, 256, 136, {0, -1}},
    {"IQ4NL", 20, 32, 18, {0, -1}},
};
struct Shape {
    const char* name;
    int n_in, n_out;
};
// n_in x n_out as the verify graph calls them (GGUF dims are [n_in, n_out])
const Shape SHAPES[] = {
    {"attn_qkv", 2560, 10240}, {"attn_gate", 2560, 6144}, {"ssm_out", 6144, 2560},
    {"attn_q", 2560, 12288},   {"attn_output", 6144, 2560},
};
const int WIDTHS[] = {1, 2, 3, 4, 6};
constexpr int COPIES = 8;

uint16_t sane_half(std::mt19937& rng) {
    const uint32_t r = rng();
    return (uint16_t) (((r >> 31) << 15) | ((5u + (r >> 10) % 5u) << 10) | (r & 0x3ffu));
}

// float -> IEEE half, round to nearest even (as __float2half_rn), for the dequant parity check
uint16_t half_rne(float f) {
    uint32_t x;
    std::memcpy(&x, &f, 4);
    const uint32_t sign = (x >> 16) & 0x8000u;
    const uint32_t ax = x & 0x7fffffffu;
    if (ax >= 0x7f800000u) return (uint16_t) (sign | (ax > 0x7f800000u ? 0x7e00u : 0x7c00u));
    if (ax >= 0x477ff000u) return (uint16_t) (sign | 0x7c00u);   // rounds to inf
    if (ax < 0x38800000u) {                                      // half subnormal (or zero)
        if (ax < 0x33000000u) return (uint16_t) sign;
        const int e = (int) (ax >> 23);
        const uint32_t m = (ax & 0x7fffffu) | 0x800000u;
        const int shift = 126 - e;                               // m * 2^(e-150) in units of 2^-24
        uint32_t v = m >> shift;
        const uint32_t rem = m & ((1u << shift) - 1u), halfway = 1u << (shift - 1);
        if (rem > halfway || (rem == halfway && (v & 1u))) ++v;
        return (uint16_t) (sign | v);
    }
    uint32_t v = ((ax >> 13) - (112u << 10));
    const uint32_t rem = ax & 0x1fffu;
    if (rem > 0x1000u || (rem == 0x1000u && (v & 1u))) ++v;
    return (uint16_t) (sign | v);
}

}  // namespace

int main(int argc, char** argv) {
    const int iters = argc > 1 ? std::atoi(argv[1]) : 200;
    cudaStream_t s;
    cudaStreamCreate(&s);
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    std::mt19937 rng(7);
    std::printf("%-12s %-5s %3s %10s %9s %8s\n", "shape", "type", "T", "us/call", "GB/s", "vs Q8_0");
    for (const Shape& sh : SHAPES) {
        for (int T : WIDTHS) {
            double q8_us = 0;
            for (const Type& ty : TYPES) {
                if (!strata::kernels::native_mmvq_supported(ty.type) || sh.n_in % ty.block_elems != 0) continue;
                const size_t wb = strata::kernels::native_mmvq_weight_bytes(ty.type, sh.n_in, sh.n_out);
                std::vector<uint8_t> w(wb);
                for (auto& b : w) b = (uint8_t) (rng() & 0xff);
                const size_t nb = wb / (size_t) ty.block_bytes;
                for (size_t k = 0; k < nb; ++k)
                    for (int at : ty.scale_at)
                        if (at >= 0) {
                            const uint16_t h = sane_half(rng);
                            std::memcpy(&w[k * (size_t) ty.block_bytes + (size_t) at], &h, 2);
                        }
                void* dw[COPIES];
                for (int c = 0; c < COPIES; ++c) {
                    cudaMalloc(&dw[c], wb);
                    cudaMemcpy(dw[c], w.data(), wb, cudaMemcpyHostToDevice);
                }
                std::vector<float> x((size_t) T * sh.n_in);
                std::normal_distribution<float> nd(0.f, 1.f);
                for (auto& v : x) v = nd(rng);
                float* dx = nullptr;
                float* dy = nullptr;
                void* xq = nullptr;
                cudaMalloc(&dx, x.size() * 4);
                cudaMalloc(&dy, (size_t) T * sh.n_out * 4);
                cudaMalloc(&xq, strata::kernels::native_q8_1_bytes(sh.n_in, T));
                cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice);
                strata::kernels::quantize_q8_1_rows(dx, T, sh.n_in, xq, s);
                for (int i = 0; i < 20; ++i)
                    strata::kernels::native_mmvq(ty.type, dw[i % COPIES], xq, dy, sh.n_in, sh.n_out, T, s);
                cudaEventRecord(e0, s);
                for (int i = 0; i < iters; ++i)
                    strata::kernels::native_mmvq(ty.type, dw[i % COPIES], xq, dy, sh.n_in, sh.n_out, T, s);
                cudaEventRecord(e1, s);
                cudaEventSynchronize(e1);
                float ms = 0;
                cudaEventElapsedTime(&ms, e0, e1);
                const double us = 1000.0 * ms / iters;
                if (ty.type == 8) q8_us = us;
                std::printf("%-12s %-5s %3d %10.2f %9.1f %8.3f\n", sh.name, ty.name, T, us, (double) wb / (us * 1e3),
                            q8_us > 0 ? us / q8_us : 0.0);
                // the verify graph's call for 2..4 columns of a K-quant (verify.cpp `mm`): the interleaved kernel,
                // timed with its interleave (the graph writes it once per input; an upper bound per matrix)
                if (strata::kernels::native_mmvq_il_supported(ty.type, T, sh.n_out)) {
                    void* xil = nullptr;
                    cudaMalloc(&xil, strata::kernels::native_q8_1_il_bytes(sh.n_in, T));
                    for (int i = 0; i < 20; ++i) {
                        strata::kernels::native_q8_1_interleave(xq, xil, sh.n_in, T, s);
                        strata::kernels::native_mmvq_il(ty.type, dw[i % COPIES], xq, xil, dy, sh.n_in, sh.n_out, T, s);
                    }
                    cudaEventRecord(e0, s);
                    for (int i = 0; i < iters; ++i) {
                        strata::kernels::native_q8_1_interleave(xq, xil, sh.n_in, T, s);
                        strata::kernels::native_mmvq_il(ty.type, dw[i % COPIES], xq, xil, dy, sh.n_in, sh.n_out, T, s);
                    }
                    cudaEventRecord(e1, s);
                    cudaEventSynchronize(e1);
                    cudaEventElapsedTime(&ms, e0, e1);
                    const double il_us = 1000.0 * ms / iters;
                    std::printf("%-12s %-5s %3d %10.2f %9.1f %8.3f  (il)\n", sh.name, ty.name, T, il_us,
                                (double) wb / (il_us * 1e3), q8_us > 0 ? il_us / q8_us : 0.0);
                    cudaFree(xil);
                }
                for (int c = 0; c < COPIES; ++c) cudaFree(dw[c]);
                cudaFree(dx);
                cudaFree(dy);
                cudaFree(xq);
            }
        }
    }
    // the prompt path's dequant (dequant_f16 into the GEMM scratch, every chunk): us per matrix, the output's GB/s,
    // and its bits against dequant_f32 (the thread-per-group kernel) rounded to half on the host
    std::printf("\n%-12s %-5s %10s %9s %s\n", "shape", "type", "dq us", "out GB/s", "parity");
    int bad_total = 0;
    for (const Shape& sh : SHAPES) {
        for (const Type& ty : TYPES) {
            if (sh.n_in % ty.block_elems != 0) continue;
            const size_t wb = strata::kernels::native_mmvq_weight_bytes(ty.type, sh.n_in, sh.n_out);
            std::vector<uint8_t> w(wb);
            for (auto& b : w) b = (uint8_t) (rng() & 0xff);
            const size_t nb = wb / (size_t) ty.block_bytes;
            for (size_t k = 0; k < nb; ++k)
                for (int at : ty.scale_at)
                    if (at >= 0) {
                        const uint16_t h = sane_half(rng);
                        std::memcpy(&w[k * (size_t) ty.block_bytes + (size_t) at], &h, 2);
                    }
            const size_t n = (size_t) sh.n_in * sh.n_out;
            void* dw[COPIES];
            for (int c = 0; c < COPIES; ++c) {
                cudaMalloc(&dw[c], wb);
                cudaMemcpy(dw[c], w.data(), wb, cudaMemcpyHostToDevice);
            }
            uint16_t* d16 = nullptr;
            float* d32 = nullptr;
            cudaMalloc(&d16, n * 2);
            cudaMalloc(&d32, n * 4);
            // weights are [n_out rows][n_in cols]
            strata::kernels::dequant_f16(ty.type, dw[0], 0, sh.n_out, sh.n_in, d16, s);
            strata::kernels::dequant_f32(ty.type, dw[0], 0, sh.n_out, sh.n_in, d32, s);
            std::vector<uint16_t> h16(n);
            std::vector<float> h32(n);
            cudaMemcpy(h16.data(), d16, n * 2, cudaMemcpyDeviceToHost);
            cudaMemcpy(h32.data(), d32, n * 4, cudaMemcpyDeviceToHost);
            size_t bad = 0;
            for (size_t i = 0; i < n; ++i)
                if (h16[i] != half_rne(h32[i]) && !(std::isnan(h32[i]) && (h16[i] & 0x7c00u) == 0x7c00u)) ++bad;
            bad_total += bad != 0;
            for (int i = 0; i < 10; ++i)
                strata::kernels::dequant_f16(ty.type, dw[i % COPIES], 0, sh.n_out, sh.n_in, d16, s);
            cudaEventRecord(e0, s);
            for (int i = 0; i < iters; ++i)
                strata::kernels::dequant_f16(ty.type, dw[i % COPIES], 0, sh.n_out, sh.n_in, d16, s);
            cudaEventRecord(e1, s);
            cudaEventSynchronize(e1);
            float ms = 0;
            cudaEventElapsedTime(&ms, e0, e1);
            const double us = 1000.0 * ms / iters;
            std::printf("%-12s %-5s %10.2f %9.1f %s\n", sh.name, ty.name, us, (double) n * 2 / (us * 1e3),
                        bad ? "DIFFERS" : "equal");
            if (bad) std::printf("  %zu of %zu values differ\n", bad, n);
            for (int c = 0; c < COPIES; ++c) cudaFree(dw[c]);
            cudaFree(d16);
            cudaFree(d32);
        }
    }
    if (bad_total) std::printf("PARITY FAILED in %d matrices\n", bad_total);
    const cudaError_t err = cudaGetLastError();
    std::printf("%s\n", err == cudaSuccess ? "DONE" : cudaGetErrorString(err));
    return err == cudaSuccess ? 0 : 1;
}
