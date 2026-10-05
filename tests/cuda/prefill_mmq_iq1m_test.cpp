// prefill_mmq_iq1m_test - #169: the prompt path's MMQ product for IQ1_M gate/up (the Swift 1.5 IQ2_XS pack's layers
// 8, 13 and 37, 1280 x 2560), against ggml's own IQ1_M dequantization.  IQ1_M had no MMQ tile, so those layers ran
// CUDA0's FP16 fallback and stayed off the two-card expert split (#113).
//
// The weights are random IQ1_M blocks: every byte pattern is a valid block (11-bit grid indices, 3-bit scales, the
// shift bits), and ggml cannot quantize IQ1_M without an importance matrix.  Two references, per output:
//   - exact: the weights from ggml's dequantize_row_iq1_m times the activations as MMQ's quantizer rounds them
//     (q8_1, one scale per 32 values: d = amax / 127, q = round(x * 127 / amax)), in double.  IQ1_M is
//     d16 * (8 g - 8 +- 1) / 8 with integer codes, so the tile holds the weights exactly and only the float sums differ;
//   - screen: the same weights times the float activations (the bound tests/cuda/prefill_mmq_kquant_test.cpp uses).
// Several experts per product, permuted rows, an all-zero row.
#include "strata/prefill/moe_mmq.hpp"

#include "ggml.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
namespace mmq = strata::prefill::mmq;

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

struct Dev {
    void* p = nullptr;
    explicit Dev(size_t n) { ck(cudaMalloc(&p, n), "cudaMalloc"); }
    ~Dev() { cudaFree(p); }
    Dev(const Dev&) = delete;
    Dev& operator=(const Dev&) = delete;
};

constexpr int64_t QK = 256;
constexpr size_t BLOCK = 56;   // ggml's block_iq1_m: qs[32], qh[16], scales[8] (the fp16 base in the top nibbles)

uint32_t mix(uint32_t x) {
    x ^= x >> 16; x *= 0x7feb352d; x ^= x >> 15; x *= 0x846ca68b; x ^= x >> 16;
    return x;
}

// rows x cols of random IQ1_M blocks; the base scale of each block in [0.004, 0.02)
std::vector<uint8_t> make_matrix(int64_t rows, int64_t cols, int expert, int trial) {
    const int64_t nb = rows * (cols / QK);
    std::vector<uint8_t> out((size_t) nb * BLOCK);
    for (int64_t b = 0; b < nb; ++b) {
        uint8_t* p = out.data() + (size_t) b * BLOCK;
        const uint32_t seed = (uint32_t) (b * 2654435761u) ^ (uint32_t) (expert * 40503 + trial * 9973 + 17);
        for (size_t i = 0; i < 48; ++i) p[i] = (uint8_t) mix(seed + (uint32_t) i);   // qs, qh: grid indices and shifts
        const float base = 0.004f + 0.016f * (float) (mix(seed ^ 0xabcdu) % 1000) / 1000.0f;
        const uint16_t h = ggml_fp32_to_fp16(base);
        uint16_t sc[4];
        for (int j = 0; j < 4; ++j) {
            const uint16_t three_bit_scales = (uint16_t) (mix(seed + 100u + (uint32_t) j) & 0x0fff);
            sc[j] = (uint16_t) (three_bit_scales | (((h >> (4 * j)) & 0xf) << 12));
        }
        std::memcpy(p + 48, sc, sizeof(sc));
    }
    return out;
}

std::vector<float> make_x(int rows, int64_t cols, int trial) {
    std::vector<float> x((size_t) rows * (size_t) cols);
    for (int r = 0; r < rows; ++r)
        for (int64_t k = 0; k < cols; ++k)
            x[(size_t) r * (size_t) cols + (size_t) k] =
                0.45f * std::sin(0.009f * (float) (k + 1) + 0.23f * (float) r + 0.07f * (float) trial) +
                0.17f * std::cos(0.021f * (float) (k + 3) - 0.19f * (float) r) +
                0.002f * (float) (((k * 7 + r * 13 + trial * 5) % 19) - 9);
    std::fill(x.end() - cols, x.end(), 0.0f);   // an all-zero row: exactly zero out
    return x;
}

// the activations as MMQ's q8_1 quantizer leaves them (one scale per 32 values)
std::vector<float> q8_1_round(const std::vector<float>& x) {
    std::vector<float> y(x.size());
    for (size_t b = 0; b < x.size(); b += 32) {
        float amax = 0.0f;
        for (size_t i = b; i < b + 32; ++i) amax = std::max(amax, std::fabs(x[i]));
        if (amax == 0.0f) continue;   // y stays 0
        const float d_inv = 127.0f / amax, d = 1.0f / d_inv;
        for (size_t i = b; i < b + 32; ++i) y[i] = d * std::round(x[i] * d_inv);
    }
    return y;
}

std::vector<int32_t> perm(int n, int shift, bool reverse) {
    std::vector<int32_t> p((size_t) n);
    for (int i = 0; i < n; ++i) p[(size_t) i] = (int32_t) (((reverse ? n - 1 - i : i) + shift) % n);
    return p;
}

void product(mmq::Context& ctx, cudaStream_t s, const std::string& name, int64_t out_rows, int64_t cols,
             const std::vector<int>& counts, int trial) {
    const ggml_type t = GGML_TYPE_IQ1_M;
    if (!mmq::supported((int) t)) throw std::runtime_error(name + ": IQ1_M is not covered by this build's MMQ");
    if (!mmq::fits((int) t, out_rows)) throw std::runtime_error(name + ": no MMQ tile fits");
    const int n = (int) counts.size(), rows = std::accumulate(counts.begin(), counts.end(), 0);
    std::vector<int32_t> bounds((size_t) n + 1, 0);
    for (int e = 0; e < n; ++e) bounds[(size_t) e + 1] = bounds[(size_t) e] + counts[(size_t) e];
    const auto src = perm(rows, trial + 1, false), dst = perm(rows, trial + 2, true);
    const std::vector<float> x = make_x(rows, cols, trial), xq = q8_1_round(x);
    const size_t eb = mmq::matrix_bytes((int) t, out_rows, cols);
    if (eb != (size_t) out_rows * ggml_row_size(t, cols) || ggml_row_size(t, QK) != BLOCK)
        throw std::runtime_error(name + ": matrix_bytes / block size");
    std::vector<std::vector<uint8_t>> ws;
    std::vector<uint8_t> w((size_t) n * eb + 4096, 0);
    for (int e = 0; e < n; ++e) {
        ws.push_back(make_matrix(out_rows, cols, e, trial));
        std::copy(ws.back().begin(), ws.back().end(), w.begin() + (ptrdiff_t) ((size_t) e * eb));
    }
    // the row tables padded as prefill.cpp's are (a tile reads the ids of its whole J rows, masked after the read)
    constexpr size_t PAD = 256;
    std::vector<int32_t> src_p(src), dst_p(dst);
    src_p.resize(src.size() + PAD, 0);
    dst_p.resize(dst.size() + PAD, 0);
    Dev dx(x.size() * 4), dsrc(src_p.size() * 4), ddst(dst_p.size() * 4), db(bounds.size() * 4), dw(w.size()),
        dxq(mmq::q8_bytes(rows, cols)), dy((size_t) rows * (size_t) out_rows * 4);
    ck(cudaMemcpy(dx.p, x.data(), x.size() * 4, cudaMemcpyHostToDevice), "x");
    ck(cudaMemcpy(dsrc.p, src_p.data(), src_p.size() * 4, cudaMemcpyHostToDevice), "src");
    ck(cudaMemcpy(ddst.p, dst_p.data(), dst_p.size() * 4, cudaMemcpyHostToDevice), "dst");
    ck(cudaMemcpy(db.p, bounds.data(), bounds.size() * 4, cudaMemcpyHostToDevice), "bounds");
    ck(cudaMemcpy(dw.p, w.data(), w.size(), cudaMemcpyHostToDevice), "w");
    // a pageable cudaMemcpy returns before its DMA lands, and the product's non-blocking stream does not wait for it:
    // without this the kernels read the previous product's freed buffers at the same addresses (order-dependent)
    ck(cudaDeviceSynchronize(), "uploads");
    // on the product's own stream: a legacy-stream memset is not ordered with a non-blocking stream
    ck(cudaMemsetAsync(dy.p, 0xff, (size_t) rows * (size_t) out_rows * 4, s), "sentinel");
    mmq::quantize((const float*) dx.p, (const int32_t*) dsrc.p, dxq.p, (int) t, cols, cols, rows, s);
    mmq::Product p;
    p.w = dw.p; p.type = (int) t; p.w_rows = out_rows; p.w_cols = cols; p.expert_bytes = eb; p.n = n;
    p.xq = dxq.p; p.bounds = (const int32_t*) db.p; p.ids = (const int32_t*) ddst.p; p.total_rows = rows;
    p.max_rows = *std::max_element(counts.begin(), counts.end()); p.dst = (float*) dy.p; p.ld_dst = out_rows;
    ctx.run(p, s);
    ck(cudaGetLastError(), "launch");
    ck(cudaStreamSynchronize(s), "sync");
    std::vector<float> got((size_t) rows * (size_t) out_rows);
    ck(cudaMemcpy(got.data(), dy.p, got.size() * 4, cudaMemcpyDeviceToHost), "y");

    const auto* tr = ggml_get_type_traits(t);
    std::vector<double> exact(got.size(), 0.0), screen(got.size(), 0.0);
    std::vector<float> wd((size_t) out_rows * (size_t) cols);
    for (int e = 0; e < n; ++e) {
        const size_t rb = ggml_row_size(t, cols);
        for (int64_t o = 0; o < out_rows; ++o)
            tr->to_float(ws[(size_t) e].data() + (size_t) o * rb, wd.data() + (size_t) o * (size_t) cols, cols);
        for (int r = bounds[(size_t) e]; r < bounds[(size_t) e + 1]; ++r) {
            const size_t xr = (size_t) src[(size_t) r] * (size_t) cols;
            for (int64_t o = 0; o < out_rows; ++o) {
                double a = 0, b = 0;
                const float* wr = wd.data() + (size_t) o * (size_t) cols;
                for (int64_t k = 0; k < cols; ++k) {
                    a += (double) wr[k] * xq[xr + (size_t) k];
                    b += (double) wr[k] * x[xr + (size_t) k];
                }
                exact[(size_t) dst[(size_t) r] * (size_t) out_rows + (size_t) o] = a;
                screen[(size_t) dst[(size_t) r] * (size_t) out_rows + (size_t) o] = b;
            }
        }
    }
    double e2 = 0, s2 = 0, ref2 = 0, max_exact = 0, zero_max = 0;
    size_t bad = 0, first_bad = 0;
    for (size_t i = 0; i < got.size(); ++i)
        if (!std::isfinite(got[i]) && bad++ == 0) first_bad = i;
    if (bad)
        throw std::runtime_error(name + ": " + std::to_string(bad) + " unwritten or non-finite outputs, the first at dst row " +
                                 std::to_string(first_bad / (size_t) out_rows) + " column " +
                                 std::to_string(first_bad % (size_t) out_rows));
    for (size_t i = 0; i < got.size(); ++i) {
        e2 += (got[i] - exact[i]) * (got[i] - exact[i]);
        s2 += (got[i] - screen[i]) * (got[i] - screen[i]);
        ref2 += screen[i] * screen[i];
        max_exact = std::max(max_exact, std::fabs(got[i] - exact[i]));
    }
    for (int r = 0; r < rows; ++r)
        if (src[(size_t) r] == rows - 1)   // the zero row
            for (int64_t o = 0; o < out_rows; ++o)
                zero_max = std::max(zero_max, (double) std::fabs(got[(size_t) dst[(size_t) r] * (size_t) out_rows + (size_t) o]));
    const double rms = std::sqrt(ref2 / (double) got.size());
    const double rel_exact = std::sqrt(e2 / (double) got.size()) / rms, rel_screen = std::sqrt(s2 / (double) got.size()) / rms;
    std::printf("%-26s rows %2d experts %d  ref_rms %.4g  rel_l2 exact %.2e  max/rms %.2e  screen %.5f  zero row %.3g\n",
                name.c_str(), rows, n, rms, rel_exact, max_exact / rms, rel_screen, zero_max);
    if (rel_exact > 1e-5 || max_exact / rms > 1e-4) throw std::runtime_error(name + ": not the IQ1_M weights");
    if (rel_screen > 0.04 || zero_max > std::max(1e-5, 1e-4 * rms)) throw std::runtime_error(name + ": outside the screen");
}
}  // namespace

int main() {
    try {
        if (!mmq::built()) { std::printf("no MMQ in this build\n"); return 1; }
        cudaStream_t s = nullptr;
        ck(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking), "stream");
        {
            mmq::Context ctx;
            const std::vector<std::vector<int>> batches{{1, 3, 3}, {4, 1, 2}, {2, 3}, {17}, {40, 9}};
            int trial = 0;
            for (const auto& counts : batches) {
                product(ctx, s, "IQ1_M gate/up-" + std::to_string(trial), 1280, 2560, counts, trial);
                ++trial;
            }
        }
        ck(cudaStreamDestroy(s), "destroy");
        std::printf("prefill MMQ IQ1_M test passed\n");
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "prefill MMQ IQ1_M test failed: %s\n", e.what());
        return 1;
    }
}
