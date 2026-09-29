// #5 prefill launch count: the QSA indexer append of a whole prompt chunk in one launch must leave the indexer
// state (tail, dead, pooled, block_pos) byte-identical to the token-by-token appends it replaces. The prompt path
// launched one append per token (2,047 launches per attention layer at 2K), and those launches, not the GPU,
// were the cost (nsys, strata-claude-prefill-nsys24).
#include "strata/kernels/native_qsa_indexer.hpp"
#include "strata/kernels/qsa.hpp"

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {
using namespace strata::kernels;

bool ok(cudaError_t e, const char* what) {
    if (e == cudaSuccess) return true;
    std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e));
    return false;
}

struct State {
    float *tail = nullptr, *dead = nullptr, *pooled = nullptr;
    int32_t* block_pos = nullptr;
    size_t pooled_floats = 0;
    bool alloc(int64_t max_cells) {
        pooled_floats = (size_t) (max_cells / 4 + 1) * 128;
        return ok(cudaMalloc((void**) &tail, 3 * 128 * 4), "tail") && ok(cudaMalloc((void**) &dead, 128 * 4), "dead") &&
               ok(cudaMalloc((void**) &pooled, pooled_floats * 4), "pooled") &&
               ok(cudaMalloc((void**) &block_pos, 4), "block_pos") && ok(cudaMemset(tail, 0, 3 * 128 * 4), "t0") &&
               ok(cudaMemset(dead, 0, 128 * 4), "d0") && ok(cudaMemset(pooled, 0, pooled_floats * 4), "p0") &&
               ok(cudaMemset(block_pos, 0, 4), "b0");
    }
    QsaIndexerBuffers buffers() const { return QsaIndexerBuffers{tail, dead, pooled, block_pos}; }
    std::vector<uint8_t> bytes() const {
        std::vector<uint8_t> out(3 * 128 * 4 + 128 * 4 + pooled_floats * 4 + 4);
        uint8_t* p = out.data();
        cudaMemcpy(p, tail, 3 * 128 * 4, cudaMemcpyDeviceToHost); p += 3 * 128 * 4;
        cudaMemcpy(p, dead, 128 * 4, cudaMemcpyDeviceToHost); p += 128 * 4;
        cudaMemcpy(p, pooled, pooled_floats * 4, cudaMemcpyDeviceToHost); p += pooled_floats * 4;
        cudaMemcpy(p, block_pos, 4, cudaMemcpyDeviceToHost);
        return out;
    }
};
}  // namespace

int main() {
    const QsaShapes s = qsa_real_shapes();
    const int64_t max_cells = 8192;
    const float eps = 1e-6f, freq = qsa_freq_base();
    cudaStream_t st;
    if (!ok(cudaStreamCreate(&st), "stream")) return 1;
    uint32_t rng = 7u;
    auto rnd = [&] { rng = rng * 1664525u + 1013904223u; return (float) (rng >> 8) / 16777216.0f - 0.5f; };
    std::vector<float> gamma(128);
    for (auto& g : gamma) g = 1.0f + rnd();
    float* d_gamma = nullptr;
    if (!ok(cudaMalloc((void**) &d_gamma, 128 * 4), "gamma")) return 1;
    cudaMemcpy(d_gamma, gamma.data(), 128 * 4, cudaMemcpyHostToDevice);

    // a first chunk from position 0 (the spare path) and a second, unaligned-length chunk after it
    const int64_t chunks[][2] = {{0, 1027}, {1027, 513}};   // {first position, tokens}
    State a, b;
    if (!a.alloc(max_cells) || !b.alloc(max_cells)) return 1;
    int failures = 0;
    for (const auto& ch : chunks) {
        const int64_t p0 = ch[0], T = ch[1];
        std::vector<float> raw((size_t) T * 128);
        for (auto& r : raw) r = rnd() * 8.0f;
        std::vector<int32_t> steps((size_t) T * kStepCount, 0);
        for (int64_t t = 0; t < T; ++t) steps[(size_t) (t * kStepCount + kStepPos)] = (int32_t) (p0 + t);
        float* d_raw = nullptr;
        int32_t* d_steps = nullptr;
        if (!ok(cudaMalloc((void**) &d_raw, raw.size() * 4), "raw") ||
            !ok(cudaMalloc((void**) &d_steps, steps.size() * 4), "steps")) return 1;
        cudaMemcpy(d_raw, raw.data(), raw.size() * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(d_steps, steps.data(), steps.size() * 4, cudaMemcpyHostToDevice);
        for (int64_t t = 0; t < T; ++t)
            native_qsa_indexer_append(d_raw + t * 128, d_steps + t * kStepCount + kStepPos, 0, d_gamma, eps, a.buffers(), s,
                                      max_cells, freq, st);
        native_qsa_indexer_append_batch(d_raw, T, d_steps + kStepPos, kStepCount, 0, d_gamma, eps, b.buffers(), s,
                                        max_cells, freq, st);
        if (!ok(cudaStreamSynchronize(st), "sync")) return 1;
        const std::vector<uint8_t> x = a.bytes(), y = b.bytes();
        size_t diff = 0;
        for (size_t i = 0; i < x.size(); ++i) diff += x[i] != y[i];
        std::printf("chunk at %lld, %lld tokens: %zu of %zu state bytes differ\n", (long long) p0, (long long) T, diff,
                    x.size());
        failures += diff != 0;
        cudaFree(d_raw);
        cudaFree(d_steps);
    }
    std::printf(failures ? "FAIL\n" : "PASS\n");
    return failures ? 1 : 0;
}
