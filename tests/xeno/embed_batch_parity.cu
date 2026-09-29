// #31: the prompt path embeds a chunk with one iq_embed_rows launch (the verify window's and the drafter's gather)
// instead of one iq_dequant_f32 launch per token (8,023 launches, ~0.6 s of host time before layer 0 on an 8K
// prompt, docs/reports/2026-09-29-pipeline-timeline.md).  The two must write the same bytes for every IQ type the
// model files use: random blocks, random token ids (repeats included), memcmp of every row.
#include "strata/kernels/iq_kernels.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

int main() {
    const int types[] = {16, 17, 18, 20, 21, 22, 23, 29, 42};
    const int64_t n_embd = 2048, n_vocab = 512, n_tok = 300;
    std::mt19937 rng(33);
    int fails = 0;
    for (int t : types) {
        const size_t row = strata::kernels::iq_row_bytes(t, (int) n_embd);
        std::vector<uint8_t> table(row * (size_t) n_vocab);
        for (auto& b : table) b = (uint8_t) rng();
        // fp16 block scales from random bytes can be inf/NaN: keep them, both paths must copy the same bits
        std::vector<int32_t> tok((size_t) n_tok);
        for (auto& x : tok) x = (int32_t) (rng() % (uint32_t) n_vocab);
        uint8_t* d_table = nullptr;
        int32_t* d_tok = nullptr;
        float *d_a = nullptr, *d_b = nullptr;
        cudaMalloc(&d_table, table.size());
        cudaMalloc(&d_tok, tok.size() * 4);
        cudaMalloc(&d_a, (size_t) n_tok * n_embd * 4);
        cudaMalloc(&d_b, (size_t) n_tok * n_embd * 4);
        cudaMemcpy(d_table, table.data(), table.size(), cudaMemcpyHostToDevice);
        cudaMemcpy(d_tok, tok.data(), tok.size() * 4, cudaMemcpyHostToDevice);
        cudaMemset(d_a, 0xAB, (size_t) n_tok * n_embd * 4);
        cudaMemset(d_b, 0xCD, (size_t) n_tok * n_embd * 4);
        strata::kernels::iq_embed_rows(t, d_table, row, d_tok, n_tok, n_embd, d_a, nullptr);
        for (int64_t i = 0; i < n_tok; ++i)
            strata::kernels::iq_dequant_f32(t, d_table + (size_t) tok[(size_t) i] * row, n_embd, d_b + i * n_embd, nullptr);
        std::vector<float> a((size_t) n_tok * n_embd), b(a.size());
        cudaMemcpy(a.data(), d_a, a.size() * 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(b.data(), d_b, b.size() * 4, cudaMemcpyDeviceToHost);
        const cudaError_t e = cudaGetLastError();
        size_t diff = 0;
        for (size_t i = 0; i < a.size(); ++i) diff += std::memcmp(&a[i], &b[i], 4) != 0;
        std::printf("type %2d: %zu of %zu values differ%s%s\n", t, diff, a.size(), e != cudaSuccess ? ": " : "",
                    e != cudaSuccess ? cudaGetErrorString(e) : "");
        fails += diff != 0 || e != cudaSuccess;
        cudaFree(d_table);
        cudaFree(d_tok);
        cudaFree(d_a);
        cudaFree(d_b);
    }
    std::printf(fails == 0 ? "PASS\n" : "FAIL\n");
    return fails == 0 ? 0 : 1;
}
