// Native Q2_0 row arithmetic on the target AVX-VNNI CPU, compared bitwise
// with the AVX2 fallback on the same deterministic bytes and activations.
#include "strata/kernels/cpu/expert.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace cpu = strata::kernels::cpu;

static bool check(int width, int rows, uint32_t seed) {
    constexpr int tokens = 3;
    const int blocks = width / cpu::QK;
    const size_t row_bytes = (size_t) blocks * cpu::BB;
    std::mt19937 rng(seed);
    std::normal_distribution<float> normal(0.f, 1.f);
    std::vector<uint8_t> weights((size_t) rows * row_bytes);
    for (int r = 0; r < rows; ++r)
        for (int b = 0; b < blocks; ++b) {
            uint8_t* w = weights.data() + (size_t) r * row_bytes + b * cpu::BB;
            const uint16_t scale = (uint16_t) (0x1c00 + ((r + b) % 7) * 0x400);
            std::memcpy(w, &scale, 2);
            for (int j = 2; j < cpu::BB; ++j) w[j] = (uint8_t) rng();
        }
    std::vector<cpu::ActQ> acts(tokens);
    std::vector<float> avx2((size_t) tokens * rows), vnni((size_t) tokens * rows);
    const cpu::ActQ* ap[tokens];
    float* aout[tokens];
    float* vout[tokens];
    for (int t = 0; t < tokens; ++t) {
        std::vector<float> x(width);
        for (float& v : x) v = normal(rng);
        cpu::act_quant_q8_1_avx2(x.data(), width, acts[(size_t) t]);
        ap[t] = &acts[(size_t) t];
        aout[t] = avx2.data() + (size_t) t * rows;
        vout[t] = vnni.data() + (size_t) t * rows;
    }
    cpu::q2_0_gguf_rows_multi_avx2(weights.data(), row_bytes, blocks, ap, tokens, aout, 0, rows);
    cpu::q2_0_gguf_rows_multi_avxvnni(weights.data(), row_bytes, blocks, ap, tokens, vout, 0, rows);
    const bool equal = std::memcmp(avx2.data(), vnni.data(), avx2.size() * sizeof(float)) == 0;
    std::printf("Q2_0 width=%d rows=%d tokens=%d: %s\n", width, rows, tokens, equal ? "bit-exact" : "DIFFERENT");
    return equal;
}

int main() {
    const bool gu = check(cpu::H, cpu::FF, 1107);
    const bool down = check(cpu::FF, cpu::H, 2718);
    return gu && down ? 0 : 1;
}
