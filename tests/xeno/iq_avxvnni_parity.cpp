// tests/xeno/iq_avxvnni_parity.cpp - #106: the i-quant expert rows on AVX-VNNI (vpdpwssd for each token's madd + add)
// against the AVX2 rows, bitwise.
//
// Every format iq_avx2.cpp computes - IQ2_XXS (16), IQ2_XS (17), IQ3_XXS (18), IQ3_S (21), IQ2_S (22) - on random
// blocks (a finite fp16 scale, every other byte random, so every grid index, sign and sub-scale value occurs), both
// entry points (gate/up with SiLU, 640 rows x 2560; plain rows, 2560 x 2560 here), 1-7 tokens, rows from 3 on (the
// untouched rows must stay untouched).  Exit 0: identical everywhere; 77: no AVX-VNNI here; 1: a difference.
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/cpu/iq_avx2.hpp"

#define GGML_COMMON_DECL_CPP
#include "ggml-common.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace cpu = strata::kernels::cpu;

static size_t block_bytes(int type) {
    switch (type) {
        case 16: return sizeof(block_iq2_xxs);
        case 17: return sizeof(block_iq2_xs);
        case 18: return sizeof(block_iq3_xxs);
        case 21: return sizeof(block_iq3_s);
        case 22: return sizeof(block_iq2_s);
        default: return 0;
    }
}

static void fill_rows(std::vector<uint8_t>& w, int rows, size_t row_bytes, size_t bb, std::mt19937& rng) {
    w.resize((size_t) rows * row_bytes);
    for (auto& b : w) b = (uint8_t) rng();
    for (size_t off = 0; off < w.size(); off += bb) {   // each block starts with its fp16 d: keep it finite
        const uint16_t d = (uint16_t) (0x1400 + (rng() % 0x0800));   // 2^-10 .. 2^-8 range
        std::memcpy(w.data() + off, &d, 2);
    }
}

int main() {
    if (!cpu::cpu_avxvnni_ok()) {
        std::printf("SKIP: this CPU (%s) or build has no AVX-VNNI\n", cpu::cpu_name().c_str());
        return 77;
    }
    const int N = 2560, FF = 640, NB = N / QK_K;
    std::mt19937 rng(106);
    std::vector<block_q8_K> act((size_t) 7 * NB);
    for (auto& b : act) {
        b.d = 1.0f / 127.0f;
        for (auto& q : b.qs) q = (int8_t) ((int) (rng() % 255) - 127);
        std::memset(b.bsums, 0, sizeof(b.bsums));
    }
    const void* ap[7];
    for (int t = 0; t < 7; ++t) ap[t] = act.data() + (size_t) t * NB;
    int bad = 0, cases = 0;
    for (int type : {16, 17, 18, 21, 22}) {
        const size_t bb = block_bytes(type), row_bytes = bb * NB;
        std::vector<uint8_t> gu, w;
        fill_rows(gu, 2 * FF, row_bytes, bb, rng);   // gate rows, then up rows
        fill_rows(w, N, row_bytes, bb, rng);
        for (int nt = 1; nt <= 7; ++nt) {
            for (int which = 0; which < 2; ++which) {
                const int rows = which == 0 ? FF : N, r0 = 3;
                std::vector<float> ref((size_t) nt * rows, -1.f), got((size_t) nt * rows, -1.f);
                float* pr[7];
                float* pg[7];
                for (int t = 0; t < nt; ++t) { pr[t] = ref.data() + (size_t) t * rows; pg[t] = got.data() + (size_t) t * rows; }
                if (which == 0) {
                    cpu::iq256_gu_rows_avx2(type, gu.data(), row_bytes, (size_t) FF * row_bytes, N, ap, nt, pr, r0, rows);
                    cpu::iq256_gu_rows_avxvnni(type, gu.data(), row_bytes, (size_t) FF * row_bytes, N, ap, nt, pg, r0, rows);
                } else {
                    cpu::iq256_rows_avx2(type, w.data(), row_bytes, N, ap, nt, pr, r0, rows);
                    cpu::iq256_rows_avxvnni(type, w.data(), row_bytes, N, ap, nt, pg, r0, rows);
                }
                const bool same = std::memcmp(ref.data(), got.data(), ref.size() * sizeof(float)) == 0;
                ++cases;
                if (!same) {
                    ++bad;
                    std::printf("FAIL type %d %s, %d token(s): DIFFERENT\n", type, which == 0 ? "gate/up" : "rows", nt);
                }
            }
        }
    }
    if (bad == 0) std::printf("PASS: %d cases, the AVX-VNNI i-quant rows are bitwise the AVX2 rows\n", cases);
    else std::printf("FAIL: %d of %d cases differ\n", bad, cases);
    return bad ? 1 : 0;
}
