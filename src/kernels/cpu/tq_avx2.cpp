// src/kernels/cpu/tq_avx2.cpp - the trellis expert rows (TQ2_T / TQK6 / TQK7) on AVX2 + FMA (tq_cpu.hpp).
// Compiled for AVX2 only and reached behind cpu_avx2_ok(), so nothing here can fault on an older CPU; the AVX-VNNI
// copy is compiled under its own target attribute and called only where cpu_avxvnni_ok().
//
// Per 128-weight block: the 32 states in scalar code (tq_states.hpp: one 32-bit read, a shift and a mask each, used
// as table indices straight away), four 8-byte state-table rows per YMM register, then per token VPMADDWD against
// the unpacked int16 activations and a VPHADDD tree that leaves the four exact sub-block sums S[k] in one XMM, so the
// float part is the scalar kernel's, lane for lane.  The weights are decoded once per block for all the tokens of
// the group.  The body is tq_avx2_rows.inl, in four variants (xeno #220): the rows gathered or assembled (G), and
// VPDPWSSD where the CPU has AVX-VNNI - the fork's IQ rule (iq_avx2.cpp: iq256_variant).  The same bits in all four.
//
// Measured on Zen 5 (cycles per block, one token, AVX2): ~48 with the assembled rows; computing the states with byte
// shuffles instead (VPSHUFB / VPSRLVD per 8 states) and moving them to GPRs or through memory was slower (60-80),
// and ggml-cpu's AVX2 vec_dot for these types (agentionai llama.cpp, the same state table) takes about 2.3x as long.
#include "strata/kernels/cpu/tq_cpu.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"

#include "tq_states.hpp"

#include <cmath>
#include <cstdlib>
#include <immintrin.h>
#include <utility>

// STRATA_AVXVNNI (CMake: the compiler has the AVX-VNNI intrinsics), as in iq_avx2.cpp
#if !defined(STRATA_AVXVNNI)
#define STRATA_AVXVNNI 0
#endif
#if STRATA_AVXVNNI && (defined(__GNUC__) || defined(__clang__))
#define STRATA_AVXVNNI_FN __attribute__((target("avxvnni")))
#else
#define STRATA_AVXVNNI_FN
#endif

// the row kernel's helpers must inline into it (MSVC otherwise leaves decode8 a call, wv[] in memory: xeno #220)
#if defined(_MSC_VER) && !defined(__clang__)
#define STRATA_TQ_INLINE __forceinline
#else
#define STRATA_TQ_INLINE inline __attribute__((always_inline))
#endif

namespace strata::kernels::cpu::tq_detail {
namespace {

namespace plain {
#define STRATA_TQ_ROWS_VNNI 0
#define STRATA_TQ_ROWS_FN
#include "tq_avx2_rows.inl"
#undef STRATA_TQ_ROWS_VNNI
#undef STRATA_TQ_ROWS_FN
}  // namespace plain
#if STRATA_AVXVNNI
namespace vnni {
#define STRATA_TQ_ROWS_VNNI 1
#define STRATA_TQ_ROWS_FN STRATA_AVXVNNI_FN
#include "tq_avx2_rows.inl"
#undef STRATA_TQ_ROWS_VNNI
#undef STRATA_TQ_ROWS_FN
}  // namespace vnni
#endif

}  // namespace

int tq_avx2_variant() noexcept {
    // AVX-VNNI where cpu_avxvnni_ok() (STRATA_NO_AVXVNNI=1 turns it off).  Gather (STRATA_TQ_GATHER=1) and the codebook
    // (STRATA_TQ_CODEBOOK=1) are off unless asked for: on an i5-13500 (tq_cpu_parity --bench, TQ2_T, one token) the
    // gathered rows cost 557 against 459 us per expert on a P-core and 3,438 against 1,532 on an E-core, and the
    // codebook 884 / 1,760 - the table rows are not the bound once the decode is unrolled (xeno #220).
    static const int gather = [] {
        const char* v = std::getenv("STRATA_TQ_GATHER");
        return v != nullptr && *v != '\0' ? (std::atoi(v) != 0 ? 1 : 0) : -1;
    }();
    static const bool vnni = STRATA_AVXVNNI && cpu_avxvnni_ok();
    static const int codebook = [] {
        const char* v = std::getenv("STRATA_TQ_CODEBOOK");
        return v != nullptr && *v != '\0' ? (std::atoi(v) != 0 ? 1 : 0) : -1;
    }();
    // TQ2_T's q64 decode is the default; STRATA_TQ_BYTE_STATES=1 brings back the byte-wise one for A/B (kTqByteStates)
    static const bool byte_states = [] {
        const char* v = std::getenv("STRATA_TQ_BYTE_STATES");
        return v != nullptr && std::atoi(v) != 0;
    }();
    return (gather == 1 ? kTqGather : 0) | (vnni ? kTqVnni : 0) | (codebook == 1 ? kTqCodebook : 0) |
           (byte_states ? kTqByteStates : 0);
}

namespace {
template <bool V>
void rows_any(int variant, int kb, const uint8_t* w, size_t row_bytes, int n, const TqAct* const* a, int nt,
              float* const* out, int r0, int r1) {
    const bool g = (variant & kTqGather) != 0, c = (variant & kTqCodebook) != 0;
    const bool q = (variant & kTqByteStates) == 0;   // q64 (TQ2_T, assembled rows, the state table)
#if STRATA_AVXVNNI
    if constexpr (V) {
        if (g) c ? vnni::rows_kb<true, true>(kb, w, row_bytes, n, a, nt, out, r0, r1)
                 : vnni::rows_kb<true, false>(kb, w, row_bytes, n, a, nt, out, r0, r1);
        else if (c) vnni::rows_kb<false, true>(kb, w, row_bytes, n, a, nt, out, r0, r1);
        else q ? vnni::rows_kb<false, false, true>(kb, w, row_bytes, n, a, nt, out, r0, r1)
               : vnni::rows_kb<false, false>(kb, w, row_bytes, n, a, nt, out, r0, r1);
        return;
    }
#endif
    if (g) c ? plain::rows_kb<true, true>(kb, w, row_bytes, n, a, nt, out, r0, r1)
             : plain::rows_kb<true, false>(kb, w, row_bytes, n, a, nt, out, r0, r1);
    else if (c) plain::rows_kb<false, true>(kb, w, row_bytes, n, a, nt, out, r0, r1);
    else q ? plain::rows_kb<false, false, true>(kb, w, row_bytes, n, a, nt, out, r0, r1)
           : plain::rows_kb<false, false>(kb, w, row_bytes, n, a, nt, out, r0, r1);
}
}  // namespace

void tq_rows_avx2_v(int variant, int kb, const uint8_t* w, size_t row_bytes, int n, const TqAct* const* a, int nt,
                    float* const* out, int r0, int r1) {
#if STRATA_AVXVNNI
    if ((variant & kTqVnni) && cpu_avxvnni_ok()) {
        rows_any<true>(variant, kb, w, row_bytes, n, a, nt, out, r0, r1);
        return;
    }
#endif
    rows_any<false>(variant, kb, w, row_bytes, n, a, nt, out, r0, r1);
}

void tq_rows_avx2(int kb, const uint8_t* w, size_t row_bytes, int n, const TqAct* const* a, int nt,
                  float* const* out, int r0, int r1) {
    tq_rows_avx2_v(tq_avx2_variant(), kb, w, row_bytes, n, a, nt, out, r0, r1);
}

}  // namespace strata::kernels::cpu::tq_detail

namespace strata::kernels::cpu {

// The APR rotation (hadamard.cpp) for block 128 in AVX2 (#220: the host's "act quantize" was 1.1 ms per decode round
// on Gyro-M against 0.2 on Swift, the scalar transform on the verify window's critical path).  Every output value is
// the scalar code's: the same sign multiply, the same a + c / a - c per butterfly with the operands in the same order
// (strides 1, 2, 4 inside a register by a lane swap and a blend, 8..64 across registers), the same scale multiply.
void hadamard_rotate128_avx2(float* x, int n, const float* signs) {
    const __m256 scale = _mm256_set1_ps(1.0f / std::sqrt(128.0f));
    for (int b0 = 0; b0 < n; b0 += 128) {
        float* v = x + b0;
        __m256 r[16];
        for (int k = 0; k < 16; ++k) {
            r[k] = _mm256_loadu_ps(v + 8 * k);
            if (signs) r[k] = _mm256_mul_ps(r[k], _mm256_loadu_ps(signs + b0 + 8 * k));
        }
        for (int k = 0; k < 16; ++k) {
            __m256 s = _mm256_permute_ps(r[k], 0xB1);                                           // stride 1
            r[k] = _mm256_blend_ps(_mm256_add_ps(r[k], s), _mm256_sub_ps(s, r[k]), 0xAA);
            s = _mm256_permute_ps(r[k], 0x4E);                                                  // stride 2
            r[k] = _mm256_blend_ps(_mm256_add_ps(r[k], s), _mm256_sub_ps(s, r[k]), 0xCC);
            s = _mm256_permute2f128_ps(r[k], r[k], 1);                                          // stride 4
            r[k] = _mm256_blend_ps(_mm256_add_ps(r[k], s), _mm256_sub_ps(s, r[k]), 0xF0);
        }
        for (int len = 1; len < 16; len <<= 1)                                                  // strides 8..64
            for (int i = 0; i < 16; i += len << 1)
                for (int j = i; j < i + len; ++j) {
                    const __m256 a = r[j], c = r[j + len];
                    r[j] = _mm256_add_ps(a, c);
                    r[j + len] = _mm256_sub_ps(a, c);
                }
        for (int k = 0; k < 16; ++k) _mm256_storeu_ps(v + 8 * k, _mm256_mul_ps(r[k], scale));
    }
}

}  // namespace strata::kernels::cpu
