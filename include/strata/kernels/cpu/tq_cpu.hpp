// include/strata/kernels/cpu/tq_cpu.hpp - the trellis expert formats (TQ2_T / TQK6 / TQK7, ggml types 144-146 of the
// agention llama.cpp fork) on the CPU: decode-and-dot against Q8_0 activations (ggml's block_q8_0), one or several
// tokens per call, on AVX-512, AVX2 or plain C++.
//
// THE FORMAT (third_party/ggml/ggml-common-tq.h): a 128-weight block is {fp16 d; 32 steps of K bits} (K = 8 / 6 / 7).
// Step t has a 15-bit state; the state's hash picks two points (pairs) of a 2048-point fp16 codebook, which are
// weights 4t..4t+3, times d.
//
// THE DECODE: a 32768 x 4 int16 table maps a state straight to its four weights at scale 2^13 (256 KiB, the
// approach of the fork's ggml-cpu vec_dot): one 8-byte lookup per four weights, no hash.  The int16 rounding is
// exact for codebook values >= 2^-3 in magnitude and within 2^-14 below that; it is the same table, rounded the
// same way, as the fork's ggml-cpu kernels.
//
// THE ARITHMETIC, identical on every ISA so every path gives the same bits (and so does any token-group size):
//   S[i][k] = sum_j w16[i][32k + j] * q8[i][k][j]        exact in int32 (|w16| < 2^15, |q8| <= 127)
//   acc[k]  = fma((float) S[i][k], d_w[i] * d_a[i][k], acc[k])      for blocks i in order, k = 0..3
//   dot     = ((acc[0] + acc[1]) + (acc[2] + acc[3])) * 2^-13
// i.e. four float accumulators, one per Q8_0 sub-block position, each a fixed sequential chain.  It differs from
// ggml-cpu's vec_dot (same integers, other float summation order) by float rounding only.
//
// THE KERNELS: `tq_isa()` picks AVX-512 (F/BW/VL/VBMI/VNNI, cpu_avx512_ok) or AVX2 (cpu_avx2_ok) at run time; the
// scalar kernel is the reference (on a CPU without AVX2 the engine keeps ggml-cpu's vec_dot, see tq_kernels_enabled).
// STRATA_TQ_ISA=scalar|avx2|avx512 caps the choice (tests, A/B); STRATA_NO_TQ_KERNELS=1 makes native_expert.cpp use
// ggml-cpu's vec_dot instead (the previous path).
#pragma once

#include <cstddef>
#include <cstdint>

namespace strata::kernels::cpu {

enum class TqIsa : int { scalar = 0, avx2 = 1, avx512 = 2 };

/// TQ2_T (144), TQK6 (145), TQK7 (146).
bool tq_supported(int ggml_type) noexcept;
/// Bits per step: 8, 6, 7 (0 for any other type).
int tq_kbits(int ggml_type) noexcept;
/// The best ISA this CPU runs (capped by STRATA_TQ_ISA).
TqIsa tq_isa();
/// Whether this CPU can run `isa` at all (scalar: always).
bool tq_isa_ok(TqIsa isa);
/// Whether the engine's native expert rows take these kernels for the trellis types: on an AVX2 or AVX-512 CPU
/// unless STRATA_NO_TQ_KERNELS=1; the scalar kernel only when STRATA_TQ_ISA=scalar asks for it (otherwise a CPU
/// without AVX2 keeps ggml-cpu's vec_dot, compiled for the build's floor).
bool tq_kernels_enabled();
const char* tq_isa_name(TqIsa isa);

/// The 32768 x 4 int16 state table (filled on first use; thread-safe).
const int16_t* tq_state_table();
/// The 2048-point codebook as int16 pairs at the table's scale and rounding (8 KiB; filled with the table).
const int16_t* tq_codebook_i16();
/// The state table re-indexed for TQ2_T's AVX2 decode (65536 x 4 int16, filled on first use): row w is state
/// (w & 0x7f) << 8 | w >> 8 for a little-endian 16-bit window w of the block bytes with bit 7 clear (xeno #220).
const int16_t* tq_state_table_w16();

/// Rows [r0, r1) of a trellis matrix (`row_bytes` per row, `n` weights per row) against `nt` (<= 8) Q8_0
/// activations of n values: out[t][r] = w_r . a[t].
void tq_rows(TqIsa isa, int ggml_type, const uint8_t* w, size_t row_bytes, int n, const void* const* act, int nt,
             float* const* out, int r0, int r1);
/// Gate and up rows [r0, r1) (gate at blob, up at blob + up_off): ff[t][r] = silu(g) * u, the SwiGLU expression
/// of native_gu_rows.
void tq_gu_rows(TqIsa isa, int ggml_type, const uint8_t* blob, size_t gu_row, size_t up_off, int n,
                const void* const* act, int nt, float* const* ff, int r0, int r1);

/// Largest token group per call and widest row the kernels take.
inline constexpr int kTqMaxTokens = 8;
inline constexpr int kTqMaxN = 4096;

namespace tq_detail {
/// One token's Q8_0 activation unpacked for the kernels: the int8 values widened to int16 and the scales as floats.
struct TqAct {
    alignas(64) int16_t q[kTqMaxN];
    alignas(64) float d[kTqMaxN / 32];
};
void unpack_act(const void* q8_0, int n, TqAct& a);
// the per-ISA row kernels (dot products only; `out[t][r]` for rows [r0, r1))
void tq_rows_scalar(int kb, const uint8_t* w, size_t row_bytes, int n, const TqAct* const* a, int nt,
                    float* const* out, int r0, int r1);
void tq_rows_avx2(int kb, const uint8_t* w, size_t row_bytes, int n, const TqAct* const* a, int nt,
                  float* const* out, int r0, int r1);
/// The AVX2 kernel's variants (xeno #220), the same bits in all: kTqVnni - VPDPWSSD, on by default where
/// cpu_avxvnni_ok() (STRATA_NO_AVXVNNI=1 turns it off); kTqGather - the state-table rows by VPGATHERQQ, off by default
/// (STRATA_TQ_GATHER=1: slower on an i5-13500's P- and E-cores).
inline constexpr int kTqGather = 1, kTqVnni = 2;
/// kTqCodebook: the weights from tq_codebook_i16 at the hash's two points (computed per state) instead of the 256 KiB
/// state table; with kTqGather the indices are computed in SIMD and gathered.  Off by default (STRATA_TQ_CODEBOOK=1:
/// slower on an i5-13500, the table rows are not the bound).
inline constexpr int kTqCodebook = 4;
/// kTqByteStates: TQ2_T's states built byte by byte (two byte loads, shift, or, mask: the decode before xeno #220's
/// q64).  Off by default: the default reads one 8-byte window per four states and indexes tq_state_table_w16 with a
/// shift and a mask, one load per state fewer - an i5-13500, one token, us per expert: P-core 464 -> 437, E-core
/// 1,162 -> 1,030, the same bits.  STRATA_TQ_BYTE_STATES=1 turns it back on for A/B.
inline constexpr int kTqByteStates = 8;
int tq_avx2_variant() noexcept;
/// tq_rows_avx2 with a given variant (tests and benches; kTqVnni is honoured only where cpu_avxvnni_ok()).
void tq_rows_avx2_v(int variant, int kb, const uint8_t* w, size_t row_bytes, int n, const TqAct* const* a, int nt,
                    float* const* out, int r0, int r1);
void tq_rows_avx512(int kb, const uint8_t* w, size_t row_bytes, int n, const TqAct* const* a, int nt,
                    float* const* out, int r0, int r1);
}  // namespace tq_detail

}  // namespace strata::kernels::cpu
