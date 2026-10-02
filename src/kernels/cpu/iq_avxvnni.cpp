// src/kernels/cpu/iq_avxvnni.cpp - xeno #106: the i-quant expert rows on AVX-VNNI (Intel Core 12th-14th gen and Core
// Ultra without AVX-512, AMD Zen 5).  The same source as iq_avx2.cpp, compiled again with STRATA_IQ_AVXVNNI: each
// token's `madd` + `add` becomes one `vpdpwssd`, the same integer sum (iq_avx2.cpp says why nothing saturates).
// Only this file uses AVX-VNNI, and its rows run only where cpu_avxvnni_ok() holds (iq256_gu_rows / iq256_rows).
#if defined(__AVXVNNI__) || (defined(_MSC_VER) && !defined(__clang__))
#define STRATA_IQ_AVXVNNI 1
#include "iq_avx2.cpp"
#else
// a compiler without AVX-VNNI: the names exist (cpu_avxvnni_ok() is false on such a build, so they are not called)
#include "strata/kernels/cpu/iq_avx2.hpp"
namespace strata::kernels::cpu {
void iq256_gu_rows_avxvnni(int type, const uint8_t* blob, size_t gu_row, size_t up_off, int n, const void* const* act,
                           int nt, float* const* ff, int r0, int r1) {
    iq256_gu_rows_avx2(type, blob, gu_row, up_off, n, act, nt, ff, r0, r1);
}
void iq256_rows_avxvnni(int type, const uint8_t* w, size_t row_bytes, int n, const void* const* act, int nt,
                        float* const* out, int r0, int r1) {
    iq256_rows_avx2(type, w, row_bytes, n, act, nt, out, r0, r1);
}
}  // namespace strata::kernels::cpu
#endif
