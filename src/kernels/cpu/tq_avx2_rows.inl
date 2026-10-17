// src/kernels/cpu/tq_avx2_rows.inl - the AVX2 trellis row kernel's body, included twice by tq_avx2.cpp: once plain,
// once as the AVX-VNNI copy (STRATA_TQ_ROWS_VNNI 1, compiled under STRATA_TQ_ROWS_FN = target("avxvnni") on GCC /
// Clang).  The integer sums are the same in both, so every variant gives the scalar kernel's bits (tq_cpu_parity).
//
//   G (gather): the four state-table rows of a YMM come from one VPGATHERQQ instead of a load and three broadcast
//   loads blended in - the fork's IQ rule (cpu_gather_fast_here: a P-core of an AVX-VNNI Intel CPU gathers faster;
//   its E-cores assemble faster).
//   VNNI: each pair of VPMADDWD + VPADDD becomes VPMADDWD + VPDPWSSD (src + the same pairwise products).

template <bool G>
STRATA_TQ_ROWS_FN STRATA_TQ_INLINE __m256i rows4(const long long* tab, uint32_t s0, uint32_t s1, uint32_t s2, uint32_t s3) {
    if constexpr (G) {
        return _mm256_i32gather_epi64(tab, _mm_setr_epi32((int) s0, (int) s1, (int) s2, (int) s3), 8);
    } else {
        // a load and three broadcast loads blended in.  Broadcast loads and VPBLENDD avoid the shuffle port that
        // VPINSRQ / VINSERTI128 need (the two measure the same on Zen 5).
        __m256i v = _mm256_castsi128_si256(_mm_loadl_epi64((const __m128i*) (tab + s0)));
        v = _mm256_blend_epi32(v, _mm256_set1_epi64x(tab[s1]), 0x0C);
        v = _mm256_blend_epi32(v, _mm256_set1_epi64x(tab[s2]), 0x30);
        return _mm256_blend_epi32(v, _mm256_set1_epi64x(tab[s3]), 0xC0);
    }
}

// w0 . q0 + w1 . q1 per int32 lane (pairwise int16 products), exact
STRATA_TQ_ROWS_FN STRATA_TQ_INLINE __m256i pair_dot(__m256i w0, const __m256i* q0, __m256i w1, const __m256i* q1) {
#if STRATA_TQ_ROWS_VNNI
    return _mm256_dpwssd_avx_epi32(_mm256_madd_epi16(w0, _mm256_load_si256(q0)), w1, _mm256_load_si256(q1));
#else
    return _mm256_add_epi32(_mm256_madd_epi16(w0, _mm256_load_si256(q0)), _mm256_madd_epi16(w1, _mm256_load_si256(q1)));
#endif
}

// CB (codebook): a state's four weights straight from the 2048-point int16 codebook (tq_codebook_i16: 8 KiB, held in
// L1) at the hash's two points, x = s * 0x9e3779b1: x >> 21 and (x >> 10) & 2047 - the same entries the 256 KiB
// state table holds, which L1 cannot (xeno #220).  With G the 8 dword indices of four states are computed in SIMD
// and gathered (a P-core's fast gather on an L1-resident table); without, scalar multiplies and loads.
template <bool G>
STRATA_TQ_ROWS_FN STRATA_TQ_INLINE __m256i rows4_cb(const uint32_t* cb, const uint32_t* st) {
    if constexpr (G) {
        const __m128i x = _mm_mullo_epi32(_mm_loadu_si128((const __m128i*) st), _mm_set1_epi32((int) 0x9e3779b1u));
        const __m128i p0 = _mm_srli_epi32(x, 21);
        const __m128i p1 = _mm_and_si128(_mm_srli_epi32(x, 10), _mm_set1_epi32(2047));
        const __m256i idx = _mm256_inserti128_si256(_mm256_castsi128_si256(_mm_unpacklo_epi32(p0, p1)),
                                                    _mm_unpackhi_epi32(p0, p1), 1);
        return _mm256_i32gather_epi32((const int*) cb, idx, 4);
    } else {
        long long r[4];
        for (int k = 0; k < 4; ++k) {
            const uint32_t x = st[k] * 0x9e3779b1u;
            r[k] = (long long) ((uint64_t) cb[x >> 21] | ((uint64_t) cb[(x >> 10) & 2047u] << 32));
        }
        return _mm256_set_epi64x(r[3], r[2], r[1], r[0]);
    }
}

// The eight YMM of a block unrolled at compile time: MSVC does not unroll the m loop (the `#pragma GCC unroll` is
// GCC's), which left the state offsets computed at run time and wv[] spilled to the stack between the decode and the
// token loop (xeno #220: dumpbin of the loop, 34 stack references).  A fold over M gives every state a constant offset
// and keeps wv in registers on every compiler.
template <int KB, bool G, int... M>
STRATA_TQ_ROWS_FN STRATA_TQ_INLINE void decode8(const long long* tab, const uint8_t* qs, uint64_t wr, __m256i* wv,
                                      std::integer_sequence<int, M...>) {
    ((wv[M] = rows4<G>(tab, tq_state<KB>(qs, wr, 4 * M), tq_state<KB>(qs, wr, 4 * M + 1),
                       tq_state<KB>(qs, wr, 4 * M + 2), tq_state<KB>(qs, wr, 4 * M + 3))), ...);
}

// Q (q64, TQ2_T only): one 8-byte window per four states, read big-endian, so state qs[t-1] << 8 | qs[t] (15 bits)
// is a shift and a mask of it - (be >> (24 - 8k)) & 0x7fff - straight into the 256 KiB state table.  Against the
// byte-wise decode it saves a load, a shift and an OR per state (xeno #220, tqab: P-core -6 %, E-core -11 %); the
// big-endian read (one byte swap per window, not per state) replaced a 512 KiB re-indexed table with the base one:
// half the pages, P-core -9 %, E-core -3 % more (Gemini's MOVBE suggestion, measured; a per-state swap had lost).
// Windows: bytes 4M-4 .. 4M+3 for M >= 1, all inside the block; M = 0 reads bytes 0..7 and qs[31] for t = 0.
// The four rows go into the ymm as two xmm pairs and one insert (MOVQ, MOVQ, PUNPCKLQDQ per pair, VINSERTI128)
// rather than the broadcast/blend of rows4: a Gracemont E-core splits a ymm blend into two uops and runs the pairs
// faster (tqab: E-core -4.5 % against q64's blend, P-core the same; xeno #220).
STRATA_TQ_ROWS_FN STRATA_TQ_INLINE __m256i rows4x(const long long* tab, uint32_t s0, uint32_t s1, uint32_t s2, uint32_t s3) {
    const __m128i a = _mm_unpacklo_epi64(_mm_loadl_epi64((const __m128i*) (tab + s0)),
                                         _mm_loadl_epi64((const __m128i*) (tab + s1)));
    const __m128i b = _mm_unpacklo_epi64(_mm_loadl_epi64((const __m128i*) (tab + s2)),
                                         _mm_loadl_epi64((const __m128i*) (tab + s3)));
    return _mm256_inserti128_si256(_mm256_castsi128_si256(a), b, 1);
}

STRATA_TQ_ROWS_FN STRATA_TQ_INLINE uint64_t tq_load_be64(const uint8_t* p) {
    uint64_t x;
    std::memcpy(&x, p, 8);
#if defined(_MSC_VER) && !defined(__clang__)
    return _byteswap_uint64(x);
#else
    return __builtin_bswap64(x);
#endif
}

template <int M>
STRATA_TQ_ROWS_FN STRATA_TQ_INLINE __m256i rows4q(const long long* tab, const uint8_t* qs) {
    if constexpr (M == 0) {
        const uint64_t be = tq_load_be64(qs);   // bytes 0..7, byte 0 on top
        const uint32_t i0 = (((uint32_t) qs[31] & 0x7fu) << 8) | qs[0];
        return rows4x(tab, i0, (uint32_t) (be >> 48) & 0x7fffu, (uint32_t) (be >> 40) & 0x7fffu,
                      (uint32_t) (be >> 32) & 0x7fffu);
    } else {
        const uint64_t be = tq_load_be64(qs + 4 * M - 4);   // bytes 4M-4 .. 4M+3
        return rows4x(tab, (uint32_t) (be >> 24) & 0x7fffu, (uint32_t) (be >> 16) & 0x7fffu,
                      (uint32_t) (be >> 8) & 0x7fffu, (uint32_t) be & 0x7fffu);
    }
}

template <int... M>
STRATA_TQ_ROWS_FN STRATA_TQ_INLINE void decode8q(const long long* tab, const uint8_t* qs, __m256i* wv,
                                                std::integer_sequence<int, M...>) {
    ((wv[M] = rows4q<M>(tab, qs)), ...);
}

template <int KB, bool G, bool CB = false, bool Q = false, int NT = 0, bool F16 = false>
STRATA_TQ_ROWS_FN void rows_t(const uint8_t* w, size_t row_bytes, int n, const TqAct* const* a, int nt_,
                              float* const* out, int r0, int r1) {
    const int nt = NT > 0 ? NT : nt_;   // #220: a constant count keeps acc[] and a[] in registers
    const int nb = n / 128;
    constexpr size_t bsz = 2 + 4 * KB;
    constexpr bool q64 = Q && KB == 8 && !G && !CB;
    const long long* tab = (const long long*) tq_state_table();
    const uint32_t* cb = (const uint32_t*) tq_codebook_i16();
    for (int r = r0; r < r1; ++r) {
        const uint8_t* x = w + (size_t) r * row_bytes;
        __m128 acc[kTqMaxTokens];
        for (int t = 0; t < nt; ++t) acc[t] = _mm_setzero_ps();
        for (int i = 0; i < nb; ++i) {
            const uint8_t* blk = x + (size_t) i * bsz;
            const uint8_t* qs = blk + 2;
            const uint64_t wr = tq_wrap<KB>(qs);
            __m256i wv[8];   // wv[m] = weights 16m .. 16m+15 (states 4m .. 4m+3)
            if constexpr (q64) {
                decode8q(tab, qs, wv, std::make_integer_sequence<int, 8>{});
            } else if constexpr (CB) {
                alignas(32) uint32_t st[32];
                tq_states<KB>(qs, st);
                for (int m = 0; m < 8; ++m) wv[m] = rows4_cb<G>(cb, st + 4 * m);
            } else {
                decode8<KB, G>(tab, qs, wr, wv, std::make_integer_sequence<int, 8>{});
            }
            const __m128 dx = F16 ? _mm_broadcastss_ps(_mm_cvtph_ps(_mm_cvtsi32_si128((int) tq_ld16(blk)))) : _mm_set1_ps(tq_half_to_float(tq_ld16(blk)));
            for (int t = 0; t < nt; ++t) {
                const __m256i* q = (const __m256i*) (a[t]->q + 128 * i);
                const __m256i p0 = pair_dot(wv[0], q + 0, wv[1], q + 1);
                const __m256i p1 = pair_dot(wv[2], q + 2, wv[3], q + 3);
                const __m256i p2 = pair_dot(wv[4], q + 4, wv[5], q + 5);
                const __m256i p3 = pair_dot(wv[6], q + 6, wv[7], q + 7);
                // per 128-bit lane [p0, p1, p2, p3] partial sums, then the two lanes added: S[0..3], exact
                const __m256i h = _mm256_hadd_epi32(_mm256_hadd_epi32(p0, p1), _mm256_hadd_epi32(p2, p3));
                const __m128i s = _mm_add_epi32(_mm256_castsi256_si128(h), _mm256_extracti128_si256(h, 1));
                const __m128 sc = _mm_mul_ps(dx, _mm_loadu_ps(a[t]->d + 4 * i));
                acc[t] = _mm_fmadd_ps(_mm_cvtepi32_ps(s), sc, acc[t]);
            }
        }
        for (int t = 0; t < nt; ++t) {
            alignas(16) float v[4];
            _mm_store_ps(v, acc[t]);
            out[t][r] = ((v[0] + v[1]) + (v[2] + v[3])) * (1.0f / 8192.0f);
        }
    }
}

template <bool F16>
STRATA_TQ_ROWS_FN void rows_q64_nt(const uint8_t* w, size_t row_bytes, int n, const TqAct* const* a, int nt,
                                   float* const* out, int r0, int r1) {
    switch (nt) {   // #220: one body per token count
        case 1: rows_t<8, false, false, true, 1, F16>(w, row_bytes, n, a, nt, out, r0, r1); break;
        case 2: rows_t<8, false, false, true, 2, F16>(w, row_bytes, n, a, nt, out, r0, r1); break;
        case 3: rows_t<8, false, false, true, 3, F16>(w, row_bytes, n, a, nt, out, r0, r1); break;
        case 4: rows_t<8, false, false, true, 4, F16>(w, row_bytes, n, a, nt, out, r0, r1); break;
        case 5: rows_t<8, false, false, true, 5, F16>(w, row_bytes, n, a, nt, out, r0, r1); break;
        case 6: rows_t<8, false, false, true, 6, F16>(w, row_bytes, n, a, nt, out, r0, r1); break;
        case 7: rows_t<8, false, false, true, 7, F16>(w, row_bytes, n, a, nt, out, r0, r1); break;
        default: rows_t<8, false, false, true, 8, F16>(w, row_bytes, n, a, nt, out, r0, r1); break;
    }
}

template <bool G, bool CB, bool Q = false>
STRATA_TQ_ROWS_FN void rows_kb(int kb, const uint8_t* w, size_t row_bytes, int n, const TqAct* const* a, int nt,
                               float* const* out, int r0, int r1) {
    if (kb == 8) rows_t<8, G, CB, Q>(w, row_bytes, n, a, nt, out, r0, r1);
    else if (kb == 6) rows_t<6, G, CB>(w, row_bytes, n, a, nt, out, r0, r1);
    else rows_t<7, G, CB>(w, row_bytes, n, a, nt, out, r0, r1);
}
