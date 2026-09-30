// #32 S3: expert_split runs some of a layer's routed experts on the 4070 (sm_89) instead of the 5060 (sm_120).  The
// prompt path's MoE layer - per-token q8 activations, each group's sub-products through mmq::expert_rows (row gather,
// gate/up MMQ, swiglu, q8, down MMQ), then moe_combine - must write the same bytes on both cards.  A synthetic layer:
// 512 tokens, top-10 of 24 Q2_0 experts, two MMQ groups (16 + 8), the first split into two sub-products as the
// prompt path splits a group whose rows exceed its scratch.  CUDA_VISIBLE_DEVICES=1,0: device 0 is the 5060 Ti.
#include "strata/prefill/kernels.hpp"
#include "strata/prefill/moe_mmq.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <random>
#include <vector>

namespace mmq = strata::prefill::mmq;

namespace {
constexpr int64_t T = 512, K = 10, N = 2560, NFF = 640, NEX = 24, GROUP = 16;
constexpr int QT = 42;   // GGML_TYPE_Q2_0
constexpr size_t TAIL = 4096;

struct Layer {
    std::vector<float> x, w, shared, sg;
    std::vector<uint8_t> gu, down;          // NEX experts, matrix_bytes apart
    std::vector<int32_t> src, slot, cnt;    // src: token of each row (rows sorted by expert); slot: (t, k) -> row
};

void q2_blocks(std::vector<uint8_t>& W, std::mt19937& rng) {
    std::uniform_real_distribution<float> ud(0.01f, 0.03f);
    for (size_t b = 0; b + 18 <= W.size(); b += 18) {   // block_q2_0: fp16 d, then 16 bytes of 2-bit codes
        const __half d = __float2half(ud(rng));
        std::memcpy(&W[b], &d, 2);
        for (int i = 2; i < 18; ++i) W[b + i] = (uint8_t) rng();
    }
}

Layer make_layer() {
    Layer L;
    std::mt19937 rng(32);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    L.x.resize((size_t) (T * N));
    for (auto& v : L.x) v = nd(rng);
    L.w.resize((size_t) (T * K));
    for (auto& v : L.w) v = std::abs(nd(rng)) * 0.2f;
    L.shared.resize((size_t) (T * N));
    for (auto& v : L.shared) v = nd(rng);
    L.sg.resize((size_t) T);
    for (auto& v : L.sg) v = nd(rng);
    L.gu.resize((size_t) NEX * mmq::matrix_bytes(QT, 2 * NFF, N));
    L.down.resize((size_t) NEX * mmq::matrix_bytes(QT, N, NFF));
    q2_blocks(L.gu, rng);
    q2_blocks(L.down, rng);
    // top-10 of 24 per token, skewed so experts differ in row counts
    std::vector<int32_t> ex((size_t) (T * K));
    for (int64_t t = 0; t < T; ++t) {
        std::vector<int32_t> p((size_t) NEX);
        std::iota(p.begin(), p.end(), 0);
        std::shuffle(p.begin(), p.end(), rng);
        std::stable_sort(p.begin(), p.begin() + NEX, [&](int32_t a, int32_t b) { return (a % 5) < (b % 5); });
        for (int64_t k = 0; k < K; ++k) ex[(size_t) (t * K + k)] = p[(size_t) ((k + t) % NEX)];
    }
    L.cnt.assign((size_t) NEX, 0);
    for (int32_t e : ex) ++L.cnt[(size_t) e];
    std::vector<int32_t> off((size_t) NEX + 1, 0);
    for (int64_t e = 0; e < NEX; ++e) off[(size_t) e + 1] = off[(size_t) e] + L.cnt[(size_t) e];
    L.src.resize((size_t) (T * K));
    L.slot.resize((size_t) (T * K));
    std::vector<int32_t> at(off.begin(), off.end() - 1);
    for (int64_t i = 0; i < T * K; ++i) {
        const int32_t r = at[(size_t) ex[(size_t) i]]++;
        L.src[(size_t) r] = (int32_t) (i / K);
        L.slot[(size_t) i] = r;
    }
    return L;
}

template <typename V>
void* up(const V& v) {
    void* d = nullptr;
    cudaMalloc(&d, v.size() * sizeof(v[0]) + TAIL);
    cudaMemset(d, 0, v.size() * sizeof(v[0]) + TAIL);
    cudaMemcpy(d, v.data(), v.size() * sizeof(v[0]), cudaMemcpyHostToDevice);
    return d;
}

// the layer on the current device: expert rows (Dm) and the combined output (bo)
bool run(const Layer& L, std::vector<float>& Dm, std::vector<float>& bo) {
    cudaStream_t s;
    cudaStreamCreate(&s);
    mmq::Context ctx;
    const size_t gub = mmq::matrix_bytes(QT, 2 * NFF, N), db = mmq::matrix_bytes(QT, N, NFF);
    auto* dx = (float*) up(L.x);
    auto* dw = (float*) up(L.w);
    auto* dsh = (float*) up(L.shared);
    auto* dsg = (float*) up(L.sg);
    auto* dgu = (uint8_t*) up(L.gu);
    auto* ddn = (uint8_t*) up(L.down);
    auto* dsrc = (int32_t*) up(L.src);
    auto* dslot = (int32_t*) up(L.slot);
    const int64_t R = T * K;
    void *xtok = nullptr, *xq = nullptr, *hq = nullptr;
    float *gu = nullptr, *h = nullptr, *dm = nullptr, *dbo = nullptr;
    int32_t *ids = nullptr, *bounds = nullptr;
    cudaMalloc(&xtok, mmq::q8_bytes(T, N));
    cudaMalloc(&xq, mmq::q8_bytes(R, N));
    cudaMalloc(&hq, mmq::q8_bytes(R, NFF));
    cudaMalloc((void**) &gu, (size_t) (R * 2 * NFF) * 4);
    cudaMalloc((void**) &h, (size_t) (R * NFF) * 4);
    cudaMalloc((void**) &dm, (size_t) (R * N) * 4);
    cudaMalloc((void**) &dbo, (size_t) (T * N) * 4);
    cudaMalloc((void**) &ids, (size_t) R * 4);
    cudaMalloc((void**) &bounds, 64 * 4);
    mmq::iota(ids, R, s);
    mmq::quantize(dx, nullptr, xtok, QT, N, N, T, s);
    // sub-products: experts [q0, q1) of the layer (group 0 split at 8, then group 1 whole)
    const int64_t subs[3][2] = {{0, 8}, {8, 16}, {16, NEX}};
    std::vector<int32_t> abs_off((size_t) NEX + 1, 0);
    for (int64_t e = 0; e < NEX; ++e) abs_off[(size_t) e + 1] = abs_off[(size_t) e] + L.cnt[(size_t) e];
    for (const auto& sb : subs) {
        std::vector<int32_t> b;
        int64_t maxr = 0;
        for (int64_t e = sb[0]; e <= sb[1]; ++e) b.push_back(abs_off[(size_t) e] - abs_off[(size_t) sb[0]]);
        for (int64_t e = sb[0]; e < sb[1]; ++e) maxr = std::max<int64_t>(maxr, L.cnt[(size_t) e]);
        cudaMemcpyAsync(bounds, b.data(), b.size() * 4, cudaMemcpyHostToDevice, s);
        const int64_t r0 = abs_off[(size_t) sb[0]];
        mmq::ExpertRows a;
        a.xtok = xtok; a.xtok_rows = T; a.rows = dsrc + r0; a.nr = abs_off[(size_t) sb[1]] - r0; a.max_rows = maxr;
        a.n = (int) (sb[1] - sb[0]);
        a.gu = dgu + (size_t) sb[0] * gub; a.gu_type = QT; a.gu_bytes = gub;
        a.down = ddn + (size_t) sb[0] * db; a.down_type = QT; a.down_bytes = db;
        a.bounds = bounds; a.ids = ids; a.n_embd = N; a.n_ff = NFF; a.interleaved = false;
        a.xq = xq; a.gu_out = gu; a.h = h; a.hq = hq; a.dst = dm + r0 * N;
        mmq::expert_rows(ctx, a, s);
        cudaStreamSynchronize(s);   // the bounds block is reused by the next sub-product
    }
    strata::prefill::moe_combine(dm, dslot, dw, dsh, dsg, dbo, T, s);
    cudaStreamSynchronize(s);
    Dm.resize((size_t) (R * N));
    bo.resize((size_t) (T * N));
    cudaMemcpy(Dm.data(), dm, Dm.size() * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(bo.data(), dbo, bo.size() * 4, cudaMemcpyDeviceToHost);
    const cudaError_t e = cudaGetLastError();
    for (void* p : {(void*) dx, (void*) dw, (void*) dsh, (void*) dsg, (void*) dgu, (void*) ddn, (void*) dsrc,
                    (void*) dslot, xtok, xq, hq, (void*) gu, (void*) h, (void*) dm, (void*) dbo, (void*) ids,
                    (void*) bounds})
        cudaFree(p);
    cudaStreamDestroy(s);
    if (e != cudaSuccess) { std::printf("%s\n", cudaGetErrorString(e)); return false; }
    return true;
}
}  // namespace

// #41: the same layer with its experts relabelled (new id = order[old]), so they fall in different MMQ groups and
// sub-products and run in a different order.  Weights move with their expert; every (t, k) row must keep its bytes.
Layer relabel(const Layer& L, const std::vector<int32_t>& order) {
    std::vector<int32_t> off((size_t) NEX + 1, 0), ex((size_t) (T * K));
    for (int64_t e = 0; e < NEX; ++e) off[(size_t) e + 1] = off[(size_t) e] + L.cnt[(size_t) e];
    for (int64_t i = 0; i < T * K; ++i) {   // the expert of (t, k): the one whose row range holds its row
        const int32_t r = L.slot[(size_t) i];
        ex[(size_t) i] = (int32_t) (std::upper_bound(off.begin(), off.end(), r) - off.begin() - 1);
    }
    Layer R = L;
    const size_t gub = mmq::matrix_bytes(QT, 2 * NFF, N), db = mmq::matrix_bytes(QT, N, NFF);
    for (int64_t e = 0; e < NEX; ++e) {
        std::memcpy(&R.gu[(size_t) order[(size_t) e] * gub], &L.gu[(size_t) e * gub], gub);
        std::memcpy(&R.down[(size_t) order[(size_t) e] * db], &L.down[(size_t) e * db], db);
    }
    R.cnt.assign((size_t) NEX, 0);
    for (int32_t& e : ex) { e = order[(size_t) e]; ++R.cnt[(size_t) e]; }
    std::vector<int32_t> off2((size_t) NEX + 1, 0);
    for (int64_t e = 0; e < NEX; ++e) off2[(size_t) e + 1] = off2[(size_t) e] + R.cnt[(size_t) e];
    std::vector<int32_t> at(off2.begin(), off2.end() - 1);
    for (int64_t i = 0; i < T * K; ++i) {
        const int32_t r = at[(size_t) ex[(size_t) i]]++;
        R.src[(size_t) r] = (int32_t) (i / K);
        R.slot[(size_t) i] = r;
    }
    return R;
}

int main() {
    int n_dev = 0;
    cudaGetDeviceCount(&n_dev);
    if (n_dev < 2 || !mmq::built()) { std::printf("needs two devices (CUDA_VISIBLE_DEVICES=1,0) and MMQ\n"); return 1; }
    const Layer L = make_layer();
    std::vector<float> Dm[2], bo[2];
    for (int dev = 0; dev < 2; ++dev) {
        cudaSetDevice(dev);
        if (!run(L, Dm[dev], bo[dev])) return 1;
    }
    // not vacuous: every expert row was written, with finite non-zero values
    size_t bad = 0, zero_rows = 0;
    for (float v : bo[0]) bad += !std::isfinite(v);
    for (int64_t r = 0; r < T * K; ++r) {
        bool any = false;
        for (int64_t d = 0; d < N && !any; ++d) any = Dm[0][(size_t) (r * N + d)] != 0.0f;
        zero_rows += !any;
    }
    size_t dd = 0, db = 0;
    for (size_t i = 0; i < Dm[0].size(); ++i) dd += std::memcmp(&Dm[0][i], &Dm[1][i], 4) != 0;
    for (size_t i = 0; i < bo[0].size(); ++i) db += std::memcmp(&bo[0][i], &bo[1][i], 4) != 0;
    std::printf("MoE layer sm_120 vs sm_89: expert rows %zu of %zu differ, output %zu of %zu differ "
                "(%zu non-finite outputs, %zu all-zero rows)\n", dd, Dm[0].size(), db, bo[0].size(), bad, zero_rows);
    // #41: relabelled experts (reversed ids: other groups, other sub-products, other order), on the 5060 Ti
    std::vector<int32_t> order((size_t) NEX);
    for (int64_t e = 0; e < NEX; ++e) order[(size_t) e] = (int32_t) (NEX - 1 - e);
    const Layer L2 = relabel(L, order);
    std::vector<float> Dm2, bo2;
    cudaSetDevice(0);
    if (!run(L2, Dm2, bo2)) return 1;
    size_t rd = 0, rb = 0;
    for (int64_t i = 0; i < T * K; ++i)
        rd += std::memcmp(&Dm[0][(size_t) L.slot[(size_t) i] * N], &Dm2[(size_t) L2.slot[(size_t) i] * N],
                          (size_t) N * 4) != 0;
    for (size_t i = 0; i < bo[0].size(); ++i) rb += std::memcmp(&bo[0][i], &bo2[i], 4) != 0;
    std::printf("relabelled experts (#41): %zu of %lld (t, k) rows differ, output %zu of %zu differ\n", rd,
                (long long) (T * K), rb, bo[0].size());
    const bool pass = dd == 0 && db == 0 && bad == 0 && zero_rows == 0 && rd == 0 && rb == 0;
    std::printf("%s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
