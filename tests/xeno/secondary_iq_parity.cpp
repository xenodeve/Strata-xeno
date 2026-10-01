// #11: the 4070 tier on an i-quant pack.  SecondaryRunner refused every layout that is not Q2_0 / Q2_0
// ("invalid launch geometry"), so Swift 1.5 IQ3_XXS ran on the 5060 alone.  The 5060's verify path computes such a
// layer with native_expert_grouped on q8_1 activations from quantize_q8_1_rows, with no activation scales unless
// both formats are Q2_0 (verify.cpp).  The runner must give the same bytes for the same experts and activations.
//
// With a second layer (of other formats), one runner with captured graphs serves L, L2, L in turn: a graph bakes
// in its layer's formats, so a cache keyed by token and row counts alone would replay L's graph for L2.
//
//     xeno_secondary_iq_parity <shard1.gguf> <layer> [layer2]
#include "strata/artifact/gguf_reader.hpp"
#include "strata/core/secondary_arena.hpp"
#include "strata/core/secondary_budget.hpp"
#include "strata/core/secondary_runner.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/kernels/iq_kernels.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace cpu = strata::kernels::cpu;
constexpr int H = cpu::H, FF = cpu::FF, ENTRIES = 3;

static void ck(cudaError_t e, const char* at) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", at, cudaGetErrorString(e));
        std::exit(2);
    }
}

// One layer's two experts (E, E+1), its activations, and the 5060 verify path's output for them.
struct Layer {
    int index = 0, e = 0;
    cpu::NativeFmt f;
    strata::kernels::NativeExpertLayout layout;
    std::vector<uint8_t> blob, blob2;
    std::vector<float> x, reference;
};

static bool load(const char* shard1, int L, int E, int seed, Layer& out) {
    // a layer's experts live in one shard (Swift 1.5: layers 12-47 in shard 2): shard 1, else its sibling shard 2
    std::string path = shard1;
    auto gguf = std::make_unique<strata::GgufFile>(path);
    auto find = [&](const char* role) -> const strata::TensorInfo* {
        for (const auto& ti : gguf->tensors())
            if (ti.name == "blk." + std::to_string(L) + "." + role + ".weight") return &ti;
        return nullptr;
    };
    if (find("ffn_gate_exps") == nullptr) {
        const size_t at = path.rfind("-00001-of-");
        if (at != std::string::npos) path.replace(at + 1, 5, "00002");
        gguf = std::make_unique<strata::GgufFile>(path);
    }
    const strata::TensorInfo *gate = find("ffn_gate_exps"), *up = find("ffn_up_exps"), *down = find("ffn_down_exps");
    if (!gate || !up || !down) { std::fprintf(stderr, "layer %d has no experts in %s\n", L, path.c_str()); return false; }
    std::string err;
    if (!cpu::native_fmt((int) gate->type, (int) down->type, H, FF, out.f, err)) {
        std::fprintf(stderr, "native format: %s\n", err.c_str());
        return false;
    }
    const cpu::NativeFmt& f = out.f;
    if (f.gu_type == 42 && f.d_type == 42) { std::fprintf(stderr, "layer %d is Q2_0: not an i-quant layer\n", L); return false; }
    auto blob_of = [&](int e) {
        std::vector<uint8_t> b(f.bytes);
        std::memcpy(b.data(), gguf->tensor_data(*gate) + (size_t) e * f.up_off, f.up_off);
        std::memcpy(b.data() + f.up_off, gguf->tensor_data(*up) + (size_t) e * f.up_off, f.up_off);
        std::memcpy(b.data() + f.down_off, gguf->tensor_data(*down) + (size_t) e * (f.bytes - f.down_off),
                    f.bytes - f.down_off);
        return b;
    };
    out.index = L;
    out.e = E;
    out.blob = blob_of(E);
    out.blob2 = blob_of((E + 1) % cpu::NE);
    out.layout = strata::kernels::native_expert_layout(f.gu_type, f.d_type, H, FF);
    std::mt19937 rng(seed);
    std::normal_distribution<float> normal(0.f, 1.f);
    out.x.resize((size_t) ENTRIES * H);
    for (auto& v : out.x) v = normal(rng);

    // the 5060's verify path: plain q8_1 activations, no scales (i-quant); entries 0, 1 use E, entry 2 uses E+1
    ck(cudaSetDevice(0), "primary device");
    void *dblob = nullptr, *dx = nullptr, *dxq = nullptr, *scratch = nullptr;
    float* dout = nullptr;
    unsigned long long* dptr = nullptr;
    int32_t *dstart = nullptr, *dn = nullptr, *ddst = nullptr, *dtok = nullptr;
    const size_t xb = out.x.size() * sizeof(float);
    ck(cudaMalloc(&dblob, 2 * out.blob.size()), "blob alloc");
    ck(cudaMalloc(&dx, xb), "x alloc");
    ck(cudaMalloc(&dxq, (size_t) ENTRIES * H / 32 * 36), "q8 alloc");
    ck(cudaMalloc(&scratch, strata::kernels::native_expert_scratch_bytes(ENTRIES, FF, H)), "scratch alloc");
    ck(cudaMalloc((void**) &dout, xb), "out alloc");
    ck(cudaMalloc((void**) &dptr, 2 * sizeof(unsigned long long)), "pointer alloc");
    ck(cudaMalloc((void**) &dstart, 3 * sizeof(int32_t)), "start alloc");
    ck(cudaMalloc((void**) &dn, sizeof(int32_t)), "count alloc");
    ck(cudaMalloc((void**) &ddst, ENTRIES * sizeof(int32_t)), "dst alloc");
    ck(cudaMalloc((void**) &dtok, ENTRIES * sizeof(int32_t)), "tok alloc");
    ck(cudaMemcpy(dblob, out.blob.data(), out.blob.size(), cudaMemcpyHostToDevice), "blob copy");
    ck(cudaMemcpy((uint8_t*) dblob + out.blob.size(), out.blob2.data(), out.blob2.size(), cudaMemcpyHostToDevice),
       "blob2 copy");
    ck(cudaMemcpy(dx, out.x.data(), xb, cudaMemcpyHostToDevice), "x copy");
    const unsigned long long ptr[2] = {(unsigned long long) dblob,
                                       (unsigned long long) ((uint8_t*) dblob + out.blob.size())};
    const int32_t start[3] = {0, 2, ENTRIES}, groups = 2, dst[ENTRIES] = {0, 1, 2}, tok[ENTRIES] = {0, 1, 2};
    ck(cudaMemcpy(dptr, ptr, sizeof ptr, cudaMemcpyHostToDevice), "pointer copy");
    ck(cudaMemcpy(dstart, start, sizeof start, cudaMemcpyHostToDevice), "start copy");
    ck(cudaMemcpy(dn, &groups, sizeof groups, cudaMemcpyHostToDevice), "count copy");
    ck(cudaMemcpy(ddst, dst, sizeof dst, cudaMemcpyHostToDevice), "dst copy");
    ck(cudaMemcpy(dtok, tok, sizeof tok, cudaMemcpyHostToDevice), "tok copy");
    cudaStream_t stream = nullptr;
    ck(cudaStreamCreate(&stream), "stream");
    strata::kernels::quantize_q8_1_rows((const float*) dx, ENTRIES, H, dxq, stream);
    strata::kernels::native_expert_grouped(out.layout, dptr, dstart, dn, ddst, dtok, 2, ENTRIES, dxq, scratch, dout,
                                           stream, nullptr);
    ck(cudaStreamSynchronize(stream), "primary expert kernel");
    out.reference.resize(out.x.size());
    ck(cudaMemcpy(out.reference.data(), dout, xb, cudaMemcpyDeviceToHost), "primary out");
    cudaFree(dblob); cudaFree(dx); cudaFree(dxq); cudaFree(scratch); cudaFree(dout);
    cudaFree(dptr); cudaFree(dstart); cudaFree(dn); cudaFree(ddst); cudaFree(dtok);
    cudaStreamDestroy(stream);
    return true;
}

// The runner on `slots` (the layer's two experts) against the layer's 5060 reference: differing floats.
static int compare(strata::core::SecondaryRunner& runner, const strata::core::SecondaryArena& arena, const Layer& l,
                   int32_t s0, int32_t s1, const char* what) {
    const int32_t selected[ENTRIES] = {s0, s0, s1};
    std::vector<float> got(l.x.size(), 0.f);
    std::string err;
    if (!runner.launch(l.layout, arena, l.x.data(), selected, ENTRIES, 1, err) || !runner.finish(got.data(), err)) {
        std::fprintf(stderr, "secondary runner (%s): %s\n", what, err.c_str());
        return (int) got.size();
    }
    int different = 0;
    double max_abs = 0.;
    for (size_t i = 0; i < got.size(); ++i) {
        different += std::memcmp(&l.reference[i], &got[i], sizeof(float)) != 0;
        max_abs = std::max(max_abs, (double) std::fabs(l.reference[i] - got[i]));
    }
    std::printf("%s: layer %d (gate/up %d, down %d), experts %d/%d: 4070 runner vs 5060 verify path, differing "
                "%d/%zu floats, max |diff| %.3g\n", what, l.index, l.f.gu_type, l.f.d_type, l.e, (l.e + 1) % cpu::NE,
                different, got.size(), max_abs);
    return different;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: xeno_secondary_iq_parity <shard1.gguf> <layer> [layer2]\n");
        return 2;
    }
    const int L = std::atoi(argv[2]), L2 = argc > 3 ? std::atoi(argv[3]) : -1;
    Layer a, b;
    if (!load(argv[1], L, 7, 1107, a)) return 2;
    if (L2 >= 0 && !load(argv[1], L2, 100, 31, b)) return 2;
    if (L2 >= 0 && a.f.gu_type == b.f.gu_type && a.f.d_type == b.f.d_type) {
        std::fprintf(stderr, "layers %d and %d share their formats: the graph check needs two\n", L, L2);
        return 2;
    }

    ck(cudaSetDevice(1), "secondary device");
    std::vector<uint64_t> sizes = {a.blob.size(), a.blob2.size()};
    if (L2 >= 0) { sizes.push_back(b.blob.size()); sizes.push_back(b.blob2.size()); }
    strata::core::SecondaryArena arena;
    std::string err;
    bool ok = arena.open(1, sizes, 16ull << 20, err);
    const std::vector<const std::vector<uint8_t>*> blobs = {&a.blob, &a.blob2, &b.blob, &b.blob2};
    for (uint64_t s = 0; ok && s < sizes.size(); ++s)
        ok = arena.fill_slot(s, blobs[s]->data(), blobs[s]->size(), err) &&
             arena.verify_slot(s, blobs[s]->data(), blobs[s]->size(), err);
    if (!ok) { std::fprintf(stderr, "secondary arena: %s\n", err.c_str()); return 2; }
    // a 4070 swap copies a newcomer into its victim's slot, possibly from another layer: fits() is the check
    if (arena.slot_bytes(0) != (a.blob.size() + 255) / 256 * 256 || arena.slot_bytes(sizes.size()) != 0 ||
        !arena.fits(0, a.blob.size()) || arena.fits(0, arena.slot_bytes(0) + 1) || arena.fits(0, 0)) {
        std::fprintf(stderr, "secondary arena: slot_bytes %llu / fits for a %zu-byte blob\n",
                     (unsigned long long) arena.slot_bytes(0), a.blob.size());
        return 1;
    }

    int different = 0;
    {
        strata::core::SecondaryRunner runner;   // eager launches
        if (!runner.init(ENTRIES, ENTRIES, H, FF, err, strata::core::kSecondaryReserveBytes)) {
            std::fprintf(stderr, "secondary runner: %s\n", err.c_str());
            return 1;
        }
        different += compare(runner, arena, a, 0, 1, "eager");
    }
    if (L2 >= 0) {
        strata::core::SecondaryRunner runner;   // captured graphs, two layer formats in turn
        if (!runner.init(ENTRIES, ENTRIES, H, FF, err, strata::core::kSecondaryReserveBytes) ||
            !runner.set_graph(true, err)) {
            std::fprintf(stderr, "secondary runner: %s\n", err.c_str());
            return 1;
        }
        different += compare(runner, arena, a, 0, 1, "graph L");
        different += compare(runner, arena, b, 2, 3, "graph L2");
        different += compare(runner, arena, a, 0, 1, "graph L again");
    }
    return different ? 1 : 0;
}
