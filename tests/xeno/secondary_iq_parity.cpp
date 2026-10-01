// #11: the 4070 tier on an i-quant pack.  SecondaryRunner refused every layout that is not Q2_0 / Q2_0
// ("invalid launch geometry"), so Swift 1.5 IQ3_XXS ran on the 5060 alone.  The 5060's verify path computes such a
// layer with native_expert_grouped on q8_1 activations from quantize_q8_1_rows, with no activation scales unless
// both formats are Q2_0 (verify.cpp).  The runner must give the same bytes for the same experts and activations.
//
//     xeno_secondary_iq_parity <shard1.gguf> <layer> [expert] [seed]
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
#include <random>
#include <string>
#include <vector>

namespace cpu = strata::kernels::cpu;

static void ck(cudaError_t e, const char* at) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", at, cudaGetErrorString(e));
        std::exit(2);
    }
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: xeno_secondary_iq_parity <shard1.gguf> <layer> [expert] [seed]\n");
        return 2;
    }
    constexpr int H = cpu::H, FF = cpu::FF, ENTRIES = 3;
    const int L = std::atoi(argv[2]);
    const int E = argc > 3 ? std::atoi(argv[3]) : 7;
    const int seed = argc > 4 ? std::atoi(argv[4]) : 1107;
    // a layer's experts live in one shard (Swift 1.5: layers 12-47 in shard 2): shard 1, else its sibling shard 2
    std::string path = argv[1];
    auto has_layer = [&](const strata::GgufFile& g) {
        for (const auto& ti : g.tensors())
            if (ti.name == "blk." + std::to_string(L) + ".ffn_gate_exps.weight") return true;
        return false;
    };
    if (!has_layer(strata::GgufFile(path))) {
        const size_t at = path.rfind("-00001-of-");
        if (at != std::string::npos) path.replace(at + 1, 5, "00002");
    }
    strata::GgufFile gguf(path);
    const strata::TensorInfo *gate = nullptr, *up = nullptr, *down = nullptr;
    for (const auto& ti : gguf.tensors()) {
        if (ti.name == "blk." + std::to_string(L) + ".ffn_gate_exps.weight") gate = &ti;
        if (ti.name == "blk." + std::to_string(L) + ".ffn_up_exps.weight") up = &ti;
        if (ti.name == "blk." + std::to_string(L) + ".ffn_down_exps.weight") down = &ti;
    }
    if (!gate || !up || !down) { std::fprintf(stderr, "layer %d has no experts in %s\n", L, path.c_str()); return 2; }
    cpu::NativeFmt f;
    std::string err;
    if (!cpu::native_fmt((int) gate->type, (int) down->type, H, FF, f, err)) {
        std::fprintf(stderr, "native format: %s\n", err.c_str());
        return 2;
    }
    if (f.gu_type == 42 && f.d_type == 42) { std::fprintf(stderr, "layer %d is Q2_0: not an i-quant layer\n", L); return 2; }
    auto blob_of = [&](int e) {
        std::vector<uint8_t> b(f.bytes);
        std::memcpy(b.data(), gguf.tensor_data(*gate) + (size_t) e * f.up_off, f.up_off);
        std::memcpy(b.data() + f.up_off, gguf.tensor_data(*up) + (size_t) e * f.up_off, f.up_off);
        std::memcpy(b.data() + f.down_off, gguf.tensor_data(*down) + (size_t) e * (f.bytes - f.down_off),
                    f.bytes - f.down_off);
        return b;
    };
    const std::vector<uint8_t> blob = blob_of(E), blob2 = blob_of((E + 1) % cpu::NE);
    std::mt19937 rng(seed);
    std::normal_distribution<float> normal(0.f, 1.f);
    std::vector<float> x((size_t) ENTRIES * H);
    for (auto& v : x) v = normal(rng);
    const auto layout = strata::kernels::native_expert_layout(f.gu_type, f.d_type, H, FF);

    // the 5060's verify path: plain q8_1 activations, no scales (i-quant)
    ck(cudaSetDevice(0), "primary device");
    void *dblob = nullptr, *dx = nullptr, *dxq = nullptr, *scratch = nullptr;
    float* dout = nullptr;
    unsigned long long* dptr = nullptr;
    int32_t *dstart = nullptr, *dn = nullptr, *ddst = nullptr, *dtok = nullptr;
    ck(cudaMalloc(&dblob, 2 * blob.size()), "blob alloc");
    ck(cudaMalloc(&dx, x.size() * sizeof(float)), "x alloc");
    ck(cudaMalloc(&dxq, (size_t) ENTRIES * H / 32 * 36), "q8 alloc");
    ck(cudaMalloc(&scratch, strata::kernels::native_expert_scratch_bytes(ENTRIES, FF, H)), "scratch alloc");
    ck(cudaMalloc((void**) &dout, x.size() * sizeof(float)), "out alloc");
    ck(cudaMalloc((void**) &dptr, 2 * sizeof(unsigned long long)), "pointer alloc");
    ck(cudaMalloc((void**) &dstart, 3 * sizeof(int32_t)), "start alloc");
    ck(cudaMalloc((void**) &dn, sizeof(int32_t)), "count alloc");
    ck(cudaMalloc((void**) &ddst, ENTRIES * sizeof(int32_t)), "dst alloc");
    ck(cudaMalloc((void**) &dtok, ENTRIES * sizeof(int32_t)), "tok alloc");
    ck(cudaMemcpy(dblob, blob.data(), blob.size(), cudaMemcpyHostToDevice), "blob copy");
    ck(cudaMemcpy((uint8_t*) dblob + blob.size(), blob2.data(), blob2.size(), cudaMemcpyHostToDevice), "blob2 copy");
    ck(cudaMemcpy(dx, x.data(), x.size() * sizeof(float), cudaMemcpyHostToDevice), "x copy");
    // entries 0 and 1 use expert E, entry 2 uses E+1 (the runner's grouping of slots {0, 0, 1})
    const unsigned long long ptr[2] = {(unsigned long long) dblob, (unsigned long long) ((uint8_t*) dblob + blob.size())};
    const int32_t start[3] = {0, 2, ENTRIES}, groups = 2, dst[ENTRIES] = {0, 1, 2}, tok[ENTRIES] = {0, 1, 2};
    ck(cudaMemcpy(dptr, ptr, sizeof ptr, cudaMemcpyHostToDevice), "pointer copy");
    ck(cudaMemcpy(dstart, start, sizeof start, cudaMemcpyHostToDevice), "start copy");
    ck(cudaMemcpy(dn, &groups, sizeof groups, cudaMemcpyHostToDevice), "count copy");
    ck(cudaMemcpy(ddst, dst, sizeof dst, cudaMemcpyHostToDevice), "dst copy");
    ck(cudaMemcpy(dtok, tok, sizeof tok, cudaMemcpyHostToDevice), "tok copy");
    cudaStream_t stream = nullptr;
    ck(cudaStreamCreate(&stream), "stream");
    strata::kernels::quantize_q8_1_rows((const float*) dx, ENTRIES, H, dxq, stream);
    strata::kernels::native_expert_grouped(layout, dptr, dstart, dn, ddst, dtok, 2, ENTRIES, dxq, scratch, dout, stream,
                                            nullptr);
    ck(cudaStreamSynchronize(stream), "primary expert kernel");
    std::vector<float> reference(x.size()), runner_out(x.size(), 0.f);
    ck(cudaMemcpy(reference.data(), dout, reference.size() * sizeof(float), cudaMemcpyDeviceToHost), "primary out");

    // the 4070 tier: two arena slots and the runner
    ck(cudaSetDevice(1), "secondary device");
    strata::core::SecondaryArena arena;
    if (!arena.open(1, {(uint64_t) blob.size(), (uint64_t) blob2.size()}, 8ull << 20, err) ||
        !arena.fill_slot(0, blob.data(), blob.size(), err) || !arena.verify_slot(0, blob.data(), blob.size(), err) ||
        !arena.fill_slot(1, blob2.data(), blob2.size(), err) || !arena.verify_slot(1, blob2.data(), blob2.size(), err)) {
        std::fprintf(stderr, "secondary arena: %s\n", err.c_str());
        return 2;
    }
    // a 4070 swap copies a newcomer into its victim's slot, possibly from another layer: slot_bytes is its room
    if (arena.slot_bytes(0) != (blob.size() + 255) / 256 * 256 || arena.slot_bytes(2) != 0) {
        std::fprintf(stderr, "secondary arena: slot_bytes %llu for a %zu-byte blob\n",
                     (unsigned long long) arena.slot_bytes(0), blob.size());
        return 1;
    }
    strata::core::SecondaryRunner runner;
    const int32_t selected_slots[ENTRIES] = {0, 0, 1};
    if (!runner.init(ENTRIES, ENTRIES, H, FF, err, strata::core::kSecondaryReserveBytes) ||
        !runner.launch(layout, arena, x.data(), selected_slots, ENTRIES, 1, err) ||
        !runner.finish(runner_out.data(), err)) {
        std::fprintf(stderr, "secondary runner: %s\n", err.c_str());
        return 1;
    }
    int different = 0;
    double max_abs = 0.;
    for (size_t i = 0; i < x.size(); ++i) {
        different += std::memcmp(&reference[i], &runner_out[i], sizeof(float)) != 0;
        max_abs = std::max(max_abs, (double) std::fabs(reference[i] - runner_out[i]));
    }
    std::printf("layer %d (gate/up %d, down %d), experts %d/%d: 4070 runner vs 5060 verify path, differing %d/%zu "
                "floats, max |diff| %.3g\n", L, f.gu_type, f.d_type, E, (E + 1) % cpu::NE, different, x.size(), max_abs);
    return different ? 1 : 0;
}
