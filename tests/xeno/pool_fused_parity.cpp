// #147: the CPU expert pool's native multi-token path in one phase per batch - each expert's down rows start once its
// own gate/up rows are done (the worker that finishes them quantizes the SwiGLU output) - instead of two phases with
// a barrier and a host quantize between them.  The same kernels over the same row ranges: the output must be
// bit-identical to the two-phase path, for any expert count, token count and worker count.
// usage: xeno_pool_fused_parity <i-quant first shard .gguf> [layer]
#include "strata/artifact/gguf_reader.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/kernels/cpu/pool.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace cpu = strata::kernels::cpu;

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: xeno_pool_fused_parity <gguf> [layer]\n"); return 2; }
    constexpr int H = cpu::H, FF = cpu::FF;
    const int L = argc > 2 ? std::atoi(argv[2]) : 5;   // the IQ2_XS first shard holds layers 0-12
    strata::GgufFile gguf(argv[1]);
    const strata::TensorInfo *gate = nullptr, *up = nullptr, *down = nullptr;
    for (const auto& ti : gguf.tensors()) {
        if (ti.name == "blk." + std::to_string(L) + ".ffn_gate_exps.weight") gate = &ti;
        if (ti.name == "blk." + std::to_string(L) + ".ffn_up_exps.weight") up = &ti;
        if (ti.name == "blk." + std::to_string(L) + ".ffn_down_exps.weight") down = &ti;
    }
    if (!gate || !up || !down) { std::fprintf(stderr, "layer %d not in this shard\n", L); return 2; }
    cpu::NativeFmt f;
    std::string err;
    if (!cpu::native_fmt((int) gate->type, (int) down->type, H, FF, f, err)) { std::fprintf(stderr, "%s\n", err.c_str()); return 2; }
    constexpr int NEXP = 12;
    std::vector<std::vector<uint8_t>> blobs(NEXP, std::vector<uint8_t>(f.bytes));
    for (int i = 0; i < NEXP; ++i) {
        const int e = 37 * i + 5;
        std::memcpy(blobs[i].data(), gguf.tensor_data(*gate) + (size_t) e * f.up_off, f.up_off);
        std::memcpy(blobs[i].data() + f.up_off, gguf.tensor_data(*up) + (size_t) e * f.up_off, f.up_off);
        std::memcpy(blobs[i].data() + f.down_off, gguf.tensor_data(*down) + (size_t) e * (f.bytes - f.down_off),
                    f.bytes - f.down_off);
    }
    std::mt19937 rng(147);
    std::normal_distribution<float> normal(0.f, 1.f);
    constexpr int T = 4;
    std::vector<float> x((size_t) T * H);
    for (auto& v : x) v = normal(rng);
    std::vector<cpu::ActQ> act(T);
    std::vector<uint8_t> nact((size_t) T * cpu::kNativeActBytes);
    for (int t = 0; t < T; ++t) {
        cpu::act_quant_any(x.data() + (size_t) t * H, H, act[t]);
        cpu::native_quant_act(f, x.data() + (size_t) t * H, nact.data() + (size_t) t * cpu::kNativeActBytes);
    }
    int bad = 0, cases = 0;
    for (int workers : {1, 4, 13})
        for (int n : {1, 2, 5, NEXP}) {
            std::vector<float> two((size_t) NEXP * T * H, 0.f), one((size_t) NEXP * T * H, 0.f);
            for (int fused = 1; fused >= 0; --fused) {
                // a pool per arm: one arm's intermediate buffers must not stand in for the other's
                cpu::ExpertPool pool(workers, false, true);
                std::vector<float>& out = fused ? one : two;
                std::vector<cpu::ExpertJobMulti> jobs((size_t) n);
                for (int e = 0; e < n; ++e) {
                    jobs[e].blob = blobs[e].data();
                    jobs[e].nt = 1 + (e % T);   // 1-4 tokens per expert
                    for (int t = 0; t < jobs[e].nt; ++t) {
                        jobs[e].act[t] = &act[t];
                        jobs[e].nact[t] = nact.data() + (size_t) t * cpu::kNativeActBytes;
                        jobs[e].out[t] = out.data() + ((size_t) e * T + t) * H;
                    }
                }
                pool.set_fused(fused != 0);
                pool.run_split_multi_native(f, jobs.data(), n);
            }
            ++cases;
            if (std::memcmp(two.data(), one.data(), two.size() * sizeof(float)) != 0) {
                ++bad;
                std::printf("FAIL: %d workers, %d experts: the one-phase output differs\n", workers, n);
            }
        }
    std::printf("%s: %d of %d cases bit-identical (layer %d, gate/up type %d, down type %d)\n", bad ? "FAIL" : "PASS",
                cases - bad, cases, L, f.gu_type, f.d_type);
    return bad ? 1 : 0;
}
