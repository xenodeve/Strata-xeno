// The actual native-Q2_0 miss pool versus the verifier's grouped GPU hit.
// Uses one GGUF expert and one deterministic activation, so placement is the only variable.
#include "strata/artifact/gguf_reader.hpp"
#include "strata/core/secondary_arena.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/cpu/pool.hpp"
#include "strata/kernels/iq_kernels.hpp"

#include <cuda_runtime.h>
#include <cuda_fp16.h>

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
    if (argc < 2 || argc > 6) {
        std::fprintf(stderr, "usage: native_q2_pool_hit_parity <q2_0-first-shard.gguf> [layer] [expert] [seed] [input.bin|--secondary]\n");
        return 2;
    }
    constexpr int H = cpu::H, FF = cpu::FF, ENTRIES = 3;
    const int L = argc > 2 ? std::atoi(argv[2]) : 0;
    const int E = argc > 3 ? std::atoi(argv[3]) : 7;
    const int seed = argc > 4 ? std::atoi(argv[4]) : 1107;
    const bool secondary = argc > 5 && std::strcmp(argv[5], "--secondary") == 0;
    if (secondary) {
        ck(cudaSetDevice(0), "primary device");
        ck(cudaFree(nullptr), "primary context");
        ck(cudaSetDevice(1), "secondary device");
    }
    if (L < 0 || L >= 48 || E < 0 || E >= cpu::NE) return 2;
    strata::GgufFile gguf(argv[1]);
    const strata::TensorInfo* gate = nullptr;
    const strata::TensorInfo* up = nullptr;
    const strata::TensorInfo* down = nullptr;
    for (const auto& ti : gguf.tensors()) {
        if (ti.name == "blk." + std::to_string(L) + ".ffn_gate_exps.weight") gate = &ti;
        if (ti.name == "blk." + std::to_string(L) + ".ffn_up_exps.weight") up = &ti;
        if (ti.name == "blk." + std::to_string(L) + ".ffn_down_exps.weight") down = &ti;
    }
    if (!gate || !up || !down) return 2;
    cpu::NativeFmt f;
    std::string err;
    if (!cpu::native_fmt((int) gate->type, (int) down->type, H, FF, f, err) || f.gu_type != 42 || f.d_type != 42) {
        std::fprintf(stderr, "native Q2_0 format: %s\n", err.c_str());
        return 2;
    }
    std::vector<uint8_t> blob(f.bytes);
    std::memcpy(blob.data(), gguf.tensor_data(*gate) + (size_t) E * f.up_off, f.up_off);
    std::memcpy(blob.data() + f.up_off, gguf.tensor_data(*up) + (size_t) E * f.up_off, f.up_off);
    std::memcpy(blob.data() + f.down_off, gguf.tensor_data(*down) + (size_t) E * (f.bytes - f.down_off),
                f.bytes - f.down_off);
    const int E2 = (E + 1) % cpu::NE;
    std::vector<uint8_t> blob2(f.bytes);
    std::memcpy(blob2.data(), gguf.tensor_data(*gate) + (size_t) E2 * f.up_off, f.up_off);
    std::memcpy(blob2.data() + f.up_off, gguf.tensor_data(*up) + (size_t) E2 * f.up_off, f.up_off);
    std::memcpy(blob2.data() + f.down_off, gguf.tensor_data(*down) + (size_t) E2 * (f.bytes - f.down_off),
                f.bytes - f.down_off);

    std::mt19937 rng(seed);
    std::normal_distribution<float> normal(0.f, 1.f);
    std::vector<float> x((size_t) ENTRIES * H), pool_out((size_t) ENTRIES * H), hit_out((size_t) ENTRIES * H);
    if (argc > 5 && !secondary) {
        std::FILE* f = std::fopen(argv[5], "rb");
        if (!f) return 2;
        int32_t header[3] = {};
        if (std::fread(header, sizeof header, 1, f) != 1 || header[1] != L || header[2] < 1 || header[2] > 15) return 2;
        std::vector<int32_t> ids((size_t) header[2]);
        std::vector<float> source(H);
        if (std::fread(ids.data(), sizeof(int32_t), ids.size(), f) != ids.size() ||
            std::fread(source.data(), sizeof(float), source.size(), f) != source.size()) return 2;
        std::fclose(f);
        for (int t = 0; t < ENTRIES; ++t)
            std::memcpy(x.data() + (size_t) t * H, source.data(), H * sizeof(float));
    } else if (seed == -1 || seed == -2) {
        for (int c = 0; c < ENTRIES * H / 32; ++c) {
            const float maximum = seed == -1 ? 1.f : 1.000123f * (1.f + 0.001f * c);
            x[32 * c] = maximum;
            for (int j = 1; j < 32; ++j) {
                float value = ((float) (j - 16) + 0.5f) * (maximum / 127.f);
                if (seed == -2) value = std::nextafter(value, j & 1 ? -INFINITY : INFINITY);
                x[32 * c + j] = value;
            }
        }
    } else {
        for (auto& v : x) v = normal(rng);
    }
    std::vector<cpu::ActQ> act(ENTRIES);
    cpu::ExpertJobMulti jobs[2];
    jobs[0].blob = blob.data();
    jobs[0].nt = 2;
    jobs[1].blob = blob2.data();
    jobs[1].nt = 1;
    for (int t = 0; t < ENTRIES; ++t) {
        cpu::act_quant_any(x.data() + (size_t) t * H, H, act[t]);
        const int group = t == 2 ? 1 : 0;
        const int entry = t == 2 ? 0 : t;
        jobs[group].act[entry] = &act[t];
        jobs[group].out[entry] = pool_out.data() + (size_t) t * H;
    }
    cpu::ExpertPool pool(1, false, true);
    pool.run_split_multi_native(f, jobs, 2);

    const auto layout = strata::kernels::native_expert_layout(f.gu_type, f.d_type, H, FF);
    strata::core::SecondaryArena secondary_arena;
    void *dblob = nullptr, *dx = nullptr, *dxq = nullptr, *scratch = nullptr;
    float* dx_scales = nullptr;
    float* dout = nullptr;
    unsigned long long* dptr = nullptr;
    int32_t *dstart = nullptr, *dn = nullptr, *ddst = nullptr, *dtok = nullptr;
    if (secondary) {
        std::string secondary_err;
        if (!secondary_arena.open(1, {(uint64_t) blob.size(), (uint64_t) blob2.size()},
                                  8ull << 20, secondary_err) ||
            !secondary_arena.fill_slot(0, blob.data(), blob.size(), secondary_err) ||
            !secondary_arena.verify_slot(0, blob.data(), blob.size(), secondary_err) ||
            !secondary_arena.fill_slot(1, blob2.data(), blob2.size(), secondary_err) ||
            !secondary_arena.verify_slot(1, blob2.data(), blob2.size(), secondary_err)) {
            std::fprintf(stderr, "secondary parity: %s\n", secondary_err.c_str());
            return 2;
        }
        std::printf("secondary parity: two CUDA contexts, two verified 4070 slots, lower free %.3f GiB\n",
                    (double) secondary_arena.lower_free_after() / 1073741824.0);
    } else {
        ck(cudaMalloc(&dblob, 2 * blob.size()), "blob alloc");
    }
    ck(cudaMalloc(&dx, x.size() * sizeof(float)), "x alloc");
    ck(cudaMalloc(&dxq, (size_t) ENTRIES * H / 32 * 36), "q8 alloc");
    ck(cudaMalloc((void**) &dx_scales, (size_t) ENTRIES * H / 32 * sizeof(float)), "scale alloc");
    ck(cudaMalloc(&scratch, strata::kernels::native_expert_scratch_bytes(ENTRIES, FF)), "scratch alloc");
    ck(cudaMalloc((void**) &dout, (size_t) ENTRIES * H * sizeof(float)), "out alloc");
    ck(cudaMalloc((void**) &dptr, 2 * sizeof(unsigned long long)), "pointer alloc");
    ck(cudaMalloc((void**) &dstart, 3 * sizeof(int32_t)), "start alloc");
    ck(cudaMalloc((void**) &dn, sizeof(int32_t)), "count alloc");
    ck(cudaMalloc((void**) &ddst, ENTRIES * sizeof(int32_t)), "dst alloc");
    ck(cudaMalloc((void**) &dtok, ENTRIES * sizeof(int32_t)), "tok alloc");
    if (!secondary) {
        ck(cudaMemcpy(dblob, blob.data(), blob.size(), cudaMemcpyHostToDevice), "blob copy");
        ck(cudaMemcpy((uint8_t*) dblob + blob.size(), blob2.data(), blob2.size(), cudaMemcpyHostToDevice), "blob2 copy");
    }
    ck(cudaMemcpy(dx, x.data(), x.size() * sizeof(float), cudaMemcpyHostToDevice), "x copy");
    const unsigned long long ptr[2] = {
        (unsigned long long) (secondary ? secondary_arena.slot_ptr(0) : (uint8_t*) dblob),
        (unsigned long long) (secondary ? secondary_arena.slot_ptr(1) : (uint8_t*) dblob + blob.size())};
    const int32_t start[3] = {0, 2, ENTRIES}, groups = 2, dst[ENTRIES] = {0, 1, 2}, tok[ENTRIES] = {0, 1, 2};
    ck(cudaMemcpy(dptr, ptr, sizeof ptr, cudaMemcpyHostToDevice), "pointer copy");
    ck(cudaMemcpy(dstart, start, sizeof start, cudaMemcpyHostToDevice), "start copy");
    ck(cudaMemcpy(dn, &groups, sizeof groups, cudaMemcpyHostToDevice), "count copy");
    ck(cudaMemcpy(ddst, dst, sizeof dst, cudaMemcpyHostToDevice), "dst copy");
    ck(cudaMemcpy(dtok, tok, sizeof tok, cudaMemcpyHostToDevice), "tok copy");
    cudaStream_t stream = nullptr;
    ck(cudaStreamCreate(&stream), "stream");
    strata::kernels::quantize_q8_1_rows_scaled((const float*) dx, ENTRIES, H, dxq, dx_scales, stream);
    strata::kernels::native_expert_grouped(layout, dptr, dstart, dn, ddst, dtok, 2, ENTRIES, dxq, scratch, dout, stream,
                                            dx_scales);
    ck(cudaStreamSynchronize(stream), "expert kernel");
    std::vector<uint8_t> q8((size_t) ENTRIES * H / 32 * 36);
    std::vector<float> gpu_scales((size_t) ENTRIES * H / 32);
    ck(cudaMemcpy(q8.data(), dxq, q8.size(), cudaMemcpyDeviceToHost), "activation copy");
    ck(cudaMemcpy(gpu_scales.data(), dx_scales, gpu_scales.size() * sizeof(float), cudaMemcpyDeviceToHost), "scale copy");
    ck(cudaMemcpy(hit_out.data(), dout, hit_out.size() * sizeof(float), cudaMemcpyDeviceToHost), "output copy");
    int different_codes = 0;
    for (int i = 0; i < ENTRIES * H; ++i)
        different_codes += act[i / H].q[i % H] != (int8_t) q8[(size_t) (i / 32) * 36 + 4 + (i % 32)];
    uint16_t gpu_scale_bits = 0;
    std::memcpy(&gpu_scale_bits, q8.data(), sizeof gpu_scale_bits);
    std::printf("input quantization: differing codes %d/%d, first CPU scale %.9g, GPU fp16 bits 0x%04x\n",
                different_codes, ENTRIES * H, act[0].scale[0], gpu_scale_bits);
    int different_scales = 0;
    for (int c = 0; c < ENTRIES * H / 32; ++c)
        different_scales += act[c / (H / 32)].scale[c % (H / 32)] != gpu_scales[c];
    std::printf("input fp32 scales: differing %d/%d, first GPU scale %.9g\n", different_scales,
                ENTRIES * H / 32, gpu_scales[0]);
    const cpu::ActQ* acts[] = {&act[0]};
    std::vector<float> cpu_gate(FF), gpu_gate(FF), rounded_gate(FF);
    float* cpu_gate_ptr[] = {cpu_gate.data()};
    cpu::q2_rows_any(blob.data(), f.gu_row, H / 64, acts, 1, cpu_gate_ptr, 0, FF);
    std::vector<float> grouped_gate(FF), grouped_up(FF), grouped_hidden(FF), cpu_up(FF);
    float* cpu_up_ptr[] = {cpu_up.data()};
    cpu::q2_rows_any(blob.data() + f.up_off, f.gu_row, H / 64, acts, 1, cpu_up_ptr, 0, FF);
    ck(cudaMemcpy(grouped_gate.data(), scratch, FF * sizeof(float), cudaMemcpyDeviceToHost), "all gate copy");
    ck(cudaMemcpy(grouped_up.data(), (uint8_t*) scratch + ENTRIES * FF * sizeof(float), FF * sizeof(float),
                  cudaMemcpyDeviceToHost), "all up copy");
    ck(cudaMemcpy(grouped_hidden.data(), (uint8_t*) scratch + 2 * ENTRIES * FF * sizeof(float), FF * sizeof(float),
                  cudaMemcpyDeviceToHost), "hidden copy");
    int gate_different = 0, up_different = 0, hidden_different = 0;
    for (int i = 0; i < FF; ++i) {
        gate_different += cpu_gate[i] != grouped_gate[i];
        up_different += cpu_up[i] != grouped_up[i];
        const float cpu_hidden = (cpu_gate[i] / (1.f + (float) std::exp(-(double) cpu_gate[i]))) * cpu_up[i];
        if (cpu_hidden != grouped_hidden[i]) {
            ++hidden_different;
            const float double_exp = (float) std::exp((double) -cpu_gate[i]);
            const float double_hidden = (cpu_gate[i] / (1.f + double_exp)) * cpu_up[i];
            std::printf("hidden mismatch row %d gate %.9g up %.9g CPU %.9g GPU %.9g CPU-double %.9g\n",
                        i, cpu_gate[i], cpu_up[i], cpu_hidden, grouped_hidden[i], double_hidden);
        }
    }
    std::printf("gate/up/hidden differing: %d/%d, %d/%d, %d/%d\n", gate_different, FF,
                up_different, FF, hidden_different, FF);
    float* dgate = nullptr;
    ck(cudaMalloc((void**) &dgate, FF * sizeof(float)), "gate alloc");
    strata::kernels::iq_mmvq(42, secondary ? (void*) secondary_arena.slot_ptr(0) : dblob,
                              dxq, dgate, H, FF, 1, stream);
    ck(cudaStreamSynchronize(stream), "gate kernel");
    ck(cudaMemcpy(gpu_gate.data(), dgate, FF * sizeof(float), cudaMemcpyDeviceToHost), "gate copy");
    cpu::ActQ rounded_act = act[0];
    for (int c = 0; c < H / 32; ++c) {
        __half hs;
        std::memcpy(&hs, q8.data() + (size_t) c * 36, sizeof hs);
        rounded_act.scale[c] = __half2float(hs);
        rounded_act.hx[c] = rounded_act.scale[c] * rounded_act.sum[c];
    }
    const cpu::ActQ* rounded_acts[] = {&rounded_act};
    float* rounded_gate_ptr[] = {rounded_gate.data()};
    cpu::q2_rows_any(blob.data(), f.gu_row, H / 64, rounded_acts, 1, rounded_gate_ptr, 0, FF);
    double gate_l1 = 0., rounded_l1 = 0., gate_mag = 0.;
    for (int i = 0; i < FF; ++i) {
        gate_l1 += std::fabs((double) cpu_gate[i] - gpu_gate[i]);
        rounded_l1 += std::fabs((double) rounded_gate[i] - gpu_gate[i]);
        gate_mag += std::fabs((double) gpu_gate[i]);
    }
    std::printf("gate relative L1: CPU fp32 %.9g, CPU with GPU scales %.9g\n",
                gate_l1 / (gate_mag + 1e-30), rounded_l1 / (gate_mag + 1e-30));
    int different = 0;
    double l1 = 0., reference = 0., max_abs = 0.;
    for (int i = 0; i < ENTRIES * H; ++i) {
        const double diff = std::fabs((double) pool_out[i] - hit_out[i]);
        different += pool_out[i] != hit_out[i];
        l1 += diff;
        reference += std::fabs((double) pool_out[i]);
        if (diff > max_abs) max_abs = diff;
    }
    std::printf("layer %d expert %d: differing %d/%d, relative L1 %.9g, max abs %.9g\n", L, E, different, ENTRIES * H,
                l1 / (reference + 1e-30), max_abs);
    if (!secondary) cudaFree(dblob);
    cudaFree(dx); cudaFree(dxq); cudaFree(dx_scales); cudaFree(scratch); cudaFree(dout);
    cudaFree(dptr); cudaFree(dstart); cudaFree(dn); cudaFree(ddst); cudaFree(dtok); cudaFree(dgate);
    cudaStreamDestroy(stream);
    return (different_codes || different_scales || gate_different || up_different || hidden_different || different) ? 1 : 0;
}
