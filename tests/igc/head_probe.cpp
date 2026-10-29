// Isolated diagnostic: replay the existing native head from a final HC residual.
// No decoder, adapter, arithmetic executor, training or language-state write.
#include "strata/core/native_head.hpp"
#include "strata/core/weights.hpp"
#include "strata/kernels/fused_gr.hpp"
#include "head_vjp.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <set>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace {
constexpr int N = 2560, HC = 4, LR = 320, V = 248320;
std::filesystem::path file_path(const std::string& text) {
    return std::filesystem::path(std::u8string(text.begin(), text.end()));
}
void check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
struct Buffer {
    void* data = nullptr;
    explicit Buffer(size_t bytes) { check(cudaMalloc(&data, bytes)); }
    ~Buffer() { if (data) cudaFree(data); }
    Buffer(const Buffer&) = delete;
    float* f() const { return static_cast<float*>(data); }
};
struct Stream {
    cudaStream_t value = nullptr;
    Stream() { check(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking)); }
    ~Stream() { if (value) { cudaStreamSynchronize(value); cudaStreamDestroy(value); } }
};
std::vector<float> read_vector(const std::string& path, size_t count, bool header) {
    std::ifstream file(file_path(path), std::ios::binary);
    if (!file) throw std::runtime_error("input_open");
    if (header) {
        int32_t shape[2]{};
        file.read(reinterpret_cast<char*>(shape), sizeof(shape));
        if (!file || shape[0] != HC || shape[1] != N) throw std::runtime_error("residual_geometry");
    }
    std::vector<float> result(count);
    file.read(reinterpret_cast<char*>(result.data()), static_cast<std::streamsize>(count * sizeof(float)));
    if (!file || file.peek() != std::char_traits<char>::eof()) throw std::runtime_error("input_bytes");
    for (float value : result)
        if (!std::isfinite(value)) throw std::runtime_error(header ? "residual_nonfinite" : "logits_nonfinite");
    return result;
}
void write_vector(const std::string& path, const std::vector<float>& values) {
    if (path.empty()) return;
    std::ofstream file(file_path(path), std::ios::binary);
    file.write(reinterpret_cast<const char*>(values.data()), static_cast<std::streamsize>(values.size() * sizeof(float)));
    if (!file) throw std::runtime_error("output_write");
}
}

int main(int argc, char** argv) try {
    std::string pack, model, residual, mixed_input, expected, mixed_out, logits_out, vjp_input, vjp_out;
    bool zero = false;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        if (key == "--zero") { zero = true; continue; }
        if (++i >= argc) throw std::runtime_error("missing_argument");
        const std::string value = argv[i];
        if (key == "--pack") pack = value;
        else if (key == "--model") model = value;
        else if (key == "--residual") residual = value;
        else if (key == "--mixed-input") mixed_input = value;
        else if (key == "--expected-logits") expected = value;
        else if (key == "--mixed-out") mixed_out = value;
        else if (key == "--logits-out") logits_out = value;
        else if (key == "--vjp-input") vjp_input = value;
        else if (key == "--vjp-out") vjp_out = value;
        else throw std::runtime_error("unknown_argument");
    }
    const int inputs = static_cast<int>(zero) + static_cast<int>(!residual.empty()) + static_cast<int>(!mixed_input.empty());
    if (pack.empty() || model.empty() || inputs != 1) throw std::runtime_error("arguments");
    if (vjp_input.empty() != vjp_out.empty()) throw std::runtime_error("vjp_arguments");
    const auto input = residual.empty() ? std::vector<float>(HC * N, 0.0f) : read_vector(residual, HC * N, true);
    const auto supplied_mixed = mixed_input.empty() ? std::vector<float>{} : read_vector(mixed_input, N, false);
    const auto wanted = expected.empty() ? std::vector<float>{} : read_vector(expected, V, false);
    const auto delta = vjp_input.empty() ? std::vector<float>{} : read_vector(vjp_input, V, false);
    // Validate input before touching the model, weights or device.
    const std::set<std::string> keep{"output_hc_norm.weight", "output_hc_down.weight", "output_hc_up.weight"};
    std::set<std::string> skip;
    std::ifstream index(file_path(pack) / "index.txt");
    if (!index) throw std::runtime_error("pack_index");
    for (std::string line; std::getline(index, line);) {
        std::istringstream row(line);
        std::string name;
        if (row >> name && name[0] != '#' && !keep.contains(name)) skip.insert(name);
    }
    uint64_t weight_bytes = 0;
    std::string error;
    if (!strata::core::WeightTable::pool_bytes(pack, weight_bytes, error, &skip)) throw std::runtime_error(error);
    if (weight_bytes > (32ull << 20)) throw std::runtime_error("head_pack_budget");
    Buffer weights(weight_bytes);
    strata::core::WeightTable table;
    if (!table.load(pack, weights.data, weight_bytes, error, &skip)) throw std::runtime_error(error);
    const auto* norm = table.find("output_hc_norm.weight");
    const auto* down = table.find("output_hc_down.weight");
    const auto* up = table.find("output_hc_up.weight");
    using strata::core::WeightKind;
    if (!norm || !down || !up || norm->kind != WeightKind::F32 || norm->elements != HC * N ||
            down->kind != WeightKind::Bf16InF32 || down->ne0 != HC * N || down->ne1 != LR ||
            up->kind != WeightKind::Bf16InF32 || up->ne0 != LR || up->ne1 != HC * N)
        throw std::runtime_error("head_weight_geometry");
    strata::core::NativeHead head;
    if (!head.load({model}, N, V, error)) throw std::runtime_error(error);
    if (!delta.empty() && head.type() != 8) throw std::runtime_error("vjp_requires_q8_0_weights");
    Stream stream;
    Buffer R(HC * N * sizeof(float)), xn(HC * N * sizeof(float)), lo(LR * sizeof(float));
    Buffer rs(HC * sizeof(float)), mixed(N * sizeof(float)), logits(V * sizeof(float));
    check(cudaMemcpyAsync(R.data, input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice, stream.value));
    strata::kernels::FusedGrArgs args;
    args.R = R.f(); args.R_out = R.f(); args.apply = false;
    args.w_norm = static_cast<const float*>(norm->data);
    args.w_down = static_cast<const uint16_t*>(down->data);
    args.w_up = static_cast<const uint16_t*>(up->data);
    args.eps = 1e-6f; args.lo = lo.f(); args.rs = rs.f(); args.mixed = mixed.f();
    std::vector<float> first(V), again(V), host_mixed(N);
    for (auto* output : {&first, &again}) {
        if (supplied_mixed.empty()) strata::kernels::fused_gr_read_multi(&args, 1, xn.f(), stream.value);
        else check(cudaMemcpyAsync(mixed.data, supplied_mixed.data(), N * sizeof(float), cudaMemcpyHostToDevice, stream.value));
        if (!head.run(mixed.f(), logits.f(), stream.value, error)) throw std::runtime_error(error);
        check(cudaMemcpyAsync(output->data(), logits.data, V * sizeof(float), cudaMemcpyDeviceToHost, stream.value));
        check(cudaStreamSynchronize(stream.value));
    }
    if (std::memcmp(first.data(), again.data(), V * sizeof(float))) throw std::runtime_error("repeat_mismatch");
    for (float value : first)
        if (!std::isfinite(value) || (zero && value != 0.0f)) throw std::runtime_error("zero_or_finite_control");
    check(cudaMemcpy(host_mixed.data(), mixed.data, N * sizeof(float), cudaMemcpyDeviceToHost));
    write_vector(mixed_out, host_mixed);
    write_vector(logits_out, first);
    if (!delta.empty()) {
        Buffer device_delta(V * sizeof(float)), device_gradient(N * sizeof(float));
        check(cudaMemcpyAsync(device_delta.data, delta.data(), V * sizeof(float), cudaMemcpyHostToDevice, stream.value));
        igc_head_weight_vjp(head.weights(), device_delta.f(), device_gradient.f(), N, V, stream.value);
        std::vector<float> gradient(N);
        check(cudaMemcpyAsync(gradient.data(), device_gradient.data, N * sizeof(float), cudaMemcpyDeviceToHost, stream.value));
        check(cudaStreamSynchronize(stream.value));
        for (float value : gradient) if (!std::isfinite(value)) throw std::runtime_error("vjp_nonfinite");
        write_vector(vjp_out, gradient);
        std::puts("gradient=dequant_weight_ste");
    }
    const auto pick = static_cast<int>(std::max_element(first.begin(), first.end()) - first.begin());
    if (!wanted.empty()) {
        size_t different = 0;
        double max_delta = 0;
        for (int i = 0; i < V; ++i) {
            different += std::memcmp(&first[i], &wanted[i], sizeof(float)) != 0;
            max_delta = std::max(max_delta, std::abs(static_cast<double>(first[i]) - wanted[i]));
        }
        const auto expected_pick = static_cast<int>(std::max_element(wanted.begin(), wanted.end()) - wanted.begin());
        std::printf("compare bitwise_differences=%zu max_abs=%.9g argmax=%d expected_argmax=%d\n", different, max_delta, pick, expected_pick);
        if (different) return 1;
    }
    std::printf("PASS %s logits=%d repeat=bitwise argmax=%d head_bytes=%llu mixer_bytes=%llu\n",
                zero ? "zero" : "replay", V, pick, static_cast<unsigned long long>(head.weight_bytes()),
                static_cast<unsigned long long>(weight_bytes));
    return 0;
} catch (const std::exception& error) {
    std::fprintf(stderr, "igc_head_probe: %s\n", error.what());
    return 2;
}
