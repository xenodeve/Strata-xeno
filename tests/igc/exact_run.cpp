#include "strata/core/igc_exact.hpp"
#include <cuda_runtime.h>
#include <bit>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <vector>

namespace {
void check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
struct Device {
    void* input = nullptr;
    void* output = nullptr;
    cudaStream_t stream = nullptr;
    ~Device() {
        if (stream) cudaStreamSynchronize(stream);
        if (output) cudaFree(output);
        if (input) cudaFree(input);
        if (stream) cudaStreamDestroy(stream);
    }
};
}
int main(int argc, char** argv) try {
    using namespace strata::igc;
    static_assert(std::endian::native == std::endian::little);
    if (argc != 3 && argc != 4) throw std::runtime_error("expected input and output record paths");
    const bool sequence = argc == 4 && std::strcmp(argv[3], "--sequence") == 0;
    if (argc == 4 && !sequence) throw std::runtime_error("unknown_argument");
    std::ifstream source(argv[1], std::ios::binary | std::ios::ate);
    if (!source) throw std::runtime_error("record_input_open");
    const auto bytes = source.tellg();
    if (bytes <= 0 || bytes > static_cast<std::streamoff>(sizeof(ExactRequest) * kMaxBatch) ||
            bytes % sizeof(ExactRequest)) throw std::runtime_error("record_input_bytes");
    const auto count = static_cast<uint32_t>(bytes / sizeof(ExactRequest));
    std::vector<ExactRequest> input(count);
    source.seekg(0);
    source.read(reinterpret_cast<char*>(input.data()), bytes);
    if (!source) throw std::runtime_error("record_input_read");
    Device device;
    check(cudaStreamCreateWithFlags(&device.stream, cudaStreamNonBlocking));
    check(cudaMalloc(&device.input, static_cast<size_t>(bytes)));
    check(cudaMalloc(&device.output, count * sizeof(ExactResult)));
    check(cudaMemcpyAsync(device.input, input.data(), static_cast<size_t>(bytes), cudaMemcpyHostToDevice, device.stream));
    std::vector<ExactResult> first(count), repeated(count);
    std::string error;
    for (auto* values : {&first, &repeated}) {
        if (sequence) {
            for (uint32_t i = 0; i < count; ++i) {
                check(cudaMemcpyAsync(device.input, &input[i], sizeof(ExactRequest), cudaMemcpyHostToDevice, device.stream));
                if (!execute_exact(static_cast<const ExactRequest*>(device.input), static_cast<ExactResult*>(device.output),
                                    1, device.stream, error)) throw std::runtime_error(error);
                check(cudaMemcpyAsync(&(*values)[i], device.output, sizeof(ExactResult), cudaMemcpyDeviceToHost, device.stream));
            }
        } else {
            if (!execute_exact(static_cast<const ExactRequest*>(device.input), static_cast<ExactResult*>(device.output),
                                count, device.stream, error)) throw std::runtime_error(error);
            check(cudaMemcpyAsync(values->data(), device.output, count * sizeof(ExactResult), cudaMemcpyDeviceToHost, device.stream));
        }
        check(cudaStreamSynchronize(device.stream));
    }
    if (std::memcmp(first.data(), repeated.data(), count * sizeof(ExactResult))) throw std::runtime_error("repeat_mismatch");
    std::ofstream target(argv[2], std::ios::binary);
    target.write(reinterpret_cast<const char*>(first.data()), count * sizeof(ExactResult));
    if (!target) throw std::runtime_error("record_output_write");
    std::printf("PASS gpu_exact count=%u repeat=bitwise request_bytes=%zu result_bytes=%zu mode=%s\n", count,
                sizeof(ExactRequest), sizeof(ExactResult), sequence ? "sequence" : "batch");
    return 0;
} catch (const std::exception& error) {
    std::fprintf(stderr, "igc_exact_run: %s\n", error.what());
    return 2;
}
