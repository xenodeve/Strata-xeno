#include "strata/core/secondary_budget.hpp"
#include "strata/core/secondary_vram.hpp"

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <string>

namespace {
constexpr size_t kBytes = 1u << 20;

__global__ void transform(const uint8_t* input, uint8_t* output, size_t bytes) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < bytes) output[i] = input[i] ^ 0x5a;
}

bool check(cudaError_t error, const char* operation) {
    if (error == cudaSuccess) return true;
    std::fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(error));
    return false;
}
} // namespace

int main() {
    if (!check(cudaSetDevice(0), "primary device") ||
        !check(cudaFree(nullptr), "primary context")) return 1;
    uint8_t *source = nullptr, *result = nullptr;
    if (!check(cudaHostAlloc((void**) &source, kBytes, cudaHostAllocPortable | cudaHostAllocMapped),
               "primary pinned source") ||
        !check(cudaHostAlloc((void**) &result, kBytes, cudaHostAllocPortable | cudaHostAllocMapped),
               "primary pinned result")) return 1;
    for (size_t i = 0; i < kBytes; ++i) { source[i] = (uint8_t) i; result[i] = 0; }

    if (!check(cudaSetDevice(1), "secondary device")) return 1;
    cudaDeviceProp primary{}, secondary{};
    if (!check(cudaGetDeviceProperties(&primary, 0), "primary identity") ||
        !check(cudaGetDeviceProperties(&secondary, 1), "secondary identity") ||
        std::string(primary.name).find("5060 Ti") == std::string::npos ||
        std::string(secondary.name).find("4070 SUPER") == std::string::npos) {
        std::fprintf(stderr, "expected CUDA_VISIBLE_DEVICES=1,0\n");
        return 1;
    }
    uint8_t *device_input = nullptr, *device_output = nullptr;
    cudaStream_t stream = nullptr;
    cudaEvent_t done = nullptr;
    if (!check(cudaMalloc((void**) &device_input, kBytes), "secondary input") ||
        !check(cudaMalloc((void**) &device_output, kBytes), "secondary output") ||
        !check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "secondary stream") ||
        !check(cudaEventCreateWithFlags(&done, cudaEventDisableTiming), "secondary event")) return 1;
    if (!check(cudaMemcpyAsync(device_input, source, kBytes, cudaMemcpyHostToDevice, stream), "H2D")) return 1;
    transform<<<(unsigned) ((kBytes + 255) / 256), 256, 0, stream>>>(device_input, device_output, kBytes);
    if (!check(cudaGetLastError(), "transform launch") ||
        !check(cudaMemcpyAsync(result, device_output, kBytes, cudaMemcpyDeviceToHost, stream), "D2H") ||
        !check(cudaEventRecord(done, stream), "record completion") ||
        !check(cudaEventSynchronize(done), "wait completion")) return 1;
    for (size_t i = 0; i < kBytes; ++i) {
        if (result[i] != (uint8_t) (source[i] ^ 0x5a)) {
            std::fprintf(stderr, "cross-device pinned result differs at byte %zu\n", i);
            return 1;
        }
    }
    size_t cuda_free = 0, cuda_total = 0;
    uint64_t nvml_free = 0;
    std::string err;
    if (!check(cudaMemGetInfo(&cuda_free, &cuda_total), "secondary free") ||
        !strata::core::secondary_nvml_free_bytes(1, nvml_free, err) ||
        strata::core::secondary_effective_free(cuda_free, nvml_free) <
            strata::core::kSecondaryReserveBytes) {
        std::fprintf(stderr, "secondary reserve failed: %s\n", err.c_str());
        return 1;
    }
    std::printf("two CUDA contexts: pinned H2D, kernel, event and pinned D2H passed; lower free %.3f GiB\n",
                (double) strata::core::secondary_effective_free(cuda_free, nvml_free) / 1073741824.0);
    cudaEventDestroy(done);
    cudaStreamDestroy(stream);
    cudaFree(device_output);
    cudaFree(device_input);
    cudaSetDevice(0);
    cudaFreeHost(result);
    cudaFreeHost(source);
    return 0;
}
