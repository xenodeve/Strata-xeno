#include "head_vjp.hpp"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdexcept>

namespace {
struct Q80Block { half d; signed char q[32]; };
static_assert(sizeof(Q80Block) == 34);
// Each warp loads one weight row's 32 contiguous columns. Reduction order is
// fixed; no float atomics or updates to frozen model weights.
__global__ void transpose_product(const Q80Block* weights, const float* delta,
                                   float* output, int blocks_per_row, int rows) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    float sum = 0;
    for (int row = warp; row < rows; row += 8) {
        const float coefficient = delta[row];
        if (coefficient != 0.0f) {
            const Q80Block& block = weights[static_cast<size_t>(row) * blocks_per_row + blockIdx.x];
            sum += (__half2float(block.d) * static_cast<float>(block.q[lane])) * coefficient;
        }
    }
    __shared__ float partial[8][32];
    partial[warp][lane] = sum;
    __syncthreads();
    if (warp == 0) {
        float total = 0;
        for (int i = 0; i < 8; ++i) total += partial[i][lane];
        output[blockIdx.x * 32 + lane] = total;
    }
}
}

void igc_head_weight_vjp(const void* weights, const float* delta, float* result,
                         int width, int rows, void* stream) {
    if (!weights || !delta || !result || !stream || width <= 0 || width > 2560 ||
            width % 32 || rows <= 0 || rows > 248320)
        throw std::invalid_argument("head_vjp_geometry");
    transpose_product<<<width / 32, 256, 0, static_cast<cudaStream_t>(stream)>>>(
        static_cast<const Q80Block*>(weights), delta, result, width / 32, rows);
    const auto status = cudaPeekAtLastError();
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
