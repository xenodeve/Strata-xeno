// H2 finding: the verify window copied every slot row of the CPU's mapped output over the x4 link, then added the
// GPU hits on top. `moe_hit_merge_mapped` reads only the rows that are not GPU hits and must give the SAME BYTES
// as `copy_from_mapped` + `moe_hit_add` (the ADR 0001 bit-exact gate), including -0.0 hit values.
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/s2_expert_grouped.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

int main() {
    const int rows = 40, n = 2560;
    const size_t bytes = (size_t) rows * n * sizeof(float);
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> u(-3.0f, 3.0f);
    // hit rows: the CPU pool writes +0 there (expert_source.cpp "the row must be written")
    const std::vector<int32_t> dst = {3, 0, 17, 39, 22, 8, 31};
    std::vector<char> hit_row(rows, 0);
    for (int32_t d : dst) hit_row[(size_t) d] = 1;

    float* ymiss = nullptr;
    if (cudaHostAlloc((void**) &ymiss, bytes, cudaHostAllocMapped) != cudaSuccess) return 1;
    std::vector<float> hit(rows * (size_t) n);
    for (int r = 0; r < rows; ++r)
        for (int i = 0; i < n; ++i) {
            ymiss[(size_t) r * n + i] = hit_row[(size_t) r] ? 0.0f : u(rng);
            hit[(size_t) r * n + i] = hit_row[(size_t) r] ? (i % 97 == 0 ? -0.0f : u(rng)) : 12345.0f;
        }
    float* m_ymiss = nullptr;
    if (cudaHostGetDevicePointer((void**) &m_ymiss, ymiss, 0) != cudaSuccess) return 1;
    float *d_hit, *ref, *got;
    int32_t *d_dst, *d_count;
    cudaMalloc(&d_hit, bytes); cudaMalloc(&ref, bytes); cudaMalloc(&got, bytes);
    cudaMalloc(&d_dst, 64 * sizeof(int32_t)); cudaMalloc(&d_count, sizeof(int32_t));
    const int32_t count = (int32_t) dst.size();
    cudaMemcpy(d_hit, hit.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_dst, dst.data(), dst.size() * sizeof(int32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_count, &count, sizeof(int32_t), cudaMemcpyHostToDevice);
    cudaMemset(got, 0xff, bytes);

    strata::kernels::copy_from_mapped(ref, m_ymiss, (int64_t) rows * n, nullptr);
    strata::kernels::moe_hit_add(ref, d_hit, d_dst, d_count, 64, n, nullptr);
    strata::kernels::moe_hit_merge_mapped(got, m_ymiss, d_hit, d_dst, d_count, rows, n, nullptr);
    if (cudaDeviceSynchronize() != cudaSuccess) { std::fprintf(stderr, "cuda error\n"); return 1; }

    std::vector<float> a(rows * (size_t) n), b(rows * (size_t) n);
    cudaMemcpy(a.data(), ref, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(b.data(), got, bytes, cudaMemcpyDeviceToHost);
    if (std::memcmp(a.data(), b.data(), bytes) != 0) {
        for (size_t i = 0; i < a.size(); ++i)
            if (std::memcmp(&a[i], &b[i], 4) != 0) {
                std::fprintf(stderr, "row %zu col %zu: ref %a got %a\n", i / n, i % n, a[i], b[i]);
                break;
            }
        return 1;
    }
    std::printf("hit_merge_mapped: %d rows bit-exact (%d hits)\n", rows, count);
    return 0;
}
