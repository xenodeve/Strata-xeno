// #29: the prompt path gathers an MMQ group's experts with one gather_native_group launch instead of one
// gather_native launch per expert.  Both must write the same bytes into the group slots, for every group size
// 1..kGatherGroupMax, blobs in any order (ring slots and cache slots mix in a group), and must not touch the bytes
// past the last slot (the MMQ tail is zeroed separately).
#include "strata/prefill/moe_mmq.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

int main() {
    namespace mmq = strata::prefill::mmq;
    const size_t half = 460800, down = 460800, up_off = half, down_off = 2 * half, blob = 2 * half + down;
    const int pool = 40;
    std::mt19937 rng(29);
    std::vector<uint8_t> host(blob * pool);
    for (auto& b : host) b = (uint8_t) rng();
    uint8_t* d_pool = nullptr;
    cudaMalloc(&d_pool, host.size());
    cudaMemcpy(d_pool, host.data(), host.size(), cudaMemcpyHostToDevice);
    const size_t gu_stride = 2 * half, d_stride = down;
    const size_t gu_bytes = gu_stride * mmq::kGatherGroupMax + 4096, d_bytes = d_stride * mmq::kGatherGroupMax + 4096;
    uint8_t *gu_a = nullptr, *gu_b = nullptr, *d_a = nullptr, *d_b = nullptr;
    cudaMalloc(&gu_a, gu_bytes);
    cudaMalloc(&gu_b, gu_bytes);
    cudaMalloc(&d_a, d_bytes);
    cudaMalloc(&d_b, d_bytes);
    int fails = 0;
    for (int n = 1; n <= mmq::kGatherGroupMax; ++n) {
        std::vector<const uint8_t*> blobs((size_t) n);
        for (auto& p : blobs) p = d_pool + (size_t) (rng() % pool) * blob;
        cudaMemset(gu_a, 0x5A, gu_bytes);
        cudaMemset(gu_b, 0x5A, gu_bytes);
        cudaMemset(d_a, 0xA5, d_bytes);
        cudaMemset(d_b, 0xA5, d_bytes);
        for (int i = 0; i < n; ++i)
            mmq::gather_native(blobs[(size_t) i], blobs[(size_t) i] + up_off, half, blobs[(size_t) i] + down_off, down,
                               gu_a + (size_t) i * gu_stride, d_a + (size_t) i * d_stride, nullptr);
        mmq::gather_native_group(blobs.data(), n, up_off, down_off, half, down, gu_b, gu_stride, d_b, d_stride, nullptr);
        std::vector<uint8_t> ha(gu_bytes), hb(gu_bytes), da(d_bytes), db(d_bytes);
        cudaMemcpy(ha.data(), gu_a, gu_bytes, cudaMemcpyDeviceToHost);
        cudaMemcpy(hb.data(), gu_b, gu_bytes, cudaMemcpyDeviceToHost);
        cudaMemcpy(da.data(), d_a, d_bytes, cudaMemcpyDeviceToHost);
        cudaMemcpy(db.data(), d_b, d_bytes, cudaMemcpyDeviceToHost);
        const bool same = ha == hb && da == db;
        const cudaError_t e = cudaGetLastError();
        if (!same || e != cudaSuccess) {
            size_t diff = 0;
            for (size_t i = 0; i < ha.size(); ++i) diff += ha[i] != hb[i];
            for (size_t i = 0; i < da.size(); ++i) diff += da[i] != db[i];
            std::printf("group of %2d: %zu bytes differ %s\n", n, diff, e != cudaSuccess ? cudaGetErrorString(e) : "");
            ++fails;
        }
    }
    std::printf(fails == 0 ? "PASS\n" : "FAIL\n");
    return fails == 0 ? 0 : 1;
}
