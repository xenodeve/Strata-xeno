// #22 probe (not a test): do scattered expert-slot copies cost less as one cudaMemcpyBatchAsync than as one
// cudaMemcpyAsync each?  A swap round moves a handful of 1,382,400 B blobs from pinned host memory into slots spread
// over the expert cache, so the question is the host's issue time (WDDM: 5-50 us per call) as much as the link's.
//
// For each card and batch size (1 / 2 / 4 / 8 / 16 / 32 blobs), 20 rounds of each form into slots scattered across a
// 512-slot device buffer: the host time to issue the round, and the device time from the first copy to the last.
// CUDA_VISIBLE_DEVICES=1,0: device 0 is the 5060 Ti (x4), device 1 the 4070 SUPER (x16).
#include <cuda_runtime.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CK(x)                                                                                       \
    do {                                                                                            \
        cudaError_t e_ = (x);                                                                       \
        if (e_ != cudaSuccess) {                                                                    \
            std::printf("%s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_));          \
            std::exit(1);                                                                           \
        }                                                                                           \
    } while (0)

int main() {
    constexpr size_t kBlob = 1382400;
    constexpr int kSlots = 512, kMax = 32, kRounds = 20;
    int n_dev = 0;
    CK(cudaGetDeviceCount(&n_dev));
    for (int dev = 0; dev < n_dev; ++dev) {
        CK(cudaSetDevice(dev));
        cudaDeviceProp pr{};
        CK(cudaGetDeviceProperties(&pr, dev));
        uint8_t *d = nullptr, *h = nullptr;
        if (cudaMalloc(&d, (size_t) kSlots * kBlob) != cudaSuccess) {   // 708 MB: the tier's scale, not all of it
            std::printf("device %d: cannot allocate the slots\n", dev);
            cudaGetLastError();
            continue;
        }
        CK(cudaHostAlloc(&h, (size_t) kMax * kBlob, cudaHostAllocPortable));
        std::memset(h, 1, (size_t) kMax * kBlob);
        cudaStream_t s;
        CK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
        cudaEvent_t e0, e1;
        CK(cudaEventCreate(&e0));
        CK(cudaEventCreate(&e1));
        std::printf("device %d (%s)\n", dev, pr.name);
        std::printf("  blobs   per-call: host us  device ms  GB/s   |  batch: host us  device ms  GB/s\n");
        for (int n : {1, 2, 4, 8, 16, 32}) {
            double host[2] = {0, 0}, dms[2] = {0, 0};
            for (int form = 0; form < 2; ++form) {
                for (int r = 0; r < kRounds; ++r) {
                    std::vector<void*> dst(n);
                    std::vector<const void*> src(n);
                    std::vector<size_t> sz(n, kBlob);
                    for (int i = 0; i < n; ++i) {   // scattered slots, a different set each round
                        dst[i] = d + (size_t) ((i * 97 + r * 31) % kSlots) * kBlob;
                        src[i] = h + (size_t) i * kBlob;
                    }
                    CK(cudaStreamSynchronize(s));
                    CK(cudaEventRecord(e0, s));
                    const auto t0 = std::chrono::steady_clock::now();
                    if (form == 0) {
                        for (int i = 0; i < n; ++i) CK(cudaMemcpyAsync(dst[i], src[i], kBlob, cudaMemcpyHostToDevice, s));
                    } else {
                        cudaMemcpyAttributes at{};
                        at.srcAccessOrder = cudaMemcpySrcAccessOrderStream;
                        size_t idx = 0;
                        CK(cudaMemcpyBatchAsync(dst.data(), src.data(), sz.data(), (size_t) n, &at, &idx, 1, s));
                    }
                    host[form] += std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
                    CK(cudaEventRecord(e1, s));
                    CK(cudaEventSynchronize(e1));
                    float ms = 0;
                    CK(cudaEventElapsedTime(&ms, e0, e1));
                    if (r > 0) dms[form] += ms;   // the first round warms the path
                    else host[form] = 0;
                }
            }
            const double gb = (double) n * kBlob * (kRounds - 1) / 1e6;
            std::printf("  %5d   %14.1f  %9.3f  %5.2f   |  %13.1f  %9.3f  %5.2f\n", n, host[0] / (kRounds - 1),
                        dms[0] / (kRounds - 1), gb / dms[0], host[1] / (kRounds - 1), dms[1] / (kRounds - 1),
                        gb / dms[1]);
        }
        CK(cudaEventDestroy(e0));
        CK(cudaEventDestroy(e1));
        CK(cudaStreamDestroy(s));
        CK(cudaFree(d));
        CK(cudaFreeHost(h));
    }
    return 0;
}
