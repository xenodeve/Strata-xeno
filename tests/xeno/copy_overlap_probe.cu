// #31 probe (not a test): does an H2D copy from pinned memory progress while a long compute kernel runs on another
// stream of the same device?  The prompt path's timeline (tl3, docs/reports/2026-09-29-pipeline-timeline.md) shows
// copies in flight stretching to 10-50 ms while `gdn` / `qsa attn` run, against 0.2 ms otherwise.
//
// For each kernel kind (a clock spin that touches no memory, a VRAM-streaming kernel, and a kernel whose blocks
// fill every SM for the whole duration), the probe runs:
//   alone:    32 copies of one expert blob (1.38 MB) on the copy stream, nothing else on the device;
//   during:   the same copies, issued right after a ~100 ms kernel is launched on the compute stream;
// and prints each copy's device time (events on the copy stream) and how many copies finished before the kernel.
// CUDA_VISIBLE_DEVICES=1 runs it on the 5060 Ti (x4).
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CK(x)                                                                                       \
    do {                                                                                            \
        cudaError_t e_ = (x);                                                                       \
        if (e_ != cudaSuccess) {                                                                    \
            std::printf("%s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_));          \
            std::exit(1);                                                                           \
        }                                                                                           \
    } while (0)

__global__ void spin_kernel(long long cycles, int* sink) {
    const long long t0 = clock64();
    while (clock64() - t0 < cycles) {}
    if (threadIdx.x == 0 && blockIdx.x == 0) *sink = 1;
}

__global__ void stream_kernel(const float4* src, float4* dst, size_t n, int reps) {
    for (int r = 0; r < reps; ++r)
        for (size_t i = blockIdx.x * (size_t) blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x) {
            float4 v = src[i];
            v.x += 1.0f;
            dst[i] = v;
        }
}

int main() {
    CK(cudaSetDevice(0));
    cudaDeviceProp prop{};
    CK(cudaGetDeviceProperties(&prop, 0));
    int clock_khz = 0;
    CK(cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, 0));
    std::printf("device %s, %d SMs, clock %d kHz, async engines %d\n", prop.name, prop.multiProcessorCount, clock_khz,
                prop.asyncEngineCount);
    cudaStream_t cs, cp;
    CK(cudaStreamCreateWithFlags(&cs, cudaStreamNonBlocking));
    CK(cudaStreamCreateWithFlags(&cp, cudaStreamNonBlocking));
    const size_t blob = 1382400, ncopy = 32;
    uint8_t *host = nullptr, *dev = nullptr;
    CK(cudaHostAlloc((void**) &host, blob * ncopy, cudaHostAllocDefault));
    CK(cudaMalloc((void**) &dev, blob * ncopy));
    for (size_t i = 0; i < blob * ncopy; ++i) host[i] = (uint8_t) i;
    const size_t nf = (size_t) 256 << 20;   // 256 Mi floats (1 GiB) each way
    float4 *a = nullptr, *b = nullptr;
    CK(cudaMalloc((void**) &a, nf / 4 * sizeof(float4)));
    CK(cudaMalloc((void**) &b, nf / 4 * sizeof(float4)));
    int* sink = nullptr;
    CK(cudaMalloc((void**) &sink, 4));
    std::vector<cudaEvent_t> ev(2 * ncopy);
    for (auto& e : ev) CK(cudaEventCreate(&e));
    cudaEvent_t k0, k1;
    CK(cudaEventCreate(&k0));
    CK(cudaEventCreate(&k1));
    const long long cycles = (long long) clock_khz * 100;   // ~100 ms

    auto run = [&](int kind, bool during) {
        CK(cudaDeviceSynchronize());
        if (during) {
            CK(cudaEventRecord(k0, cs));
            if (kind == 0) spin_kernel<<<1, 32, 0, cs>>>(cycles, sink);
            else if (kind == 1) stream_kernel<<<prop.multiProcessorCount * 4, 256, 0, cs>>>(a, b, nf / 4, 12);
            else spin_kernel<<<prop.multiProcessorCount * 16, 1024, 0, cs>>>(cycles, sink);
            CK(cudaEventRecord(k1, cs));
            (void) cudaStreamQuery(cs);   // WDDM: submit the kernel now
        }
        for (size_t i = 0; i < ncopy; ++i) {
            CK(cudaEventRecord(ev[2 * i], cp));
            CK(cudaMemcpyAsync(dev + i * blob, host + i * blob, blob, cudaMemcpyHostToDevice, cp));
            CK(cudaEventRecord(ev[2 * i + 1], cp));
        }
        (void) cudaStreamQuery(cp);
        CK(cudaDeviceSynchronize());
        std::vector<float> d(ncopy);
        float span = 0, kms = 0;
        for (size_t i = 0; i < ncopy; ++i) CK(cudaEventElapsedTime(&d[i], ev[2 * i], ev[2 * i + 1]));
        CK(cudaEventElapsedTime(&span, ev[0], ev[2 * ncopy - 1]));
        int before_kernel_end = -1;
        if (during) {
            CK(cudaEventElapsedTime(&kms, k0, k1));
            before_kernel_end = 0;
            for (size_t i = 0; i < ncopy; ++i) {
                float t = 0;
                CK(cudaEventElapsedTime(&t, k1, ev[2 * i + 1]));
                before_kernel_end += t < 0;
            }
        }
        std::vector<float> s = d;
        std::sort(s.begin(), s.end());
        const char* kn[] = {"spin 1 block", "VRAM stream", "spin all SMs"};
        std::printf("%-13s %-6s copies: p50 %.3f ms, max %.3f ms, all %zu in %.2f ms (%.2f GB/s)", kn[kind],
                    during ? "during" : "alone", s[ncopy / 2], s.back(), ncopy, span,
                    blob * ncopy / (span * 1e-3) / 1e9);
        if (during) std::printf("; kernel %.1f ms, %d copies done before it ended", kms, before_kernel_end);
        std::printf("\n");
    };
    for (int rep = 0; rep < 2; ++rep)
        for (int kind = 0; kind < 3; ++kind) {
            run(kind, false);
            run(kind, true);
        }
    return 0;
}
