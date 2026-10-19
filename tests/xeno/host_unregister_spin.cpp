// #228 mechanism check: does cudaHostUnregister of memory registered for device 1 (not portable) return while device 0
// runs a kernel that spins on a mapped host flag (the verify window's wait_flag_ge)?  If it waits for device 0, an
// unregister made while the decode window waits on the host deadlocks with that host.
//
//     xeno_host_unregister_spin   (CUDA_VISIBLE_DEVICES=1,0: device 0 = 5060 Ti, device 1 = 4070 SUPER)
#include "strata/kernels/verify_kernels.hpp"

#include <cuda_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>

static void ck(cudaError_t e, const char* at) {
    if (e != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", at, cudaGetErrorString(e)); std::exit(2); }
}

int main() {
    uint32_t* flag = nullptr;
    ck(cudaSetDevice(0), "device 0");
    ck(cudaHostAlloc((void**) &flag, sizeof(uint32_t), cudaHostAllocMapped | cudaHostAllocPortable), "flag");
    *flag = 0;
    cudaStream_t s0 = nullptr;
    ck(cudaStreamCreateWithFlags(&s0, cudaStreamNonBlocking), "stream");
    uint32_t* dflag = nullptr;
    ck(cudaHostGetDevicePointer((void**) &dflag, flag, 0), "device pointer");
    std::vector<uint8_t> buf(8u << 20);
    for (auto& b : buf) b = 1;
    const bool portable = std::getenv("PORTABLE") != nullptr;
    int fails = 0;
    for (const char* op : {"register", "unregister"}) {
        // register first (outside the spin) when testing the unregister
        if (op[0] == 'u') {
            ck(cudaSetDevice(1), "device 1");
            ck(cudaHostRegister(buf.data(), buf.size(), portable ? cudaHostRegisterPortable : cudaHostRegisterDefault),
               "register");
        }
        *(volatile uint32_t*) flag = 0;
        ck(cudaSetDevice(0), "device 0");
        strata::kernels::wait_flag_ge(dflag, 1, s0);
        (void) cudaStreamQuery(s0);   // WDDM: hand it to the GPU now
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        double took = -1;
        std::thread t([&] {
            cudaSetDevice(1);
            const auto t0 = std::chrono::steady_clock::now();
            const cudaError_t e = op[0] == 'r'
                ? cudaHostRegister(buf.data(), buf.size(), portable ? cudaHostRegisterPortable : cudaHostRegisterDefault)
                : cudaHostUnregister(buf.data());
            took = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            if (e != cudaSuccess) std::fprintf(stderr, "%s: %s\n", op, cudaGetErrorString(e));
        });
        const auto t0 = std::chrono::steady_clock::now();
        while (took < 0 && std::chrono::steady_clock::now() - t0 < std::chrono::seconds(3))
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        const bool blocked = took < 0;
        *(volatile uint32_t*) flag = 1;   // release the spin
        t.join();
        ck(cudaStreamSynchronize(s0), "spin drained");
        std::printf("%s (%s) while device 0 spins on a host flag: %s (%.1f ms)\n", op,
                    portable ? "portable" : "device 1 only", blocked ? "BLOCKED until the spin ended" : "returned",
                    took);
        fails += blocked;
        if (op[0] == 'r') ck(cudaHostUnregister(buf.data()), "unregister after register test");
    }
    return fails ? 1 : 0;
}
