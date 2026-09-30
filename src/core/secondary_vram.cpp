#include "strata/core/secondary_vram.hpp"

#if defined(STRATA_HAS_NVML)
#include <cuda_runtime.h>
#include <nvml.h>
#endif

namespace strata::core {

bool secondary_nvml_free_bytes(int cuda_ordinal, uint64_t& free_bytes, std::string& err) {
    free_bytes = 0;
#if defined(STRATA_HAS_NVML)
    char pci_bus_id[32]{};
    const cudaError_t cuda_err = cudaDeviceGetPCIBusId(pci_bus_id, sizeof(pci_bus_id), cuda_ordinal);
    if (cuda_err != cudaSuccess) {
        err = std::string("cudaDeviceGetPCIBusId: ") + cudaGetErrorString(cuda_err);
        return false;
    }
    const nvmlReturn_t init = nvmlInit_v2();
    if (init != NVML_SUCCESS) {
        err = std::string("nvmlInit_v2: ") + nvmlErrorString(init);
        return false;
    }
    nvmlDevice_t device = nullptr;
    nvmlReturn_t result = nvmlDeviceGetHandleByPciBusId_v2(pci_bus_id, &device);
    nvmlMemory_t memory{};
    if (result == NVML_SUCCESS) result = nvmlDeviceGetMemoryInfo(device, &memory);
    const nvmlReturn_t shutdown = nvmlShutdown();
    if (result != NVML_SUCCESS) {
        err = std::string("NVML memory query for PCI ") + pci_bus_id + ": " + nvmlErrorString(result);
        return false;
    }
    if (shutdown != NVML_SUCCESS) {
        err = std::string("nvmlShutdown: ") + nvmlErrorString(shutdown);
        return false;
    }
    free_bytes = memory.free;
    return true;
#else
    (void) cuda_ordinal;
    err = "NVML is unavailable; secondary placement is disabled";
    return false;
#endif
}

} // namespace strata::core
