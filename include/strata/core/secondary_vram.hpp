#pragma once

#include <cstdint>
#include <string>

namespace strata::core {

// Query physical framebuffer memory for the same GPU as a CUDA ordinal.
// PCI bus ID, not NVML index, binds the two ordinal spaces.
bool secondary_nvml_free_bytes(int cuda_ordinal, uint64_t& free_bytes, std::string& err);

} // namespace strata::core
