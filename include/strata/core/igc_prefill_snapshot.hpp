#pragma once
#include "strata/core/early_activations.hpp"
#include <cstddef>
#include <cstdint>
#include <string>

namespace strata::igc {
// Per-prefill-lane tail storage. The owner emits at the existing ordered wave
// tail, before this lane reuses its buffers. Not a CUDA graph host callback.
class PrefillSnapshot {
public:
    PrefillSnapshot()=default;
    ~PrefillSnapshot();
    PrefillSnapshot(const PrefillSnapshot&)=delete;
    PrefillSnapshot& operator=(const PrefillSnapshot&)=delete;
    bool allocate(void* created_stream,std::string&);
    bool stage(core::EarlyActivationTap,const float* source,int64_t pos0,int64_t count,void* created_stream,std::string&);
    bool emit(const core::EarlyActivationConsumer&,std::string&);
    std::size_t device_bytes()const;
private:
    struct Impl;Impl* impl_=nullptr;
};
}
