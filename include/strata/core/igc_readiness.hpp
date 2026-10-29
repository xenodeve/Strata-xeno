#pragma once
#include "strata/core/igc_adapter.hpp"
#include <cstddef>
#include <cstdint>
#include <string>
namespace strata::igc {
// Diagnostic kind3 binary readout of actual final-HC mixed activations.
// Buffers are caller-owned and nonoverlapping. Status0 is valid NN forward,
// status2 nonfinite/overflow; it is never a guarantee of semantic correctness.
class ReadinessMapper {
public:
    ReadinessMapper()=default;
    ~ReadinessMapper();
    ReadinessMapper(const ReadinessMapper&)=delete;
    ReadinessMapper& operator=(const ReadinessMapper&)=delete;
    bool load(const adapter::Artifact&,void* created_stream,uint32_t capacity,std::string&);
    bool run(const float* mixed,float* logits,uint32_t* status,uint32_t count,
             void* created_stream,bool enabled,std::string&);
    std::size_t device_bytes()const;
private:
    struct Impl;Impl* impl_=nullptr;
};
}
