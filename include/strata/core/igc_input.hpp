#pragma once
#include "strata/core/igc_adapter.hpp"
#include <cstddef>
#include <cstdint>
#include <string>

namespace strata::igc {
// Kind1 diagnostic neural forward. Caller owns right-aligned[batch,80,4,2560]
// native states, byte masks[batch,80] and output[batch,52]. No token IDs/parser.
class InputMapper {
public:
    InputMapper()=default;
    ~InputMapper();
    InputMapper(const InputMapper&)=delete;
    InputMapper& operator=(const InputMapper&)=delete;
    bool load(const adapter::Artifact&,void* created_stream,uint32_t capacity,std::string&);
    // Status0: neural forward valid;1: bad/empty mask;2: nonfinite/overflow.
    // Nonzero preserves original output bytes;0 is not a quality/eligibility gate.
    bool run(const float* states,const uint8_t* valid,float* logits,uint32_t* status,
             uint32_t count,void* created_stream,bool enabled,std::string&);
    std::size_t device_bytes()const;
private:
    struct Impl;
    Impl* impl_=nullptr;
};
}
