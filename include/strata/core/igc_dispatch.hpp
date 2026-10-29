#pragma once
#include "strata/core/igc_exact.hpp"
namespace strata::igc {
// Internal device arrays, not a serialized pointer-bearing user ABI.
struct SelectionInputs {
    const float *primary=nullptr,*secondary=nullptr,*readiness=nullptr;
    const uint32_t *primary_status=nullptr,*secondary_status=nullptr,*readiness_status=nullptr;
    const uint64_t *invocation=nullptr,*anchor=nullptr;
};
// Status0 ready,1 no call,2 upstream/owner,3 nonfinite,4 disagreement,5 canonical.
// Every enabled call overwrites inactive requests too, preventing stale execution.
bool assemble_selective_exact(const SelectionInputs&,ExactRequest*,uint32_t* status,
                              uint32_t count,void* created_stream,bool enabled,std::string&);
}
