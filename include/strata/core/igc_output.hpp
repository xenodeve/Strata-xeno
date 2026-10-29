#pragma once
#include "strata/core/igc_adapter.hpp"
#include "strata/core/igc_exact.hpp"
#include <cstddef>
#include <cstdint>
#include <string>

namespace strata::igc {
// Diagnostic native forward only. A quality-ineligible artifact cannot enable
// serving merely by passing this numerical probe. One workspace per stream.
class OutputMapper {
public:
    OutputMapper()=default;
    ~OutputMapper();
    OutputMapper(const OutputMapper&)=delete;
    OutputMapper& operator=(const OutputMapper&)=delete;
    bool load(const adapter::Artifact&,void* created_stream,std::string&);
    // mixed is updated transactionally per row. Non-success status leaves its
    // original bytes untouched. No allocation, synchronization or host math.
    // Status:0 applied,1 upstream failure,2 malformed/unowned,3 nonfinite.
    bool run(float* mixed,const ExactResult*,const uint64_t* invocation,
             const uint64_t* anchor,uint32_t* status,uint32_t count,
             void* created_stream,bool enabled,std::string&);
    std::size_t device_bytes() const;
private:
    struct Impl;
    Impl* impl_=nullptr;
};
} // namespace strata::igc
