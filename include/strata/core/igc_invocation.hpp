#pragma once
#include "strata/core/igc_exact.hpp"

namespace strata::igc {
// Internal device-owned request lifetime. Not a user ABI or a conversation
// checkpoint: caller fences this stream before resetting/releasing the storage.
enum class InvocationStatus : uint32_t { Pending=0, Applied=1, NoCall=2, Rejected=3, OwnerMismatch=4 };
struct Invocation {
    uint64_t id=0, position=0;
    uint32_t armed=0, attempts=0, evaluations=0;
    InvocationStatus terminal=InvocationStatus::Rejected;
    ExactResult result{};
};
static_assert(std::is_trivially_copyable_v<Invocation>);
// Start a fresh owned attempt, clearing any previous answer. One created stream,
// stable caller-owned device addresses, count1..64; no allocation or host sync.
bool arm_invocations(Invocation*,const uint64_t* ids,const uint64_t* positions,
                     uint32_t count,void* stream,bool enabled,std::string&);
// First enabled launch consumes an armed attempt even if selection fails. Later
// graph launches preserve the terminal state/result. Only selection_status==0
// and matching ownership can call the exact arithmetic kernel's calculation.
bool execute_invocations(Invocation*,const ExactRequest*,const uint32_t* selection_status,
                         uint32_t count,void* stream,bool enabled,std::string&);
// Publish the retained result to a fixed row batch for OutputMapper. This does
// not execute arithmetic or mutate the invocation. Failure results have no value.
bool broadcast_invocation(const Invocation*,ExactResult*,uint32_t rows,void* stream,
                          bool enabled,std::string&);
}
