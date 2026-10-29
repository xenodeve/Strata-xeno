#pragma once
#include "strata/core/igc_exact.hpp"

namespace strata::igc::integer_extended {
// Separate experimental component ABI. No current learned controller routes
// these records; ABI1 entrypoints and artifacts retain their original meaning.
constexpr uint32_t kAbi=2;
enum class Operation : uint32_t { FloorDivide=1,Modulo=2,Gcd=3,Power=4 };
enum class Status : uint32_t { Success=0,Malformed=1,Unsupported=2,Overflow=3,Fault=4,DomainError=5 };
struct Request {
    uint32_t version=kAbi,domain=kIntegerDomain;
    Operation operation=Operation::FloorDivide;
    uint32_t reserved=0;
    uint64_t invocation=0,position=0;
    Integer lhs{},rhs{};
};
struct Result {
    uint32_t version=kAbi;Status status=Status::Malformed;
    uint64_t invocation=0,position=0;
    Integer value{};
};
static_assert(sizeof(Request)==160 && sizeof(Result)==88);
static_assert(std::is_standard_layout_v<Request> && std::is_trivially_copyable_v<Request> && std::is_trivially_copyable_v<Result>);
// Disjoint caller-owned device arrays, one explicit created stream, count1..64. Work is
// statically bounded; no host calculation, parsing, allocation or synchronization.
bool execute(const Request*,Result*,uint32_t count,void* stream,std::string&);
}
