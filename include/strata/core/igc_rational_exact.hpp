#pragma once
#include "strata/core/igc_integer_extended.hpp"

namespace strata::igc::rational_exact {
constexpr uint32_t kAbi = 3, kRationalDomain = 2, kDecimalDomain = 3;
using Status = integer_extended::Status;
enum class Operation : uint32_t { Add=1, Subtract=2, Multiply=3, Divide=4 };
struct Operand {
    Integer numerator{}, denominator{};
    uint32_t scale=0, reserved=0;
};
struct Rational { Integer numerator{}, denominator{}; };
struct Request {
    uint32_t version=kAbi, domain=kRationalDomain;
    Operation operation=Operation::Add;
    uint32_t reserved=0;
    uint64_t invocation=0, position=0;
    Operand lhs{}, rhs{};
};
struct Result {
    uint32_t version=kAbi;
    Status status=Status::Malformed;
    uint64_t invocation=0, position=0;
    Rational value{}; // Both lengths remain zero on every failure.
};
static_assert(sizeof(Operand)==136 && sizeof(Request)==304 && sizeof(Result)==152);
static_assert(offsetof(Request,lhs)==32 && offsetof(Request,rhs)==168 && offsetof(Result,value)==24);
static_assert(std::is_trivially_copyable_v<Request> && std::is_trivially_copyable_v<Result>);
// Disjoint caller-owned device arrays; explicit created stream; count1..64.
// Rational: signed numerator/denominator, scale0. Decimal: denominator+1,
// coefficient*10^-scale (0..61), Add/Subtract/Multiply only. Exact reduced
// rational output, not rounded decimal text. No host arithmetic or explicit
// device allocation/sync. Error strings may allocate on the host.
bool execute(const Request*,Result*,uint32_t count,void* stream,std::string& error);
}
