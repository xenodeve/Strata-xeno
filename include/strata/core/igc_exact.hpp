#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include <type_traits>

namespace strata::igc {
constexpr uint32_t kExactAbi = 1, kIntegerDomain = 1, kMaxBatch = 64;
constexpr int kMaxDigits = 62;
enum class Operation : uint32_t { Add = 1, Subtract = 2, Multiply = 3 };
enum class Status : uint32_t { Success = 0, Malformed = 1, Unsupported = 2, Overflow = 3, Fault = 4 };

// Canonical signed magnitude: little-endian decimal digits 0..9, length 1..62.
// Zero is {length=1,negative=0,digits=0}; unused digits must be zero.
struct Integer {
    uint8_t length = 0, negative = 0;
    uint8_t digits[kMaxDigits]{};
};
struct ExactRequest {
    uint32_t version = kExactAbi, domain = kIntegerDomain;
    Operation operation = Operation::Add;
    uint32_t reserved = 0;
    uint64_t invocation = 0, position = 0;
    Integer lhs{}, rhs{};
};
struct ExactResult {
    uint32_t version = kExactAbi;
    Status status = Status::Malformed;
    uint64_t invocation = 0, position = 0;
    Integer value{}; // length=0 on every failure: never an unverified zero answer
};
static_assert(sizeof(Integer) == 64 && sizeof(ExactRequest) == 160 && sizeof(ExactResult) == 88);
static_assert(offsetof(ExactRequest, lhs) == 32 && offsetof(ExactResult, value) == 24);
static_assert(std::is_trivially_copyable_v<ExactRequest> && std::is_trivially_copyable_v<ExactResult>);

// Internal engine API: input/output are caller-owned device arrays on the current
// device and the caller's explicit created stream. No allocation, host arithmetic,
// token/string parsing or synchronization. The caller fences before consuming data.
// Each request has a compile-time bounded amount of work, independent of its data.
bool execute_exact(const ExactRequest* input, ExactResult* output, uint32_t count,
                   void* stream, std::string& error);
// Copy one already completed result at output[0] to rows1..count-1 on the same
// explicit stream. Row0 is never written, so there is no read/write alias race.
bool broadcast_exact_result(ExactResult* output,uint32_t count,void* stream,std::string& error);
} // namespace strata::igc
