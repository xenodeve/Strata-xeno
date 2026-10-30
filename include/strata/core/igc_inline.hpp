#pragma once
// Finds `A op B =` at the end of generated tokens, for exact execution inside reasoning.
// Token ids are the Gyro-M pack's; each decimal digit is one token. Host parsing only:
// the value always comes from the device executor.
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>
#include "strata/core/igc_exact.hpp"

namespace strata::igc {
struct InlineCall {
    Operation operation = Operation::Add;
    Integer lhs{}, rhs{};
    bool spaced = false;  // ` =` keeps a space before the value
};

namespace inline_tokens {
inline constexpr int32_t kZero = 15, kNine = 24, kSpace = 220, kComma = 11, kDot = 13, kMinus = 12, kSpacedMinus = 471;
inline constexpr int32_t kEquals = 28, kSpacedEquals = 283, kOpen = 7, kClose = 8, kSlash = 14;
inline bool digit(int32_t t) { return t >= kZero && t <= kNine; }
inline std::optional<Operation> operation(int32_t t) {
    switch (t) {
        case 10: case 478: return Operation::Add;
        case kMinus: case kSpacedMinus: return Operation::Subtract;
        case 9: case 348: case 17044: case 23985: return Operation::Multiply;
        default: return std::nullopt;
    }
}
}

// Digits are read right to left; the executor's limit decides the operand bound.
inline bool read_number(std::span<const int32_t> tokens, std::ptrdiff_t& at, Integer& value) {
    using namespace inline_tokens;
    while (at >= 0 && tokens[(size_t) at] == kSpace) --at;
    std::vector<uint8_t> digits;
    while (at >= 0 && (digit(tokens[(size_t) at]) || tokens[(size_t) at] == kComma)) {
        if (tokens[(size_t) at] != kComma) digits.push_back((uint8_t) (tokens[(size_t) at] - kZero));
        else if (digits.empty()) return false;
        --at;
    }
    if (at + 1 >= 0 && at + 1 < (std::ptrdiff_t) tokens.size() && tokens[(size_t) (at + 1)] == kComma) return false;
    while (digits.size() > 1 && digits.back() == 0) digits.pop_back();
    if (digits.empty() || digits.size() > kMaxDigits / 2) return false;
    value = Integer{};
    value.length = (uint8_t) digits.size();
    for (size_t i = 0; i < digits.size(); ++i) value.digits[i] = digits[i];
    return true;
}

inline std::optional<InlineCall> parse_inline(std::span<const int32_t> tokens) {
    using namespace inline_tokens;
    if (tokens.empty() || (tokens.back() != kEquals && tokens.back() != kSpacedEquals)) return std::nullopt;
    InlineCall call;
    call.spaced = tokens.back() == kSpacedEquals;
    std::ptrdiff_t at = (std::ptrdiff_t) tokens.size() - 2;
    if (!read_number(tokens, at, call.rhs)) return std::nullopt;
    while (at >= 0 && tokens[(size_t) at] == kSpace) --at;
    if (at < 0) return std::nullopt;
    const auto op = operation(tokens[(size_t) at]);
    if (!op) return std::nullopt;
    call.operation = *op;
    --at;
    if (!read_number(tokens, at, call.lhs)) return std::nullopt;
    while (at >= 0 && tokens[(size_t) at] == kSpace) --at;
    // Precedence and signs stay the model's: chains, parentheses and decimals are not calls.
    if (at >= 0) {
        const int32_t before = tokens[(size_t) at];
        if (operation(before) || before == kDot || before == kOpen || before == kClose || before == kSlash) return std::nullopt;
    }
    return call;
}

// A learned operand counts only as a whole number written in the prompt, commas apart:
// a two-digit head reading 68824 as 68 must not reach the answer.
inline bool written_in(std::span<const int64_t> prompt, const Integer& value) {
    using namespace inline_tokens;
    std::vector<uint8_t> run;
    auto matches = [&] {
        if (run.size() != value.length) return false;
        for (size_t i = 0; i < run.size(); ++i)
            if (run[run.size() - 1 - i] != value.digits[i]) return false;
        return true;
    };
    for (const int64_t t : prompt) {
        if (digit((int32_t) t)) run.push_back((uint8_t) (t - kZero));
        else if (t != kComma || run.empty()) {
            if (matches()) return true;
            run.clear();
        }
    }
    return matches();
}

inline std::vector<int32_t> value_tokens(const Integer& value, bool spaced) {
    using namespace inline_tokens;
    std::vector<int32_t> out;
    if (value.negative) out.push_back(spaced ? kSpacedMinus : kMinus);
    else if (spaced) out.push_back(kSpace);
    for (int i = value.length - 1; i >= 0; --i) out.push_back(kZero + value.digits[i]);
    return out;
}
}
