#pragma once
// Finds `A op B =` at the end of generated tokens, for exact execution inside reasoning.
// Token ids are the Gyro-M pack's; each decimal digit is one token. Host parsing only:
// values always come from the device executors.
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>
#include "strata/core/igc_exact.hpp"
#include "strata/core/igc_rational_exact.hpp"

namespace strata::igc {
enum class InlineOperation : uint32_t { Add = 1, Subtract = 2, Multiply = 3, Divide = 4 };
struct InlineNumber {
    Integer coefficient{};
    uint32_t scale = 0;  // value = coefficient * 10^-scale
};
struct InlineCall {
    std::vector<InlineNumber> numbers;  // left to right; one step per operation
    std::vector<InlineOperation> operations;
    bool spaced = false;                // ` =` keeps a space before the value
    bool exact_integers() const {
        for (const auto op : operations) if (op == InlineOperation::Divide) return false;
        for (const auto& n : numbers) if (n.scale) return false;
        return true;
    }
};

namespace inline_tokens {
inline constexpr int32_t kZero = 15, kNine = 24, kSpace = 220, kComma = 11, kDot = 13, kMinus = 12, kSpacedMinus = 471;
inline constexpr int32_t kEquals = 28, kSpacedEquals = 283, kOpen = 7, kClose = 8;
inline bool digit(int32_t t) { return t >= kZero && t <= kNine; }
inline std::optional<InlineOperation> operation(int32_t t) {
    switch (t) {
        case 10: case 478: return InlineOperation::Add;
        case kMinus: case kSpacedMinus: return InlineOperation::Subtract;
        case 9: case 348: case 17044: case 23985: return InlineOperation::Multiply;
        case 14: case 593: case 169814: return InlineOperation::Divide;
        default: return std::nullopt;
    }
}
}

// Digits are read right to left; the executor's limit decides the operand bound.
inline bool read_number(std::span<const int32_t> tokens, std::ptrdiff_t& at, InlineNumber& value) {
    using namespace inline_tokens;
    while (at >= 0 && tokens[(size_t) at] == kSpace) --at;
    std::vector<uint8_t> digits;
    uint32_t scale = 0;
    bool point = false;
    while (at >= 0) {
        const int32_t t = tokens[(size_t) at];
        if (digit(t)) digits.push_back((uint8_t) (t - kZero));
        else if (t == kComma && !digits.empty() && !point) { /* thousands */ }
        else if (t == kDot && !digits.empty() && !point && at > 0 && digit(tokens[(size_t) at - 1])) {
            point = true;
            scale = (uint32_t) digits.size();
        } else break;
        --at;
    }
    if (at + 1 < (std::ptrdiff_t) tokens.size() && tokens[(size_t) (at + 1)] == kComma) return false;
    while (digits.size() > 1 && digits.back() == 0) digits.pop_back();
    if (digits.empty() || digits.size() > kMaxDigits / 2 || scale >= digits.size() + 30) return false;
    value = InlineNumber{};
    value.scale = scale;
    value.coefficient.length = (uint8_t) digits.size();
    for (size_t i = 0; i < digits.size(); ++i) value.coefficient.digits[i] = digits[i];
    return true;
}

inline std::optional<InlineCall> parse_inline(std::span<const int32_t> tokens) {
    using namespace inline_tokens;
    if (tokens.empty() || (tokens.back() != kEquals && tokens.back() != kSpacedEquals)) return std::nullopt;
    InlineCall call;
    call.spaced = tokens.back() == kSpacedEquals;
    std::ptrdiff_t at = (std::ptrdiff_t) tokens.size() - 2;
    for (;;) {
        InlineNumber number;
        if (!read_number(tokens, at, number)) return std::nullopt;
        call.numbers.insert(call.numbers.begin(), number);
        const std::ptrdiff_t end = at;
        while (at >= 0 && tokens[(size_t) at] == kSpace) --at;
        auto op = at >= 0 ? operation(tokens[(size_t) at]) : std::nullopt;
        if (!op && at > 0 && tokens[(size_t) at] == 115 && tokens[(size_t) at - 1] == 1624) {  // ` ÷`
            op = InlineOperation::Divide;
            --at;
        }
        if (!op) {
            at = end;
            break;
        }
        if (call.operations.size() == 15) return std::nullopt;
        call.operations.insert(call.operations.begin(), *op);
        --at;
    }
    if (call.operations.empty()) return std::nullopt;
    // Precedence and signs stay the model's: mixed precedence, parentheses and a leading sign are not calls.
    const auto multiplicative = [](InlineOperation op) { return op == InlineOperation::Multiply || op == InlineOperation::Divide; };
    for (const InlineOperation op : call.operations)
        if (multiplicative(op) != multiplicative(call.operations[0])) return std::nullopt;
    if (at >= 0) {
        const int32_t before = tokens[(size_t) at];
        if (before == kDot || before == kOpen || before == kClose) return std::nullopt;
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

// Presentation only: long division of the executor's reduced fraction into a terminating
// decimal. A value with no exact short decimal stays the model's, nothing is rounded.
inline std::optional<std::vector<int32_t>> rational_tokens(const rational_exact::Rational& value, bool spaced) {
    using namespace inline_tokens;
    if (!value.denominator.length || value.denominator.length > 17) return std::nullopt;
    uint64_t denominator = 0;
    for (int i = value.denominator.length - 1; i >= 0; --i) denominator = denominator * 10 + value.denominator.digits[i];
    if (!denominator) return std::nullopt;
    std::vector<int32_t> out;
    if (value.numerator.negative) out.push_back(spaced ? kSpacedMinus : kMinus);
    else if (spaced) out.push_back(kSpace);
    uint64_t remainder = 0;
    bool started = false;
    for (int i = value.numerator.length - 1; i >= 0; --i) {
        remainder = remainder * 10 + value.numerator.digits[i];
        const uint64_t q = remainder / denominator;
        remainder %= denominator;
        if (q || started) {
            out.push_back(kZero + (int32_t) q);
            started = true;
        }
    }
    if (!started) out.push_back(kZero);
    if (remainder) out.push_back(kDot);
    for (int places = 0; remainder; ++places) {
        if (places == 12) return std::nullopt;
        remainder *= 10;
        out.push_back(kZero + (int32_t) (remainder / denominator));
        remainder %= denominator;
    }
    return out;
}
}
