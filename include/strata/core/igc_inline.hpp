#pragma once
// Finds `A op B =` at the end of generated tokens, for exact execution inside reasoning.
// Token ids are the Gyro-M pack's; each decimal digit is one token. Host parsing only:
// values always come from the device executors.
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <optional>
#include <span>
#include <unordered_map>
#include <vector>
#include "strata/core/igc_binding_tokens.hpp"
#include "strata/core/igc_exact.hpp"
#include "strata/core/igc_integer_extended.hpp"
#include "strata/core/igc_rational_exact.hpp"

namespace strata::igc {
enum class InlineOperation : uint32_t { Add = 1, Subtract = 2, Multiply = 3, Divide = 4, Power = 5 };
struct InlineNumber {
    Integer coefficient{};
    uint32_t scale = 0;  // value = coefficient * 10^-scale
    bool percent = false;
};
struct InlineStep {
    bool number = false;
    InlineOperation operation = InlineOperation::Add;
    InlineNumber value{};
};
struct InlineCall {
    std::vector<InlineStep> program;  // postfix, in the expression's own precedence
    bool spaced = false;              // ` =` keeps a space before the value
    bool exact_integers() const {
        for (const auto& step : program)
            if (step.number ? step.value.scale != 0 : step.operation == InlineOperation::Divide) return false;
        return true;
    }
};

namespace inline_tokens {
inline constexpr int32_t kZero = 15, kNine = 24, kSpace = 220, kComma = 11, kDot = 13, kMinus = 12, kSpacedMinus = 471;
inline constexpr int32_t kEquals = 28, kSpacedEquals = 283, kOpen = 7, kClose = 8;
inline constexpr int32_t kPercent = 4, kOf = 314, kDollar = 3, kSpacedDollar = 393, kTimesDollar = 37726;
inline bool digit(int32_t t) { return t >= kZero && t <= kNine; }
inline std::optional<InlineOperation> operation(int32_t t) {
    switch (t) {
        case 10: case 478: return InlineOperation::Add;
        case kMinus: case kSpacedMinus: case 32160: case 23437: return InlineOperation::Subtract;
        case 9: case 348: case 17044: case 23985: case kTimesDollar: case 13535: case 9306: return InlineOperation::Multiply;
        case 14: case 593: case 169814: return InlineOperation::Divide;
        case 61: case 6108: return InlineOperation::Power;
        default: return std::nullopt;
    }
}
// `\times`, `\cdot`, `\div` are two tokens: the backslash, then the name.
inline std::optional<InlineOperation> latex_operation(int32_t slash, int32_t name) {
    if (slash == 4795) return name == 1672 ? std::optional(InlineOperation::Multiply) : std::nullopt;
    if (slash != 1088 && slash != 59) return std::nullopt;
    if (name == 14695 || name == 49142) return InlineOperation::Multiply;
    if (name == 596) return InlineOperation::Divide;
    return std::nullopt;
}
// The tokenizer's one-token brackets, as the tokens the walk and the parser read.
inline std::vector<int32_t> split_brackets(std::span<const int32_t> tokens) {
    std::vector<int32_t> out;
    out.reserve(tokens.size() + 8);
    auto put = [&](std::initializer_list<int32_t> parts) { out.insert(out.end(), parts); };
    for (const int32_t t : tokens) switch (t) {
        case 4653: put({kClose, 9}); break;                  // )*
        case 7030: put({kClose, 10}); break;                 // )+
        case 7060: put({kClose, kMinus}); break;             // )-
        case 5443: put({kClose, 14}); break;                 // )/
        case 28803: put({kClose, 61}); break;                // )^
        case 11393: put({kClose, kEquals}); break;           // )=
        case 6391: put({9, kOpen}); break;                   // *(
        case 12975: put({10, kOpen}); break;                 // +(
        case 7762: put({kMinus, kOpen}); break;              // -(
        case 11545: put({14, kOpen}); break;                 // /(
        case 12883: put({61, kOpen}); break;                 // ^(
        case 1148: put({kOpen, kOpen}); break;               // ((
        case 578: put({kClose, kClose}); break;              // ))
        case 7461: put({kClose, kClose, kClose}); break;     // )))
        case 30824: put({kClose, kClose, 9}); break;         // ))*
        case 17201: put({kClose, 9, kOpen}); break;          // )*(
        case 40415: put({kClose, 10, kOpen}); break;         // )+(
        case 49629: put({kClose, kMinus, kOpen}); break;     // )-(
        case 23493: put({kClose, 14, kOpen}); break;         // )/(
        case 80851: put({kClose, kClose, 9, kOpen}); break;  // ))*(
        case 4391: put({kEquals, kOpen}); break;             // =(
        case 42163: put({kSpacedEquals, kOpen}); break;      // ` =(`
        case 318: put({kSpace, kOpen}); break;               // ` (`
        case 12492: put({348, kOpen}); break;                // ` *(`
        case 59025: put({478, kOpen}); break;                // ` +(`
        case 27478: put({kSpacedMinus, kOpen}); break;       // ` -(`
        case 68477: put({593, kOpen}); break;                // ` /(`
        default: out.push_back(t);
    }
    return out;
}
inline bool closes_expression(int32_t t) { return t == kEquals || t == kSpacedEquals || t == 11393; }
// `<think>\n\n</think>\n\n` ends a request with the thinking off: what follows is the reply from its first token.
inline bool thinking_off(std::span<const int64_t> prompt, int64_t think_close) {
    return prompt.size() >= 2 && prompt[prompt.size() - 2] == think_close && prompt.back() == 271;
}
// The tail is one exact cycle. In 1410 measured runs no text that repeated six times ever left, 2263 rounds at most.
// At least 64 tokens: digits are one token each, so `2102500000000` and `0.33333333` are eight rounds of one token.
// At most 1024 tokens or three rounds: of 7073 runs none left a block of 300 characters after three rounds, and of
// 1472 runs long enough the model left no block of 1000 characters after three rounds by itself (#282).
inline bool in_cycle(std::span<const int32_t> tokens, int longest, int rounds) {
    for (int period = 1; period <= longest; ++period) {
        const std::size_t whole = static_cast<std::size_t>(period);
        const std::size_t span = std::max(std::clamp<std::size_t>(whole * rounds, 64, 1024), whole * 3);
        if (tokens.size() < span) return false;
        const auto tail = tokens.last(span);
        if (std::equal(tail.begin() + period, tail.end(), tail.begin())) return true;
    }
    return false;
}
// A loop past in_cycle's reach still says the same lines: one of about 1,000 lines ran to the end of the context, 261,496 tokens,
// with a line back 37 times. Of 7,437 stored runs, a line of 12 tokens came back 16 times in 3 that ended by themselves, one of them
// right and 2,535 tokens from its end, and in all 4 that ran to the end of the context, at 11,202 to 128,741 tokens.
inline constexpr int kLineTokens = 12, kLineRounds = 16;
// A run that counts by hand changes its numbers, so neither check sees it, and it feeds on these values: 8,104 of them in 261,266
// tokens. Of 2,471 stored runs that took a value and ended by themselves, one took more than 256, and it was wrong; a right one took 78.
inline constexpr int kMaxValues = 256;
class LineRepeat {
public:
    bool step(int32_t token) {
        if (token != 198 && token != 271) {
            hash_ = (hash_ ^ static_cast<uint32_t>(token)) * 0x100000001B3ull;
            ++length_;
            return false;
        }
        const bool counts = length_ >= kLineTokens;
        const uint64_t line = hash_;
        hash_ = 0xCBF29CE484222325ull;
        length_ = 0;
        return counts && ++seen_[line] >= kLineRounds;
    }
private:
    uint64_t hash_ = 0xCBF29CE484222325ull;
    int length_ = 0;
    std::unordered_map<uint64_t, int> seen_;
};
// The note on a value whose expression is a sum at its top level. Left open, the model takes the value for the
// first term and writes the last one again, `a + b = V (exact) + b`: 28 times in 297 runs, none in 1290 runs with a `.`.
enum class SumNote : int { Open = 0, Stop = 1, None = 2 };
// The note on a repeat. `stop checking` can end the thinking with work left: 2 of 6 runs of one sum stopped after its
// first term and answered -95 for 5, and 5 of the 10 runs in 294 that ended on the note had a step to go (#281).
enum class RepeatNote : int { Stop = 0, Bare = 1, Next = 2 };
// What follows a value delivered inside reasoning, when this expression was written `seen` times before.
// The `close_at`th time closes the thinking, 0 never. The close carries its own `.`, `notes` only the two marks.
inline std::vector<int32_t> after_value(int seen, bool notes, int close_at, int32_t think_close, SumNote sum = SumNote::Open, RepeatNote repeat = RepeatNote::Stop) {
    if (close_at > 0 && seen + 1 >= close_at) return {13, 198, think_close, 271};
    if (!notes || sum == SumNote::None) return {};
    const int32_t end = sum == SumNote::Stop ? 553 : 8;   // `).` or `)`
    if (!seen) return {318, 44839, end};                  // ` (exact)`
    if (repeat == RepeatNote::Bare) return {318, 43967, 10457, end};            // ` (already confirmed)`
    if (repeat == RepeatNote::Next) return {318, 43967, 10457, 11, 1727, 2923, end};   // ` (already confirmed, next step)`
    return {318, 43967, 10457, 11, 2842, 12910, end};     // ` (already confirmed, stop checking)`
}
// `(` written against a word is a function call or a product we cannot read.
inline bool opens_cleanly(int32_t before) {
    return before == kEquals || before == kSpacedEquals || before == 198 || before == 271 || before == 25;
}
}

// A word that scales the number after it: `half of 68 + 10` is 44 to a reader and 78 as a sum.
inline bool scales_next(std::span<const int32_t> tokens, std::ptrdiff_t at) {
    using namespace inline_tokens;
    static constexpr int32_t kThe = 279;
    static constexpr int32_t kWords[] = {4139, 35778, 25015, 41212, 4688, 30686, 20226, 35750, 89931,   // half, third, thirds
                                         8079, 30961, 33252, 83618, 30439, 19069, 17175, 22132,         // quarter, quarters, fifth
                                         24238, 64714, 29398, 72086, 35174, 93154, 41127, 82798, 53797, // sixth to tenth
                                         10598, 90345, 1923, 4190, 6872, 7144, 22983, 35436, 79968};    // twice, double, triple
    bool of = false;
    while (at > 0 && (tokens[(size_t) at] == kOf || tokens[(size_t) at] == kThe)) {
        of = of || tokens[(size_t) at] == kOf;
        --at;
    }
    const int32_t t = tokens[(size_t) at];
    return (of && digit(t)) || std::find(std::begin(kWords), std::end(kWords), t) != std::end(kWords);   // `1/4 of 20 + 5`
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
        else if (t == kComma && !digits.empty() && !point) {
            if (digits.size() % 3 != 0) return false;   // `[6,9+4=` is a list, not 69
        } else if (t == kDot && !digits.empty() && !point && at > 0 && digit(tokens[(size_t) at - 1])) {
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

// One step of the postfix program: a number, or an operation on the two values below it.
inline bool numeric_token(int32_t t) {
    using namespace inline_tokens;
    return digit(t) || t == kComma || t == kDot;
}

// Precedence climbing over the tokens of one expression, left to right.
struct InlineParser {
    std::span<const int32_t> tokens;
    size_t at = 0;
    InlineCall& call;
    bool latex = false;       // written as `\times`: shown working
    int product_terms = 0;    // terms of the outer sums that multiply or divide
    void skip() { while (at < tokens.size() && tokens[at] == inline_tokens::kSpace) ++at; }
    std::optional<InlineOperation> peek(size_t& width) {
        skip();
        width = 1;
        if (at >= tokens.size()) return std::nullopt;
        if (tokens[at] == inline_tokens::kOf && at > 0 && tokens[at - 1] == inline_tokens::kPercent)
            return InlineOperation::Multiply;  // `30% of 180`
        if (at + 1 < tokens.size() && tokens[at] == 1624 && tokens[at + 1] == 115) {  // ` ÷`
            width = 2;
            return InlineOperation::Divide;
        }
        if (at + 1 < tokens.size())
            if (const auto latex_op = inline_tokens::latex_operation(tokens[at], tokens[at + 1])) {
                width = 2;
                return latex_step(*latex_op);
            }
        return inline_tokens::operation(tokens[at]);
    }
    InlineOperation latex_step(InlineOperation op) {
        latex = true;
        return op;
    }
    bool primary() {
        using namespace inline_tokens;
        skip();
        if (at >= tokens.size()) return false;
        if (tokens[at] == kOpen) {
            ++at;
            if (!sum()) return false;
            skip();
            if (at >= tokens.size() || tokens[at] != kClose) return false;
            ++at;
            return true;
        }
        if (tokens[at] == kDollar || tokens[at] == kSpacedDollar) ++at;
        size_t last = at;
        while (last < tokens.size() && numeric_token(tokens[last])) ++last;
        if (last == at) return false;
        std::ptrdiff_t back = (std::ptrdiff_t) last - 1;
        InlineNumber number;
        const auto run = tokens.subspan(at, last - at);
        back = (std::ptrdiff_t) run.size() - 1;
        if (!read_number(run, back, number) || back != -1) return false;
        if (last < tokens.size() && tokens[last] == kPercent) {
            number.scale += 2;
            number.percent = true;
            ++last;
        }
        call.program.push_back(InlineStep{true, InlineOperation::Add, number});
        at = last;
        return true;
    }
    bool power() {
        if (!primary()) return false;
        size_t width = 1;
        if (peek(width) != InlineOperation::Power) return true;
        at += width;
        if (!power()) return false;  // right to left
        call.program.push_back(InlineStep{false, InlineOperation::Power, {}});
        return true;
    }
    bool product() {
        if (!power()) return false;
        int multiplied = 0;
        for (size_t width = 1;;) {
            const auto op = peek(width);
            if (!op || (*op != InlineOperation::Multiply && *op != InlineOperation::Divide)) {
                product_terms += multiplied;
                return true;
            }
            multiplied = 1;
            at += width;
            if (!power()) return false;
            call.program.push_back(InlineStep{false, *op, {}});
        }
    }
    bool sum() {
        if (!product()) return false;
        for (size_t width = 1;;) {
            const auto op = peek(width);
            if (!op || (*op != InlineOperation::Add && *op != InlineOperation::Subtract)) return true;
            at += width;
            if (!product()) return false;
            call.program.push_back(InlineStep{false, *op, {}});
        }
    }
};

inline std::optional<InlineCall> parse_inline(std::span<const int32_t> written) {
    using namespace inline_tokens;
    if (written.empty() || !closes_expression(written.back())) return std::nullopt;
    const auto split = split_brackets(written.last(std::min<size_t>(written.size(), 128)));
    const std::span<const int32_t> tokens = split;
    InlineCall call;
    call.spaced = tokens.back() == kSpacedEquals;
    // Walk back over the expression; an unmatched `(` belongs to the prose around it.
    std::ptrdiff_t first = (std::ptrdiff_t) tokens.size() - 1;
    int depth = 0;
    while (first > 0) {
        const int32_t t = tokens[(size_t) first - 1];
        if (t == kDot && tokens[(size_t) first] == kSpace) break;   // a sentence ended here, not a decimal
        if (first >= 2 && latex_operation(tokens[(size_t) first - 2], t)) {
            first -= 2;
            continue;
        }
        if (t == kClose) ++depth;
        else if (t == kOpen) {
            if (!depth) break;
            --depth;
        } else if (t == kOf ? !(first >= 2 && tokens[(size_t) first - 2] == kPercent)
                            : !(numeric_token(t) || t == kSpace || operation(t) || t == 115 || t == 1624
                                || t == kPercent || t == kDollar || t == kSpacedDollar)) break;
        --first;
    }
    while (first + 1 < (std::ptrdiff_t) tokens.size() && tokens[(size_t) first] == kSpace) ++first;
    const auto body = tokens.subspan((size_t) first, tokens.size() - 1 - (size_t) first);
    if (body.empty() || body.size() > 96) return std::nullopt;
    // A sign before the first number belongs to words we cannot see.
    if (first > 0 && tokens[(size_t) first - 1] == kDot) return std::nullopt;
    // The tokenizer glues signs to what comes before: `=-1+5=` and `)*4+1=` would read as whole expressions.
    std::ptrdiff_t before = first - 1;
    while (before >= 0 && tokens[(size_t) before] == kSpace) --before;
    if (before >= 0) {
        const int32_t t = tokens[(size_t) before];
        if (scales_next(tokens, before)) return std::nullopt;
        const bool glued = tokens[(size_t) first - 1] != kSpace;
        if (binds_next(t) || (tokens[(size_t) first] == kOpen && glued && !opens_cleanly(t))) return std::nullopt;
        if (glued && tokens[(size_t) first] != kOpen && !number_may_follow(t)) return std::nullopt;   // `d12^2=`, `sin60^3=`, `a1 +1 =`
        if (t == 332 && before > 0) {   // `**` is bold at the start of a line, a power after a value
            const int32_t left = tokens[(size_t) before - 1];
            if (digit(left) || left == kClose || left == 87) return std::nullopt;
        }
    }
    InlineParser parser{body, 0, call};
    if (!parser.sum()) return std::nullopt;
    parser.skip();
    if (parser.at != body.size()) return std::nullopt;
    // `2 \times 6 + 4 \times 1.5 = 12 + 6` shows its steps: the full value there reads as a term and gets added to again.
    if (parser.latex && parser.product_terms > 1) return std::nullopt;
    size_t steps = 0;
    for (const auto& step : call.program) steps += !step.number;
    if (!steps || steps > 15) return std::nullopt;
    // `100 - 30%` means 70 to a reader and 99.7 to arithmetic: percent is only taken as a factor.
    std::vector<bool> percent;
    for (const auto& step : call.program) {
        if (step.number) {
            percent.push_back(step.value.percent);
            continue;
        }
        const bool rhs = percent.back();
        percent.pop_back();
        const bool lhs = percent.back();
        percent.pop_back();
        const bool factor = step.operation == InlineOperation::Multiply || step.operation == InlineOperation::Divide;
        if (!factor && (lhs || rhs)) return std::nullopt;
        percent.push_back(false);
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
