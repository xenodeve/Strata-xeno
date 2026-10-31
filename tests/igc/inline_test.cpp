#include "strata/core/igc_inline.hpp"
#include <iostream>
#include <optional>
#include <stdexcept>
#include <vector>
using namespace strata::igc;
static void require(bool v, const char* e) { if (!v) throw std::runtime_error(e); }
static int number(const InlineNumber& value) {
    const Integer& v = value.coefficient;
    int n = 0;
    for (int i = v.length - 1; i >= 0; --i) n = n * 10 + v.digits[i];
    return n;
}
int main() try {
    // The program as text: digits, then the operation letters.
    auto program = [](const std::optional<InlineCall>& call) {
        std::vector<int> out;
        for (const auto& step : call->program) out.push_back(step.number ? number(step.value) : -(int) step.operation);
        return out;
    };
    auto a = parse_inline(std::vector<int32_t>{16, 17, 9, 18, 19, 28});
    require(a && program(a) == std::vector<int>{12, 34, -3} && !a->spaced, "tight");
    auto b = parse_inline(std::vector<int32_t>{16, 17, 23985, 220, 18, 19, 283});
    require(b && program(b) == std::vector<int>{12, 34, -3} && b->spaced, "spaced");
    auto c = parse_inline(std::vector<int32_t>{17, 11, 22, 20, 17, 478, 220, 18, 11, 16, 15, 15, 283});
    require(c && program(c) == std::vector<int>{2752, 3100, -1}, "commas");
    auto sentence = parse_inline(std::vector<int32_t>{24, 15, 24, 13, 220, 16, 17, 9, 18, 19, 283});  // 909. 12*34 =
    require(sentence && program(sentence) == std::vector<int>{12, 34, -3}, "sentence end");
    auto point = parse_inline(std::vector<int32_t>{16, 13, 20, 9, 17, 28});  // 1.5*2=
    require(point.has_value(), "decimal stays a number");
    auto chain = parse_inline(std::vector<int32_t>{16, 21, 12, 18, 12, 19, 28});  // 16-3-4=
    require(chain && program(chain) == std::vector<int>{16, 3, -2, 4, -2}, "chain");
    auto mixed = parse_inline(std::vector<int32_t>{17, 10, 18, 9, 19, 28});  // 2+3*4=
    require(mixed && program(mixed) == std::vector<int>{2, 3, 4, -3, -1}, "precedence");
    auto bracket = parse_inline(std::vector<int32_t>{7, 16, 10, 17, 8, 9, 18, 28});  // (1+2)*3=
    require(bracket && program(bracket) == std::vector<int>{1, 2, -1, 3, -3}, "parentheses");
    auto prose = parse_inline(std::vector<int32_t>{500, 7, 19, 15, 478, 220, 17, 15, 283});  // word (40 + 20 =
    require(prose && program(prose) == std::vector<int>{40, 20, -1}, "open bracket in prose");
    require(!parse_inline(std::vector<int32_t>{17, 7, 18, 10, 19, 8, 28}), "implicit multiply");
    auto decimal = parse_inline(std::vector<int32_t>{16, 13, 20, 10, 17, 28});  // 1.5+2=
    require(decimal && number(decimal->program[0].value) == 15 && decimal->program[0].value.scale == 1 && !decimal->exact_integers(), "decimal");
    auto divided = parse_inline(std::vector<int32_t>{16, 17, 1624, 115, 220, 18, 283});  // 12 ÷ 3 =
    require(divided && program(divided) == std::vector<int>{12, 3, -4} && !divided->exact_integers(), "division");
    require(!parse_inline(std::vector<int32_t>{13, 20, 10, 17, 28}), "bare point");
    auto squares = parse_inline(std::vector<int32_t>{18, 61, 17, 478, 220, 19, 61, 17, 283});  // 3^2 + 4^2 =
    require(squares && program(squares) == std::vector<int>{3, 2, -5, 4, 2, -5, -1}, "powers");
    auto percent = parse_inline(std::vector<int32_t>{18, 15, 4, 314, 220, 16, 23, 15, 283});  // 30% of 180 =
    require(percent && program(percent) == std::vector<int>{30, 180, -3} && percent->program[0].value.scale == 2 && !percent->exact_integers(), "percent of");
    require(!parse_inline(std::vector<int32_t>{16, 15, 15, 471, 220, 18, 15, 4, 283}), "percent as a discount");  // 100 - 30% =
    auto dollars = parse_inline(std::vector<int32_t>{3, 20, 348, 220, 18, 283});  // $5 * 3 =
    require(dollars && program(dollars) == std::vector<int>{5, 3, -3}, "dollars");
    auto joined = parse_inline(std::vector<int32_t>{16, 13, 20, 37726, 17, 15, 28});  // 1.5*$20=
    require(joined && program(joined) == std::vector<int>{15, 20, -3}, "times dollar");
    require(!parse_inline(std::vector<int32_t>{500, 314, 220, 16, 23, 15, 283}), "of without percent is prose");
    require(!parse_inline(std::vector<int32_t>{9, 18, 28}), "no left operand");
    require(!parse_inline(std::vector<int32_t>{16, 17, 18}), "no equals");
    require(!parse_inline(std::vector<int32_t>{16, 17, 28}), "number alone");
    rational_exact::Rational half{};
    half.numerator.length = 1; half.numerator.digits[0] = 7;
    half.denominator.length = 1; half.denominator.digits[0] = 2;
    require(rational_tokens(half, false) == std::vector<int32_t>{18, 13, 20}, "terminating decimal");
    half.denominator.digits[0] = 3;
    require(!rational_tokens(half, false), "repeating decimal stays the model's");
    std::vector<int32_t> tooLong(32, 16);
    tooLong.push_back(10); tooLong.push_back(16); tooLong.push_back(28);
    require(!parse_inline(tooLong), "operand bound");
    Integer negative{};
    negative.length = 2; negative.negative = 1; negative.digits[0] = 4; negative.digits[1] = 3;
    require(value_tokens(negative, true) == std::vector<int32_t>{471, 18, 19}, "negative spaced");
    require(value_tokens(negative, false) == std::vector<int32_t>{12, 18, 19}, "negative tight");
    negative.negative = 0;
    require(value_tokens(negative, true) == std::vector<int32_t>{220, 18, 19}, "positive spaced");
    // 'What is 68824 times 42716?' digits, with words between; a truncated reading is refused
    const std::vector<int64_t> prompt{1, 21, 23, 23, 17, 19, 2, 19, 17, 22, 16, 21, 3};
    Integer truncated{};
    truncated.length = 2; truncated.digits[0] = 8; truncated.digits[1] = 6;
    require(!written_in(prompt, truncated), "truncated operand");
    Integer whole{};
    whole.length = 5; whole.digits[0] = 4; whole.digits[1] = 2; whole.digits[2] = 8; whole.digits[3] = 8; whole.digits[4] = 6;
    require(written_in(prompt, whole), "whole operand");
    const std::vector<int64_t> commas{1, 16, 11, 15, 15, 15, 2};
    Integer thousand{};
    thousand.length = 4; thousand.digits[3] = 1;
    require(written_in(commas, thousand), "comma operand");
    std::cout << "PASS igc inline parser\n";
    return 0;
} catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 2; }
