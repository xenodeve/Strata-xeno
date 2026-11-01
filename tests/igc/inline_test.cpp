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
    {
        auto tail = [](std::vector<int32_t> round, int rounds) {
            std::vector<int32_t> out;
            for (int i = 0; i < 100; ++i) out.push_back(i * i % 97);
            for (int i = 0; i < rounds; ++i) out.insert(out.end(), round.begin(), round.end());
            return out;
        };
        const std::vector<int32_t> phrase{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
        require(inline_tokens::in_cycle(tail(phrase, 8), 128, 8), "eight rounds of a phrase are a cycle");
        {
            const auto rounds = [](int length, int tokens_between) {
                inline_tokens::LineRepeat lines;
                int fired = 0;
                for (int round = 1; round <= 40 && !fired; ++round) {
                    for (int i = 0; i < length; ++i) lines.step(1000 + i);
                    if (lines.step(198)) fired = round;
                    for (int i = 0; i < tokens_between; ++i) lines.step(5000 + round * 20 + i);
                    lines.step(271);
                }
                return fired;
            };
            require(rounds(12, 0) == inline_tokens::kLineRounds, "a long line back sixteen times is a loop");
            require(rounds(12, 15) == inline_tokens::kLineRounds, "other lines between its rounds change nothing");
            require(rounds(11, 0) == 0, "a short line can recur for ever");
            inline_tokens::LineRepeat lines;
            bool fired = false;
            for (int round = 0; round < 40; ++round) {
                for (int i = 0; i < 12; ++i) lines.step(round * 100 + i);
                fired = lines.step(198) || fired;
            }
            require(!fired, "different lines are not a loop");
        }
        require(!inline_tokens::in_cycle(tail(phrase, 7), 128, 8), "seven rounds can still leave");
        require(inline_tokens::in_cycle(tail({7, 8}, 32), 128, 8), "a short loop, 64 tokens");
        require(!inline_tokens::in_cycle(tail({7, 8}, 31), 128, 8), "a short run is not yet one");
        require(!inline_tokens::in_cycle(tail({15}, 13), 128, 8), "the zeros of 2102500000000");
        require(!inline_tokens::in_cycle(tail({18}, 40), 128, 8), "a repeating decimal");
        std::vector<int32_t> plain;
        for (int i = 0; i < 2000; ++i) plain.push_back(i * i % 9973);
        require(!inline_tokens::in_cycle(plain, 128, 8), "varied text is no cycle");
        std::vector<int32_t> run(1024, 5);
        require(inline_tokens::in_cycle(run, 128, 8), "one token repeated");
        require(!inline_tokens::in_cycle(std::vector<int32_t>{}, 128, 8), "nothing written");
        std::vector<int32_t> block;
        for (int i = 0; i < 350; ++i) block.push_back(1000 + i);
        require(inline_tokens::in_cycle(tail(block, 3), 512, 8), "a long block needs three rounds");
        require(!inline_tokens::in_cycle(tail(block, 2), 512, 8), "two rounds of a block can still leave");
        std::vector<int32_t> whole_list;
        for (int i = 0; i < 3630; ++i) whole_list.push_back(2000 + i);
        require(inline_tokens::in_cycle(tail(whole_list, 3), 4096, 8), "a list of 3630 tokens, three rounds");
        require(!inline_tokens::in_cycle(tail(whole_list, 2), 4096, 8), "two rounds of it can still leave");
        require(!inline_tokens::in_cycle(tail(whole_list, 3), 512, 8), "and a short reach does not see it");
        std::vector<int32_t> mid(block.begin(), block.begin() + 128);
        require(inline_tokens::in_cycle(tail(mid, 8), 512, 8), "128 tokens, eight rounds");
        require(!inline_tokens::in_cycle(tail(mid, 7), 512, 8), "128 tokens, seven rounds");
        const std::vector<int32_t> exact{318, 44839, 8}, confirmed{318, 43967, 10457, 11, 2842, 12910, 8}, closing{13, 198, 77, 271};
        require(inline_tokens::after_value(0, true, 3, 77) == exact, "the first value is marked");
        require(inline_tokens::after_value(1, true, 3, 77) == confirmed, "the first repeat is confirmed");
        require(inline_tokens::after_value(2, true, 3, 77) == closing, "the third closes");
        require(inline_tokens::after_value(9, true, 3, 77) == closing, "any later one closes");
        require(inline_tokens::after_value(9, true, 0, 77) == confirmed, "0 never closes");
        require(inline_tokens::after_value(1, false, 3, 77).empty(), "no note, no close yet");
        require(inline_tokens::after_value(2, false, 3, 77) == closing, "the close does not depend on notes");
        require(inline_tokens::after_value(0, false, 0, 77).empty(), "the value alone");
        using inline_tokens::SumNote;
        const std::vector<int32_t> exact_stop{318, 44839, 553}, confirmed_stop{318, 43967, 10457, 11, 2842, 12910, 553};
        require(inline_tokens::after_value(0, true, 3, 77, SumNote::Stop) == exact_stop, "a sum closes its note");
        require(inline_tokens::after_value(1, true, 3, 77, SumNote::Stop) == confirmed_stop, "a repeated sum closes its note");
        require(inline_tokens::after_value(2, true, 3, 77, SumNote::Stop) == closing, "the third still closes a sum");
        require(inline_tokens::after_value(0, false, 3, 77, SumNote::Stop).empty(), "no notes, nothing to close");
        require(inline_tokens::after_value(0, true, 3, 77, SumNote::None).empty(), "a sum alone");
        require(inline_tokens::after_value(1, true, 3, 77, SumNote::None).empty(), "a repeated sum alone");
        require(inline_tokens::after_value(2, true, 3, 77, SumNote::None) == closing, "the third still closes without a note");
        const std::vector<int64_t> off{5, 248068, 271, 248069, 271}, on{5, 248068, 198}, history{248069, 271, 5, 248068, 198}, one{271};
        require(inline_tokens::thinking_off(off, 248069), "a closed thinking ends the prompt");
        require(!inline_tokens::thinking_off(on, 248069), "an open one does not");
        require(!inline_tokens::thinking_off(history, 248069), "nor an earlier turn's close");
        require(!inline_tokens::thinking_off(one, 248069), "a short prompt");
        using inline_tokens::RepeatNote;
        const std::vector<int32_t> bare{318, 43967, 10457, 8}, bare_stop{318, 43967, 10457, 553}, next{318, 43967, 10457, 11, 1727, 2923, 8};
        require(inline_tokens::after_value(1, true, 3, 77, SumNote::Open, RepeatNote::Stop) == confirmed, "stop checking stays the default");
        require(inline_tokens::after_value(1, true, 3, 77, SumNote::Open, RepeatNote::Bare) == bare, "a bare repeat note");
        require(inline_tokens::after_value(1, true, 3, 77, SumNote::Stop, RepeatNote::Bare) == bare_stop, "closed on a sum");
        require(inline_tokens::after_value(1, true, 3, 77, SumNote::Open, RepeatNote::Next) == next, "next step");
        require(inline_tokens::after_value(0, true, 3, 77, SumNote::Open, RepeatNote::Bare) == exact, "the first note does not change");
        require(inline_tokens::after_value(2, true, 3, 77, SumNote::Open, RepeatNote::Next) == closing, "nor does the close");
        require(inline_tokens::after_value(1, true, 3, 77, SumNote::None, RepeatNote::Next).empty(), "no note on the sum, none on its repeat");
    }
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
    auto refused = [](std::vector<int32_t> tokens) { return !parse_inline(tokens).has_value(); };
    require(refused({64, 10512, 16, 10, 20, 28}), "=-1+5= keeps its sign or stays out");
    require(refused({4653, 19, 10, 16, 10, 17, 28}), ")*4+1+2=");
    require(refused({87, 332, 17, 10, 16, 28}), "x**2+1=");
    require(refused({58, 21, 11, 24, 10, 19, 28}), "[6,9+4= is a list");
    require(parse_inline(std::vector<int32_t>{16, 11, 15, 15, 15, 10, 20, 28}).has_value(), "1,000+5=");
    require(refused({67, 16, 17, 61, 17, 28}), "d12^2= is a name");
    require(refused({64, 16, 478, 16, 283}), "a1 +1 =");
    require(refused({15470, 21, 15, 61, 18, 28}), "sin60^3=");
    require(refused({26026, 19, 16, 471, 16, 24, 283}), "sqrt41 -19 =");
    require(refused({87, 62, 16, 10, 17, 28}), "x_1+2=");
    require(refused({26026, 18, 14, 18, 283}), "sqrt3/3 =");
    require(refused({169614, 17, 10, 16, 28}), "a root sign binds");
    require(refused({4139, 314, 220, 21, 23, 478, 220, 16, 15, 283}), "half of 68 + 10 =");
    require(refused({41212, 314, 220, 21, 23, 478, 220, 16, 15, 283}), "Half of 68 + 10 =");
    require(refused({4139, 279, 220, 21, 23, 478, 220, 16, 15, 283}), "half the 68 + 10 =");
    require(refused({4139, 314, 279, 220, 21, 23, 478, 220, 16, 15, 283}), "half of the 68 + 10 =");
    require(refused({4139, 314, 318, 21, 23, 478, 220, 16, 15, 8, 283}), "half of (68 + 10) =");
    require(refused({4139, 314, 220, 21, 23, 9, 18, 283}), "half of 68*3 =");
    require(refused({10598, 220, 20, 478, 220, 18, 283}), "twice 5 + 3 =");
    require(refused({16, 14, 19, 314, 220, 17, 15, 478, 220, 20, 283}), "1/4 of 20 + 5 =");
    require(refused({15, 13, 20, 314, 220, 21, 23, 478, 220, 16, 15, 283}), "0.5 of 68 + 10 =");
    require(parse_inline(std::vector<int32_t>{2542, 314, 220, 21, 23, 478, 220, 16, 15, 283}).has_value(), "sum of 68 + 10 =");
    require(parse_inline(std::vector<int32_t>{2702, 220, 21, 23, 478, 220, 16, 15, 283}).has_value(), "total 68 + 10 =");
    require(parse_inline(std::vector<int32_t>{17, 20, 4, 314, 220, 19, 15, 478, 220, 18, 283}).has_value(), "25% of 40 + 3 =");
    require(parse_inline(std::vector<int32_t>{7354, 220, 17, 10, 17, 28}).has_value(), "a word and a space");
    require(parse_inline(std::vector<int32_t>{3, 17, 10, 17, 28}).has_value(), "$2+2=");
    require(parse_inline(std::vector<int32_t>{332, 17, 10, 17, 28}).has_value(), "**2+2=");
    require(refused({18, 2942, 220, 19, 478, 220, 16, 283}), "3 times 4 + 1 =");
    require(refused({830, 220, 19, 478, 220, 16, 283}), "3 x 4 + 1 =");
    require(refused({1088, 14695, 220, 16, 12, 17, 283}), "a bare \\times has no left side");
    require(refused({13791, 7, 16, 23, 15, 12, 19, 21, 8, 283}), " tan(180-46) =");
    auto bold = parse_inline(std::vector<int32_t>{271, 332, 17, 478, 220, 16, 283});  // \n\n**2 + 1 =
    require(bold && program(bold) == std::vector<int>{2, 1, -1}, "bold opens a line");
    auto glued = parse_inline(std::vector<int32_t>{271, 7, 18, 10, 19, 8, 9, 17, 28});  // \n\n(3+4)*2=
    require(glued && program(glued) == std::vector<int>{3, 4, -1, 2, -3}, "a bracket opening a line");
    auto latex = parse_inline(std::vector<int32_t>{17237, 19, 1088, 14695, 220, 16, 17, 20, 471, 220, 17, 283});  // \(4 \times 125 - 2 =
    require(latex && program(latex) == std::vector<int>{4, 125, -3, 2, -2}, "\\times");
    auto tight = parse_inline(std::vector<int32_t>{19, 4795, 1672, 220, 20, 28});  // 4\times 5=
    require(tight && program(tight) == std::vector<int>{4, 5, -3}, "tight \\times");
    auto cdot = parse_inline(std::vector<int32_t>{19, 59, 49142, 20, 28});  // 4\cdot5=
    require(cdot && program(cdot) == std::vector<int>{4, 5, -3}, "\\cdot");
    auto over = parse_inline(std::vector<int32_t>{19, 1088, 596, 220, 20, 283});  // 4 \div 5 =
    require(over && program(over) == std::vector<int>{4, 5, -4}, "\\div");
    // \\(2 \\times 6 + 4 \\times 1 =
    require(refused({17237, 17, 1088, 14695, 220, 21, 478, 220, 19, 1088, 14695, 220, 16, 283}), "shown working stays the model's");
    auto plain = parse_inline(std::vector<int32_t>{17, 9, 21, 10, 19, 9, 16, 28});  // 2*6+4*1=
    require(plain.has_value(), "the same sum in plain text is read");
    auto closes = parse_inline(std::vector<int32_t>{19, 6391, 20, 10, 16, 11393});  // 4*(5+1)=
    require(closes && program(closes) == std::vector<int>{4, 5, 1, -1, -3} && !closes->spaced, "a bracket glued to =");
    auto after = parse_inline(std::vector<int32_t>{7, 17, 10, 18, 4653, 19, 28});  // (2+3)*4=
    require(after && program(after) == std::vector<int>{2, 3, -1, 4, -3}, "a bracket glued to its operation");
    auto squared = parse_inline(std::vector<int32_t>{7, 16, 15, 15, 12, 16, 28803, 17, 10, 24, 24, 283});  // (100-1)^2+99 =
    require(squared && program(squared) == std::vector<int>{100, 1, -2, 2, -5, 99, -1}, "a bracket glued to its power");
    auto opened = parse_inline(std::vector<int32_t>{64, 318, 18, 10, 19, 8, 9, 17, 28});  // a (3+4)*2=
    require(opened && program(opened) == std::vector<int>{3, 4, -1, 2, -3}, "a spaced bracket");
    require(refused({87, 7, 16, 8, 7030, 17, 9, 18, 28}), "x(1)+2*3=");
    require(refused({87, 8, 7030, 17, 9, 18, 28}), "x)+2*3=");
    require(refused({17, 7030, 18, 28}), "2)+3=");
    auto dots = parse_inline(std::vector<int32_t>{17, 13535, 18, 9306, 220, 19, 28});  // 2·3 · 4=
    require(dots && program(dots) == std::vector<int>{2, 3, -3, 4, -3}, "middle dot");
    auto minus = parse_inline(std::vector<int32_t>{20, 32160, 17, 23437, 220, 16, 28});  // 5−2 − 1=
    require(minus && program(minus) == std::vector<int>{5, 2, -2, 1, -2}, "unicode minus");
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
