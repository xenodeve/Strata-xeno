#include "strata/core/igc_inline.hpp"
#include <iostream>
#include <stdexcept>
#include <vector>
using namespace strata::igc;
static void require(bool v, const char* e) { if (!v) throw std::runtime_error(e); }
static int number(const Integer& v) {
    int n = 0;
    for (int i = v.length - 1; i >= 0; --i) n = n * 10 + v.digits[i];
    return n;
}
int main() try {
    // '12*34=' and '12 × 34 = ' as the pack tokenizes them
    auto a = parse_inline(std::vector<int32_t>{16, 17, 9, 18, 19, 28});
    require(a && a->operation == Operation::Multiply && number(a->lhs) == 12 && number(a->rhs) == 34 && !a->spaced, "tight");
    auto b = parse_inline(std::vector<int32_t>{16, 17, 23985, 220, 18, 19, 283});
    require(b && b->operation == Operation::Multiply && b->spaced, "spaced");
    auto c = parse_inline(std::vector<int32_t>{17, 11, 22, 20, 17, 478, 220, 18, 11, 16, 15, 15, 283});
    require(c && number(c->lhs) == 2752 && number(c->rhs) == 3100, "commas");
    require(!parse_inline(std::vector<int32_t>{16, 22, 12, 18, 12, 19, 28}), "chain is the model's");
    require(!parse_inline(std::vector<int32_t>{7, 16, 10, 17, 8, 9, 18, 28}), "parenthesis");
    require(!parse_inline(std::vector<int32_t>{16, 13, 20, 10, 17, 28}), "decimal");
    require(!parse_inline(std::vector<int32_t>{16, 17, 18}), "no equals");
    require(!parse_inline(std::vector<int32_t>{9, 18, 28}), "no left operand");
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
