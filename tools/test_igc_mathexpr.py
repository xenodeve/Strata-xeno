"""tools/igc/mathexpr.py: every way a model writes arithmetic, read exactly. python tools/test_igc_mathexpr.py"""
import sys, unittest
from fractions import Fraction as F
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.igc.mathexpr import evaluate, expression_before


class Evaluate(unittest.TestCase):
    def test_every_spelling_of_the_operators(self):
        for text in ('6*7', '6 \\times 7', '6 \\cdot 7', '6 × 7', '6·7', '6 ⋅ 7', '6 \\ast 7', '84/2', '84 \\div 2', '84 ÷ 2', '50 - 8', '50 − 8',
                     '40+2', '2^{5} + 10', '2^5+10', '2**5 + 10'):
            self.assertEqual(evaluate(text), 42, text)

    def test_fractions_at_any_depth(self):
        self.assertEqual(evaluate('\\frac{3}{4} + 1/4'), 1)
        self.assertEqual(evaluate('\\dfrac{1}{2} + \\tfrac{1}{2}'), 1)
        self.assertEqual(evaluate('\\frac{\\frac{-153}{357} \\div \\frac{2}{7}}{\\frac{6}{-16} \\cdot \\frac{-480}{-650} \\div -6 \\cdot \\frac{-10}{24}}'), 78)
        self.assertEqual(evaluate('\\frac{9 \\times 8 \\times 7}{3 \\times 2 \\times 1}'), 84)

    def test_brackets_of_every_kind(self):
        self.assertEqual(evaluate('(-1*-638*(-7)/(-7))/(((-62)/(-279))/(((-170)/3060)/(1/(-2))))'), 319)
        self.assertEqual(evaluate('\\left( 2 + 3 \\right) \\times [4 - 1]'), 15)
        self.assertEqual(evaluate('\\bigl(1 + 1\\bigr) \\cdot \\{2\\}'), 4)
        self.assertEqual(evaluate('{2 + 3}^{2}'), 25)

    def test_numbers_however_written(self):
        self.assertEqual(evaluate('12,510,204 \\times 100'), 1251020400)
        self.assertEqual(evaluate('0.1 + 0.2'), F(3, 10))
        self.assertEqual(evaluate('1.5e3 + 1'), 1501)
        self.assertEqual(evaluate('3 \\times 10^{-2}'), F(3, 100))
        self.assertEqual(evaluate('1\\,000 + 1'), 1001)
        self.assertEqual(evaluate('3²+4²'), 25)
        self.assertEqual(evaluate('5! / 3!'), 20)
        self.assertEqual(evaluate('20\\% \\times 50'), 10)
        self.assertEqual(evaluate('15% * 200'), 30)
        self.assertEqual(evaluate('\\sqrt{144} + 1'), 13)
        self.assertEqual(evaluate('2(3 + 4)'), 14)
        self.assertEqual(evaluate('(2)(3)(7)'), 42)

    def test_what_is_not_exact_arithmetic(self):
        for text in ('2x + 3', 'x', '\\sqrt{2}', '9^{999}', '2\\frac{1}{3}', '(3 + 4', '3 + ', '1.2.3', '', '\\text{a} + 1', '2^{1/2}', '1/0',
                     '\\pi \\times 2', 'a_1 + 2'):
            self.assertIsNone(evaluate(text), text)


class ExpressionBefore(unittest.TestCase):
    def test_the_longest_expression_that_ends_at_the_equals(self):
        self.assertEqual(expression_before('So the total is 3 \\times 4 '), ('3 \\times 4', 12))
        self.assertEqual(expression_before('$12,510,204 \\times 100 '), ('12,510,204 \\times 100', 1251020400))   # not cut at its comma
        self.assertEqual(expression_before('so -(25 * 130) '), ('-(25 * 130)', -3250))                          # its own sign
        self.assertEqual(expression_before('= 189 + 7 '), ('189 + 7', 196))

    def test_a_minus_after_a_symbol_is_not_a_sign(self):
        self.assertEqual(expression_before('x - 3 \\times 4 '), ('3 \\times 4', 12))
        self.assertEqual(expression_before('b_2 - 3 + 4 '), ('3 + 4', 7))

    def test_a_lone_number_or_algebra_is_none(self):
        for text in ('so x ', 'it is 17 ', 'a3 + 4 ', '2x + 3 ', ''):
            self.assertIsNone(expression_before(text), text)


if __name__ == '__main__':
    unittest.main()
