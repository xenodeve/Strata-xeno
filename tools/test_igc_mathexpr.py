"""tools/igc/mathexpr.py: every way a model writes arithmetic, read exactly. python tools/test_igc_mathexpr.py"""
import sys, unittest
from fractions import Fraction as F
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.igc.mathexpr import evaluate, expression_before, wrong_relations


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
        self.assertIsNone(evaluate('2 (2)'))   # a space before the bracket: `= 2 (2)` is a value and a note, not 2 x 2
        self.assertEqual(evaluate('(2)(3)(7)'), 42)

    def test_forms_found_in_the_traces(self):
        # tier 1 of the catalogue from 929 stored answers (probe/forms.py), each with the trace's own spelling
        self.assertEqual(evaluate('2016 \\bmod 1000'), 16)
        self.assertEqual(evaluate('2016 mod 1000'), 16)
        self.assertEqual(evaluate('2016 \\pmod{1000}'), 16)
        self.assertEqual(evaluate('180^\\circ - 30^\\circ'), 150)
        self.assertEqual(evaluate('180° - 30°'), 150)
        self.assertEqual(evaluate('|2304 - 2025|'), 279)
        self.assertEqual(evaluate('\\left|3 - 10\\right| + |{-2}|'), 9)
        self.assertEqual(evaluate('\\lfloor 729/27 \\rfloor'), 27)
        self.assertEqual(evaluate('\\lfloor 5/2 \\rfloor + \\lceil 5/2 \\rceil'), 5)
        self.assertEqual(evaluate('\\binom{9}{3}'), 84)
        self.assertEqual(evaluate('\\dbinom{9}{4, 3, 2}'), 1260)
        self.assertEqual(evaluate('\\frac{9!}{4! 3! 2!}'), 1260)
        self.assertIsNone(evaluate('\\binom{n}{2}'))

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

    def test_the_tail_of_an_expression_on_a_symbol_is_none(self):
        # `x - 3 \times 4 =` goes on as `x - 12`: a value of 12 put in there would be wrong, and `Z / (7/3)` is no value of 7/3
        for text in ('x - 3 \\times 4 ', 'b_2 - 3 + 4 ', 'W = Z / (7/3) '):
            self.assertIsNone(expression_before(text), text)
        self.assertEqual(expression_before('so the sum - 3 \\times 4 '), ('- 3 \\times 4', -12))   # after a word, the minus is a sign

    def test_a_lone_number_or_algebra_is_none(self):
        for text in ('so x ', 'it is 17 ', 'a3 + 4 ', '2x + 3 ', ''):
            self.assertIsNone(expression_before(text), text)


class WrongRelations(unittest.TestCase):
    """Every relation on a line, checked: what the hand check corrects."""
    def test_a_wrong_value_and_a_wrong_rewrite(self):
        self.assertEqual(wrong_relations('so 144 - 144 - 572 - 168 + 96 = -744.'), [('144 - 144 - 572 - 168 + 96', '-744', '-644')])
        self.assertEqual(wrong_relations('$9 \\cdot 21 + 7 = 189 + 17 = 196$'), [('9 \\cdot 21 + 7', '189 + 17', '196')])

    def test_right_ones_pass(self):
        for line in ('$9 \\cdot 21 + 7 = 189 + 7 = 196$.', 'b 151.45 - 51.47 = 99.98', '$97/17 \\approx 5.7$', '$2016 \\equiv 16 \\pmod{1000}$',
                     '$7+3+5 \\to 15$', '- $13 \\div 2 = 6$ remainder $1$', '$x + 3 = 5$', 'Count = $\\lfloor 729/27 \\rfloor = 27$.',
                     '$|N - 2025| = |2304 - 2025| = 279$', '$180^\\circ - 30^\\circ = 150^\\circ$', '1 -> -6 (No)'):
            self.assertEqual(wrong_relations(line), [], line)

    def test_false_corrections_of_dm_check_pass(self):
        # each "corrected" a right hand value in the first --check-hand run (dm mul_big 7808, div_big 2289, div_big 6890, mul_div 230, add_or_sub 8989)
        for line in ('$600 \\times 8 = 4.8k$', '$37 / 1.8329827 \\times 10^7 = 20.1852... \\times 10^{-7}$',
                     '1.  $1.000000000 \\div 187897983 = 0$', '$438 + 2/3 = 438.666$.', '  $1 - 7 = 4$ (borrow)', '$x = 3$, $y = 5$'):
            self.assertEqual(wrong_relations(line), [], line)

    def test_carries_estimates_and_spaced_brackets_pass(self):
        # the first line checker "corrected" these on stored base answers (dm mul_big 1649, 5409, 8376, 1618)
        for line in ('$5 \\times 2 = 10 + 1 = 11$', '$4.2 \\times 53 = 4 \\times 53 = 212$', '$0.00431 \\times 58463 = 0.004 \\times 58500$',
                     'so $0+0+0+0+1+1 = 2$ (2)', '$25327 \\times 276.2011 = 25000 \\times 276$'):
            self.assertEqual(wrong_relations(line), [], line)
        self.assertEqual(wrong_relations('$222 \\times 1327 = 292594$'), [('222 \\times 1327', '292594', '294594')])   # still caught
        for line in ('$$ 0.7412898 \\times 10^{-9} = 7.412898 \\times 10^{-10} $$',      # a tiny value is a step too (div_big 3578)
                     '$$ W = Z / (7/3) = \\frac{-1/3}{7/3} $$',                          # the tail of Z / (7/3), not a value (mul_div 663)
                     '$63.5 \\times 10^6 / 29 \\times 10^3 = 2190$', '$27443490133 \\div 22114013 = 1240$',   # rounded to its places
                     '$10^9 / 187897983 = 5.322000000000000$', '$10^9 / 187897983 = 5.3220000000000005$',   # a float's tail
                     '$\\frac{4.99}{4.18} \\times 10^4 = 1.19 \\times 10000$',              # 10000 as a power of ten
                     '$27443490133 \\div 22114013 = \\frac{27.44 \\times 10^9}{22.11 \\times 10^6}$',   # the same on rounded numbers
                     '$80 / 189 = 4$'):                                                 # a long division's digit, 800 / 189
            self.assertEqual(wrong_relations(line), [], line)
        self.assertEqual(wrong_relations('$9 \\times 78621 = 707889$'), [('9 \\times 78621', '707889', '707589')])
        self.assertEqual(wrong_relations('$1/731721563 \\approx 1.36664166 \\times 10^{-9}$'),   # wrong in its sixth digit
                         [('1/731721563', '1.36664166 \\times 10^{-9}', '1.36664006 \\times 10^{-9}')])
        self.assertEqual(wrong_relations('$-3312 / 96 = -104 / 3$'), [('-3312 / 96', '-104 / 3', '-34.5')])   # 0.5 % off is wrong

    def test_aime_style_lines_that_are_right(self):
        # real lines of the 4B's AIME / HMMT answers the line checker once "corrected"
        for line in ('*   $12^2 = 6^2 + (\\frac{s-r}{2})^2$.', '$$576 + (144 \\times 3) = 2(r^2 + s^2)$$',
                     'Alternatively, $9! / (6! 2! 1!) = \\binom{9}{6} \\binom{3}{2} \\binom{1}{1} = \\binom{9}{6} \\binom{3}{2} = 84 \\times 3 = 252$.',
                     '$3+1 \\times 1+1 = 8$.', 'Check: $12(-66^2) - (-66)(99) - 6(99^2) = 0$. Yes.',
                     'If $(x,y)$ is in both, then $3/4 = -2/3$ (for $y \\ne 0$).',
                     "$y' = \\frac{\\sqrt{3}}{2}(1.31) + \\frac{1}{2}(-2.28) \\approx 0.866(1.31) - 1.14 \\approx 1.13 - 1.14 \\approx -0.01$.",
                     "$x'^2 - 4 \\approx 2.6269^2 - 4 \\approx 6.89 - 4 = 2.89$.", '$9(-1.55)^4 \\approx 9(5.78) = 52$.',
                     '$-508/9 + 152 = -56.44 + 152 \\neq 0$.', '$0.23 + 1.152 - 22.88 + 33.6 + 96 = 130.98 - 22.88 = 108$.',
                     '$464.7 - 483 = -18$.', '$6^9 \\approx 10^7$.', '*   $= 126 + 50/3 = 126 + 16.67 = 142.67$.',
                     'Then $u^2 (1 + \\frac{169 \\times 3}{841}) = 169$.'):
            self.assertEqual(wrong_relations(line), [], line)

    def test_aime_style_lines_that_are_wrong(self):
        for line, right in (('Why did I think $288 \\times 315 = 90360$?', '90720'), ('$\\frac{9!}{4!} = 9 \\times 8 \\times 7 \\times 6 = 3024$.', '15120'),
                            ('*   So expected number of intersections with base lines is $25 \\times 2/3 + 25 \\times 2/3 = 50/3$.', '100/3'),
                            ('*   So $E[R] = 29 + 478/3 = 562/3$.', '565/3'), ('*   $R = 1 + 3 + (3-1) = 5$.', '6'),
                            ('$9.74 + 19.10 - 148.78 + 181.68 = 187.92 - 148.78 = 39.14$.', '61.74'),
                            ('$94080 = 9408 \\times 10 = 2 \\times 4704 \\times 5 = 10 \\times 4704$.', '94080')):
            got = wrong_relations(line)
            self.assertTrue(got and got[0][2] == right, (line, got))

    def test_other_relations(self):
        self.assertEqual(wrong_relations('$115/18 \\approx 6.83$'), [('115/18', '6.83', '6.39')])
        self.assertEqual(wrong_relations('$2016 \\equiv 18 \\pmod{1000}$'), [('2016', '18 \\pmod{1000}', '16 \\pmod{1000}')])
        self.assertEqual(wrong_relations('$8+4+3 \\to 16$'), [('8+4+3', '16', '15')])
        # a value written in scientific form is answered in the same form: 37/18329827 is 20.18567... x 10^-7
        self.assertEqual(wrong_relations('$37 / 1.8329827 \\times 10^7 = 20.1852 \\times 10^{-7}$'),
                         [('37 / 1.8329827 \\times 10^7', '20.1852 \\times 10^{-7}', '20.1857 \\times 10^{-7}')])


if __name__ == '__main__':
    unittest.main()
