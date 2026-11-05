"""tools/igc/mathbackend/symbolic: python tools/test_mathbackend_symbolic.py (standard library only, no model, no network)."""
import random
import sys
import unittest
from decimal import Decimal
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.igc.mathbackend import BadInput, NotCovered, registry, run
from tools.igc.mathbackend import symbolic as S


def P(t):
    return S.parse(t)


class ParserTests(unittest.TestCase):
    def test_infix_forms(self):
        self.assertEqual(run('symbolic.canon', '(x+1)^2'), 'x^2 + 2*x + 1')
        self.assertEqual(run('symbolic.canon', '2x(x+1)'), '2*x^2 + 2*x')
        self.assertEqual(run('symbolic.canon', '-x^2'), '-x^2')
        self.assertEqual(run('symbolic.canon', '2^3^2'), '512')
        self.assertEqual(run('symbolic.canon', 'x**2'), 'x^2')
        self.assertEqual(run('symbolic.canon', '1,000 + 1'), '1001')
        self.assertEqual(run('symbolic.canon', 'x_1 + x_{2}'), 'x_1 + x_2')

    def test_latex_forms(self):
        self.assertEqual(run('symbolic.canon', '\\frac{x^2+1}{2}'), 'x^2/2 + 1/2')
        self.assertEqual(run('symbolic.canon', '\\dfrac{1}{\\sqrt{x}}'), '1/x^(1/2)')
        self.assertEqual(run('symbolic.canon', '2\\cdot x^{3}\\times 3'), '6*x^3')
        self.assertEqual(run('symbolic.canon', '\\sqrt[3]{27}x'), '3*x')
        self.assertEqual(run('symbolic.canon', '\\left(x+1\\right)^{2}'), 'x^2 + 2*x + 1')
        self.assertEqual(run('symbolic.canon', '\\sin^2 x'), 'sin(x)^2')
        self.assertEqual(run('symbolic.canon', '\\log_2 8'), '3')
        self.assertEqual(run('symbolic.canon', '\\sin\\left(\\frac{\\pi}{6}\\right)'), '1/2')
        self.assertEqual(run('symbolic.canon', '|x-1|'), 'abs(x - 1)')
        self.assertEqual(run('symbolic.canon', '\\lfloor 7/2 \\rfloor'), '3')

    def test_euler_constant_and_e(self):
        self.assertEqual(run('symbolic.canon', 'E^x'), 'exp(x)')
        self.assertEqual(run('symbolic.canon', 'e + 1'), 'e + 1')                     # e is a plain symbol unless asked
        self.assertEqual(run('symbolic.canon', 'e^x', e_is_euler=True), 'exp(x)')
        with self.assertRaises(NotCovered):
            run('symbolic.canon', 'e^x')                                              # a symbol to a symbolic power: refused, not guessed

    def test_ambiguous_or_malformed_is_bad_input(self):
        for t in ['', '(x', 'x +', 'x2', 'sin 2x', '3 4', '5!', '\\int x', 'x = 1', 'sin^{-1} x', '1 2', 'x$']:
            with self.assertRaises(BadInput, msg=t):
                run('symbolic.canon', t)

    def test_roundtrip_of_random_trees(self):
        rs = random.Random(7)

        def tree(d):
            if d == 0:
                return rs.choice(['x', 'y', str(rs.randint(1, 5)), '1/2', '3/4'])
            a, b = tree(d - 1), tree(d - 1)
            k = rs.randint(0, 9)
            return [f'({a})+({b})', f'({a})-({b})', f'({a})*({b})', f'({a})^2', f'({a})/(({b})^2+1)', f'sin({a})', f'exp(({a})/3)', f'log(({a})^2+1)',
                    f'sqrt(({a})^2+2)', f'({a})^(1/2)*({b})'][k]
        for _ in range(250):
            t = tree(rs.randint(1, 3))
            try:
                e = P(t)
            except (NotCovered, BadInput):
                continue
            self.assertEqual(P(S.to_text(e)), e, t)


class CanonicalFormTests(unittest.TestCase):
    def test_expansion_and_like_terms(self):
        self.assertEqual(run('symbolic.expand', '(x+1)^3'), 'x^3 + 3*x^2 + 3*x + 1')
        self.assertEqual(run('symbolic.canon', '(x+1)^3-(x-1)^3'), '6*x^2 + 2')
        self.assertEqual(run('symbolic.canon', '(x+y)^2-(x-y)^2'), '4*x*y')

    def test_radicals(self):
        self.assertEqual(run('symbolic.canon', 'sqrt(8)'), '2*2^(1/2)')
        self.assertEqual(run('symbolic.canon', 'sqrt(2)*sqrt(2)'), '2')
        self.assertEqual(run('symbolic.canon', 'sqrt(6)'), '2^(1/2)*3^(1/2)')
        self.assertEqual(run('symbolic.canon', '(1+sqrt(2))^2'), '3 + 2*2^(1/2)')
        self.assertEqual(run('symbolic.canon', '8^(1/3)'), '2')

    def test_no_guessed_identities(self):
        self.assertEqual(run('symbolic.canon', 'sqrt(x^2)'), 'abs(x)')              # never x
        self.assertEqual(run('symbolic.canon', 'sqrt(x)^2'), 'x')                   # true wherever the left side is defined
        self.assertEqual(run('symbolic.canon', 'sin(x)^2+cos(x)^2'), 'cos(x)^2 + sin(x)^2')
        with self.assertRaises(NotCovered):
            run('symbolic.equal', 'sin(x)^2+cos(x)^2', '1')                         # an identity the form does not know: refused, not "False"
        self.assertEqual(run('symbolic.canon', 'log(x^2)'), 'log(x^2)')            # not 2 log(x)
        self.assertEqual(run('symbolic.canon', '(x*y)^(1/2)'), '(x*y)^(1/2)')      # not sqrt(x) sqrt(y)

    def test_exp_log_trig_tables(self):
        self.assertEqual(run('symbolic.canon', 'exp(x)*exp(-x)'), '1')
        self.assertEqual(run('symbolic.canon', 'exp(2*log(3))'), '9')
        self.assertEqual(run('symbolic.canon', 'log(12)'), '2*log(2) + log(3)')
        self.assertEqual(run('symbolic.canon', 'sin(pi/6)+cos(pi/3)'), '1')
        self.assertEqual(run('symbolic.canon', 'sin(-x)'), '-sin(x)')
        self.assertEqual(run('symbolic.canon', '2^x*2^(x+1)'), '2*exp(2*x*log(2))')
        self.assertEqual(run('symbolic.canon', '(-1)^k*(-1)^(k+1)'), '-1')

    def test_sum_powers_normal(self):
        # (x+1)^(3/2) is (x+1)*(x+1)^(1/2): equal forms
        self.assertTrue(run('symbolic.equal', '(x+1)^(3/2)', 'x*sqrt(x+1)+sqrt(x+1)'))
        self.assertTrue(run('symbolic.equal', '1/(x+1)^2', '(x+1)^(-2)'))
        self.assertEqual(run('symbolic.canon', '1/(x+1)^2'), '(x + 1)^(-2)')

    def test_equal(self):
        self.assertTrue(run('symbolic.equal', '(x+1)^2', 'x^2+2x+1'))
        self.assertFalse(run('symbolic.equal', '(x+1)^2', 'x^2+2x'))
        self.assertTrue(run('symbolic.equal', '1/(x-1)-1/(x+1)', '2/(x^2-1)'))
        self.assertFalse(run('symbolic.equal', '1/(x-1)', '1/(x+1)'))
        self.assertTrue(run('symbolic.equal', 'sqrt(3)*sqrt(3)', '3'))

    def test_simplify_and_rational_form(self):
        self.assertEqual(run('symbolic.simplify', '(x^2-1)/(x-1)'), 'x + 1')
        self.assertEqual(run('symbolic.simplify', '(x^3-1)/(x^2-1)'), '(x^2 + x + 1)/(x + 1)')
        self.assertEqual(run('symbolic.rational_form', '1/x + 1/(x+1)', 'x'), ('2*x + 1', 'x^2 + x'))
        self.assertEqual(run('symbolic.rational_form', '(2*x+2)/(4*x+4)', 'x'), ('1', '2'))


class CalculusTests(unittest.TestCase):
    def num(self, expr, **vals):
        return run('symbolic.evaluate_numeric', expr, {k: str(v) for k, v in vals.items()}, 30)

    def test_derivative_table(self):
        cases = {'x^3*sin(x)': 'x^3*cos(x) + 3*x^2*sin(x)', 'log(x^2+1)': '2*x/(x^2 + 1)', 'atan(x)': '1/(x^2 + 1)', 'exp(2x)': '2*exp(2*x)',
                 'sqrt(x+1)': '1/(2*(x + 1)^(1/2))', '1/x': '-1/x^2', 'tan(x)': 'tan(x)^2 + 1', 'abs(x)': 'abs(x)/x', 'log(x)': '1/x', 'x^x_': None}
        for f, want in cases.items():
            if want is None:
                continue
            self.assertTrue(run('symbolic.equal', run('symbolic.diff', f, 'x'), want), f)
        self.assertTrue(run('symbolic.equal', run('symbolic.diff', 'exp(2x)*x', 'x', 2), '4*x*exp(2*x)+4*exp(2*x)'))
        with self.assertRaises(NotCovered):
            run('symbolic.diff', 'floor(x)', 'x')

    def test_log_derivative_divides_by_the_atom(self):
        # bug caught in development: d(log u) was returned as u'/u * log(u) because the atom itself was not divided out
        self.assertEqual(run('symbolic.diff', 'log(x)', 'x'), '1/x')
        self.assertEqual(run('symbolic.diff', 'log(x)^2', 'x'), '2*log(x)/x')

    def test_derivative_against_finite_difference(self):
        h = Decimal('1e-12')
        for f in ['x^3*sin(x)', 'exp(x)/(1+x^2)', 'log(x^2+1)*atan(x)', 'sqrt(x^2+3)', 'cos(x)^3']:
            d = run('symbolic.diff', f, 'x')
            x0 = Decimal('0.7')
            fd = (self.num(f, x=x0 + h) - self.num(f, x=x0 - h)) / (2 * h)
            self.assertLess(abs(fd - self.num(d, x=x0)), Decimal('1e-15'), f)

    def test_apart(self):
        self.assertEqual(run('symbolic.apart', '1/(x^2-1)', 'x'), '1/(2*(x - 1)) - 1/(2*(x + 1))')
        self.assertTrue(run('symbolic.equal', run('symbolic.apart', '(x^3+1)/(x^2*(x-1))', 'x'), '(x^3+1)/(x^2*(x-1))'))
        self.assertTrue(run('symbolic.equal', run('symbolic.apart', '1/(x*(x^2+1))', 'x'), '1/(x*(x^2+1))'))
        with self.assertRaises(NotCovered):
            run('symbolic.apart', '1/(x^4+1)', 'x')                                   # irreducibility of the quartic is not decided

    def test_taylor_and_series(self):
        self.assertEqual(run('symbolic.series_coefficients', 'exp(x)', 'x', 0, 4), ['1', '1', '1/2', '1/6', '1/24'])
        self.assertEqual(run('symbolic.series_coefficients', 'tan(x)', 'x', 0, 5), ['0', '1', '0', '1/3', '0', '2/15'])
        self.assertEqual(run('symbolic.series_coefficients', 'log(x)', 'x', 1, 3), ['0', '1', '-1/2', '1/3'])
        self.assertEqual(run('symbolic.series_coefficients', '1/(1-x)', 'x', 0, 3), ['1', '1', '1', '1'])
        self.assertEqual(run('symbolic.taylor', 'cos(x)', 'x', 0, 4), 'x^4/24 - x^2/2 + 1')
        for f in ['sqrt(x)', '1/x', 'abs(x)', 'floor(x)', 'x^(1/3)', 'log(x)']:
            with self.assertRaises(NotCovered, msg=f):
                run('symbolic.taylor', f, 'x', 0, 3)                                  # abs: the two sides differ (caught in development)

    def test_limits(self):
        table = [('sin(x)/x', 'x', 0, '1'), ('(1-cos(x))/x^2', 'x', 0, '1/2'), ('(x^2+1)/(2*x^2-3)', 'x', 'oo', '1/2'), ('x*log(x)', 'x', 0, '0', '+'),
                 ('sqrt(x^2+x)-x', 'x', 'oo', '1/2'), ('x^2*exp(-x)', 'x', 'oo', '0'), ('(exp(x)-1)/x', 'x', 0, '1'), ('atan(x)', 'x', 'oo', 'pi/2'),
                 ('atan(x)', 'x', '-oo', '-pi/2'), ('1/x', 'x', 0, 'oo', '+'), ('1/x', 'x', 0, '-oo', '-'), ('(x^2-4)/(x-2)', 'x', 2, '4'),
                 ('(tan(x)-x)/x^3', 'x', 0, '1/3'), ('(x^3-x)/(x^2+1)', 'x', 'oo', 'oo'), ('sqrt(x^2+1)/x', 'x', '-oo', '-1')]
        for row in table:
            f, v, p, want = row[:4]
            d = row[4] if len(row) > 4 else 'both'
            self.assertEqual(run('symbolic.limit', f, v, p, d), want, f)

    def test_limit_refusals(self):
        with self.assertRaises(NotCovered):
            run('symbolic.limit', '1/x', 'x', 0)                                      # sides differ
        with self.assertRaises(NotCovered):
            run('symbolic.limit', 'exp(x)/x', 'x', 'oo')                              # exp of a growing argument
        with self.assertRaises(NotCovered):
            run('symbolic.limit', 'sin(1/x)', 'x', 0)

    def test_integrate_verified_by_derivative(self):
        for f in ['3*x^2+2*x-1', 'x*exp(x)', 'x^2*sin(x)', 'x^3*exp(-2*x)', '1/(x^2-1)', '1/(x^2+1)', 'x/(x^2+x+1)', '1/(x^2+1)^2', 'log(x)', 'x*log(x+1)',
                  'sqrt(2*x+1)', 'x*sqrt(x+1)', '1/x', '(2*x+3)/((x-1)*(x+2))', 'cos(3*x+1)', '2^x', 'x^(3/2)', '1/sqrt(x+2)']:
            F = run('symbolic.integrate', f, 'x')
            self.assertTrue(run('symbolic.equal', run('symbolic.diff', F, 'x'), f), f)
        self.assertEqual(run('symbolic.integrate', 'x*exp(x)', 'x'), 'x*exp(x) - exp(x)')
        self.assertEqual(run('symbolic.integrate', '1/x', 'x'), 'log(abs(x))')

    def test_integrate_refusals(self):
        for f in ['exp(x^2)', 'sin(x)/x', 'tan(x)', 'sin(x)^2', 'exp(x)*sin(x)', '1/(x^4+1)', '1/(x^3+x+1)', 'abs(x)']:
            with self.assertRaises(NotCovered, msg=f):
                run('symbolic.integrate', f, 'x')

    def test_definite_integrals(self):
        self.assertEqual(run('symbolic.integrate_definite', 'x^2', 'x', 0, 3), '9')
        self.assertEqual(run('symbolic.integrate_definite', '1/x', 'x', 1, 2), 'log(2)')
        self.assertEqual(run('symbolic.integrate_definite', 'exp(-x)', 'x', 0, 'oo'), '1')
        self.assertEqual(run('symbolic.integrate_definite', '1/(x^2+1)', 'x', '-oo', 'oo'), 'pi')
        self.assertEqual(run('symbolic.integrate_definite', '1/sqrt(x)', 'x', 0, 4), '4')
        self.assertEqual(run('symbolic.integrate_definite', 'sin(x)', 'x', 0, 'pi'), '2')
        self.assertEqual(run('symbolic.integrate_definite', '1/(x*(x+1))', 'x', 1, 'oo'), 'log(2)')
        self.assertEqual(run('symbolic.integrate_definite', 'x', 'x', 3, 1), '-4')

    def test_definite_refusals(self):
        for f, lo, hi in [('1/x^2', -1, 1), ('1/x', 0, 1), ('1/x', -1, 1), ('1/(x-1)^2', 0, 3), ('1/x^2', 0, 1), ('sqrt(x-1)', 0, 2)]:
            with self.assertRaises(NotCovered, msg=f):
                run('symbolic.integrate_definite', f, 'x', lo, hi)                    # a pole or branch point inside, or divergent


class SumTests(unittest.TestCase):
    def brute(self, f, lo, hi):
        return run('symbolic.sum_range', f, 'k', lo, hi)

    def test_closed_forms_match_explicit_sums(self):
        for f, lo in [('k', 1), ('k^2', 1), ('k^3', 0), ('3*k^2-k+2', 0), ('2^k', 0), ('k*2^k', 1), ('k^2*3^k', 0), ('(-1)^k', 0), ('(-1)^k*k', 1), ('(-1)^k*k^2', 1),
                      ('1/(k*(k+1))', 1), ('1/(k*(k+3))', 1), ('1/((2*k-1)*(2*k+1))', 1), ('1/(k*(k+1)*(k+2))', 1), ('3^k-2^k', 0), ('k^4', 1)]:
            closed = run('symbolic.sum_closed', f, 'k', lo, 'n')
            for n in range(lo, lo + 8):
                self.assertEqual(run('symbolic.evaluate', closed, {'n': n}), self.brute(f, lo, n), f'{f} n={n}')

    def test_infinite_sums(self):
        self.assertEqual(run('symbolic.sum_closed', '(1/2)^k', 'k', 0, 'oo'), '2')
        self.assertEqual(run('symbolic.sum_closed', 'k/3^k', 'k', 1, 'oo'), '3/4')
        self.assertEqual(run('symbolic.sum_closed', '1/(k*(k+1))', 'k', 1, 'oo'), '1')
        self.assertEqual(run('symbolic.sum_closed', '1/(k*(k+2))', 'k', 1, 'oo'), '3/4')
        for f in ['2^k', 'k', '1/k', '1/k^2']:
            with self.assertRaises(NotCovered, msg=f):
                run('symbolic.sum_closed', f, 'k', 1, 'oo')

    def test_numeric_bounds_by_evaluation(self):
        self.assertEqual(run('symbolic.sum_range', '1/(k*(k+1))', 'k', 1, 9), Fraction(9, 10))
        self.assertEqual(run('symbolic.sum_range', 'floor(k/2)', 'k', 1, 10), 25)
        self.assertEqual(run('symbolic.sum_closed', 'k^2', 'k', 1, 10), '385')
        self.assertEqual(run('symbolic.sum_range', 'k', 'k', 5, 4), 0)
        with self.assertRaises(NotCovered):
            run('symbolic.sum_range', 'k', 'k', 1, 10 ** 6)
        with self.assertRaises(BadInput):
            run('symbolic.sum_range', '1/(k-3)', 'k', 1, 5)


class SolveTests(unittest.TestCase):
    def check(self, eq, want):
        self.assertEqual(run('symbolic.solve', eq, 'x'), want, eq)

    def test_polynomials_and_rational(self):
        self.check('x^2 - 3*x + 2 = 0', ['1', '2'])
        self.check('x^3 - 6x^2 + 11x - 6 = 0', ['1', '2', '3'])
        self.check('x^2 - 2 = 0', ['-2^(1/2)', '2^(1/2)'])
        self.check('x^2 + 1 = 0', [])
        self.check('(x^2-1)/(x-1) = 0', ['-1'])                                       # x = 1 is not a solution
        self.check('x + 1/x = 5/2', ['1/2', '2'])
        self.check('x^2 - 2*x - 1 = 0', ['1 - 2^(1/2)', '1 + 2^(1/2)'])
        self.check('x + 1 = x', [])

    def test_exp_log(self):
        self.check('2^x = 8', ['3'])
        self.check('4^x - 5*2^x + 4 = 0', ['0', '2'])
        self.check('exp(x) = 5', ['log(5)'])
        self.check('3*exp(x) = -1', [])
        self.check('log(x) + log(x-1) = log(12)', ['4'])                              # x = -3 is not a solution: log of a negative number
        self.check('2*log(x) = log(9)', ['3'])

    def test_abs(self):
        self.check('|x-1| = 2', ['-1', '3'])
        self.check('|x-1| + |x+1| = 4', ['-2', '2'])
        self.check('x^2 - 3|x| + 2 = 0', ['-2', '-1', '1', '2'])
        self.check('|x| = x + 3', ['-3/2'])
        self.check('|x| = -1', [])

    def test_refusals(self):
        for eq in ['x^5 - 1 = 0', 'sin(x) = 1/2', 'x*exp(x) = 1', 'sqrt(x) = 3', 'x = x', 'x^3 + x + 1 = 0', 'a*x + 1 = 0']:
            with self.assertRaises(NotCovered, msg=eq):
                run('symbolic.solve', eq, 'x')

    def test_roots_satisfy_the_equation(self):
        for eq in ['x^2 - 2*x - 7 = 0', '2*x^2 + 3*x - 5 = 0', 'x^2 - 3|x| + 1 = 0']:
            lhs, rhs = eq.split('=')
            for r in run('symbolic.solve', eq, 'x'):
                v = run('symbolic.evaluate_numeric', lhs, {'x': r}, 40) - run('symbolic.evaluate_numeric', rhs, {'x': r}, 40)
                self.assertLess(abs(v), Decimal('1e-30'), (eq, r))


class EvaluationTests(unittest.TestCase):
    def test_exact_and_numeric(self):
        self.assertEqual(run('symbolic.evaluate', 'x^2 + 1/3', {'x': '1/2'}), Fraction(7, 12))
        self.assertEqual(run('symbolic.evaluate', 'floor(x) + abs(y)', {'x': '7/2', 'y': '-2'}), Fraction(5))
        with self.assertRaises(NotCovered):
            run('symbolic.evaluate', 'sqrt(x)', {'x': 2})
        with self.assertRaises(BadInput):
            run('symbolic.evaluate', 'x + y', {'x': 1})
        with self.assertRaises(BadInput):
            run('symbolic.evaluate', '1/x', {'x': 0})
        self.assertEqual(str(run('symbolic.evaluate_numeric', 'sin(1)', None, 10)), '0.8414709848')
        self.assertEqual(str(run('symbolic.evaluate_numeric', 'atan(1)*4', None, 20)), '3.1415926535897932385')
        self.assertEqual(str(run('symbolic.evaluate_numeric', 'exp(1)', None, 15)), '2.71828182845905')
        self.assertEqual(str(run('symbolic.evaluate_numeric', 'sin(1000000)', None, 15)), '-0.349993502171293')
        self.assertEqual(run('symbolic.substitute', 'x^2 + y', 'x', 'y + 1'), 'y^2 + 3*y + 1')


class ContractTests(unittest.TestCase):
    def test_registration_and_docs(self):
        ops = {k: v for k, v in registry().items() if k.startswith('symbolic.')}
        self.assertGreaterEqual(len(ops), 18)
        for k, v in ops.items():
            self.assertTrue(v['doc'], k)
            self.assertTrue(v['bound'], k)
            self.assertIn('->', v['doc'], k)                                           # every docstring carries an example

    def test_results_use_contract_types_and_are_deterministic(self):
        self.assertIsInstance(run('symbolic.canon', 'x+1'), str)
        self.assertIsInstance(run('symbolic.solve', 'x^2=4', 'x'), list)
        self.assertIsInstance(run('symbolic.evaluate', '1/3'), Fraction)
        self.assertIsInstance(run('symbolic.equal', 'x', 'x'), bool)
        self.assertEqual(run('symbolic.integrate', 'x*exp(x)', 'x'), run('symbolic.integrate', 'x*exp(x)', 'x'))

    def test_runaway_inputs_are_refused_not_crashes(self):
        for t in ['(' * 500 + 'x' + ')' * 500, '-' * 5000 + 'x', 'sin(' * 40 + 'x' + ')' * 40, '2^100000', '(x+y+z+w)^200']:
            with self.assertRaises((NotCovered, BadInput)):
                run('symbolic.canon', t)

    def test_standard_library_only(self):
        text = (Path(__file__).resolve().parents[1] / 'tools' / 'igc' / 'mathbackend' / 'symbolic.py').read_text(encoding='utf-8')
        for line in text.splitlines():
            if line.startswith('import ') or line.startswith('from '):
                mod = line.split()[1]
                self.assertIn(mod.split('.')[0], {'re', 'functools', 'decimal', 'fractions', 'math', 'tools'}, line)


if __name__ == '__main__':
    unittest.main()
