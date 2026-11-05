"""tools/igc/mathbackend/algebra unit tests: hand-computed values and certificates, standard library only. python tools/test_mathbackend_algebra.py"""
import sys
import unittest
from decimal import Decimal
from fractions import Fraction as F
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.igc.mathbackend import BadInput, NotCovered, Surd, registry, run
from tools.igc.mathbackend import algebra as A


def r(name, *a, **k):
    return run('algebra.' + name, *a, **k)


class PolyTests(unittest.TestCase):
    def test_parse_and_print(self):
        self.assertEqual(r('poly_from_str', '(x+1)^2 - 3x/2'), [1, F(1, 2), 1])
        self.assertEqual(r('poly_from_str', '\\frac{x^{2}}{2} + 3 x'), [F(1, 2), 3, 0])
        self.assertEqual(r('poly_from_str', '2(x-1)(x+1)'), [2, 0, -2])
        self.assertEqual(r('poly_from_str', '0'), [])
        self.assertEqual(r('poly_to_str', [1, 0, F(-1, 2)]), 'x^2 - 1/2')
        for c in ([3, F(-1, 3), 0, 7], [-1, 0, 5], [F(1, 2)]):
            self.assertEqual(r('poly_from_str', r('poly_to_str', c)), c)
        with self.assertRaises(NotCovered):
            r('poly_from_str', 'x*y + 1')
        with self.assertRaises(NotCovered):
            r('poly_from_str', '1/x')
        with self.assertRaises(BadInput):
            r('poly_from_str', '(x+1')

    def test_arithmetic(self):
        self.assertEqual(r('poly_mul', [1, 1], [1, -1]), [1, 0, -1])
        self.assertEqual(r('poly_add', [1, 2], [1, -2]), [2, 0])
        self.assertEqual(r('poly_sub', [1, 2], [1, 2]), [])
        self.assertEqual(r('poly_pow', [1, 1], 4), [1, 4, 6, 4, 1])
        self.assertEqual(r('poly_divmod', [1, 0, 0, -1], [1, -1]), ([1, 1, 1], []))
        self.assertEqual(r('poly_divmod', [1, 0, 1], [1, 1]), ([1, -1], [2]))
        with self.assertRaises(BadInput):
            r('poly_divmod', [1], [])
        self.assertEqual(r('poly_gcd', [1, 0, -1], [1, 3, 2]), [1, 1])
        self.assertEqual(r('poly_lcm', [1, -1], [1, 1]), [1, 0, -1])
        s, t, g = r('poly_gcdex', [1, 0, -1], [1, 3, 2])
        self.assertEqual(r('poly_add', r('poly_mul', s, [1, 0, -1]), r('poly_mul', t, [1, 3, 2])), g)
        self.assertEqual(g, [1, 1])
        self.assertEqual(r('poly_compose', [1, 0, 1], [1, 1]), [1, 2, 2])
        self.assertEqual(r('poly_derivative', [1, 0, 0, 1]), [3, 0, 0])
        self.assertEqual(r('poly_derivative', [1, 0, 0, 1], 3), [6])
        self.assertEqual(r('poly_antiderivative', [3, 0, 0]), [1, 0, 0, 0])
        self.assertEqual(r('poly_integral', [1, 0, 0], 0, 3), 9)
        self.assertEqual(r('poly_eval', [1, 0, -2], 3), 7)
        self.assertEqual(r('poly_eval', [1, 0, -2], Surd(0, 1, 2)), 0)
        self.assertEqual(r('poly_taylor_shift', [1, 0, 0], 2), [1, 4, 4])
        self.assertEqual(r('poly_content_primitive', [2, 4, -6]), (2, [1, 2, -3]))
        self.assertEqual(r('poly_content_primitive', [-2, 4]), (-2, [1, -2]))
        self.assertEqual(r('poly_squarefree_part', [1, -2, 1]), [1, -1])
        self.assertEqual(r('poly_squarefree_decomposition', [2, -6, 6, -2]), (2, [([1, -1], 3)]))
        self.assertEqual(r('poly_from_roots', [1, 2, 2]), [1, -5, 8, -4])
        self.assertEqual(r('poly_transform_roots', [1, -3, 2], 'shift', 1), [1, -5, 6])
        self.assertEqual(r('poly_transform_roots', [1, -3, 2], 'scale', 2), [1, -6, 8])
        self.assertEqual(r('poly_transform_roots', [1, -3, 2], 'reciprocal'), [2, -3, 1])

    def test_bounds(self):
        with self.assertRaises(NotCovered):
            r('poly_pow', [1, 1], 5000)
        with self.assertRaises(BadInput):
            r('poly_add', [0.5, 1], [1])


class RootTests(unittest.TestCase):
    def test_rational_roots(self):
        self.assertEqual(r('rational_roots', [1, -5, 8, -4]), [(1, 1), (2, 2)])
        self.assertEqual(r('rational_roots', [2, -1, 0]), [(0, 1), (F(1, 2), 1)])
        self.assertEqual(r('rational_roots', [1, 0, 1]), [])

    def test_sturm(self):
        self.assertEqual(r('count_real_roots', [1, 0, -2]), 2)
        self.assertEqual(r('count_real_roots', [1, 0, -2], 0, 1), 0)
        self.assertEqual(r('count_real_roots', [1, 0, -2], 0, 2), 1)
        self.assertEqual(r('count_real_roots', [1, -3, 3, -1]), 1)
        self.assertEqual(r('count_real_roots', [1, -3, 3, -1], with_multiplicity=True), 3)
        self.assertEqual(r('count_real_roots', [1, 0, 1]), 0)
        self.assertEqual(r('count_real_roots', [1, -6, 11, -6], 1, 3), 2)         # roots 1,2,3 : (1,3] holds 2 and 3
        seq = r('sturm_sequence', [1, 0, -2])
        self.assertEqual(seq, [[1, 0, -2], [1, 0], [1]])

    def test_isolate_and_refine(self):
        iv = r('isolate_real_roots', [1, 0, -2])
        self.assertEqual(len(iv), 2)
        (l1, h1, m1), (l2, h2, m2) = iv
        self.assertTrue(l1 < -1 and h1 < 0 and l1 * l1 > 2 > h1 * h1 or l1 * l1 > 2 > h1 * h1)
        self.assertTrue(l2 * l2 < 2 < h2 * h2)
        roots = r('real_roots', [1, 0, -2], 20)
        self.assertEqual(str(roots[1]['root']), '1.41421356237309504880')
        self.assertTrue(roots[1]['numeric'] and not roots[1]['exact'])
        exact = r('real_roots', [1, -3, 2])
        self.assertEqual([(d['root'], d['exact']) for d in exact], [(1, True), (2, True)])
        self.assertEqual(r('real_roots', [1, 0, 0])[0]['multiplicity'], 2)

    def test_complex_roots(self):
        zs = r('complex_roots', [1, 0, 1], 10)
        self.assertEqual([(str(a), str(b)) for a, b in zs], [('0E-10', '-1.0000000000'), ('0E-10', '1.0000000000')])
        z3 = r('complex_roots', [1, 0, 0, -1], 12)
        self.assertEqual(len(z3), 3)
        self.assertIn((Decimal('1.000000000000'), Decimal('0.000000000000')), z3)
        self.assertEqual(str(max(b for _, b in z3)), '0.866025403784')

    def test_quadratic_and_solve(self):
        (a1, b1), (a2, b2) = r('quadratic_roots', 1, -2, -1)
        self.assertEqual((a1, a2), (Surd(1, -1, 2), Surd(1, 1, 2)))
        self.assertEqual((b1, b2), (Surd(0), Surd(0)))
        (a1, b1), (a2, b2) = r('quadratic_roots', 1, 0, 1)
        self.assertEqual((a1, b1, a2, b2), (Surd(0), Surd(-1), Surd(0), Surd(1)))
        (a1, b1), _ = r('quadratic_roots', 1, 1, 1)
        self.assertEqual((a1, b1), (Surd(F(-1, 2)), Surd(0, F(-1, 2), 3)))
        s = r('solve_polynomial', [1, 0, -3, 2])
        self.assertEqual([(f['poly'], f['multiplicity']) for f in s['factors']], [([1, -1], 2), ([1, 2], 1)])
        self.assertTrue(s['all_exact'])
        s = r('solve_polynomial_str', 'x^3 = 2')
        self.assertFalse(s['all_exact'])
        self.assertEqual(len(s['factors'][0]['real_intervals']), 1)
        s = r('solve_polynomial_str', 'x^2 = 2x + 3')
        self.assertEqual(sorted(f['roots'][0][0] for f in s['factors']), [-1, 3])


class FactorTests(unittest.TestCase):
    def test_factor(self):
        self.assertEqual(r('factor_z', [1, 0, -1])['factors'], [([1, -1], 1), ([1, 1], 1)])
        res = r('factor_z', [1, 0, 0, 0, 4])                      # x^4 + 4 = (x^2 - 2x + 2)(x^2 + 2x + 2)
        self.assertEqual([f for f, _ in res['factors']], [[1, -2, 2], [1, 2, 2]])
        res = r('factor_z', [1, 0, -10, 0, 1])                    # irreducible over Q although reducible modulo every prime
        self.assertEqual(len(res['factors']), 1)
        res = r('factor_z', [2, 4, 2, 0])                         # 2x(x+1)^2
        self.assertEqual(res['content'], 2)
        self.assertEqual(res['factors'], [([1, 0], 1), ([1, 1], 2)])
        res = r('factor_z', [1] + [0] * 11 + [-1])                # x^12 - 1 = prod of cyclotomics 1,2,3,4,6,12
        self.assertEqual([len(f) - 1 for f, _ in res['factors']], [1, 1, 2, 2, 2, 4])
        self.assertTrue(r('is_irreducible_z', [1, 1, 1]))
        self.assertFalse(r('is_irreducible_z', [1, 0, -1]))
        self.assertFalse(r('is_irreducible_z', [5]))

    def test_cyclotomic(self):
        self.assertEqual(r('cyclotomic', 1), [1, -1])
        self.assertEqual(r('cyclotomic', 6), [1, -1, 1])
        self.assertEqual(r('cyclotomic', 12), [1, 0, -1, 0, 1])
        self.assertEqual(r('cyclotomic', 105)[:9], [1, 1, 1, 0, 0, -1, -1, -2, -1])

    def test_resultant_discriminant(self):
        self.assertEqual(r('resultant', [1, 0, -1], [1, -2]), 3)
        self.assertEqual(r('discriminant', [1, 0, 0, 1]), -27)
        self.assertEqual(r('discriminant', [1, 2, 1]), 0)
        self.assertEqual(r('discriminant', [2, 3, 5]), 9 - 40)

    def test_symmetric(self):
        self.assertEqual(r('vieta', [1, -6, 11, -6]), [6, 11, 6])
        self.assertEqual(r('power_sums', [1, -3, 2], 4), [3, 5, 9, 17])
        self.assertEqual(r('power_sums', [1, -3, 2], 2, True), [F(3, 2), F(5, 4)])
        self.assertEqual(r('roots_power_poly', [1, -3, 2], 2), [1, -5, 4])


class InterpolationTests(unittest.TestCase):
    def test_all(self):
        self.assertEqual(r('interpolate', [(0, 1), (1, 3), (2, 7)]), [1, 1, 1])
        self.assertEqual(r('interpolate_eval', [(0, 1), (1, 3), (2, 7)], 3), 13)
        self.assertEqual(r('finite_differences', [1, 4, 9, 16]), [[1, 4, 9, 16], [3, 5, 7], [2, 2], [0]])
        with self.assertRaises(BadInput):
            r('interpolate', [(1, 1), (1, 2)])


class ModTests(unittest.TestCase):
    def test_all(self):
        self.assertEqual(r('polymod_mul', [1, 1], [1, 1], 2), [1, 0, 1])
        self.assertEqual(r('polymod_divmod', [1, 0, 0, 1], [1, 1], 5), ([1, 4, 1], []))
        self.assertEqual(r('polymod_gcd', [1, 0, 0, 1], [1, 1], 7), [1, 1])
        self.assertEqual(r('polymod_powmod', [1, 0], 10, [1, 0, 1], 3), [2])          # x^10 = (x^2)^5 = (-1)^5 = -1 = 2
        self.assertEqual(r('polymod_inverse', [1, 0], [1, 1, 1], 2), [1, 1])
        self.assertEqual(r('polymod_roots', [1, 0, -1], 7), [(1, 1), (6, 1)])
        self.assertEqual(r('polymod_factor', [1, 0, 1], 5), (1, [([1, 2], 1), ([1, 3], 1)]))
        self.assertEqual(r('polymod_factor', [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], 2)[1], [([1, 1], 16)])    # (x+1)^16 over F_2
        self.assertTrue(r('polymod_irreducible', [1, 1, 1], 2))
        self.assertFalse(r('polymod_irreducible', [1, 0, 1], 2))
        with self.assertRaises(BadInput):
            r('polymod_mul', [1], [1], 4)


class MatrixTests(unittest.TestCase):
    A = [[1, 2], [3, 4]]

    def test_basic(self):
        self.assertEqual(r('mat_mul', self.A, [[0, 1], [1, 0]]), [[2, 1], [4, 3]])
        self.assertEqual(r('mat_trace', self.A), 5)
        self.assertEqual(r('mat_det', self.A), -2)
        self.assertEqual(r('mat_det', [[2, 0, 1], [1, 3, 2], [1, 1, 1]]), 2 * (3 - 2) - 0 + 1 * (1 - 3))
        self.assertEqual(r('mat_inverse', self.A), [[-2, 1], [F(3, 2), F(-1, 2)]])
        with self.assertRaises(BadInput):
            r('mat_inverse', [[1, 2], [2, 4]])
        self.assertEqual(r('mat_rank', [[1, 2], [2, 4]]), 1)
        self.assertEqual(r('mat_rref', self.A), ([[1, 0], [0, 1]], [0, 1]))
        self.assertEqual(r('mat_nullspace', [[1, 2], [2, 4]]), [[-2, 1]])
        self.assertEqual(r('mat_charpoly', self.A), [1, -5, -2])
        self.assertEqual(r('mat_minpoly', [[2, 0], [0, 2]]), [1, -2])
        self.assertEqual(r('mat_minpoly', [[2, 1], [0, 2]]), [1, -4, 4])
        self.assertEqual(r('mat_pow', [[1, 1], [1, 0]], 10), [[89, 55], [55, 34]])
        self.assertEqual(r('mat_pow', self.A, -1), r('mat_inverse', self.A))
        self.assertEqual(r('mat_pow', [[1, 1], [1, 0]], 10 ** 18, mod=1000000007)[0][1], 209783453)    # F(10^18) mod 1e9+7
        with self.assertRaises(NotCovered):
            r('mat_pow', [[2, 1], [1, 1]], 10 ** 6)
        with self.assertRaises(BadInput):
            r('mat_det', [[1, 2, 3], [4, 5, 6]])

    def test_solve(self):
        s = r('mat_solve', [[1, 1], [1, -1]], [3, 1])
        self.assertEqual((s['status'], s['particular'], s['nullspace']), ('unique', [2, 1], []))
        self.assertEqual(r('mat_solve', [[1, 1], [2, 2]], [1, 3])['status'], 'none')
        s = r('mat_solve', [[1, 1], [2, 2]], [1, 2])
        self.assertEqual((s['status'], s['particular'], s['nullspace']), ('infinite', [1, 0], [[-1, 1]]))
        s = r('solve_linear_str', ['x + y = 3', 'x - y = 1'])
        self.assertEqual(s['solution'], {'x': 2, 'y': 1})
        s = r('solve_linear_str', ['2(x + y) = 6', '3x + 3y = 9'])
        self.assertEqual((s['status'], s['free']), ('infinite', ['y']))
        self.assertEqual(r('solve_linear_str', ['x = 1', 'x = 2'])['status'], 'none')

    def test_lu_eigen(self):
        res = r('mat_lu', [[0, 1], [1, 0]])
        self.assertEqual((res['P'], res['L'], res['U']), ([[0, 1], [1, 0]], [[1, 0], [0, 1]], [[1, 0], [0, 1]]))
        M = [[2, 1, 1], [4, 3, 3], [8, 7, 9]]
        res = r('mat_lu', M)
        self.assertEqual(r('mat_mul', res['P'], M), r('mat_mul', res['L'], res['U']))
        e = r('mat_eigen', [[2, 0], [0, 3]])
        self.assertEqual([(d['value'], d['algebraic'], d['geometric']) for d in e['eigenvalues']], [(2, 1, 1), (3, 1, 1)])
        self.assertTrue(e['diagonalizable_over_Q'])
        e = r('mat_eigen', [[2, 1], [0, 2]])
        self.assertFalse(e['diagonalizable_over_Q'])
        e = r('mat_eigen', [[0, -1], [1, 0]])
        self.assertEqual((e['eigenvalues'], e['rest']), ([], [1, 0, 1]))


class RecurrenceTests(unittest.TestCase):
    def test_nth(self):
        self.assertEqual(r('recurrence_nth', [1, 1], [0, 1], 10), 55)
        self.assertEqual(r('recurrence_nth', [1, 1], [0, 1], 90), 2880067194370816120)
        self.assertEqual(r('recurrence_nth', [1, 1], [0, 1], 10 ** 18, mod=10 ** 9 + 7), 209783453)
        self.assertEqual(r('recurrence_nth', [2], [1], 5, const=1), 63)             # a_n = 2a_(n-1) + 1: 1,3,7,15,31,63
        self.assertEqual(r('recurrence_nth', [3, -2], [1, 3], 6), 2 ** 7 - 1)
        self.assertEqual(r('recurrence_char_poly', [1, 1]), [1, -1, -1])
        with self.assertRaises(NotCovered):
            r('recurrence_nth', [1, 1], [0, 1], 10 ** 7)

    def test_closed_form(self):
        f = r('recurrence_closed_form', [1, 1], [0, 1])
        for n in range(30):
            self.assertEqual(A.closed_form_value(f, n), r('recurrence_nth', [1, 1], [0, 1], n))
        f = r('recurrence_closed_form', [4, -4], [1, 2])                              # double root 2
        self.assertEqual([(t['root'], t['multiplicity']) for t in f['terms']], [(2, 2)])
        for n in range(20):
            self.assertEqual(A.closed_form_value(f, n), r('recurrence_nth', [4, -4], [1, 2], n))
        f = r('recurrence_closed_form', [4, 0], [1, 3])                               # root 0 once: valid from n = 1
        self.assertEqual(f['valid_from'], 1)
        with self.assertRaises(NotCovered):
            r('recurrence_closed_form', [0, -1], [1, 0])                              # complex roots

    def test_guess(self):
        g = r('recurrence_guess', [1, 1, 2, 3, 5, 8, 13])
        self.assertEqual((g['order'], g['coeffs']), (2, [1, 1]))
        g = r('recurrence_guess', [1, 3, 7, 15, 31, 63, 127])
        self.assertEqual((g['order'], g['coeffs']), (2, [3, -2]))
        with self.assertRaises(NotCovered):
            r('recurrence_guess', [1, 1, 2])
        self.assertEqual(r('recurrence_guess', [0, 0, 0, 0])['order'], 0)


class RegistryTests(unittest.TestCase):
    def test_registered_with_docstring(self):
        ops = {k: v for k, v in registry().items() if k.startswith('algebra.')}
        self.assertGreaterEqual(len(ops), 60)
        for k, v in ops.items():
            self.assertTrue(v['doc'], k)
            self.assertTrue(v['kind'] and v['bound'], k)

    def test_backend_imports_only_stdlib(self):
        import ast
        src = (Path(__file__).resolve().parent / 'igc' / 'mathbackend' / 'algebra.py').read_text(encoding='utf-8')
        mods = set()
        for node in ast.walk(ast.parse(src)):
            if isinstance(node, ast.Import):
                mods.update(a.name.split('.')[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom):
                mods.add(node.module.split('.')[0])
        self.assertTrue(mods <= set(sys.stdlib_module_names) | {'tools'}, mods)


if __name__ == '__main__':
    unittest.main()
