"""tools/igc/mathbackend/search.py unit tests. python tools/test_mathbackend_search.py (standard library only; the oracle simulation is sim_search.py)."""
import itertools
import sys
import unittest
from decimal import Decimal
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.igc.mathbackend import BadInput, NotCovered, Surd, registry, run
from tools.igc.mathbackend import search as S


class Registry(unittest.TestCase):
    def test_every_operation_is_registered_with_a_docstring_and_bound(self):
        ops = {k: v for k, v in registry().items() if k.startswith('search.')}
        self.assertGreaterEqual(len(ops), 43)
        for name, o in ops.items():
            self.assertTrue(o['doc'], name)
            self.assertTrue(o['bound'], name)
            self.assertTrue(o['kind'], name)

    def test_floats_are_refused_in_exact_operations(self):
        with self.assertRaises(BadInput):
            run('search.mean', [0.1, 0.2])
        self.assertEqual(run('search.mean', ['0.1', '0.2']), Fraction(3, 20))


class SAT(unittest.TestCase):
    def test_model_satisfies_every_clause(self):
        cl = [[1, 2, -3], [-1, 3], [2, 3], [-2, -3, 1]]
        r = run('search.sat_solve', cl)
        self.assertTrue(r['sat'])
        m = set(r['model'])
        self.assertTrue(all(any(l in m for l in c) for c in cl))

    def test_unsat_and_edges(self):
        self.assertFalse(run('search.sat_solve', [[1], [-1]])['sat'])
        self.assertFalse(run('search.sat_solve', [[]])['sat'])
        self.assertTrue(run('search.sat_solve', [])['sat'])
        self.assertTrue(run('search.sat_solve', [[1, -1]])['sat'])      # a tautology

    def test_pigeonhole_is_unsat(self):
        holes = 4
        var = lambda p, h: p * holes + h + 1
        cl = [[var(p, h) for h in range(holes)] for p in range(holes + 1)]
        cl += [[-var(p1, h), -var(p2, h)] for h in range(holes) for p1 in range(holes + 1) for p2 in range(p1 + 1, holes + 1)]
        self.assertFalse(run('search.sat_solve', cl)['sat'])

    def test_count_matches_truth_table(self):
        cl = [[1, 2], [-1, 3], [-2, -3]]
        n = 4
        brute = sum(1 for bits in itertools.product((False, True), repeat=n)
                    if all(any(bits[abs(l) - 1] == (l > 0) for l in c) for c in cl))
        self.assertEqual(run('search.sat_count', cl, n), brute)
        self.assertEqual(run('search.sat_count', [[1, 2]]), 3)
        self.assertEqual(run('search.sat_count', [], 5), 32)
        self.assertEqual(run('search.sat_count', [[1], [-1]], 3), 0)

    def test_bad_literal(self):
        with self.assertRaises(BadInput):
            run('search.sat_solve', [[0]])


class CSP(unittest.TestCase):
    def test_queens(self):
        for q, cnt in ((4, 2), (5, 10), (6, 4)):
            dom = {f'q{r}': f'0..{q - 1}' for r in range(q)}
            cons = [f'abs(q{a} - q{b}) != {b - a}' for a in range(q) for b in range(a + 1, q)]
            self.assertEqual(run('search.csp_count', dom, cons, [list(dom)]), cnt)

    def test_solutions_order_limit_and_division(self):
        r = run('search.csp_solutions', {'a': '1..4', 'b': '1..4'}, ['a / b == 2'])
        self.assertEqual(r['solutions'], [{'a': 2, 'b': 1}, {'a': 4, 'b': 2}])
        r = run('search.csp_solutions', {'a': '1..9'}, ['a % 2 == 1'], limit=2)
        self.assertEqual(r, {'solutions': [{'a': 1}, {'a': 3}], 'complete': False})

    def test_division_by_zero_is_not_a_solution(self):
        self.assertEqual(run('search.csp_count', {'a': '0..3', 'b': '0..3'}, ['a / b == 1']), 3)

    def test_optimize(self):
        r = run('search.csp_optimize', {'a': '0..5', 'b': '0..5'}, 'a*b', ['a + b <= 6'])
        self.assertEqual((r['value'], r['solution'], r['optimal_count']), (9, {'a': 3, 'b': 3}, 1))
        self.assertIsNone(run('search.csp_optimize', {'a': '0..2'}, 'a', ['a > 5'])['value'])

    def test_unsafe_text_is_refused(self):
        for bad in ('__import__("os")', 'a.real == 1', 'a == 1.5', 'open("x")', 'b == 1'):
            with self.assertRaises(BadInput, msg=bad):
                run('search.csp_count', {'a': '0..3'}, [bad])

    def test_budget(self):
        dom = {c: '0..99' for c in 'abcdef'}
        with self.assertRaises(NotCovered):
            run('search.csp_count', dom, ['a + b + c + d + e + f == 7'])


class DP(unittest.TestCase):
    def test_subset_sum(self):
        r = run('search.subset_sum', [3, 34, 4, 12, 5, 2], 9)
        self.assertEqual((r['exists'], r['count']), (True, 2))
        self.assertEqual(sum([3, 34, 4, 12, 5, 2][i] for i in r['subset']), 9)
        self.assertEqual(run('search.subset_sum', [1, -1, 2], 0)['count'], 2)        # {} and {1, -1}
        self.assertFalse(run('search.subset_sum', [2, 4, 6], 5)['exists'])
        with self.assertRaises(NotCovered):
            run('search.subset_sum', [10 ** 6] * 5, 1)

    def test_knapsack_kinds(self):
        self.assertEqual(run('search.knapsack', [1, 3, 4, 5], [1, 4, 5, 7], 7)['value'], 9)
        r = run('search.knapsack', [3, 4], [4, 5], 10, kind='unbounded')
        self.assertEqual(r['value'], 13)                                                # 2*3 + 1*4 = 10 -> 8 + 5
        r = run('search.knapsack', [3, 4], [4, 5], 10, kind='bounded', counts=[1, 1])
        self.assertEqual((r['value'], r['counts']), (9, [1, 1]))
        self.assertEqual(run('search.knapsack', [2], ['1/2'], 5, kind='unbounded')['value'], 1)

    def test_coin_change(self):
        self.assertEqual(run('search.coin_change_count', [1, 5, 10, 25, 50], 100), 292)
        self.assertEqual(run('search.coin_change_count', [1, 2], 4, ordered=True), 5)
        self.assertEqual(run('search.coin_change_count', [1, 2, 5], 5, limits=[2, 1, 1]), 1)   # only {5}: 1+1+2 = 4 is the most the small coins reach
        self.assertEqual(run('search.coin_change_count', [3], 7), 0)
        self.assertEqual(run('search.coin_change_count', [3], 0), 1)
        r = run('search.coin_change_min', [1, 5, 10, 25], 63)
        self.assertEqual((r['count'], r['coins']), (6, {1: 3, 10: 1, 25: 2}))
        self.assertIsNone(run('search.coin_change_min', [5, 10], 3)['count'])
        self.assertEqual(run('search.coin_change_min', [1, 3, 4], 6)['count'], 2)       # greedy would take 4+1+1


class LP(unittest.TestCase):
    def test_basic_and_degenerate(self):
        r = run('search.lp_solve', [3, 2], [[1, 1], [1, 3]], [4, 6])
        self.assertEqual((r['status'], r['value'], r['x']), ('optimal', 12, [4, 0]))
        r = run('search.lp_solve', [1, 1], [[1, 1], [-1, -1]], [2, -2])                 # a degenerate equality in two rows
        self.assertEqual(r['value'], 2)
        self.assertEqual(run('search.lp_solve', [1], [[1]], [1], A_eq=[[1]], b_eq=[3])['status'], 'infeasible')
        self.assertEqual(run('search.lp_solve', [1, 0], [[1, -1]], [3])['status'], 'unbounded')
        r = run('search.lp_solve', [1, 2], [[1, 1]], [10], bounds=[(None, None), (-2, 5)], maximize=False)
        self.assertEqual(r['status'], 'unbounded')                                      # x -> -inf
        r = run('search.lp_solve', [1, 2], [[1, 1], [-1, -1]], [10, 5], bounds=[(None, None), (-2, 5)], maximize=False)
        self.assertEqual(r['status'], 'optimal')
        x = r['x']
        self.assertTrue(-5 <= x[0] + x[1] <= 10 and -2 <= x[1] <= 5)

    def test_minimise_and_fractions(self):
        r = run('search.lp_solve', [1, 1], [[-2, -1], [-1, -3]], [-3, -5], maximize=False)
        self.assertEqual(r['value'], Fraction(11, 5))
        self.assertTrue(all(isinstance(v, Fraction) for v in r['x']))

    def test_ilp(self):
        r = run('search.ilp_solve', [5, 4], [[6, 4], [1, 2]], [24, 6])
        self.assertEqual(r['value'], 20)
        r = run('search.ilp_solve', [1, 1], [[2, 2]], [3])                                # relaxation 1.5, integer optimum 1
        self.assertEqual(r['value'], 1)
        self.assertEqual(run('search.ilp_solve', [1], [[2]], [1], A_eq=[[2]], b_eq=[1])['status'], 'infeasible')
        with self.assertRaises(NotCovered):
            run('search.ilp_solve', [1, 0], [[1, -1]], [3])


class Graphs(unittest.TestCase):
    def test_shortest_paths(self):
        r = run('search.shortest_path', [(1, 2, 7), (1, 3, 9), (2, 3, 1), (3, 4, 2)], 1, 4)
        self.assertEqual((r['dist'][4], r['path']), (10, [1, 2, 3, 4]))
        r = run('search.shortest_path', [('a', 'b', 4), ('b', 'c', -2), ('a', 'c', 3)], 'a', 'c')
        self.assertEqual((r['dist']['c'], r['negative_cycle']), (2, False))             # Bellman-Ford with a negative edge
        r = run('search.shortest_path', [(0, 1, 1), (1, 2, -3), (2, 1, 1)], 0)
        self.assertTrue(r['negative_cycle'])
        r = run('search.shortest_path', [(0, 1, 5)], 0, 2)
        self.assertIsNone(r['path'])
        self.assertEqual(run('search.shortest_path', [(0, 1, 2), (1, 2, 3)], 2, 0, directed=False)['dist'][0], 5)
        d = run('search.all_pairs_shortest', [(1, 2, 3), (2, 3, 4), (3, 1, 1)])['dist']
        self.assertEqual((d[1][3], d[3][2], d[2][1]), (7, 4, 5))

    def test_max_flow_min_cut(self):
        r = run('search.max_flow', [(1, 2, 3), (1, 3, 2), (2, 4, 2), (3, 4, 3), (2, 3, 1)], 1, 4)
        self.assertEqual(r['value'], 5)
        cut = sum(w for (u, v, w), k in zip([(1, 2, 3), (1, 3, 2), (2, 4, 2), (3, 4, 3), (2, 3, 1)], range(5)) if k in r['cut']['edges'])
        self.assertEqual(cut, 5)
        self.assertEqual(run('search.max_flow', [(1, 2, 2), (2, 3, 2)], 3, 1, directed=False)['value'], 2)
        self.assertEqual(run('search.max_flow', [(1, 2, 2)], 2, 1)['value'], 0)

    def test_mst(self):
        r = run('search.mst', [(1, 2, 1), (2, 3, 2), (1, 3, 3), (3, 4, 1)])
        self.assertEqual((r['weight'], r['edges'], r['connected']), (4, [0, 1, 3], True))
        r = run('search.mst', [(1, 2, 1), (3, 4, 5)])
        self.assertEqual((r['weight'], r['connected']), (6, False))
        self.assertEqual(run('search.mst', [(1, 2, 1), (2, 3, 2), (1, 3, 3)], maximize=True)['weight'], 5)

    def test_assignment(self):
        r = run('search.assignment', [[4, 1, 3], [2, 0, 5], [3, 2, 2]])
        self.assertEqual((r['cost'], r['pairs']), (5, [(0, 1), (1, 0), (2, 2)]))
        r = run('search.assignment', [[4, 1, 3], [2, 0, 5]], maximize=True)
        self.assertEqual(r['cost'], 9)                                                  # 4 + 5
        r = run('search.assignment', [[1], [2], [3]])
        self.assertEqual((r['cost'], r['pairs']), (1, [(0, 0)]))
        self.assertEqual(run('search.assignment', [['1/2', '1/3'], ['1/3', '1/2']])['cost'], Fraction(2, 3))


class Statistics(unittest.TestCase):
    def test_exact_values(self):
        self.assertEqual(run('search.mean', [1, 2, 4]), Fraction(7, 3))
        self.assertEqual(run('search.median', [1, 3, 4, 10]), Fraction(7, 2))
        self.assertEqual(run('search.mode', [1, 2, 2, 3, 3]), [2, 3])
        self.assertEqual(run('search.variance', [2, 4, 4, 4, 5, 5, 7, 9]), 4)
        self.assertEqual(run('search.variance', [1, 2, 3, 4], sample=True), Fraction(5, 3))
        self.assertEqual(run('search.stdev', [2, 4, 4, 4, 5, 5, 7, 9]), Surd(2))
        s = run('search.stdev', [1, 2])
        self.assertEqual(s * s, Surd(Fraction(1, 4)))
        self.assertEqual(run('search.covariance', [1, 2, 3], [1, 2, 4]), Fraction(1))
        self.assertEqual(run('search.correlation', [1, 2, 3], [2, 4, 6]), Surd(1))
        self.assertEqual(run('search.correlation', [1, 2, 3], [6, 4, 2]), Surd(-1))
        r = run('search.linear_regression', [1, 2, 3], [2, 4, 7])
        self.assertEqual((r['slope'], r['intercept'], r['r_squared']), (Fraction(5, 2), Fraction(-2, 3), Fraction(75, 76)))
        with self.assertRaises(BadInput):
            run('search.linear_regression', [1, 1], [1, 2])
        with self.assertRaises(BadInput):
            run('search.mean', [])

    def test_distributions(self):
        self.assertEqual(run('search.binomial_pmf', 4, '1/2', 2), Fraction(3, 8))
        self.assertEqual(run('search.binomial_cdf', 4, '1/2', 2), Fraction(11, 16))
        self.assertEqual(run('search.binomial_cdf', 4, '1/2', 9), 1)
        self.assertEqual(run('search.hypergeom_pmf', 10, 4, 3, 2), Fraction(3, 10))
        self.assertEqual(run('search.hypergeom_cdf', 10, 4, 3, 1), Fraction(2, 3))
        self.assertEqual(run('search.geometric_pmf', '1/6', 3), Fraction(25, 216))
        self.assertEqual(run('search.geometric_cdf', '1/6', 3), Fraction(91, 216))
        self.assertEqual(run('search.negbinomial_pmf', 4, '1/3', 5), Fraction(8, 243))
        self.assertEqual(run('search.poisson_partial_sum', 2, 3), Fraction(19, 3))
        with self.assertRaises(BadInput):
            run('search.binomial_pmf', 4, '3/2', 2)

    def test_poisson_and_normal_numeric(self):
        self.assertEqual(str(run('search.poisson_pmf', 2, 3, digits=10).value), '0.1804470443')
        self.assertEqual(str(run('search.poisson_cdf', 2, 3, digits=10).value), '0.8571234605')
        self.assertEqual(str(run('search.normal_cdf', 0, digits=10).value), '0.5000000000')
        self.assertEqual(str(run('search.normal_cdf', 1, digits=12).value), '0.841344746069')
        self.assertEqual(str(run('search.normal_cdf', -8, digits=8).value), '6.2209606E-16')
        self.assertEqual(str(run('search.normal_ppf', '0.975', digits=10).value), '1.959963985')
        self.assertTrue(run('search.normal_cdf', 3, digits=5).numeric)
        with self.assertRaises(NotCovered):
            run('search.normal_cdf', 50)


class NumericCalculus(unittest.TestCase):
    def test_result_shape(self):
        r = run('search.num_eval', 'sin(pi/6)', digits=12)
        self.assertEqual(len(r), 2)
        self.assertIsInstance(r.value, Decimal)
        self.assertTrue(r.numeric)
        self.assertEqual(str(r.value), '0.500000000000')

    def test_num_eval_functions(self):
        self.assertEqual(str(run('search.num_eval', 'atan(1)*4', digits=20).value), '3.1415926535897932385')
        self.assertEqual(str(run('search.num_eval', 'exp(1)', digits=15).value), '2.71828182845905')
        self.assertEqual(str(run('search.num_eval', 'x^2 + 1/x', {'x': '1/3'}, digits=8).value), '3.1111111')
        with self.assertRaises(BadInput):
            run('search.num_eval', '2x')
        with self.assertRaises(BadInput):
            run('search.num_eval', '__import__("os")')
        with self.assertRaises(NotCovered):
            run('search.num_eval', 'log(x)', {'x': -1})

    def test_derivative(self):
        self.assertEqual(str(run('search.derivative', 'x^3', 2, digits=10).value), '12.00000000')
        self.assertEqual(str(run('search.derivative', 'sin(x)', 0, digits=12, order=3).value), '-1.00000000000')
        self.assertEqual(str(run('search.derivative', 'exp(x)', 1, digits=15).value), '2.71828182845905')
        with self.assertRaises(NotCovered):                                              # a kink: central differences alone would say 0
            run('search.derivative', 'abs(x)', 0)
        with self.assertRaises(NotCovered):
            run('search.derivative', 'sqrt(x)', 0)

    def test_integrate(self):
        self.assertEqual(str(run('search.integrate', 'x^2', 0, 3, digits=10).value), '9.000000000')
        self.assertEqual(str(run('search.integrate', '1/sqrt(x)', 0, 1, digits=15).value), '2.00000000000000')
        self.assertEqual(str(run('search.integrate', 'exp(-x^2)', '-inf', 'inf', digits=15).value), '1.77245385090552')
        self.assertEqual(str(run('search.integrate', 'sin(x)', 0, 'pi', digits=12).value), '2.00000000000')
        self.assertEqual(str(run('search.integrate', 'x/(exp(x)-1)', 0, 'inf', digits=10).value), '1.644934067')
        r = run('search.integrate', 'abs(x)', -1, 2, digits=12, points=[0])
        self.assertEqual(str(r.value), '2.50000000000')
        self.assertEqual(str(run('search.integrate', 'x', 3, 0, digits=6).value), '-4.50000')    # reversed limits
        for bad in (('1/x', 0, 1), ('1/x^2', -1, 1), ('exp(x)', 0, 'inf'), ('1/(x-1/3)', 0, 1)):
            with self.assertRaises(NotCovered, msg=bad[0]):
                run('search.integrate', *bad, digits=10)

    def test_limit(self):
        self.assertEqual(str(run('search.limit', 'sin(x)/x', 0, digits=12).value), '1.00000000000')
        self.assertEqual(str(run('search.limit', '(1+1/x)^x', 'inf', digits=10).value), '2.718281828')
        self.assertEqual(run('search.limit', 'x*log(x)', 0, digits=8).value, 0)
        self.assertEqual(str(run('search.limit', 'abs(x)/x', 0, digits=8, direction='-').value), '-1.0000000')
        for bad in (('1/x', 0), ('sin(1/x)', 0), ('x', 'inf'), ('exp(x)', 'inf')):
            with self.assertRaises(NotCovered, msg=bad[0]):
                run('search.limit', *bad, digits=10)
        with self.assertRaises(NotCovered):
            run('search.limit', 'abs(x)/x', 0, digits=8, direction='both')

    def test_series(self):
        self.assertEqual(str(run('search.series_sum', '1/n^2', digits=12).value), '1.64493406685')
        self.assertEqual(str(run('search.series_sum', '(-1)^n/n', digits=12).value), '-0.693147180560')
        self.assertEqual(str(run('search.series_sum', '1/2^n', 0, digits=12).value), '2.00000000000')
        self.assertEqual(str(run('search.series_sum', '1/(n*(n+1))', 1, 99, digits=12).value), '0.990000000000')
        for t in ('1/n', 'n', '2^n', '1/sqrt(n)'):
            with self.assertRaises(NotCovered, msg=t):
                run('search.series_sum', t, digits=10)

    def test_findroot(self):
        r = run('search.findroot', 'x^2 - 2', 1, 2, digits=15)
        self.assertEqual(str(r.value), '1.41421356237310')
        self.assertLess(r.error, Decimal('1E-12'))
        self.assertEqual(str(run('search.findroot', 'cos(x) - x', 0, 1, digits=12).value), '0.739085133215')
        self.assertEqual(run('search.findroot', 'x', -1, 1, digits=6).value, 0)
        with self.assertRaises(BadInput):
            run('search.findroot', 'x^2 + 1', -1, 2)
        with self.assertRaises(NotCovered):                                              # a sign change at a pole
            run('search.findroot', 'tan(x)', 1, 2)

    def test_ode(self):
        r = run('search.ode_ivp', 'y', 1, 0, 1, digits=10)
        self.assertEqual(str(r.value[0]), '2.718281828')
        r = run('search.ode_ivp', ['y2', '-y1'], [0, 1], 0, 3, digits=10)
        self.assertEqual([str(v) for v in r.value], ['0.1411200081', '-0.9899924966'])
        r = run('search.ode_ivp', 'y', 1, 1, 0, digits=8)                                # backwards
        self.assertEqual(str(r.value[0]), '0.36787944')
        with self.assertRaises(NotCovered):
            run('search.ode_ivp', 'y^2', 1, 0, 2)                                        # blows up at t = 1

    def test_module_has_no_third_party_imports(self):
        src = (Path(S.__file__)).read_text(encoding='utf-8')
        import re
        mods = set(re.findall(r'^(?:from|import) ([A-Za-z_][\w.]*)', src, re.M))
        self.assertTrue(mods <= {'ast', 'heapq', 're', 'collections', 'decimal', 'fractions', 'functools', 'math', 'typing', 'tools.igc.mathbackend'}, mods)


if __name__ == '__main__':
    unittest.main()
