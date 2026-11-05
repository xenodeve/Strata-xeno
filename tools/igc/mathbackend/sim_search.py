"""Simulation of tools/igc/mathbackend/search.py against independent oracles, no model involved.

    PYTHONPATH=. PYTHONUTF8=1 <venv python with sympy and mpmath> -m tools.igc.mathbackend.sim_search --seed 1 --n 40

Each case is (op, args, kw, oracle, oracle_name, cmp). The oracles are SymPy / mpmath where the backend did not copy their code (search.py copies none), an independent brute
force written in this file (truth tables, itertools, subset and permutation enumeration, minimum cut by subsets, vertex enumeration), or a certificate that is checked
exactly (a SAT model satisfies every clause, a flow obeys conservation, a root is bracketed by a sign change evaluated by mpmath). The oracle is named in every case.
A case can expect a refusal (REFUSE: the right answer is NotCovered, e.g. a divergent integral) or allow either (EITHER: an answer must be right, a refusal is accepted).
Report per operation: generated, answered, agree, refused (with the reasons), disagree (must be 0), errors (an unexpected exception, counted as disagree), and for the
numeric operations how many errors estimates held. Written to build-mathbackend/search.json.
"""
import argparse
import itertools
import json
import math
import random
import re
import sys
import time
from decimal import Decimal, getcontext, localcontext
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable, NamedTuple

from tools.igc.mathbackend import BadInput, NotCovered, Surd, registry, run
from tools.igc.mathbackend.search import Numeric

import mpmath as mp
import statistics
import sympy
from sympy import Rational as SR

mp.mp.dps = 70
getcontext().prec = 150      # the comparisons below must not round
REFUSE = 'REFUSE'      # the right answer is a refusal
BAD = 'BAD'            # the right answer is BadInput
EITHER = 'EITHER'      # a refusal is accepted; an answer must match `oracle` (wrapped as (EITHER, oracle))


class Case(NamedTuple):
    op: str
    args: tuple
    kw: dict
    oracle: Any
    oracle_name: str
    cmp: Callable


# ----------------------------------------------------------------------------------------------------------------------------------------
# comparison helpers
# ----------------------------------------------------------------------------------------------------------------------------------------

def eq(res, oracle):
    return res == oracle


def mpd(x, n=60):
    """mpmath number -> Decimal."""
    return Decimal(mp.nstr(x, n, min_fixed=-10 ** 9, max_fixed=10 ** 9))


def quantum(v):
    return Decimal(1).scaleb(v.as_tuple().exponent)


def num_cmp(res, oracle):
    """res is a Numeric; oracle a Decimal (or a tuple of them). Agree when within one unit of the last digit returned; also returns whether the error estimate
    covered the difference (half a unit plus the estimate)."""
    if not isinstance(res, Numeric) or not res.numeric:
        return False
    vals = res.value if isinstance(res.value, tuple) else (res.value,)
    ors = oracle if isinstance(oracle, tuple) else (oracle,)
    if len(vals) != len(ors):
        return False
    ok, honest = True, True
    for v, o in zip(vals, ors):
        q = quantum(v)
        d = abs(v - o)
        if d > q:
            ok = False
        if d > res.error + q / 2:
            honest = False
    return (ok, honest)


# ----------------------------------------------------------------------------------------------------------------------------------------
# SAT
# ----------------------------------------------------------------------------------------------------------------------------------------

def _truth_table(nv, clauses):
    """(count of satisfying assignments, one model) by enumerating all 2^nv assignments with bit operations."""
    cnt, model = 0, None
    cl = [[(abs(l) - 1, l > 0) for l in c] for c in clauses]
    for bits in range(1 << nv):
        if all(any(((bits >> v) & 1) == (1 if pos else 0) for v, pos in c) for c in cl):
            cnt += 1
            if model is None:
                model = bits
    return cnt, model


def _model_ok(clauses, model):
    s = set(model)
    return all(any(l in s for l in c) for c in clauses)


def _rand_cnf(rng, nv, m, k=3):
    return [[rng.choice((-1, 1)) * v for v in rng.sample(range(1, nv + 1), min(k, nv))] for _ in range(m)]


def gen_sat_solve(rng, n):
    out = []
    for i in range(n):
        kind = i % 5
        if kind <= 2:                                               # random 3-SAT near and around the threshold, truth table decides
            nv = rng.randint(3, 14)
            cl = _rand_cnf(rng, nv, int(nv * rng.uniform(2.0, 6.0)))
            cnt, _ = _truth_table(nv, cl)
            out.append(Case('search.sat_solve', (cl, nv), {}, cnt > 0, 'truth table (2^n enumeration)',
                            lambda r, o, cl=cl: r['sat'] == o and (not o or _model_ok(cl, r['model']))))
        elif kind == 3:                                              # planted solution: certificate only
            nv = rng.randint(20, 60)
            sol = [rng.choice((-1, 1)) * v for v in range(1, nv + 1)]
            cl = []
            while len(cl) < int(nv * 4.0):
                c = [rng.choice((-1, 1)) * v for v in rng.sample(range(1, nv + 1), 3)]
                if any(l in set(sol) for l in c):
                    cl.append(c)
            out.append(Case('search.sat_solve', (cl, nv), {}, True, 'certificate: the returned model satisfies every clause (instance built around a planted solution)',
                            lambda r, o, cl=cl: r['sat'] is True and _model_ok(cl, r['model'])))
        else:                                                        # pigeonhole: n+1 pigeons in n holes is unsatisfiable (a theorem), plus edge cases
            if i // 5 % 2 == 0:
                holes = rng.randint(2, 5)
                pig = holes + 1
                var = lambda p, h: p * holes + h + 1
                cl = [[var(p, h) for h in range(holes)] for p in range(pig)]
                cl += [[-var(p1, h), -var(p2, h)] for h in range(holes) for p1 in range(pig) for p2 in range(p1 + 1, pig)]
                out.append(Case('search.sat_solve', (cl,), {}, False, 'pigeonhole principle (n+1 pigeons, n holes is unsatisfiable)',
                                lambda r, o: r['sat'] is False))
            else:
                edge = [([], None), ([[]], False), ([[1, -1]], True), ([[1], [1, 2], [1, 2, 3]], True), ([[1, 1, 2], [-2, -2]], True)][rng.randint(0, 4)]
                cl, ex = edge
                if ex is None:
                    out.append(Case('search.sat_solve', (cl, 3), {}, True, 'by definition: no clauses is satisfiable',
                                    lambda r, o: r['sat'] is True and len(r['model']) == 3))
                else:
                    out.append(Case('search.sat_solve', (cl,), {}, ex, 'by definition (empty clause / tautology / duplicates)', lambda r, o: r['sat'] == o))
    return out


def gen_sat_count(rng, n):
    out = []
    for i in range(n):
        nv = rng.randint(2, 15)
        extra = rng.choice((0, 0, 1, 3))
        cl = _rand_cnf(rng, nv, rng.randint(0, int(nv * 3)), k=rng.choice((1, 2, 3)))
        cnt, _ = _truth_table(nv + extra, cl)
        out.append(Case('search.sat_count', (cl, nv + extra), {}, cnt, 'truth table (2^n enumeration)', eq))
    return out


# ----------------------------------------------------------------------------------------------------------------------------------------
# constraint search
# ----------------------------------------------------------------------------------------------------------------------------------------

def _brute_csp(domains, constraints, alldiff=()):
    names = list(domains)
    doms = []
    for nm in names:
        d = domains[nm]
        if isinstance(d, str):
            lo, hi = map(int, d.split('..'))
            doms.append(list(range(lo, hi + 1)))
        else:
            doms.append(sorted(set(d)))
    codes = [compile(c.replace('^', '**'), 'c', 'eval') for c in constraints]
    sols = []
    for vals in itertools.product(*doms):
        env = dict(zip(names, vals))
        if any(len({env[g] for g in grp}) != len(grp) for grp in alldiff):
            continue
        if all(eval(c, {'__builtins__': {}, 'abs': abs, 'min': min, 'max': max}, env) for c in codes):
            sols.append(vals)
    return names, sols


_CSP_TEMPLATES = ['a + b == c', 'a * b < c + d', '(a + b) % 3 == 0', 'a != b', 'a * a + b * b == c * c', 'a < b', 'b < c', 'a + b + c + d == 12', 'abs(a - b) >= 2',
                  'c % 2 == 0', 'a * b == 12', 'max(a, b) < c', 'a ^ 2 + b > c * 2', 'a + 2 * b == c + d', 'min(c, d) > 2', 'd - a == 3']


def gen_csp_count(rng, n):
    out = []
    queens_known = {4: 2, 5: 10, 6: 4, 7: 40}
    for i in range(n):
        if i % 10 == 9:
            q = rng.choice((4, 5, 6, 7))
            dom = {f'q{r}': f'0..{q - 1}' for r in range(q)}
            cons = [f'abs(q{r1} - q{r2}) != {r2 - r1}' for r1 in range(q) for r2 in range(r1 + 1, q)]
            out.append(Case('search.csp_count', (dom, cons, [list(dom)]), {}, queens_known[q], f'known count of {q}-queens solutions (OEIS A000170)', eq))
            continue
        nv = rng.randint(2, 4)
        names = 'abcd'[:nv]
        dom = {nm: (f'{rng.randint(0, 2)}..{rng.randint(5, 9)}' if rng.random() < 0.6 else sorted(rng.sample(range(-3, 10), rng.randint(3, 8)))) for nm in names}
        cons = [t for t in rng.sample(_CSP_TEMPLATES, rng.randint(1, 3)) if set(re.findall(r'[a-d]', t)) <= set(names)]
        ad = [list(names)] if rng.random() < 0.3 else []
        _, sols = _brute_csp(dom, cons, ad)
        out.append(Case('search.csp_count', (dom, cons, ad), {}, len(sols), 'itertools.product enumeration with Python eval of the same constraint text', eq))
    return out


def gen_csp_solutions(rng, n):
    out = []
    for i in range(n):
        if i % 20 == 19:
            dom = {'S': '1..9', 'E': '0..9', 'N': '0..9', 'D': '0..9', 'M': '1..9', 'O': '0..9', 'R': '0..9', 'Y': '0..9'}
            cons = ['1000*S + 100*E + 10*N + D + 1000*M + 100*O + 10*R + E == 10000*M + 1000*O + 100*N + 10*E + Y']
            out.append(Case('search.csp_solutions', (dom, cons, [list(dom)]), {},
                            [{'S': 9, 'E': 5, 'N': 6, 'D': 7, 'M': 1, 'O': 0, 'R': 8, 'Y': 2}], 'the unique solution of SEND + MORE = MONEY (9567 + 1085 = 10652)',
                            lambda r, o: r['complete'] and r['solutions'] == o))
            continue
        nv = rng.randint(2, 4)
        names = 'abcd'[:nv]
        dom = {nm: f'{rng.randint(0, 2)}..{rng.randint(4, 8)}' for nm in names}
        cons = [t for t in rng.sample(_CSP_TEMPLATES, rng.randint(1, 3)) if set(re.findall(r'[a-d]', t)) <= set(names)]
        lim = rng.choice((1, 3, 5, 1000))
        nm_, sols = _brute_csp(dom, cons)
        exp = [dict(zip(nm_, s)) for s in sols]
        out.append(Case('search.csp_solutions', (dom, cons), {'limit': lim}, (exp[:lim], len(exp) <= lim),
                        'itertools.product enumeration with Python eval (lexicographic order)',
                        lambda r, o: r['solutions'] == o[0] and r['complete'] == o[1]))
    return out


def gen_csp_optimize(rng, n):
    out = []
    objs = ['a * b', 'a + 2 * b - c', 'a * a - b', 'max(a, b) * c', 'a * b * c - d']
    for i in range(n):
        nv = rng.randint(3, 4)
        names = 'abcd'[:nv]
        dom = {nm: f'{rng.randint(0, 2)}..{rng.randint(4, 7)}' for nm in names}
        obj = rng.choice([o for o in objs if set(re.findall(r'[a-d]', o)) <= set(names)])
        cons = [t for t in rng.sample(_CSP_TEMPLATES, rng.randint(0, 2)) if set(re.findall(r'[a-d]', t)) <= set(names)]
        mx = rng.random() < 0.5
        nm_, sols = _brute_csp(dom, cons)
        oc = compile(obj, 'o', 'eval')
        vals = [eval(oc, {'__builtins__': {}, 'max': max}, dict(zip(nm_, s))) for s in sols]
        if vals:
            best = max(vals) if mx else min(vals)
            first = sols[vals.index(best)]
            exp = {'value': best, 'solution': dict(zip(nm_, first)), 'optimal_count': vals.count(best)}
        else:
            exp = {'value': None, 'solution': None, 'optimal_count': 0}
        out.append(Case('search.csp_optimize', (dom, obj, cons), {'maximize': mx}, exp, 'itertools.product enumeration, max/min over feasible points', eq))
    return out


# ----------------------------------------------------------------------------------------------------------------------------------------
# dynamic programming
# ----------------------------------------------------------------------------------------------------------------------------------------

def gen_subset_sum(rng, n):
    out = []
    for i in range(n):
        k = rng.randint(1, 14)
        vals = [rng.randint(-15, 40) if rng.random() < 0.3 else rng.randint(1, 40) for _ in range(k)]
        if i % 7 == 0:
            vals = [rng.randint(1, 9) for _ in range(k)]
        tgt = rng.randint(-20, 120)
        cnt, first = 0, None
        for mask in range(1 << k):
            if sum(vals[j] for j in range(k) if (mask >> j) & 1) == tgt:
                cnt += 1
        out.append(Case('search.subset_sum', (vals, tgt), {}, cnt, 'enumeration of all 2^n subsets',
                        lambda r, o, vals=vals, tgt=tgt: r['count'] == o and r['exists'] == (o > 0)
                        and (o == 0 or (r['subset'] == sorted(set(r['subset'])) and sum(vals[j] for j in r['subset']) == tgt))))
    return out


def _knap_brute(w, v, cap, caps):
    """caps[i] = max copies of item i (1 for 0/1). Returns the best value."""
    best = Fraction(0)
    for cnt in itertools.product(*[range(c + 1) for c in caps]):
        if sum(c * x for c, x in zip(cnt, w)) <= cap:
            best = max(best, sum(c * x for c, x in zip(cnt, v)))
    return best


def gen_knapsack(rng, n):
    out = []
    for i in range(n):
        kind = ('0/1', 'unbounded', 'bounded')[i % 3]
        k = rng.randint(1, 10) if kind == '0/1' else rng.randint(1, 4)
        w = [rng.randint(1, 12) for _ in range(k)]
        frac = rng.random() < 0.25
        v = [Fraction(rng.randint(1, 30), rng.choice((1, 2, 3))) if frac else rng.randint(1, 30) for _ in range(k)]
        cap = rng.randint(0, 40)
        if kind == '0/1':
            caps = [1] * k
        elif kind == 'unbounded':
            caps = [cap // wi for wi in w]
        else:
            caps = [rng.randint(0, 3) for _ in range(k)]
        best = _knap_brute(w, v, cap, caps)
        kw = {'kind': kind}
        if kind == 'bounded':
            kw['counts'] = caps

        def cmpf(r, o, w=w, v=v, cap=cap, caps=caps):
            c = r['counts']
            return (Fraction(r['value']) == o and len(c) == len(w) and all(0 <= a <= b for a, b in zip(c, caps)) and sum(a * x for a, x in zip(c, w)) <= cap
                    and sum(a * x for a, x in zip(c, v)) == o)
        out.append(Case('search.knapsack', (w, v, cap), kw, best, f'enumeration of every count vector ({kind}); the returned choice is checked feasible and worth the value', cmpf))
    return out


def gen_coin_change_count(rng, n):
    out = []
    known = [(([1, 5, 10, 25, 50], 100), {}, 292, 'known: 292 ways to change a dollar with 1,5,10,25,50 cents'),
             (([1, 2, 5, 10, 20, 50, 100, 200], 200), {}, 73682, 'known: Project Euler 31 (73682)'),
             (([1, 5, 10, 25], 100), {}, 242, 'known: 242 ways to make a dollar from pennies, nickels, dimes, quarters')]
    for i in range(n):
        if i < len(known):
            a, kw, e, name = known[i]
            out.append(Case('search.coin_change_count', a, kw, e, name, eq))
            continue
        k = rng.randint(1, 4)
        coins = rng.sample(range(1, 13), k)
        amt = rng.randint(0, 40)
        mode = i % 3
        if mode == 0:
            cnt = sum(1 for c in itertools.product(*[range(amt // x + 1) for x in coins]) if sum(a * b for a, b in zip(c, coins)) == amt)
            out.append(Case('search.coin_change_count', (coins, amt), {}, cnt, 'enumeration of every count vector', eq))
        elif mode == 1:
            a2 = rng.randint(0, 13)

            def seqs(rem):
                if rem == 0:
                    return 1
                return sum(seqs(rem - c) for c in coins if c <= rem)
            out.append(Case('search.coin_change_count', (coins, a2), {'ordered': True}, seqs(a2), 'recursive enumeration of ordered sequences', eq))
        else:
            lims = [rng.randint(0, 4) for _ in coins]
            cnt = sum(1 for c in itertools.product(*[range(min(l, amt // x) + 1) for x, l in zip(coins, lims)]) if sum(a * b for a, b in zip(c, coins)) == amt)
            out.append(Case('search.coin_change_count', (coins, amt), {'limits': lims}, cnt, 'enumeration of every bounded count vector', eq))
    return out


def gen_coin_change_min(rng, n):
    out = []
    for i in range(n):
        k = rng.randint(1, 5)
        coins = rng.sample(range(1, 30), k)
        amt = rng.randint(0, 120)
        # oracle: breadth-first search over remaining amounts (independent of the DP)
        dist, frontier, d = {amt: 0}, [amt], 0
        while frontier and 0 not in dist:
            d += 1
            nxt = []
            for r in frontier:
                for c in coins:
                    if r - c >= 0 and (r - c) not in dist:
                        dist[r - c] = d
                        nxt.append(r - c)
            frontier = nxt
        exp = dist.get(0)

        def cmpf(r, o, coins=coins, amt=amt):
            if o is None:
                return r['count'] is None
            return r['count'] == o and sum(c * m for c, m in r['coins'].items()) == amt and sum(r['coins'].values()) == o and all(c in coins for c in r['coins'])
        out.append(Case('search.coin_change_min', (coins, amt), {}, exp, 'breadth-first search over remainders; the returned coins are checked to sum to the amount', cmpf))
    return out


# ----------------------------------------------------------------------------------------------------------------------------------------
# linear programming
# ----------------------------------------------------------------------------------------------------------------------------------------

def _rand_lp(rng):
    n = rng.randint(2, 4)
    m = rng.randint(1, 4)
    c = [rng.randint(-5, 8) for _ in range(n)]
    A = [[rng.randint(-3, 6) for _ in range(n)] for _ in range(m)]
    b = [rng.randint(-2, 20) for _ in range(m)]
    return n, m, c, A, b


def _vertex_enum(c, rows, n):
    """rows: list of (coeffs, rhs) meaning coeffs.x <= rhs (bounds already included as rows, x free). Maximise c.x over the polyhedron by listing every point where n
    linearly independent rows are tight (exact Gaussian elimination, written here). Returns (status, value) where status is 'optimal' or 'infeasible' or 'unbounded'."""
    def solve(M, rhs):
        k = len(M)
        a = [list(map(Fraction, M[i])) + [Fraction(rhs[i])] for i in range(k)]
        for col in range(k):
            piv = next((r for r in range(col, k) if a[r][col] != 0), None)
            if piv is None:
                return None
            a[col], a[piv] = a[piv], a[col]
            pv = a[col][col]
            a[col] = [x / pv for x in a[col]]
            for r in range(k):
                if r != col and a[r][col] != 0:
                    f = a[r][col]
                    a[r] = [x - f * y for x, y in zip(a[r], a[col])]
        return [a[i][k] for i in range(k)]
    pts = []
    for sub in itertools.combinations(range(len(rows)), n):
        x = solve([rows[i][0] for i in sub], [rows[i][1] for i in sub])
        if x is None:
            continue
        if all(sum(Fraction(p) * q for p, q in zip(r[0], x)) <= r[1] for r in rows):
            pts.append(x)
    if not pts:
        return 'infeasible', None
    return 'optimal', max(sum(Fraction(ci) * xi for ci, xi in zip(c, x)) for x in pts)


def _lp_cmp_factory(c, Au, bu, Ae, be, bounds, maximize):
    def cmpf(r, o):
        if o[0] != r['status']:
            return False
        if o[0] != 'optimal':
            return True
        x = r['x']
        lo_hi_ok = all((lo is None or xi >= lo) and (hi is None or xi <= hi) for xi, (lo, hi) in zip(x, bounds))
        feas = (all(sum(Fraction(a) * xi for a, xi in zip(row, x)) <= b for row, b in zip(Au, bu))
                and all(sum(Fraction(a) * xi for a, xi in zip(row, x)) == b for row, b in zip(Ae, be)))
        val = sum(Fraction(ci) * xi for ci, xi in zip(c, x))
        return lo_hi_ok and feas and val == r['value'] == o[1]
    return cmpf


def gen_lp_solve(rng, n):
    out = []
    for i in range(n):
        nv, m, c, A, b = _rand_lp(rng)
        kind = i % 5
        bounds = [(Fraction(0), None)] * nv
        Ae, be = [], []
        if kind == 1:                                        # an equality row and mixed bounds
            Ae, be = [[rng.randint(0, 3) for _ in range(nv)]], [rng.randint(0, 12)]
        if kind == 2:                                        # box bounds with negative lower ends
            bounds = [(Fraction(rng.randint(-4, 1)), Fraction(rng.randint(2, 8))) for _ in range(nv)]
        if kind == 3:                                        # free variables boxed by rows, so every oracle can see them
            bounds = [(None, None)] * nv
            A = A + [[1 if j == k else 0 for j in range(nv)] for k in range(nv)] + [[-1 if j == k else 0 for j in range(nv)] for k in range(nv)]
            b = b + [rng.randint(3, 9)] * nv + [rng.randint(3, 9)] * nv
        if kind == 4:                                        # an upper bound only (x = hi - y)
            bounds = [(None, Fraction(rng.randint(2, 8)))] + [(Fraction(0), None)] * (nv - 1)
        maximize = rng.random() < 0.6
        # oracle 1: vertex enumeration over the polyhedron (rows + bounds + equalities as two inequalities), bounded by a big box so unbounded cases are detected by growth
        rows = [(r, bb) for r, bb in zip(A, b)]
        for r, bb in zip(Ae, be):
            rows += [(r, bb), ([-x for x in r], -bb)]
        for j, (lo, hi) in enumerate(bounds):
            e = [1 if k == j else 0 for k in range(nv)]
            if lo is not None:
                rows.append(([-x for x in e], -lo))
            if hi is not None:
                rows.append((e, hi))
        cc = c if maximize else [-x for x in c]
        BIG = 10 ** 6
        boxed = rows + [([1 if k == j else 0 for k in range(nv)], BIG) for j in range(nv)] + [([-1 if k == j else 0 for k in range(nv)], BIG) for j in range(nv)]
        st, val = _vertex_enum(cc, boxed, nv)
        if st == 'optimal':
            st2, val2 = _vertex_enum(cc, rows + [([1 if k == j else 0 for k in range(nv)], BIG * 100) for j in range(nv)]
                                     + [([-1 if k == j else 0 for k in range(nv)], BIG * 100) for j in range(nv)], nv)
            if val2 > val:
                st, val = 'unbounded', None                      # the optimum grew when the box grew: the LP is unbounded
        exp = (st, (val if maximize else -val) if val is not None else None)
        out.append(Case('search.lp_solve', (c, A, b, Ae or None, be or None, bounds), {'maximize': maximize}, exp,
                        'vertex enumeration with exact Gaussian elimination (box 1e6 and 1e8 to detect unboundedness); the returned x is checked feasible and worth the value',
                        _lp_cmp_factory(c, A, b, Ae, be, bounds, maximize)))
    # a few cross-checks against SymPy's exact simplex (variables >= 0, rows only)
    return out


def gen_lp_solve_sympy(rng, n):
    from sympy.solvers.simplex import linprog, InfeasibleLPError, UnboundedLPError
    out = []
    for i in range(n):
        nv, m, c, A, b = _rand_lp(rng)
        try:
            val, _ = linprog(c, A, b)                                    # sympy minimises
            exp = ('optimal', -SR(val))                                  # we maximise -c.x below by passing -c
        except UnboundedLPError:
            exp = ('unbounded', None)
        except InfeasibleLPError:
            exp = ('infeasible', None)
        # ours: minimise c.x == sympy: pass maximize=False
        if exp[0] == 'optimal':
            exp = ('optimal', SR(val))
        out.append(Case('search.lp_solve', (c, A, b), {'maximize': False}, (exp[0], Fraction(int(exp[1].p), int(exp[1].q)) if exp[1] is not None else None),
                        'SymPy 1.14 linprog (exact simplex, independent code); the returned x is checked feasible',
                        _lp_cmp_factory(c, A, b, [], [], [(Fraction(0), None)] * nv, False)))
    return out


def gen_ilp_solve(rng, n):
    out = []
    for i in range(n):
        nv = rng.randint(2, 3)
        m = rng.randint(1, 3)
        c = [rng.randint(-3, 9) for _ in range(nv)]
        A = [[rng.randint(-2, 7) for _ in range(nv)] for _ in range(m)]
        b = [rng.randint(0, 25) for _ in range(m)]
        hi = rng.randint(3, 6)
        bounds = [(Fraction(0), Fraction(hi))] * nv
        maximize = rng.random() < 0.6
        best = None
        for x in itertools.product(range(hi + 1), repeat=nv):
            if all(sum(a * xi for a, xi in zip(row, x)) <= bb for row, bb in zip(A, b)):
                v = sum(ci * xi for ci, xi in zip(c, x))
                if best is None or (v > best if maximize else v < best):
                    best = v
        exp = ('infeasible', None) if best is None else ('optimal', Fraction(best))

        def cmpf(r, o, c=c, A=A, b=b, hi=hi):
            if r['status'] != o[0]:
                return False
            if o[0] != 'optimal':
                return True
            x = r['x']
            return (all(xi.denominator == 1 and 0 <= xi <= hi for xi in x) and all(sum(a * xi for a, xi in zip(row, x)) <= bb for row, bb in zip(A, b))
                    and sum(ci * xi for ci, xi in zip(c, x)) == r['value'] == o[1])
        out.append(Case('search.ilp_solve', (c, A, b, None, None, bounds), {'maximize': maximize}, exp,
                        'enumeration of every integer point in the box; the returned x is checked integral, feasible and worth the value', cmpf))
    return out


# ----------------------------------------------------------------------------------------------------------------------------------------
# graphs
# ----------------------------------------------------------------------------------------------------------------------------------------

def _rand_graph(rng, directed, neg=False, wfrac=False, nmax=7):
    n = rng.randint(2, nmax)
    m = rng.randint(1, min(n * (n - 1) // (1 if directed else 2), 14))
    pairs = [(u, v) for u in range(n) for v in range(n) if u != v and (directed or u < v)]
    es = []
    for u, v in rng.sample(pairs, min(m, len(pairs))):
        w = rng.randint(-3 if neg else 0, 9)
        es.append((u, v, Fraction(w, rng.choice((1, 2, 4))) if wfrac else w))
    return n, es


def _simple_paths(adj, s, t):
    out = []

    def rec(u, seen, w, path):
        if u == t:
            out.append((w, list(path)))
            return
        for v, c in adj.get(u, ()):
            if v not in seen:
                seen.add(v)
                path.append(v)
                rec(v, seen, w + c, path)
                path.pop()
                seen.discard(v)
    rec(s, {s}, 0, [s])
    return out


def _neg_cycle_reachable(adj, nodes, s):
    reach = {s}
    stack = [s]
    while stack:
        u = stack.pop()
        for v, _ in adj.get(u, ()):
            if v not in reach:
                reach.add(v)
                stack.append(v)
    for c in reach:                                              # a simple cycle through c with negative total
        def rec(u, seen, w):
            for v, cost in adj.get(u, ()):
                if v == c and w + cost < 0:
                    return True
                if v not in seen and v in reach and v > c:         # each cycle is found from its smallest node
                    seen.add(v)
                    if rec(v, seen, w + cost):
                        return True
                    seen.discard(v)
            return False
        if rec(c, {c}, 0):
            return True
    return False


def _adj(es, directed):
    adj = {}
    for u, v, w in es:
        adj.setdefault(u, []).append((v, w))
        if not directed:
            adj.setdefault(v, []).append((u, w))
    return adj


def gen_shortest_path(rng, n):
    out = []
    for i in range(n):
        directed = rng.random() < 0.75
        neg = directed and rng.random() < 0.35
        nv, es = _rand_graph(rng, directed, neg, wfrac=rng.random() < 0.2)
        s = rng.randrange(nv)
        t = rng.randrange(nv)
        adj = _adj(es, directed)
        negcyc = _neg_cycle_reachable(adj, range(nv), s)
        dist = {}
        if not negcyc:
            for v in range(nv):
                ps = _simple_paths(adj, s, v) if v != s else [(0, [s])]
                if ps:
                    dist[v] = min(p[0] for p in ps)
        exp = ('neg',) if negcyc else (dist,)

        def cmpf(r, o, es=es, directed=directed, s=s, t=t):
            if o == ('neg',):
                return r['negative_cycle'] is True and r['dist'] is None
            d = o[0]
            if r['negative_cycle'] or {k: Fraction(v) for k, v in r['dist'].items()} != {k: Fraction(v) for k, v in d.items()}:
                return False
            if t not in d:
                return r['path'] is None
            p = r['path']
            if p[0] != s or p[-1] != t:
                return False
            tot = Fraction(0)
            for a, b in zip(p, p[1:]):
                cands = [w for u, v, w in es if (u, v) == (a, b) or (not directed and (v, u) == (a, b))]
                if not cands:
                    return False
                tot += min(cands)
            return tot == d[t]
        out.append(Case('search.shortest_path', (es, s, t), {'directed': directed}, exp,
                        'enumeration of all simple paths (and of simple cycles for the negative-cycle verdict); the returned path is checked edge by edge', cmpf))
    return out


def gen_all_pairs_shortest(rng, n):
    out = []
    for i in range(n):
        directed = rng.random() < 0.7
        neg = directed and rng.random() < 0.3
        nv, es = _rand_graph(rng, directed, neg, nmax=6)
        adj = _adj(es, directed)
        nodes = sorted({u for u, v, w in es} | {v for u, v, w in es})
        negcyc = any(_neg_cycle_reachable(adj, nodes, s) for s in nodes)
        if negcyc:
            exp = None
        else:
            exp = {}
            for s in nodes:
                exp[s] = {}
                for t in nodes:
                    ps = _simple_paths(adj, s, t) if s != t else [(0, [s])]
                    if ps:
                        exp[s][t] = min(p[0] for p in ps)

        def cmpf(r, o):
            if o is None:
                return r['negative_cycle'] is True
            return (not r['negative_cycle']) and {u: {v: Fraction(d) for v, d in row.items()} for u, row in r['dist'].items()} == {u: {v: Fraction(d) for v, d in row.items()} for u, row in o.items()}
        out.append(Case('search.all_pairs_shortest', (es,), {'directed': directed}, exp, 'enumeration of all simple paths and simple cycles', cmpf))
    return out


def gen_max_flow(rng, n):
    out = []
    for i in range(n):
        directed = rng.random() < 0.7
        nv, es = _rand_graph(rng, directed, nmax=7)
        es = [(u, v, abs(w) + 1) for u, v, w in es]
        s, t = rng.sample(range(nv), 2)
        best = None
        others = [x for x in range(nv) if x not in (s, t)]
        for mask in range(1 << len(others)):
            S = {s} | {others[j] for j in range(len(others)) if (mask >> j) & 1}
            cap = sum(w for u, v, w in es if (u in S and v not in S) or (not directed and v in S and u not in S))
            best = cap if best is None else min(best, cap)

        def cmpf(r, o, es=es, directed=directed, s=s, t=t):
            if r['value'] != o:
                return False
            fl = r['flow']
            if len(fl) != len(es):
                return False
            net = {}
            for (u, v, w), f in zip(es, fl):
                if (f < -w or f > w) if not directed else (f < 0 or f > w):
                    return False
                net[u] = net.get(u, 0) + f
                net[v] = net.get(v, 0) - f
            if any(val != 0 for k, val in net.items() if k not in (s, t)) or net.get(s, 0) != o:
                return False
            S = set(r['cut']['source_side'])
            capacity = sum(w for u, v, w in es if (u in S and v not in S) or (not directed and v in S and u not in S))
            return s in S and t not in S and capacity == o
        out.append(Case('search.max_flow', (es, s, t), {'directed': directed}, best,
                        'minimum cut by enumerating every source-side vertex set (max-flow min-cut theorem); the returned flow is checked for conservation and capacity, its cut for capacity = value', cmpf))
    return out


def gen_mst(rng, n):
    out = []
    for i in range(n):
        nv, es = _rand_graph(rng, False, nmax=6)
        es = [(u, v, w + 1) for u, v, w in es]
        nodes = sorted({u for u, v, w in es} | {v for u, v, w in es})
        mx = rng.random() < 0.3

        def comps():
            p = {x: x for x in nodes}

            def f(a):
                while p[a] != a:
                    a = p[a]
                return a
            for u, v, w in es:
                p[f(u)] = f(v)
            return len({f(x) for x in nodes})
        k = len(nodes) - comps()
        best = None
        for sub in itertools.combinations(range(len(es)), k):
            p = {x: x for x in nodes}

            def f(a):
                while p[a] != a:
                    a = p[a]
                return a
            ok = True
            for j in sub:
                a, b = f(es[j][0]), f(es[j][1])
                if a == b:
                    ok = False
                    break
                p[a] = b
            if ok:
                w = sum(es[j][2] for j in sub)
                best = w if best is None else (max(best, w) if mx else min(best, w))
        connected = comps() == 1

        def cmpf(r, o, es=es, k=k, connected=connected):
            if r['weight'] != o[0] or r['connected'] != connected or len(r['edges']) != k:
                return False
            return sum(es[j][2] for j in r['edges']) == o[0]
        out.append(Case('search.mst', (es,), {'maximize': mx}, (best,), 'enumeration of every acyclic edge subset of the spanning size; the returned edges are checked to weigh the total', cmpf))
    return out


def gen_assignment(rng, n):
    out = []
    for i in range(n):
        r = rng.randint(1, 6)
        c = rng.randint(1, 6)
        frac = rng.random() < 0.2
        M = [[Fraction(rng.randint(-5, 20), rng.choice((1, 2, 3))) if frac else rng.randint(-5, 20) for _ in range(c)] for _ in range(r)]
        mx = rng.random() < 0.4
        best = None
        if r <= c:
            for cols in itertools.permutations(range(c), r):
                t = sum(M[i][cols[i]] for i in range(r))
                best = t if best is None else (max(best, t) if mx else min(best, t))
        else:
            for rows in itertools.permutations(range(r), c):
                t = sum(M[rows[j]][j] for j in range(c))
                best = t if best is None else (max(best, t) if mx else min(best, t))

        def cmpf(res, o, M=M, r=r, c=c):
            pairs = res['pairs']
            return (Fraction(res['cost']) == o and len(pairs) == min(r, c) and len({p[0] for p in pairs}) == len(pairs) and len({p[1] for p in pairs}) == len(pairs)
                    and sum(M[a][b] for a, b in pairs) == o)
        out.append(Case('search.assignment', (M,), {'maximize': mx}, Fraction(best), 'enumeration of every injective assignment (itertools.permutations); the pairs are checked', cmpf))
    return out


# ----------------------------------------------------------------------------------------------------------------------------------------
# statistics
# ----------------------------------------------------------------------------------------------------------------------------------------

def _rand_data(rng, lo=2, hi=12):
    k = rng.randint(lo, hi)
    if rng.random() < 0.4:
        return [Fraction(rng.randint(-20, 40), rng.choice((1, 2, 3, 7))) for _ in range(k)]
    return [rng.randint(-20, 40) for _ in range(k)]


def gen_mean(rng, n):
    return [Case('search.mean', (d,), {}, statistics.mean([Fraction(x) for x in d]), 'statistics.mean on Fractions (exact)', eq) for d in (_rand_data(rng, 1) for _ in range(n))]


def gen_median(rng, n):
    return [Case('search.median', (d,), {}, Fraction(statistics.median([Fraction(x) for x in d])), 'statistics.median on Fractions', eq) for d in (_rand_data(rng, 1) for _ in range(n))]


def gen_mode(rng, n):
    out = []
    for _ in range(n):
        d = [rng.randint(0, 6) for _ in range(rng.randint(1, 15))]
        out.append(Case('search.mode', (d,), {}, sorted(Fraction(x) for x in statistics.multimode(d)), 'statistics.multimode', eq))
    return out


def gen_variance(rng, n):
    out = []
    for i in range(n):
        d = _rand_data(rng, 2)
        sample = i % 2 == 1
        fd = [Fraction(x) for x in d]
        o = statistics.variance(fd) if sample else statistics.pvariance(fd)
        out.append(Case('search.variance', (d,), {'sample': sample}, Fraction(o), 'statistics.variance / pvariance on Fractions', eq))
    return out


def gen_stdev(rng, n):
    out = []
    for i in range(n):
        d = _rand_data(rng, 2)
        sample = i % 2 == 1
        fd = [Fraction(x) for x in d]
        v = Fraction(statistics.variance(fd) if sample else statistics.pvariance(fd))
        out.append(Case('search.stdev', (d,), {'sample': sample}, v,
                        'certificate: the Surd squared equals statistics.(p)variance exactly, and is non-negative',
                        lambda r, o: isinstance(r, Surd) and r.sign() >= 0 and r * r == Surd(o)))
    return out


def _xy(rng, lo=3):
    k = rng.randint(lo, 10)
    xs = [rng.randint(-10, 20) for _ in range(k)]
    if len(set(xs)) < 2:
        xs[0] += 1
    ys = [Fraction(rng.randint(-10, 20), rng.choice((1, 2))) for _ in range(k)]
    return xs, ys


def gen_covariance(rng, n):
    out = []
    for i in range(n):
        xs, ys = _xy(rng, 2)
        sample = i % 2 == 1
        k = len(xs)
        mx, my = sympy.Rational(sum(xs), k), sum(sympy.Rational(y.numerator, y.denominator) for y in ys) / k
        e = sum(sympy.Rational(x) * sympy.Rational(y.numerator, y.denominator) for x, y in zip(xs, ys)) / k - mx * my      # E[xy] - E[x]E[y]
        if sample:
            e = e * k / (k - 1)
        out.append(Case('search.covariance', (xs, ys), {'sample': sample}, Fraction(int(e.p), int(e.q)), 'SymPy Rational algebra with the identity E[xy] - E[x]E[y]', eq))
    return out


def gen_correlation(rng, n):
    out = []
    for i in range(n):
        xs, ys = _xy(rng, 3)
        if len(set(ys)) < 2:
            ys[0] += 1
        fx, fy = [float(x) for x in xs], [float(y) for y in ys]
        mxf, myf = sum(fx) / len(fx), sum(fy) / len(fy)
        r = sum((a - mxf) * (b - myf) for a, b in zip(fx, fy)) / math.sqrt(sum((a - mxf) ** 2 for a in fx) * sum((b - myf) ** 2 for b in fy))
        sx, sy = sympy.Matrix([sympy.Rational(x) for x in xs]), sympy.Matrix([sympy.Rational(y.numerator, y.denominator) for y in ys])
        sxx = ((sx.T * sx)[0] - sum(sx) ** 2 / len(xs))
        syy = ((sy.T * sy)[0] - sum(sy) ** 2 / len(xs))
        sxy = ((sx.T * sy)[0] - sum(sx) * sum(sy) / len(xs))
        r2 = sympy.Rational(sxy ** 2 / (sxx * syy))
        r2f = Fraction(int(r2.p), int(r2.q))
        out.append(Case('search.correlation', (xs, ys), {}, (r2f, r),
                        'certificate: Surd squared equals r^2 from SymPy sums-of-squares algebra, sign equals sign of the covariance, float agrees with plain float arithmetic',
                        lambda res, o: isinstance(res, Surd) and res * res == Surd(o[0]) and abs(float(res) - o[1]) < 1e-9))
    return out


def gen_linear_regression(rng, n):
    out = []
    i = -1
    while len(out) < n:
        i += 1
        xs, ys = _xy(rng, 3)
        k = len(xs)
        X = sympy.Matrix([[k, sum(xs)], [sum(xs), sum(x * x for x in xs)]])
        yv = [sympy.Rational(y.numerator, y.denominator) for y in ys]
        rhs = sympy.Matrix([sum(yv), sum(sympy.Rational(x) * y for x, y in zip(xs, yv))])
        if X.det() == 0:
            continue
        sol = X.LUsolve(rhs)
        b0, b1 = sol[0], sol[1]
        res = sum((y - b0 - b1 * x) ** 2 for x, y in zip(xs, yv))
        tot = sum((y - sum(yv) / k) ** 2 for y in yv)
        r2 = sympy.Integer(1) - res / tot if tot != 0 else sympy.Integer(1)
        fr = lambda q: Fraction(int(q.p), int(q.q))
        out.append(Case('search.linear_regression', (xs, ys), {}, {'slope': fr(b1), 'intercept': fr(b0), 'r_squared': fr(r2), 'n': k},
                        'SymPy: normal equations by LU solve; R^2 = 1 - SSres/SStot', eq))
    return out


def _sp_pmf(dist, k):
    d = dist.pspace.distribution
    return d.pmf(k) if hasattr(d, 'pmf') else d.pdf(k)


def _fr(q):
    q = sympy.Rational(q)
    return Fraction(int(q.p), int(q.q))


def gen_binomial_pmf(rng, n):
    from sympy.stats import Binomial
    out = []
    for _ in range(n):
        nn, p = rng.randint(0, 25), Fraction(rng.randint(0, 9), rng.choice((9, 10, 7)))
        p = min(p, Fraction(1))
        k = rng.randint(-1, nn + 1)
        e = _sp_pmf(Binomial('X', nn, SR(p.numerator, p.denominator)), k) if 0 <= k <= nn else 0
        out.append(Case('search.binomial_pmf', (nn, p, k), {}, _fr(e), 'sympy.stats Binomial pmf', eq))
    return out


def gen_binomial_cdf(rng, n):
    from sympy.stats import Binomial
    out = []
    for _ in range(n):
        nn, p = rng.randint(0, 25), Fraction(rng.randint(0, 9), rng.choice((9, 10, 7)))
        p = min(p, Fraction(1))
        k = rng.randint(-1, nn + 1)
        X = Binomial('X', nn, SR(p.numerator, p.denominator))
        e = sum(_sp_pmf(X, j) for j in range(0, min(k, nn) + 1)) if k >= 0 else 0
        out.append(Case('search.binomial_cdf', (nn, p, k), {}, _fr(e), 'sum of sympy.stats Binomial pmf', eq))
    return out


def gen_hypergeom_pmf(rng, n):
    from sympy.stats import Hypergeometric
    out = []
    for _ in range(n):
        N = rng.randint(2, 40)
        K = rng.randint(0, N)
        d = rng.randint(1, N)
        k = rng.randint(0, d)
        e = _sp_pmf(Hypergeometric('H', N, K, d), k)
        out.append(Case('search.hypergeom_pmf', (N, K, d, k), {}, _fr(e), 'sympy.stats Hypergeometric pmf', eq))
    return out


def gen_hypergeom_cdf(rng, n):
    from sympy.stats import Hypergeometric
    out = []
    for _ in range(n):
        N = rng.randint(2, 40)
        K = rng.randint(0, N)
        d = rng.randint(1, N)
        k = rng.randint(0, d)
        H = Hypergeometric('H', N, K, d)
        e = sum(_sp_pmf(H, j) for j in range(0, k + 1))
        out.append(Case('search.hypergeom_cdf', (N, K, d, k), {}, _fr(e), 'sum of sympy.stats Hypergeometric pmf', eq))
    return out


def gen_geometric_pmf(rng, n):
    from sympy.stats import Geometric
    out = []
    for _ in range(n):
        p = Fraction(rng.randint(1, 9), rng.choice((10, 12, 9)))
        p = min(p, Fraction(1))
        k = rng.randint(1, 30)
        out.append(Case('search.geometric_pmf', (p, k), {}, _fr(_sp_pmf(Geometric('G', SR(p.numerator, p.denominator)), k)), 'sympy.stats Geometric pmf', eq))
    return out


def gen_geometric_cdf(rng, n):
    from sympy.stats import Geometric
    out = []
    for _ in range(n):
        p = Fraction(rng.randint(1, 9), rng.choice((10, 12, 9)))
        p = min(p, Fraction(1))
        k = rng.randint(1, 25)
        G = Geometric('G', SR(p.numerator, p.denominator))
        out.append(Case('search.geometric_cdf', (p, k), {}, _fr(sum(_sp_pmf(G, j) for j in range(1, k + 1))), 'sum of sympy.stats Geometric pmf', eq))
    return out


def gen_negbinomial_pmf(rng, n):
    from sympy.stats import NegativeBinomial
    out = []
    for _ in range(n):
        r = rng.randint(1, 6)
        p = Fraction(rng.randint(1, 8), 9)
        k = rng.randint(r, r + 15)
        e = _sp_pmf(NegativeBinomial('N', r, SR(p.numerator, p.denominator)), k - r)
        out.append(Case('search.negbinomial_pmf', (r, p, k), {}, _fr(e), 'sympy.stats NegativeBinomial (failures before the r-th success, p = success probability) at k - r', eq))
    return out


def gen_poisson_partial_sum(rng, n):
    out = []
    for _ in range(n):
        lam = Fraction(rng.randint(0, 40), rng.choice((1, 2, 5)))
        k = rng.randint(0, 30)
        e = sum(SR(lam.numerator, lam.denominator) ** j / sympy.factorial(j) for j in range(k + 1))
        # certificate: e^-lam * S = P(X <= k) = Q(k+1, lam), the regularised upper incomplete gamma (mpmath)
        out.append(Case('search.poisson_partial_sum', (lam, k), {}, (_fr(e), lam, k), 'SymPy Rational series, and mpmath: e^-lam * S equals the regularised incomplete gamma Q(k+1, lam)',
                        lambda r, o: r == o[0] and abs(mp.mpf(r.numerator) / r.denominator * mp.exp(-mp.mpf(o[1].numerator) / o[1].denominator)
                                                       - mp.gammainc(o[2] + 1, mp.mpf(o[1].numerator) / o[1].denominator, regularized=True)) < mp.mpf(10) ** -50))
    return out


# ----------------------------------------------------------------------------------------------------------------------------------------
# numeric calculus
# ----------------------------------------------------------------------------------------------------------------------------------------

def _mpq(q):
    q = Fraction(q)
    return mp.mpf(q.numerator) / q.denominator


def _rand_arg(rng, lo, hi):
    return Fraction(rng.randint(int(lo * 8), int(hi * 8)), 8) + (Fraction(1, 3) if rng.random() < 0.3 else 0)


_EVAL_T = [
    ('sin(x)^2 + cos(x)^2', lambda x: mp.sin(x) ** 2 + mp.cos(x) ** 2, (-50, 50)),
    ('exp(x)*log(1+x)', lambda x: mp.exp(x) * mp.log(1 + x), (0, 20)),
    ('atan(x) + asin(x/10)', lambda x: mp.atan(x) + mp.asin(x / 10), (-9, 9)),
    ('sqrt(x) + cbrt(x)', lambda x: mp.sqrt(x) + mp.cbrt(x), (0, 1000)),
    ('sinh(x)/cosh(x) - tanh(x)', lambda x: mp.sinh(x) / mp.cosh(x) - mp.tanh(x), (-30, 30)),
    ('erf(x)', lambda x: mp.erf(x), (-6, 6)),
    ('log(x)/log(2)', lambda x: mp.log(x) / mp.log(2), (1, 500)),
    ('tan(x)', lambda x: mp.tan(x), (-1.5, 1.5)),
    ('exp(-x^2/2)/sqrt(2*pi)', lambda x: mp.exp(-x ** 2 / 2) / mp.sqrt(2 * mp.pi), (-8, 8)),
    ('2^x + x^(1/3)', lambda x: 2 ** x + mp.cbrt(x), (0, 30)),
    ('asinh(x) + acosh(x+1) + atanh(x/12)', lambda x: mp.asinh(x) + mp.acosh(x + 1) + mp.atanh(x / 12), (0, 11)),
    ('sin(1000*x)', lambda x: mp.sin(1000 * x), (-3, 3)),
    ('cos(x) - 1', lambda x: mp.cos(x) - 1, (-1, 1)),
    ('pi*e + log10(x)', lambda x: mp.pi * mp.e + mp.log10(x), (1, 100000)),
]


def gen_num_eval(rng, n):
    out = []
    for i in range(n):
        expr, f, (lo, hi) = _EVAL_T[i % len(_EVAL_T)]
        x = _rand_arg(rng, lo, hi)
        if x == 0 and 'log' in expr:
            x = Fraction(1, 2)
        digits = rng.choice((5, 10, 15, 20, 30, 40))
        o = mpd(f(_mpq(x)))
        out.append(Case('search.num_eval', (expr, {'x': x}, digits), {}, o, 'mpmath at 70 digits', num_cmp))
    return out


_DERIV_T = ['x^5 - 3*x^2 + 1', 'sin(x)*exp(x)', 'log(1+x^2)', 'atan(x)', 'exp(-x^2)', 'sqrt(1+x^2)', 'x^x', 'sin(x)/x', 'tanh(x)', 'asin(x/3)', '1/(1+x^2)', 'cos(2*x)+x^3']


def gen_derivative(rng, n):
    out = []
    xs = sympy.symbols('x')
    for i in range(n):
        if i % 10 == 9:
            expr, x0 = [('abs(x)', 0), ('sqrt(x)', 0), ('cbrt(x)', 0), ('abs(x - 1)', 1)][rng.randint(0, 3)]
            out.append(Case('search.derivative', (expr, x0), {'digits': 10}, REFUSE, 'the function is not differentiable at the point (a kink or an infinite slope): a refusal is right', None))
            continue
        expr = _DERIV_T[i % len(_DERIV_T)]
        order = rng.choice((1, 1, 2, 3, 4))
        x0 = Fraction(rng.randint(1, 20), 8) if expr in ('x^x', 'sin(x)/x') else Fraction(rng.randint(-16, 16), 8)
        digits = rng.choice((6, 10, 15, 20, 30))
        se = sympy.sympify(expr.replace('^', '**'), locals={'cbrt': sympy.cbrt})
        v = sympy.diff(se, xs, order).subs(xs, SR(x0.numerator, x0.denominator)).evalf(70)
        out.append(Case('search.derivative', (expr, x0, digits, order), {}, Decimal(str(v)), 'SymPy exact derivative evaluated at 70 digits', num_cmp))
    return out


_INT_T = [   # expr, a, b, expected-by (sympy closed form or 'mp'), mp function
    ('x^3 - 2*x + 1', None, None, lambda x: x ** 3 - 2 * x + 1),
    ('sin(x)*exp(-x)', 0, 'inf', lambda x: mp.sin(x) * mp.exp(-x)),
    ('1/(1+x^2)', '-inf', 'inf', lambda x: 1 / (1 + x ** 2)),
    ('exp(-x^2)*cos(x)', '-inf', 'inf', lambda x: mp.exp(-x ** 2) * mp.cos(x)),
    ('log(x)', 0, 1, lambda x: mp.log(x)),
    ('1/sqrt(x)', 0, 2, lambda x: 1 / mp.sqrt(x)),
    ('x^(-1/3)', 0, 3, lambda x: x ** (-mp.mpf(1) / 3)),
    ('sqrt(1-x^2)', -1, 1, lambda x: mp.sqrt(1 - x ** 2)),
    ('1/sqrt(1-x^2)', -1, 1, lambda x: 1 / mp.sqrt(1 - x ** 2)),
    ('x^3*exp(-x)', 0, 'inf', lambda x: x ** 3 * mp.exp(-x)),
    ('x/(exp(x)-1)', 0, 'inf', lambda x: x / (mp.exp(x) - 1)),
    ('log(x)/(1+x)', 0, 1, lambda x: mp.log(x) / (1 + x)),
    ('1/(x^2+4)', 1, 'inf', lambda x: 1 / (x ** 2 + 4)),
    ('sin(10*x)', 0, 'pi', lambda x: mp.sin(10 * x)),
    ('exp(-x)*x^2', 0, 'inf', lambda x: mp.exp(-x) * x ** 2),
    ('atan(x)/x', 0, 1, lambda x: mp.atan(x) / x),
    ('exp(x)', -3, 2, lambda x: mp.exp(x)),
    ('x*sin(x)', 0, 'pi', lambda x: x * mp.sin(x)),
]


def _mpb(b):
    return {'inf': mp.inf, '-inf': -mp.inf, 'pi': mp.pi}.get(b, None) if isinstance(b, str) else mp.mpf(b)


def gen_integrate(rng, n):
    out = []
    x = sympy.symbols('x')
    i = -1
    while len(out) < n:
        i += 1
        if i % 8 == 7:                                               # integrals that diverge or have an inner pole: a refusal is right
            expr, a, b = [('1/x', 0, 1), ('1/x^2', -1, 1), ('1/x^(3/2)', 0, 1), ('1/(x-1/3)', 0, 1), ('tan(x)', 0, 2), ('1/x', 1, 'inf'), ('exp(x)', 0, 'inf')][rng.randint(0, 6)]
            out.append(Case('search.integrate', (expr, a, b, 12), {}, REFUSE, 'the integral diverges (or has a pole inside): a refusal is right', None))
            continue
        if i % 8 == 6:                                               # kinks with breakpoints; conditionally convergent: either an answer or a refusal
            k = rng.randint(1, 5)
            if rng.random() < 0.8:
                a0 = rng.randint(-3, 0)
                b0 = rng.randint(1, 4)
                c0 = Fraction(rng.randint(-4, 4), 3)
                expr = f'abs(x - {c0})' if c0 >= 0 else f'abs(x + {-c0})'
                # exact: integral of |x - c| over [a0, b0]
                c = max(a0, min(b0, c0))
                val = Fraction((c - a0) ** 2 + (b0 - c) ** 2, 2) if True else None
                o = Decimal(val.numerator) / Decimal(val.denominator)
                out.append(Case('search.integrate', (expr, a0, b0, 15), {'points': [c0] if a0 < c0 < b0 else []}, o, 'exact: area of two triangles', num_cmp))
            else:
                out.append(Case('search.integrate', ('sin(x)/x', 0, 'inf', 8), {}, (EITHER, Decimal(mp.nstr(mp.pi / 2, 40))), 'known: Dirichlet integral pi/2 (conditionally convergent; a refusal is accepted)', None))
            continue
        expr, a, b, f = _INT_T[i % len(_INT_T)]
        if a is None:
            a, b = rng.randint(-3, 0), rng.randint(1, 4)
        digits = rng.choice((8, 12, 15, 20, 25))
        # oracle: mpmath quadrature at 70 digits (a different algorithm and code), cross-checked by SymPy's exact integral when it has one
        lo, hi = _mpb(a), _mpb(b)
        o = None
        try:
            se = sympy.sympify(expr.replace('^', '**'))
            sa = {'inf': sympy.oo, '-inf': -sympy.oo, 'pi': sympy.pi}.get(a, a) if isinstance(a, str) else a
            sb = {'inf': sympy.oo, '-inf': -sympy.oo, 'pi': sympy.pi}.get(b, b) if isinstance(b, str) else b
            ex = sympy.integrate(se, (x, sa, sb))
            if not ex.has(sympy.Integral) and ex.is_number:
                o = Decimal(str(ex.evalf(70)))
        except Exception:
            o = None
        if o is None:
            v, err = mp.quad(f, [lo, hi], error=True, maxdegree=10)
            if err > mp.mpf(10) ** -35:
                continue
            o = mpd(v)
            name = 'mpmath.quad at 70 digits (error estimate < 1e-35)'
        else:
            name = 'SymPy exact integral evaluated at 70 digits'
        out.append(Case('search.integrate', (expr, a, b, digits), {}, o, name, num_cmp))
    return out


_LIM_T = [  # expr, point, direction, sympy-expr point
    ('sin(x)/x', 0, '+'), ('(1-cos(x))/x^2', 0, '+'), ('(exp(x)-1)/x', 0, '-'), ('(1+x)^(1/x)', 0, '+'), ('x^x', 0, '+'), ('log(x)/x', 'inf', '+'),
    ('(1+3/x)^x', 'inf', '+'), ('sqrt(x^2+5*x)-x', 'inf', '+'), ('x*log(x)', 0, '+'), ('tan(2*x)/x', 0, '+'), ('(sin(x)-x)/x^3', 0, '+'), ('atan(x)', 'inf', '+'),
    ('x*sin(1/x)', 'inf', '+'), ('(x^2-4)/(x-2)', 2, '+'), ('(sqrt(1+x)-1)/x', 0, '+'), ('abs(x)/x', 0, '+'), ('abs(x)/x', 0, '-'), ('exp(-x)*x^5', 'inf', '+'),
    ('(exp(x)-1-x)/x^2', 0, '+'), ('log(1+x)/x', 0, '+'), ('(cos(x)-1+x^2/2)/x^4', 0, '+'), ('x^2*log(x)', 0, '+'), ('sin(x)/x', 'inf', '+')]


def gen_limit(rng, n):
    out = []
    xs = sympy.symbols('x')
    for i in range(n):
        if i % 8 == 7:                                               # no limit: a refusal is right
            expr, p, d = [('1/x', 0, '+'), ('x', 'inf', '+'), ('sin(1/x)', 0, '+'), ('1/x', 0, 'both'), ('abs(x)/x', 0, 'both'), ('exp(x)', 'inf', '+'), ('cos(x)', 'inf', '+')][rng.randint(0, 6)]
            out.append(Case('search.limit', (expr, p, 10, d), {}, REFUSE, 'the limit is infinite or does not exist: a refusal is right', None))
            continue
        expr, p, d = _LIM_T[i % len(_LIM_T)]
        digits = rng.choice((8, 12, 15, 20))
        se = sympy.sympify(expr.replace('^', '**'))
        sp = sympy.oo if p == 'inf' else p
        v = sympy.limit(se, xs, sp, d)
        out.append(Case('search.limit', (expr, p, digits, d), {}, Decimal(str(sympy.N(v, 60))), 'SymPy limit (exact), evaluated at 60 digits', num_cmp))
    return out


_SER_T = [  # term, start, mp function of n
    ('1/n^2', 1), ('1/n^3', 1), ('1/(n^2+1)', 1), ('(-1)^n/n', 1), ('(-1)^n/n^2', 1), ('1/2^n', 0), ('n/3^n', 1), ('1/2^n', 1),
    ('n^2/2^n', 1), ('1/(n*(n+1))', 1), ('1/(n^2+3*n+2)', 0), ('1/(4*n^2-1)', 1), ('(-1)^n/(2*n+1)', 0), ('1/(n*2^n)', 1), ('(-1)^n/(n^2+1)', 0),
    ('1/(n^2+n+1)', 0), ('n/(n^4+1)', 1), ('(-1)^n*n/(n^2+1)', 1), ('1/(3*n+1)^2', 0), ('(2/3)^n', 0), ('(-1)^n/sqrt(n)', 1), ('n^3/5^n', 1)]


def gen_series_sum(rng, n):
    out = []
    n_ = sympy.symbols('n', integer=True)
    for i in range(n):
        if i % 8 == 7:
            t, s = [('1/n', 1), ('1/sqrt(n)', 1), ('n', 1), ('2^n', 0), ('1/(n*log(n+1))', 1), ('(-2)^n', 0)][rng.randint(0, 5)]
            out.append(Case('search.series_sum', (t, s, 'inf', 10), {}, REFUSE, 'the series diverges: a refusal is right', None))
            continue
        if i % 8 == 5:                                               # finite sums: exact Fractions summed, compared as Decimals
            t = rng.choice(('1/(n^2+1)', '1/n', '1/(n*(n+1))', 'n/(n+1)'))
            s = rng.randint(1, 3)
            e = s + rng.randint(5, 60)
            fn = {'1/(n^2+1)': lambda k: Fraction(1, k * k + 1), '1/n': lambda k: Fraction(1, k), '1/(n*(n+1))': lambda k: Fraction(1, k * (k + 1)), 'n/(n+1)': lambda k: Fraction(k, k + 1)}[t]
            tot = sum(fn(k) for k in range(s, e + 1))
            with localcontext() as c:
                c.prec = 80
                o = Decimal(tot.numerator) / Decimal(tot.denominator)
            out.append(Case('search.series_sum', (t, s, e, 15), {}, o, 'exact Fraction sum, divided at 80 digits', num_cmp))
            continue
        t, s = _SER_T[i % len(_SER_T)]
        digits = rng.choice((8, 12, 15, 20))
        se = sympy.sympify(t.replace('^', '**').replace('n!', 'factorial(n)'), locals={'n': n_})
        o = None
        # oracle: SymPy exact summation when it closes, else mpmath.nsum with the same term as an mpmath lambda
        mpf_term = _series_mp(t)
        try:
            ex = sympy.summation(se, (n_, s, sympy.oo))
            if ex.is_number and not ex.has(sympy.Sum):
                o = Decimal(str(ex.evalf(70)))
        except Exception:
            o = None
        name = 'SymPy exact summation evaluated at 70 digits'
        if o is None:
            v = mp.nsum(mpf_term, [s, mp.inf])
            o = mpd(v)
            name = 'mpmath.nsum at 70 digits'
        wrap = (EITHER, o) if ('sqrt' in t) else o                    # n^(-1/2) alternating converges like a power of 1/sqrt(n): a refusal is accepted
        out.append(Case('search.series_sum', (t, s, 'inf', digits), {}, wrap, name, num_cmp))
    return out


def _series_mp(t):
    """the term text as an mpmath function of n (hand-written pairs below keep the oracle independent of the backend's parser)."""
    table = {
        '1/n^2': lambda n: 1 / n ** 2, '1/n^3': lambda n: 1 / n ** 3, '1/(n^2+1)': lambda n: 1 / (n ** 2 + 1), '(-1)^n/n': lambda n: (-1) ** int(n) / n,
        '(-1)^n/n^2': lambda n: (-1) ** int(n) / n ** 2, '1/2^n': lambda n: mp.mpf(2) ** -n, 'n/3^n': lambda n: n / mp.mpf(3) ** n, 'n^2/2^n': lambda n: n ** 2 / mp.mpf(2) ** n,
        '1/(n*(n+1))': lambda n: 1 / (n * (n + 1)), '1/(n^2+3*n+2)': lambda n: 1 / (n ** 2 + 3 * n + 2), '1/(4*n^2-1)': lambda n: 1 / (4 * n ** 2 - 1),
        '(-1)^n/(2*n+1)': lambda n: (-1) ** int(n) / (2 * n + 1), '1/(n*2^n)': lambda n: 1 / (n * mp.mpf(2) ** n), '(-1)^n/(n^2+1)': lambda n: (-1) ** int(n) / (n ** 2 + 1),
        '1/(n^2+n+1)': lambda n: 1 / (n ** 2 + n + 1), 'n/(n^4+1)': lambda n: n / (n ** 4 + 1), '(-1)^n*n/(n^2+1)': lambda n: (-1) ** int(n) * n / (n ** 2 + 1),
        '1/(3*n+1)^2': lambda n: 1 / (3 * n + 1) ** 2, '(2/3)^n': lambda n: (mp.mpf(2) / 3) ** n, '(-1)^n/sqrt(n)': lambda n: (-1) ** int(n) / mp.sqrt(n),
        'n^3/5^n': lambda n: n ** 3 / mp.mpf(5) ** n}
    return table[t]


_ROOT_T = [  # expr, mpmath fn, bracket generator range
    ('x^3 - 2*x - 5', lambda x: x ** 3 - 2 * x - 5, (-4, 4)), ('cos(x) - x', lambda x: mp.cos(x) - x, (-3, 3)), ('exp(x) - 3*x', lambda x: mp.exp(x) - 3 * x, (-2, 4)),
    ('x*exp(x) - 1', lambda x: x * mp.exp(x) - 1, (-1, 3)), ('log(x) + x - 2', lambda x: mp.log(x) + x - 2, (0.1, 6)), ('sin(x) - x/3', lambda x: mp.sin(x) - x / 3, (0.5, 4)),
    ('x^5 - x - 1', lambda x: x ** 5 - x - 1, (-2, 2)), ('atan(x) - 1/2', lambda x: mp.atan(x) - mp.mpf(1) / 2, (-3, 3)), ('tanh(x) - x/2 - 0.1', lambda x: mp.tanh(x) - x / 2 - mp.mpf('0.1'), (-3, 3)),
    ('sqrt(x) - cos(x)', lambda x: mp.sqrt(x) - mp.cos(x), (0, 4)), ('x^2 - 2', lambda x: x ** 2 - 2, (0, 3)), ('exp(-x) - x^2', lambda x: mp.exp(-x) - x ** 2, (0, 3))]


def gen_findroot(rng, n):
    out = []
    i = -1
    while len(out) < n:
        i += 1
        if i % 8 == 7:
            kind = rng.randint(0, 2)
            if kind == 0:
                out.append(Case('search.findroot', ('tan(x)', 1, 2, 10), {}, REFUSE, 'the sign change at 1..2 is a pole of tan (pi/2), not a root: a refusal is right', None))
            elif kind == 1:
                out.append(Case('search.findroot', ('1/(x-1/2)', 0, 1, 10), {}, REFUSE, 'the sign change is a pole: a refusal is right', None))
            else:
                out.append(Case('search.findroot', ('x^2 + 1', -1, 2, 10), {}, BAD, 'no sign change on the bracket: BadInput is right', None))
            continue
        expr, f, (lo, hi) = _ROOT_T[i % len(_ROOT_T)]
        tries = 0
        while True:
            a = Fraction(rng.randint(int(lo * 4), int(hi * 4) - 1), 4)
            b = a + Fraction(rng.randint(1, 16), 4)
            tries += 1
            if tries > 200:
                a, b = Fraction(lo).limit_denominator(4), Fraction(hi).limit_denominator(4)
                break
            if (f(_mpq(a)) > 0) != (f(_mpq(b)) > 0):
                break
        if (f(_mpq(a)) > 0) == (f(_mpq(b)) > 0):
            continue
        digits = rng.choice((8, 12, 15, 20, 30))

        def cmpf(r, o, f=f, digits=digits):
            if not isinstance(r, Numeric):
                return False
            v, e = r.value, r.error
            q = quantum(v)
            lo_, hi_ = v - e - q, v + e + q
            return (f(mp.mpf(str(lo_))) > 0) != (f(mp.mpf(str(hi_))) > 0)
        out.append(Case('search.findroot', (expr, a, b, digits), {}, None, 'certificate: mpmath evaluates f at 70 digits and finds a sign change within value +/- (error + one unit of the last digit)', cmpf))
    return out


def gen_ode_ivp(rng, n):
    out = []
    for i in range(n):
        digits = rng.choice((8, 10, 12))
        kind = i % 9
        t1 = Fraction(rng.randint(1, 8), 4)
        if kind == 0:
            y0 = Fraction(rng.randint(1, 4))
            o = (mpd(_mpq(y0) * mp.exp(_mpq(t1))),)
            out.append(Case('search.ode_ivp', ('y', y0, 0, t1, digits), {}, o, 'exact solution y0*e^t (mpmath)', num_cmp))
        elif kind == 1:
            o = (mpd(mp.exp(-_mpq(t1) ** 2)),)
            out.append(Case('search.ode_ivp', ('-2*t*y', 1, 0, t1, digits), {}, o, 'exact solution e^(-t^2) (mpmath)', num_cmp))
        elif kind == 2:
            tt = Fraction(rng.randint(1, 3), 4)
            o = (mpd(1 / (1 - _mpq(tt))),)
            out.append(Case('search.ode_ivp', ('y^2', 1, 0, tt, digits), {}, o, 'exact solution 1/(1-t) (mpmath)', num_cmp))
        elif kind == 3:
            o = (mpd(2 * mp.exp(_mpq(t1)) - _mpq(t1) - 1),)
            out.append(Case('search.ode_ivp', ('t + y', 1, 0, t1, digits), {}, o, 'exact solution 2e^t - t - 1 (mpmath)', num_cmp))
        elif kind == 4:
            o = (mpd(1 / (1 + mp.exp(-_mpq(t1)))),)
            out.append(Case('search.ode_ivp', ('y*(1-y)', Fraction(1, 2), 0, t1, digits), {}, o, 'exact logistic solution (mpmath)', num_cmp))
        elif kind == 5:
            o = (mpd(mp.sin(_mpq(t1))), mpd(mp.cos(_mpq(t1))))
            out.append(Case('search.ode_ivp', (['y2', '-y1'], [0, 1], 0, t1, digits), {}, o, 'exact solution (sin t, cos t) (mpmath)', num_cmp))
        elif kind == 6:                                               # Lotka-Volterra against mpmath.odefun (Taylor series method, independent)
            tt = Fraction(rng.randint(2, 8), 4)
            f = mp.odefun(lambda t, y: [y[0] - y[0] * y[1], -y[1] + y[0] * y[1]], 0, [mp.mpf(2), mp.mpf(1)], tol=mp.mpf(10) ** -25)
            v = f(_mpq(tt))
            out.append(Case('search.ode_ivp', (['y1 - y1*y2', '-y2 + y1*y2'], [2, 1], 0, tt, digits), {}, (mpd(v[0]), mpd(v[1])), 'mpmath.odefun (Taylor series method) at 1e-25', num_cmp))
        elif kind == 7:                                               # backwards in time and a zero-length span
            if rng.random() < 0.5:
                o = (mpd(mp.exp(-_mpq(t1))),)
                out.append(Case('search.ode_ivp', ('y', 1, t1, 0, digits), {}, o, 'exact solution, integrating backwards (mpmath)', num_cmp))
            else:
                out.append(Case('search.ode_ivp', ('y^2', 1, 0, 2, 8), {}, REFUSE, 'the solution 1/(1-t) blows up at t = 1 inside [0, 2]: a refusal is right', None))
        else:                                                         # nonlinear pendulum-like: against mpmath.odefun
            tt = Fraction(rng.randint(2, 8), 4)
            f = mp.odefun(lambda t, y: [y[1], -mp.sin(y[0])], 0, [mp.mpf(1), mp.mpf(0)], tol=mp.mpf(10) ** -25)
            v = f(_mpq(tt))
            out.append(Case('search.ode_ivp', (['y2', '-sin(y1)'], [1, 0], 0, tt, digits), {}, (mpd(v[0]), mpd(v[1])), 'mpmath.odefun (Taylor series method) at 1e-25', num_cmp))
    return out


def gen_normal_cdf(rng, n):
    out = []
    for i in range(n):
        x = Fraction(rng.randint(-60, 60), rng.choice((4, 8, 5)))
        mu = Fraction(rng.randint(-3, 3)) if i % 3 == 1 else 0
        sg = Fraction(rng.randint(1, 5), 2) if i % 3 == 1 else 1
        if i % 7 == 6:
            x = Fraction(rng.randint(-37, -10))                      # deep tails
        digits = rng.choice((8, 15, 20, 30, 40))
        z = (_mpq(x) - _mpq(mu)) / _mpq(sg)
        o = mpd(mp.ncdf(z))
        out.append(Case('search.normal_cdf', (x, mu, sg, digits), {}, o, 'mpmath.ncdf at 70 digits (significant digits)', num_cmp))
    return out


def gen_normal_ppf(rng, n):
    out = []
    for i in range(n):
        if i % 4 == 0:
            p = Fraction(rng.randint(1, 999), 1000)
        elif i % 4 == 1:
            p = Fraction(1, 10 ** rng.randint(2, 30))
        elif i % 4 == 2:
            p = 1 - Fraction(1, 10 ** rng.randint(2, 15))
        else:
            p = Fraction(rng.randint(1, 99), 100)
        digits = rng.choice((8, 15, 20))

        def cmpf(r, o, p=p):
            if not isinstance(r, Numeric):
                return False
            v = mp.mpf(str(r.value))
            q = mp.mpf(str(quantum(r.value)))
            pl, ph = mp.ncdf(v - q), mp.ncdf(v + q)
            pp = _mpq(p)
            return (pl <= pp <= ph) or abs(pl - pp) <= mp.mpf(10) ** -60 or abs(ph - pp) <= mp.mpf(10) ** -60
        out.append(Case('search.normal_ppf', (p, digits), {}, None, 'certificate: mpmath.ncdf at 70 digits brackets p within one unit of the last digit', cmpf))
    return out


def gen_poisson_pmf(rng, n):
    out = []
    for _ in range(n):
        lam = Fraction(rng.randint(0, 60), rng.choice((1, 2, 4)))
        k = rng.randint(0, 80)
        digits = rng.choice((8, 15, 25, 40))
        l = _mpq(lam)
        o = mpd(mp.exp(-l) * l ** k / mp.factorial(k)) if (lam > 0 or k > 0) else Decimal(1)
        out.append(Case('search.poisson_pmf', (lam, k, digits), {}, o, 'mpmath exp(-lam) lam^k / k! at 70 digits', num_cmp))
    return out


def gen_poisson_cdf(rng, n):
    out = []
    for _ in range(n):
        lam = Fraction(rng.randint(0, 60), rng.choice((1, 2, 4)))
        k = rng.randint(0, 80)
        digits = rng.choice((8, 15, 25, 40))
        l = _mpq(lam)
        o = mpd(mp.gammainc(k + 1, l, regularized=True)) if lam > 0 else Decimal(1)
        out.append(Case('search.poisson_cdf', (lam, k, digits), {}, o, 'mpmath regularised upper incomplete gamma Q(k+1, lam) at 70 digits', num_cmp))
    return out


def gen_hard(rng, n):
    """Hard fixed cases (the same for every seed): sharp, singular, oscillating, huge, tiny. An answer must be right; a refusal is accepted (EITHER)."""
    mpm = mp.mpf
    out = []

    def add(op_, args, kw, truth, why):
        if isinstance(truth, tuple):
            o = tuple(t if isinstance(t, Decimal) else mpd(t) for t in truth)
        else:
            o = truth if isinstance(truth, Decimal) else mpd(truth)
        out.append(Case(op_, args, kw, (EITHER, o), why, None))
    add('search.integrate', ('1/sqrt(abs(x-0.3))', 0, 1, 12), {'points': ['0.3']}, 2 * mp.sqrt(mpm('0.3')) + 2 * mp.sqrt(mpm('0.7')), 'closed form (singularity inside, split)')
    add('search.integrate', ('1/sqrt(abs(x-0.3))', 0, 1, 12), {}, 2 * mp.sqrt(mpm('0.3')) + 2 * mp.sqrt(mpm('0.7')), 'closed form (singularity inside, not split: a refusal is accepted)')
    add('search.integrate', ('abs(x)/x', -1, 2, 12), {}, 1, 'closed form (jump inside)')
    add('search.integrate', ('sin(100*x)', 0, 10, 12), {}, (1 - mp.cos(1000)) / 100, 'closed form')
    add('search.integrate', ('cos(x^2)', 0, 5, 12), {}, mp.quad(lambda x: mp.cos(x ** 2), mp.linspace(0, 5, 60)), 'mpmath.quad on 60 sub-intervals')
    add('search.integrate', ('exp(-x)*sin(x)^2', 0, 'inf', 15), {}, mpm(2) / 5, 'closed form')
    add('search.integrate', ('x^2*exp(-x^2)', '-inf', 'inf', 15), {}, mp.sqrt(mp.pi) / 2, 'closed form')
    add('search.integrate', ('log(x)*log(1-x)', 0, 1, 15), {}, 2 - mp.pi ** 2 / 6, 'closed form')
    add('search.integrate', ('sqrt(x)*log(x)', 0, 1, 15), {}, -mpm(4) / 9, 'closed form')
    add('search.integrate', ('x^(-0.9)', 0, 1, 10), {}, 10, 'closed form (almost non-integrable: a refusal is accepted)')
    add('search.integrate', ('1/(1+x^2)', 0, 'inf', 40), {}, mp.pi / 2, 'closed form, 40 digits')
    add('search.integrate', ('exp(-x^2)', 0, 1, 40), {}, mp.sqrt(mp.pi) / 2 * mp.erf(1), 'closed form, 40 digits')
    add('search.integrate', ('1/x^1.1', 1, 'inf', 10), {}, 10, 'closed form (slow tail: a refusal is accepted)')
    add('search.integrate', ('exp(-1000*x)', 0, 1, 12), {}, (1 - mp.exp(-1000)) / 1000, 'closed form (sharp)')
    add('search.integrate', ('x/(exp(x)-1)', 0, 'inf', 20), {}, mp.pi ** 2 / 6, 'closed form (cancellation at 0)')
    add('search.derivative', ('exp(50*x)', 2, 15), {}, 50 * mp.exp(100), 'closed form (huge value)')
    add('search.derivative', ('x^100', '1.01', 15), {}, 100 * mpm('1.01') ** 99, 'closed form')
    add('search.derivative', ('log(x)', '1/1000', 15), {}, 1000, 'closed form (near a singularity)')
    add('search.derivative', ('sin(100*x)', 1, 15), {}, 100 * mp.cos(100), 'closed form (fast wiggle)')
    add('search.derivative', ('sin(x)', 1000, 15, 2), {}, -mp.sin(1000), 'closed form (large argument)')
    add('search.derivative', ('atan(x)', 0, 50), {}, 1, 'closed form, 50 digits')
    add('search.derivative', ('x^3', 1, 15, 6), {}, 0, 'closed form (order 6)')
    add('search.limit', ('sin(x)/x', 0, 40), {}, 1, 'closed form, 40 digits')
    add('search.limit', ('(1+x)^(1/x)', 0, 30), {}, mp.e, 'closed form')
    add('search.limit', ('x^x', 0, 10), {}, 1, 'closed form (log terms: a refusal is accepted)')
    add('search.limit', ('x*sin(1/x)', 0, 10), {}, 0, 'closed form (squeeze: a refusal is accepted)')
    add('search.limit', ('(2^x-1)/x', 0, 15), {}, mp.log(2), 'closed form')
    add('search.limit', ('(x^2+1)/(2*x^2-3)', 'inf', 15), {}, mpm('0.5'), 'closed form')
    add('search.limit', ('(1+1/x)^x', '-inf', 10), {}, mp.e, 'closed form')
    add('search.limit', ('sqrt(x)', 0, 10), {}, 0, 'closed form')
    add('search.limit', ('(sin(x)-x)/x^3', 0, 20), {}, -mpm(1) / 6, 'closed form')
    add('search.series_sum', ('cos(n)/n^2', 1, 'inf', 12), {}, mp.pi ** 2 / 6 - mp.pi / 2 + mpm(1) / 4, 'closed form (oscillating signs: a refusal is accepted)')
    add('search.series_sum', ('1/(n^2+1000)', 1, 'inf', 12), {}, mp.nsum(lambda n: 1 / (n ** 2 + 1000), [1, mp.inf]), 'mpmath.nsum')
    add('search.series_sum', ('1/(n*(n+1)*(n+2))', 1, 'inf', 15), {}, mpm(1) / 4, 'closed form')
    add('search.series_sum', ('(-1)^n/log(n+1)', 1, 'inf', 10), {}, mp.nsum(lambda n: (-1) ** int(n) / mp.log(n + 1), [1, mp.inf]), 'mpmath.nsum')
    add('search.series_sum', ('n^2/(n^4+1)', 1, 'inf', 12), {}, mp.nsum(lambda n: n ** 2 / (n ** 4 + 1), [1, mp.inf]), 'mpmath.nsum')
    add('search.series_sum', ('1/n^1.5', 1, 'inf', 8), {}, mp.zeta(1.5), 'closed form zeta(3/2) (slow power: a refusal is accepted)')
    add('search.series_sum', ('0.99^n', 0, 'inf', 12), {}, 100, 'closed form (ratio 0.99)')
    add('search.series_sum', ('0.999^n', 0, 'inf', 10), {}, 1000, 'closed form (ratio 0.999: a refusal is accepted)')
    add('search.series_sum', ('1/n!', 0, 'inf', 20), {}, mp.e, 'closed form')
    add('search.series_sum', ('2^n/n!', 0, 'inf', 15), {}, mp.exp(2), 'closed form')
    add('search.findroot', ('x^3', -1, 2, 12), {}, 0, 'closed form (triple root)')
    add('search.findroot', ('sin(x)', 3, 4, 30), {}, mp.pi, 'closed form')
    add('search.findroot', ('1e-20*(x-0.3)', 0, 1, 12), {}, mpm('0.3'), 'closed form (tiny slope)')
    add('search.findroot', ('exp(x)-1e6', 0, 20, 40), {}, mp.log(10 ** 6), 'closed form, 40 digits')
    add('search.ode_ivp', ('y', 1, 0, 10, 10), {}, (mp.exp(10),), 'closed form (growth e^10)')
    add('search.ode_ivp', ('-y', 1, 0, 20, 10), {}, (mp.exp(-20),), 'closed form (decay)')
    add('search.ode_ivp', ('1/y', 1, 0, 2, 10), {}, (mp.sqrt(5),), 'closed form y = sqrt(1+2t)')
    add('search.ode_ivp', (['y2', '-y1'], [1, 0], 0, 100, 8), {}, (mp.cos(100), -mp.sin(100)), 'closed form (cos t, -sin t) over 16 periods')
    return out


GEN = {name[4:]: fn for name, fn in globals().items() if name.startswith('gen_')}
GEN['lp_solve_sympy'] = gen_lp_solve_sympy


def _bucket(msg):
    return re.sub(r'\d+(\.\d+)?(E[+-]?\d+)?', '#', msg)[:90]


def run_case(c):
    """-> (outcome, detail, honest) where outcome is 'agree', 'disagree', 'refused', 'refused_expected', 'error'."""
    try:
        res = run(c.op, *c.args, **c.kw)
    except NotCovered as e:
        if c.oracle == REFUSE:
            return 'refused_expected', str(e), None
        if isinstance(c.oracle, tuple) and len(c.oracle) == 2 and c.oracle[0] == EITHER:
            return 'refused', str(e), None
        return 'refused', str(e), None
    except BadInput as e:
        if c.oracle == BAD:
            return 'agree', 'bad input expected', None
        return 'error', f'BadInput: {e}', None
    except Exception as e:                                           # a crash is a bug
        return 'error', f'{type(e).__name__}: {e}', None
    if c.oracle == REFUSE or c.oracle == BAD:
        return 'disagree', f'answered {res!r} where a refusal was right', None
    cmp = c.cmp
    oracle = c.oracle
    if isinstance(oracle, tuple) and len(oracle) == 2 and oracle[0] == EITHER:
        oracle = oracle[1]
        cmp = num_cmp
    try:
        v = cmp(res, oracle)
    except Exception as e:
        return 'error', f'compare crashed: {type(e).__name__}: {e}', None
    honest = None
    if isinstance(v, tuple):
        v, honest = v
    return ('agree' if v else 'disagree'), (None if v else f'got {res!r} expected {oracle!r}'), honest


def generate(seed, n):
    cases = []
    for name, fn in GEN.items():
        rng = random.Random(f'{seed}:{name}')
        cases += fn(rng, n)
    return cases


def run_all(seed=1, n=40, only=None, verbose=False):
    t_all = time.time()
    per = {}
    ops_registered = sorted(k for k in registry() if k.startswith('search.'))
    for name, fn in GEN.items():
        if only and name not in only:
            continue
        rng = random.Random(f'{seed}:{name}')
        t0 = time.time()
        cases = fn(rng, n)
        for c in cases:
            r = per.setdefault(c.op, dict(generated=0, answered=0, agree=0, refused={}, refused_expected=0, disagree=0, errors=0, error_estimate_held=0, error_estimate_checked=0,
                                          seconds=0.0, oracles=set(), failures=[], refused_examples={}))
            t1 = time.time()
            out, detail, honest = run_case(c)
            r['seconds'] += time.time() - t1
            r['generated'] += 1
            r['oracles'].add(c.oracle_name)
            if out == 'agree':
                r['answered'] += 1
                r['agree'] += 1
            elif out == 'disagree':
                r['answered'] += 1
                r['disagree'] += 1
                r['failures'].append({'args': repr(c.args)[:300], 'kw': repr(c.kw), 'detail': detail[:400]})
            elif out == 'error':
                r['errors'] += 1
                r['failures'].append({'args': repr(c.args)[:300], 'kw': repr(c.kw), 'detail': detail[:400]})
            elif out == 'refused_expected':
                r['refused_expected'] += 1
                r['agree'] += 0
            else:
                b = _bucket(detail)
                r['refused'][b] = r['refused'].get(b, 0) + 1
                r['refused_examples'].setdefault(b, [])
                if len(r['refused_examples'][b]) < 3:
                    r['refused_examples'][b].append(f'{c.args!r} {c.kw!r}'[:200])
            if honest is not None:
                r['error_estimate_checked'] += 1
                r['error_estimate_held'] += 1 if honest else 0
            if verbose and out in ('disagree', 'error'):
                print('FAIL', c.op, c.args, c.kw, detail, file=sys.stderr)
    for k, r in per.items():
        r['oracles'] = sorted(r['oracles'])
        r['seconds'] = round(r['seconds'], 2)
    rep = {'module': 'search', 'seed': seed, 'n': n, 'seconds': round(time.time() - t_all, 1), 'operations_registered': ops_registered,
           'operations_without_case': [o for o in ops_registered if o not in per], 'per_operation': per}
    return rep


def print_table(rep):
    print(f"{'operation':34} {'gen':>4} {'answ':>4} {'agree':>5} {'refuse':>6} {'exp.ref':>7} {'DISAGREE':>8} {'err':>3} {'err-est':>9} {'sec':>6}")
    tot = dict(generated=0, answered=0, agree=0, refused=0, refused_expected=0, disagree=0, errors=0)
    for op_, r in sorted(rep['per_operation'].items()):
        nref = sum(r['refused'].values())
        est = f"{r['error_estimate_held']}/{r['error_estimate_checked']}" if r['error_estimate_checked'] else '-'
        print(f"{op_:34} {r['generated']:>4} {r['answered']:>4} {r['agree']:>5} {nref:>6} {r['refused_expected']:>7} {r['disagree']:>8} {r['errors']:>3} {est:>9} {r['seconds']:>6}")
        for k in ('generated', 'answered', 'agree', 'refused_expected', 'disagree', 'errors'):
            tot[k] += r[k]
        tot['refused'] += nref
    print(f"{'TOTAL':34} {tot['generated']:>4} {tot['answered']:>4} {tot['agree']:>5} {tot['refused']:>6} {tot['refused_expected']:>7} {tot['disagree']:>8} {tot['errors']:>3}")
    print('operations registered:', len(rep['operations_registered']), '| without a case:', rep['operations_without_case'] or 'none', '| seconds:', rep['seconds'])
    for op_, r in sorted(rep['per_operation'].items()):
        if r['refused']:
            print(f'  refusals of {op_}:')
            for reason, cnt in sorted(r['refused'].items(), key=lambda kv: -kv[1]):
                print(f'    {cnt:3}  {reason}')
                for ex in r['refused_examples'].get(reason, []):
                    print(f'           e.g. {ex}')
        for f in r['failures'][:3]:
            print(f'  FAILURE {op_}: {f}')


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--n', type=int, default=40)
    ap.add_argument('--only', nargs='*')
    ap.add_argument('--out', default='build-mathbackend/search.json')
    ap.add_argument('-v', '--verbose', action='store_true')
    a = ap.parse_args(argv)
    rep = run_all(a.seed, a.n, a.only, a.verbose)
    p = Path(a.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(rep, indent=1, default=str), encoding='utf-8')
    print_table(rep)
    bad = sum(r['disagree'] + r['errors'] for r in rep['per_operation'].values())
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
