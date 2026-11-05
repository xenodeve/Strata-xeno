"""MIP math backend, module `search`: SAT, constraint search, dynamic programming, exact LP/ILP, graph algorithms, exact statistics, numeric calculus.

Written for this repository from the textbook algorithms (DPLL, backtracking, the two-phase simplex with Bland's rule, Dijkstra / Bellman-Ford / Floyd-Warshall,
Edmonds-Karp, Kruskal, the Hungarian method, tanh-sinh quadrature, Neville extrapolation, Gragg-Bulirsch-Stoer, Cohen-Villegas-Zagier). No third-party code is copied;
see third_party/mathbackend/NOTICES_search.md. Standard library only.

Exact operations take int, Fraction or str ('3/4', '0.125') and return int or Fraction; a float is refused (BadInput) because it is not exact.
Every numeric operation returns a `Numeric(value, error)`: `value` is a decimal.Decimal (a tuple of them for ode_ivp) rounded to the digits asked, `error` is the
operation's own estimate of its absolute error (it is an estimate from the method, not a proof; the simulation checks it against an independent oracle), and
`.numeric` is always True. Digits mean: the result is within one unit of its last printed digit, where the last digit is the `digits`-th significant digit when
|value| >= 1 and the `digits`-th decimal place when |value| < 1 ('abs' accuracy); probabilities (normal_cdf, poisson_*) are `digits` significant digits ('rel').
Functions of x are given as text ('sin(x)*exp(-x^2)', `^` is a power); names: x (or the named variables), pi, e, sin cos tan asin acos atan sinh cosh tanh asinh acosh atanh
exp log ln log10 log2 sqrt cbrt abs erf min max. Implicit multiplication ('2x') is not read.
Every operation has a bound (size, steps or nodes) and raises NotCovered past it; undefined input raises BadInput.
"""
import ast
import heapq
import re
from collections import deque
from decimal import Context, Decimal, Overflow, ROUND_HALF_EVEN, getcontext, localcontext
from fractions import Fraction
from functools import lru_cache
from math import comb, factorial, gcd, isqrt
from typing import NamedTuple

from tools.igc.mathbackend import BadInput, NotCovered, Surd, op

# ----------------------------------------------------------------------------------------------------------------------------------------
# common helpers
# ----------------------------------------------------------------------------------------------------------------------------------------

NOT_COVERED = {
    'search.roots_in_interval': 'all real roots of a function in an interval cannot be bounded: a sign scan misses double roots and roots closer than the grid; use algebra.* for polynomials',
    'search.sat_certificate_unsat': 'a refutation proof (DRAT) is not produced; UNSAT is a complete DPLL search, bounded by a node budget',
    'search.minimize_nonconvex': 'global optimisation of a non-convex function has no exact bounded method',
    'search.integrate_oscillatory_infinite': 'integrals like int_0^inf sin(x)/x dx do not converge absolutely, so tanh-sinh refuses them',
    'search.limit_log_terms': 'limits whose approach has log terms are answered only when Wynn extrapolation converges to the digits asked (x*ln(x) at 0 does, x^x at 0 may not); otherwise refused',
    'search.ode_stiff_or_singular': 'stiff systems and solutions with a singularity inside the interval are refused (step budget); a right side that is not Lipschitz (sqrt(y) at y = 0) has several solutions and the one the integrator follows is returned, not detected',
    'search.polynomial_regression': 'not in the floor list; linear regression only',
    'search.sat_weighted_maxsat': 'MaxSAT / weighted SAT is not covered',
    'search.ilp_unbounded': 'an integer program whose LP relaxation is unbounded is refused: infeasible/unbounded cannot be told apart cheaply',
}

_MAX_DIGITS = 100


class Numeric(NamedTuple):
    """(value, error): a numeric result. `value` is a Decimal (a tuple of Decimals for ode_ivp), `error` the method's own estimate of the absolute error."""
    value: object
    error: Decimal

    @property
    def numeric(self):
        return True

    def __str__(self):
        return f'{self.value} (error ~ {self.error})'


def _q(x, what='value'):
    """int, Fraction, Decimal or str -> Fraction. A float is refused: it is not exact."""
    if isinstance(x, bool):
        raise BadInput(f'{what}: a bool is not a number')
    if isinstance(x, int):
        return Fraction(x)
    if isinstance(x, Fraction):
        return x
    if isinstance(x, Decimal):
        if not x.is_finite():
            raise BadInput(f'{what}: not finite')
        return Fraction(x)
    if isinstance(x, str):
        try:
            return Fraction(x.strip())
        except (ValueError, ZeroDivisionError):
            raise BadInput(f'{what}: cannot read {x!r} as a rational')
    if isinstance(x, float):
        raise BadInput(f'{what}: a float is not exact; pass an int, a Fraction or a string such as "0.1"')
    raise BadInput(f'{what}: need int, Fraction or str, got {type(x).__name__}')


def _int(x, what='value'):
    q = _q(x, what)
    if q.denominator != 1:
        raise BadInput(f'{what}: need an integer, got {q}')
    return int(q)


def _qlist(xs, what='data'):
    if isinstance(xs, (str, bytes)) or not hasattr(xs, '__iter__'):
        raise BadInput(f'{what}: need a list')
    return [_q(v, what) for v in xs]


def _clean(q):
    """A Fraction with denominator 1 becomes an int (graph and DP results)."""
    return int(q) if isinstance(q, Fraction) and q.denominator == 1 else q


def _nkey(v):
    return (isinstance(v, str), v)


def _check_digits(digits):
    if not isinstance(digits, int) or isinstance(digits, bool) or digits < 1:
        raise BadInput('digits must be a positive integer')
    if digits > _MAX_DIGITS:
        raise NotCovered(f'digits > {_MAX_DIGITS}')
    return digits


class _Budget:
    """A work counter: `tick(k)` raises NotCovered past the limit, so no operation hangs."""
    __slots__ = ('left', 'what')

    def __init__(self, limit, what='search limit'):
        self.left, self.what = limit, what

    def tick(self, k=1):
        self.left -= k
        if self.left < 0:
            raise NotCovered(self.what)


# ----------------------------------------------------------------------------------------------------------------------------------------
# SAT: DPLL with unit propagation (model finding, model counting)
# ----------------------------------------------------------------------------------------------------------------------------------------

def _read_cnf(clauses, nvars):
    if isinstance(clauses, (str, bytes)) or not hasattr(clauses, '__iter__'):
        raise BadInput('clauses: need a list of lists of non-zero ints (DIMACS literals)')
    out, top = [], 0
    for cl in clauses:
        lits = []
        for l in cl:
            if isinstance(l, bool) or not isinstance(l, int) or l == 0:
                raise BadInput(f'bad literal {l!r}: literals are non-zero ints')
            lits.append(l)
            top = max(top, abs(l))
        s = set(lits)
        if any(-l in s for l in s):
            continue                      # a tautology: always true
        out.append(tuple(sorted(s, key=lambda v: (abs(v), v))))   # an empty clause stays: it is false, so the formula is unsatisfiable
    n = top if nvars is None else _int(nvars, 'nvars')
    if n < top:
        raise BadInput('nvars is smaller than the largest variable in the clauses')
    return out, n


def _assign(clauses, lit):
    """The clauses after `lit` is made true: satisfied clauses go, -lit is struck out; None on an empty clause."""
    out = []
    for c in clauses:
        if lit in c:
            continue
        if -lit in c:
            c = tuple(x for x in c if x != -lit)
            if not c:
                return None
        out.append(c)
    return out


def _propagate(clauses, trail, bud, pure):
    """Unit propagation (and pure literals when `pure`) to a fixed point. Returns the remaining clauses or None on a conflict."""
    while True:
        bud.tick(len(clauses) + 1)
        unit = None
        for c in clauses:
            if len(c) == 1:
                unit = c[0]
                break
        if unit is None and pure:
            pos, neg = set(), set()
            for c in clauses:
                for l in c:
                    (pos if l > 0 else neg).add(abs(l))
            both = pos & neg
            for v in sorted(pos | neg):
                if v not in both:
                    unit = v if v in pos else -v
                    break
        if unit is None:
            return clauses
        trail.append(unit)
        clauses = _assign(clauses, unit)
        if clauses is None:
            return None


def _branch_var(clauses):
    """The variable that occurs most in the shortest clauses (ties: the smallest variable)."""
    m = min(len(c) for c in clauses)
    score = {}
    for c in clauses:
        if len(c) == m:
            for l in c:
                score[abs(l)] = score.get(abs(l), 0) + 1
    return min(score, key=lambda v: (-score[v], v))


def _dpll_solve(clauses, trail, bud):
    clauses = _propagate(clauses, trail, bud, True)
    if clauses is None:
        return False
    if not clauses:
        return True
    v = _branch_var(clauses)
    for lit in (v, -v):
        mark = len(trail)
        trail.append(lit)
        sub = _assign(clauses, lit)
        if sub is not None and _dpll_solve(sub, trail, bud):
            return True
        del trail[mark:]
    return False


def _dpll_count(clauses, nfree, bud):
    """Number of assignments to the `nfree` variables not yet fixed that satisfy `clauses`."""
    trail = []
    clauses = _propagate(clauses, trail, bud, False)
    if clauses is None:
        return 0
    nfree -= len(trail)
    if not clauses:
        return 1 << nfree
    inside = {abs(l) for c in clauses for l in c}
    free = nfree - len(inside)
    v = _branch_var(clauses)
    total = 0
    for lit in (v, -v):
        sub = _assign(clauses, lit)
        if sub is not None:
            total += _dpll_count(sub, len(inside) - 1, bud)
    return total << free


@op('search.sat_solve', kind='satisfiability', bound='<= 400 variables, <= 5000 clauses, 3e7 literal visits')
def sat_solve(clauses, nvars=None):
    """Satisfiability by DPLL with unit propagation and pure literals. `clauses` are DIMACS lists of non-zero ints.
    Returns {'sat': bool, 'model': [signed literal for variable 1..nvars] or None}; the model satisfies every clause (checked before returning).
    sat_solve([[1, 2], [-1, 2], [1, -2]]) -> {'sat': True, 'model': [1, 2]}"""
    cl, n = _read_cnf(clauses, nvars)
    if n > 400 or len(cl) > 5000:
        raise NotCovered('too large: more than 400 variables or 5000 clauses')
    if any(len(c) == 0 for c in cl):
        return {'sat': False, 'model': None}
    if not cl:
        return {'sat': True, 'model': [-v for v in range(1, n + 1)]}
    bud = _Budget(30_000_000, 'search limit (3e7 literal visits)')
    trail = []
    if not _dpll_solve(cl, trail, bud):
        return {'sat': False, 'model': None}
    val = {abs(l): l > 0 for l in trail}
    model = [v if val.get(v, False) else -v for v in range(1, n + 1)]
    ms = set(model)
    if not all(any(l in ms for l in c) for c in cl):          # a safety net: a model that fails is a bug, never an answer
        raise NotCovered('internal check failed: model does not satisfy the clauses')
    return {'sat': True, 'model': model}


@op('search.sat_count', kind='satisfiability', bound='<= 120 variables, <= 2000 clauses, 3e7 literal visits')
def sat_count(clauses, nvars=None):
    """The number of assignments of variables 1..nvars that satisfy every clause (exact model counting by DPLL with unit propagation).
    sat_count([[1, 2]]) -> 3     (of the four assignments of two variables, all but both-false)"""
    cl, n = _read_cnf(clauses, nvars)
    if n > 120 or len(cl) > 2000:
        raise NotCovered('too large: more than 120 variables or 2000 clauses')
    if any(len(c) == 0 for c in cl):
        return 0
    return _dpll_count(cl, n, _Budget(30_000_000, 'search limit (3e7 literal visits)'))


# ----------------------------------------------------------------------------------------------------------------------------------------
# exhaustive constraint search with all-different and arithmetic constraints
# ----------------------------------------------------------------------------------------------------------------------------------------

class _Undefined(ArithmeticError):
    pass


def _cdiv(a, b):
    if b == 0:
        raise _Undefined('division by zero')
    return Fraction(a) / Fraction(b)


def _cpow(a, b):
    if isinstance(b, Fraction):
        if b.denominator != 1:
            raise _Undefined('non-integer power')
        b = int(b)
    if not isinstance(b, int) or abs(b) > 400:
        raise _Undefined('power out of range')
    if b >= 0:
        return a ** b
    if a == 0:
        raise _Undefined('zero to a negative power')
    return Fraction(1) / Fraction(a) ** (-b)


def _csum(*a):
    return sum(a)


_CALLS = {'abs': abs, 'min': min, 'max': max, 'sum': _csum, 'gcd': gcd}
_OK_NODES = (ast.Expression, ast.BoolOp, ast.BinOp, ast.UnaryOp, ast.Compare, ast.Call, ast.Name, ast.Constant, ast.Load, ast.IfExp,
             ast.And, ast.Or, ast.Add, ast.Sub, ast.Mult, ast.FloorDiv, ast.Mod, ast.Div, ast.Pow, ast.USub, ast.UAdd, ast.Not,
             ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE)


class _Rewrite(ast.NodeTransformer):
    def visit_BinOp(self, node):
        self.generic_visit(node)
        if isinstance(node.op, ast.Div):
            return ast.copy_location(ast.Call(ast.Name('_div', ast.Load()), [node.left, node.right], []), node)
        if isinstance(node.op, ast.Pow):
            return ast.copy_location(ast.Call(ast.Name('_pow', ast.Load()), [node.left, node.right], []), node)
        return node


_GLOBALS = {'__builtins__': {}, '_div': _cdiv, '_pow': _cpow, **_CALLS}


def _compile_int(text, names):
    """A constraint or objective text -> (code, set of variable names used). Python operators with exact meaning: / is an exact Fraction division,
    // and % floor, ^ is read as a power (not xor). Names must be in `names`. Only int constants and the calls abs min max sum gcd are allowed."""
    if not isinstance(text, str):
        raise BadInput('a constraint is a string such as "a + b == 10"')
    try:
        tree = ast.parse(text.strip().replace('^', '**'), mode='eval')
    except SyntaxError as e:
        raise BadInput(f'cannot read constraint {text!r}: {e.msg}')
    used = set()
    for nd in ast.walk(tree):
        if not isinstance(nd, _OK_NODES):
            raise BadInput(f'constraint {text!r}: {type(nd).__name__} is not allowed')
        if isinstance(nd, ast.Constant) and (isinstance(nd.value, float) or not isinstance(nd.value, (int, bool))):
            raise BadInput(f'constraint {text!r}: only integer constants (use a / b for a fraction)')
        if isinstance(nd, ast.Call):
            if not isinstance(nd.func, ast.Name) or nd.func.id not in _CALLS or nd.keywords:
                raise BadInput(f'constraint {text!r}: only the calls {sorted(_CALLS)} are allowed')
        elif isinstance(nd, ast.Name):
            if nd.id in _CALLS:
                continue
            if nd.id not in names:
                raise BadInput(f'constraint {text!r}: unknown name {nd.id!r}')
            used.add(nd.id)
    tree = ast.fix_missing_locations(_Rewrite().visit(tree))
    return compile(tree, '<constraint>', 'eval'), used


def _domain(spec, name):
    if isinstance(spec, str):
        m = re.fullmatch(r'\s*(-?\d+)\s*\.\.\s*(-?\d+)\s*', spec)
        if not m:
            raise BadInput(f'domain of {name}: write "lo..hi" or a list of ints')
        lo, hi = int(m.group(1)), int(m.group(2))
        vals = list(range(lo, hi + 1))
    else:
        vals = sorted({_int(v, f'domain of {name}') for v in spec})
    if len(vals) > 100000:
        raise NotCovered(f'domain of {name} is larger than 100000')
    return vals


class _CSP:
    def __init__(self, domains, constraints, all_different, nodes):
        if not isinstance(domains, dict) or not domains:
            raise BadInput('domains: a dict name -> list of ints or "lo..hi"')
        self.names = list(domains)
        for nm in self.names:
            if not isinstance(nm, str) or not nm.isidentifier() or nm in _CALLS:
                raise BadInput(f'bad variable name {nm!r}')
        if len(self.names) > 40:
            raise NotCovered('more than 40 variables')
        self.dom = [_domain(domains[nm], nm) for nm in self.names]
        pos = {nm: i for i, nm in enumerate(self.names)}
        self.at = [[] for _ in self.names]
        self.always = []
        for text in constraints:
            code, used = _compile_int(text, set(self.names))
            if not used:
                self.always.append(code)
            else:
                self.at[max(pos[u] for u in used)].append(code)
        self.peers = [[] for _ in self.names]          # all-different: the earlier variables of every group that holds this one
        for grp in all_different:
            grp = list(grp)
            if any(g not in pos for g in grp) or len(set(grp)) != len(grp):
                raise BadInput(f'all_different group {grp!r}: unknown or repeated variable')
            members = sorted(pos[g] for g in grp)
            for k, i in enumerate(members):
                self.peers[i].extend(members[:k])
        self.bud = _Budget(nodes, f'search limit ({nodes} nodes)')
        self.env = {}

    def _ok(self, codes):
        env = self.env
        for code in codes:
            try:
                if not eval(code, _GLOBALS, env):
                    return False
            except (ArithmeticError, ValueError, OverflowError):
                return False                              # an undefined value (division by zero ...) does not satisfy a constraint
        return True

    def walk(self, leaf):
        """Calls leaf(values) for every solution in lexicographic order of the variables as given; leaf returns True to stop."""
        if not self._ok(self.always):
            return
        n, env, names, dom, at, peers, bud = len(self.names), self.env, self.names, self.dom, self.at, self.peers, self.bud
        vals = [0] * n

        def rec(i):
            if i == n:
                return leaf(tuple(vals))
            nm, pr, cs = names[i], peers[i], at[i]
            for v in dom[i]:
                if pr and any(vals[j] == v for j in pr):
                    continue
                bud.tick()
                vals[i] = v
                env[nm] = v
                if cs and not self._ok(cs):
                    continue
                if rec(i + 1):
                    return True
            return False
        rec(0)


@op('search.csp_solutions', kind='constraint search', bound='<= 40 variables, domains <= 100000, 4e6 search nodes (about 15 s)')
def csp_solutions(domains, constraints=(), all_different=(), limit=1000):
    """Every assignment (up to `limit`) of integer variables that satisfies the constraints, found by backtracking in the order of `domains`.
    domains: {'a': '1..9', 'b': [2, 4, 6]}; constraints: strings such as 'a + b == 10' ('/' exact, '^' power, a constraint that divides by zero is not satisfied);
    all_different: lists of names. Returns {'solutions': [{'a': ..}, ...] in lexicographic order, 'complete': bool}.
    csp_solutions({'a': '1..3', 'b': '1..3'}, ['a < b']) -> {'solutions': [{'a': 1, 'b': 2}, {'a': 1, 'b': 3}, {'a': 2, 'b': 3}], 'complete': True}"""
    lim = _int(limit, 'limit')
    if lim < 1 or lim > 100000:
        raise BadInput('limit must be 1..100000')
    cs = _CSP(domains, list(constraints), [list(g) for g in all_different], 4_000_000)
    sols, state = [], {'cut': False}

    def leaf(vals):
        if len(sols) >= lim:
            state['cut'] = True
            return True
        sols.append(dict(zip(cs.names, vals)))
        return False
    cs.walk(leaf)
    return {'solutions': sols, 'complete': not state['cut']}


@op('search.csp_count', kind='constraint search', bound='<= 40 variables, domains <= 100000, 4e6 search nodes (about 15 s)')
def csp_count(domains, constraints=(), all_different=()):
    """The number of integer assignments that satisfy the constraints (same input as csp_solutions).
    csp_count({'a': '1..10', 'b': '1..10'}, ['a + b == 10']) -> 9"""
    cs = _CSP(domains, list(constraints), [list(g) for g in all_different], 4_000_000)
    cnt = [0]

    def leaf(vals):
        cnt[0] += 1
        return False
    cs.walk(leaf)
    return cnt[0]


@op('search.csp_optimize', kind='constraint search', bound='<= 40 variables, domains <= 100000, 4e6 search nodes (about 15 s)')
def csp_optimize(domains, objective, constraints=(), all_different=(), maximize=True):
    """The best value of an integer expression over all assignments that satisfy the constraints.
    Returns {'value': Fraction-or-int, 'solution': first optimal assignment in lexicographic order, 'optimal_count': how many assignments reach it} or {'value': None}.
    csp_optimize({'a': '0..5', 'b': '0..5'}, 'a*b', ['a + b <= 6']) -> {'value': 9, 'solution': {'a': 3, 'b': 3}, 'optimal_count': 1}"""
    cs = _CSP(domains, list(constraints), [list(g) for g in all_different], 4_000_000)
    obj, used = _compile_int(objective, set(cs.names))
    best = {'v': None, 'sol': None, 'n': 0}

    def leaf(vals):
        try:
            v = eval(obj, _GLOBALS, cs.env)
        except (ArithmeticError, ValueError, OverflowError):
            return False
        v = _clean(Fraction(v)) if not isinstance(v, bool) else int(v)
        b = best['v']
        if b is None or (v > b if maximize else v < b):
            best['v'], best['sol'], best['n'] = v, dict(zip(cs.names, vals)), 1
        elif v == b:
            best['n'] += 1
        return False
    cs.walk(leaf)
    if best['v'] is None:
        return {'value': None, 'solution': None, 'optimal_count': 0}
    return {'value': best['v'], 'solution': best['sol'], 'optimal_count': best['n']}


# ----------------------------------------------------------------------------------------------------------------------------------------
# dynamic programming: subset sum, knapsack, change-making
# ----------------------------------------------------------------------------------------------------------------------------------------

@op('search.subset_sum', kind='dynamic programming', bound='|sum range| <= 2e6, n <= 2000, 5e7 DP steps')
def subset_sum(values, target):
    """Subsets of `values` (integers, negatives allowed) that sum to `target`. Returns {'exists': bool, 'count': number of index subsets (the empty one
    counts for target 0), 'subset': one subset as a sorted list of indices, or None}.
    subset_sum([3, 34, 4, 12, 5, 2], 9) -> {'exists': True, 'count': 2, 'subset': [2, 4]}"""
    vals = [_int(v, 'values') for v in values]
    t = _int(target, 'target')
    n = len(vals)
    lo, hi = sum(v for v in vals if v < 0), sum(v for v in vals if v > 0)
    if hi - lo > 2_000_000 or n > 2000:
        raise NotCovered('too large: range of sums > 2e6 or more than 2000 values')
    if t < lo or t > hi:
        return {'exists': False, 'count': 0, 'subset': None}
    off = -lo
    width = hi - lo + 1
    mask = (1 << width) - 1
    layers = [1 << off]                                  # layers[i]: bitset of sums reachable with the first i values
    for v in vals:
        b = layers[-1]
        layers.append((b | (b << v if v >= 0 else b >> -v)) & mask)
    if not (layers[n] >> (t + off)) & 1:
        return {'exists': False, 'count': 0, 'subset': None}
    # one subset, walking back
    cur, pick = t, []
    for i in range(n - 1, -1, -1):
        if (layers[i] >> (cur + off)) & 1:
            continue
        pick.append(i)
        cur -= vals[i]
    pick.sort()
    # the count: a dictionary DP over reachable sums only
    bud = _Budget(50_000_000, 'search limit (5e7 DP steps)')
    ways = {0: 1}
    for v in vals:
        bud.tick(len(ways))
        new = dict(ways)
        for s, c in ways.items():
            new[s + v] = new.get(s + v, 0) + c
        ways = new
    return {'exists': True, 'count': ways.get(t, 0), 'subset': pick}


def _scaled_values(values):
    """Rational values -> (integer values, denominator)."""
    qs = [_q(v, 'values') for v in values]
    den = 1
    for q in qs:
        den = den * q.denominator // gcd(den, q.denominator)
    return [int(q * den) for q in qs], den


@op('search.knapsack', kind='dynamic programming', bound='capacity <= 1e6, items * capacity <= 1e7')
def knapsack(weights, values, capacity, kind='0/1', counts=None):
    """Best total value within an integer capacity. kind: '0/1' (each item once), 'unbounded', or 'bounded' (item i at most counts[i] times).
    Weights are positive ints, values rational. Returns {'value': best total, 'counts': how many of each item (one optimal choice)}.
    knapsack([1, 3, 4, 5], [1, 4, 5, 7], 7) -> {'value': 9, 'counts': [0, 1, 1, 0]}"""
    w = [_int(x, 'weights') for x in weights]
    iv, den = _scaled_values(values)
    cap = _int(capacity, 'capacity')
    if len(w) != len(iv):
        raise BadInput('weights and values differ in length')
    if cap < 0 or any(x <= 0 for x in w):
        raise BadInput('weights must be positive and the capacity non-negative')
    if kind not in ('0/1', 'unbounded', 'bounded'):
        raise BadInput("kind is '0/1', 'unbounded' or 'bounded'")
    if cap > 1_000_000 or len(w) * (cap + 1) > 10_000_000:
        raise NotCovered('too large: capacity > 1e6 or items * capacity > 1e7')
    n = len(w)
    if kind == 'unbounded':
        best = [0] * (cap + 1)
        pick = [-1] * (cap + 1)
        for c in range(1, cap + 1):
            bv, bi = best[c - 1], -2                       # -2: carry the previous capacity
            for i in range(n):
                if w[i] <= c and iv[i] > 0:
                    cand = best[c - w[i]] + iv[i]
                    if cand > bv:
                        bv, bi = cand, i
            best[c], pick[c] = bv, bi
        cnt, c = [0] * n, cap
        while c > 0:
            if pick[c] == -2:
                c -= 1
            else:
                cnt[pick[c]] += 1
                c -= w[pick[c]]
        return {'value': _clean(Fraction(best[cap], den)), 'counts': cnt}
    if kind == 'bounded':
        if counts is None or len(counts) != n:
            raise BadInput('bounded knapsack needs counts, one per item')
        lim = [_int(c, 'counts') for c in counts]
        if any(c < 0 for c in lim):
            raise BadInput('counts must be non-negative')
        items = []                                          # binary splitting: (original index, multiplicity, weight, value)
        for i in range(n):
            k, m = lim[i], 1
            while k > 0:
                t = min(m, k)
                items.append((i, t, w[i] * t, iv[i] * t))
                k -= t
                m *= 2
    else:
        items = [(i, 1, w[i], iv[i]) for i in range(n)]
    if len(items) * (cap + 1) > 10_000_000:
        raise NotCovered('too large: items * capacity > 1e7')
    best = [0] * (cap + 1)
    took = []
    for (_, _, wt, v) in items:
        flags = bytearray(cap + 1)
        if wt <= cap and v > 0:
            for c in range(cap, wt - 1, -1):
                cand = best[c - wt] + v
                if cand > best[c]:
                    best[c] = cand
                    flags[c] = 1
        took.append(flags)
    cnt, c = [0] * n, cap
    for k in range(len(items) - 1, -1, -1):
        if took[k][c]:
            i, mult, wt, _ = items[k]
            cnt[i] += mult
            c -= wt
    return {'value': _clean(Fraction(best[cap], den)), 'counts': cnt}


@op('search.coin_change_count', kind='dynamic programming', bound='amount * coins <= 2e7')
def coin_change_count(coins, amount, ordered=False, limits=None):
    """Number of ways to make `amount` from coin values (any number of each, or at most limits[i] of coin i). ordered=False counts multisets, ordered=True counts
    sequences (compositions). Duplicate coin values are the same coin. coin_change_count([1, 5, 10, 25, 50], 100) -> 292"""
    cs = [_int(c, 'coins') for c in coins]
    a = _int(amount, 'amount')
    if a < 0 or any(c <= 0 for c in cs):
        raise BadInput('coins must be positive and the amount non-negative')
    if limits is not None:
        if ordered:
            raise NotCovered('ordered counts with limits')
        lm = [_int(x, 'limits') for x in limits]
        if len(lm) != len(cs) or any(x < 0 for x in lm):
            raise BadInput('limits: one non-negative int per coin')
        if len(set(cs)) != len(cs):
            raise BadInput('limits with duplicate coin values are ambiguous')
        pairs = list(zip(cs, lm))
    else:
        pairs = [(c, None) for c in sorted(set(cs))]
    if a * max(1, len(pairs)) > 20_000_000:
        raise NotCovered('too large: amount * coins > 2e7')
    ways = [0] * (a + 1)
    ways[0] = 1
    if ordered:
        for s in range(1, a + 1):
            ways[s] = sum(ways[s - c] for c, _ in pairs if c <= s)
        return ways[a]
    for c, u in pairs:
        if u is None:
            for s in range(c, a + 1):
                ways[s] += ways[s - c]
        else:                                               # at most u of this coin: new[s] = new[s-c] + old[s] - old[s-(u+1)c]
            old = ways
            new = old[:]
            for s in range(c, a + 1):
                new[s] = new[s - c] + old[s] - (old[s - (u + 1) * c] if s - (u + 1) * c >= 0 else 0)
            ways = new
    return ways[a]


@op('search.coin_change_min', kind='dynamic programming', bound='amount * coins <= 2e7')
def coin_change_min(coins, amount):
    """The fewest coins that make `amount` (any number of each coin). Returns {'count': k or None when impossible, 'coins': {value: how many}}.
    coin_change_min([1, 5, 10, 25], 63) -> {'count': 6, 'coins': {1: 3, 10: 1, 25: 2}}"""
    cs = sorted({_int(c, 'coins') for c in coins})
    a = _int(amount, 'amount')
    if a < 0 or any(c <= 0 for c in cs):
        raise BadInput('coins must be positive and the amount non-negative')
    if a * max(1, len(cs)) > 20_000_000:
        raise NotCovered('too large: amount * coins > 2e7')
    INF = a + 2
    best = [INF] * (a + 1)
    best[0] = 0
    last = [0] * (a + 1)
    for s in range(1, a + 1):
        for c in cs:
            if c > s:
                break
            if best[s - c] + 1 < best[s]:
                best[s] = best[s - c] + 1
                last[s] = c
    if best[a] >= INF:
        return {'count': None, 'coins': {}}
    used, s = {}, a
    while s > 0:
        used[last[s]] = used.get(last[s], 0) + 1
        s -= last[s]
    return {'count': best[a], 'coins': dict(sorted(used.items()))}


# ----------------------------------------------------------------------------------------------------------------------------------------
# exact linear programming (two-phase simplex over Fractions, Bland's rule) and integer programming (branch and bound)
# ----------------------------------------------------------------------------------------------------------------------------------------

def _pivot(T, r, c):
    pr = T[r]
    pv = pr[c]
    if pv != 1:
        pr = T[r] = [x / pv for x in pr]
    for i, row in enumerate(T):
        if i != r:
            f = row[c]
            if f != 0:
                T[i] = [a - f * b for a, b in zip(row, pr)]


def _run_simplex(T, basis, allowed, bud):
    """Maximise using the last row of T as reduced costs (positive = improving). Bland's rule. Returns 'optimal' or 'unbounded'."""
    m = len(T) - 1
    while True:
        obj = T[m]
        enter = next((j for j in allowed if obj[j] > 0), None)
        if enter is None:
            return 'optimal'
        best, row = None, None
        for i in range(m):
            a = T[i][enter]
            if a > 0:
                ratio = T[i][-1] / a
                if best is None or ratio < best or (ratio == best and basis[i] < basis[row]):
                    best, row = ratio, i
        if row is None:
            return 'unbounded'
        bud.tick(m + 1)
        _pivot(T, row, enter)
        basis[row] = enter


def _simplex_core(cvec, rows, nvar, bud):
    """maximise cvec.y subject to rows [(coeffs, sense, rhs)] with y >= 0. Returns ('optimal', y) | ('infeasible', None) | ('unbounded', None)."""
    rows = [(list(a), s, b) for a, s, b in rows]
    for k, (a, s, b) in enumerate(rows):
        if b < 0:
            rows[k] = ([-x for x in a], {'<=': '>=', '>=': '<=', '=': '='}[s], -b)
    m = len(rows)
    nslack = sum(1 for _, s, _ in rows if s != '=')
    nart = sum(1 for _, s, _ in rows if s != '<=')
    total = nvar + nslack + nart
    T = []
    basis = []
    si, ai = nvar, nvar + nslack
    for a, s, b in rows:
        row = [Fraction(x) for x in a] + [Fraction(0)] * (nslack + nart) + [Fraction(b)]
        if s == '<=':
            row[si] = Fraction(1)
            basis.append(si)
            si += 1
        elif s == '>=':
            row[si] = Fraction(-1)
            si += 1
            row[ai] = Fraction(1)
            basis.append(ai)
            ai += 1
        else:
            row[ai] = Fraction(1)
            basis.append(ai)
            ai += 1
        T.append(row)
    first_art = nvar + nslack

    if nart:
        c1 = [Fraction(0)] * first_art + [Fraction(-1)] * nart
        obj = c1 + [Fraction(0)]
        for i, bcol in enumerate(basis):
            if c1[bcol] != 0:
                f = c1[bcol]
                obj = [x - f * y for x, y in zip(obj, T[i])]
        T.append(obj)
        _run_simplex(T, basis, list(range(total)), bud)
        if T[m][-1] != 0:                                   # value = -obj[rhs] < 0: some artificial stays positive
            return 'infeasible', None
        T.pop()
        i = 0
        while i < len(T):                                   # drive artificials out of the basis
            if basis[i] >= first_art:
                j = next((j for j in range(first_art) if T[i][j] != 0), None)
                if j is None:
                    del T[i], basis[i]                      # a redundant row
                    continue
                _pivot(T, i, j)
                basis[i] = j
            i += 1
    cc = [Fraction(x) for x in cvec] + [Fraction(0)] * (total - nvar)
    obj = cc[:first_art] + [Fraction(0)] * nart + [Fraction(0)]
    for i, bcol in enumerate(basis):
        f = obj[bcol]
        if f != 0:
            obj = [x - f * y for x, y in zip(obj, T[i])]
    T.append(obj)
    status = _run_simplex(T, basis, list(range(first_art)), bud)
    if status == 'unbounded':
        return 'unbounded', None
    y = [Fraction(0)] * nvar
    for i, bcol in enumerate(basis):
        if bcol < nvar:
            y[bcol] = T[i][-1]
    return 'optimal', y


def _lp(c, A_ub, b_ub, A_eq, b_eq, bounds, bud):
    """maximise c.x. Returns ('optimal', value, x) | ('infeasible', None, None) | ('unbounded', None, None)."""
    n = len(c)
    off = [Fraction(0)] * n
    cols = []                                              # per original variable: list of (column, coefficient)
    ncol = 0
    extra = []                                             # upper-bound rows on shifted variables: (column, bound)
    for i, (lo, hi) in enumerate(bounds):
        if lo is None and hi is None:
            cols.append([(ncol, 1), (ncol + 1, -1)])
            ncol += 2
        elif lo is not None:
            if hi is not None and hi < lo:
                return 'infeasible', None, None
            off[i] = lo
            cols.append([(ncol, 1)])
            if hi is not None:
                extra.append((ncol, hi - lo))
            ncol += 1
        else:
            off[i] = hi
            cols.append([(ncol, -1)])
            ncol += 1

    def row_of(a):
        r = [Fraction(0)] * ncol
        const = Fraction(0)
        for i, ai in enumerate(a):
            if ai != 0:
                const += ai * off[i]
                for col, cf in cols[i]:
                    r[col] += ai * cf
        return r, const
    rows = []
    for a, b in zip(A_ub, b_ub):
        r, k = row_of(a)
        rows.append((r, '<=', b - k))
    for a, b in zip(A_eq, b_eq):
        r, k = row_of(a)
        rows.append((r, '=', b - k))
    for col, ub in extra:
        r = [Fraction(0)] * ncol
        r[col] = Fraction(1)
        rows.append((r, '<=', ub))
    cvec, c0 = row_of(c)
    status, y = _simplex_core(cvec, rows, ncol, bud)
    if status != 'optimal':
        return status, None, None
    x = [off[i] + sum(cf * y[col] for col, cf in cols[i]) for i in range(n)]
    return 'optimal', sum(ci * xi for ci, xi in zip(c, x)), x


def _lp_input(c, A_ub, b_ub, A_eq, b_eq, bounds, maximize):
    cq = _qlist(c, 'c')
    n = len(cq)
    if n == 0:
        raise BadInput('no variables')
    if n > 60:
        raise NotCovered('more than 60 variables')

    def mat(A, name):
        return [] if A is None else [_qlist(r, name) for r in A]
    Au, Ae = mat(A_ub, 'A_ub'), mat(A_eq, 'A_eq')
    bu = [] if b_ub is None else _qlist(b_ub, 'b_ub')
    be = [] if b_eq is None else _qlist(b_eq, 'b_eq')
    if len(Au) != len(bu) or len(Ae) != len(be) or any(len(r) != n for r in Au + Ae):
        raise BadInput('constraint matrix and right-hand side do not match the number of variables')
    if bounds is None:
        bnd = [(Fraction(0), None)] * n
    else:
        if len(bounds) != n:
            raise BadInput('bounds: one (lo, hi) pair per variable')
        bnd = [(None if lo is None else _q(lo, 'bounds'), None if hi is None else _q(hi, 'bounds')) for lo, hi in bounds]
    if (len(Au) + len(Ae) + sum(1 for lo, hi in bnd if hi is not None)) * (n + len(Au) + len(Ae)) > 6000:
        raise NotCovered('too large: rows * columns > 6000')
    if not maximize:
        cq = [-x for x in cq]
    return cq, Au, bu, Ae, be, bnd


@op('search.lp_solve', kind='linear programming', bound='rows * columns <= 6000, 20000 pivots')
def lp_solve(c, A_ub=None, b_ub=None, A_eq=None, b_eq=None, bounds=None, maximize=True):
    """Exact linear program over Fractions: maximise (or minimise) c.x subject to A_ub x <= b_ub, A_eq x = b_eq and bounds[i] = (lo, hi) (None = no bound;
    default (0, None)). Two-phase simplex with Bland's rule. Returns {'status': 'optimal'|'infeasible'|'unbounded', 'value': Fraction, 'x': [Fraction ...]}.
    lp_solve([3, 2], [[1, 1], [1, 3]], [4, 6]) -> {'status': 'optimal', 'value': Fraction(12, 1), 'x': [Fraction(4, 1), Fraction(0, 1)]}"""
    cq, Au, bu, Ae, be, bnd = _lp_input(c, A_ub, b_ub, A_eq, b_eq, bounds, maximize)
    status, val, x = _lp(cq, Au, bu, Ae, be, bnd, _Budget(20000, 'search limit (20000 pivots)'))
    if status != 'optimal':
        return {'status': status, 'value': None, 'x': None}
    return {'status': 'optimal', 'value': val if maximize else -val, 'x': x}


@op('search.ilp_solve', kind='linear programming', bound='rows * columns <= 6000, 4000 branch-and-bound nodes')
def ilp_solve(c, A_ub=None, b_ub=None, A_eq=None, b_eq=None, bounds=None, maximize=True, integer=None):
    """Exact integer program by branch and bound on the exact LP relaxation. `integer` lists the indices of the integer variables (default: all).
    Same input and output as lp_solve. An unbounded relaxation raises NotCovered.
    ilp_solve([5, 4], [[6, 4], [1, 2]], [24, 6]) -> {'status': 'optimal', 'value': Fraction(20, 1), 'x': [Fraction(4, 1), Fraction(0, 1)]}"""
    cq, Au, bu, Ae, be, bnd = _lp_input(c, A_ub, b_ub, A_eq, b_eq, bounds, maximize)
    n = len(cq)
    ints = list(range(n)) if integer is None else sorted({_int(i, 'integer') for i in integer})
    if any(i < 0 or i >= n for i in ints):
        raise BadInput('integer: index out of range')
    bud = _Budget(4000, 'search limit (4000 branch-and-bound nodes)')
    pivots = _Budget(200000, 'search limit (200000 pivots)')
    best = {'v': None, 'x': None}

    def rec(bounds_now):
        bud.tick()
        st, val, x = _lp(cq, Au, bu, Ae, be, bounds_now, pivots)
        if st == 'infeasible':
            return
        if st == 'unbounded':
            raise NotCovered('LP relaxation is unbounded')
        if best['v'] is not None and val <= best['v']:
            return
        frac = next((i for i in ints if x[i].denominator != 1), None)
        if frac is None:
            best['v'], best['x'] = val, x
            return
        lo, hi = bounds_now[frac]
        fl = x[frac].numerator // x[frac].denominator
        up = list(bounds_now)
        up[frac] = (Fraction(fl + 1), hi)
        rec(up)
        down = list(bounds_now)
        down[frac] = (lo, Fraction(fl))
        rec(down)
    rec(bnd)
    if best['v'] is None:
        return {'status': 'infeasible', 'value': None, 'x': None}
    return {'status': 'optimal', 'value': best['v'] if maximize else -best['v'], 'x': best['x']}


# ----------------------------------------------------------------------------------------------------------------------------------------
# graphs: shortest paths, max flow, minimum spanning tree, assignment
# ----------------------------------------------------------------------------------------------------------------------------------------

def _edges(edges, directed, what='edges'):
    """[(u, v)] or [(u, v, w)] -> list of (u, v, w Fraction); nodes are ints or strs."""
    out = []
    for e in edges:
        if len(e) == 2:
            u, v, w = e[0], e[1], Fraction(1)
        elif len(e) == 3:
            u, v, w = e[0], e[1], _q(e[2], 'weight')
        else:
            raise BadInput(f'{what}: an edge is (u, v) or (u, v, weight)')
        for z in (u, v):
            if isinstance(z, bool) or not isinstance(z, (int, str)):
                raise BadInput(f'{what}: nodes are ints or strs')
        out.append((u, v, w))
    if len(out) > 20000:
        raise NotCovered('more than 20000 edges')
    return out


def _nodes_of(es, extra=()):
    s = set(extra)
    for u, v, _ in es:
        s.add(u)
        s.add(v)
    return sorted(s, key=_nkey)


@op('search.shortest_path', kind='graph', bound='<= 20000 edges; Dijkstra, or Bellman-Ford when a weight is negative')
def shortest_path(edges, source, target=None, directed=True):
    """Shortest distances from `source` (Dijkstra for non-negative weights, Bellman-Ford otherwise). edges: (u, v) or (u, v, weight); nodes ints or strs.
    Returns {'dist': {node: distance} for reachable nodes, 'path': node list to `target` or None, 'negative_cycle': True when a negative cycle is reachable
    (then dist and path are None)}. shortest_path([(1, 2, 7), (1, 3, 9), (2, 3, 1)], 1, 3) -> {'dist': {1: 0, 2: 7, 3: 8}, 'path': [1, 2, 3], 'negative_cycle': False}"""
    es = _edges(edges, directed)
    nodes = _nodes_of(es, [source] + ([] if target is None else [target]))
    adj = {u: [] for u in nodes}
    for u, v, w in es:
        adj[u].append((v, w))
        if not directed:
            adj[v].append((u, w))
    neg = any(w < 0 for _, _, w in es)
    if neg and not directed:
        raise NotCovered('negative weights in an undirected graph (every such edge is a negative cycle)')
    dist, prev = {source: Fraction(0)}, {}
    if not neg:
        heap, cnt, done = [(Fraction(0), 0, source)], 1, set()
        while heap:
            d, _, u = heapq.heappop(heap)
            if u in done:
                continue
            done.add(u)
            for v, w in adj[u]:
                nd = d + w
                if v not in dist or nd < dist[v]:
                    dist[v], prev[v] = nd, u
                    heapq.heappush(heap, (nd, cnt, v))
                    cnt += 1
    else:
        for _ in range(len(nodes) - 1):
            ch = False
            for u in nodes:
                if u in dist:
                    for v, w in adj[u]:
                        if v not in dist or dist[u] + w < dist[v]:
                            dist[v], prev[v] = dist[u] + w, u
                            ch = True
            if not ch:
                break
        for u in nodes:
            if u in dist:
                for v, w in adj[u]:
                    if dist[u] + w < dist[v]:
                        return {'dist': None, 'path': None, 'negative_cycle': True}
    path = None
    if target is not None and target in dist:
        path, z = [target], target
        while z != source:
            z = prev[z]
            path.append(z)
        path.reverse()
    return {'dist': {k: _clean(dist[k]) for k in sorted(dist, key=_nkey)}, 'path': path, 'negative_cycle': False}


@op('search.all_pairs_shortest', kind='graph', bound='<= 150 nodes (Floyd-Warshall)')
def all_pairs_shortest(edges, directed=True):
    """All-pairs shortest distances by Floyd-Warshall. Returns {'dist': {u: {v: distance}} for reachable pairs, 'negative_cycle': bool}.
    all_pairs_shortest([(1, 2, 3), (2, 3, 4)])['dist'][1][3] -> 7"""
    es = _edges(edges, directed)
    nodes = _nodes_of(es)
    if len(nodes) > 150:
        raise NotCovered('more than 150 nodes')
    d = {u: {u: Fraction(0)} for u in nodes}
    for u, v, w in es:
        if v not in d[u] or w < d[u][v]:
            d[u][v] = w
        if not directed and (u not in d[v] or w < d[v][u]):
            d[v][u] = w
    for k in nodes:
        dk = d[k]
        for i in nodes:
            di = d[i]
            if k in di:
                dik = di[k]
                for j, dkj in dk.items():
                    s = dik + dkj
                    if j not in di or s < di[j]:
                        di[j] = s
    if any(d[u][u] < 0 for u in nodes):
        return {'dist': None, 'negative_cycle': True}
    return {'dist': {u: {v: _clean(d[u][v]) for v in sorted(d[u], key=_nkey)} for u in nodes}, 'negative_cycle': False}


@op('search.max_flow', kind='graph', bound='<= 20000 edges (Edmonds-Karp)')
def max_flow(edges, source, sink, directed=True):
    """Maximum flow from source to sink (Edmonds-Karp over exact capacities). edges: (u, v, capacity). Returns {'value': flow, 'flow': [flow on each input edge,
    in the input order; for an undirected edge, positive when it runs u -> v], 'cut': {'source_side': nodes, 'edges': indices of the edges of a minimum cut}}.
    max_flow([(1, 2, 3), (1, 3, 2), (2, 4, 2), (3, 4, 3)], 1, 4) -> value 4"""
    es = _edges(edges, directed)
    if any(w < 0 for _, _, w in es):
        raise BadInput('capacities must be non-negative')
    if source == sink:
        raise BadInput('source equals sink')
    nodes = _nodes_of(es, [source, sink])
    idx = {u: i for i, u in enumerate(nodes)}
    to, cap, adj = [], [], [[] for _ in nodes]
    for u, v, w in es:
        adj[idx[u]].append(len(to))
        to.append(idx[v])
        cap.append(w)
        adj[idx[v]].append(len(to))
        to.append(idx[u])
        cap.append(w if not directed else Fraction(0))
    s, t = idx[source], idx[sink]
    value = Fraction(0)
    while True:
        par = {s: None}
        q = deque([s])
        while q and t not in par:
            u = q.popleft()
            for a in adj[u]:
                if cap[a] > 0 and to[a] not in par:
                    par[to[a]] = a
                    q.append(to[a])
        if t not in par:
            break
        path, z, push = [], t, None
        while z != s:
            a = par[z]
            path.append(a)
            push = cap[a] if push is None else min(push, cap[a])
            z = to[a ^ 1]
        for a in path:
            cap[a] -= push
            cap[a ^ 1] += push
        value += push
    side = set(par)
    flow = []
    for k, (u, v, w) in enumerate(es):
        flow.append(_clean(w - cap[2 * k]))
    cutedges = [k for k, (u, v, w) in enumerate(es) if (idx[u] in side) != (idx[v] in side) and (not directed or idx[u] in side)]
    return {'value': _clean(value), 'flow': flow, 'cut': {'source_side': [nodes[i] for i in sorted(side)], 'edges': cutedges}}


@op('search.mst', kind='graph', bound='<= 20000 edges (Kruskal)')
def mst(edges, nodes=None, maximize=False):
    """Minimum (or maximum) spanning tree of an undirected graph by Kruskal. Ties break by input order. Returns {'weight': total, 'edges': indices of the chosen
    input edges, ascending, 'connected': bool}; for a disconnected graph it is a spanning forest.
    mst([(1, 2, 1), (2, 3, 2), (1, 3, 3)]) -> {'weight': 3, 'edges': [0, 1], 'connected': True}"""
    es = _edges(edges, False)
    ns = _nodes_of(es, [] if nodes is None else nodes)
    parent = {u: u for u in ns}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    order = sorted(range(len(es)), key=lambda k: (-es[k][2] if maximize else es[k][2], k))
    chosen, total = [], Fraction(0)
    for k in order:
        u, v, w = es[k]
        ru, rv = find(u), find(v)
        if ru != rv:
            parent[ru] = rv
            chosen.append(k)
            total += w
    return {'weight': _clean(total), 'edges': sorted(chosen), 'connected': len(chosen) == max(0, len(ns) - 1)}


@op('search.assignment', kind='graph', bound='<= 150 rows and columns (Hungarian method)')
def assignment(cost, maximize=False):
    """Assignment problem: choose one entry per row, no two in a column (rectangular allowed: the shorter side is fully matched), minimising (or maximising)
    the total. Exact Hungarian method. Returns {'cost': total, 'pairs': [(row, column)] sorted by row}.
    assignment([[4, 1, 3], [2, 0, 5], [3, 2, 2]]) -> {'cost': 5, 'pairs': [(0, 1), (1, 0), (2, 2)]}"""
    if not hasattr(cost, '__iter__') or not cost:
        raise BadInput('cost: a non-empty matrix')
    a = [_qlist(r, 'cost') for r in cost]
    if any(len(r) != len(a[0]) for r in a) or not a[0]:
        raise BadInput('cost: rows differ in length')
    if len(a) > 150 or len(a[0]) > 150:
        raise NotCovered('more than 150 rows or columns')
    transposed = len(a) > len(a[0])
    if transposed:
        a = [list(r) for r in zip(*a)]
    if maximize:
        a = [[-x for x in r] for r in a]
    n, m = len(a), len(a[0])
    INF = None
    u, v, p, way = [Fraction(0)] * (n + 1), [Fraction(0)] * (m + 1), [0] * (m + 1), [0] * (m + 1)
    for i in range(1, n + 1):
        p[0], j0 = i, 0
        minv, used = [INF] * (m + 1), [False] * (m + 1)
        while True:
            used[j0] = True
            i0, delta, j1 = p[j0], INF, 0
            for j in range(1, m + 1):
                if not used[j]:
                    cur = a[i0 - 1][j - 1] - u[i0] - v[j]
                    if minv[j] is INF or cur < minv[j]:
                        minv[j], way[j] = cur, j0
                    if delta is INF or minv[j] < delta:
                        delta, j1 = minv[j], j
            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                elif minv[j] is not INF:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while j0:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
    pairs = [(p[j] - 1, j - 1) for j in range(1, m + 1) if p[j]]
    orig = [[(-x if maximize else x) for x in r] for r in a]
    total = sum(orig[i][j] for i, j in pairs)
    if transposed:
        pairs = [(j, i) for i, j in pairs]
    return {'cost': _clean(total), 'pairs': sorted(pairs)}


# ----------------------------------------------------------------------------------------------------------------------------------------
# exact statistics
# ----------------------------------------------------------------------------------------------------------------------------------------

def _data(xs, what='data', minimum=1):
    q = _qlist(xs, what)
    if len(q) < minimum:
        raise BadInput(f'{what}: need at least {minimum} value(s)')
    if len(q) > 1_000_000:
        raise NotCovered('more than 1e6 values')
    return q


@op('search.mean', kind='statistics', bound='<= 1e6 values')
def mean(data):
    """Arithmetic mean as an exact Fraction. mean([1, 2, 4]) -> Fraction(7, 3)"""
    q = _data(data)
    return sum(q) / len(q)


@op('search.median', kind='statistics', bound='<= 1e6 values')
def median(data):
    """Median as an exact Fraction (the mean of the two middle values when the count is even). median([1, 3, 4, 10]) -> Fraction(7, 2)"""
    q = sorted(_data(data))
    n = len(q)
    return q[n // 2] if n % 2 else (q[n // 2 - 1] + q[n // 2]) / 2


@op('search.mode', kind='statistics', bound='<= 1e6 values')
def mode(data):
    """All most-frequent values, sorted ascending (a list). mode([1, 2, 2, 3, 3]) -> [Fraction(2, 1), Fraction(3, 1)]"""
    q = _data(data)
    cnt = {}
    for v in q:
        cnt[v] = cnt.get(v, 0) + 1
    top = max(cnt.values())
    return sorted(v for v, c in cnt.items() if c == top)


@op('search.variance', kind='statistics', bound='<= 1e6 values')
def variance(data, sample=False):
    """Variance as an exact Fraction: the population variance (divide by n) by default, the sample variance (divide by n - 1) when sample=True.
    variance([2, 4, 4, 4, 5, 5, 7, 9]) -> Fraction(4, 1)"""
    q = _data(data, minimum=2 if sample else 1)
    m = sum(q) / len(q)
    return sum((v - m) ** 2 for v in q) / (len(q) - 1 if sample else len(q))


@op('search.stdev', kind='statistics', bound='<= 1e6 values')
def stdev(data, sample=False):
    """Standard deviation as an exact Surd (the square root of the variance). stdev([2, 4, 4, 4, 5, 5, 7, 9]) -> Surd(2)"""
    return Surd.sqrt(variance(data, sample))


@op('search.covariance', kind='statistics', bound='<= 1e6 values')
def covariance(xs, ys, sample=False):
    """Covariance of two equally long lists as an exact Fraction (population by default, sample when sample=True).
    covariance([1, 2, 3], [1, 2, 4]) -> Fraction(1, 1)"""
    x, y = _data(xs, 'xs', 2 if sample else 1), _data(ys, 'ys', 2 if sample else 1)
    if len(x) != len(y):
        raise BadInput('xs and ys differ in length')
    mx, my = sum(x) / len(x), sum(y) / len(y)
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / (len(x) - 1 if sample else len(x))


@op('search.correlation', kind='statistics', bound='<= 1e6 values')
def correlation(xs, ys):
    """Pearson correlation coefficient as an exact Surd (sign times the square root of the rational r squared). Constant data is BadInput.
    correlation([1, 2, 3], [2, 4, 6]) -> Surd(1)"""
    x, y = _data(xs, 'xs'), _data(ys, 'ys')
    if len(x) != len(y):
        raise BadInput('xs and ys differ in length')
    mx, my = sum(x) / len(x), sum(y) / len(y)
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    sxx = sum((a - mx) ** 2 for a in x)
    syy = sum((b - my) ** 2 for b in y)
    if sxx == 0 or syy == 0:
        raise BadInput('a constant list has no correlation')
    r = Surd.sqrt(sxy * sxy / (sxx * syy))
    return -r if sxy < 0 else r


@op('search.linear_regression', kind='statistics', bound='<= 1e6 values')
def linear_regression(xs, ys):
    """Least-squares line y = slope*x + intercept, all exact Fractions. Returns {'slope', 'intercept', 'r_squared', 'n'}; BadInput when all x are equal.
    linear_regression([1, 2, 3], [2, 4, 7]) -> slope 5/2, intercept -2/3, r_squared 75/76 ... exact"""
    x, y = _data(xs, 'xs', 2), _data(ys, 'ys', 2)
    if len(x) != len(y):
        raise BadInput('xs and ys differ in length')
    n = len(x)
    mx, my = sum(x) / n, sum(y) / n
    sxx = sum((a - mx) ** 2 for a in x)
    if sxx == 0:
        raise BadInput('all x are equal: the slope is undefined')
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    syy = sum((b - my) ** 2 for b in y)
    slope = sxy / sxx
    return {'slope': slope, 'intercept': my - slope * mx, 'r_squared': (sxy * sxy / (sxx * syy)) if syy != 0 else Fraction(1), 'n': n}


def _prob(p, what='p'):
    p = _q(p, what)
    if p < 0 or p > 1:
        raise BadInput(f'{what} must be in [0, 1]')
    return p


@op('search.binomial_pmf', kind='probability', bound='n <= 3000')
def binomial_pmf(n, p, k):
    """P(X = k) for X ~ Binomial(n, p), exact Fraction. binomial_pmf(4, '1/2', 2) -> Fraction(3, 8)"""
    n, k, p = _int(n, 'n'), _int(k, 'k'), _prob(p)
    if n < 0:
        raise BadInput('n must be non-negative')
    if n > 3000:
        raise NotCovered('n > 3000')
    return comb(n, k) * p ** k * (1 - p) ** (n - k) if 0 <= k <= n else Fraction(0)


@op('search.binomial_cdf', kind='probability', bound='n <= 3000')
def binomial_cdf(n, p, k):
    """P(X <= k) for X ~ Binomial(n, p), exact Fraction. binomial_cdf(4, '1/2', 2) -> Fraction(11, 16)"""
    n, k, p = _int(n, 'n'), _int(k, 'k'), _prob(p)
    if n < 0:
        raise BadInput('n must be non-negative')
    if n > 3000:
        raise NotCovered('n > 3000')
    if k < 0:
        return Fraction(0)
    if k >= n:
        return Fraction(1)
    return sum(comb(n, j) * p ** j * (1 - p) ** (n - j) for j in range(k + 1))


@op('search.hypergeom_pmf', kind='probability', bound='N <= 100000')
def hypergeom_pmf(N, K, n, k):
    """P(X = k) when n items are drawn without replacement from N items of which K are marked, exact Fraction. hypergeom_pmf(10, 4, 3, 2) -> Fraction(3, 10)"""
    N, K, n, k = (_int(v, nm) for v, nm in ((N, 'N'), (K, 'K'), (n, 'n'), (k, 'k')))
    if not (0 <= K <= N and 0 <= n <= N):
        raise BadInput('need 0 <= K <= N and 0 <= n <= N')
    if N > 100000:
        raise NotCovered('N > 100000')
    if k < max(0, n - (N - K)) or k > min(n, K):
        return Fraction(0)
    return Fraction(comb(K, k) * comb(N - K, n - k), comb(N, n))


@op('search.hypergeom_cdf', kind='probability', bound='N <= 100000')
def hypergeom_cdf(N, K, n, k):
    """P(X <= k) for the hypergeometric distribution of hypergeom_pmf, exact Fraction. hypergeom_cdf(10, 4, 3, 1) -> Fraction(2, 3)"""
    k = _int(k, 'k')
    lo = max(0, _int(n, 'n') - (_int(N, 'N') - _int(K, 'K')))
    return sum((hypergeom_pmf(N, K, n, j) for j in range(lo, k + 1)), Fraction(0))


@op('search.geometric_pmf', kind='probability', bound='k <= 100000')
def geometric_pmf(p, k):
    """P(first success on trial k) = (1-p)^(k-1) p for k >= 1, exact Fraction. geometric_pmf('1/6', 3) -> Fraction(25, 216)"""
    p, k = _prob(p), _int(k, 'k')
    if k > 100000:
        raise NotCovered('k > 100000')
    return (1 - p) ** (k - 1) * p if k >= 1 else Fraction(0)


@op('search.geometric_cdf', kind='probability', bound='k <= 100000')
def geometric_cdf(p, k):
    """P(first success within k trials) = 1 - (1-p)^k, exact Fraction. geometric_cdf('1/6', 3) -> Fraction(91, 216)"""
    p, k = _prob(p), _int(k, 'k')
    if k > 100000:
        raise NotCovered('k > 100000')
    return 1 - (1 - p) ** k if k >= 1 else Fraction(0)


@op('search.negbinomial_pmf', kind='probability', bound='k <= 3000')
def negbinomial_pmf(r, p, k):
    """P(the r-th success comes on trial k) = C(k-1, r-1) p^r (1-p)^(k-r), exact Fraction. negbinomial_pmf(2, '1/2', 4) -> Fraction(3, 16)"""
    r, p, k = _int(r, 'r'), _prob(p), _int(k, 'k')
    if r < 1:
        raise BadInput('r must be >= 1')
    if k > 3000:
        raise NotCovered('k > 3000')
    return comb(k - 1, r - 1) * p ** r * (1 - p) ** (k - r) if k >= r else Fraction(0)


@op('search.poisson_partial_sum', kind='probability', bound='k <= 100000')
def poisson_partial_sum(lam, k):
    """The exact finite sum 1 + lam + lam^2/2! + ... + lam^k/k! as a Fraction (the truncated exponential series: P(X <= k) = e^-lam times this).
    poisson_partial_sum(2, 3) -> Fraction(19, 3)"""
    lam, k = _q(lam, 'lam'), _int(k, 'k')
    if lam < 0:
        raise BadInput('lam must be non-negative')
    if k > 100000:
        raise NotCovered('k > 100000')
    s, t = Fraction(0), Fraction(1)
    for j in range(0, k + 1):
        s += t
        t = t * lam / (j + 1)
    return s


# ----------------------------------------------------------------------------------------------------------------------------------------
# decimal arithmetic: pi, sin, cos, atan, erf ... (the standard library decimal has only exp, ln, log10, sqrt)
# ----------------------------------------------------------------------------------------------------------------------------------------

def _ctx(prec):
    """A working context: huge exponent range, and Overflow does not trap (exp(1e30) is Infinity, so x/(exp(x)-1) is 0 there as it should be);
    a function value that ends up infinite is refused in _safe."""
    c = Context(prec=prec, rounding=ROUND_HALF_EVEN, Emax=10 ** 9, Emin=-10 ** 9)
    c.traps[Overflow] = False
    return localcontext(c)


_PI_CACHE = {}


def _pi(prec):
    """pi to at least `prec` digits (Machin's formula in integer arithmetic)."""
    for p, v in _PI_CACHE.items():
        if p >= prec:
            return v
    guard = ((prec + 20) // 50 + 1) * 50
    one = 10 ** guard

    def atan_inv(n):
        x = one // n
        s, k, sign, n2 = x, 1, 1, n * n
        while x:
            x //= n2
            k += 2
            sign = -sign
            s += sign * (x // k)
        return s
    pi = 4 * (4 * atan_inv(5) - atan_inv(239))
    v = Decimal(f'{pi}E-{guard}')
    _PI_CACHE[guard] = v
    return v


def _sincos(x):
    """(sin x, cos x) at the current context precision."""
    p = getcontext().prec
    with _ctx(p + max(0, x.adjusted()) + 12) as c:
        half = _pi(c.prec) / 2
        k = (x / half).to_integral_value(ROUND_HALF_EVEN)
        r = x - k * half
        r2 = r * r
        eps = Decimal(10) ** -(c.prec + 1)
        term, s, n = r, r, 1
        while abs(term) > eps:
            term = -term * r2 / ((n + 1) * (n + 2))
            s += term
            n += 2
        term, co, n = Decimal(1), Decimal(1), 0
        while abs(term) > eps:
            term = -term * r2 / ((n + 1) * (n + 2))
            co += term
            n += 2
        q = int(k) % 4
        s, co = [(s, co), (co, -s), (-s, -co), (-co, s)][q]
    return +s, +co


def _sin(x):
    return _sincos(x)[0]


def _cos(x):
    return _sincos(x)[1]


def _tan(x):
    s, c = _sincos(x)
    return s / c


def _atan(x):
    p = getcontext().prec
    if x == 0:
        return Decimal(0)
    with _ctx(p + 12) as c:
        neg = x < 0
        z = abs(x)
        inv = z > 1
        if inv:
            z = 1 / z
        k = 0
        while z > Decimal('0.03'):
            z = z / (1 + (1 + z * z).sqrt())
            k += 1
        z2 = z * z
        eps = Decimal(10) ** -(c.prec + 1)
        term, s, n = z, z, 1
        while abs(term) > eps:
            term = -term * z2
            n += 2
            s += term / n
        s *= 2 ** k
        if inv:
            s = _pi(c.prec) / 2 - s
        if neg:
            s = -s
    return +s


def _asin(x):
    if abs(x) > 1:
        raise _Undefined('asin out of range')
    p = getcontext().prec
    with _ctx(p + 5):
        if abs(x) == 1:
            r = _pi(p + 5) / 2 * (1 if x > 0 else -1)
        else:
            r = _atan(x / (1 - x * x).sqrt())
    return +r


def _acos(x):
    p = getcontext().prec
    with _ctx(p + 5):
        r = _pi(p + 5) / 2 - _asin(x)
    return +r


def _small_guard(x):
    return max(0, -x.adjusted()) + 6 if x != 0 else 0


def _sinh(x):
    with _ctx(getcontext().prec + _small_guard(x) + 4):
        e = x.exp()
        r = (e - 1 / e) / 2
    return +r


def _cosh(x):
    with _ctx(getcontext().prec + 4):
        e = x.exp()
        r = (e + 1 / e) / 2
    return +r


def _tanh(x):
    with _ctx(getcontext().prec + _small_guard(x) + 4):
        e = (2 * x).exp()
        r = (e - 1) / (e + 1)
    return +r


def _asinh(x):
    with _ctx(getcontext().prec + _small_guard(x) + 6):
        z = abs(x)
        r = (z + (z * z + 1).sqrt()).ln()
        r = -r if x < 0 else r
    return +r


def _acosh(x):
    if x < 1:
        raise _Undefined('acosh below 1')
    with _ctx(getcontext().prec + 6):
        r = (x + (x * x - 1).sqrt()).ln()
    return +r


def _atanh(x):
    if abs(x) >= 1:
        raise _Undefined('atanh out of range')
    with _ctx(getcontext().prec + _small_guard(x) + 6):
        r = ((1 + x) / (1 - x)).ln() / 2
    return +r


def _cbrt(x):
    if x == 0:
        return Decimal(0)
    with _ctx(getcontext().prec + 5):
        r = (abs(x).ln() / 3).exp()
        r = -r if x < 0 else r
    return +r


def _erf_series(z):
    """erf(z) for z >= 0 by the all-positive series 2/sqrt(pi) e^(-z^2) sum 2^n z^(2n+1)/(2n+1)!!."""
    if z == 0:
        return Decimal(0)
    p = getcontext().prec
    with _ctx(p + 10) as c:
        z2 = z * z
        eps = Decimal(10) ** -(c.prec + 2)
        term, s, n = z, z, 0
        while True:
            n += 1
            term = term * 2 * z2 / (2 * n + 1)
            s += term
            if term < eps * s:
                break
        r = 2 * s * (-z2).exp() / _pi(c.prec).sqrt()
    return +r


def _erf(x):
    if abs(x) > 38:
        raise _Undefined('erf argument beyond 38')
    r = _erf_series(abs(x))
    return -r if x < 0 else r


def _dpow(b, e):
    if e == 0:
        if b == 0:
            raise _Undefined('0^0')
        return Decimal(1)
    return b ** e


def _dlog(x):
    if x <= 0:
        raise _Undefined('log of a non-positive number')
    return x.ln()


_DEC1 = {'sin': _sin, 'cos': _cos, 'tan': _tan, 'asin': _asin, 'acos': _acos, 'atan': _atan, 'sinh': _sinh, 'cosh': _cosh, 'tanh': _tanh,
         'asinh': _asinh, 'acosh': _acosh, 'atanh': _atanh, 'exp': lambda x: x.exp(), 'log': _dlog, 'ln': _dlog,
         'log10': lambda x: x.log10(), 'log2': lambda x: _dlog(x) / _dlog(Decimal(2)), 'sqrt': lambda x: x.sqrt(), 'cbrt': _cbrt,
         'abs': lambda x: abs(x), 'erf': _erf}
def _factorial(x):
    if x != x.to_integral_value() or x < 0 or x > 3000:
        raise _Undefined('factorial needs an integer in 0..3000')
    return Decimal(factorial(int(x)))


def _comb(n, k):
    if n != n.to_integral_value() or k != k.to_integral_value() or n < 0 or n > 3000:
        raise _Undefined('comb needs integers, 0 <= n <= 3000')
    return Decimal(comb(int(n), int(k))) if 0 <= k <= n else Decimal(0)


_DEC1['factorial'] = _factorial
_DEC2 = {'min': min, 'max': max, 'comb': _comb, 'binomial': _comb}
_NUM = re.compile(r'(?<![\w.])(\d+\.?\d*(?:[eE][+-]?\d+)?|\.\d+(?:[eE][+-]?\d+)?)')


@lru_cache(maxsize=512)
def _dec_compile(expr, argnames):
    """Text -> a function of the named arguments (Decimals) that returns a Decimal in the current context. Raises BadInput on a text it cannot read."""
    if not isinstance(expr, str):
        raise BadInput('a function is a string such as "sin(x)*exp(-x^2)"')
    text = re.sub(r'(\w+|\([^()]*\))!', r'factorial(\1)', expr.strip())          # n! and (n+1)!
    text = _NUM.sub(lambda m: f'N("{m.group(1)}")', text.replace('^', '**'))
    try:
        tree = ast.parse(text, mode='eval')
    except SyntaxError as e:
        raise BadInput(f'cannot read {expr!r}: {e.msg} (write * for products)')
    ix = {nm: i for i, nm in enumerate(argnames)}

    def build(nd):
        if isinstance(nd, ast.Expression):
            return build(nd.body)
        if isinstance(nd, ast.BinOp):
            l, r = build(nd.left), build(nd.right)
            t = type(nd.op)
            if t is ast.Add:
                return lambda a: l(a) + r(a)
            if t is ast.Sub:
                return lambda a: l(a) - r(a)
            if t is ast.Mult:
                return lambda a: l(a) * r(a)
            if t is ast.Div:
                return lambda a: l(a) / r(a)
            if t is ast.Pow:
                return lambda a: _dpow(l(a), r(a))
            raise BadInput(f'{expr!r}: operator {t.__name__} is not allowed')
        if isinstance(nd, ast.UnaryOp):
            f = build(nd.operand)
            if isinstance(nd.op, ast.USub):
                return lambda a: -f(a)
            if isinstance(nd.op, ast.UAdd):
                return f
            raise BadInput(f'{expr!r}: unary operator not allowed')
        if isinstance(nd, ast.Name):
            if nd.id in ix:
                i = ix[nd.id]
                return lambda a: a[i]
            if nd.id == 'pi':
                return lambda a: +_pi(getcontext().prec + 5)
            if nd.id == 'e':
                return lambda a: Decimal(1).exp()
            raise BadInput(f'{expr!r}: unknown name {nd.id!r}')
        if isinstance(nd, ast.Call):
            if not isinstance(nd.func, ast.Name) or nd.keywords:
                raise BadInput(f'{expr!r}: bad call')
            nm = nd.func.id
            if nm == 'N':
                v = Decimal(nd.args[0].value)
                return lambda a: v
            args = [build(x) for x in nd.args]
            if nm in _DEC1 and len(args) == 1:
                f, g = _DEC1[nm], args[0]
                return lambda a: f(g(a))
            if nm in _DEC2 and len(args) == 2:
                f, g, h = _DEC2[nm], args[0], args[1]
                return lambda a: f(g(a), h(a))
            raise BadInput(f'{expr!r}: unknown function {nm!r} or wrong number of arguments')
        raise BadInput(f'{expr!r}: {type(nd).__name__} is not allowed')
    return build(tree)


def _dec(x, what='value'):
    """int, Fraction, Decimal or str -> Decimal in the current context (a Fraction is divided at the context precision). 'inf' / '-inf' give infinities.
    A str that is not a rational is read as a constant expression: 'pi', 'pi/2', 'sqrt(2)'."""
    if isinstance(x, str):
        t = x.strip()
        if t.lower().lstrip('+-') in ('inf', 'infinity'):
            return Decimal('-Infinity' if t.startswith('-') else 'Infinity')
        try:
            q = Fraction(t)
        except (ValueError, ZeroDivisionError):
            return _safe(_dec_compile(t, ()), ())
        return Decimal(q.numerator) / Decimal(q.denominator)
    if isinstance(x, Decimal):
        return x
    q = _q(x, what)
    return Decimal(q.numerator) / Decimal(q.denominator)


def _fun(f, argnames=('x',)):
    """Text or a Python callable -> a function of Decimals."""
    if callable(f):
        return f
    c = _dec_compile(f, tuple(argnames))
    return lambda *a: c(a)


def _safe(fn, *a):
    try:
        r = fn(*a)
    except ArithmeticError as e:
        raise NotCovered(f'the function is undefined where it was sampled ({type(e).__name__})')
    if isinstance(r, Decimal) and not r.is_finite():
        raise NotCovered('the function is not finite where it was sampled')
    return r


def _tol(digits, extra=2):
    return Decimal(10) ** -(digits + extra)


def _sig3(e):
    """An error estimate to 3 significant digits."""
    e = abs(e)
    if e == 0 or not e.is_finite():
        return e
    with _ctx(400):
        return e.quantize(Decimal(1).scaleb(e.adjusted() - 2))


def _round(v, digits, rel=False):
    from decimal import InvalidOperation
    zero = Decimal((0, (0,) * (digits + 1), -digits))          # prints as 0.000...
    if v == 0:
        return zero

    def exp_of(x):
        return x.adjusted() + 1 - digits if rel else max(x.adjusted() + 1, 0) - digits
    with _ctx(400):
        try:
            r = v.quantize(Decimal(1).scaleb(exp_of(v)))
            if exp_of(r) != exp_of(v) and r != 0:               # rounding carried into a new digit (0.99999 -> 1.0000): use the digits of the rounded value
                r = r.quantize(Decimal(1).scaleb(exp_of(r)))
        except InvalidOperation:
            with _ctx(digits):
                r = +v
        return zero if r == 0 else r


def _finish(v, err, digits, rel=False):
    """Round to the digits asked and wrap with the error estimate (3 digits)."""
    return Numeric(_round(v, digits, rel), _sig3(err))


def _scale(v):
    return max(Decimal(1), abs(v))


def _neville0(xs, ys):
    """Polynomial extrapolation to x = 0 (Neville). Returns the table T[i][k]: the value using points i-k..i."""
    n = len(xs)
    T = [[None] * n for _ in range(n)]
    for i in range(n):
        T[i][0] = ys[i]
        for k in range(1, i + 1):
            T[i][k] = (xs[i] * T[i - 1][k - 1] - xs[i - k] * T[i][k - 1]) / (xs[i] - xs[i - k])
    return T


def _best_entry(T):
    """The table entry whose two neighbours agree best: (value, error estimate)."""
    best = None
    n = len(T)
    for i in range(1, n):
        for k in range(1, i + 1):
            e = max(abs(T[i][k] - T[i][k - 1]), abs(T[i][k] - T[i - 1][k - 1]))
            if best is None or e < best[1]:
                best = (T[i][k], e)
    return best


# ----------------------------------------------------------------------------------------------------------------------------------------
# numeric calculus
# ----------------------------------------------------------------------------------------------------------------------------------------

@op('search.num_eval', kind='numeric calculus', bound='digits <= 100')
def num_eval(expr, vars=None, digits=30):
    """Evaluate a text expression at the digits asked. vars: {'x': value} (int, Fraction or str). The error is the difference between two working precisions.
    num_eval('sin(pi/6)', digits=10) -> Numeric(value=Decimal('0.5000000000'), error=...)"""
    _check_digits(digits)
    vs = {k: _q(v, k) for k, v in (vars or {}).items()}
    names = tuple(sorted(vs))
    c = _dec_compile(expr, names)
    res = []
    for extra in (20, 32):
        with _ctx(digits + extra):
            args = tuple(Decimal(vs[k].numerator) / Decimal(vs[k].denominator) for k in names)
            res.append(_safe(c, args))
    return _finish(res[1], abs(res[1] - res[0]) + Decimal(10) ** -(digits + 12) * _scale(res[1]), digits)


@op('search.derivative', kind='numeric calculus', bound='order <= 6, digits <= 60')
def derivative(f, x0, digits=15, order=1):
    """The order-th derivative of f at x0 by central differences with Richardson extrapolation (Neville in h^2) at a working precision well above the digits
    asked. For order 1 it also checks the two one-sided derivatives and raises NotCovered when they differ (a kink), so |x| at 0 is refused rather than answered 0.
    Returns Numeric(value, error). derivative('x^3', 2, digits=10) -> Numeric(value=Decimal('12.00000000'), ...)"""
    _check_digits(digits)
    order = _int(order, 'order')
    if not 1 <= order <= 6 or digits > 60:
        raise NotCovered('order must be 1..6 and digits <= 60')
    W = 2 * digits + 50 + 6 * order
    with _ctx(W):
        fn = _fun(f)
        x = _dec(x0, 'x0')
        last = None
        for shrink in (1, 16, 256, 4096):                       # start steps 1/64, 1/1024, ...: a nearer singularity or a faster wiggle needs a smaller one
            h0 = Decimal(1) / (64 * shrink)
            hs = [h0 / 2 ** i for i in range(9)]
            try:
                ys = []
                for h in hs:
                    tot = Decimal(0)
                    for k in range(order + 1):
                        tot += (-1) ** k * comb(order, k) * _safe(fn, x + (Decimal(order) / 2 - k) * h)
                    ys.append(tot / h ** order)
            except NotCovered as e:
                last = e
                continue
            val, err = _best_entry(_neville0([h * h for h in hs], ys))
            tol = _tol(digits, 1) * _scale(val)
            if err <= tol:
                break
            last = NotCovered(f'derivative did not converge to {digits} digits (estimated error {err:.2E}); the function may not be smooth at x0')
        else:
            raise last
        if order == 1:
            for sgn in (1, -1):
                hh = [h0 / 4 / Decimal(2) ** i for i in range(11)]
                f0 = _safe(fn, x)
                yy = [sgn * (_safe(fn, x + sgn * h) - f0) / h for h in hh]
                sv, se = _best_entry(_neville0(hh, yy))
                if abs(sv - val) > 100 * _tol(digits, 1) * _scale(val) + 10 * se:
                    raise NotCovered('the one-sided derivatives differ: f is not differentiable at x0')
        return _finish(val, err + Decimal(10) ** -(digits + 5), digits)


def _ts_nodes(W, level):
    """tanh-sinh nodes of one level on (0, 1): ([(d, weight)] with d the distance of the node to the nearer end, the centre weight or None)."""
    key = (W, level)
    got = _TS_CACHE.get(key)
    if got is not None:
        return got
    with _ctx(W + 5):
        pi = _pi(W + 10)
        h = Decimal(1) / (1 << level)
        step = 1 if level == 0 else 2
        k = 0 if level == 0 else 1
        out, centre = [], None
        cut = Decimal(10) ** -W
        while True:
            if k == 0:
                centre = pi / 4
                k += step
                continue
            e = (k * h).exp()
            sh = (e - 1 / e) / 2
            ch = (e + 1 / e) / 2
            E = (-pi * sh).exp()
            d = E / (1 + E)
            if d < cut:
                break
            w = pi / 2 * ch * 4 * E / (1 + E) ** 2 / 2
            out.append((+d, +w))
            k += step
        res = (out, centre)
    _TS_CACHE[key] = res
    return res


_TS_CACHE = {}


def _tanh_sinh(fn, a, b, W, digits):
    """int_a^b f by tanh-sinh, a and b Decimals (b may be +inf, a may be -inf, not both). Returns (value, error estimate)."""
    inf_hi, inf_lo = b.is_infinite(), a.is_infinite()
    if inf_lo and inf_hi:
        raise _Undefined('both ends infinite')
    L = b - a if not (inf_hi or inf_lo) else None

    def point(d, right):
        """(x, jacobian) of the node at distance d from the left end (right False) or the right end (right True), t in (0,1)."""
        if L is not None:
            return (b - L * d if right else a + L * d), L
        if inf_hi:                                              # x = a + t/(1-t)
            return (a + (1 - d) / d, 1 / (d * d)) if right else (a + d / (1 - d), 1 / ((1 - d) ** 2))
        # inf_lo: x = b - t/(1-t), t measured from b
        return (b - (1 - d) / d, 1 / (d * d)) if right else (b - d / (1 - d), 1 / ((1 - d) ** 2))

    def ev(x):
        return _safe(fn, x)
    total, absum = Decimal(0), Decimal(0)
    prev = prev_abs = None
    tol = _tol(digits, 2)
    near = Decimal(10) ** -(W // 2)
    dead = [None, None]                  # per end: the distance from which the function could not be evaluated (cancellation very close to the end)
    last_edge = Decimal(0)
    for level in range(0, 11):
        nodes, centre = _ts_nodes(W, level)
        if centre is not None:
            x, jac = (a + L / 2, L) if L is not None else (point(Decimal('0.5'), False))
            v = ev(x) * jac * centre
            total += v
            absum += abs(v)
        lastv = [Decimal(0), Decimal(0)]
        for d, w in nodes:
            for side in (0, 1):
                if dead[side] is not None and d <= dead[side]:
                    continue
                x, jac = point(d, bool(side))
                try:
                    fx = ev(x)
                except NotCovered:
                    if d >= near:
                        raise
                    dead[side] = d               # the rest of this end is cut here; the tail estimate below sees the last node that did evaluate
                    continue
                v = fx * jac * w
                total += v
                absum += abs(v)
                lastv[side] = abs(v)
        last_edge = max(lastv)
        h = Decimal(1) / (1 << level)
        cur, cur_abs = total * h, absum * h
        if prev is not None:
            diff = abs(cur - prev)
            if level >= 2 and diff <= tol * _scale(cur) and abs(cur_abs - prev_abs) <= Decimal('1e-3') * _scale(cur_abs):
                err = diff + 2 * last_edge * h
                if err > _tol(digits, 0) * _scale(cur):
                    raise NotCovered('the integrand does not decay or is singular at an end: the cut-off tail exceeds the digits asked')
                return cur, err
        prev, prev_abs = cur, cur_abs
    raise NotCovered('integral did not converge to the digits asked (a singularity or a kink inside, or a non-decaying tail); split it with `points`')


@op('search.integrate', kind='numeric calculus', bound='digits <= 60; 11 tanh-sinh levels')
def integrate(f, a, b, digits=15, points=()):
    """The definite integral of f over [a, b] by tanh-sinh quadrature (handles endpoint singularities and infinite limits 'inf' / '-inf'). `points` are interior
    breakpoints for a function with a kink (|x|) or a jump. An integrand with a pole or a kink inside, or that does not converge absolutely, raises NotCovered.
    Returns Numeric(value, error). integrate('x^2', 0, 3, digits=10) -> Numeric(value=Decimal('9.000000000'), ...)"""
    _check_digits(digits)
    if digits > 60:
        raise NotCovered('digits <= 60')
    W = 2 * digits + 25
    with _ctx(W):
        fn = _fun(f)
        lo, hi = _dec(a, 'a'), _dec(b, 'b')
        if lo == hi:
            return _finish(Decimal(0), Decimal(0), digits)
        sign = 1
        if lo > hi:
            lo, hi, sign = hi, lo, -1
        cuts = sorted({_dec(p, 'points') for p in points})
        if lo.is_infinite() and hi.is_infinite():
            cuts = sorted(set(cuts) | {Decimal(0)})
        if any(c <= lo or c >= hi for c in cuts):
            raise BadInput('points must lie strictly inside (a, b)')
        edges = [lo] + cuts + [hi]
        total, err = Decimal(0), Decimal(0)
        for i in range(len(edges) - 1):
            p, q = edges[i], edges[i + 1]
            if p.is_infinite() and q.is_infinite():
                raise _Undefined('unreachable')
            v, e = _tanh_sinh(fn, p, q, W, digits)
            total += v
            err += e
        return _finish(sign * total, err, digits)


@op('search.limit', kind='numeric calculus', bound='digits <= 40; 18 sample points')
def limit(f, point, digits=15, direction='+'):
    """The limit of f(x) as x -> point (an int, Fraction, str, or 'inf' / '-inf'), by sampling x = point + h for h = h0/2^k and extrapolating to h = 0
    (Neville in h, then Wynn's epsilon when the approach is not a power series). direction '+' (from above), '-' (from below) or 'both' (the two one-sided
    values must agree, else NotCovered). A limit that is infinite, does not exist or has log terms is refused. Returns Numeric(value, error).
    limit('sin(x)/x', 0, digits=12) -> Numeric(value=Decimal('1.00000000000'), ...)"""
    _check_digits(digits)
    if digits > 40 or direction not in ('+', '-', 'both'):
        raise NotCovered("digits <= 40; direction is '+', '-' or 'both'")
    W = 2 * digits + 60
    with _ctx(W):
        fn = _fun(f)
        p = _dec(point, 'point')

        def one_side(sgn):
            n = 18
            if p.is_infinite():
                s = 1 if p > 0 else -1
                xs = [s * Decimal(2) ** (6 + k) for k in range(n)]
                hs = [1 / abs(x) for x in xs]
            else:
                h0 = Decimal(1) / 64
                hs = [h0 / Decimal(2) ** k for k in range(n)]
                xs = [p + sgn * h for h in hs]
            ys = [_safe(fn, x) for x in xs]
            val, err = _best_entry(_neville0(hs, ys))
            tol = _tol(digits, 1) * _scale(val)
            if err <= tol:
                return val, err
            # Wynn epsilon on the sequence of values (geometric approach with a non-integer power)
            diffs = [abs(ys[i + 1] - ys[i]) for i in range(len(ys) - 1)]
            converging = all(diffs[i + 1] < Decimal('0.95') * diffs[i] for i in range(len(diffs) - 9, len(diffs) - 1)) if all(d > 0 for d in diffs[-9:]) else (diffs[-1] == 0)
            if not converging:
                raise NotCovered(f'the limit did not converge to {digits} digits (estimated error {err:.2E}): the values do not settle (infinite, absent or oscillating)')
            seq = ys
            e_prev = [Decimal(0)] * (len(seq) + 1)
            e_cur = list(seq)
            best = None
            col = 0
            while len(e_cur) > 1:
                new = []
                for i in range(len(e_cur) - 1):
                    dd = e_cur[i + 1] - e_cur[i]
                    new.append(e_prev[i + 1] + (1 / dd if dd != 0 else Decimal(10) ** (W // 2)))
                e_prev, e_cur = e_cur, new
                col += 1
                if col % 2 == 0 and len(e_cur) >= 2:
                    ee = abs(e_cur[-1] - e_cur[-2])
                    if best is None or ee < best[1]:
                        best = (e_cur[-1], ee)
            if best is not None and best[1] <= tol:
                return best
            raise NotCovered(f'the limit did not converge to {digits} digits (estimated error {min(err, best[1] if best else err):.2E}): infinite, absent, or with log terms')
        if direction == 'both' and not p.is_infinite():
            a, ea = one_side(-1)
            b, eb = one_side(1)
            if abs(a - b) > _tol(digits, 0) * _scale(a) + ea + eb:
                raise NotCovered(f'the one-sided limits differ ({a:.10} from below, {b:.10} from above)')
            return _finish((a + b) / 2, ea + eb + abs(a - b), digits)
        v, e = one_side(-1 if direction == '-' else 1)
        return _finish(v, e, digits)


@op('search.series_sum', kind='numeric calculus', bound='digits <= 40; 32768 terms')
def series_sum(term, start=1, stop='inf', digits=15):
    """Sum of term(n) for n = start..stop, `term` a text in n. A finite stop is summed directly; stop='inf' uses (1) the direct sum with a proven-by-ratio tail
    when the terms fall geometrically, (2) the Cohen-Villegas-Zagier acceleration when the signs strictly alternate, (3) Richardson extrapolation of the partial
    sums in 1/N for algebraic decay (1/n^2, 1/(n^2+1) ...). A divergent series, or a decay such as 1/(n ln n)^2, is refused. Returns Numeric(value, error).
    series_sum('1/n^2', digits=12) -> Numeric(value=Decimal('1.64493406685'), ...)"""
    _check_digits(digits)
    if digits > 40:
        raise NotCovered('digits <= 40')
    n0 = _int(start, 'start')
    W = 2 * digits + 40
    with _ctx(W):
        fn = _fun(term, ('n',))
        if not (isinstance(stop, str) and stop.strip().lower() in ('inf', 'infinity', '+inf')):
            n1 = _int(stop, 'stop')
            if n1 < n0:
                return _finish(Decimal(0), Decimal(0), digits)
            if n1 - n0 > 200000:
                raise NotCovered('more than 200000 terms')
            s = sum((_safe(fn, Decimal(n)) for n in range(n0, n1 + 1)), Decimal(0))
            return _finish(s, Decimal(10) ** -(W - 5) * _scale(s), digits)
        tol = _tol(digits, 1)
        # (1) geometric: the terms fall below 10^-(digits+10) within 4000 terms, with a steady ratio below 0.97 over the last 25 (an empirical tail bound)
        s, terms, ratios, small = Decimal(0), [], [], False
        for n in range(n0, n0 + 4000):
            t = _safe(fn, Decimal(n))
            terms.append(t)
            s += t
            if len(terms) > 1 and terms[-2] != 0:
                ratios.append(abs(t / terms[-2]))
            if len(terms) > 40 and abs(t) < Decimal(10) ** -(digits + 10) * _scale(s):
                small = True
                break
        if small:
            rmax = max(ratios[-25:], default=Decimal(0))
            if rmax < Decimal('0.97'):
                return _finish(s, abs(terms[-1]) * rmax / (1 - rmax) + Decimal(10) ** -(W - 10) * _scale(s), digits)
        # (2) alternating
        first = [_safe(fn, Decimal(n0 + k)) for k in range(60)]
        if (all(v != 0 for v in first) and all((first[k] > 0) != (first[k + 1] > 0) for k in range(59))
                and all(abs(first[k + 1]) < abs(first[k]) for k in range(30, 59))):       # strictly alternating, magnitudes falling: a divergent (-2)^n is not summed
            sg = 1 if first[0] > 0 else -1

            def cvz(m):
                a = [abs(_safe(fn, Decimal(n0 + k))) for k in range(m)]
                d = (3 + Decimal(8).sqrt()) ** m
                d = (d + 1 / d) / 2
                bb, c, tot = Decimal(-1), -d, Decimal(0)
                for k in range(m):
                    c = bb - c
                    tot += c * a[k]
                    bb = (k + m) * (k - m) * bb / ((k + Decimal('0.5')) * (k + 1))
                return sg * tot / d
            m = int(1.31 * (digits + 12)) + 8
            v1, v2 = cvz(m), cvz(m + 12)
            err = abs(v1 - v2)
            if err <= _tol(digits, 1) * _scale(v2):
                return _finish(v2, err, digits)
            raise NotCovered('alternating series did not converge to the digits asked (terms not smoothly decreasing?)')
        # (3) algebraic decay: Richardson on the partial sums S(N), N = 16 * 2^k
        Ns = [16 << k for k in range(11)]
        sums, acc, n = [], Decimal(0), n0
        for N in Ns:
            while n < n0 + N:
                acc += _safe(fn, Decimal(n))
                n += 1
            sums.append(acc)
        xs = [Decimal(1) / N for N in Ns]
        val, err = _best_entry(_neville0(xs, sums))
        if err <= tol * _scale(val):
            return _finish(val, err, digits)
        raise NotCovered(f'the series did not converge to {digits} digits by any of the three methods (divergent, or slower than a power of 1/n; estimate {err:.2E})')


@op('search.findroot', kind='numeric calculus', bound='digits <= 80; 4000 iterations')
def findroot(f, a, b, digits=15):
    """A root of f in the bracket [a, b] (f(a) and f(b) of opposite signs, else BadInput) by bisection safeguarded Newton steps (derivative by a central
    difference), so the root always stays inside a shrinking bracket. The error is the final bracket width. A sign change at a pole (tan on [1, 2]) is
    refused: the value at the end must be small compared to the end values. Returns Numeric(value, error).
    findroot('x^2 - 2', 1, 2, digits=12) -> Numeric(value=Decimal('1.41421356237'), ...)"""
    _check_digits(digits)
    if digits > 80:
        raise NotCovered('digits <= 80')
    W = digits + 30
    with _ctx(W):
        fn = _fun(f)
        lo, hi = _dec(a, 'a'), _dec(b, 'b')
        if lo > hi:
            lo, hi = hi, lo
        flo, fhi = _safe(fn, lo), _safe(fn, hi)
        if flo == 0:
            return _finish(lo, Decimal(0), digits)
        if fhi == 0:
            return _finish(hi, Decimal(0), digits)
        if (flo > 0) == (fhi > 0):
            raise BadInput('f(a) and f(b) have the same sign: no bracket')
        tol = _tol(digits, 2) * max(Decimal(1), abs(lo), abs(hi))
        fa0, fb0 = flo, fhi
        x, use_newton = (lo + hi) / 2, False

        def squeeze(z, fz):
            nonlocal lo, hi, flo, fhi
            if fz == 0:
                lo = hi = z
            elif (fz > 0) == (flo > 0):
                lo, flo = z, fz
            else:
                hi, fhi = z, fz
        for _ in range(4000):
            width0 = hi - lo
            fx = _safe(fn, x)
            squeeze(x, fx)
            if hi - lo <= tol:
                break
            nx = None
            if use_newton:
                hh = (hi - lo) / 1000
                try:
                    d = (_safe(fn, x + hh) - _safe(fn, x - hh)) / (2 * hh)
                    nx = x - fx / d if d != 0 else None
                except NotCovered:
                    nx = None
            if nx is not None and lo < nx < hi:
                delta = tol / 2                                   # close the bracket around the Newton point so the width is an honest error bound
                for cand in (nx - delta, nx + delta):
                    if lo < cand < hi:
                        squeeze(cand, _safe(fn, cand))
                x = nx if lo < nx < hi else (lo + hi) / 2
            else:
                x = (lo + hi) / 2
            if hi - lo > width0 / 2:
                use_newton = False                                # Newton did not halve the bracket: bisect next
            else:
                use_newton = True
            if hi - lo <= tol:
                break
        else:
            raise NotCovered('findroot: 4000 iterations without closing the bracket')
        mid = (lo + hi) / 2
        if hi != lo and abs(_safe(fn, mid)) > Decimal('1e-3') * max(abs(fa0), abs(fb0)):
            raise NotCovered('the sign change looks like a pole, not a root (|f| is not small inside the final bracket)')
        return _finish(mid, (hi - lo) + Decimal(10) ** -(digits + 5), digits)


def _ode_rhs(rhs, n):
    names = ('t',) + tuple(f'y{i + 1}' for i in range(n))
    if n == 1:
        names = ('t', 'y1', 'y')
    fs = [_dec_compile(r, names) for r in rhs]
    if n == 1:
        return lambda t, y: [_safe(fs[0], (t, y[0], y[0]))]
    return lambda t, y: [_safe(f, (t,) + tuple(y)) for f in fs]


def _gbs(F, t0, t1, y0, W, tol_global):
    """Gragg-Bulirsch-Stoer with step control. Returns (y at t1, summed local error estimate, steps)."""
    n = len(y0)
    seqn = [2 * (j + 1) for j in range(8)]
    t, y = t0, list(y0)
    span = t1 - t0
    H = span / 10
    err_total, steps = Decimal(0), 0
    sgn = 1 if span > 0 else -1
    while sgn * (t1 - t) > 0:
        steps += 1
        if steps > 3000:
            raise NotCovered('ode_ivp: more than 3000 steps (stiff, or a singularity)')
        if sgn * (t + H - t1) > 0:
            H = t1 - t
        tol_local = tol_global * abs(H / span)
        f0 = F(t, y)
        rows = []
        accepted = False
        for j, m in enumerate(seqn):
            h = H / m
            z0, z1 = list(y), [yi + h * fi for yi, fi in zip(y, f0)]
            for i in range(1, m):
                fz = F(t + i * h, z1)
                z0, z1 = z1, [a + 2 * h * b for a, b in zip(z0, fz)]
            fz = F(t + H, z1)
            yj = [(a + b + h * c) / 2 for a, b, c in zip(z0, z1, fz)]
            rows.append(yj)
            if j >= 2:
                xs = [(H / seqn[i]) ** 2 for i in range(j + 1)]
                comps, errs = [], []
                for c in range(n):
                    T = _neville0(xs, [rows[i][c] for i in range(j + 1)])
                    comps.append(T[j][j])
                    errs.append(abs(T[j][j] - T[j][j - 1]))
                e = max(errs)
                if e <= tol_local * max(Decimal(1), max(abs(v) for v in comps)):
                    y, t = comps, t + H
                    err_total += e
                    accepted = True
                    if j <= 3:
                        H *= Decimal('1.6')
                    break
        if not accepted:
            H /= 2
            if abs(H) < abs(span) * Decimal(10) ** -12:
                raise NotCovered('ode_ivp: step size underflow')
        if any(abs(v) > Decimal(10) ** 60 for v in y):
            raise NotCovered('ode_ivp: the solution blows up')
    return y, err_total, steps


@op('search.ode_ivp', kind='numeric calculus', bound='digits <= 30; 3000 steps per run; up to 4 refinements')
def ode_ivp(rhs, y0, t0, t1, digits=12):
    """Initial value problem y' = rhs(t, y) from t0 to t1 by Gragg-Bulirsch-Stoer extrapolation with step control. rhs: one text ('-2*t*y', variables t and y) or a
    list of texts for a system (variables t, y1, y2, ...); y0: the initial value(s). The run is repeated at a tighter tolerance and the two results are compared,
    so the error estimate includes the growth of the local errors; it must be below the digits asked or the call raises NotCovered. Returns Numeric(value, error)
    where value is a tuple of Decimals (one per component). ode_ivp('y', 1, 0, 1, digits=10) -> value (Decimal('2.718281828'),)"""
    _check_digits(digits)
    if digits > 30:
        raise NotCovered('digits <= 30')
    rl = [rhs] if isinstance(rhs, str) else list(rhs)
    y0l = [y0] if not isinstance(y0, (list, tuple)) else list(y0)
    if len(rl) != len(y0l) or not rl or len(rl) > 8:
        raise BadInput('rhs and y0 must have the same length (1..8)')
    tol_goal = _tol(digits, 1)
    prev = None
    for rnd in range(4):
        W = digits + 20 + 6 * rnd
        with _ctx(W):
            F = _ode_rhs(rl, len(rl))
            ys = [_dec(v, 'y0') for v in y0l]
            ta, tb = _dec(t0, 't0'), _dec(t1, 't1')
            if ta == tb:
                return _finish_vec(ys, Decimal(0), digits)
            tol_run = _tol(digits, 3 + 3 * rnd)
            y, e, steps = _gbs(F, ta, tb, ys, W, tol_run)
            if prev is not None:
                diff = max(abs(a - b) for a, b in zip(prev, y))
                if diff <= tol_goal * max(Decimal(1), max(abs(v) for v in y)):
                    return _finish_vec(y, diff + e, digits)
            prev = y
    raise NotCovered('ode_ivp: two runs at different tolerances disagree beyond the digits asked (unstable or stiff problem)')


def _finish_vec(ys, err, digits):
    return Numeric(tuple(_round(v, digits) for v in ys), _sig3(err))


# ----------------------------------------------------------------------------------------------------------------------------------------
# normal and Poisson distributions (numeric)
# ----------------------------------------------------------------------------------------------------------------------------------------

def _phi(z, W):
    """Standard normal cdf of a Decimal z at relative accuracy for W digits (extra digits for the cancellation in the tail)."""
    extra = int(z * z / Decimal('4.6')) + 12
    with _ctx(W + extra):
        w = z / Decimal(2).sqrt()
        e = _erf(abs(w))
        return (1 + e) / 2 if z >= 0 else (1 - e) / 2


@op('search.normal_cdf', kind='probability', bound='|z| <= 38, digits <= 80')
def normal_cdf(x, mu=0, sigma=1, digits=30):
    """P(X <= x) for X ~ Normal(mu, sigma): the cdf to `digits` significant digits (erf by its all-positive series, extra digits for the tails).
    x, mu, sigma are int, Fraction or str; sigma > 0; |x - mu| / sigma <= 38. Returns Numeric(value, error).
    normal_cdf(0, digits=10) -> Numeric(value=Decimal('0.5000000000'), ...)"""
    _check_digits(digits)
    if digits > 80:
        raise NotCovered('digits <= 80')
    xq, mq, sq = _q(x, 'x'), _q(mu, 'mu'), _q(sigma, 'sigma')
    if sq <= 0:
        raise BadInput('sigma must be positive')
    zq = (xq - mq) / sq
    if abs(zq) > 38:
        raise NotCovered('|z| > 38')
    res = []
    for g in (15, 28):
        with _ctx(digits + g):
            z = Decimal(zq.numerator) / Decimal(zq.denominator)
            res.append(_phi(z, digits + g))
    return _finish(res[1], abs(res[1] - res[0]) + abs(res[1]) * Decimal(10) ** -(digits + 8), digits, rel=True)


@op('search.normal_ppf', kind='probability', bound='1e-300 < p < 1, digits <= 60')
def normal_ppf(p, digits=20, mu=0, sigma=1):
    """The quantile z with P(X <= z) = p for X ~ Normal(mu, sigma): bisection on the cdf, then Newton steps; p is an exact rational (int, Fraction, str).
    Returns Numeric(value, error). normal_ppf('0.975', digits=10) -> Numeric(value=Decimal('1.959963985'), ...)"""
    _check_digits(digits)
    if digits > 60:
        raise NotCovered('digits <= 60')
    pq, mq, sq = _q(p, 'p'), _q(mu, 'mu'), _q(sigma, 'sigma')
    if not 0 < pq < 1:
        raise BadInput('p must be strictly between 0 and 1')
    if sq <= 0:
        raise BadInput('sigma must be positive')
    flip = pq > Fraction(1, 2)
    qq = 1 - pq if flip else pq                                   # the lower tail, exact
    if qq < Fraction(1, 10 ** 300):
        raise NotCovered('p within 1e-300 of 0 or 1')
    W = digits + 20
    with _ctx(W + 10):
        target = Decimal(qq.numerator) / Decimal(qq.denominator)
        lo, hi = Decimal(-38), Decimal(0)
        for _ in range(40):                                      # bisection to about 1e-10 relative to the width
            mid = (lo + hi) / 2
            if _phi(mid, 15) < target:
                lo = mid
            else:
                hi = mid
        z = (lo + hi) / 2
        pi2 = (2 * _pi(W + 15)).sqrt()
        step = Decimal(1)
        for _ in range(60):
            pdf = (-z * z / 2).exp() / pi2
            step = (_phi(z, W) - target) / pdf
            z -= step
            if abs(step) < Decimal(10) ** -(digits + 6) * max(Decimal(1), abs(z)):
                break
        else:
            raise NotCovered('normal_ppf: Newton did not converge')
        if flip:
            z = -z
        m = Decimal(mq.numerator) / Decimal(mq.denominator)
        s = Decimal(sq.numerator) / Decimal(sq.denominator)
        val = m + s * z
        return _finish(val, abs(step) * s + Decimal(10) ** -(digits + 8) * _scale(val), digits)


def _poisson_lam(lam):
    lam = _q(lam, 'lam')
    if lam < 0:
        raise BadInput('lam must be non-negative')
    if lam > 2000:
        raise NotCovered('lam > 2000')
    return lam


@op('search.poisson_pmf', kind='probability', bound='lam <= 2000, k <= 100000, digits <= 60')
def poisson_pmf(lam, k, digits=30):
    """P(X = k) for X ~ Poisson(lam), to `digits` significant digits: e^-lam lam^k / k! with the exact Fraction lam^k/k! and a Decimal e^-lam.
    Returns Numeric(value, error). poisson_pmf(2, 3, digits=10) -> Numeric(value=Decimal('0.1804470443'), ...)"""
    _check_digits(digits)
    lam, k = _poisson_lam(lam), _int(k, 'k')
    if k > 100000 or digits > 60:
        raise NotCovered('k > 100000 or digits > 60')
    if k < 0:
        return _finish(Decimal(0), Decimal(0), digits, rel=True)
    frac = lam ** k / factorial(k)
    res = []
    for g in (15, 28):
        with _ctx(digits + g + int(lam) // 2 + 5):
            lamd = Decimal(lam.numerator) / Decimal(lam.denominator)
            res.append((Decimal(frac.numerator) / Decimal(frac.denominator)) * (-lamd).exp())
    return _finish(res[1], abs(res[1] - res[0]) + abs(res[1]) * Decimal(10) ** -(digits + 8), digits, rel=True)


@op('search.poisson_cdf', kind='probability', bound='lam <= 2000, k <= 100000, digits <= 60')
def poisson_cdf(lam, k, digits=30):
    """P(X <= k) for X ~ Poisson(lam), to `digits` significant digits: e^-lam times the exact truncated series poisson_partial_sum(lam, k).
    Returns Numeric(value, error). poisson_cdf(2, 3, digits=10) -> Numeric(value=Decimal('0.8571234605'), ...)"""
    _check_digits(digits)
    lam, k = _poisson_lam(lam), _int(k, 'k')
    if k > 100000 or digits > 60:
        raise NotCovered('k > 100000 or digits > 60')
    if k < 0:
        return _finish(Decimal(0), Decimal(0), digits, rel=True)
    s = poisson_partial_sum(lam, k)
    res = []
    for g in (15, 28):
        with _ctx(digits + g + int(lam) // 2 + 5):
            lamd = Decimal(lam.numerator) / Decimal(lam.denominator)
            res.append((Decimal(s.numerator) / Decimal(s.denominator)) * (-lamd).exp())
    return _finish(res[1], abs(res[1] - res[0]) + abs(res[1]) * Decimal(10) ** -(digits + 8), digits, rel=True)
