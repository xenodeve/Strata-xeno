"""Simulation of tools/igc/mathbackend/algebra.py against independent oracles (no model).

    PYTHONPATH=. PYTHONUTF8=1 <venv-with-sympy>/python -m tools.igc.mathbackend.sim_algebra --seed 1 --n 60

Every case is (op_name, args, oracle, oracle_name). `oracle` is the expected value, a callable that certifies the backend's answer (a certificate: the factors
multiply back, a root satisfies the polynomial, A * inverse = I, ...), or Raises(ExceptionType) when the input must be refused as bad. The backend code was written
from the textbook algorithms and copies nothing from SymPy or mpmath, so SymPy / mpmath / itertools brute force are independent oracles for it; where a case uses a
certificate the name says so ('cert:...').

generate(seed, n) -> list[Case]      run(seed, n) -> dict (also written to build-mathbackend/algebra.json at the repo root)
"""
import argparse
import json
import random
import sys
import time
from collections import Counter, namedtuple
from decimal import Decimal
from fractions import Fraction
from itertools import combinations
from math import gcd, prod
from pathlib import Path

import sympy
from sympy.polys import galoistools as gt
from sympy import ZZ, QQ

from tools.igc.mathbackend import BadInput, NotCovered, Surd, run as backend_run, registry
from tools.igc.mathbackend import algebra as A

Case = namedtuple('Case', 'op args oracle oracle_name')
x = sympy.Symbol('x')


class Raises:
    def __init__(self, exc):
        self.exc = exc


class MayRefuse:
    """The case may be refused by the operation's own stated scope or bound (counted as an expected refusal); if it answers, `check` must hold."""
    def __init__(self, check):
        self.check = check


class Answers:
    """The case is inside the operation's stated scope: a refusal (NotCovered) is a failure; the answer is checked by `check` (value or callable)."""
    def __init__(self, check):
        self.check = check


# ------------------------------------------------------------------ helpers

def fr(r):
    r = sympy.Rational(r)
    return Fraction(int(r.p), int(r.q))


def sy(f):
    f = Fraction(f)
    return sympy.Rational(f.numerator, f.denominator)


def P(co):
    return sympy.Poly([sy(c) for c in co], x, domain='QQ') if len(co) else sympy.Poly(0, x, domain='QQ')


def co(poly):
    poly = sympy.Poly(poly, x, domain='QQ') if not isinstance(poly, sympy.Poly) else poly
    if poly.is_zero:
        return []
    return [fr(c) for c in poly.all_coeffs()]


def expr(cf):
    return sum(sy(c) * x ** (len(cf) - 1 - i) for i, c in enumerate(cf))


def norm(v):
    if isinstance(v, (list, tuple)):
        return [norm(a) for a in v]
    if isinstance(v, dict):
        return {k: norm(a) for k, a in v.items()}
    if isinstance(v, int) and not isinstance(v, bool):
        return Fraction(v)
    return v


def same(a, b):
    return norm(a) == norm(b)


def rint(rng, lo=-9, hi=9):
    return rng.randint(lo, hi)


def rpoly(rng, deg=None, lo=-9, hi=9, frac=False):
    deg = rng.randint(0, 6) if deg is None else deg
    cf = [Fraction(rint(rng, lo, hi), rng.choice([1, 2, 3, 4, 6]) if frac else 1) for _ in range(deg + 1)]
    while cf[0] == 0:
        cf[0] = Fraction(rint(rng, 1, hi) * rng.choice([1, -1]), 1)
    return cf


def rroots_poly(rng, k, den=True):
    """(roots list as Fractions, monic-with-lead polynomial from them) built by direct multiplication in the sim."""
    roots = [Fraction(rint(rng, -7, 7), rng.choice([1, 1, 2, 3]) if den else 1) for _ in range(k)]
    lead = rng.choice([1, 1, 2, -3, 5]) if den else 1
    cf = [Fraction(lead)]
    for r in roots:
        nxt = cf + [Fraction(0)]
        for i, c in enumerate(cf):
            nxt[i + 1] -= r * c
        cf = nxt
    return roots, cf


def poly_to_text(cf, style, rng):
    s = str(expr(cf))
    if style == 'py':
        return s
    if style == 'latex':
        return sympy.latex(expr(cf))
    if style == 'caret':
        return s.replace('**', '^').replace('*', '')
    return s.replace('**', '^')


# ------------------------------------------------------------------ generators: each returns a list of (args, oracle, oracle_name)

def g_poly_from_str(rng, n):
    out = []
    fixed = [('(x+1)^2 - 3x/2', [1, Fraction(1, 2), 1]), ('2x^3 - x + 1/2', [2, 0, -1, Fraction(1, 2)]), ('x^2', [1, 0, 0]), ('0', []), ('7', [7]),
             ('(x-1)(x+1)', [1, 0, -1]), ('(x+1)^5', [1, 5, 10, 10, 5, 1]), ('\\frac{x^2}{2} + \\frac{3}{4}', [Fraction(1, 2), 0, Fraction(3, 4)]),
             ('-x^2 + x', [-1, 1, 0]), ('3(x-2)^2', [3, -12, 12]), ('x**3 - 2*x', [1, 0, -2, 0]), ('0.5x + 0.25', [Fraction(1, 2), Fraction(1, 4)])]
    for s, c in fixed:
        out.append(((s,), [Fraction(v) for v in c], 'hand-computed'))
    while len(out) < n:
        c = rpoly(rng, rng.randint(0, 7), frac=rng.random() < 0.4)
        style = rng.choice(['py', 'latex', 'caret'])
        k = rng.random()
        if k < 0.6:
            out.append(((poly_to_text(c, style, rng),), c, 'SymPy printer round trip (Poly.all_coeffs of the expression)'))
        elif k < 0.85:
            c2 = rpoly(rng, rng.randint(0, 3))
            s = f'({poly_to_text(c, "caret", rng)})({poly_to_text(c2, "caret", rng)})'
            out.append(((s,), co(P(c) * P(c2)), 'SymPy Poly product'))
        else:
            c2 = rpoly(rng, 1)
            e = rng.randint(2, 5)
            s = f'({poly_to_text(c2, "caret", rng)})^{e}'
            out.append(((s,), co(P(c2) ** e), 'SymPy Poly power'))
    return out


def g_poly_to_str(rng, n):
    out = []
    while len(out) < n:
        c = rpoly(rng, rng.randint(0, 7), frac=rng.random() < 0.5)
        def cert(r, c=c):
            return sympy.Poly(sympy.sympify(r.replace('^', '**')), x, domain='QQ') == P(c)
        out.append(((c,), cert, 'cert: SymPy reads the text back to the same polynomial'))
    return out


def two_polys(rng, allow_zero=True):
    a = rpoly(rng, rng.randint(0, 6), frac=rng.random() < 0.3)
    b = rpoly(rng, rng.randint(0, 5), frac=rng.random() < 0.3)
    r = rng.random()
    if allow_zero and r < 0.06:
        a = []
    elif allow_zero and r < 0.1:
        b = []
    return a, b


def simple_binary(fn, name):
    def g(rng, n):
        out = []
        while len(out) < n:
            a, b = two_polys(rng)
            out.append(((a, b), co(fn(P(a), P(b))), name))
        return out
    return g


g_poly_add = simple_binary(lambda p, q: p + q, 'SymPy Poly over QQ')
g_poly_sub = simple_binary(lambda p, q: p - q, 'SymPy Poly over QQ')
g_poly_mul = simple_binary(lambda p, q: p * q, 'SymPy Poly over QQ')


def g_poly_pow(rng, n):
    out = []
    while len(out) < n:
        a = rpoly(rng, rng.randint(0, 4), frac=rng.random() < 0.3)
        e = rng.randint(0, 8)
        out.append(((a, e), co(P(a) ** e), 'SymPy Poly over QQ'))
    out.append((([], 0), [Fraction(1)], 'hand: 0^0 = 1 (empty product)'))
    out.append((([1, 1], -1), Raises(BadInput), 'negative exponent'))
    return out


def g_poly_divmod(rng, n):
    out = []
    while len(out) < n:
        a, b = two_polys(rng, False)
        if rng.random() < 0.3:
            b = co(P(b) * P(rpoly(rng, 2)))
            a = co(P(a) * P(b))
        q, r = P(a).div(P(b))
        out.append(((a, b), (co(q), co(r)), 'SymPy Poly.div'))
    out.append((([1, 2], []), Raises(BadInput), 'division by the zero polynomial'))
    return out


def common_factor_pair(rng):
    f = rpoly(rng, rng.randint(0, 3))
    a = co(P(f) * P(rpoly(rng, rng.randint(0, 3))))
    b = co(P(f) * P(rpoly(rng, rng.randint(0, 3))))
    return a, b


def g_poly_gcd(rng, n):
    out = []
    while len(out) < n:
        a, b = two_polys(rng) if rng.random() < 0.4 else common_factor_pair(rng)
        g = P(a).gcd(P(b))
        out.append(((a, b), co(g.monic()) if not g.is_zero else [], 'SymPy Poly.gcd (monic)'))
    return out


def g_poly_lcm(rng, n):
    out = []
    while len(out) < n:
        a, b = two_polys(rng, False) if rng.random() < 0.5 else common_factor_pair(rng)
        if not a or not b:
            continue
        l = P(a).lcm(P(b))
        out.append(((a, b), co(l.monic()), 'SymPy Poly.lcm (monic)'))
    out.append((([], [1, 2]), [], 'hand: lcm with zero is zero'))
    return out


def g_poly_gcdex(rng, n):
    out = []
    while len(out) < n:
        a, b = two_polys(rng, False) if rng.random() < 0.5 else common_factor_pair(rng)
        if not a or not b:
            continue
        g = P(a).gcd(P(b)).monic()

        def cert(r, a=a, b=b, g=g):
            s, t, gg = r
            return P(s) * P(a) + P(t) * P(b) == P(gg) and P(gg) == g
        out.append(((a, b), cert, 'cert: s*p + t*q equals the SymPy monic gcd'))
    return out


def g_poly_compose(rng, n):
    out = []
    while len(out) < n:
        a = rpoly(rng, rng.randint(0, 4), frac=rng.random() < 0.3)
        b = rpoly(rng, rng.randint(0, 3), frac=rng.random() < 0.3)
        out.append(((a, b), co(P(a).compose(P(b))), 'SymPy Poly.compose'))
    return out


def g_poly_derivative(rng, n):
    out = []
    while len(out) < n:
        a = rpoly(rng, rng.randint(0, 8), frac=rng.random() < 0.3)
        k = rng.randint(0, 4)
        out.append(((a, k), co(P(a).diff((x, k))) if k else co(P(a)), 'SymPy Poly.diff'))
    return out


def g_poly_antiderivative(rng, n):
    out = []
    while len(out) < n:
        a = rpoly(rng, rng.randint(0, 8), frac=rng.random() < 0.3)
        out.append(((a,), co(P(a).integrate()), 'SymPy Poly.integrate'))
    return out


def g_poly_integral(rng, n):
    out = []
    while len(out) < n:
        a = rpoly(rng, rng.randint(0, 6), frac=rng.random() < 0.3)
        lo, hi = Fraction(rint(rng, -5, 5), rng.choice([1, 2, 3])), Fraction(rint(rng, -5, 5), rng.choice([1, 2, 3]))
        v = sympy.integrate(expr(a), (x, sy(lo), sy(hi)))
        out.append(((a, lo, hi), fr(v), 'SymPy integrate'))
    return out


def g_poly_eval(rng, n):
    out = []
    while len(out) < n:
        a = rpoly(rng, rng.randint(0, 8), frac=rng.random() < 0.3)
        v = Fraction(rint(rng, -9, 9), rng.choice([1, 2, 3, 7]))
        out.append(((a, v), fr(P(a).eval(sy(v))), 'SymPy Poly.eval'))
    for _ in range(max(0, n // 6)):               # Surd argument: certificate by SymPy's exact algebra
        a = rpoly(rng, rng.randint(1, 4))
        d = rng.choice([2, 3, 5, 6, 7])
        p0, q0 = rint(rng, -4, 4), rint(rng, 1, 4)
        v = Surd(p0, q0, d)

        def cert(r, a=a, p0=p0, q0=q0, d=d):
            e = sympy.expand(expr(a).subs(x, p0 + q0 * sympy.sqrt(d)))
            ca = e.coeff(sympy.sqrt(d), 1)
            cb = e - ca * sympy.sqrt(d)
            return isinstance(r, Surd) and r.q == fr(ca) and r.p == fr(cb)
        out.append(((a, v), cert, 'cert: SymPy expands p(p0 + q0*sqrt(d)) exactly'))
    return out


def g_poly_taylor_shift(rng, n):
    out = []
    while len(out) < n:
        a = rpoly(rng, rng.randint(0, 7), frac=rng.random() < 0.3)
        s = Fraction(rint(rng, -6, 6), rng.choice([1, 2, 3]))
        out.append(((a, s), co(P(a).shift(sy(s))), 'SymPy Poly.shift'))
    return out


def g_poly_content_primitive(rng, n):
    out = []
    while len(out) < n:
        a = rpoly(rng, rng.randint(0, 6), frac=True)
        c = Fraction(rint(rng, 1, 6), rng.choice([1, 2, 3, 5]))
        a = [c * v for v in a]

        def cert(r, a=a):
            ct, pr = r
            ints = all(Fraction(v).denominator == 1 for v in pr)
            g = 0
            for v in pr:
                g = gcd(g, int(v))
            return ints and g == 1 and pr[0] > 0 and [ct * v for v in pr] == [Fraction(v) for v in a]
        out.append(((a,), cert, 'cert: content * primitive = p, primitive integer with gcd 1 and positive lead'))
    out.append((([],), (Fraction(0), []), 'hand: zero polynomial'))
    return out


def g_poly_squarefree_part(rng, n):
    out = []
    while len(out) < n:
        f = rpoly(rng, rng.randint(1, 3))
        h = rpoly(rng, rng.randint(0, 2))
        a = co(P(f) ** rng.randint(1, 3) * P(h))
        sp = P(a).sqf_part().monic()

        def cert(r, a=a, sp=sp):
            return P(r).monic() == sp and P(r).degree() == sp.degree() and not P(a).rem(P(r))
        out.append(((a,), cert, 'cert: same monic polynomial as SymPy sqf_part, divides p'))
    return out


def g_poly_squarefree_decomposition(rng, n):
    out = []
    while len(out) < n:
        f1, f2 = rpoly(rng, rng.randint(1, 2)), rpoly(rng, rng.randint(1, 2))
        a = co(P(f1) ** rng.randint(1, 3) * P(f2) ** rng.randint(1, 3) * rpoly(rng, 0)[0])
        lc, lst = sympy.Poly(expr(a), x).sqf_list()
        want = {}
        for f, k in lst:
            want[k] = want.get(k, sympy.Poly(1, x)) * f.monic()

        def cert(r, a=a, want=want):
            lcc, fs = r
            tot = P([lcc])
            seen = {}
            for g, m in fs:
                tot = tot * P(g) ** m
                seen[m] = seen.get(m, sympy.Poly(1, x, domain='QQ')) * P(g)
            return tot == P(a) and {k: v.monic() for k, v in seen.items()} == {k: sympy.Poly(v.as_expr(), x, domain='QQ').monic() for k, v in want.items()}
        out.append(((a,), cert, 'cert: lc * prod g^m = p, and the grouping by multiplicity equals SymPy sqf_list'))
    return out


def g_poly_from_roots(rng, n):
    out = []
    while len(out) < n:
        k = rng.randint(0, 6)
        roots = [Fraction(rint(rng, -6, 6), rng.choice([1, 2, 3])) for _ in range(k)]
        lead = Fraction(rint(rng, 1, 5) * rng.choice([1, -1]), rng.choice([1, 2]))
        e = sympy.Poly(sympy.prod([x - sy(r) for r in roots]) * sy(lead), x, domain='QQ')
        out.append(((roots, lead), co(e), 'SymPy product of (x - r)'))
    return out


def g_poly_transform_roots(rng, n):
    out = []
    while len(out) < n:
        a = rpoly(rng, rng.randint(1, 6), frac=rng.random() < 0.3)
        kind = rng.choice(['shift', 'scale', 'reciprocal'])
        c = Fraction(rint(rng, -5, 5), rng.choice([1, 2, 3]))
        deg = len(a) - 1
        pe = expr(a)
        if kind == 'shift':
            want = co(sympy.Poly(pe.subs(x, x - sy(c)), x, domain='QQ'))
        elif kind == 'scale':
            if c == 0:
                c = Fraction(2)
            want = co(sympy.Poly(sympy.expand(pe.subs(x, x / sy(c)) * sy(c) ** deg), x, domain='QQ'))
        else:
            if a[-1] == 0:
                continue
            want = co(sympy.Poly(sympy.expand(x ** deg * pe.subs(x, 1 / x)), x, domain='QQ'))
        out.append(((a, kind, c), want, 'SymPy substitution'))
    return out


# --- roots

def rational_poly_with_roots(rng, extra=True):
    k = rng.randint(1, 4)
    roots = [Fraction(rint(rng, -6, 6), rng.choice([1, 2, 3])) for _ in range(k)]
    if rng.random() < 0.4:
        roots = roots + [roots[0]] * rng.randint(1, 2)
    e = sympy.prod([x - sy(r) for r in roots]) * rng.choice([1, 2, -3])
    if extra and rng.random() < 0.5:
        e *= rng.choice([x ** 2 + 1, x ** 2 - 2, x ** 3 - 2, x ** 2 + x + 1, x ** 4 + 1])
    return co(sympy.Poly(e, x, domain='QQ'))


def g_rational_roots(rng, n):
    out = []
    while len(out) < n:
        a = rational_poly_with_roots(rng) if rng.random() < 0.8 else rpoly(rng, rng.randint(1, 5))
        d = P(a).ground_roots()
        want = sorted((fr(r), int(m)) for r, m in d.items())
        out.append(((a,), want, 'SymPy Poly.ground_roots'))
    out.append((([],), Raises(BadInput), 'zero polynomial'))
    return out


def g_sturm_sequence(rng, n):
    out = []
    while len(out) < n:
        a = rational_poly_with_roots(rng) if rng.random() < 0.5 else rpoly(rng, rng.randint(1, 6))

        def cert(seq, a=a):
            s = [P(v) for v in seq]
            g = P(a).quo(P(a).gcd(P(a).diff(x)))
            if s[0].monic() != g.monic():
                return False
            if len(s) > 1 and s[1].monic() != g.diff(x).monic():
                return False
            for i in range(1, len(s) - 1):
                r = s[i - 1].rem(s[i])
                if r.is_zero or r.degree() != s[i + 1].degree():
                    return False
                lam = r.LC() / s[i + 1].LC()
                if lam * s[i + 1] != r or lam >= 0:
                    return False        # remainder must be a NEGATIVE multiple of the next member
            return s[-1].degree() == 0 or len(s) == 1
        out.append(((a,), cert, 'cert: each member is a negative multiple of the remainder of the two before (SymPy rem)'))
    return out


def real_roots_of(a):
    """Counter {sympy real root: multiplicity} of the polynomial a, exact via SymPy's CRootOf isolation."""
    return Counter(P(a).real_roots())


def cnt_in(rts, lo, hi):
    n = 0
    for r, m in rts.items():
        if (lo is None or r > sy(lo)) and (hi is None or r <= sy(hi)):
            n += 1
    return n


def g_count_real_roots(rng, n):
    out = []
    while len(out) < n:
        a = rational_poly_with_roots(rng) if rng.random() < 0.6 else rpoly(rng, rng.randint(1, 7))
        rts = real_roots_of(a)
        lo = hi = None
        if rng.random() < 0.7:
            lo = Fraction(rint(rng, -8, 8), rng.choice([1, 2, 3]))
            hi = lo + Fraction(rint(rng, 0, 12), rng.choice([1, 2, 3]))
        elif rng.random() < 0.5:
            lo = Fraction(rint(rng, -8, 8), 2)
        mult = rng.random() < 0.25
        if mult:
            want = sum(m for r, m in rts.items() if (lo is None or r > sy(lo)) and (hi is None or r <= sy(hi)))
            out.append(((a, lo, hi, True), want, 'SymPy real_roots (exact CRootOf) counted in (a, b], with multiplicity'))
        else:
            out.append(((a, lo, hi), cnt_in(rts, lo, hi), 'SymPy real_roots (exact CRootOf) counted in (a, b]'))
    return out


def g_isolate_real_roots(rng, n):
    out = []
    while len(out) < n:
        a = rational_poly_with_roots(rng) if rng.random() < 0.6 else rpoly(rng, rng.randint(1, 7))
        rts = real_roots_of(a)

        def cert(r, a=a, rts=rts):
            if len(r) != len(rts):
                return False
            used = set()
            prev_hi = None
            for lo, hi, m in r:
                if prev_hi is not None and lo < prev_hi:
                    return False
                prev_hi = hi
                if lo == hi:
                    if P(a).eval(sy(lo)) != 0:
                        return False
                    hit = [t for t in rts if t == sy(lo)]
                else:
                    hit = [t for t in rts if sy(lo) < t < sy(hi)]
                if len(hit) != 1 or rts[hit[0]] != m or hit[0] in used:
                    return False
                used.add(hit[0])
            return True
        out.append(((a,), cert, 'cert: each interval holds exactly one SymPy real root (exact), multiplicity equal, intervals disjoint, none missing'))
    return out


def g_real_roots(rng, n):
    out = []
    while len(out) < n:
        a = rational_poly_with_roots(rng) if rng.random() < 0.5 else rpoly(rng, rng.randint(1, 6))
        d = rng.choice([3, 8, 15, 30])
        rts = real_roots_of(a)

        def cert(r, a=a, d=d, rts=rts):
            if len(r) != len(rts):
                return False
            for item, (t, m) in zip(r, sorted(rts.items(), key=lambda kv: float(kv[0]))):
                if item['multiplicity'] != m:
                    return False
                if item['exact'] != (t.is_Rational):
                    return False
                v = sympy.Rational(item['root']) if isinstance(item['root'], Fraction) else sympy.Rational(str(item['root']))
                if item['exact']:
                    if v != t:
                        return False
                elif abs(sympy.N(t - v, d + 15)) >= sympy.Float(10) ** (-d) or not item['numeric']:
                    return False
                lo, hi = item['interval']
                if not (sy(lo) <= t <= sy(hi)):
                    return False
            return True
        out.append(((a, d), cert, 'cert: |root - SymPy root (N to d+15 digits)| < 10^-d, exactness and multiplicity as SymPy, interval holds the root'))
    return out


def check_complex(a, digits, res):
    import mpmath
    mpmath.mp.dps = digits + 30
    pe = sympy.Poly(expr(a), x, domain='QQ')
    lc, lst = pe.sqf_list()
    tuples = Counter(res)
    zeros = Counter()
    tol = mpmath.mpf(10) ** (-digits + 1)
    for z, cnt in tuples.items():
        zz = mpmath.mpc(mpmath.mpf(str(z[0])), mpmath.mpf(str(z[1])))
        assigned = []
        for f, k in lst:
            fc = [mpmath.mpf(int(c.p)) / int(c.q) for c in f.all_coeffs()]
            v = mpmath.polyval(fc, zz)
            dv = mpmath.polyval([c * (len(fc) - 1 - i) for i, c in enumerate(fc[:-1])], zz)
            if dv != 0 and abs(v / dv) < tol:
                assigned.append(k)
        if len(assigned) != 1 or assigned[0] != cnt:
            return False
        zeros[assigned[0]] += 1
    for f, k in lst:
        if zeros[k] < 0:
            return False
    want = Counter()
    for f, k in lst:
        want[k] += f.degree()
    if Counter(zeros) != want:
        return False
    # pairwise distinct
    keys = list(tuples)
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            if abs(mpmath.mpc(mpmath.mpf(str(keys[i][0])) - mpmath.mpf(str(keys[j][0])), mpmath.mpf(str(keys[i][1])) - mpmath.mpf(str(keys[j][1])))) < tol:
                return False
    return True


def g_complex_roots(rng, n):
    out = []
    while len(out) < n:
        a = rational_poly_with_roots(rng) if rng.random() < 0.5 else rpoly(rng, rng.randint(1, 7))
        d = rng.choice([8, 15, 25])
        out.append(((a, d), (lambda r, a=a, d=d: len(r) == len(a) - 1 and check_complex(a, d, r)), 'cert: Newton correction below 10^-(d-1) at every root of the square-free factor, multiplicities, count = degree (mpmath)'))
    return out


def g_quadratic_roots(rng, n):
    out = []
    while len(out) < n:
        a, b, c = rint(rng, -9, 9), rint(rng, -9, 9), rint(rng, -9, 9)
        if rng.random() < 0.3:
            a, b, c = Fraction(a, rng.choice([1, 2, 3])), Fraction(b, rng.choice([1, 2, 5])), Fraction(c, rng.choice([1, 3, 4]))
        if a == 0:
            continue

        def cert(r, a=a, b=b, c=c):
            if len(r) != 2:
                return False
            a_, b_, c_ = Fraction(a), Fraction(b), Fraction(c)
            D = b_ * b_ - 4 * a_ * c_
            for re, im in r:
                # exact: a z^2 + b z + c = 0 with z = re + i*im
                zr, zi = re, im
                real = a_ * (zr * zr - zi * zi) + b_ * zr + c_
                imag = (2 * a_ * zr + b_) * zi
                if not (real == 0) or not (imag == 0):
                    return False
            if D < 0:
                return r[0][1] < 0 < r[1][1] and r[0][0] == r[1][0]
            return r[0][0] <= r[1][0]
        out.append(((a, b, c), cert, 'cert: exact substitution of each root (Surd arithmetic) gives 0'))
    out.append(((0, 1, 1), Raises(BadInput), 'a = 0'))
    return out


def solve_cert(a, res):
    """Compare a solve_polynomial result with SymPy factor_list and numeric roots."""
    pe = sympy.Poly(expr(a), x, domain='QQ')
    lc, fl = sympy.factor_list(pe.as_expr(), x)
    want = Counter()
    for f, k in fl:
        fp = sympy.Poly(f, x, domain='QQ')
        want[(tuple(co(fp.monic())), k)] += 1
    got = Counter()
    for e in res['factors']:
        got[(tuple(co(P(e['poly']).monic())), e['multiplicity'])] += 1
    if want != got:
        return False
    tot = P([res['content']])
    for e in res['factors']:
        tot = tot * P(e['poly']) ** e['multiplicity']
    if tot != pe:
        return False
    for e in res['factors']:
        f = P(e['poly'])
        if e['degree'] >= 3:
            if e['roots'] is not None:
                return False
            if len([t for t in f.real_roots() if True]) != len(e['real_intervals']):
                return False
        else:
            if e['roots'] is None:
                return False
            ex = f.as_expr()
            for re, im in e['roots']:
                z = sy(Fraction(0)) + sympy.Float(str(re.to_decimal(40)), 40) + sympy.I * sympy.Float(str(im.to_decimal(40)), 40)
                if abs(sympy.N(ex.subs(x, z), 30)) > 1e-25:
                    return False
    return res['all_exact'] == all(e['degree'] <= 2 for e in res['factors'])


def g_solve_polynomial(rng, n):
    out = []
    while len(out) < n:
        a = rational_poly_with_roots(rng) if rng.random() < 0.6 else rpoly(rng, rng.randint(1, 6))
        out.append(((a,), (lambda r, a=a: solve_cert(a, r)), 'cert: factors equal SymPy factor_list, product reconstructs, roots of degree<=2 factors satisfy the factor (SymPy N 30 digits), degree>=3 factors carry intervals'))
    return out


def g_solve_polynomial_str(rng, n):
    out = []
    while len(out) < n:
        a = rational_poly_with_roots(rng) if rng.random() < 0.6 else rpoly(rng, rng.randint(1, 5))
        r = rng.random()
        if r < 0.4:
            s = poly_to_text(a, 'caret', rng) + ' = 0'
        elif r < 0.7:
            b = rpoly(rng, rng.randint(0, 2))
            lhs = co(P(a) + P(b))
            s = poly_to_text(lhs, 'caret', rng) + ' = ' + poly_to_text(b, 'caret', rng)
        else:
            s = poly_to_text(a, 'latex', rng) + ' = 0'
        out.append(((s,), (lambda res, a=a: solve_cert(a, res)), 'cert: as solve_polynomial, on the polynomial lhs - rhs'))
    out.append((('x = x',), Raises(BadInput), 'identity: every number is a solution'))
    return out


def sylvester_det(a, b):
    m, n = len(a) - 1, len(b) - 1
    S = sympy.zeros(m + n, m + n)
    for i in range(n):
        for j, c in enumerate(a):
            S[i, i + j] = sy(c)
    for i in range(m):
        for j, c in enumerate(b):
            S[n + i, i + j] = sy(c)
    return fr(S.det())


def g_resultant(rng, n):
    # The oracle is the determinant of the Sylvester matrix (the definition), built here. SymPy's own resultant() was tried first and returned the wrong
    # sign in 5 of the first 60 cases (e.g. Res(2x+7, 4x^3+6x^2-6) is -832 by the definition and by Res(f,g) = lc(f)^deg(g) * prod g(root of f); SymPy gives
    # 832 for both argument orders, which breaks Res(f,g) = (-1)^(mn) Res(g,f)).
    out = []
    while len(out) < n:
        a = rpoly(rng, rng.randint(1, 5), frac=rng.random() < 0.3)
        b = rpoly(rng, rng.randint(1, 5), frac=rng.random() < 0.3)
        out.append(((a, b), sylvester_det(a, b), 'determinant of the Sylvester matrix (SymPy Matrix.det)'))
    return out


def g_discriminant(rng, n):
    out = []
    while len(out) < n:
        a = rational_poly_with_roots(rng) if rng.random() < 0.3 else rpoly(rng, rng.randint(1, 6), frac=rng.random() < 0.3)
        out.append(((a,), fr(sympy.discriminant(expr(a), x)), 'SymPy discriminant'))
    return out


def g_vieta(rng, n):
    out = []
    while len(out) < n:
        roots, cf = rroots_poly(rng, rng.randint(1, 6))
        want = [sum((prod(c) for c in combinations(roots, k)), Fraction(0)) for k in range(1, len(roots) + 1)]
        out.append(((cf,), want, 'brute force: elementary symmetric sums of the chosen roots (itertools.combinations)'))
    return out


def g_power_sums(rng, n):
    out = []
    while len(out) < n:
        roots, cf = rroots_poly(rng, rng.randint(1, 5))
        k = rng.randint(1, 8)
        if rng.random() < 0.3 and all(r != 0 for r in roots):
            out.append(((cf, k, True), [sum((Fraction(1) / r ** j for r in roots), Fraction(0)) for j in range(1, k + 1)], 'brute force: sum of r^-j over the chosen roots'))
        else:
            out.append(((cf, k), [sum((r ** j for r in roots), Fraction(0)) for j in range(1, k + 1)], 'brute force: sum of r^j over the chosen roots'))
    return out


def g_roots_power_poly(rng, n):
    out = []
    while len(out) < n:
        roots, cf = rroots_poly(rng, rng.randint(1, 4), den=False)
        k = rng.randint(1, 4)
        want = co(sympy.Poly(sympy.prod([x - sy(r) ** k for r in roots]), x, domain='QQ'))
        out.append(((cf, k), want, 'SymPy product of (x - r^k) over the chosen roots'))
    return out


# --- interpolation, differences, cyclotomic

def g_interpolate(rng, n):
    out = []
    while len(out) < n:
        k = rng.randint(1, 8)
        xs = rng.sample(range(-8, 9), k)
        pts = [(Fraction(xv), Fraction(rint(rng, -9, 9), rng.choice([1, 2, 3]))) for xv in xs]
        want = co(sympy.Poly(sympy.interpolate([(sy(a), sy(b)) for a, b in pts], x), x, domain='QQ')) if k > 1 else ([pts[0][1]] if pts[0][1] != 0 else [])

        def cert(r, pts=pts, want=want):
            return len(r) <= len(pts) and all(P(r).eval(sy(a)) == sy(b) for a, b in pts) and same(r, want)
        out.append(((pts,), cert, 'cert: passes through every point, degree < k, equals SymPy interpolate'))
    out.append((([(1, 1), (1, 2)],), Raises(BadInput), 'two points share an x'))
    return out


def g_interpolate_eval(rng, n):
    out = []
    while len(out) < n:
        k = rng.randint(1, 7)
        xs = rng.sample(range(-8, 9), k)
        pts = [(Fraction(xv), Fraction(rint(rng, -9, 9), rng.choice([1, 2, 3]))) for xv in xs]
        at = Fraction(rint(rng, -12, 12), rng.choice([1, 2]))
        pol = sympy.interpolate([(sy(a), sy(b)) for a, b in pts], x) if k > 1 else sy(pts[0][1])
        out.append(((pts, at), fr(sympy.sympify(pol).subs(x, sy(at))), 'SymPy interpolate'))
    return out


def g_finite_differences(rng, n):
    out = []
    while len(out) < n:
        s = [Fraction(rint(rng, -9, 9), rng.choice([1, 2])) for _ in range(rng.randint(1, 9))]
        order = rng.choice([None, None, 1, 2, 3])
        rows = [list(s)]
        while len(rows[-1]) > 1 and (order is None or len(rows) <= order):
            r = rows[-1]
            rows.append([r[i + 1] - r[i] for i in range(len(r) - 1)])
        out.append(((s, order), rows, 'brute force differences in the sim (independent loop)'))
    return out


def g_cyclotomic(rng, n):
    out = []
    ns = list(range(1, 31)) + [36, 45, 60, 63, 64, 72, 84, 90, 105, 120, 210, 360, 420, 1155]
    for m in ns[:n] if n <= len(ns) else ns:
        out.append(((m,), co(sympy.Poly(sympy.cyclotomic_poly(m, x), x, domain='QQ')), 'SymPy cyclotomic_poly'))
    while len(out) < n:
        m = rng.randint(1, 400)
        out.append(((m,), co(sympy.Poly(sympy.cyclotomic_poly(m, x), x, domain='QQ')), 'SymPy cyclotomic_poly'))
    return out


# --- factoring

SD = [[1, 0, -10, 0, 1], [1, 0, -40, 0, 352, 0, -960, 0, 576], [1, 0, 0, 0, 1], [1, 0, 0, 0, 4], [1, 0, 0, 0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
      [1, -1, 1, -1, 1, -1, 1, -1, 1], [1, 0, 0, -1], [1, 0, 0, 0, 0, 0, -1], [1, 0, -2], [1, 0, 0, 0, -16], [1, 0, 0, 0, 0, -32], [3, 0, -2, 0, 5],
      [1, 0, -4, 0, 4], [1, 4, 6, 4, 5], [1, 0, 1, 0, 1], [1, 0, 0, -3, 0, 0, 0, 0, 1, 0]]


def g_factor_z(rng, n):
    out = []
    for c in SD:
        out.append(((c,), 'sympy', 'SymPy factor_list'))
    while len(out) < n:
        r = rng.random()
        if r < 0.35:
            parts = [rpoly(rng, rng.randint(1, 3)) for _ in range(rng.randint(2, 4))]
            a = co(sympy.Poly(sympy.prod([expr(p_) for p_ in parts]), x, domain='QQ'))
        elif r < 0.55:
            f = rpoly(rng, rng.randint(1, 3))
            a = co(sympy.Poly(expr(f) ** rng.randint(2, 3) * expr(rpoly(rng, rng.randint(1, 3))), x, domain='QQ'))
        elif r < 0.75:
            m = rng.randint(2, 60)
            a = co(sympy.Poly(sympy.cyclotomic_poly(m, x) * sympy.cyclotomic_poly(rng.randint(1, 30), x), x, domain='QQ'))
        elif r < 0.9:
            a = rpoly(rng, rng.randint(4, 14), lo=-30, hi=30)
        else:
            a = co(sympy.Poly(expr(rpoly(rng, rng.randint(6, 12), lo=-5, hi=5)) * expr(rpoly(rng, rng.randint(6, 12), lo=-5, hi=5)), x, domain='QQ'))
        out.append(((a,), 'sympy', 'SymPy factor_list'))
    final = []
    for args, _, name in out[:n]:
        a = args[0]
        pe = P(a).as_expr()
        lc, fl = sympy.factor_list(pe, x)
        want = Counter()
        for f, k in fl:
            fp = sympy.Poly(f, x, domain='ZZ' if all(sympy.Rational(c).q == 1 for c in sympy.Poly(f, x).all_coeffs()) else 'QQ')
            ints = [int(c) for c in fp.all_coeffs()]
            g_ = 0
            for v in ints:
                g_ = gcd(g_, v)
            ints = [v // g_ for v in ints]
            if ints[0] < 0:
                ints = [-v for v in ints]
            want[(tuple(ints), k)] += 1

        def cert(res, a=a, want=want):
            got = Counter((tuple(int(c) for c in f), m) for f, m in res['factors'])
            tot = P([res['content']])
            for f, m in res['factors']:
                tot = tot * P(f) ** m
            return got == want and tot == P(a) and all(f[0] > 0 for f, _ in res['factors'])
        final.append(((a,), cert, 'cert: factors multiply back to p, and the (factor, multiplicity) multiset equals SymPy factor_list'))
    return final


def g_is_irreducible_z(rng, n):
    out = []
    pool = [c for c in SD] + [[1, 0, 1], [2, 1], [1, 1, 1], [1, 0, -1], [4], [1, 2, 1]]
    while len(out) < n:
        a = rng.choice(pool) if rng.random() < 0.5 else rpoly(rng, rng.randint(1, 8), lo=-12, hi=12)
        pe = P(a)
        want = bool(pe.is_irreducible) and pe.degree() >= 1
        out.append(((a,), want, 'SymPy Poly.is_irreducible'))
    return out


# --- polynomials mod p

PRIMES_SMALL = [2, 3, 5, 7, 11, 13, 31, 101]
PRIMES_ANY = PRIMES_SMALL + [257, 65537, 1000000007, 2305843009213693951]


def gf(cf, p):
    return [int(c) % p for c in cf]


def mod_poly(rng, p, deg):
    cf = [rng.randrange(p) for _ in range(deg + 1)]
    cf[0] = cf[0] or 1
    return cf


def gtrim(l):
    i = 0
    while i < len(l) and l[i] == 0:
        i += 1
    return l[i:]


def g_polymod_mul(rng, n):
    out = []
    while len(out) < n:
        p = rng.choice(PRIMES_ANY)
        a, b = mod_poly(rng, p, rng.randint(0, 6)), mod_poly(rng, p, rng.randint(0, 6))
        out.append(((a, b, p), gtrim([int(v) for v in gt.gf_mul(a, b, p, ZZ)]), 'SymPy galoistools gf_mul'))
    return out


def g_polymod_divmod(rng, n):
    out = []
    while len(out) < n:
        p = rng.choice(PRIMES_ANY)
        a, b = mod_poly(rng, p, rng.randint(0, 8)), mod_poly(rng, p, rng.randint(0, 5))
        q, r = gt.gf_div(a, b, p, ZZ)
        out.append(((a, b, p), (gtrim([int(v) for v in q]), gtrim([int(v) for v in r])), 'SymPy galoistools gf_div'))
    return out


def g_polymod_gcd(rng, n):
    out = []
    while len(out) < n:
        p = rng.choice(PRIMES_ANY)
        f = mod_poly(rng, p, rng.randint(0, 3))
        a = gt.gf_mul(f, mod_poly(rng, p, rng.randint(0, 3)), p, ZZ)
        b = gt.gf_mul(f, mod_poly(rng, p, rng.randint(0, 3)), p, ZZ)
        g = gt.gf_gcd(a, b, p, ZZ)
        out.append(((a, b, p), gtrim([int(v) for v in gt.gf_monic(g, p, ZZ)[1]]) if g else [], 'SymPy galoistools gf_gcd (monic)'))
    return out


def g_polymod_powmod(rng, n):
    out = []
    while len(out) < n:
        p = rng.choice(PRIMES_ANY)
        f = mod_poly(rng, p, rng.randint(1, 6))
        a = mod_poly(rng, p, rng.randint(0, 6))
        e = rng.choice([0, 1, 2, 5, 10, 100, 12345, 10 ** 9 + 7, 10 ** 18, 2 ** 100 + 12345])
        out.append(((a, e, f, p), gtrim([int(v) for v in gt.gf_pow_mod(a, e, f, p, ZZ)]), 'SymPy galoistools gf_pow_mod'))
    return out


def g_polymod_inverse(rng, n):
    out = []
    while len(out) < n:
        p = rng.choice(PRIMES_SMALL + [257, 65537, 1000000007])
        f = mod_poly(rng, p, rng.randint(1, 6))
        a = mod_poly(rng, p, rng.randint(0, 6))
        coprime = len(gt.gf_gcd(a, f, p, ZZ)) == 1
        if not coprime:
            out.append(((a, f, p), Raises(BadInput), 'gcd(a, f) != 1: SymPy gf_gcd'))
            continue

        def cert(r, a=a, f=f, p=p):
            prod_ = gt.gf_rem(gt.gf_mul(a, r, p, ZZ), f, p, ZZ)
            return len(r) <= len(f) - 1 and gtrim([int(v) for v in prod_]) == [1]
        out.append(((a, f, p), cert, 'cert: a * inverse = 1 mod f (SymPy gf_mul, gf_rem)'))
    return out


def g_polymod_roots(rng, n):
    out = []
    while len(out) < n:
        p = rng.choice(PRIMES_SMALL)
        if rng.random() < 0.5:
            roots = [rng.randrange(p) for _ in range(rng.randint(1, 4))]
            f = [1]
            for r in roots:
                f = gt.gf_mul(f, [1, (-r) % p], p, ZZ)
            if rng.random() < 0.5:
                f = gt.gf_mul(f, mod_poly(rng, p, 2), p, ZZ)
        else:
            f = mod_poly(rng, p, rng.randint(1, 6))
        want = []
        for r in range(p):
            g_, m = list(f), 0
            while True:
                q, rem = gt.gf_div(g_, [1, (-r) % p], p, ZZ)
                if rem:
                    break
                g_, m = q, m + 1
            if m:
                want.append((r, m))
        out.append(((f, p), want, 'brute force: every residue tried and divided out'))
    return out


def g_polymod_factor(rng, n):
    out = []
    while len(out) < n:
        p = rng.choice(PRIMES_SMALL + [257])
        f = [1]
        for _ in range(rng.randint(1, 4)):
            f = gt.gf_mul(f, mod_poly(rng, p, rng.randint(1, 4)), p, ZZ)
        if rng.random() < 0.3:
            g_ = mod_poly(rng, p, rng.randint(1, 2))
            f = gt.gf_mul(f, gt.gf_mul(g_, g_, p, ZZ), p, ZZ)
        lc, facs = gt.gf_factor(f, p, ZZ)
        want = sorted((tuple(int(v) for v in fa), int(m)) for fa, m in facs)

        def cert(r, f=f, p=p, lc=lc, want=want):
            l, fs = r
            tot = [l]
            for fa, m in fs:
                for _ in range(m):
                    tot = gt.gf_mul(tot, fa, p, ZZ)
            return int(l) == int(lc) and gtrim([int(v) for v in tot]) == gtrim(list(f)) and sorted((tuple(fa), m) for fa, m in fs) == want
        out.append(((f, p), cert, 'cert: lead * product = f mod p, and (factor, multiplicity) set equals SymPy gf_factor'))
    return out


def g_polymod_irreducible(rng, n):
    out = []
    while len(out) < n:
        p = rng.choice(PRIMES_SMALL + [257])
        f = mod_poly(rng, p, rng.randint(1, 8))
        out.append(((f, p), bool(gt.gf_irreducible_p(f, p, ZZ)), 'SymPy galoistools gf_irreducible_p'))
    return out


# --- matrices

def rmat(rng, m, k, lo=-5, hi=5, frac=False):
    return [[Fraction(rint(rng, lo, hi), rng.choice([1, 2, 3]) if frac else 1) for _ in range(k)] for _ in range(m)]


def sm(M):
    return sympy.Matrix([[sy(c) for c in r] for r in M])


def fm(M):
    return [[fr(c) for c in M.row(i)] for i in range(M.rows)]


def rank_deficient(rng, m, k):
    r = rng.randint(1, max(1, min(m, k) - 1)) if min(m, k) > 1 else 1
    return fm(sm(rmat(rng, m, r)) * sm(rmat(rng, r, k)))


def rand_matrix_any(rng, square=False, maxn=5, frac=False):
    m = rng.randint(1, maxn)
    k = m if square else rng.randint(1, maxn)
    if rng.random() < 0.3 and min(m, k) > 1:
        return rank_deficient(rng, m, k)
    return rmat(rng, m, k, frac=frac)


def g_mat_mul(rng, n):
    out = []
    while len(out) < n:
        m, k, l = rng.randint(1, 5), rng.randint(1, 5), rng.randint(1, 5)
        a, b = rmat(rng, m, k, frac=rng.random() < 0.3), rmat(rng, k, l)
        out.append(((a, b), fm(sm(a) * sm(b)), 'SymPy Matrix product'))
    return out


def g_mat_trace(rng, n):
    out = []
    while len(out) < n:
        a = rand_matrix_any(rng, True, 7, frac=rng.random() < 0.3)
        out.append(((a,), fr(sm(a).trace()), 'SymPy Matrix.trace'))
    return out


def g_mat_det(rng, n):
    out = []
    while len(out) < n:
        a = rand_matrix_any(rng, True, 7, frac=rng.random() < 0.3)
        out.append(((a,), fr(sm(a).det()), 'SymPy Matrix.det'))
    out.append((([[1, 2, 3], [4, 5, 6], [7, 8, 9]],), Fraction(0), 'hand: singular'))
    return out


def g_mat_inverse(rng, n):
    out = []
    while len(out) < n:
        a = rand_matrix_any(rng, True, 6, frac=rng.random() < 0.3)
        M = sm(a)
        if M.det() == 0:
            out.append(((a,), Raises(BadInput), 'singular: SymPy det = 0'))
            continue

        def cert(r, a=a):
            return fm(sm(a) * sm(r)) == fm(sympy.eye(len(a)))
        out.append(((a,), cert, 'cert: A * inverse = I (SymPy product)'))
    return out


def g_mat_rank(rng, n):
    out = []
    while len(out) < n:
        a = rand_matrix_any(rng, False, 6)
        out.append(((a,), sm(a).rank(), 'SymPy Matrix.rank'))
    return out


def g_mat_rref(rng, n):
    out = []
    while len(out) < n:
        a = rand_matrix_any(rng, False, 6, frac=rng.random() < 0.3)
        R, piv = sm(a).rref()
        out.append(((a,), (fm(R), list(piv)), 'SymPy Matrix.rref'))
    return out


def g_mat_nullspace(rng, n):
    out = []
    while len(out) < n:
        a = rand_matrix_any(rng, False, 6)
        M = sm(a)

        def cert(r, a=a, M=M):
            if len(r) != M.cols - M.rank():
                return False
            if any(M * sympy.Matrix([sy(c) for c in v]) != sympy.zeros(M.rows, 1) for v in r):
                return False
            return not r or sympy.Matrix([[sy(c) for c in v] for v in r]).rank() == len(r)
        out.append(((a,), cert, 'cert: A v = 0 for each, independent, count = columns - rank (SymPy)'))
    return out


def g_mat_solve(rng, n):
    out = []
    while len(out) < n:
        m, k = rng.randint(1, 5), rng.randint(1, 5)
        a = rank_deficient(rng, m, k) if (rng.random() < 0.35 and min(m, k) > 1) else rmat(rng, m, k, frac=rng.random() < 0.3)
        if rng.random() < 0.65:
            x0 = [Fraction(rint(rng, -5, 5)) for _ in range(k)]
            b = [fr(c) for c in (sm(a) * sympy.Matrix([sy(v) for v in x0]))]
        else:
            b = [Fraction(rint(rng, -5, 5)) for _ in range(m)]
        M = sm(a)
        rk = M.rank()
        rk2 = M.row_join(sympy.Matrix([sy(v) for v in b])).rank()
        status = 'none' if rk2 > rk else ('unique' if rk == k else 'infinite')

        def cert(r, M=M, b=b, rk=rk, k=k, status=status):
            if r['status'] != status:
                return False
            if status == 'none':
                return r['particular'] is None
            x0 = sympy.Matrix([sy(v) for v in r['particular']])
            if M * x0 != sympy.Matrix([sy(v) for v in b]):
                return False
            ns = r['nullspace']
            return len(ns) == k - rk and all(M * sympy.Matrix([sy(v) for v in vv]) == sympy.zeros(M.rows, 1) for vv in ns)
        out.append(((a, b), cert, 'cert: A x0 = b, nullspace vectors satisfy A v = 0, count = n - rank, status from SymPy ranks'))
    return out


def g_mat_charpoly(rng, n):
    out = []
    while len(out) < n:
        a = rand_matrix_any(rng, True, 6, frac=rng.random() < 0.3)
        M = sm(a)
        out.append(((a,), co(sympy.Poly(M.charpoly(x).as_expr(), x, domain='QQ')), 'SymPy Matrix.charpoly'))
    return out


def g_mat_minpoly(rng, n):
    out = []
    while len(out) < n:
        k = rng.randint(1, 5)
        if rng.random() < 0.5:                       # block structure with repeated eigenvalues: minimal polynomial < characteristic polynomial
            d = [rng.choice([1, 2, 3]) for _ in range(k)]
            a = [[Fraction(d[i]) if i == j else Fraction(0) for j in range(k)] for i in range(k)]
            if k > 1 and rng.random() < 0.5:
                a[0][1] = Fraction(1)
        else:
            a = rmat(rng, k, k)
        M = sm(a)

        def cert(r, a=a, M=M):
            d = len(r) - 1
            if r[0] != 1:
                return False
            tot = sympy.zeros(M.rows, M.cols)
            for i, c in enumerate(r):
                tot += sy(c) * M ** (d - i)
            if tot != sympy.zeros(M.rows, M.cols):
                return False
            vecs = sympy.Matrix([list(M ** j) for j in range(d)]) if d else sympy.zeros(0, 1)
            return d == 0 or vecs.rank() == d       # I, A, ..., A^(d-1) independent: no polynomial of lower degree annihilates A
        out.append(((a,), cert, 'cert: p(A) = 0 and I..A^(d-1) are linearly independent (SymPy rank), so d is minimal'))
    return out


def g_mat_eigen(rng, n):
    out = []
    while len(out) < n:
        k = rng.randint(1, 4)
        if rng.random() < 0.6:
            T = sm(rmat(rng, k, k, -2, 2))
            if T.det() == 0:
                continue
            D = sympy.diag(*[rng.choice([-1, 1, 2, 3]) for _ in range(k)])
            if k > 1 and rng.random() < 0.4:
                D[0, 1] = 1
            a = fm(T * D * T.inv())
        else:
            a = rmat(rng, k, k)
        M = sm(a)
        cpoly = sympy.Poly(M.charpoly(x).as_expr(), x, domain='QQ')
        ev = {fr(r_): int(m_) for r_, m_ in cpoly.ground_roots().items()}      # rational eigenvalues: linear factors of the characteristic polynomial (SymPy)

        def cert(r, a=a, M=M, ev=ev):
            got = {e['value']: e['algebraic'] for e in r['eigenvalues']}
            if got != ev:
                return False
            cp = sympy.Poly(M.charpoly(x).as_expr(), x, domain='QQ')
            tot = P(r['rest'])
            for e in r['eigenvalues']:
                tot = tot * sympy.Poly(x - sy(e['value']), x, domain='QQ') ** e['algebraic']
                v = e['eigenvectors']
                if len(v) != e['geometric'] or e['geometric'] != (M - sy(e['value']) * sympy.eye(M.rows)).nullspace().__len__():
                    return False
                if any(M * sympy.Matrix([sy(c) for c in vv]) != sy(e['value']) * sympy.Matrix([sy(c) for c in vv]) for vv in v):
                    return False
            diag = len(r['rest']) == 1 and all(e['algebraic'] == (M - sy(e['value']) * sympy.eye(M.rows)).nullspace().__len__() for e in r['eigenvalues'])
            return tot == cp and r['diagonalizable_over_Q'] == diag
        out.append(((a,), cert, 'cert: rational eigenvalues = SymPy eigenvals (rational part), A v = lambda v, geometric = SymPy nullity, rest * linear factors = charpoly'))
    return out


def g_mat_pow(rng, n):
    out = []
    while len(out) < n:
        k = rng.randint(1, 4)
        r = rng.random()
        a = rmat(rng, k, k, -3, 3, frac=rng.random() < 0.2)
        if r < 0.6:
            e = rng.randint(0, 25)
            out.append(((a, e), fm(sm(a) ** e), 'SymPy Matrix power'))
        elif r < 0.75:
            M = sm(a)
            if M.det() == 0:
                continue
            e = -rng.randint(1, 8)
            out.append(((a, e), fm(M ** e), 'SymPy Matrix power (negative: inverse)'))
        else:
            ai = [[Fraction(int(c)) for c in rw] for rw in rmat(rng, k, k, -3, 3)]
            m = rng.choice([7, 10, 1000000007, 998244353, 2 ** 31 - 1])
            e = rng.choice([10 ** 9, 10 ** 18, 123456789123456789, 5, 100])
            # brute reference: plain Python integer matrix multiplication (independent loop), not the backend's code
            def mm(X, Y):
                return [[sum(X[i][t] * Y[t][j] for t in range(k)) % m for j in range(k)] for i in range(k)]
            R = [[int(i == j) for j in range(k)] for i in range(k)]
            B = [[int(c) % m for c in rw] for rw in ai]
            ee = e
            while ee:
                if ee & 1:
                    R = mm(R, B)
                B = mm(B, B)
                ee >>= 1
            out.append(((ai, e, m), [[Fraction(c) for c in rw] for rw in R], 'brute: square-and-multiply on Python ints in the sim'))
    return out


def g_mat_lu(rng, n):
    out = []
    while len(out) < n:
        m, k = rng.randint(1, 5), rng.randint(1, 5)
        a = rank_deficient(rng, m, k) if (rng.random() < 0.3 and min(m, k) > 1) else rmat(rng, m, k, 0, 5)
        if rng.random() < 0.3:
            a = rmat(rng, m, k, -3, 3)

        def cert(r, a=a):
            Pm, L, U = sm(r['P']), sm(r['L']), sm(r['U'])
            if Pm * sm(a) != L * U:
                return False
            if any(L[i, j] != 0 for i in range(L.rows) for j in range(i + 1, L.cols)) or any(L[i, i] != 1 for i in range(L.rows)):
                return False
            piv = -1
            for i in range(U.rows):                 # echelon: first nonzero strictly to the right of the previous row's
                row = [j for j in range(U.cols) if U[i, j] != 0]
                if not row:
                    piv = U.cols
                    continue
                if row[0] <= piv and piv < U.cols:
                    return False
                if piv == U.cols:
                    return False
                piv = row[0]
            return sorted(sum(1 for v in rw if v != 0) for rw in sm(r['P']).tolist()) == [1] * len(a)
        out.append(((a,), cert, 'cert: P A = L U (SymPy), L unit lower triangular, U in echelon form, P a permutation'))
    return out


# --- linear systems in text

def g_solve_linear_str(rng, n):
    out = []
    names = ['x', 'y', 'z', 'w']
    while len(out) < n:
        k = rng.randint(1, 4)
        m = rng.randint(1, 5)
        vs = names[:k]
        a = rmat(rng, m, k, -6, 6)
        if rng.random() < 0.65:
            x0 = [Fraction(rint(rng, -5, 5)) for _ in range(k)]
            b = [sum(a[i][j] * x0[j] for j in range(k)) for i in range(m)]
        else:
            b = [Fraction(rint(rng, -9, 9)) for _ in range(m)]
        eqs = []
        for i in range(m):
            lhs = ' '.join(('+' if a[i][j] >= 0 else '-') + f' {abs(a[i][j])}{vs[j]}' for j in range(k) if a[i][j] != 0) or f'0*{vs[0]}'
            lhs = lhs.lstrip('+ ')
            style = rng.random()
            if style < 0.25:
                eqs.append(f'{lhs} = {b[i]}')
            elif style < 0.5:
                eqs.append(f'{b[i]} = {lhs}')
            elif style < 0.75:
                eqs.append(f'2({lhs}) = {2 * b[i]}')
            else:
                eqs.append(f'{lhs} - {b[i]} = 0')
        M = sm(a)
        bb = sympy.Matrix([sy(v) for v in b])
        rk = M.rank()
        rk2 = M.row_join(bb).rank()
        # variables that appear in at least one equation
        used = [j for j in range(k) if any(a[i][j] != 0 for i in range(m))] or [0]
        status = 'none' if rk2 > rk else ('unique' if rk == len(used) else 'infinite')

        def cert(r, a=a, b=b, vs=vs, status=status, used=used, rk=rk):
            if r['status'] != status:
                return False
            if status == 'none':
                return r['solution'] is None
            sol = r['solution']
            if sorted(sol) != sorted(r['variables']):
                return False
            for i in range(len(a)):
                if sum(a[i][j] * sol.get(vs[j], 0) for j in range(len(vs))) != b[i]:
                    return False
            return len(r['nullspace']) == len(r['variables']) - rk
        out.append(((eqs,), cert, 'cert: the solution satisfies every equation exactly; status and nullity from SymPy ranks'))
    out.append(((['x + y = 3', 'x - y = 1'],), {'variables': ['x', 'y'], 'status': 'unique', 'solution': {'x': Fraction(2), 'y': Fraction(1)}, 'free': [], 'nullspace': []}, 'hand-solved'))
    out.append(((['x*y = 3'],), Raises(NotCovered), 'not linear'))
    return out


# --- recurrences

def rrec(rng, k=None):
    k = k or rng.randint(1, 4)
    c = [Fraction(rint(rng, -3, 3), rng.choice([1, 1, 2])) for _ in range(k)]
    if c[-1] == 0:
        c[-1] = Fraction(1)
    a0 = [Fraction(rint(rng, -4, 4)) for _ in range(k)]
    return c, a0


def iterate(c, a0, n, const=0):
    s = list(a0)
    while len(s) <= n:
        m = len(s)
        s.append(sum((c[i] * s[m - 1 - i] for i in range(len(c))), Fraction(0)) + const)
    return s


def g_recurrence_char_poly(rng, n):
    out = []
    while len(out) < n:
        c, _ = rrec(rng)
        out.append(((c,), co(sympy.Poly(x ** len(c) - sum(sy(v) * x ** (len(c) - 1 - i) for i, v in enumerate(c)), x, domain='QQ')), 'SymPy Poly of x^k - sum c_i x^(k-i)'))
    return out


def g_recurrence_nth(rng, n):
    out = []
    while len(out) < n:
        c, a0 = rrec(rng)
        r = rng.random()
        if r < 0.5:
            m = rng.randint(0, 200)
            const = Fraction(rint(rng, -3, 3)) if rng.random() < 0.3 else 0
            out.append(((c, a0, m, None, const), iterate(c, a0, m, const)[m], 'brute: iterate the recurrence term by term (Fractions)'))
        else:
            ci = [Fraction(int(v)) for v in c]
            ci = [v if v.denominator == 1 else Fraction(1) for v in ci]
            m = rng.choice([10, 1000, 10 ** 6, 10 ** 12, 10 ** 18, 123456789012345678901234567890])
            mod = rng.choice([7, 10, 1000000007, 998244353, 10 ** 18 + 9])
            k = len(ci)
            # brute reference: companion-matrix power with Python ints (a different algorithm from Kitamasa)
            comp = [[int(ci[j]) % mod if i == 0 else int(j == i - 1) for j in range(k)] for i in range(k)]

            def mm(X, Y):
                return [[sum(X[i][t] * Y[t][j] for t in range(k)) % mod for j in range(k)] for i in range(k)]
            R = [[int(i == j) for j in range(k)] for i in range(k)]
            B = comp
            e = m
            while e:
                if e & 1:
                    R = mm(R, B)
                B = mm(B, B)
                e >>= 1
            # a_m = (R applied to the state vector (a_(k-1), ..., a_0)) first component, for m >= k-1
            state = [int(a0[k - 1 - i]) % mod for i in range(k)]
            val = sum(R[0][j] * state[j] for j in range(k)) % mod
            # R = comp^m maps the state at index k-1 to the state at index k-1+m, so val is a_(k-1+m); ask for index k-1+m
            out.append(((ci, [Fraction(int(v)) for v in a0], k - 1 + m, mod, 0), val, 'brute: companion-matrix power on Python ints in the sim'))
    return out


def g_recurrence_closed_form(rng, n):
    out = []
    fixed = [([1, 1], [0, 1]), ([2, -1], [1, 2]), ([3, -2], [1, 5]), ([4, -4], [1, 2]), ([0, 4], [1, 3]), ([6, -11, 6], [1, 2, 3]), ([5, -6], [2, 7]), ([1, 1, 0], [1, 1, 2])]
    for c, a0 in fixed:
        out.append(((c, a0), None, ''))
    while len(out) < n:
        r = rng.random()
        if r < 0.5:                                    # rational roots from chosen roots
            roots = [Fraction(rint(rng, -3, 3)) for _ in range(rng.randint(1, 3))]
            poly = [Fraction(1)]
            for rt in roots:
                nxt = poly + [Fraction(0)]
                for i, cf_ in enumerate(poly):
                    nxt[i + 1] -= rt * cf_
                poly = nxt
            c = [-v for v in poly[1:]]
        else:
            c, _ = rrec(rng, rng.randint(1, 3))
        a0 = [Fraction(rint(rng, -4, 4)) for _ in range(len(c))]
        out.append(((c, a0), None, ''))
    final = []
    for args, _, _ in out[:n]:
        c, a0 = args
        c = [Fraction(v) for v in c]
        a0 = [Fraction(v) for v in a0]

        def cert(r, c=c, a0=a0):
            seq = iterate(c, a0, 40)
            for m in range(r['valid_from'], 40):
                if A.closed_form_value(r, m) != seq[m]:
                    return False
            return True
        q = x ** len(c) - sum(sy(v) * x ** (len(c) - 1 - i) for i, v in enumerate(c))
        fl = sympy.factor_list(q, x)[1]
        in_scope = sum(1 for f_, k in fl if sympy.degree(f_, x) >= 2) <= 1 and all(
            sympy.degree(f_, x) == 1 or (sympy.degree(f_, x) == 2 and k == 1 and sympy.discriminant(f_, x) > 0) for f_, k in fl)
        # in scope (rational roots, or one real quadratic pair): a refusal is a failure; out of scope: a refusal is the expected answer, an answer must still verify
        final.append(((c, a0), Answers(cert) if in_scope else MayRefuse(cert),
                      'cert: the closed form reproduces the iterated sequence exactly for n = valid_from..39' + ('' if not in_scope else ' (in scope: refusal counts as failure)')))
    return final


def g_recurrence_guess(rng, n):
    out = []
    while len(out) < n:
        c, a0 = rrec(rng, rng.randint(1, 4))
        k = len(c)
        N = rng.randint(2 * k, 2 * k + 6)
        s = iterate(c, a0, N - 1)

        def cert(r, s=s):
            L, cf = r['order'], r['coeffs']
            if len(cf) != L or 2 * L + 1 > len(s):
                return False
            for m in range(L, len(s)):
                if sum((cf[i] * s[m - 1 - i] for i in range(L)), Fraction(0)) != s[m]:
                    return False
            H = sympy.Matrix(len(s) // 2, len(s) - len(s) // 2, lambda i, j: sy(s[i + j]))   # Hankel matrix of the terms
            return H.rank() == L or L <= H.rank()
        out.append(((s,), Answers(cert) if N >= 2 * k + 1 else MayRefuse(cert),
                    'cert: the recurrence regenerates every term, order not above the Hankel rank (SymPy); with N = 2*order the bound (2*order + 1 terms) may refuse'))
    out.append((([1, 2, 4, 8, 16, 32, 64],), (lambda r: r['order'] == 1 and r['coeffs'] == [2]), 'hand: powers of two'))
    return out


GENS = {
    'poly_from_str': g_poly_from_str, 'poly_to_str': g_poly_to_str, 'poly_add': g_poly_add, 'poly_sub': g_poly_sub, 'poly_mul': g_poly_mul, 'poly_pow': g_poly_pow,
    'poly_divmod': g_poly_divmod, 'poly_gcd': g_poly_gcd, 'poly_lcm': g_poly_lcm, 'poly_gcdex': g_poly_gcdex, 'poly_compose': g_poly_compose,
    'poly_derivative': g_poly_derivative, 'poly_antiderivative': g_poly_antiderivative, 'poly_integral': g_poly_integral, 'poly_eval': g_poly_eval,
    'poly_taylor_shift': g_poly_taylor_shift, 'poly_content_primitive': g_poly_content_primitive, 'poly_squarefree_part': g_poly_squarefree_part,
    'poly_squarefree_decomposition': g_poly_squarefree_decomposition, 'poly_from_roots': g_poly_from_roots, 'poly_transform_roots': g_poly_transform_roots,
    'rational_roots': g_rational_roots, 'sturm_sequence': g_sturm_sequence, 'count_real_roots': g_count_real_roots, 'isolate_real_roots': g_isolate_real_roots,
    'real_roots': g_real_roots, 'complex_roots': g_complex_roots, 'quadratic_roots': g_quadratic_roots, 'solve_polynomial': g_solve_polynomial,
    'solve_polynomial_str': g_solve_polynomial_str, 'factor_z': g_factor_z, 'is_irreducible_z': g_is_irreducible_z, 'resultant': g_resultant,
    'discriminant': g_discriminant, 'vieta': g_vieta, 'power_sums': g_power_sums, 'roots_power_poly': g_roots_power_poly, 'interpolate': g_interpolate,
    'interpolate_eval': g_interpolate_eval, 'finite_differences': g_finite_differences, 'cyclotomic': g_cyclotomic,
    'polymod_mul': g_polymod_mul, 'polymod_divmod': g_polymod_divmod, 'polymod_gcd': g_polymod_gcd, 'polymod_powmod': g_polymod_powmod,
    'polymod_inverse': g_polymod_inverse, 'polymod_roots': g_polymod_roots, 'polymod_factor': g_polymod_factor, 'polymod_irreducible': g_polymod_irreducible,
    'mat_mul': g_mat_mul, 'mat_trace': g_mat_trace, 'mat_det': g_mat_det, 'mat_inverse': g_mat_inverse, 'mat_rank': g_mat_rank, 'mat_rref': g_mat_rref,
    'mat_nullspace': g_mat_nullspace, 'mat_solve': g_mat_solve, 'mat_charpoly': g_mat_charpoly, 'mat_minpoly': g_mat_minpoly, 'mat_eigen': g_mat_eigen,
    'mat_pow': g_mat_pow, 'mat_lu': g_mat_lu, 'solve_linear_str': g_solve_linear_str,
    'recurrence_char_poly': g_recurrence_char_poly, 'recurrence_nth': g_recurrence_nth, 'recurrence_closed_form': g_recurrence_closed_form,
    'recurrence_guess': g_recurrence_guess,
}


def generate(seed, n, only=None):
    """All cases, n per operation, from a fixed seed (only: a list of operation names without the module prefix)."""
    cases = []
    for name, gen in GENS.items():
        if only and name not in only:
            continue
        rng = random.Random(f'{seed}:{name}')
        for args, oracle, oname in gen(rng, n):
            cases.append(Case('algebra.' + name, args, oracle, oname))
    return cases


def check(case, result):
    o = case.oracle.check if isinstance(case.oracle, (Answers, MayRefuse)) else case.oracle
    if callable(o):
        return bool(o(result))
    return same(result, o)


def run(seed=1, n=60, only=None, write=None):
    """The report dict ({'ops': {op: {generated, answered, agree, refused, refusals, expected_refusals_ok, disagree, seconds}}, ...}); written to
    build-mathbackend/algebra.json at the repo root unless `only` is given or write=False."""
    t0 = time.time()
    cases = generate(seed, n, only)
    per = {}
    bad = []
    for c in cases:
        st = per.setdefault(c.op, dict(generated=0, answered=0, agree=0, refused=0, refusals={}, expected_refusals_ok=0, disagree=0, seconds=0.0))
        st['generated'] += 1
        t1 = time.time()
        try:
            res = backend_run(c.op, *c.args)
        except NotCovered as e:
            if isinstance(c.oracle, Raises) and c.oracle.exc is NotCovered:
                st['expected_refusals_ok'] += 1
            elif isinstance(c.oracle, MayRefuse):
                st['expected_refusals_ok'] += 1
            elif isinstance(c.oracle, Answers):
                st['disagree'] += 1
                bad.append(dict(op=c.op, args=repr(c.args)[:300], got=f'NotCovered: {e} (the case is inside the stated scope)', oracle=c.oracle_name))
            else:
                st['refused'] += 1
                key = str(e)[:90]
                st['refusals'][key] = st['refusals'].get(key, 0) + 1
        except BadInput as e:
            st['answered'] += 1
            if isinstance(c.oracle, Raises) and c.oracle.exc is BadInput:
                st['agree'] += 1
            else:
                st['disagree'] += 1
                bad.append(dict(op=c.op, args=repr(c.args)[:300], got=f'BadInput: {e}', oracle=c.oracle_name))
        except Exception as e:                 # a crash is a disagreement
            st['answered'] += 1
            st['disagree'] += 1
            bad.append(dict(op=c.op, args=repr(c.args)[:300], got=f'{type(e).__name__}: {e}', oracle=c.oracle_name))
        else:
            st['answered'] += 1
            if isinstance(c.oracle, Raises):
                st['disagree'] += 1
                bad.append(dict(op=c.op, args=repr(c.args)[:300], got=repr(res)[:200], oracle=c.oracle_name + ' (expected a refusal)'))
            else:
                try:
                    ok = check(c, res)
                except Exception as e:
                    ok = False
                    res = f'oracle check crashed: {type(e).__name__}: {e}; result {res!r}'
                if ok:
                    st['agree'] += 1
                else:
                    st['disagree'] += 1
                    bad.append(dict(op=c.op, args=repr(c.args)[:300], got=repr(res)[:300], oracle=c.oracle_name))
        st['seconds'] += time.time() - t1
    registered = sorted(k for k in registry() if k.startswith('algebra.'))
    report = dict(module='algebra', seed=seed, n_per_operation=n, seconds=round(time.time() - t0, 1), registered=len(registered),
                  ops=per, no_case=[k for k in registered if k not in per], disagreements=bad[:50],
                  totals=dict(generated=sum(s['generated'] for s in per.values()), answered=sum(s['answered'] for s in per.values()),
                              agree=sum(s['agree'] for s in per.values()), refused=sum(s['refused'] for s in per.values()),
                              disagree=sum(s['disagree'] for s in per.values())),
                  not_covered=A.NOT_COVERED)
    if write or (write is None and not only):
        out = Path(__file__).resolve().parents[3] / 'build-mathbackend'
        out.mkdir(exist_ok=True)
        (out / 'algebra.json').write_text(json.dumps(report, indent=1, default=str), encoding='utf-8')
    return report


def table(rep):
    lines = [f"{'operation':36s} {'gen':>4s} {'ans':>4s} {'agree':>5s} {'refused':>7s} {'DISAGREE':>8s} {'sec':>6s}"]
    for k, s in rep['ops'].items():
        lines.append(f"{k:36s} {s['generated']:4d} {s['answered']:4d} {s['agree']:5d} {s['refused']:7d} {s['disagree']:8d} {s['seconds']:6.1f}"
                     + (f"   (+{s['expected_refusals_ok']} refused as the stated bound/scope intends)" if s['expected_refusals_ok'] else ''))
        for r, c in s['refusals'].items():
            lines.append(f'      refused x{c}: {r}')
    t = rep['totals']
    lines.append(f"{'TOTAL':36s} {t['generated']:4d} {t['answered']:4d} {t['agree']:5d} {t['refused']:7d} {t['disagree']:8d} {rep['seconds']:6.1f}")
    if rep['no_case']:
        lines.append('operations with no case: ' + ', '.join(rep['no_case']))
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--n', type=int, default=60)
    ap.add_argument('--only', nargs='*')
    ap.add_argument('--no-json', action='store_true')
    a = ap.parse_args()
    rep = run(a.seed, a.n, a.only, write=False if a.no_json else None)
    print(table(rep))
    for d in rep['disagreements'][:15]:
        print('DISAGREE', json.dumps(d, default=str)[:700])
    if not a.no_json and not a.only:
        print('wrote', Path(__file__).resolve().parents[3] / 'build-mathbackend' / 'algebra.json')
    return 1 if rep['totals']['disagree'] else 0


if __name__ == '__main__':
    sys.exit(main())
