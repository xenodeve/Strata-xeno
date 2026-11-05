"""MIP math backend, algebra: univariate polynomials over Q, polynomials mod a prime, exact matrices over Q, linear systems, linear recurrences.

Standard library only. Written from the textbook algorithms (Euclid, Yun, Sturm, Cantor-Zassenhaus, Hensel lifting, Bareiss, Faddeev-LeVerrier,
Berlekamp-Massey); nothing is copied from another project (third_party/mathbackend/NOTICES_algebra.md).

Conventions
- A polynomial is a list of coefficients, highest degree first, as written: x^3 + 2x + 1 is [1, 0, 2, 1]. The zero polynomial is []. Inputs may be ints,
  Fractions, strings ('1/2', '0.25') or a string such as '2x^3 - x/3 + 1'; results are lists of Fractions without leading zeros.
- A matrix is a list of rows. Entries are exact (int, Fraction, string); floats are refused (BadInput).
- A real root that is not rational is given by an exact rational interval, and a decimal value flagged 'numeric': True. An exact answer is never rounded.
- An operation whose input is past its bound, or whose method cannot finish, raises NotCovered; it never returns a partial factorisation as a complete one.
- Bounds are stated in each @op. Results are deterministic (the one random choice, Cantor-Zassenhaus, uses a fixed seed).
"""
import random
import re
from decimal import Decimal, ROUND_HALF_EVEN, localcontext
from fractions import Fraction
from itertools import combinations
from math import gcd, isqrt

from tools.igc.mathbackend import BadInput, NotCovered, Surd, op

MAX_DEG = 2000            # longest polynomial any operation takes
MAX_DIM = 60              # largest matrix side
MAX_DIGITS = 200          # most digits a numeric result is asked for

NOT_COVERED = {
    'exact roots of an irreducible polynomial of degree 3 or more (radicals)': 'only isolating rational intervals and decimals are given; Cardano/Ferrari/Galois-group tests are not written',
    'factorisation over an extension field, over R or over C': 'factor_z factors over Z (Q) only; quadratics are split by quadratic_roots into surds',
    'polynomials in several variables (factor, gcd, resultant, Groebner bases)': 'the backend is univariate; a system that is not linear and not one-variable is refused',
    'nonlinear systems of equations': 'no elimination: only linear systems and one-variable polynomial equations',
    'Jordan form, Smith form, irrational or complex eigenvalues': 'mat_eigen gives the rational eigenvalues, their eigenvectors and the characteristic factor left over',
    'matrices with symbolic entries': 'entries are exact numbers only',
    'recurrence closed forms with complex roots or several different irrational roots': 'recurrence_closed_form handles rational roots (with multiplicity) and at most one real quadratic pair',
    'factor_z past its bounds': 'degree 120, 40 000 recombination tests (a polynomial with very many modular factors is refused, not guessed)',
}


# ---------------------------------------------------------------- numbers and polynomials over Q

def _q(x):
    if isinstance(x, Fraction):
        return x
    if isinstance(x, bool):
        raise BadInput('a boolean is no number')
    if isinstance(x, int):
        return Fraction(x)
    if isinstance(x, str):
        try:
            return Fraction(x.strip().replace(' ', ''))
        except (ValueError, ZeroDivisionError):
            raise BadInput(f'not an exact number: {x!r}')
    if isinstance(x, Decimal):
        if not x.is_finite():
            raise BadInput('not a finite number')
        return Fraction(x)
    if isinstance(x, float):
        raise BadInput('a float is not exact: pass an int, a Fraction or a string')
    raise BadInput(f'not an exact number: {x!r}')


def _trim(p):
    i = 0
    while i < len(p) and p[i] == 0:
        i += 1
    return list(p[i:])


def _poly(p):
    """A polynomial argument -> trimmed list of Fractions (a string is parsed, a number is a constant)."""
    if isinstance(p, str):
        return _parse_univariate(p)[0]
    if isinstance(p, (int, Fraction)) and not isinstance(p, bool):
        return _trim([Fraction(p)])
    if not isinstance(p, (list, tuple)):
        raise BadInput(f'not a polynomial: {p!r}')
    if len(p) > MAX_DEG + 1:
        raise NotCovered(f'degree past {MAX_DEG}')
    return _trim([_q(c) for c in p])


def _deg(p):
    return len(p) - 1


def _add(p, q):
    if len(p) < len(q):
        p, q = q, p
    r = list(p)
    off = len(p) - len(q)
    for i, c in enumerate(q):
        r[off + i] += c
    return _trim(r)


def _neg(p):
    return [-c for c in p]


def _sub(p, q):
    return _add(p, _neg(q))


def _scale(p, c):
    return [] if c == 0 else [c * a for a in p]


def _mul(p, q):
    if not p or not q:
        return []
    if len(p) + len(q) - 2 > MAX_DEG:
        raise NotCovered(f'degree past {MAX_DEG}')
    r = [Fraction(0)] * (len(p) + len(q) - 1)
    for i, a in enumerate(p):
        if a == 0:
            continue
        for j, b in enumerate(q):
            r[i + j] += a * b
    return r


def _divmod(p, q):
    if not q:
        raise BadInput('division by the zero polynomial')
    if len(p) < len(q):
        return [], list(p)
    r = list(p)
    qd = len(q) - 1
    lc = q[0]
    out = []
    for i in range(len(p) - qd):
        c = r[i] / lc
        out.append(c)
        if c != 0:
            for j in range(1, len(q)):
                r[i + j] -= c * q[j]
    return _trim(out), _trim(r[len(p) - qd:])


def _monic(p):
    return [c / p[0] for c in p] if p else []


def _iprim(a):
    g = 0
    for c in a:
        g = gcd(g, c)
    return [c // g for c in a] if g > 1 else list(a)


def _iprem(a, b):
    """Pseudo-remainder of integer lists (a times a power of lc(b), reduced mod b), kept small by removing the content at every step."""
    r = list(a)
    db, lb = len(b), b[0]
    while len(r) >= db and r:
        c = r[0]
        r = [lb * v for v in r]
        for j in range(db):
            r[j] -= c * b[j]
        i = 0
        while i < len(r) and r[i] == 0:
            i += 1
        r = r[i:]
        if r:
            r = _iprim(r)
    return r


def _gcd(p, q):
    """Monic gcd by the primitive remainder sequence over Z (no coefficient swell); [] when both are zero."""
    if not p:
        return _monic(q)
    if not q:
        return _monic(p)
    a, b = _int_content(p)[1], _int_content(q)[1]
    if len(a) < len(b):
        a, b = b, a
    while True:
        r = _iprem(a, b)
        if not r:
            break
        a, b = b, _iprim(r)
    return _monic([Fraction(c) for c in b])


def _gcdex(p, q):
    """(s, t, g) with s*p + t*q = g, g monic (or zero)."""
    r0, r1, s0, s1, t0, t1 = p, q, [Fraction(1)], [], [], [Fraction(1)]
    while r1:
        qq, r = _divmod(r0, r1)
        r0, r1 = r1, r
        s0, s1 = s1, _sub(s0, _mul(qq, s1))
        t0, t1 = t1, _sub(t0, _mul(qq, t1))
    if not r0:
        return [], [], []
    c = 1 / r0[0]
    return _scale(s0, c), _scale(t0, c), _scale(r0, c)


def _deriv(p):
    n = len(p) - 1
    return _trim([c * (n - i) for i, c in enumerate(p[:-1])])


def _eval(p, x):
    if not p:
        return Fraction(0)
    acc = p[0]
    for c in p[1:]:
        acc = acc * x + c
    return acc


def _compose(p, q):
    out = []
    for c in p:
        out = _add(_mul(out, q), [c] if c != 0 else [])
    return out


def _pow(p, n):
    r, b = [Fraction(1)], p
    while n:
        if n & 1:
            r = _mul(r, b)
        n >>= 1
        if n:
            b = _mul(b, b)
    return r


def _int_content(p):
    """(content > 0, primitive integer list with the sign of p kept): p = content * primitive, positive scaling only."""
    if not p:
        return Fraction(0), []
    den = 1
    for c in p:
        den = den * c.denominator // gcd(den, c.denominator)
    ints = [int(c * den) for c in p]
    g = 0
    for c in ints:
        g = gcd(g, c)
    return Fraction(g, den), [c // g for c in ints]


def _clear(p):
    """p = content * primitive, the primitive an integer polynomial with a positive leading coefficient (the content carries the sign)."""
    c, pr = _int_content(p)
    if pr and pr[0] < 0:
        c, pr = -c, [-a for a in pr]
    return c, pr


def _sqf_decomp(p):
    """Yun: list of (monic square-free g_i, i) with p = lc * prod g_i^i. p non-constant."""
    if len(p) <= 1:
        return []
    dp = _deriv(p)
    g = _gcd(p, dp)
    b = _divmod(p, g)[0]
    c = _divmod(dp, g)[0]
    d = _sub(c, _deriv(b))
    i, out = 1, []
    while len(b) > 1:
        a = _gcd(b, d)
        if len(a) > 1:
            out.append((a, i))
        b = _divmod(b, a)[0]
        c = _divmod(d, a)[0]
        d = _sub(c, _deriv(b))
        i += 1
    return out


def _nonzero(p, what='polynomial'):
    if not p:
        raise BadInput(f'the zero {what} has no such answer')


MAX_POW_BITS = 400000       # largest result a power may produce, in bits of one coefficient (about 120000 digits)


def _check_pow_size(coefficients, e):
    bits = max(max(abs(Fraction(c).numerator).bit_length(), Fraction(c).denominator.bit_length()) for c in coefficients)
    if bits > 1 and e * bits > MAX_POW_BITS:
        raise NotCovered('power too large: the result would have over 120000 digits')


@op('algebra.poly_from_str', kind='polynomials over Q', bound='degree <= 2000, one variable')
def poly_from_str(s, var=None):
    """Coefficients (highest degree first) of a one-variable polynomial written in text; products and powers of brackets are expanded.
    poly_from_str('(x+1)^2 - 3x/2') -> [1, 1/2, 1]"""
    p, v = _parse_univariate(s)
    if var is not None and v is not None and v != var:
        raise BadInput(f'the variable is {v}, not {var}')
    return p


@op('algebra.poly_to_str', kind='polynomials over Q', bound='degree <= 2000')
def poly_to_str(p, var='x'):
    """The polynomial as text, highest power first. poly_to_str([1, 0, -1/2]) -> 'x^2 - 1/2'"""
    p = _poly(p)
    if not p:
        return '0'
    n = len(p) - 1
    out = []
    for i, c in enumerate(p):
        if c == 0:
            continue
        e = n - i
        mag = abs(c)
        body = ''
        if e == 0:
            body = str(mag)
        else:
            coef = '' if mag == 1 else (str(mag) + '*' if mag.denominator == 1 else '(' + str(mag) + ')*')
            body = coef + var + ('' if e == 1 else f'^{e}')
        out.append(('-' if c < 0 else '+', body))
    s = ('-' if out[0][0] == '-' else '') + out[0][1]
    for sign, body in out[1:]:
        s += f' {sign} {body}'
    return s


@op('algebra.poly_add', kind='polynomials over Q', bound='degree <= 2000')
def poly_add(p, q):
    """poly_add([1, 2], [1, -2]) -> [2, 0]"""
    return _add(_poly(p), _poly(q))


@op('algebra.poly_sub', kind='polynomials over Q', bound='degree <= 2000')
def poly_sub(p, q):
    """poly_sub([1, 0, 1], [1, 1]) -> [1, -1, 0]"""
    return _sub(_poly(p), _poly(q))


@op('algebra.poly_mul', kind='polynomials over Q', bound='degree of the product <= 2000')
def poly_mul(p, q):
    """poly_mul([1, 1], [1, -1]) -> [1, 0, -1]"""
    return _mul(_poly(p), _poly(q))


@op('algebra.poly_pow', kind='polynomials over Q', bound='degree of the result <= 2000, n >= 0')
def poly_pow(p, n):
    """poly_pow([1, 1], 3) -> [1, 3, 3, 1]"""
    p = _poly(p)
    if not isinstance(n, int) or isinstance(n, bool) or n < 0:
        raise BadInput('the exponent must be a non-negative integer')
    if p and n * (len(p) - 1) > MAX_DEG:
        raise NotCovered(f'degree past {MAX_DEG}')
    if not p:
        return [Fraction(1)] if n == 0 else []
    _check_pow_size(p, n)
    return _pow(p, n)


@op('algebra.poly_divmod', kind='polynomials over Q', bound='degree <= 2000')
def poly_divmod(p, q):
    """Quotient and remainder, deg r < deg q. poly_divmod([1, 0, 0, -1], [1, -1]) -> ([1, 1, 1], [])"""
    return _divmod(_poly(p), _poly(q))


@op('algebra.poly_gcd', kind='polynomials over Q', bound='degree <= 2000')
def poly_gcd(p, q):
    """Monic greatest common divisor (gcd of two zero polynomials is []). poly_gcd([1, 0, -1], [1, 3, 2]) -> [1, 1]"""
    return _gcd(_poly(p), _poly(q))


@op('algebra.poly_lcm', kind='polynomials over Q', bound='degree <= 2000')
def poly_lcm(p, q):
    """Monic least common multiple (the zero polynomial if either is zero). poly_lcm([1, -1], [1, 1]) -> [1, 0, -1]"""
    p, q = _poly(p), _poly(q)
    if not p or not q:
        return []
    return _monic(_mul(_divmod(p, _gcd(p, q))[0], q))


@op('algebra.poly_gcdex', kind='polynomials over Q', bound='degree <= 2000')
def poly_gcdex(p, q):
    """(s, t, g) with s*p + t*q = g = monic gcd (Bezout). poly_gcdex([1, 0, -1], [1, 3, 2]) -> ([0], [1/3 ...]) with s*p + t*q = [1, 1]"""
    return _gcdex(_poly(p), _poly(q))


@op('algebra.poly_compose', kind='polynomials over Q', bound='degree of the result <= 2000')
def poly_compose(p, q):
    """p(q(x)). poly_compose([1, 0, 1], [1, 1]) -> [1, 2, 2]"""
    p, q = _poly(p), _poly(q)
    if p and q and (len(p) - 1) * max(len(q) - 1, 0) > MAX_DEG:
        raise NotCovered(f'degree past {MAX_DEG}')
    return _compose(p, q)


@op('algebra.poly_derivative', kind='polynomials over Q', bound='degree <= 2000')
def poly_derivative(p, k=1):
    """k-th derivative. poly_derivative([1, 0, 0, 1]) -> [3, 0, 0]"""
    p = _poly(p)
    for _ in range(k):
        p = _deriv(p)
    return p


@op('algebra.poly_antiderivative', kind='polynomials over Q', bound='degree <= 2000')
def poly_antiderivative(p):
    """The antiderivative with constant term 0. poly_antiderivative([3, 0, 0]) -> [1, 0, 0, 0]"""
    p = _poly(p)
    n = len(p)
    return [c / (n - i) for i, c in enumerate(p)] + ([Fraction(0)] if p else [])


@op('algebra.poly_integral', kind='polynomials over Q', bound='degree <= 2000')
def poly_integral(p, a, b):
    """Exact definite integral from a to b. poly_integral([1, 0, 0], 0, 3) -> 9"""
    p = _poly(p)
    a, b = _q(a), _q(b)
    n = len(p)
    F = [c / (n - i) for i, c in enumerate(p)] + ([Fraction(0)] if p else [])
    return _eval(F, b) - _eval(F, a)


@op('algebra.poly_eval', kind='polynomials over Q', bound='degree <= 2000')
def poly_eval(p, x):
    """Value at a rational (or a Surd) x. poly_eval([1, 0, -2], 3) -> 7"""
    p = _poly(p)
    if isinstance(x, Surd):
        return _eval(p, x) if p else Surd(0)
    return _eval(p, _q(x))


@op('algebra.poly_taylor_shift', kind='polynomials over Q', bound='degree <= 2000')
def poly_taylor_shift(p, a):
    """The coefficients of p(x + a). poly_taylor_shift([1, 0, 0], 2) -> [1, 4, 4]"""
    p = _poly(p)
    a = _q(a)
    work = list(p)
    n = len(work)
    for i in range(n - 1):               # repeated synthetic division by (x - (-a))
        for j in range(1, n - i):
            work[j] += a * work[j - 1]
    return _trim(work)


@op('algebra.poly_content_primitive', kind='polynomials over Q', bound='degree <= 2000')
def poly_content_primitive(p):
    """(content, primitive part): p = content * primitive, the primitive part an integer polynomial with positive lead and gcd 1.
    poly_content_primitive([2, 4, -6]) -> (2, [1, 2, -3])"""
    p = _poly(p)
    if not p:
        return Fraction(0), []
    c, pr = _clear(p)
    return c, [Fraction(a) for a in pr]


@op('algebra.poly_squarefree_part', kind='polynomials over Q', bound='degree <= 2000')
def poly_squarefree_part(p):
    """p / gcd(p, p'), same roots without repeats (the leading coefficient is kept). poly_squarefree_part([1, -2, 1]) -> [1, -1]"""
    p = _poly(p)
    _nonzero(p)
    if len(p) <= 1:
        return p
    return _divmod(p, _gcd(p, _deriv(p)))[0]


@op('algebra.poly_squarefree_decomposition', kind='polynomials over Q', bound='degree <= 2000')
def poly_squarefree_decomposition(p):
    """(lc, [(g_i, i)]): p = lc * prod g_i^i with every g_i monic, square-free and the g_i pairwise coprime.
    poly_squarefree_decomposition([1, -3, 3, -1]) -> (1, [([1, -1], 3)])"""
    p = _poly(p)
    _nonzero(p)
    return p[0], _sqf_decomp(p)


@op('algebra.poly_from_roots', kind='polynomials over Q', bound='degree <= 2000')
def poly_from_roots(roots, lead=1):
    """lead * prod (x - r) over the roots as listed (repeat a root for its multiplicity). poly_from_roots([1, 2, 2]) -> [1, -5, 8, -4]"""
    if len(roots) > MAX_DEG:
        raise NotCovered(f'degree past {MAX_DEG}')
    p = [_q(lead)]
    if p[0] == 0:
        raise BadInput('the leading coefficient is zero')
    for r in roots:
        r = _q(r)
        p = _mul(p, [Fraction(1), -r])
    return p


@op('algebra.poly_transform_roots', kind='polynomials over Q', bound='degree <= 2000')
def poly_transform_roots(p, kind, c=1):
    """The polynomial whose roots are the roots r of p changed by `kind`: 'shift' (r + c), 'scale' (c*r, c != 0), 'reciprocal' (1/r, p(0) != 0).
    Monic when the lead of p is 1. poly_transform_roots([1, -3, 2], 'shift', 1) -> [1, -5, 6]"""
    p = _poly(p)
    _nonzero(p)
    c = _q(c)
    if kind == 'shift':
        return poly_taylor_shift(p, -c)
    if kind == 'scale':
        if c == 0:
            raise BadInput('scale by zero')
        return _trim([a * c ** i for i, a in enumerate(p)])
    if kind == 'reciprocal':
        if p[-1] == 0:
            raise BadInput('a zero root has no reciprocal')
        return list(reversed(p))
    raise BadInput("kind must be 'shift', 'scale' or 'reciprocal'")


# ---------------------------------------------------------------- text parser (one or several variables, used by the string operations)

def _norm_text(s):
    if not isinstance(s, str):
        raise BadInput('expected a string')
    s = s.replace('$', '').replace('\\left', '').replace('\\right', '').replace('\\cdot', '*').replace('\\times', '*')
    s = s.replace('−', '-').replace('·', '*').replace('×', '*').replace('**', '^')
    s = re.sub(r'\^\s*\{([^{}]*)\}', r'^(\1)', s)
    for _ in range(3):
        s = re.sub(r'\\d?frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}', r'((\1)/(\2))', s)
    return s.replace('{', '(').replace('}', ')')


_TOKEN = re.compile(r'\s*(?:(\d+(?:\.\d+)?)|([A-Za-z])|(.))', re.S)


def _tokens(s):
    out, i = [], 0
    s = _norm_text(s)
    while i < len(s):
        m = _TOKEN.match(s, i)
        if not m:
            break
        i = m.end()
        if m.group(1) is not None:
            out.append(('n', Fraction(m.group(1))))
        elif m.group(2) is not None:
            out.append(('v', m.group(2)))
        elif m.group(3) is not None and not m.group(3).isspace():
            out.append(('o', m.group(3)))
    return out


def _mp_add(a, b, sign=1):
    r = dict(a)
    for m, c in b.items():
        v = r.get(m, 0) + sign * c
        if v == 0:
            r.pop(m, None)
        else:
            r[m] = v
    return r


def _mp_mul(a, b):
    r = {}
    for m1, c1 in a.items():
        for m2, c2 in b.items():
            d = dict(m1)
            for v, e in m2:
                d[v] = d.get(v, 0) + e
            m = tuple(sorted(d.items()))
            val = r.get(m, 0) + c1 * c2
            if val == 0:
                r.pop(m, None)
            else:
                r[m] = val
    if max((sum(e for _, e in m) for m in r), default=0) > MAX_DEG:
        raise NotCovered(f'degree past {MAX_DEG}')
    return r


class _Parser:
    def __init__(self, s):
        self.t = _tokens(s)
        self.i = 0
        if len(self.t) > 4000:
            raise NotCovered('expression too long')

    def peek(self):
        return self.t[self.i] if self.i < len(self.t) else (None, None)

    def take(self):
        tok = self.peek()
        self.i += 1
        return tok

    def expr(self):
        acc = self.term()
        while self.peek() in (('o', '+'), ('o', '-')):
            sign = 1 if self.take()[1] == '+' else -1
            acc = _mp_add(acc, self.term(), sign)
        return acc

    def term(self):
        acc = self.power()
        while True:
            k, v = self.peek()
            if (k, v) == ('o', '*'):
                self.take()
                acc = _mp_mul(acc, self.power())
            elif (k, v) == ('o', '/'):
                self.take()
                d = self.power()
                if not d:
                    raise BadInput('division by zero')
                if list(d) != [()]:
                    raise NotCovered('division by an expression that is not a number')
                acc = {m: c / d[()] for m, c in acc.items()}
            elif k == 'v' or (k, v) == ('o', '('):
                acc = _mp_mul(acc, self.power())     # implicit product: 2x, x(x+1), (x+1)(x-1)
            elif k == 'n':
                raise BadInput('two numbers side by side')
            else:
                return acc

    def power(self):
        k, v = self.peek()
        if (k, v) == ('o', '-'):
            self.take()
            return {m: -c for m, c in self.power().items()}
        if (k, v) == ('o', '+'):
            self.take()
            return self.power()
        base = self.atom()
        if self.peek() == ('o', '^'):
            self.take()
            neg = False
            if self.peek() == ('o', '-'):
                self.take()
                neg = True
            k, v = self.peek()
            if (k, v) == ('o', '('):
                self.take()
                if self.peek() == ('o', '-'):
                    self.take()
                    neg = not neg
                k2, v2 = self.take()
                if k2 != 'n' or self.take() != ('o', ')'):
                    raise NotCovered('exponent is not an integer')
                e = v2
            elif k == 'n':
                self.take()
                e = v
            else:
                raise NotCovered('exponent is not an integer')
            if e.denominator != 1 or neg:
                raise NotCovered('exponent is not a non-negative integer')
            e = int(e)
            deg1 = max((sum(x for _, x in m) for m in base), default=0)
            if e * deg1 > MAX_DEG:
                raise NotCovered(f'degree past {MAX_DEG}')
            if base:
                _check_pow_size(list(base.values()), e)
            r, b = {(): Fraction(1)}, base
            while e:
                if e & 1:
                    r = _mp_mul(r, b)
                e >>= 1
                if e:
                    b = _mp_mul(b, b)
            return r
        return base

    def atom(self):
        k, v = self.take()
        if k == 'n':
            return {(): v} if v != 0 else {}
        if k == 'v':
            return {((v, 1),): Fraction(1)}
        if (k, v) == ('o', '('):
            r = self.expr()
            if self.take() != ('o', ')'):
                raise BadInput('unbalanced brackets')
            return r
        raise BadInput(f'cannot read {v!r}' if k else 'the expression ended early')


def _parse_multi(s):
    p = _Parser(s)
    try:
        r = p.expr()
    except RecursionError:
        raise NotCovered('expression nested too deeply')
    if p.i != len(p.t):
        raise BadInput(f'cannot read {p.t[p.i][1]!r}')
    return r


def _parse_univariate(s):
    """(coefficients highest first, variable name or None)."""
    return _mp_univariate(_parse_multi(s))


def _mp_univariate(mp):
    vs = sorted({v for m in mp for v, _ in m})
    if len(vs) > 1:
        raise NotCovered(f'several variables: {", ".join(vs)}')
    v = vs[0] if vs else None
    if not mp:
        return [], v
    deg = max(sum(e for _, e in m) for m in mp)
    co = [Fraction(0)] * (deg + 1)
    for m, c in mp.items():
        co[deg - sum(e for _, e in m)] = c
    return _trim(co), v


# ---------------------------------------------------------------- integers: primes and factoring (small, for rational roots and prime moduli)

_SMALL = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41)
_MR_LIMIT = 3317044064679887385961981       # deterministic with the first 13 primes below this


def _is_prime(n):
    if n < 2:
        return False
    for q in _SMALL:
        if n % q == 0:
            return n == q
    if n >= _MR_LIMIT:
        raise NotCovered('cannot certify a prime above 3.3e24')
    d, s = n - 1, 0
    while d % 2 == 0:
        d //= 2
        s += 1
    for a in _SMALL:
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(s - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def _rho(n):
    for c in range(1, 40):
        x = y = 2
        d = 1
        steps = 0
        while d == 1:
            x = (x * x + c) % n
            y = (y * y + c) % n
            y = (y * y + c) % n
            d = gcd(abs(x - y), n)
            steps += 1
            if steps > 400000:
                break
        if 1 < d < n:
            return d
    raise NotCovered('cannot factor an integer: Pollard rho gave up')


def _factor_int(n):
    f = {}
    p = 2
    while p < 20000 and p * p <= n:
        while n % p == 0:
            f[p] = f.get(p, 0) + 1
            n //= p
        p += 1 if p == 2 else 2
    stack = [n] if n > 1 else []
    while stack:
        m = stack.pop()
        if m == 1:
            continue
        if _is_prime(m):
            f[m] = f.get(m, 0) + 1
            continue
        d = _rho(m)
        stack += [d, m // d]
    return f


def _divisors(n):
    ds = [1]
    for p, e in _factor_int(n).items():
        ds = [d * p ** k for d in ds for k in range(e + 1)]
    return sorted(ds)


# ---------------------------------------------------------------- polynomials modulo m (lists of ints, highest first); division needs a unit leading coefficient

def _mt(p):
    i = 0
    while i < len(p) and p[i] == 0:
        i += 1
    return p[i:]


def _mred(p, m):
    return _mt([c % m for c in p])


def _madd(p, q, m):
    if len(p) < len(q):
        p, q = q, p
    r = list(p)
    off = len(p) - len(q)
    for i, c in enumerate(q):
        r[off + i] = (r[off + i] + c) % m
    return _mt(r)


def _msub(p, q, m):
    return _madd(p, [(-c) % m for c in q], m)


def _mmul(p, q, m):
    if not p or not q:
        return []
    r = [0] * (len(p) + len(q) - 1)
    for i, a in enumerate(p):
        if a:
            for j, b in enumerate(q):
                r[i + j] += a * b
    return _mt([c % m for c in r])


def _mdivmod(p, q, m):
    if not q:
        raise BadInput('division by the zero polynomial')
    if len(p) < len(q):
        return [], list(p)
    inv = pow(q[0], -1, m)
    r = list(p)
    qd = len(q) - 1
    out = []
    for i in range(len(p) - qd):
        c = r[i] * inv % m
        out.append(c)
        if c:
            for j in range(1, len(q)):
                r[i + j] = (r[i + j] - c * q[j]) % m
    return _mt(out), _mt([c % m for c in r[len(p) - qd:]])


def _mmonic(p, m):
    if not p:
        return []
    inv = pow(p[0], -1, m)
    return [c * inv % m for c in p]


def _mgcd(p, q, m):
    while q:
        p, q = q, _mdivmod(p, q, m)[1]
    return _mmonic(p, m)


def _mgcdex(p, q, m):
    r0, r1, s0, s1, t0, t1 = p, q, [1], [], [], [1]
    while r1:
        qq, r = _mdivmod(r0, r1, m)
        r0, r1 = r1, r
        s0, s1 = s1, _msub(s0, _mmul(qq, s1, m), m)
        t0, t1 = t1, _msub(t0, _mmul(qq, t1, m), m)
    if not r0:
        return [], [], []
    inv = pow(r0[0], -1, m)
    return [c * inv % m for c in s0], [c * inv % m for c in t0], [c * inv % m for c in r0]


def _mpowmod(b, e, f, m):
    r = [1]
    b = _mdivmod(b, f, m)[1]
    while e:
        if e & 1:
            r = _mdivmod(_mmul(r, b, m), f, m)[1]
        e >>= 1
        if e:
            b = _mdivmod(_mmul(b, b, m), f, m)[1]
    return r


def _mderiv(p, m):
    n = len(p) - 1
    return _mt([c * (n - i) % m for i, c in enumerate(p[:-1])])


def _fp_sqf(f, p):
    """Square-free decomposition of a monic polynomial over F_p: list of (monic square-free g, multiplicity)."""
    out = []
    if len(f) <= 1:
        return out
    c = _mgcd(f, _mderiv(f, p), p)
    w = _mdivmod(f, c, p)[0]
    i = 1
    while len(w) > 1:
        y = _mgcd(w, c, p)
        z = _mdivmod(w, y, p)[0]
        if len(z) > 1:
            out.append((_mmonic(z, p), i))
        i += 1
        w = y
        c = _mdivmod(c, y, p)[0]
    if len(c) > 1:      # c is a p-th power: c = h(x^p), and over F_p h(x^p) = h(x)^p
        h = [c[k] for k in range(0, len(c), p)]
        for g, mult in _fp_sqf(_mmonic(h, p), p):
            out.append((g, mult * p))
    return out


def _fp_ddf(f, p):
    """Distinct-degree factorisation of a monic square-free f: list of (product of the irreducible factors of degree d, d)."""
    out = []
    h = [1, 0]
    d = 0
    while len(f) - 1 >= 2 * (d + 1):
        d += 1
        h = _mpowmod(h, p, f, p)
        g = _mgcd(_msub(h, [1, 0], p), f, p)
        if len(g) > 1:
            out.append((g, d))
            f = _mdivmod(f, g, p)[0]
            h = _mdivmod(h, f, p)[1]
    if len(f) > 1:
        out.append((f, len(f) - 1))
    return out


def _fp_edf(f, d, p, rng):
    """Split a monic f, a product of irreducibles of degree d, into them (Cantor-Zassenhaus; the trace map when p = 2)."""
    n = len(f) - 1
    if n == d:
        return [f]
    while True:
        a = _mt([rng.randrange(p) for _ in range(n)])
        if len(a) < 2:
            continue
        if p == 2:
            t, cur = a, a
            for _ in range(d - 1):
                cur = _mdivmod(_mmul(cur, cur, 2), f, 2)[1]
                t = _madd(t, cur, 2)
            g = _mgcd(t, f, 2)
        else:
            b = _mpowmod(a, (p ** d - 1) // 2, f, p)
            g = _mgcd(_msub(b, [1], p), f, p)
        if 1 < len(g) < len(f):
            return _fp_edf(g, d, p, rng) + _fp_edf(_mdivmod(f, g, p)[0], d, p, rng)


def _fp_factor(f, p):
    """Monic f over F_p -> sorted list of (monic irreducible, multiplicity)."""
    rng = random.Random(1234567)
    out = []
    for g, mult in _fp_sqf(f, p):
        for h, d in _fp_ddf(g, p):
            for irr in _fp_edf(h, d, p, rng):
                out.append((irr, mult))
    out.sort(key=lambda t: (len(t[0]), t[0]))
    return out


# ---------------------------------------------------------------- factorisation over Z (Zassenhaus: factor mod p, Hensel lift, recombine)

MAX_FACTOR_DEG = 120
MAX_RECOMBINE_TESTS = 40000


def _ipoly_mul(p, q):
    if not p or not q:
        return []
    r = [0] * (len(p) + len(q) - 1)
    for i, a in enumerate(p):
        if a:
            for j, b in enumerate(q):
                r[i + j] += a * b
    return r


def _idiv_exact(f, g):
    """Quotient of integer polynomials when g divides f exactly, else None."""
    if len(g) > len(f):
        return None
    r = list(f)
    out = []
    gd = len(g) - 1
    for i in range(len(f) - gd):
        if r[i] % g[0]:
            return None
        c = r[i] // g[0]
        out.append(c)
        if c:
            for j in range(1, len(g)):
                r[i + j] -= c * g[j]
    if any(r[len(f) - gd:]):
        return None
    return out


def _hensel_pair(F, g, h, p, k):
    """F monic mod p^k, g, h monic mod p, F = g*h mod p, g and h coprime mod p. Return monic g, h mod p^k with F = g*h."""
    s, t, _ = _mgcdex(g, h, p)       # s*g + t*h = 1 mod p
    g, h = list(g), list(h)
    pj = p
    for _ in range(k - 1):
        pj1 = pj * p
        prod = _mmul(g, h, pj1)
        # e = (F - g*h) / p^j, reduced mod p
        diff = _msub(_mred(F, pj1), prod, pj1)
        e = _mred([c // pj for c in diff], p) if diff else []
        dh = _mdivmod(_mmul(e, s, p), h, p)[1]
        dg = _mdivmod(_mmul(e, t, p), g, p)[1]
        g = _madd(g, [c * pj for c in dg], pj1)
        h = _madd(h, [c * pj for c in dh], pj1)
        pj = pj1
    return g, h


def _hensel(f, hs, p, k):
    """Monic lifts mod p^k of the monic factors hs (mod p) of f (mod p), product = f / lc(f) mod p^k."""
    m = p ** k
    inv = pow(f[0], -1, m)
    fm = [c * inv % m for c in f]

    def lift(F, hl):
        if len(hl) == 1:
            return [F]
        mid = len(hl) // 2
        g0, h0 = [1], [1]
        for x in hl[:mid]:
            g0 = _mmul(g0, x, p)
        for x in hl[mid:]:
            h0 = _mmul(h0, x, p)
        g, h = _hensel_pair(F, g0, h0, p, k)
        return lift(g, hl[:mid]) + lift(h, hl[mid:])
    return lift(fm, hs)


def _prim_int_poly(g):
    c = 0
    for a in g:
        c = gcd(c, a)
    g = [a // c for a in g]
    return [-a for a in g] if g[0] < 0 else g


def _factor_prim_sqfree(f):
    """f: integer list, primitive, square-free, positive lead, degree >= 1 -> list of irreducible integer lists (complete or NotCovered)."""
    n = len(f) - 1
    if n == 1:
        return [f]
    if n > MAX_FACTOR_DEG:
        raise NotCovered(f'factor_z: degree past {MAX_FACTOR_DEG}')
    lead = f[0]
    best, tried = None, 0
    p = 2
    while tried < 6 and p < 3000:
        p += 1
        if not _is_prime(p) or lead % p == 0:
            continue
        fp = _mred(f, p)
        fm = _mmonic(fp, p)
        if len(_mgcd(fm, _mderiv(fm, p), p)) > 1:
            continue
        facs = [g for g, _ in _fp_factor(fm, p)]
        tried += 1
        if len(facs) == 1:
            return [f]              # irreducible mod p with p not dividing the lead: irreducible over Z
        if best is None or len(facs) < len(best[1]):
            best = (p, facs)
        if len(facs) <= 2:
            break
    if best is None:
        raise NotCovered('factor_z: no good prime found')
    p, facs = best
    norm2 = isqrt(sum(a * a for a in f)) + 1
    bound = 2 * lead * (2 ** n) * norm2 + 1
    k = 1
    while p ** k <= bound:
        k += 1
    m = p ** k
    hs = _hensel(f, facs, p, k)
    half = m // 2

    def sym(c):
        c %= m
        return c - m if c > half else c

    found, T, fs, s, tests = [], list(range(len(hs))), list(f), 1, 0
    while 2 * s <= len(T):
        hit = False
        for S in combinations(T, s):
            tests += 1
            if tests > MAX_RECOMBINE_TESTS:
                raise NotCovered('factor_z: too many factors mod p to recombine within the bound')
            g = [fs[0] % m]
            for i in S:
                g = _mmul(g, hs[i], m)
            g = [sym(c) for c in g]
            gp = _prim_int_poly(g)
            if gp[-1] == 0 or fs[-1] % gp[-1] or fs[0] % gp[0]:
                continue
            q = _idiv_exact(fs, gp)
            if q is not None:
                found.append(gp)
                fs = q
                T = [i for i in T if i not in S]
                hit = True
                break
        if not hit:
            s += 1
    if len(fs) > 1:
        found.append(fs)
    return found


@op('algebra.factor_z', kind='factorisation', bound='degree <= 120; at most 40000 recombination tests, else NotCovered (never a partial answer)')
def factor_z(p):
    """Complete factorisation over Z (= over Q) into irreducibles: {'content': c, 'factors': [(f, multiplicity)]}, p = c * prod f^m, every f a primitive
    integer polynomial with positive lead, sorted by degree. factor_z([1, 0, -1]) -> {'content': 1, 'factors': [([1, -1], 1), ([1, 1], 1)]}"""
    p = _poly(p)
    _nonzero(p)
    if len(p) > MAX_FACTOR_DEG + 1:
        raise NotCovered(f'factor_z: degree past {MAX_FACTOR_DEG}')
    factors = []
    q = list(p)
    z = 0
    while len(q) > 1 and q[-1] == 0:
        q.pop()
        z += 1
    if z:
        factors.append(([Fraction(1), Fraction(0)], z))
    for g, mult in _sqf_decomp(q):
        _, gi = _clear(g)
        for h in _factor_prim_sqfree(gi):
            factors.append(([Fraction(a) for a in h], mult))
    factors.sort(key=lambda t: (len(t[0]), t[0], t[1]))
    c = p[0]
    for h, mult in factors:
        c /= h[0] ** mult
    return {'content': c, 'factors': factors}


@op('algebra.is_irreducible_z', kind='factorisation', bound='as factor_z')
def is_irreducible_z(p):
    """True when p (over Q) is irreducible of degree >= 1. is_irreducible_z([1, 0, 1]) -> True"""
    r = factor_z(p)
    return len(r['factors']) == 1 and r['factors'][0][1] == 1 and len(r['factors'][0][0]) > 1


# ---------------------------------------------------------------- rational and real roots

def _rational_roots(p):
    """[(root, multiplicity)] sorted, p a trimmed non-zero Fraction list."""
    out = []
    p = list(p)
    z = 0
    while len(p) > 1 and p[-1] == 0:
        p.pop()
        z += 1
    if z:
        out.append((Fraction(0), z))
    if len(p) <= 1:
        return out
    for g, mult in _sqf_decomp(p):
        _, gi = _clear(g)
        n = len(gi) - 1
        if n == 1:
            out.append((Fraction(-gi[1], gi[0]), mult))
            continue
        try:
            d0, dn = _divisors(abs(gi[-1])), _divisors(gi[0])
            if len(d0) * len(dn) > 40000:
                raise NotCovered('many candidates')
            cands = [Fraction(u, v) for u in d0 for v in dn if gcd(u, v) == 1]
            for r in cands:
                for s in (r, -r):
                    if sum(c * s.numerator ** (n - i) * s.denominator ** i for i, c in enumerate(gi)) == 0:
                        out.append((s, mult))
        except NotCovered:
            for h in _factor_prim_sqfree(gi):
                if len(h) == 2:
                    out.append((Fraction(-h[1], h[0]), mult))
    out.sort()
    return out


@op('algebra.rational_roots', kind='roots', bound='divisor search up to 40000 candidates, else factor_z')
def rational_roots(p):
    """All rational roots with multiplicity [(root, m)], sorted. rational_roots([1, -5, 8, -4]) -> [(1, 1), (2, 2)]"""
    p = _poly(p)
    _nonzero(p)
    return _rational_roots(p)


def _pos_prim(p):
    c, pr = _int_content(p)
    return [Fraction(a) for a in pr] if pr else []


def _sturm(g):
    seq = [_pos_prim(g), _pos_prim(_deriv(g))]
    while True:
        r = _divmod(seq[-2], seq[-1])[1]
        if not r:
            break
        seq.append(_pos_prim(_neg(r)))
    return seq


def _sgn(x):
    return (x > 0) - (x < 0)


def _variations(seq, x):
    prev, v = 0, 0
    for p in seq:
        s = _sgn(_eval(p, x))
        if s == 0:
            continue
        if prev and s != prev:
            v += 1
        prev = s
    return v


def _variations_inf(seq, sign):
    prev, v = 0, 0
    for p in seq:
        s = _sgn(p[0]) * (sign ** (len(p) - 1) if sign < 0 else 1)
        if prev and s != prev:
            v += 1
        prev = s
    return v


def _count(seq, a, b):
    """Distinct real roots of the square-free polynomial the Sturm sequence belongs to, in (a, b]; None = infinity."""
    va = _variations_inf(seq, -1) if a is None else _variations(seq, a)
    vb = _variations_inf(seq, 1) if b is None else _variations(seq, b)
    return va - vb


@op('algebra.sturm_sequence', kind='roots', bound='degree <= 2000')
def sturm_sequence(p):
    """The Sturm sequence of the square-free part (each member scaled by a positive number to a primitive integer polynomial).
    sturm_sequence([1, 0, -2]) -> [[1, 0, -2], [1, 0], [1]]"""
    p = _poly(p)
    _nonzero(p)
    if len(p) <= 1:
        return [p]
    g = _divmod(p, _gcd(p, _deriv(p)))[0]
    return _sturm(g)


@op('algebra.count_real_roots', kind='roots', bound='degree <= 2000; a, b rational or None (infinite)')
def count_real_roots(p, a=None, b=None, with_multiplicity=False):
    """Number of real roots in (a, b] by Sturm's theorem (distinct roots; with_multiplicity=True counts each root as often as it repeats).
    count_real_roots([1, 0, -2]) -> 2; count_real_roots([1, 0, -2], 0, 1) -> 0"""
    p = _poly(p)
    _nonzero(p)
    a = None if a is None else _q(a)
    b = None if b is None else _q(b)
    if a is not None and b is not None and a >= b:
        return 0
    total = 0
    z = 0
    q = list(p)
    while len(q) > 1 and q[-1] == 0:
        q.pop()
        z += 1
    if z and (a is None or a < 0) and (b is None or b >= 0):
        total += z if with_multiplicity else 1
    for g, mult in _sqf_decomp(q):
        if len(g) == 1:
            continue
        c = _count(_sturm(g), a, b)
        total += c * (mult if with_multiplicity else 1)
    return total


def _cauchy(g):
    return 2 + max((abs(c / g[0]) for c in g[1:]), default=Fraction(0))


def _isolate(g, seq):
    B = _cauchy(g)
    stack = [(-B, B, _count(seq, -B, B))]
    out = []
    while stack:
        lo, hi, c = stack.pop()
        if c == 0:
            continue
        if c == 1:
            out.append((hi, hi) if _eval(g, hi) == 0 else (lo, hi))
            continue
        mid = (lo + hi) / 2
        cl = _count(seq, lo, mid)
        stack.append((lo, mid, cl))
        stack.append((mid, hi, c - cl))
    out.sort()
    return out


def _refine(g, seq, lo, hi, eps):
    while hi - lo > eps:
        mid = (lo + hi) / 2
        gm = _eval(g, mid)
        if gm == 0:
            return mid, mid
        glo = _eval(g, lo)
        if glo != 0:
            if _sgn(glo) != _sgn(gm):
                hi = mid
            else:
                lo = mid
        elif _count(seq, lo, mid) == 1:
            hi = mid
        else:
            lo = mid
    return lo, hi


def _split_rational(g):
    """(rational roots of the square-free g, g divided by them). A search that cannot finish is skipped: the root then shows as a tiny interval."""
    try:
        rats = [r for r, _ in _rational_roots(g)]
    except NotCovered:
        return [], g
    cof = g
    for r in rats:
        cof = _divmod(cof, [Fraction(1), -r])[0]
    return rats, cof


def _roots_sqfree(g):
    """All distinct real roots of the monic square-free g, sorted: (lo, hi, poly, sturm) with lo == hi for an exact rational root (poly, sturm None)."""
    rats, cof = _split_rational(g)
    out = [(r, r, None, None) for r in rats]
    if len(cof) > 1:
        seq = _sturm(cof)
        for lo, hi in _isolate(cof, seq):
            while any(lo <= r <= hi for r in rats):        # an interval must hold no rational root already reported
                lo, hi = _refine(cof, seq, lo, hi, (hi - lo) / 2)
            out.append((lo, hi, cof, seq))
    out.sort(key=lambda t: (t[0], t[1]))
    return out


def _all_roots(p):
    """Every distinct real root of p as [lo, hi, multiplicity, poly, sturm]; the intervals are pairwise disjoint and hold no exact root of another entry."""
    items = []
    for g, mult in _sqf_decomp(p):
        for lo, hi, gg, seq in _roots_sqfree(g):
            items.append([lo, hi, mult, gg, seq])
    while True:
        clash = None
        for i, a in enumerate(items):
            for j in range(i + 1, len(items)):
                b = items[j]
                if not (a[1] < b[0] or b[1] < a[0]):
                    clash = (a, b)
                    break
            if clash:
                break
        if not clash:
            break
        for it in clash:
            if it[0] != it[1]:
                it[0], it[1] = _refine(it[3], it[4], it[0], it[1], (it[1] - it[0]) / 2)
    items.sort(key=lambda t: (t[0], t[1]))
    return items


@op('algebra.isolate_real_roots', kind='roots', bound='degree <= 200 (Sturm with exact Fractions)')
def isolate_real_roots(p):
    """Disjoint exact rational intervals (lo, hi, multiplicity), each holding exactly one distinct real root in its open interval; lo == hi is an exact
    rational root. Sorted. isolate_real_roots([1, 0, -2]) -> [(-3/2.., ..), ...] two intervals around -sqrt2 and sqrt2"""
    p = _poly(p)
    _nonzero(p)
    if len(p) > 201:
        raise NotCovered('degree past 200')
    return [(lo, hi, mult) for lo, hi, mult, _, _ in _all_roots(p)]


def _dec(x, digits):
    with localcontext() as ctx:
        ctx.prec = len(str(abs(x.numerator))) + digits + 10
        v = Decimal(x.numerator) / Decimal(x.denominator)
        return v.quantize(Decimal(1).scaleb(-digits), rounding=ROUND_HALF_EVEN)


def _check_digits(digits):
    if not isinstance(digits, int) or isinstance(digits, bool) or digits < 1 or digits > MAX_DIGITS:
        raise BadInput(f'digits must be an integer from 1 to {MAX_DIGITS}')


@op('algebra.real_roots', kind='roots', bound='degree <= 200, digits <= 200')
def real_roots(p, digits=15):
    """Every distinct real root: {'root': Fraction (exact) or Decimal with `digits` places after the point (error < 10^-digits), 'exact': bool,
    'numeric': bool, 'multiplicity': m, 'interval': (lo, hi) exact rationals holding it}. real_roots([1, 0, -2], 5) -> roots -1.41421 and 1.41421"""
    _check_digits(digits)
    p = _poly(p)
    _nonzero(p)
    if len(p) > 201:
        raise NotCovered('degree past 200')
    eps = Fraction(1, 10 ** (digits + 3))
    out = []
    for lo, hi, mult, g, seq in _all_roots(p):
        if lo != hi:
            lo, hi = _refine(g, seq, lo, hi, eps)
        if lo == hi:
            out.append(dict(root=lo, exact=True, numeric=False, multiplicity=mult, interval=(lo, hi)))
        else:
            out.append(dict(root=_dec((lo + hi) / 2, digits), exact=False, numeric=True, multiplicity=mult, interval=(lo, hi)))
    out.sort(key=lambda d: d['interval'][0] + d['interval'][1])
    return out


def _cx_horner(g, z, prec):
    """(g(z), g'(z)) for z = (re, im) Decimals."""
    pr, pi = Decimal(0), Decimal(0)
    dr, di = Decimal(0), Decimal(0)
    zr, zi = z
    for c in g:
        dr, di = dr * zr - di * zi + pr, dr * zi + di * zr + pi
        pr, pi = pr * zr - pi * zi + c, pr * zi + pi * zr
    return (pr, pi), (dr, di)


def _aberth(cf):
    """Float approximations of all roots of the monic polynomial with float coefficients cf (highest first)."""
    import cmath
    n = len(cf) - 1
    r0 = max(abs(cf[-1]) ** (1.0 / n), 1e-3) if cf[-1] != 0 else 1.0
    z = [r0 * cmath.exp(1j * (2 * cmath.pi * k / n + 0.4)) for k in range(n)]
    for _ in range(800):
        worst = 0.0
        for k in range(n):
            pv, dv = 0j, 0j
            for c in cf:
                dv = dv * z[k] + pv
                pv = pv * z[k] + c
            if pv == 0:
                continue
            if dv == 0:
                dv = 1e-30
            ratio = pv / dv
            s = sum(1 / (z[k] - z[j]) for j in range(n) if j != k and z[k] != z[j])
            w = ratio / (1 - ratio * s)
            z[k] -= w
            worst = max(worst, abs(w) / max(1.0, abs(z[k])))
        if worst < 1e-14:
            break
    return z


@op('algebra.complex_roots', kind='roots', bound='degree <= 40, digits <= 50; numeric (Aberth in floats, Newton in Decimal)')
def complex_roots(p, digits=15):
    """All complex roots with multiplicity as (re, im) Decimal pairs with `digits` places after the point, sorted; numeric. Real roots come from the exact
    Sturm isolation and the non-real ones in exact conjugate pairs. complex_roots([1, 0, 1]) -> [(0, -1), (0, 1)]"""
    _check_digits(digits)
    if digits > 50:
        raise NotCovered('digits past 50')
    p = _poly(p)
    _nonzero(p)
    if len(p) > 41:
        raise NotCovered('degree past 40')
    res = []
    zero = Decimal(0).quantize(Decimal(1).scaleb(-digits))
    prec = digits + 12
    unit = Decimal(1).scaleb(-digits)
    for g, mult in _sqf_decomp(p):
        n = len(g) - 1
        eps = Fraction(1, 10 ** (digits + 3))
        reals = []
        for lo, hi, gg, seq in _roots_sqfree(g):
            if lo != hi:
                lo, hi = _refine(gg, seq, lo, hi, eps)
            reals.append(_dec((lo + hi) / 2, digits))
        k = len(reals)
        nonreal = n - k
        found = []
        if nonreal:
            try:
                cf = [float(c) for c in g]
            except OverflowError:
                raise NotCovered('coefficients too large for the numeric start')
            if any(abs(c) > 1e150 for c in cf):
                raise NotCovered('coefficients too large for the numeric start')
            zs = _aberth(cf)
            pos = sorted((w for w in zs if w.imag > 0), key=lambda w: -w.imag)
            if len(pos) < nonreal // 2:
                raise NotCovered('numeric start did not separate the complex roots')
            with localcontext() as ctx:
                ctx.prec = prec + len(str(max(abs(c.numerator) for c in g))) + n
                gd = [Decimal(c.numerator) / Decimal(c.denominator) for c in g]
                for w in pos[:nonreal // 2]:
                    zr, zi = Decimal(w.real), Decimal(w.imag)
                    for it in range(60):
                        (pr, pi), (dr, di) = _cx_horner(gd, (zr, zi), ctx.prec)
                        den = dr * dr + di * di
                        if den == 0:
                            raise NotCovered('derivative vanished while refining a root')
                        sr = (pr * dr + pi * di) / den
                        si = (pi * dr - pr * di) / den
                        zr, zi = zr - sr, zi - si
                        if abs(sr) + abs(si) < unit / 1000:
                            break
                    else:
                        raise NotCovered('Newton refinement did not converge')
                    if zi <= 0:
                        raise NotCovered('refinement left the upper half plane')
                    found.append((zr.quantize(unit, ROUND_HALF_EVEN), zi.quantize(unit, ROUND_HALF_EVEN)))
            for zr, zi in found:
                res += [(zr, zi)] * mult
                res += [(zr, -zi)] * mult
        for r in reals:
            res += [(r, zero)] * mult
    res.sort()
    return res


@op('algebra.quadratic_roots', kind='roots', bound='any size; exact (Surd)')
def quadratic_roots(a, b, c):
    """Both roots of a*x^2 + b*x + c as exact (re, im) pairs of Surd, ascending (by real part, then imaginary part).
    quadratic_roots(1, -2, -1) -> [(1 - sqrt(2), 0), (1 + sqrt(2), 0)]; quadratic_roots(1, 0, 1) -> [(0, -1), (0, 1)]"""
    a, b, c = _q(a), _q(b), _q(c)
    if a == 0:
        raise BadInput('a = 0: not a quadratic')
    D = b * b - 4 * c * a
    if D >= 0:
        s = Surd.sqrt(D)
        r1 = (Surd(-b) - s) / (2 * a)
        r2 = (Surd(-b) + s) / (2 * a)
        lo, hi = (r1, r2) if r1 <= r2 else (r2, r1)
        return [(lo, Surd(0)), (hi, Surd(0))]
    s = Surd.sqrt(-D) / (2 * abs(a))
    return [(Surd(-b / (2 * a)), -s), (Surd(-b / (2 * a)), s)]


def _solve_poly(p):
    p = list(p)
    z = 0
    while len(p) > 1 and p[-1] == 0:
        p.pop()
        z += 1
    r = factor_z(p) if len(p) > 1 else {'content': p[0] if p else Fraction(0), 'factors': []}
    facs = []
    if z:
        facs.append(dict(poly=[Fraction(1), Fraction(0)], multiplicity=z, degree=1, roots=[(Surd(0), Surd(0))], real_intervals=[(Fraction(0), Fraction(0))]))
    for f, m in r['factors']:
        d = len(f) - 1
        ent = dict(poly=f, multiplicity=m, degree=d)
        if d == 1:
            ent['roots'] = [(Surd(-f[1] / f[0]), Surd(0))]
            ent['real_intervals'] = [(-f[1] / f[0], -f[1] / f[0])]
        elif d == 2:
            ent['roots'] = quadratic_roots(*f)
            ent['real_intervals'] = [(lo, hi) for lo, hi, _ in isolate_real_roots(f)]
        else:
            ent['roots'] = None
            ent['real_intervals'] = [(lo, hi) for lo, hi, _ in isolate_real_roots(f)]
        facs.append(ent)
    facs.sort(key=lambda e: (e['degree'], e['poly']))
    return dict(content=r['content'], factors=facs, all_exact=all(e['roots'] is not None for e in facs))


@op('algebra.solve_polynomial', kind='roots', bound='factor_z bounds (degree <= 120)')
def solve_polynomial(p):
    """Solve p(x) = 0 over C: factor over Z; every factor of degree 1 or 2 gets its exact roots (Surd pairs (re, im)); a factor of degree >= 3 gets None and
    its isolating real intervals (use real_roots / complex_roots for decimals). {'content', 'factors': [{'poly', 'multiplicity', 'degree', 'roots', 'real_intervals'}],
    'all_exact': bool}. solve_polynomial([1, 0, -3, 2]) -> factors x - 1 (m=2) and x + 2"""
    p = _poly(p)
    _nonzero(p)
    if len(p) == 1:
        return dict(content=p[0], factors=[], all_exact=True)
    return _solve_poly(p)


@op('algebra.solve_polynomial_str', kind='roots', bound='one variable; as solve_polynomial')
def solve_polynomial_str(eq):
    """Solve an equation written in text, 'lhs = rhs' (a side without '=' means = 0), as solve_polynomial. solve_polynomial_str('x^2 = 2x + 3') -> roots -1 and 3"""
    if eq.count('=') > 1:
        raise BadInput('more than one =')
    lhs, _, rhs = eq.partition('=')
    mp = _mp_add(_parse_multi(lhs), _parse_multi(rhs) if rhs.strip() else {}, -1)
    p = _mp_univariate(mp)[0]
    _nonzero(p, 'equation (every number is a solution)')
    return solve_polynomial(p)


# ---------------------------------------------------------------- resultant, discriminant, symmetric functions

MAX_SYLVESTER = 120        # deg f + deg g past this is refused (Bareiss on the Sylvester matrix, a few seconds at most)


def _resultant(f, g):
    if not f or not g:
        return Fraction(0)
    m, n = len(f) - 1, len(g) - 1
    if m == 0:
        return f[0] ** n
    if n == 0:
        return g[0] ** m
    if m + n > MAX_SYLVESTER:
        raise NotCovered(f'resultant: degrees add up to more than {MAX_SYLVESTER}')
    cf, fi = _int_content(f)           # Res(cf*F, cg*G) = cf^n * cg^m * Res(F, G), F and G integer polynomials (contents are positive)
    cg, gi = _int_content(g)
    N = m + n
    S = [[Fraction(0)] * N for _ in range(N)]
    for i in range(n):
        for j, c in enumerate(fi):
            S[i][i + j] = Fraction(c)
    for i in range(m):
        for j, c in enumerate(gi):
            S[n + i][i + j] = Fraction(c)
    return cf ** n * cg ** m * _det(S)


@op('algebra.resultant', kind='polynomials over Q', bound='degree <= 2000')
def resultant(p, q):
    """The resultant Res(p, q) (determinant of the Sylvester matrix), exact. resultant([1, 0, -1], [1, -2]) -> 3"""
    p, q = _poly(p), _poly(q)
    _nonzero(p)
    _nonzero(q)
    return _resultant(p, q)


@op('algebra.discriminant', kind='polynomials over Q', bound='degree <= 2000')
def discriminant(p):
    """The discriminant, a_n^(2n-2) * prod (r_i - r_j)^2 over i<j; zero exactly when p has a repeated root. discriminant([1, 0, 0, 1]) -> -27 (x^3 + 1)"""
    p = _poly(p)
    _nonzero(p)
    n = len(p) - 1
    if n < 1:
        raise BadInput('a constant has no discriminant')
    d = _resultant(p, _deriv(p)) / p[0]
    return (-1) ** (n * (n - 1) // 2) * d


@op('algebra.vieta', kind='symmetric functions', bound='degree <= 2000')
def vieta(p):
    """The elementary symmetric functions [e1, ..., en] of the roots (with multiplicity, over C): e_k = (-1)^k a_(n-k) / a_n. vieta([1, -6, 11, -6]) -> [6, 11, 6]"""
    p = _poly(p)
    _nonzero(p)
    n = len(p) - 1
    if n < 1:
        raise BadInput('a constant has no roots')
    return [(-1) ** k * p[k] / p[0] for k in range(1, n + 1)]


@op('algebra.power_sums', kind='symmetric functions', bound='k <= 2000, degree <= 2000')
def power_sums(p, k, negative=False):
    """[s_1, ..., s_k] with s_j = sum of r^j over the roots (Newton's identities); negative=True gives sums of r^-j (needs p(0) != 0).
    power_sums([1, -3, 2], 4) -> [3, 5, 9, 17]"""
    p = _poly(p)
    _nonzero(p)
    if not isinstance(k, int) or k < 1 or k > MAX_DEG:
        raise BadInput(f'k must be an integer from 1 to {MAX_DEG}')
    if negative:
        if p[-1] == 0:
            raise BadInput('a zero root has no negative power')
        p = list(reversed(p))
    n = len(p) - 1
    if n < 1:
        raise BadInput('a constant has no roots')
    e = [Fraction(1)] + [(-1) ** j * p[j] / p[0] for j in range(1, n + 1)]
    s = []
    for j in range(1, k + 1):
        t = Fraction(0)
        for i in range(1, min(j - 1, n) + 1):
            t += (-1) ** (i - 1) * e[i] * s[j - i - 1]
        if j <= n:
            t += (-1) ** (j - 1) * j * e[j]
        s.append(t)
    return s


@op('algebra.roots_power_poly', kind='symmetric functions', bound='degree * k <= 2000')
def roots_power_poly(p, k):
    """The monic polynomial whose roots are the k-th powers of the roots of p (k >= 1), by Newton's identities. roots_power_poly([1, 0, -2], 2) -> [1, -4, 4]... for x^2 - 2: roots +-sqrt2, squares 2, 2"""
    p = _poly(p)
    _nonzero(p)
    n = len(p) - 1
    if n < 1:
        raise BadInput('a constant has no roots')
    if not isinstance(k, int) or k < 1 or n * k > MAX_DEG:
        raise BadInput('k must be a positive integer with degree*k <= 2000')
    s = power_sums(p, n * k)
    P = [s[k * j - 1] for j in range(1, n + 1)]
    e = [Fraction(1)]
    for j in range(1, n + 1):
        t = Fraction(0)
        for i in range(1, j + 1):
            t += (-1) ** (i - 1) * e[j - i] * P[i - 1]
        e.append(t / j)
    return [(-1) ** j * e[j] for j in range(n + 1)]


# ---------------------------------------------------------------- interpolation, differences, cyclotomic

def _points(pts):
    if not isinstance(pts, (list, tuple)) or not pts:
        raise BadInput('give a non-empty list of (x, y) points')
    if len(pts) > 400:
        raise NotCovered('more than 400 points')
    xs = [_q(p[0]) for p in pts]
    ys = [_q(p[1]) for p in pts]
    if len(set(xs)) != len(xs):
        raise BadInput('two points share an x')
    return xs, ys


@op('algebra.interpolate', kind='interpolation', bound='<= 400 points')
def interpolate(points):
    """The unique polynomial of degree < len(points) through the points (Newton divided differences, exact). interpolate([(0, 1), (1, 3), (2, 7)]) -> [1, 1, 1]"""
    xs, ys = _points(points)
    n = len(xs)
    dd = list(ys)
    coef = [dd[0]]
    for j in range(1, n):
        dd = [(dd[i + 1] - dd[i]) / (xs[i + j] - xs[i]) for i in range(n - j)]
        coef.append(dd[0])
    p = [coef[-1]]
    for j in range(n - 2, -1, -1):
        p = _add(_mul(p, [Fraction(1), -xs[j]]), [coef[j]] if coef[j] != 0 else [])
    return _trim(p)


@op('algebra.interpolate_eval', kind='interpolation', bound='<= 400 points')
def interpolate_eval(points, x):
    """Value at x of the interpolating polynomial (Lagrange, exact). interpolate_eval([(0, 1), (1, 3), (2, 7)], 3) -> 13"""
    xs, ys = _points(points)
    x = _q(x)
    total = Fraction(0)
    for i, xi in enumerate(xs):
        t = ys[i]
        for j, xj in enumerate(xs):
            if j != i:
                t *= (x - xj) / (xi - xj)
        total += t
    return total


@op('algebra.finite_differences', kind='interpolation', bound='<= 2000 terms')
def finite_differences(seq, order=None):
    """The forward difference rows [seq, delta seq, delta^2 seq, ...] up to `order` (all the way down by default); the first entries give the Newton form.
    finite_differences([1, 4, 9, 16]) -> [[1, 4, 9, 16], [3, 5, 7], [2, 2], [0]]"""
    s = [_q(x) for x in seq]
    if len(s) > MAX_DEG:
        raise NotCovered(f'more than {MAX_DEG} terms')
    rows = [s]
    while len(rows[-1]) > 1 and (order is None or len(rows) <= order):
        r = rows[-1]
        rows.append([r[i + 1] - r[i] for i in range(len(r) - 1)])
    return rows


_CYC_CACHE = {}


def _cyclotomic_int(n):
    if n in _CYC_CACHE:
        return _CYC_CACHE[n]
    num = [1] + [0] * (n - 1) + [-1]
    for d in _divisors(n):
        if d < n:
            num = _idiv_exact(num, _cyclotomic_int(d))
    _CYC_CACHE[n] = num
    return num


@op('algebra.cyclotomic', kind='polynomials over Q', bound='n <= 10000')
def cyclotomic(n):
    """The n-th cyclotomic polynomial Phi_n (integer coefficients, degree phi(n)). cyclotomic(6) -> [1, -1, 1]"""
    if not isinstance(n, int) or isinstance(n, bool) or n < 1:
        raise BadInput('n must be a positive integer')
    if n > 10000:
        raise NotCovered('n past 10000')
    return [Fraction(c) for c in _cyclotomic_int(n)]


# ---------------------------------------------------------------- polynomials modulo a prime

def _prime(p):
    if not isinstance(p, int) or isinstance(p, bool) or p < 2:
        raise BadInput('the modulus must be a prime')
    if not _is_prime(p):
        raise BadInput(f'{p} is not prime')
    return p


def _pm(poly, p):
    """A polynomial argument reduced mod p (denominators must be invertible)."""
    out = []
    for c in _poly(poly):
        if c.denominator % p == 0:
            raise BadInput('a denominator is divisible by the modulus')
        out.append(c.numerator * pow(c.denominator, -1, p) % p)
    return _mt(out)


@op('algebra.polymod_mul', kind='polynomials mod p', bound='p prime < 3.3e24, degree <= 2000')
def polymod_mul(a, b, p):
    """Product in F_p[x] (coefficients in 0..p-1). polymod_mul([1, 1], [1, 1], 2) -> [1, 0, 1]"""
    p = _prime(p)
    return _mmul(_pm(a, p), _pm(b, p), p)


@op('algebra.polymod_divmod', kind='polynomials mod p', bound='p prime < 3.3e24, degree <= 2000')
def polymod_divmod(a, b, p):
    """Quotient and remainder in F_p[x]. polymod_divmod([1, 0, 0, 1], [1, 1], 5) -> ([1, 4, 1], [])"""
    p = _prime(p)
    return _mdivmod(_pm(a, p), _pm(b, p), p)


@op('algebra.polymod_gcd', kind='polynomials mod p', bound='p prime < 3.3e24, degree <= 2000')
def polymod_gcd(a, b, p):
    """Monic gcd in F_p[x]. polymod_gcd([1, 0, 0, 1], [1, 1], 7) -> [1, 1]"""
    p = _prime(p)
    return _mgcd(_pm(a, p), _pm(b, p), p)


@op('algebra.polymod_powmod', kind='polynomials mod p', bound='p prime < 3.3e24, exponent < 10^40')
def polymod_powmod(a, e, f, p):
    """a^e mod f in F_p[x], e >= 0. polymod_powmod([1, 0], 10, [1, 0, 1], 3) -> x^10 mod (x^2+1) = -1 -> [2]"""
    p = _prime(p)
    f = _pm(f, p)
    if not f:
        raise BadInput('modulus polynomial is zero')
    if not isinstance(e, int) or e < 0 or e > 10 ** 40:
        raise BadInput('the exponent must be an integer from 0 to 10^40')
    return _mpowmod(_pm(a, p), e, f, p) if len(f) > 1 else []


@op('algebra.polymod_inverse', kind='polynomials mod p', bound='p prime < 3.3e24, degree <= 2000')
def polymod_inverse(a, f, p):
    """The inverse of a modulo f in F_p[x] (needs gcd(a, f) = 1), degree < deg f. polymod_inverse([1, 0], [1, 1, 1], 2) -> [1, 1]"""
    p = _prime(p)
    a, f = _pm(a, p), _pm(f, p)
    if len(f) < 2:
        raise BadInput('the modulus polynomial must have degree >= 1')
    s, t, g = _mgcdex(a, f, p)
    if g != [1]:
        raise BadInput('not invertible: gcd(a, f) is not 1')
    return _mdivmod(s, f, p)[1]


@op('algebra.polymod_roots', kind='polynomials mod p', bound='p prime < 3.3e24, degree <= 500')
def polymod_roots(f, p):
    """The roots in F_p with multiplicity [(root, m)], sorted. polymod_roots([1, 0, -1], 7) -> [(1, 1), (6, 1)]"""
    p = _prime(p)
    f = _pm(f, p)
    _nonzero(f)
    if len(f) > 501:
        raise NotCovered('degree past 500')
    out = []
    for g, m in _fp_factor(_mmonic(f, p), p):
        if len(g) == 2:
            out.append(((-g[1]) % p, m))
    out.sort()
    return out


@op('algebra.polymod_factor', kind='polynomials mod p', bound='p prime < 3.3e24, degree <= 500')
def polymod_factor(f, p):
    """Factorisation in F_p[x]: (lead, [(monic irreducible, multiplicity)]) sorted by degree. polymod_factor([1, 0, 1], 5) -> (1, [([1, 2], 1), ([1, 3], 1)])"""
    p = _prime(p)
    f = _pm(f, p)
    _nonzero(f)
    if len(f) > 501:
        raise NotCovered('degree past 500')
    return f[0], _fp_factor(_mmonic(f, p), p)


@op('algebra.polymod_irreducible', kind='polynomials mod p', bound='p prime < 3.3e24, degree <= 500')
def polymod_irreducible(f, p):
    """True when f is irreducible in F_p[x] (degree >= 1). polymod_irreducible([1, 1, 1], 2) -> True"""
    p = _prime(p)
    f = _pm(f, p)
    _nonzero(f)
    if len(f) < 2:
        return False
    fa = _fp_factor(_mmonic(f, p), p)
    return len(fa) == 1 and fa[0][1] == 1


# ---------------------------------------------------------------- matrices over Q

def _mat(A, rect=True):
    if not isinstance(A, (list, tuple)) or not A or not all(isinstance(r, (list, tuple)) for r in A):
        raise BadInput('a matrix is a non-empty list of rows')
    w = len(A[0])
    if w == 0 or any(len(r) != w for r in A):
        raise BadInput('the rows must have the same non-zero length')
    if len(A) > MAX_DIM or w > MAX_DIM:
        raise NotCovered(f'matrix larger than {MAX_DIM}')
    M = [[_q(c) for c in r] for r in A]
    if not rect and len(M) != w:
        raise BadInput('the matrix must be square')
    return M


def _vec(v, n=None):
    if not isinstance(v, (list, tuple)):
        raise BadInput('a vector is a list')
    r = [_q(c) for c in v]
    if n is not None and len(r) != n:
        raise BadInput('vector length does not match the matrix')
    return r


def _ident(n):
    return [[Fraction(int(i == j)) for j in range(n)] for i in range(n)]


def _matmul(A, B):
    if len(A[0]) != len(B):
        raise BadInput('inner dimensions differ')
    Bt = list(zip(*B))
    return [[sum((a * b for a, b in zip(row, col)), Fraction(0)) for col in Bt] for row in A]


def _rref(M):
    M = [list(r) for r in M]
    rows, cols = len(M), len(M[0])
    piv, r = [], 0
    for c in range(cols):
        if r >= rows:
            break
        k = next((i for i in range(r, rows) if M[i][c] != 0), None)
        if k is None:
            continue
        M[r], M[k] = M[k], M[r]
        inv = 1 / M[r][c]
        M[r] = [x * inv for x in M[r]]
        for i in range(rows):
            if i != r and M[i][c] != 0:
                f = M[i][c]
                M[i] = [x - f * y for x, y in zip(M[i], M[r])]
        piv.append(c)
        r += 1
    return M, piv


def _nullspace(M):
    R, piv = _rref(M)
    cols = len(M[0])
    free = [c for c in range(cols) if c not in piv]
    basis = []
    for f in free:
        v = [Fraction(0)] * cols
        v[f] = Fraction(1)
        for i, pc in enumerate(piv):
            v[pc] = -R[i][f]
        basis.append(v)
    return basis


def _det(M):
    n = len(M)
    scale = Fraction(1)
    I = []
    for r in M:
        den = 1
        for c in r:
            den = den * c.denominator // gcd(den, c.denominator)
        scale *= den
        I.append([int(c * den) for c in r])
    sign, prev = 1, 1
    for k in range(n - 1):
        if I[k][k] == 0:
            sw = next((i for i in range(k + 1, n) if I[i][k] != 0), None)
            if sw is None:
                return Fraction(0)
            I[k], I[sw] = I[sw], I[k]
            sign = -sign
        for i in range(k + 1, n):
            for j in range(k + 1, n):
                I[i][j] = (I[i][j] * I[k][k] - I[i][k] * I[k][j]) // prev
        prev = I[k][k]
    return Fraction(sign * I[n - 1][n - 1]) / scale


def _charpoly(A):
    """Faddeev-LeVerrier: [1, c_(n-1), ..., c_0]."""
    n = len(A)
    c = [Fraction(1)]
    Mk = _ident(n)       # M_1 = I
    for k in range(1, n + 1):
        AM = _matmul(A, Mk)
        ck = -sum((AM[i][i] for i in range(n)), Fraction(0)) / k
        c.append(ck)
        Mk = [[AM[i][j] + (ck if i == j else 0) for j in range(n)] for i in range(n)]
    return c


@op('algebra.mat_mul', kind='matrices over Q', bound=f'sides <= {MAX_DIM}')
def mat_mul(A, B):
    """Matrix product. mat_mul([[1, 2], [3, 4]], [[0, 1], [1, 0]]) -> [[2, 1], [4, 3]]"""
    return _matmul(_mat(A), _mat(B))


@op('algebra.mat_trace', kind='matrices over Q', bound=f'sides <= {MAX_DIM}')
def mat_trace(A):
    """Trace of a square matrix. mat_trace([[1, 2], [3, 4]]) -> 5"""
    A = _mat(A, rect=False)
    return sum((A[i][i] for i in range(len(A))), Fraction(0))


@op('algebra.mat_det', kind='matrices over Q', bound=f'square, sides <= {MAX_DIM}; Bareiss on integers')
def mat_det(A):
    """Determinant (Bareiss, fraction-free). mat_det([[1, 2], [3, 4]]) -> -2"""
    return _det(_mat(A, rect=False))


@op('algebra.mat_inverse', kind='matrices over Q', bound=f'square, sides <= {MAX_DIM}')
def mat_inverse(A):
    """Inverse by Gauss-Jordan; BadInput when singular. mat_inverse([[1, 2], [3, 4]]) -> [[-2, 1], [3/2, -1/2]]"""
    A = _mat(A, rect=False)
    n = len(A)
    R, piv = _rref([r + I for r, I in zip(A, _ident(n))])
    if piv[:n] != list(range(n)) or len(piv) < n:
        raise BadInput('the matrix is singular')
    return [r[n:] for r in R]


@op('algebra.mat_rank', kind='matrices over Q', bound=f'sides <= {MAX_DIM}')
def mat_rank(A):
    """Rank. mat_rank([[1, 2], [2, 4]]) -> 1"""
    return len(_rref(_mat(A))[1])


@op('algebra.mat_rref', kind='matrices over Q', bound=f'sides <= {MAX_DIM}')
def mat_rref(A):
    """(reduced row echelon form, pivot columns). mat_rref([[1, 2], [3, 4]]) -> ([[1, 0], [0, 1]], [0, 1])"""
    R, piv = _rref(_mat(A))
    return R, piv


@op('algebra.mat_nullspace', kind='matrices over Q', bound=f'sides <= {MAX_DIM}')
def mat_nullspace(A):
    """A basis of {x : A x = 0} (the reduced-form basis, one vector per free column). mat_nullspace([[1, 2], [2, 4]]) -> [[-2, 1]]"""
    return _nullspace(_mat(A))


@op('algebra.mat_solve', kind='linear systems', bound=f'sides <= {MAX_DIM}')
def mat_solve(A, b):
    """Solve A x = b exactly: {'status': 'unique' | 'infinite' | 'none', 'particular': x0 or None, 'nullspace': basis}; every solution is x0 + sum t_i n_i.
    mat_solve([[1, 1], [1, -1]], [3, 1]) -> {'status': 'unique', 'particular': [2, 1], 'nullspace': []}"""
    A = _mat(A)
    b = _vec(b, len(A))
    n = len(A[0])
    R, piv = _rref([r + [bi] for r, bi in zip(A, b)])
    if n in piv:
        return dict(status='none', particular=None, nullspace=[])
    x = [Fraction(0)] * n
    for i, pc in enumerate(piv):
        x[pc] = R[i][n]
    ns = _nullspace(A)
    return dict(status='unique' if not ns else 'infinite', particular=x, nullspace=ns)


@op('algebra.mat_charpoly', kind='matrices over Q', bound='square, sides <= 30; Faddeev-LeVerrier')
def mat_charpoly(A):
    """det(xI - A) as [1, c_(n-1), ..., c_0]. mat_charpoly([[1, 2], [3, 4]]) -> [1, -5, -2]"""
    A = _mat(A, rect=False)
    if len(A) > 30:
        raise NotCovered('matrix larger than 30 for the characteristic polynomial')
    return _charpoly(A)


@op('algebra.mat_minpoly', kind='matrices over Q', bound='square, sides <= 20; first dependency among I, A, A^2, ...')
def mat_minpoly(A):
    """The minimal polynomial (monic, lowest degree with p(A) = 0). mat_minpoly([[2, 0], [0, 2]]) -> [1, -2]"""
    A = _mat(A, rect=False)
    n = len(A)
    if n > 20:
        raise NotCovered('matrix larger than 20 for the minimal polynomial')
    basis = []                           # (pivot index, reduced vector, combination of A^0..A^j)
    Pk = _ident(n)
    k = 0
    while True:
        v = [x for r in Pk for x in r]
        comb = [Fraction(0)] * (k + 1)
        comb[k] = Fraction(1)
        for pi, bv, bc in basis:
            if v[pi] != 0:
                f = v[pi]
                v = [x - f * y for x, y in zip(v, bv)]
                comb = [x - f * (bc[i] if i < len(bc) else 0) for i, x in enumerate(comb)]
        pi = next((i for i, x in enumerate(v) if x != 0), None)
        if pi is None:
            return list(reversed(comb))
        f = v[pi]
        basis.append((pi, [x / f for x in v], [x / f for x in comb]))
        Pk = _matmul(A, Pk)
        k += 1


@op('algebra.mat_eigen', kind='matrices over Q', bound='square, sides <= 30')
def mat_eigen(A):
    """Rational eigenvalues: {'eigenvalues': [{'value', 'algebraic', 'geometric', 'eigenvectors'}], 'rest': characteristic factor left (degree 0 = [1] when all
    eigenvalues are rational), 'diagonalizable_over_Q': bool}. mat_eigen([[2, 0], [0, 3]]) -> eigenvalues 2 and 3, rest [1]"""
    A = _mat(A, rect=False)
    n = len(A)
    if n > 30:
        raise NotCovered('matrix larger than 30')
    cp = _charpoly(A)
    ev, rest = [], list(cp)
    for lam, m in _rational_roots(cp):
        N = _nullspace([[A[i][j] - (lam if i == j else 0) for j in range(n)] for i in range(n)])
        ev.append(dict(value=lam, algebraic=m, geometric=len(N), eigenvectors=N))
        for _ in range(m):
            rest = _divmod(rest, [Fraction(1), -lam])[0]
    return dict(eigenvalues=ev, rest=rest, diagonalizable_over_Q=len(rest) == 1 and all(e['algebraic'] == e['geometric'] for e in ev))


@op('algebra.mat_pow', kind='matrices over Q', bound='square, sides <= 30; result entries <= 200000 digits; mod allows any exponent')
def mat_pow(A, n, mod=None):
    """A^n by repeated squaring (negative n uses the inverse). With mod = m the entries are integers reduced mod m (n may be huge).
    mat_pow([[1, 1], [1, 0]], 10) -> [[89, 55], [55, 34]]"""
    A = _mat(A, rect=False)
    if len(A) > 30:
        raise NotCovered('matrix larger than 30')
    if not isinstance(n, int) or isinstance(n, bool):
        raise BadInput('the exponent must be an integer')
    if mod is not None:
        if not isinstance(mod, int) or mod < 1:
            raise BadInput('mod must be a positive integer')
        if any(c.denominator != 1 for r in A for c in r):
            raise BadInput('mod needs an integer matrix')
        if n < 0:
            raise BadInput('mod with a negative exponent is not supported')
        I = [[int(c) % mod for c in r] for r in A]
        R = [[int(i == j) % mod for j in range(len(A))] for i in range(len(A))]
        mm = lambda X, Y: [[sum(a * b for a, b in zip(row, col)) % mod for col in zip(*Y)] for row in X]
        while n:
            if n & 1:
                R = mm(R, I)
            n >>= 1
            if n:
                I = mm(I, I)
        return [[Fraction(c) for c in r] for r in R]
    if n < 0:
        A = mat_inverse(A)
        n = -n
    if n > 10 ** 7:
        raise NotCovered('exponent past 10^7 without mod')
    R = _ident(len(A))
    while n:
        if n & 1:
            R = _matmul(R, A)
        n >>= 1
        if n:
            A = _matmul(A, A)
        big = max(max(abs(c.numerator).bit_length(), c.denominator.bit_length()) for r in A for c in r)
        if big > 700000:
            raise NotCovered('the result is too large (over 200000 digits)')
    return R


@op('algebra.mat_lu', kind='matrices over Q', bound=f'sides <= {MAX_DIM}')
def mat_lu(A):
    """P A = L U with row exchanges only when a pivot is zero: {'P': permutation matrix, 'L': unit lower triangular, 'U': echelon form}. Any shape.
    mat_lu([[0, 1], [1, 0]]) -> P = [[0, 1], [1, 0]], L = I, U = I"""
    A = _mat(A)
    m, n = len(A), len(A[0])
    U = [list(r) for r in A]
    L = [[Fraction(0)] * m for _ in range(m)]
    perm = list(range(m))
    r = 0
    for c in range(n):
        if r >= m:
            break
        k = next((i for i in range(r, m) if U[i][c] != 0), None)
        if k is None:
            continue
        if k != r:
            U[r], U[k] = U[k], U[r]
            L[r], L[k] = L[k], L[r]
            perm[r], perm[k] = perm[k], perm[r]
        for i in range(r + 1, m):
            if U[i][c] != 0:
                f = U[i][c] / U[r][c]
                L[i][r] = f
                U[i] = [x - f * y for x, y in zip(U[i], U[r])]
        r += 1
    for i in range(m):
        L[i][i] = Fraction(1)
    P = [[Fraction(int(perm[i] == j)) for j in range(m)] for i in range(m)]
    return dict(P=P, L=L, U=U)


# ---------------------------------------------------------------- linear systems in text

def _linear_system(eqs):
    if not isinstance(eqs, (list, tuple)) or not eqs:
        raise BadInput('give a list of equations')
    if len(eqs) > MAX_DIM:
        raise NotCovered(f'more than {MAX_DIM} equations')
    forms = []
    for e in eqs:
        if e.count('=') != 1:
            raise BadInput(f'an equation needs exactly one =: {e!r}')
        l, r = e.split('=')
        mp = _mp_add(_parse_multi(l), _parse_multi(r), -1)
        for m in mp:
            if sum(x for _, x in m) > 1:
                raise NotCovered(f'not linear: {e!r}')
        forms.append(mp)
    vs = sorted({v for mp in forms for m in mp for v, _ in m})
    if len(vs) > MAX_DIM:
        raise NotCovered(f'more than {MAX_DIM} variables')
    A = [[mp.get(((v, 1),), Fraction(0)) for v in vs] for mp in forms]
    b = [-mp.get((), Fraction(0)) for mp in forms]
    return vs, A, b


@op('algebra.solve_linear_str', kind='linear systems', bound=f'<= {MAX_DIM} equations, linear only (brackets are expanded)')
def solve_linear_str(eqs):
    """Solve linear equations written in text. {'variables': [...], 'status': 'unique'|'infinite'|'none', 'solution': {var: value} (the particular
    solution; free variables are 0), 'free': [vars that are free], 'nullspace': [{var: coefficient}]}.
    solve_linear_str(['x + y = 3', 'x - y = 1']) -> {'x': 2, 'y': 1}"""
    vs, A, b = _linear_system(eqs)
    if not vs:
        raise BadInput('no variable in the system')
    r = mat_solve(A, b)
    if r['status'] == 'none':
        return dict(variables=vs, status='none', solution=None, free=[], nullspace=[])
    R, piv = _rref(A)
    free = [vs[c] for c in range(len(vs)) if c not in piv]
    return dict(variables=vs, status=r['status'], solution=dict(zip(vs, r['particular'])), free=free,
                nullspace=[dict(zip(vs, v)) for v in r['nullspace']])


# ---------------------------------------------------------------- linear recurrences

def _recurrence_args(coeffs, initial):
    c = [_q(x) for x in coeffs]
    a = [_q(x) for x in initial]
    if not c:
        raise BadInput('give at least one coefficient')
    if len(c) > 30:
        raise NotCovered('order past 30')
    if len(a) < len(c):
        raise BadInput('need as many initial terms as coefficients')
    return c, a


@op('algebra.recurrence_char_poly', kind='linear recurrences', bound='order <= 30')
def recurrence_char_poly(coeffs):
    """x^k - c1 x^(k-1) - ... - ck for a_n = c1 a_(n-1) + ... + ck a_(n-k). recurrence_char_poly([1, 1]) -> [1, -1, -1]"""
    c = [_q(x) for x in coeffs]
    if not c:
        raise BadInput('give at least one coefficient')
    return [Fraction(1)] + [-x for x in c]


@op('algebra.recurrence_nth', kind='linear recurrences', bound='n <= 10^5 exact, any n with mod (< 10^40); order <= 30')
def recurrence_nth(coeffs, initial, n, mod=None, const=0):
    """a_n for a_m = c1 a_(m-1) + ... + ck a_(m-k) + const, given initial = [a_0, ..., a_(k-1)] (Kitamasa: x^n mod the characteristic polynomial).
    With mod = m the terms are integers and the result is reduced mod m. recurrence_nth([1, 1], [0, 1], 10) -> 55"""
    c, a = _recurrence_args(coeffs, initial)
    const = _q(const)
    if not isinstance(n, int) or isinstance(n, bool) or n < 0:
        raise BadInput('n must be a non-negative integer')
    k = len(c)
    a = a[:k]
    if n < k:
        v = a[n]
        return v if mod is None else int(v) % mod
    q = [Fraction(1)] + [-x for x in c]
    seq = list(a)
    if const != 0:
        for m in range(k, k + 1):
            seq.append(sum(c[i] * seq[m - 1 - i] for i in range(k)) + const)
        q = _mul(q, [Fraction(1), Fraction(-1)])
        k += 1
    if mod is None:
        if n > 10 ** 5:
            raise NotCovered('n past 10^5 without mod')
        base, e = [Fraction(1), Fraction(0)], n
        r = [Fraction(1)]
        b = base
        while e:
            if e & 1:
                r = _divmod(_mul(r, b), q)[1]
            e >>= 1
            if e:
                b = _divmod(_mul(b, b), q)[1]
                if b and b[0].numerator.bit_length() > 1500000:
                    raise NotCovered('the terms are too large')
        coef = [Fraction(0)] * (k - len(r)) + r
        return sum((coef[k - 1 - i] * seq[i] for i in range(k)), Fraction(0))
    if not isinstance(mod, int) or mod < 1:
        raise BadInput('mod must be a positive integer')
    if n > 10 ** 40:
        raise NotCovered('n past 10^40')
    for x in list(c) + seq + [const]:
        if x.denominator != 1:
            raise BadInput('mod needs integer coefficients and terms')
    qm = [int(x) % mod for x in q]
    r, b, e = [1 % mod], [1 % mod, 0], n
    if mod == 1:
        return 0
    while e:
        if e & 1:
            r = _mdivmod(_mmul(r, b, mod), qm, mod)[1]
        e >>= 1
        if e:
            b = _mdivmod(_mmul(b, b, mod), qm, mod)[1]
    coef = [0] * (k - len(r)) + r
    return sum(coef[k - 1 - i] * int(seq[i]) for i in range(k)) % mod


def _solve_field(A, b):
    """Gauss elimination for a square invertible system over Fractions or Surds; returns x."""
    n = len(A)
    M = [list(r) + [bi] for r, bi in zip(A, b)]
    for c in range(n):
        k = next((i for i in range(c, n) if not (M[i][c] == 0)), None)
        if k is None:
            raise NotCovered('singular system in the closed form')
        M[c], M[k] = M[k], M[c]
        pv = M[c][c]
        M[c] = [x / pv for x in M[c]]
        for i in range(n):
            if i != c and not (M[i][c] == 0):
                f = M[i][c]
                M[i] = [x - f * y for x, y in zip(M[i], M[c])]
    return [M[i][n] for i in range(n)]


def _close(x):
    """A Surd that is rational becomes a Fraction."""
    if isinstance(x, Surd) and x.is_rational():
        return x.p
    return x


@op('algebra.recurrence_closed_form', kind='linear recurrences', bound='order <= 12; rational roots (with multiplicity) and at most one real quadratic pair')
def recurrence_closed_form(coeffs, initial):
    """a_n = sum over roots r (multiplicity m) of (c_0 + c_1 n + ... + c_(m-1) n^(m-1)) * r^n for n >= valid_from (valid_from is the multiplicity of the root 0).
    {'valid_from': z, 'terms': [{'root', 'multiplicity', 'coeffs': [c_0, ...]}]}; roots and coefficients are Fractions or Surd.
    recurrence_closed_form([1, 1], [0, 1]) -> roots (1 +- sqrt5)/2 with coefficients +-sqrt5/5 (Binet)"""
    c, a = _recurrence_args(coeffs, initial)
    k = len(c)
    if k > 12:
        raise NotCovered('order past 12')
    q = [Fraction(1)] + [-x for x in c]
    z = 0
    while q[-1] == 0:
        q.pop()
        z += 1
    if len(q) == 1:
        return dict(valid_from=z, terms=[])
    a = a[:k]
    seq = list(a)
    need = z + len(q) - 1
    while len(seq) < need:
        m = len(seq)
        seq.append(sum(c[i] * seq[m - 1 - i] for i in range(k)))
    fac = factor_z(q)
    roots = []          # (root, multiplicity)
    quad = 0
    for f, m in fac['factors']:
        d = len(f) - 1
        if d == 1:
            roots.append((-f[1] / f[0], m))
        elif d == 2 and m == 1 and f[1] * f[1] - 4 * f[0] * f[2] > 0:
            quad += 1
            (r1, _), (r2, _) = quadratic_roots(*f)
            roots += [(r1, 1), (r2, 1)]
        else:
            raise NotCovered('closed form needs rational roots or one real quadratic pair')
    if quad > 1:
        raise NotCovered('closed form with several irrational pairs')
    cols = []
    for r, m in roots:
        for j in range(m):
            cols.append((r, j))
    u = len(cols)

    def power(r, nn):
        if isinstance(r, Surd):
            out = Surd(1)
            for _ in range(nn):
                out = out * r
            return out
        return r ** nn
    A = []
    for nn in range(z, z + u):
        A.append([power(r, nn) * (nn ** j) for r, j in cols])
    x = _solve_field(A, [seq[nn] for nn in range(z, z + u)])
    terms, i = [], 0
    for r, m in roots:
        terms.append(dict(root=_close(r), multiplicity=m, coeffs=[_close(v) for v in x[i:i + m]]))
        i += m
    return dict(valid_from=z, terms=terms)


@op('algebra.recurrence_guess', kind='linear recurrences', bound='<= 400 terms; needs at least 2*order + 1 terms')
def recurrence_guess(seq):
    """The shortest linear recurrence with constant rational coefficients that generates all the terms (Berlekamp-Massey over Q):
    {'order': L, 'coeffs': [c1..cL], 'checked_terms': N - 2L}; NotCovered if the terms are too few to fix it (N < 2L + 1).
    recurrence_guess([1, 1, 2, 3, 5, 8, 13]) -> order 2, coeffs [1, 1]"""
    s = [_q(x) for x in seq]
    N = len(s)
    if N > 400:
        raise NotCovered('more than 400 terms')
    C, B = [Fraction(1)], [Fraction(1)]
    L, m, b = 0, 1, Fraction(1)
    for n in range(N):
        C = C + [Fraction(0)] * (L + 1 - len(C))
        d = s[n] + sum(C[i] * s[n - i] for i in range(1, L + 1))
        if d == 0:
            m += 1
        elif 2 * L <= n:
            T = list(C)
            f = d / b
            C = C + [Fraction(0)] * (len(B) + m - len(C))
            for i, bi in enumerate(B):
                C[i + m] -= f * bi
            L, B, b, m = n + 1 - L, T, d, 1
        else:
            f = d / b
            C = C + [Fraction(0)] * (len(B) + m - len(C))
            for i, bi in enumerate(B):
                C[i + m] -= f * bi
            m += 1
    C = (C + [Fraction(0)] * (L + 1))[:L + 1]
    if N < 2 * L + 1:
        raise NotCovered(f'too few terms to fix a recurrence of order {L}: need {2 * L + 1}, have {N}')
    return dict(order=L, coeffs=[-x for x in C[1:]], checked_terms=N - 2 * L)


def _spow(r, n):
    out, b = Surd(1), r
    while n:
        if n & 1:
            out = out * b
        n >>= 1
        if n:
            b = b * b
    return out


def closed_form_value(form, n):
    """Evaluate a recurrence_closed_form result at n >= valid_from (a helper for tests and the simulation, not an operation)."""
    if n < form['valid_from']:
        raise BadInput('n is below valid_from')
    S = lambda x: x if isinstance(x, Surd) else Surd(x)
    total = Surd(0)
    for t in form['terms']:
        poly = Surd(0)
        for j, cj in enumerate(t['coeffs']):
            poly = poly + S(cj) * (n ** j)
        total = total + poly * _spow(S(t['root']), n)
    return _close(total)
