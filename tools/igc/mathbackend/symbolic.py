"""MIP math backend, module `symbolic`: a small exact expression engine (a CAS-lite) and the operations on it.

What it is. Every expression is kept in one canonical form: a sum of terms `coefficient * product of atom**exponent`, with Fraction coefficients and Fraction
exponents. An atom is a symbol, the constant pi, a function application (log, sin, cos, tan, atan, abs, floor) with a canonical argument, one exp(...) per term,
a prime radical `p**(1/2)`, a sum raised to a power that is not a positive integer (`(x + 1)**-1`), or `(-1)**k` for an integer symbol k. Products of sums are always
multiplied out, so two expressions are equal as functions exactly when their canonical forms are equal, up to the identities the form does not know (sin^2 + cos^2 = 1,
log of a product of symbols, ...). The form never applies an identity it does not know to be true on the whole domain: `sqrt(x^2)` becomes `abs(x)`, never `x`.

What it refuses. Everything the engine does not cover raises NotCovered with the reason; nothing is guessed. A "not equal" answer is given only where the form is complete
(polynomials and rational functions of symbols and radicals); with function atoms the answer is "equal" or NotCovered.

Text in, text out. Operations take expression text (plain infix or a LaTeX subset) and return canonical text (which `parse` reads back to the same expression), or
Fraction / Decimal / list / bool. `log` is the natural logarithm. `E` is Euler's number, `e` is an ordinary symbol unless the operation is given `e_is_euler=True`.
In sums, a symbol used in `(-1)^k` is taken to be an integer.

Nothing is copied from another project: the algorithms (Yun square-free decomposition, rational-root search, Gauss elimination over Fractions, truncated Laurent series,
Faulhaber sums, integration by the standard tables and by parts) are written from their textbook statements. Standard library only.
"""
import functools
import re
from decimal import Decimal, InvalidOperation, localcontext, ROUND_FLOOR
from fractions import Fraction
from math import comb, gcd, isqrt

from tools.igc.mathbackend import BadInput, NotCovered, Surd, op

MAX_TERMS = 4000          # terms of one canonical expression
MAX_TOKENS = 4000         # tokens of one input text
MAX_DEPTH = 60            # nesting of one input text
MAX_POWER = 200           # integer power of a sum
MAX_SERIES_ORDER = 40     # Taylor order
MAX_SUM_TERMS = 20000     # a finite sum by evaluation
MAX_COEFF_DIGITS = 4000   # digits in a rational coefficient


# ----------------------------------------------------------------------------------------------------------------------------- atoms and expressions

class Atom:
    """One factor kind. kind: s symbol, k constant (pi), f function (name, arg), e exp (arg), b sum-base (arg), r prime radical (name = the prime), m (-1)**arg."""
    __slots__ = ('kind', 'name', 'arg', 'key', '_h')
    _ORDER = {'s': '1s', 'k': '2k', 'f': '3f', 'e': '4e', 'b': '5b', 'r': '6r', 'm': '7m'}

    def __init__(self, kind, name=None, arg=None):
        self.kind, self.name, self.arg = kind, name, arg
        pre = Atom._ORDER[kind]
        if kind in 'sk':
            self.key = f'{pre}:{name}'
        elif kind == 'f':
            self.key = f'{pre}:{name}({arg.key})'
        elif kind == 'r':
            self.key = f'{pre}:{name}'
        else:
            self.key = f'{pre}({arg.key})'
        self._h = hash(self.key)

    def __hash__(self):
        return self._h

    def __eq__(self, o):
        return isinstance(o, Atom) and self.key == o.key

    def has(self, var):
        if self.kind == 's':
            return self.name == var
        if self.kind in 'kr':
            return False
        return self.arg.has(var)

    def __repr__(self):
        return 'Atom(' + self.key + ')'


def _mono_key(mono):
    return '*'.join(a.key + '^' + str(x) for a, x in mono)


def _sorted_mono(d):
    return tuple(sorted(d.items(), key=lambda it: it[0].key))


class Expr:
    """Canonical sum of terms: t maps a monomial (tuple of (Atom, Fraction exponent), sorted by atom key) to a non-zero Fraction coefficient."""
    __slots__ = ('t', '_key', '_h')

    def __init__(self, t):
        if len(t) > MAX_TERMS:
            raise NotCovered('too large')
        self.t, self._key, self._h = t, None, None

    # -- identity
    @property
    def key(self):
        if self._key is None:
            self._key = '+'.join(sorted(f'{c}@{_mono_key(m)}' for m, c in self.t.items())) if self.t else '0'
        return self._key

    def __hash__(self):
        if self._h is None:
            self._h = hash(self.key)
        return self._h

    def __eq__(self, o):
        if isinstance(o, (int, Fraction)):
            o = _const(o)
        return isinstance(o, Expr) and self.t == o.t

    def __bool__(self):
        return bool(self.t)

    # -- structure
    def is_zero(self):
        return not self.t

    def as_fraction(self):
        """The value when this is a rational number, else None."""
        if not self.t:
            return Fraction(0)
        if len(self.t) == 1 and () in self.t:
            return self.t[()]
        return None

    def has(self, var):
        return any(a.has(var) for m in self.t for a, _ in m)

    def symbols(self):
        out = set()
        for m in self.t:
            for a, _ in m:
                if a.kind == 's':
                    out.add(a.name)
                elif a.kind in 'febm':
                    out |= a.arg.symbols()
        return out

    def atoms(self):
        out = set()
        for m in self.t:
            for a, _ in m:
                out.add(a)
                if a.kind in 'febm':
                    out |= a.arg.atoms()
        return out

    # -- arithmetic
    def __neg__(self):
        return Expr({m: -c for m, c in self.t.items()})

    def __add__(self, o):
        o = _wrap(o)
        if o is NotImplemented:
            return o
        if not o.t:
            return self
        if not self.t:
            return o
        d = dict(self.t)
        for m, c in o.t.items():
            v = d.get(m, 0) + c
            if v:
                d[m] = v
            else:
                d.pop(m, None)
        return Expr(d)
    __radd__ = __add__

    def __sub__(self, o):
        o = _wrap(o)
        return o if o is NotImplemented else self + (-o)

    def __rsub__(self, o):
        o = _wrap(o)
        return o if o is NotImplemented else o + (-self)

    def __mul__(self, o):
        o = _wrap(o)
        if o is NotImplemented:
            return o
        if not self.t or not o.t:
            return ZERO
        a, b = self.t, o.t
        if len(b) == 1 and () in b:
            c = b[()]
            return Expr({m: v * c for m, v in a.items()})
        if len(a) == 1 and () in a:
            c = a[()]
            return Expr({m: v * c for m, v in b.items()})
        if len(a) * len(b) > 1_500_000:
            raise NotCovered('too large')
        out = {}
        for m1, c1 in a.items():
            for m2, c2 in b.items():
                _acc(out, _mono_product(m1, c1, m2, c2))
                if len(out) > 8 * MAX_TERMS:
                    raise NotCovered('too large')
        return Expr(out)
    __rmul__ = __mul__

    def __truediv__(self, o):
        o = _wrap(o)
        if o is NotImplemented:
            return o
        return self * _pow(o, Fraction(-1))

    def __rtruediv__(self, o):
        o = _wrap(o)
        return o if o is NotImplemented else o * _pow(self, Fraction(-1))

    def __pow__(self, e):
        return _pow(self, e)

    def __repr__(self):
        return 'Expr(' + to_text(self) + ')'

    def __str__(self):
        return to_text(self)


def _wrap(o):
    if isinstance(o, Expr):
        return o
    if isinstance(o, (int, Fraction)):
        return _const(o)
    return NotImplemented


def _const(q):
    q = Fraction(q)
    if abs(q.numerator) > 10 ** MAX_COEFF_DIGITS or q.denominator > 10 ** MAX_COEFF_DIGITS:
        raise NotCovered('too large')
    return Expr({(): q}) if q else ZERO


ZERO = Expr({})
ONE = Expr({(): Fraction(1)})


def _acc(out, e):
    for m, c in e.t.items():
        v = out.get(m, 0) + c
        if v:
            out[m] = v
        else:
            out.pop(m, None)


def _atom_expr(atom, ex=1):
    return Expr({((atom, Fraction(ex)),): Fraction(1)})


def sym(name):
    return _atom_expr(Atom('s', name))


PI = _atom_expr(Atom('k', 'pi'))
_SIMPLE = set('sk')


def _mono_product(m1, c1, m2, c2):
    """The Expr of (c1*m1)*(c2*m2)."""
    raw = {}
    simple = True
    for a, x in m1:
        raw[a] = x
        if a.kind not in _SIMPLE and not (a.kind == 'f' and a.name != 'abs'):
            simple = False
    for a, x in m2:
        if a.kind not in _SIMPLE and not (a.kind == 'f' and a.name != 'abs'):
            simple = False
        v = raw.get(a, 0) + x
        if v:
            raw[a] = v
        else:
            raw.pop(a, None)
    c = c1 * c2
    if simple:
        return Expr({_sorted_mono(raw): c})
    return _build(c, raw)


def _small_primes_factor(n):
    """{prime: exponent} of a positive integer by trial division; NotCovered when a large cofactor is left."""
    out = {}
    if n < 1:
        raise BadInput('factor of a non-positive number')
    for p in (2, 3):
        while n % p == 0:
            n //= p
            out[p] = out.get(p, 0) + 1
    p = 5
    step = 2
    while p * p <= n and p <= 1_000_000:
        while n % p == 0:
            n //= p
            out[p] = out.get(p, 0) + 1
        p += step
        step = 6 - step
    if n > 1:
        if n >= 10 ** 12 and p * p <= n:
            raise NotCovered('number too large to factor')
        out[n] = out.get(n, 0) + 1
    return out


def _build(coef, raw):
    """The canonical Expr for coef * prod(atom**x for atom, x in raw), applying the merging rules of exp, radicals, sums, abs and (-1)**k."""
    mono = {}
    extra = []
    earg = None
    marg = None
    e_seen = 0
    e_single = None
    m_seen = 0
    m_single = None
    for a, x in raw.items():
        if x == 0:
            continue
        k = a.kind
        if k in 'sk':
            mono[a] = x
        elif k == 'f':
            if a.name == 'abs' and x.denominator == 1 and x.numerator % 2 == 0:
                extra.append(_pow(a.arg, x))          # abs(u)**(2m) = u**(2m)
            else:
                mono[a] = x
        elif k == 'e':
            if x == 1 and not e_seen:
                e_single = a                       # an exp atom already made by _exp is canonical as it stands
            else:
                e_single = None
            e_seen += 1
            earg = a.arg * x if earg is None else earg + a.arg * x
        elif k == 'r':
            n = x.numerator // x.denominator
            f = x - n
            if n:
                if abs(n) > 10 ** 4:
                    raise NotCovered('too large')
                coef = coef * Fraction(a.name) ** n
            if f:
                mono[a] = f
        elif k == 'b':
            if x.denominator == 1 and x > 0:
                extra.append(_pow(a.arg, x))
            elif x > 1:                                   # B**(3/2) = B * B**(1/2)
                n = x.numerator // x.denominator
                extra.append(_pow(a.arg, n))
                mono[a] = x - n
            else:
                mono[a] = x
        elif k == 'm':
            if x.denominator != 1:
                raise NotCovered('power of (-1)^k that is not an integer power')
            if x == 1 and not m_seen:
                m_single = a
            else:
                m_single = None
            m_seen += 1
            marg = a.arg * x if marg is None else marg + a.arg * x
    if marg is not None and m_seen == 1 and m_single is not None:
        mono[m_single] = Fraction(1)                     # one (-1)^k atom made by _sign_pow is canonical as it stands
        marg = None
    res = Expr({_sorted_mono(mono): coef}) if coef else ZERO
    if marg is not None:
        res = res * _sign_pow(marg)
    if earg is not None:
        if e_seen == 1 and e_single is not None:
            res = Expr({_sorted_mono({**dict(m), e_single: Fraction(1)}): c for m, c in res.t.items()}) if res.t else res
        else:
            res = res * _exp(earg)
    for ex in extra:
        res = res * ex
    return res


# ----------------------------------------------------------------------------------------------------------------------------- powers and functions

def _num_pow(c, e):
    """c**e for a non-zero Fraction c and Fraction e, as an Expr (prime radicals when e is not an integer)."""
    if e.denominator == 1:
        n = int(e)
        if abs(n) > 5000 and abs(c) != 1:
            raise NotCovered('too large')
        r = c ** n
        return _const(r)
    if c < 0:
        raise NotCovered('negative number to a fractional power (complex)')
    raw = {}
    for part, sgn in ((c.numerator, 1), (c.denominator, -1)):
        if part != 1:
            for p, k in _small_primes_factor(part).items():
                raw[Atom('r', p)] = raw.get(Atom('r', p), 0) + sgn * k * e
    return _build(Fraction(1), raw)


def _primitive(base):
    """(c, sign-normalised primitive part) with base = c * part, c a positive Fraction, for a sum."""
    cs = [c for c in base.t.values()]
    den = 1
    for c in cs:
        den = den * c.denominator // gcd(den, c.denominator)
    g = 0
    for c in cs:
        g = gcd(g, int(c * den))
    content = Fraction(g, den)
    return content, Expr({m: c / content for m, c in base.t.items()})


def _lead_negative(e):
    first = min(e.t.items(), key=_term_order)[0]
    return e.t[first] < 0


def _pow(base, e):
    base = _wrap(base)
    if isinstance(e, Expr):
        q = e.as_fraction()
        if q is None:
            return _pow_symbolic(base, e)
        e = q
    e = Fraction(e)
    if e == 0:
        if not base.t:
            raise BadInput('0^0')
        return ONE
    if not base.t:
        if e < 0:
            raise BadInput('division by zero')
        return ZERO
    if e == 1:
        return base
    if len(base.t) == 1:
        (mono, c), = base.t.items()
        res = _num_pow(c, e) if c != 1 else ONE
        raw = {}
        integer = e.denominator == 1
        wrapped = []
        if integer:
            for a, x in mono:
                raw[a] = x * e
        else:
            plain = [a for a, _ in mono if a.kind in 'sfb']
            if len(plain) > 1:
                # (x*y)**(1/2) does not split into x**(1/2) * y**(1/2) on the whole domain: keep the product as one base
                inner = Expr({mono: Fraction(1)})
                return res * _build(Fraction(1), {Atom('b', arg=inner): e})
            for a, x in mono:
                if a.kind in 'sfb' and x.denominator == 1 and x.numerator % 2 == 0:
                    ab = fn('abs', a.arg if a.kind == 'b' else _atom_expr(a))          # (x**2)**(1/2) = abs(x)
                    wrapped.append(_pow(ab, x * e))
                elif a.kind == 'm':
                    raise NotCovered('fractional power of (-1)^k')
                else:
                    raw[a] = x * e
        res = res * _build(Fraction(1), raw)
        for w in wrapped:
            res = res * w
        return res
    # a sum
    if e.denominator == 1:
        n = int(e)
        if n > 0:
            if n > MAX_POWER:
                raise NotCovered('too large')
            result, sq, k = ONE, base, n
            while k:
                if k & 1:
                    result = result * sq
                k >>= 1
                if k:
                    sq = sq * sq
            return result
        content, part = _primitive(base)
        sign = 1
        if _lead_negative(part):
            part, sign = -part, -1
        if n < -MAX_POWER:
            raise NotCovered('too large')
        coef = (sign * content) ** n
        return _const(coef) * _atom_expr(Atom('b', arg=part), n)
    content, part = _primitive(base)
    return _num_pow(content, e) * _build(Fraction(1), {Atom('b', arg=part): e})


def _is_integer_combination(e):
    """True when e is a sum of integer multiples of products of symbols (the exponent of (-1)^k must be an integer)."""
    for m, c in e.t.items():
        if c.denominator != 1:
            return False
        for a, x in m:
            if a.kind != 's' or x.denominator != 1 or x < 0:
                return False
    return True


def _sign_pow(arg):
    """(-1)**arg as an Expr; arg must be an integer combination of integer symbols."""
    if not arg.t:
        return ONE
    if not _is_integer_combination(arg):
        raise NotCovered('(-1) to a power that is not an integer')
    keep = {m: Fraction(1) for m, c in arg.t.items() if m != () and int(c) % 2}
    sgn = -1 if int(arg.t.get((), Fraction(0))) % 2 else 1
    if not keep:
        return _const(sgn)
    return _const(sgn) * _atom_expr(Atom('m', arg=Expr(keep)))


def _pow_symbolic(base, e):
    """base ** e with a non-numeric exponent: a positive constant base becomes exp(e*log(base)); base -1 becomes (-1)**e."""
    q = base.as_fraction()
    if q is None:
        if base.symbols():
            raise NotCovered('symbolic exponent needs a constant base')
        if _const_sign(base) <= 0:
            raise NotCovered('symbolic exponent of a non-positive base')
        return _exp(e * _log(base))
    if q == 0:
        raise BadInput('0 to a symbolic power')
    if q > 0:
        return _exp(e * _log(base))
    out = _sign_pow(e)
    if q != -1:
        out = out * _exp(e * _log(_const(-q)))
    return out


def _exp(arg):
    arg = _wrap(arg)
    if not arg.t:
        return ONE
    rest = {}
    res = ONE
    for m, c in arg.t.items():
        if len(m) == 1 and m[0][0].kind == 'f' and m[0][0].name == 'log' and m[0][1] == 1:
            res = res * _pow(m[0][0].arg, c)           # exp(c*log(g)) = g**c on g > 0
        else:
            rest[m] = c
    if rest:
        res = res * _atom_expr(Atom('e', arg=Expr(rest)))
    return res


def _log(arg):
    arg = _wrap(arg)
    q = arg.as_fraction()
    if q is not None:
        if q <= 0:
            raise BadInput('log of a non-positive number')
        out = ZERO
        for part, sgn in ((q.numerator, 1), (q.denominator, -1)):
            if part != 1:
                for p, k in _small_primes_factor(part).items():
                    out = out + _atom_expr(Atom('f', 'log', _const(p))) * (sgn * k)
        return out
    if len(arg.t) == 1:
        (mono, c), = arg.t.items()
        if c > 0:
            head = _log(_const(c)) if c != 1 else ZERO
            if len(mono) == 1:
                a, x = mono[0]
                if a.kind == 'e' and x == 1:
                    return head + a.arg                      # log(exp(u)) = u
                if a.kind == 's' and not (x.denominator == 1 and x.numerator % 2 == 0):
                    return head + _atom_expr(Atom('f', 'log', _atom_expr(a))) * x
            if all(a.kind == 'r' for a, _ in mono):
                out = head
                for a, x in mono:
                    out = out + _log(_const(a.name)) * x
                return out
            return head + _atom_expr(Atom('f', 'log', Expr({mono: Fraction(1)})))
    return _atom_expr(Atom('f', 'log', arg))


_PI_ANGLE = {}


def _trig_exact(name, arg):
    """sin/cos/tan of a rational multiple of pi that is a multiple of 30 or 45 degrees, as an Expr; None when not in the table."""
    if len(arg.t) != 1:
        return None
    (mono, c), = arg.t.items()
    if mono != ((Atom('k', 'pi'), Fraction(1)),):
        return None
    deg = c * 180
    if deg.denominator != 1:
        return None
    d = int(deg) % 360
    if d % 30 and d % 45:
        return None
    half, one = _const(Fraction(1, 2)), ONE
    r2, r3 = _num_pow(Fraction(2), Fraction(1, 2)), _num_pow(Fraction(3), Fraction(1, 2))
    s_tab = {0: ZERO, 30: half, 45: r2 * Fraction(1, 2), 60: r3 * Fraction(1, 2), 90: one}

    def s_of(a):
        a %= 360
        if a <= 90:
            return s_tab[a]
        if a <= 180:
            return s_tab[180 - a]
        if a <= 270:
            return -s_tab[a - 180]
        return -s_tab[360 - a]
    sv, cv = s_of(d), s_of(d + 90)
    if name == 'sin':
        return sv
    if name == 'cos':
        return cv
    if cv.is_zero():
        return None
    return sv * _pow(cv, Fraction(-1))


def _odd_normal(arg):
    """(arg or -arg, sign) with the first term (in canonical order) positive."""
    if not arg.t:
        return arg, 1
    return (-arg, -1) if _lead_negative(arg) else (arg, 1)


def fn(name, arg):
    """The canonical Expr of the function `name` applied to the Expr `arg`."""
    arg = _wrap(arg)
    if name == 'sqrt':
        return _pow(arg, Fraction(1, 2))
    if name == 'exp':
        return _exp(arg)
    if name == 'log':
        return _log(arg)
    if name in ('sin', 'cos', 'tan'):
        if not arg.t:
            return ONE if name == 'cos' else ZERO
        ex = _trig_exact(name, arg)
        if ex is not None:
            return ex
        if name == 'cos':
            a, _ = _odd_normal(arg)
            return _atom_expr(Atom('f', 'cos', a))
        a, s = _odd_normal(arg)
        return _atom_expr(Atom('f', name, a)) * s
    if name == 'atan':
        q = arg.as_fraction()
        if q == 0:
            return ZERO
        if q == 1:
            return PI * Fraction(1, 4)
        if q == -1:
            return PI * Fraction(-1, 4)
        a, s = _odd_normal(arg)
        return _atom_expr(Atom('f', 'atan', a)) * s
    if name == 'abs':
        q = arg.as_fraction()
        if q is not None:
            return _const(abs(q))
        if len(arg.t) == 1:
            (mono, c), = arg.t.items()
            core = Expr({mono: Fraction(1)})
            if len(mono) == 1 and mono[0][0].kind == 'f' and mono[0][0].name == 'abs' and mono[0][1] == 1:
                return core * abs(c)
            if len(mono) == 1 and mono[0][0].kind == 'e' and mono[0][1] == 1:
                return core * abs(c)
            if not arg.symbols():
                return core * abs(c) * (1 if _const_sign(core) > 0 else -1)
            if len(mono) == 1 and mono[0][0].kind == 's' and mono[0][1].denominator == 1 and mono[0][1].numerator % 2 == 0:
                return core * abs(c)
            return _atom_expr(Atom('f', 'abs', core)) * abs(c)
        if not arg.symbols():
            return arg * _const_sign(arg)
        a, _ = _odd_normal(arg)
        return _atom_expr(Atom('f', 'abs', a))
    if name == 'floor':
        q = arg.as_fraction()
        if q is not None:
            return _const(q.numerator // q.denominator)
        if not arg.symbols():
            v = _decimal_value(arg, {}, 50)
            n = int(v.to_integral_value(rounding=ROUND_FLOOR))
            if abs(v - n) < Decimal(10) ** -35 or abs(v - n - 1) < Decimal(10) ** -35:
                raise NotCovered('floor of a constant too close to an integer')
            return _const(n)
        return _atom_expr(Atom('f', 'floor', arg))
    raise NotCovered(f'unknown function {name}')


def _const_sign(e):
    """The sign (+1, -1) of a constant Expr; NotCovered when it cannot be told from a 60-digit evaluation (an exact zero is returned as 0 only for the zero Expr)."""
    if not e.t:
        return 0
    q = e.as_fraction()
    if q is not None:
        return 1 if q > 0 else -1
    if e.symbols():
        raise NotCovered('sign of a non-constant')
    v = _decimal_value(e, {}, 60)
    if abs(v) < Decimal(10) ** -45:
        raise NotCovered('sign of a constant too close to zero')
    return 1 if v > 0 else -1


# ----------------------------------------------------------------------------------------------------------------------------- substitution

def subs(e, var, val):
    """Replace the symbol `var` in Expr e by the Expr val (every atom is rebuilt, so the result is canonical)."""
    if not e.has(var):
        return e
    out = ZERO
    for mono, c in e.t.items():
        term = _const(c)
        for a, x in mono:
            term = term * _atom_subs(a, x, var, val)
        out = out + term
    return out


def _atom_subs(a, x, var, val):
    k = a.kind
    if k == 's':
        return _pow(val, x) if a.name == var else _atom_expr(a, x)
    if k in 'kr':
        return _atom_expr(a, x)
    if not a.arg.has(var):
        return _atom_expr(a, x)
    arg = subs(a.arg, var, val)
    if k == 'f':
        base = fn(a.name, arg)
    elif k == 'e':
        base = _exp(arg)
    elif k == 'b':
        base = arg
    else:
        base = _sign_pow(arg)
    return _pow(base, x)


def subs_many(e, mapping):
    """Simultaneous substitution of several symbols (through temporary symbols, so values containing the names are not re-substituted)."""
    tmp = {}
    for i, name in enumerate(mapping):
        tmp[name] = f'_tmp{i}'
        e = subs(e, name, sym(tmp[name]))
    for name, val in mapping.items():
        e = subs(e, tmp[name], val)
    return e


# ----------------------------------------------------------------------------------------------------------------------------- text out

def _atom_text(a):
    k = a.kind
    if k == 's':
        return a.name
    if k == 'k':
        return a.name
    if k == 'f':
        return f'{a.name}({to_text(a.arg)})'
    if k == 'e':
        return f'exp({to_text(a.arg)})'
    if k == 'b':
        return f'({to_text(a.arg)})'
    if k == 'r':
        return str(a.name)
    return f'(-1)^({to_text(a.arg)})'


def _factor_text(a, x):
    t = _atom_text(a)
    if x == 1:
        return t
    return f'{t}^{x}' if x.denominator == 1 else f'{t}^({x})'


def _term_text(mono, c):
    """(sign, text) of one term. A sum raised to a negative power is written as `(..)^(-n)` unless it is the only factor of the denominator and has power -1:
    in a denominator `(x + 1)*x` would be multiplied out when read back, so that text would not parse to the same expression."""
    num, den = [], []
    dens = [(a, -x) for a, x in mono if x < 0]
    plain_den = [(a, x) for a, x in dens if not (a.kind == 'b' and x.denominator == 1)]
    sums_den = [(a, x) for a, x in dens if a.kind == 'b' and x.denominator == 1]
    safe = len(sums_den) == 1 and not plain_den and sums_den[0][1] == 1
    for a, x in mono:
        if x > 0:
            num.append(_factor_text(a, x))
        elif a.kind == 'b' and x.denominator == 1 and not safe:
            num.append(f'{_atom_text(a)}^({x})')
        else:
            den.append(_factor_text(a, -x))
    n, d = abs(c.numerator), c.denominator
    if n != 1 or not num:
        num.insert(0, str(n))
    if d != 1:
        den.insert(0, str(d))
    top = '*'.join(num)
    if not den:
        return (-1 if c < 0 else 1), top
    bottom = den[0] if len(den) == 1 else '(' + '*'.join(den) + ')'
    return (-1 if c < 0 else 1), f'{top}/{bottom}'


def _term_order(item):
    mono, _ = item
    deg = sum((x for a, x in mono if a.kind == 's'), Fraction(0))
    return (-deg, _mono_key(mono))


def to_text(e):
    """Canonical text of an Expr; `parse` reads it back to the same Expr."""
    if not e.t:
        return '0'
    out = []
    for i, (mono, c) in enumerate(sorted(e.t.items(), key=_term_order)):
        sgn, body = _term_text(mono, c)
        if i == 0:
            out.append(('-' if sgn < 0 else '') + body)
        else:
            out.append((' - ' if sgn < 0 else ' + ') + body)
    return ''.join(out)


# ----------------------------------------------------------------------------------------------------------------------------- numbers (Decimal)

def _pi_decimal(prec):
    scale = 10 ** (prec + 12)

    def atan_inv(n):
        x = scale // n
        total, n2, k, sgn = x, n * n, 1, -1
        while x:
            x //= n2
            k += 2
            total += sgn * (x // k)
            sgn = -sgn
        return total
    return Decimal(4 * (4 * atan_inv(5) - atan_inv(239))) / Decimal(scale)


def _sincos(x, prec):
    with localcontext() as ctx:
        ctx.prec = prec + 20 + max(0, x.adjusted())
        twopi = 2 * _pi_decimal(ctx.prec)
        r = x % twopi
        r2 = r * r
        term, s, k = r, r, 1
        cterm, c = Decimal(1), Decimal(1)
        eps = Decimal(10) ** -(ctx.prec + 2)
        n = 0
        while True:
            n += 1
            cterm = -cterm * r2 / ((2 * n - 1) * (2 * n))
            term = -term * r2 / ((2 * n) * (2 * n + 1))
            c += cterm
            s += term
            if abs(term) < eps and abs(cterm) < eps:
                break
        return s, c


def _atan_decimal(x, prec):
    with localcontext() as ctx:
        ctx.prec = prec + 20
        halvings = 0
        while abs(x) > Decimal('0.1'):
            x = x / (1 + (1 + x * x).sqrt())
            halvings += 1
        x2, term, total, k = x * x, x, x, 1
        eps = Decimal(10) ** -(ctx.prec + 2)
        while abs(term) > eps:
            term = -term * x2
            k += 2
            total += term / k
        return total * (2 ** halvings)


def _decimal_value(e, env, prec):
    """The value of an Expr as a Decimal with `prec` significant digits (env: symbol -> Decimal/int/Fraction)."""
    with localcontext() as ctx:
        ctx.prec = prec + 15
        v = _dec_expr(e, env, prec + 15)
    with localcontext() as ctx:
        ctx.prec = prec
        return +v


def _to_dec(q):
    if isinstance(q, Fraction):
        return Decimal(q.numerator) / Decimal(q.denominator)
    return Decimal(q)


def _dec_expr(e, env, prec):
    total = Decimal(0)
    for mono, c in e.t.items():
        v = _to_dec(c)
        for a, x in mono:
            v *= _dec_pow(_dec_atom(a, env, prec), x)
        total += v
    return total


def _dec_pow(b, x):
    if x == 1:
        return b
    if x.denominator == 1:
        if b == 0 and x < 0:
            raise BadInput('division by zero')
        return b ** int(x)
    if b < 0:
        raise BadInput('fractional power of a negative number')
    if b == 0:
        if x < 0:
            raise BadInput('division by zero')
        return Decimal(0)
    return b ** (Decimal(x.numerator) / Decimal(x.denominator))


def _dec_atom(a, env, prec):
    k = a.kind
    if k == 's':
        if a.name not in env:
            raise BadInput(f'no value for {a.name}')
        return _to_dec(env[a.name]) if not isinstance(env[a.name], Decimal) else env[a.name]
    if k == 'k':
        return _pi_decimal(prec)
    if k == 'r':
        return Decimal(a.name)
    if k == 'b':
        return _dec_expr(a.arg, env, prec)
    v = _dec_expr(a.arg, env, prec)
    if k == 'e':
        return v.exp()
    if k == 'm':
        n = v.to_integral_value()
        if abs(v - n) > Decimal(10) ** -10:
            raise BadInput('(-1) to a non-integer power')
        return Decimal(-1) ** int(n)
    name = a.name
    if name == 'log':
        if v <= 0:
            raise BadInput('log of a non-positive number')
        return v.ln()
    if name == 'sin':
        return _sincos(v, prec)[0]
    if name == 'cos':
        return _sincos(v, prec)[1]
    if name == 'tan':
        s, c = _sincos(v, prec)
        if abs(c) < Decimal(10) ** -(prec - 5):
            raise BadInput('tan at a pole')
        return s / c
    if name == 'atan':
        return _atan_decimal(v, prec)
    if name == 'abs':
        return abs(v)
    if name == 'floor':
        return v.to_integral_value(rounding=ROUND_FLOOR)
    raise NotCovered(f'unknown function {name}')


# ----------------------------------------------------------------------------------------------------------------------------- parser

_FUNC_WORDS = ('arctan', 'atan', 'sqrt', 'floor', 'sin', 'cos', 'tan', 'exp', 'log', 'ln', 'abs')
_WORDS = _FUNC_WORDS + ('pi',)
_CMD_FUNCS = {'sin': 'sin', 'cos': 'cos', 'tan': 'tan', 'exp': 'exp', 'ln': 'ln', 'log': 'log', 'arctan': 'atan', 'atan': 'atan'}
_CMD_OPS = {'cdot': '*', 'times': '*', 'ast': '*', 'div': '/', 'lvert': '|', 'rvert': '|', 'vert': '|', 'lfloor': 'lfloor', 'rfloor': 'rfloor'}
_CMD_SKIP = {'left', 'right', 'displaystyle', 'bigl', 'bigr', 'Bigl', 'Bigr', 'big', 'Big', 'quad', 'qquad', 'mathrm', 'textstyle'}
_UNICODE_OPS = {'×': '*', '·': '*', '⋅': '*', '∙': '*', '÷': '/', '−': '-', '–': '-'}
_NUM_RE = re.compile(r'\d{1,3}(?:,\d{3})+(?:\.\d+)?(?!\d)|\d+(?:\.\d+)?|\.\d+')
_SPACING = re.compile(r'\\[,;:! ]|~')


def _tokenize(text):
    s = _SPACING.sub(' ', text)
    toks = []
    i, n = 0, len(s)
    while i < n:
        c = s[i]
        if c.isspace():
            i += 1
            continue
        m = _NUM_RE.match(s, i)
        if m and (c.isdigit() or c == '.'):
            toks.append(('num', Fraction(m.group().replace(',', ''))))
            i = m.end()
            if i < n and s[i] == '.':
                raise BadInput('malformed number')
            continue
        if c == '\\':
            m = re.match(r'\\(?:[A-Za-z]+|.)', s[i:])
            word = m.group()[1:]
            i += len(m.group())
            if word in _CMD_SKIP:
                if word == 'mathrm':
                    raise BadInput('\\mathrm is not supported')
                continue
            if word in ('frac', 'dfrac', 'tfrac'):
                toks.append(('cmd', 'frac'))
            elif word == 'sqrt':
                toks.append(('cmd', 'sqrt'))
            elif word in _CMD_OPS:
                toks.append(('op', _CMD_OPS[word]))
            elif word in _CMD_FUNCS:
                toks.append(('func', _CMD_FUNCS[word]))
            elif word == 'pi':
                toks.append(('id', 'pi'))
            elif word == 'infty':
                toks.append(('id', 'oo'))
            elif word in ('{', '}', '(', ')', '[', ']', '|'):
                toks.append(('op', {'{': '(', '}': ')'}.get(word, word)))
            else:
                raise BadInput(f'unsupported command \\{word}')
            continue
        if c.isalpha():
            j = i
            while j < n and s[j].isalpha():
                j += 1
            run = s[i:j]
            k = 0
            while k < len(run):
                for w in _WORDS:
                    if run.startswith(w, k):
                        toks.append(('func', w) if w in _FUNC_WORDS else ('id', 'pi'))
                        k += len(w)
                        break
                else:
                    name = run[k]
                    k += 1
                    if k == len(run):                       # a subscript belongs to the last letter of the run
                        j2 = j
                        if j2 < n and s[j2] == '_':
                            if j2 + 1 < n and s[j2 + 1] == '{':
                                end = s.find('}', j2)
                                if end < 0:
                                    raise BadInput('unclosed subscript')
                                name += '_' + s[j2 + 2:end].strip()
                                j = end + 1
                            else:
                                m2 = re.match(r'\d+|[A-Za-z]', s[j2 + 1:])
                                if not m2:
                                    raise BadInput('malformed subscript')
                                name += '_' + m2.group()
                                j = j2 + 1 + len(m2.group())
                        elif j2 < n and s[j2].isdigit():
                            raise BadInput(f'ambiguous {name}{s[j2]}: write {name}*{s[j2]} or {name}_{s[j2]}')
                    toks.append(('id', name))
            i = j
            continue
        if c == '*' and s.startswith('**', i):
            toks.append(('op', '^'))
            i += 2
            continue
        c = _UNICODE_OPS.get(c, c)
        if c == '_':
            toks.append(('op', '_'))
            i += 1
            continue
        if c in '+-*/^()[]{}|=,!':
            toks.append(('op', c))
            i += 1
            continue
        raise BadInput(f'unexpected character {c!r}')
    if len(toks) > MAX_TOKENS:
        raise NotCovered('too large')
    return toks


class _Parser:
    def __init__(self, toks, e_is_euler=False):
        self.t, self.i, self.depth, self.abs_depth = toks, 0, 0, 0
        self.e_is_euler = e_is_euler

    def peek(self, k=0):
        j = self.i + k
        return self.t[j] if j < len(self.t) else (None, None)

    def take(self):
        tok = self.peek()
        self.i += 1
        return tok

    def accept(self, kind, val):
        if self.peek() == (kind, val):
            self.i += 1
            return True
        return False

    def expect(self, kind, val):
        if not self.accept(kind, val):
            raise BadInput(f'expected {val!r} but found {self.peek()[1]!r}')

    def enter(self):
        self.depth += 1
        if self.depth > MAX_DEPTH:
            raise NotCovered('too large')

    def leave(self):
        self.depth -= 1

    def expr(self):
        self.enter()
        try:
            left = self.term()
            while True:
                if self.accept('op', '+'):
                    left = left + self.term()
                elif self.accept('op', '-'):
                    left = left - self.term()
                else:
                    return left
        finally:
            self.leave()

    def starts_atom(self):
        kind, val = self.peek()
        if kind in ('num', 'id', 'func', 'cmd'):
            return True
        if kind == 'op':
            if val in ('(', '[', '{', 'lfloor'):
                return True
            if val == '|':
                return self.abs_depth == 0
        return False

    def term(self):
        left = self.unary()
        while True:
            kind, val = self.peek()
            if kind == 'op' and val == '*':
                self.take()
                left = left * self.unary()
            elif kind == 'op' and val == '/':
                self.take()
                left = left * self.inverse_factor()
            elif self.starts_atom():
                if kind == 'num':
                    raise BadInput('a number directly after a factor')
                left = left * self.power()
            else:
                return left

    def inverse_factor(self):
        """1/(next factor); `1/(x+1)^2` is read as (x+1)^(-2), not as the inverse of the expanded square."""
        if self.peek() in (('op', '-'), ('op', '+')):
            return _pow(self.unary(), Fraction(-1))
        base = self.atom()
        if self.accept('op', '^'):
            ex = self.exponent()
            q = ex.as_fraction()
            return _pow(base, -q) if q is not None else _pow(_pow(base, ex), Fraction(-1))
        return _pow(base, Fraction(-1))

    def unary(self):
        self.enter()
        try:
            if self.accept('op', '-'):
                return -self.unary()
            if self.accept('op', '+'):
                return self.unary()
            return self.power()
        finally:
            self.leave()

    def power(self):
        base = self.atom()
        while self.peek() == ('op', '!'):
            raise BadInput('factorial is not supported')
        if self.accept('op', '^'):
            return _pow(base, self.exponent())
        return base

    def exponent(self):
        self.enter()
        try:
            if self.peek() == ('op', '{'):
                return self.group()
            if self.accept('op', '-'):
                return -self.exponent()
            if self.accept('op', '+'):
                return self.exponent()
            return self.power()
        finally:
            self.leave()

    def group(self):
        self.expect('op', '{')
        v = self.expr()
        self.expect('op', '}')
        return v

    def closing(self, opener):
        return {'(': ')', '[': ']', '{': '}'}[opener]

    def atom(self):
        kind, val = self.take()
        if kind is None:
            raise BadInput('unexpected end of the expression')
        if kind == 'num':
            return _const(val)
        if kind == 'id':
            if val == 'pi':
                return PI
            if val == 'E' or (val == 'e' and self.e_is_euler):
                return _exp(ONE)
            if val == 'oo':
                raise BadInput('infinity is not a value here')
            return sym(val)
        if kind == 'cmd':
            if val == 'frac':
                a = self.group()
                b = self.group()
                return a / b
            if self.peek() == ('op', '['):
                self.take()
                idx = self.expr()
                self.expect('op', ']')
                q = idx.as_fraction()
                if q is None or q.denominator != 1 or q < 1:
                    raise BadInput('the index of a root must be a positive integer')
                body = self.group_or_atom()
                return _pow(body, Fraction(1, int(q)))
            return _pow(self.group_or_atom(), Fraction(1, 2))
        if kind == 'func':
            return self.function(val)
        if kind == 'op':
            if val in ('(', '[', '{'):
                v = self.expr()
                self.expect('op', self.closing(val))
                return v
            if val == '|':
                self.abs_depth += 1
                v = self.expr()
                self.expect('op', '|')
                self.abs_depth -= 1
                return fn('abs', v)
            if val == 'lfloor':
                v = self.expr()
                self.expect('op', 'rfloor')
                return fn('floor', v)
        raise BadInput(f'unexpected {val!r}')

    def group_or_atom(self):
        if self.peek() == ('op', '{'):
            return self.group()
        kind, _ = self.peek()
        if kind in ('num', 'id') or self.peek() == ('op', '('):
            return self.atom()
        raise BadInput('a root needs a braced or parenthesised argument')

    def function(self, name):
        base = None
        if name in ('log', 'ln') and self.peek() == ('op', '_'):
            self.take()
            base = self.group() if self.peek() == ('op', '{') else self.atom()
        power = None
        if self.peek() == ('op', '^'):
            self.take()
            power = self.exponent()
            if name in ('sin', 'cos', 'tan') and power.as_fraction() == -1:
                raise BadInput(f'{name}^(-1) is ambiguous')
        if self.peek() in (('op', '('), ('op', '[')):
            opener = self.take()[1]
            self.enter()
            arg = self.expr()
            self.leave()
            self.expect('op', self.closing(opener))
        elif self.peek() == ('op', '{'):
            arg = self.group()
        elif name == 'sqrt':
            arg = self.group_or_atom()
        else:
            arg = self.power()
            kind, val = self.peek()
            if kind in ('num', 'id') or (kind == 'op' and val in ('(', '[')):
                raise BadInput(f'ambiguous argument of {name}: use parentheses')
        if name == 'ln':
            res = _log(arg)
        elif name == 'log':
            res = _log(arg) if base is None else _log(arg) / _log(base)
        elif name == 'arctan':
            res = fn('atan', arg)
        else:
            res = fn(name, arg)
        return res if power is None else _pow(res, power)


def parse(text, e_is_euler=False):
    """Read expression text (plain infix or a LaTeX subset) into a canonical Expr."""
    if not isinstance(text, str):
        raise BadInput('expression must be text')
    toks = _tokenize(text)
    if not toks:
        raise BadInput('empty expression')
    p = _Parser(toks, e_is_euler)
    v = p.expr()
    if p.i != len(toks):
        raise BadInput(f'unexpected {p.peek()[1]!r}')
    return v


def parse_equation(text, e_is_euler=False):
    """(lhs, rhs) Exprs of `lhs = rhs`; a text without `=` is read as `text = 0`."""
    toks = _tokenize(text)
    if not toks:
        raise BadInput('empty expression')
    cuts = [i for i, t in enumerate(toks) if t == ('op', '=')]
    if len(cuts) > 1:
        raise BadInput('more than one =')
    parts = [toks[:cuts[0]], toks[cuts[0] + 1:]] if cuts else [toks, [('num', Fraction(0))]]
    out = []
    for part in parts:
        if not part:
            raise BadInput('empty side of the equation')
        p = _Parser(part, e_is_euler)
        out.append(p.expr())
        if p.i != len(part):
            raise BadInput(f'unexpected {p.peek()[1]!r}')
    return out[0], out[1]


# ----------------------------------------------------------------------------------------------------------------------------- derivative

def _dlog(a, var):
    """The logarithmic derivative d(atom)/d(var) / atom, as an Expr."""
    k = a.kind
    if k == 's':
        return _pow(sym(var), Fraction(-1)) if a.name == var else ZERO
    if k in 'kr':
        return ZERO
    if not a.arg.has(var):
        return ZERO
    if k == 'e':
        return diff(a.arg, var)
    if k == 'b':
        return diff(a.arg, var) * _pow(a.arg, Fraction(-1))
    if k == 'm':
        raise NotCovered('derivative of (-1)^k in k')
    u = a.arg
    du = diff(u, var)
    name = a.name
    me = _atom_expr(a)
    if name == 'log':
        return du * _pow(u, Fraction(-1)) * _pow(me, Fraction(-1))
    if name == 'sin':
        return du * fn('cos', u) * _pow(me, Fraction(-1))
    if name == 'cos':
        return -du * fn('sin', u) * _pow(me, Fraction(-1))
    if name == 'tan':
        return du * (_pow(me, Fraction(-1)) + me)
    if name == 'atan':
        return du * _pow(ONE + u * u, Fraction(-1)) * _pow(me, Fraction(-1))
    if name == 'abs':
        return du * _pow(u, Fraction(-1))
    raise NotCovered(f'derivative of {name}')


def diff(e, var, order=1):
    """d^order e / d var^order as an Expr."""
    for _ in range(order):
        out = ZERO
        for mono, c in e.t.items():
            if not any(a.has(var) for a, _ in mono):
                continue
            dl = ZERO
            for a, x in mono:
                if a.has(var):
                    dl = dl + _dlog(a, var) * x
            if dl.t:
                out = out + Expr({mono: c}) * dl
        e = out
        if not e.t:
            break
    return e


# ----------------------------------------------------------------------------------------------------------------------------- polynomials over Q (lists, low degree first)

def _pn(p):
    p = list(p)
    while p and p[-1] == 0:
        p.pop()
    return p


def _padd(a, b):
    n = max(len(a), len(b))
    return _pn([(a[i] if i < len(a) else 0) + (b[i] if i < len(b) else 0) for i in range(n)])


def _psub(a, b):
    n = max(len(a), len(b))
    return _pn([(a[i] if i < len(a) else 0) - (b[i] if i < len(b) else 0) for i in range(n)])


def _pmul(a, b):
    if not a or not b:
        return []
    out = [Fraction(0)] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        if x:
            for j, y in enumerate(b):
                out[i + j] += x * y
    return _pn(out)


def _pscale(a, c):
    return _pn([x * c for x in a])


def _ppow(a, n):
    res = [Fraction(1)]
    for _ in range(n):
        res = _pmul(res, a)
    return res


def _pdivmod(a, b):
    a, b = _pn(a), _pn(b)
    if not b:
        raise BadInput('division by the zero polynomial')
    q = [Fraction(0)] * max(len(a) - len(b) + 1, 0)
    r = list(a)
    while len(r) >= len(b) and r:
        c = r[-1] / b[-1]
        d = len(r) - len(b)
        q[d] = c
        for i, y in enumerate(b):
            r[i + d] -= c * y
        r = _pn(r)
    return _pn(q), r


def _pmonic(a):
    a = _pn(a)
    return [x / a[-1] for x in a] if a else a


def _pgcd(a, b):
    a, b = _pn(a), _pn(b)
    while b:
        a, b = b, _pdivmod(a, b)[1]
    return _pmonic(a)


def _pder(a):
    return _pn([i * a[i] for i in range(1, len(a))])


def _peval(p, x):
    r = Fraction(0)
    for c in reversed(p):
        r = r * x + c
    return r


def _pexpr(p, var):
    v = sym(var)
    out = ZERO
    for i, c in enumerate(p):
        if c:
            out = out + _pow(v, i) * c if i else out + _const(c)
    return out


def _to_poly(e, var):
    """The coefficient list of an Expr that is a polynomial in `var` with rational coefficients, else None."""
    out = {}
    for mono, c in e.t.items():
        if mono == ():
            k = 0
        elif len(mono) == 1 and mono[0][0].kind == 's' and mono[0][0].name == var and mono[0][1].denominator == 1 and mono[0][1] > 0:
            k = int(mono[0][1])
        else:
            return None
        out[k] = out.get(k, 0) + c
    return _pn([out.get(i, Fraction(0)) for i in range(max(out) + 1)]) if out else []


def _to_ratfunc(e, var):
    """(numerator, denominator) coefficient lists, reduced, of an Expr that is a rational function of `var` with rational coefficients; else None."""
    num, den = [], [Fraction(1)]
    for mono, c in e.t.items():
        tn, td = [c], [Fraction(1)]
        for a, x in mono:
            if x.denominator != 1:
                return None
            n = int(x)
            if a.kind == 's' and a.name == var:
                part = [Fraction(0)] * abs(n) + [Fraction(1)]
            elif a.kind == 'b':
                part = _to_poly(a.arg, var)
                if part is None:
                    return None
                part = _ppow(part, abs(n))
            else:
                return None
            if n >= 0:
                tn = _pmul(tn, part)
            else:
                td = _pmul(td, part)
        num, den = _radd(num, den, tn, td)
    return num, den


def _radd(n1, d1, n2, d2):
    g = _pgcd(d1, d2)
    if len(g) > 1:
        d2g, _ = _pdivmod(d2, g)
        n = _padd(_pmul(n1, d2g), _pmul(n2, _pdivmod(d1, g)[0]))
        d = _pmul(d1, d2g)
    else:
        n = _padd(_pmul(n1, d2), _pmul(n2, d1))
        d = _pmul(d1, d2)
    if not n:
        return [], [Fraction(1)]
    g = _pgcd(n, d)
    if len(g) > 1:
        n, d = _pdivmod(n, g)[0], _pdivmod(d, g)[0]
    lc = d[-1]
    return _pscale(n, 1 / lc), _pscale(d, 1 / lc)


def _integer_form(num, den):
    """(N, D) integer-coefficient lists with the same ratio, D with a positive leading coefficient and the pair jointly primitive."""
    allc = [c for c in num + den]
    lcm_d = 1
    for c in allc:
        lcm_d = lcm_d * c.denominator // gcd(lcm_d, c.denominator)
    n = [int(c * lcm_d) for c in num]
    d = [int(c * lcm_d) for c in den]
    g = 0
    for c in n + d:
        g = gcd(g, c)
    g = g or 1
    return [Fraction(c, g) for c in n], [Fraction(c, g) for c in d]


# ----------------------------------------------------------------------------------------------------------------------------- zero test, rational form, simplify

def _is_free(e):
    """True when the canonical form of e is complete: only symbols, pi, prime radicals and sums to integer powers (so unequal forms are unequal functions)."""
    for mono in e.t:
        for a, x in mono:
            if a.kind in 'sk':
                continue
            if a.kind == 'r':
                continue
            if a.kind == 'b' and x.denominator == 1 and _is_free(a.arg):
                continue
            return False
    return True


def is_zero(e):
    """True when e is shown to be 0: its terms cancel after the denominators are cleared. False means not shown (identities such as sin^2 + cos^2 = 1 are not known)."""
    for _ in range(6):
        if not e.t:
            return True
        low = {}
        for mono in e.t:
            for a, x in mono:
                if x < 0 and x < low.get(a, 0):
                    low[a] = x
        if not low:
            return False
        for a, x in low.items():
            e = e * _atom_expr(a, -x)
    return not e.t


def _transc_split(mono, var):
    """(rational part atoms, other atoms) of a monomial with respect to `var`."""
    rat, oth = [], []
    for a, x in mono:
        if x.denominator == 1 and ((a.kind == 's' and a.name == var) or (a.kind == 'b' and _to_poly(a.arg, var) is not None)):
            rat.append((a, x))
        else:
            oth.append((a, x))
    return tuple(rat), tuple(oth)


def simplify_expr(e):
    """Normal form: the canonical expanded form, with the rational part of each group of non-rational atoms reduced to lowest terms when there is one variable."""
    syms = e.symbols()
    if len(syms) != 1:
        return e
    var = next(iter(syms))
    groups = {}
    for mono, c in e.t.items():
        rat, oth = _transc_split(mono, var)
        groups.setdefault(oth, []).append((rat, c))
    out = ZERO
    for oth, items in groups.items():
        part = Expr({})
        for rat, c in items:
            part = part + Expr({_sorted_mono(dict(rat)): c})
        rf = _to_ratfunc(part, var)
        if rf is None:
            part2 = part
        else:
            n, d = rf
            part2 = _pexpr(n, var) * _pow(_pexpr(d, var), Fraction(-1)) if len(d) > 1 else _pexpr(_pscale(n, 1 / d[0]), var)
        out = out + part2 * Expr({oth: Fraction(1)})
    return out


# ----------------------------------------------------------------------------------------------------------------------------- roots and factors over Q

def _divisors(n):
    n = abs(int(n))
    if n == 0:
        raise BadInput('divisors of 0')
    ds = [1]
    for p, k in _small_primes_factor(n).items():
        ds = [d * p ** i for d in ds for i in range(k + 1)]
    return ds


def _rational_roots(p):
    """([(root, multiplicity)], cofactor) for a non-zero polynomial: all its rational roots by the rational-root theorem, and what is left."""
    p = _pn(p)
    if not p:
        raise BadInput('the zero polynomial')
    roots = []
    zeros = 0
    while p and p[0] == 0:
        p = p[1:]
        zeros += 1
    if zeros:
        roots.append((Fraction(0), zeros))
    if len(p) <= 1:
        return roots, p
    lcm_d = 1
    for c in p:
        lcm_d = lcm_d * c.denominator // gcd(lcm_d, c.denominator)
    ip = [int(c * lcm_d) for c in p]
    da, db = _divisors(ip[0]), _divisors(ip[-1])
    if len(da) * len(db) > 200000:
        raise NotCovered('too many rational-root candidates')
    cand = set()
    for u in da:
        for v in db:
            if gcd(u, v) == 1:
                cand.add(Fraction(u, v))
                cand.add(Fraction(-u, v))
    for r in sorted(cand):
        m = 0
        while len(p) > 1 and _peval(p, r) == 0:
            p, _ = _pdivmod(p, [-r, Fraction(1)])
            m += 1
        if m:
            roots.append((r, m))
    roots.sort()
    return roots, p


def _squarefree(p):
    """Yun's square-free decomposition: [(q, j)] with p = lc * prod q**j, q monic square-free and pairwise coprime."""
    p = _pmonic(p)
    c = _pgcd(p, _pder(p))
    w = _pdivmod(p, c)[0]
    y = _pdivmod(_pder(p), c)[0]
    z = _psub(y, _pder(w))
    out, i = [], 1
    while len(w) > 1:
        g = _pgcd(w, z)
        if len(g) > 1:
            out.append((g, i))
        w = _pdivmod(w, g)[0]
        y = _pdivmod(z, g)[0]
        z = _psub(y, _pder(w))
        i += 1
    return out


def _solve_linear(rows, rhs):
    """Solution list of a square or tall system over Fractions by Gauss-Jordan, or None when singular or inconsistent."""
    m = [list(r) + [b] for r, b in zip(rows, rhs)]
    n = len(rows[0]) if rows else 0
    piv, r = [], 0
    for col in range(n):
        pr = next((i for i in range(r, len(m)) if m[i][col] != 0), None)
        if pr is None:
            return None
        m[r], m[pr] = m[pr], m[r]
        inv = 1 / m[r][col]
        m[r] = [x * inv for x in m[r]]
        for i in range(len(m)):
            if i != r and m[i][col] != 0:
                f = m[i][col]
                m[i] = [a - f * b for a, b in zip(m[i], m[r])]
        piv.append(col)
        r += 1
    if any(row[-1] != 0 for row in m[r:]):
        return None
    return [m[i][-1] for i in range(n)]


def _apart_terms(num, den):
    """(polynomial part, [(A, f, k)]) with num/den = poly + sum A/f**k, f monic and irreducible over Q (linear, or degree 2-3 without rational roots), deg A < deg f."""
    num, den = _pn(num), _pn(den)
    if not den:
        raise BadInput('division by zero')
    q, r = _pdivmod(num, den)
    lc = den[-1]
    r, den = _pscale(r, 1 / lc), _pscale(den, 1 / lc)
    q = _pscale(q, 1)
    if not r:
        return q, []
    if len(den) - 1 > 40:
        raise NotCovered('denominator degree above 40')
    roots, rest = _rational_roots(den)
    fams = [([-rt, Fraction(1)], m) for rt, m in roots]
    if len(rest) > 1:
        for qj, j in _squarefree(rest):
            if len(qj) - 1 >= 4:
                raise NotCovered('a factor of degree 4 or more without rational roots: irreducibility not decided')
            fams.append((qj, j))
    big_d = [Fraction(1)]
    for f, m in fams:
        big_d = _pmul(big_d, _ppow(f, m))
    cols, labels = [], []
    for fi, (f, m) in enumerate(fams):
        for k in range(1, m + 1):
            cof = _pdivmod(big_d, _ppow(f, k))[0]
            for i in range(len(f) - 1):
                cols.append(_pmul([Fraction(0)] * i + [Fraction(1)], cof))
                labels.append((fi, k, i))
    n = len(big_d) - 1
    rows = [[(col[j] if j < len(col) else Fraction(0)) for col in cols] for j in range(n)]
    rhs = [(r[j] if j < len(r) else Fraction(0)) for j in range(n)]
    sol = _solve_linear(rows, rhs)
    if sol is None:
        raise NotCovered('partial fractions: singular system')
    terms = {}
    for (fi, k, i), v in zip(labels, sol):
        a = terms.setdefault((fi, k), [Fraction(0)] * (len(fams[fi][0]) - 1))
        a[i] = v
    out = []
    for (fi, k), a in sorted(terms.items()):
        a = _pn(a)
        if a:
            out.append((a, fams[fi][0], k))
    return q, out


def apart_expr(e, var):
    """Partial-fraction form of a rational function of `var` as an Expr."""
    rf = _to_ratfunc(e, var)
    if rf is None:
        raise NotCovered('not a rational function of one variable with rational coefficients')
    q, terms = _apart_terms(*rf)
    out = _pexpr(q, var)
    for a, f, k in terms:
        out = out + _pexpr(a, var) * _pow(_pexpr(f, var), Fraction(-k))
    if not is_zero(out - e):
        raise NotCovered('partial fractions: check failed')
    return out


# ----------------------------------------------------------------------------------------------------------------------------- truncated Laurent series

INF = Fraction(10 ** 9)
_LOG_T = sym('_L')            # stands for log(t) in the coefficients of a series in t -> 0+
_STATE = {'cap': Fraction(8), 'flat': False}


class Ser:
    """sum of c[e] * t**e for Fraction exponents e, known for e < prec (the rest is unknown). Coefficients are constant Exprs (they may hold the symbol _L = log t)."""
    __slots__ = ('c', 'prec')

    def __init__(self, c, prec=INF):
        prec = min(prec, _STATE['cap'])
        self.c = {e: v for e, v in c.items() if e < prec and v.t}
        self.prec = prec

    def val(self):
        return min(self.c) if self.c else self.prec

    def __add__(self, o):
        out = dict(self.c)
        for e, v in o.c.items():
            w = out.get(e, ZERO) + v
            if w.t:
                out[e] = w
            else:
                out.pop(e, None)
        return Ser(out, min(self.prec, o.prec))

    def __neg__(self):
        return Ser({e: -v for e, v in self.c.items()}, self.prec)

    def __sub__(self, o):
        return self + (-o)

    def __mul__(self, o):
        va, vb = self.val(), o.val()
        prec = min(self.prec + vb, o.prec + va)
        prec = min(prec, _STATE['cap'])
        out = {}
        for e1, c1 in self.c.items():
            for e2, c2 in o.c.items():
                e = e1 + e2
                if e < prec:
                    w = out.get(e, ZERO) + c1 * c2
                    if w.t:
                        out[e] = w
                    else:
                        out.pop(e, None)
        return Ser(out, prec)

    def scale(self, k):
        return Ser({e: v * k for e, v in self.c.items()}, self.prec)

    def shift(self, d):
        return Ser({e + d: v for e, v in self.c.items()}, self.prec + d)

    def has_log(self):
        return any(('_L' in v.symbols()) for v in self.c.values())


def _const_ser(v):
    return Ser({Fraction(0): v}) if v.t else Ser({})


def _check_nonzero(c):
    """Guard against a coefficient that is canonical-non-zero but numerically zero (an identity the form does not know)."""
    if c.as_fraction() is None and not c.symbols():
        v = _decimal_value(c, {}, 60)
        if abs(v) < Decimal(10) ** -40:
            raise NotCovered('a coefficient that is zero by an identity the form does not know')


def _need_no_log(s, what):
    if s.has_log():
        raise NotCovered(f'{what} of a series with log terms')


def _unit_part(a):
    """(v, c0, u) with a = c0 * t**v * (1 + u), u of positive valuation (relative terms)."""
    if not a.c:
        raise NotCovered('series is zero up to its known order: more terms needed')
    v = a.val()
    c0 = a.c[v]
    _check_nonzero(c0)
    inv = _pow(c0, Fraction(-1))
    u = Ser({e - v: c * inv for e, c in a.c.items() if e != v}, a.prec - v)
    return v, c0, u


def _series_sum(u, coefs):
    """sum coefs[k] * u**k for k = 0..len-1, stopping when u**k is beyond the precision."""
    res = Ser({Fraction(0): _const(coefs(0))}, INF) if coefs(0) else Ser({})
    uk = u
    k = 1
    vu = u.val()
    while vu > 0 and k * vu < min(u.prec, _STATE['cap']) + 1 and k < 400:
        co = coefs(k)
        if co:
            res = res + uk.scale(co)
        uk = uk * u
        k += 1
        if not uk.c:
            break
    return Ser(res.c, min(res.prec, u.prec))


def ser_inv(a):
    _need_no_log(a, 'division')
    v, c0, u = _unit_part(a)
    s = _series_sum(u, lambda k: Fraction((-1) ** k))
    return s.scale(_pow(c0, Fraction(-1))).shift(-v)


def ser_pow(a, r):
    r = Fraction(r)
    if r == 1:
        return a
    if r.denominator == 1:
        n = int(r)
        if n == 0:
            return Ser({Fraction(0): ONE})
        base = a if n > 0 else ser_inv(a)
        n = abs(n)
        res, sq = None, base
        while n:
            if n & 1:
                res = sq if res is None else res * sq
            n >>= 1
            if n:
                sq = sq * sq
        return res
    _need_no_log(a, 'fractional power')
    v, c0, u = _unit_part(a)
    if _const_sign(c0) <= 0:
        raise NotCovered('fractional power of a non-positive leading coefficient')

    def binom(k):
        c = Fraction(1)
        for i in range(k):
            c = c * (r - i) / (i + 1)
        return c
    s = _series_sum(u, binom)
    return s.scale(_pow(c0, r)).shift(v * r)


def ser_exp(a):
    _need_no_log(a, 'exp')
    if not a.c:
        return Ser({Fraction(0): ONE}, a.prec)
    v = a.val()
    if v < 0:
        # exp of a series that blows up: a negative leading coefficient makes it smaller than every power of t (0 for a limit), a positive one is unbounded
        if _STATE['flat'] and _const_sign(a.c[v]) < 0:
            return Ser({})
        raise NotCovered('exp of an unbounded series: essential singularity')
    a0 = a.c.get(Fraction(0), ZERO)
    u = Ser({e: c for e, c in a.c.items() if e != 0}, a.prec)
    fact = [Fraction(1)]
    s = _series_sum(u, lambda k: Fraction(1, _fact(k)))
    return s.scale(_exp(a0))


def _fact(k):
    r = 1
    for i in range(2, k + 1):
        r *= i
    return r


def ser_log(a):
    _need_no_log(a, 'log')
    v, c0, u = _unit_part(a)
    if _const_sign(c0) <= 0:
        raise NotCovered('log of a non-positive leading coefficient')
    s = _series_sum(u, lambda k: Fraction((-1) ** (k + 1), k) if k else Fraction(0))
    head = _log(c0) + _LOG_T * v if v != 0 else _log(c0)
    return s + _const_ser(head)


def ser_sincos(a):
    """(sin a, cos a) for a series of non-negative valuation."""
    _need_no_log(a, 'sin/cos')
    if a.c and a.val() < 0:
        raise NotCovered('sin/cos of an unbounded series')
    a0 = a.c.get(Fraction(0), ZERO)
    u = Ser({e: c for e, c in a.c.items() if e != 0}, a.prec)
    su = _series_sum(u, lambda k: Fraction(0) if k % 2 == 0 else Fraction((-1) ** (k // 2), _fact(k)))
    cu = _series_sum(u, lambda k: Fraction(0) if k % 2 else Fraction((-1) ** (k // 2), _fact(k)))
    s0, c0 = fn('sin', a0), fn('cos', a0)
    return su.scale(c0) + cu.scale(s0), cu.scale(c0) - su.scale(s0)


def ser_atan(a):
    _need_no_log(a, 'atan')
    if not a.c:
        return Ser({}, a.prec)
    v = a.val()
    if v > 0:
        return _series_sum(a, lambda k: Fraction((-1) ** (k // 2), k) if k % 2 else Fraction(0))
    if v < 0:
        s = _const_sign(a.c[v])
        return _const_ser(PI * Fraction(s, 2)) - ser_atan(ser_inv(a))
    raise NotCovered('atan of a series with a non-zero constant term')


def _ser_of_expr(e, var, x_ser, cache):
    total = Ser({})
    for mono, c in e.t.items():
        term = _const_ser(_const(c))
        for a, x in mono:
            s = _ser_atom(a, var, x_ser, cache)
            term = term * (s if x == 1 else ser_pow(s, x))
        total = total + term
    return total


def _ser_atom(a, var, x_ser, cache):
    if a.key in cache:
        return cache[a.key]
    k = a.kind
    if k == 's':
        out = x_ser if a.name == var else _const_ser(_atom_expr(a))
    elif k in 'kr':
        out = _const_ser(_atom_expr(a))
    elif not a.arg.has(var):
        out = _const_ser(_atom_expr(a))
    elif k == 'b':
        out = _ser_of_expr(a.arg, var, x_ser, cache)
    elif k == 'm':
        raise NotCovered('(-1)^k with k the series variable')
    else:
        arg = _ser_of_expr(a.arg, var, x_ser, cache)
        if k == 'e':
            out = ser_exp(arg)
        elif a.name == 'log':
            out = ser_log(arg)
        elif a.name == 'sin':
            out = ser_sincos(arg)[0]
        elif a.name == 'cos':
            out = ser_sincos(arg)[1]
        elif a.name == 'tan':
            sn, cs = ser_sincos(arg)
            out = sn * ser_inv(cs)
        elif a.name == 'atan':
            out = ser_atan(arg)
        elif a.name == 'abs':
            if not arg.c:
                raise NotCovered('abs of a series that is zero up to its known order')
            v = arg.val()
            _check_nonzero(arg.c[v])
            out = arg if _const_sign(arg.c[v]) > 0 else -arg
        else:
            raise NotCovered(f'{a.name} in a series')
    cache[a.key] = out
    return out


def series_at(e, var, point, direction, cap):
    """The series of e in t > 0, with var = point + direction*t (finite point) or var = direction/t (point None: toward +/- infinity)."""
    _STATE['cap'] = Fraction(cap)
    if point is None:
        x_ser = Ser({Fraction(-1): _const(direction)})
    else:
        x_ser = Ser({Fraction(0): _const(point), Fraction(1): _const(direction)}) if point else Ser({Fraction(1): _const(direction)})
    return _ser_of_expr(e, var, x_ser, {})


def _l_top(c):
    """(k, coefficient) with k the highest power of _L in the constant Expr c and coefficient the part of c with that power."""
    k = 0
    for mono in c.t:
        for a, x in mono:
            if a.kind == 's' and a.name == '_L':
                k = max(k, int(x))
    if k == 0:
        return 0, c
    out = {}
    for mono, co in c.t.items():
        ex = next((int(x) for a, x in mono if a.kind == 's' and a.name == '_L'), 0)
        if ex == k:
            out[tuple((a, x) for a, x in mono if not (a.kind == 's' and a.name == '_L'))] = co
    return k, Expr(out)


def one_sided_limit(e, var, point, direction):
    """('value', Expr) or ('inf', +1/-1) for var -> point from one side (point None: var -> direction*infinity)."""
    last = None
    _STATE['flat'] = True
    try:
        for cap in (6, 12, 24, 48):
            s = series_at(e, var, point, direction, cap)
            if not s.c:
                if s.prec > 0:
                    return 'value', ZERO         # no term below t**prec: the function is O(t**prec), so the limit is 0
                last = 'more terms needed'
                continue
            v = s.val()
            k, lead = _l_top(s.c[v])
            if v > 0:
                return 'value', ZERO
            if v == 0 and k == 0:
                return 'value', lead
            sgn = _const_sign(lead) * (-1) ** k
            return 'inf', sgn
    finally:
        _STATE['flat'] = False
    raise NotCovered('series: ' + (last or 'no result'))


def limit_expr(e, var, point, direction='both'):
    """Limit as an Expr or the strings 'oo' / '-oo'. point: Fraction, or 'oo' / '-oo'."""
    if point in ('oo', '-oo'):
        d = 1 if point == 'oo' else -1
        pts = [(None, d)]
    else:
        pts = [(point, 1), (point, -1)] if direction == 'both' else [(point, 1 if direction == '+' else -1)]
    res = []
    for p, d in pts:
        res.append(one_sided_limit(e, var, p, d))
    first = res[0]
    for r in res[1:]:
        if r[0] != first[0] or (r[0] == 'inf' and r[1] != first[1]) or (r[0] == 'value' and not is_zero(r[1] - first[1])):
            raise NotCovered('the two one-sided limits differ')
    if first[0] == 'inf':
        return 'oo' if first[1] > 0 else '-oo'
    return first[1]


def taylor_coefficients(e, var, point, order):
    """[c0..c_order] with e = sum c_k (var - point)**k + O((var-point)**(order+1)); NotCovered when e is not analytic at the point
    (the series from the right must also equal the series from the left: abs and floor fail that)."""
    if order < 0 or order > MAX_SERIES_ORDER:
        raise NotCovered('order out of range')
    for cap in (order + 3, 2 * order + 6, 4 * order + 12, 8 * order + 24):
        s = series_at(e, var, point, 1, cap)
        if s.prec <= order:
            continue
        if s.has_log():
            raise NotCovered('not analytic: log terms')
        for ex in s.c:
            if ex < 0 or ex.denominator != 1:
                raise NotCovered('not analytic at the point')
        right = [s.c.get(Fraction(i), ZERO) for i in range(order + 1)]
        left = series_at(e, var, point, -1, cap)
        if left.prec <= order:
            continue
        for i in range(order + 1):
            want = right[i] * (-1) ** i
            if not is_zero(left.c.get(Fraction(i), ZERO) - want):
                raise NotCovered('not analytic at the point: the two sides differ')
        for ex in left.c:
            if ex < 0 or ex.denominator != 1:
                raise NotCovered('not analytic at the point')
        return right
    raise NotCovered('series: not enough terms')


# ----------------------------------------------------------------------------------------------------------------------------- antiderivatives

def _int_ratfunc(num, den, var):
    """An antiderivative of num/den (rational function of var, rational coefficients), as an Expr."""
    q, terms = _apart_terms(num, den)
    x = sym(var)
    out = ZERO
    for i, c in enumerate(q):
        if c:
            out = out + _pow(x, i + 1) * (c / (i + 1))
    for a, f, k in terms:
        fe = _pexpr(f, var)
        if len(f) == 2:                                   # a constant over (x - r)**k
            c = a[0]
            if k == 1:
                out = out + fn('log', fn('abs', fe)) * c
            else:
                out = out + _pow(fe, Fraction(1 - k)) * (c / (1 - k))
        elif len(f) == 3:
            p, q0 = f[1], f[0]
            disc = p * p - 4 * q0
            if disc >= 0:
                raise NotCovered('irreducible quadratic factor with real irrational roots')
            a0 = a[0] if len(a) > 0 else Fraction(0)
            a1 = a[1] if len(a) > 1 else Fraction(0)
            if a1:
                half = a1 / 2
                if k == 1:
                    out = out + fn('log', fe) * half
                else:
                    out = out + _pow(fe, Fraction(1 - k)) * (half / (1 - k))
            c = a0 - a1 * p / 2
            if c:
                out = out + _quad_power_integral(fe, p, q0, k, var) * c
        else:
            raise NotCovered('irreducible factor of degree 3: antiderivative not covered')
    return out


def _quad_power_integral(fe, p, q0, k, var):
    """Integral of 1/f**k, f = x^2 + p x + q0 with negative discriminant, by the reduction formula."""
    s = q0 - p * p / 4
    u = sym(var) + _const(p / 2)
    root = _pow(_const(s), Fraction(-1, 2))                # 1/sqrt(s)
    res = fn('atan', u * root) * root
    for j in range(2, k + 1):
        res = u * _pow(fe, Fraction(1 - j)) * Fraction(1, 2 * (j - 1)) / s + res * Fraction(2 * j - 3, 2 * (j - 1)) / s
    return res


def _classify(v, var):
    """Kind of the var-dependent part v of a monomial (a tuple of (Atom, exponent))."""
    if all((a.kind == 's' and a.name == var and x.denominator == 1) or
           (a.kind == 'b' and x.denominator == 1 and _to_poly(a.arg, var) is not None) for a, x in v):
        return ('rat',)
    pk, frac, rest = 0, None, []
    for a, x in v:
        if a.kind == 's' and a.name == var:
            if x.denominator == 1 and x >= 0:
                pk = int(x)
            elif x.denominator != 1:
                frac = x
            else:
                return None
        else:
            rest.append((a, x))
    if frac is not None:
        return ('xpow', frac) if not rest else None
    if len(rest) == 1:
        a, x = rest[0]
        if a.kind == 'e' and x == 1:
            return ('exp', a, pk)
        if a.kind == 'f' and x == 1 and a.name in ('sin', 'cos'):
            return ('trig', a, pk)
        if a.kind == 'f' and x == 1 and a.name == 'log':
            return ('log', a, pk)
        if a.kind == 'b' and x.denominator != 1:
            return ('lpow', a, pk, x)
    return None


def _poly_integral(P, var):
    """Antiderivative of an Expr that is a polynomial in var (non-negative integer powers)."""
    out = ZERO
    x = sym(var)
    for mono, c in P.t.items():
        if mono == ():
            out = out + x * c
        else:
            a, e = mono[0]
            if len(mono) != 1 or a.name != var or e.denominator != 1 or e < 0:
                raise NotCovered('polynomial factor expected')
            out = out + _pow(x, e + 1) * (c / (e + 1))
    return out


def _int_group(G, var):
    """Antiderivative of an Expr G (rational coefficients) whose terms all depend on var."""
    rat = ZERO
    exps, trigs, logs, lpows = {}, {}, {}, {}
    out = ZERO
    x = sym(var)
    for v, c in G.t.items():
        cls = _classify(v, var) if v else ('rat',)
        if cls is None:
            raise NotCovered('integrand class not covered')
        kind = cls[0]
        if kind == 'rat':
            rat = rat + Expr({v: c})
        elif kind == 'xpow':
            out = out + _pow(x, cls[1] + 1) * (c / (cls[1] + 1))
        elif kind == 'exp':
            d = exps.setdefault(cls[1], ZERO)
            exps[cls[1]] = d + _pow(x, cls[2]) * c
        elif kind == 'trig':
            d = trigs.setdefault(cls[1].arg, [ZERO, ZERO])
            d[0 if cls[1].name == 'sin' else 1] = d[0 if cls[1].name == 'sin' else 1] + _pow(x, cls[2]) * c
        elif kind == 'log':
            d = logs.setdefault(cls[1], ZERO)
            logs[cls[1]] = d + _pow(x, cls[2]) * c
        elif kind == 'lpow':
            d = lpows.setdefault((cls[1], cls[3]), ZERO)
            lpows[(cls[1], cls[3])] = d + _pow(x, cls[2]) * c
    if rat.t:
        rf = _to_ratfunc(rat, var)
        if rf is None:
            raise NotCovered('rational part is not a rational function with rational coefficients')
        out = out + _int_ratfunc(rf[0], rf[1], var)
    for atom, P in exps.items():
        alpha = diff(atom.arg, var)
        if not alpha.t or alpha.has(var):
            raise NotCovered('exp of a non-linear argument')
        ia = _pow(alpha, Fraction(-1))
        s, term, d = ZERO, ia, P
        j = 0
        while d.t:
            s = s + d * term * (1 if j % 2 == 0 else -1)
            d = diff(d, var)
            term = term * ia
            j += 1
        out = out + _atom_expr(atom) * s
    for theta, (Ps, Pc) in trigs.items():
        alpha = diff(theta, var)
        if not alpha.t or alpha.has(var):
            raise NotCovered('sin/cos of a non-linear argument')
        ia = _pow(alpha, Fraction(-1))
        sn, cs = fn('sin', theta), fn('cos', theta)

        def sc(P):
            """(S, C) with S = sum (-1)^j P^(2j) / a^(2j+1), C = sum (-1)^j P^(2j+1) / a^(2j+2)."""
            S, C = ZERO, ZERO
            d, j = P, 0
            while d.t:
                sign = 1 if (j // 2) % 2 == 0 else -1
                if j % 2 == 0:
                    S = S + d * _pow(ia, j + 1) * sign
                else:
                    C = C + d * _pow(ia, j + 1) * sign
                d = diff(d, var)
                j += 1
            return S, C
        if Ps.t:
            S, C = sc(Ps)
            out = out + (-S) * cs + C * sn
        if Pc.t:
            S, C = sc(Pc)
            out = out + S * sn + C * cs
    for atom, P in logs.items():
        L = atom.arg
        Q = _poly_integral(P, var)
        out = out + Q * _atom_expr(atom)
        rest = Q * diff(L, var) * _pow(L, Fraction(-1))
        rf = _to_ratfunc(rest, var)
        if rf is None:
            raise NotCovered('log of a non-polynomial argument')
        out = out - _int_ratfunc(rf[0], rf[1], var)
    for (atom, r), P in lpows.items():
        lin = _to_poly(atom.arg, var)
        if lin is None or len(lin) != 2:
            raise NotCovered('power of a non-linear base')
        beta, alpha = lin
        u = atom.arg
        for mono, c in P.t.items():
            k = 0 if mono == () else int(mono[0][1])
            for i in range(k + 1):
                coef = Fraction(comb(k, i)) * (-beta) ** (k - i) / alpha ** k / alpha / (r + i + 1)
                out = out + _pow(u, r + i + 1) * (c * coef)
    return out


def antiderivative(e, var):
    """An antiderivative of the Expr e in var (no constant), verified by differentiating it back."""
    if not e.has(var):
        return e * sym(var)
    groups = {}
    for mono, c in e.t.items():
        k = tuple((a, x) for a, x in mono if not a.has(var))
        v = tuple((a, x) for a, x in mono if a.has(var))
        g = groups.setdefault(k, {})
        g[v] = g.get(v, 0) + c
    total = ZERO
    for k, vs in groups.items():
        part = _int_group(Expr({v: c for v, c in vs.items() if c}), var)
        total = total + part * Expr({k: Fraction(1)})
    if not is_zero(diff(total, var) - e):
        raise NotCovered('antiderivative could not be verified by differentiation')
    return total


# ----------------------------------------------------------------------------------------------------------------------------- definite integrals

def _sturm_count(p, lo, hi):
    """Number of distinct real roots of the polynomial p in the half-open interval (lo, hi] (lo, hi rationals or None for -/+ infinity)."""
    p = _pn(p)
    if len(p) <= 1:
        return 0
    seq = [p, _pder(p)]
    while True:
        r = _pdivmod(seq[-2], seq[-1])[1]
        if not r:
            break
        seq.append(_pscale(r, -1))
    g = seq[-1]
    seq = [_pdivmod(s, g)[0] for s in seq]            # square-free: divide by the gcd

    def sign_changes(x):
        signs = []
        for s in seq:
            v = _peval(s, x)
            if v:
                signs.append(1 if v > 0 else -1)
        return sum(1 for i in range(len(signs) - 1) if signs[i] != signs[i + 1])

    def sign_changes_inf(neg):
        signs = []
        for s in seq:
            if s:
                v = s[-1] * ((-1) ** (len(s) - 1) if neg else 1)
                signs.append(1 if v > 0 else -1)
        return sum(1 for i in range(len(signs) - 1) if signs[i] != signs[i + 1])
    a = sign_changes_inf(True) if lo is None else sign_changes(lo)
    b = sign_changes_inf(False) if hi is None else sign_changes(hi)
    return a - b


def _singular_polys(e, var):
    """Polynomials (in var) that must not vanish on the interior of an interval of integration: bases of negative or fractional powers and log arguments; None if one is not polynomial."""
    out = []
    for a in e.atoms():
        if a.kind == 's' and a.name == var:
            continue
        if a.kind == 'b' and a.arg.has(var):
            p = _to_poly(a.arg, var)
            if p is None:
                return None
            out.append(p)
        elif a.kind == 'f' and a.name == 'log' and a.arg.has(var):
            p = _to_poly(a.arg, var)
            if p is None:
                return None
            out.append(p)
        elif a.kind == 'f' and a.name in ('tan', 'floor', 'abs', 'atan') and a.arg.has(var):
            if a.name in ('tan', 'floor'):
                return None
    for mono in e.t:
        for a, x in mono:
            if a.kind == 's' and a.name == var and (x < 0 or x.denominator != 1):
                out.append([Fraction(0), Fraction(1)])
    return out


def definite_integral(e, var, lo, hi):
    """Integral of the Expr e over [lo, hi] (constant Exprs, or None for -/+ infinity) as an Expr; NotCovered for poles inside, divergence or what the antiderivative does not cover."""
    qlo = lo.as_fraction() if lo is not None else None
    qhi = hi.as_fraction() if hi is not None else None
    if lo is not None and hi is not None:
        d = hi - lo
        if d.is_zero():
            return ZERO
        if _const_sign(d) < 0:
            return -definite_integral(e, var, hi, lo)
    sing = _singular_polys(e, var)
    if sing is None:
        raise NotCovered('integrand with tan/floor or a non-polynomial base: pole check not covered')
    if sing and ((lo is not None and qlo is None) or (hi is not None and qhi is None)):
        raise NotCovered('irrational limits with a singular integrand')
    for p in sing:
        n = _sturm_count(p, qlo, qhi)
        inside = n
        if qhi is not None and _peval(p, qhi) == 0:
            inside -= 1
        if inside > 0:
            raise NotCovered('a pole or branch point inside the interval')
    F = antiderivative(e, var)
    if sing:
        mid = (qlo + qhi) / 2 if qlo is not None and qhi is not None else (qhi - 1 if qhi is not None else (qlo + 1 if qlo is not None else Fraction(0)))
        for m in e.t:
            for a, x in m:
                if a.kind == 'b' and a.arg.has(var) and x.denominator != 1:
                    v = _peval(_to_poly(a.arg, var), mid)
                    if v <= 0:
                        raise NotCovered('integrand not real on the interval')
                if a.kind == 's' and a.name == var and x.denominator != 1 and mid <= 0:
                    raise NotCovered('integrand not real on the interval')
                if a.kind == 'f' and a.name == 'log' and a.arg.has(var) and _peval(_to_poly(a.arg, var), mid) <= 0:
                    raise NotCovered('integrand not real on the interval')

    def at(b, side):
        if b is None:
            return limit_expr(F, var, 'oo' if side > 0 else '-oo')
        try:
            return subs(F, var, b)
        except (BadInput, ZeroDivisionError):
            q = b.as_fraction()
            if q is None:
                raise NotCovered('antiderivative undefined at an irrational limit')
            return limit_expr(F, var, q, '-' if side > 0 else '+')
    top = at(hi, 1)
    bot = at(lo, -1)
    for v in (top, bot):
        if isinstance(v, str):
            raise NotCovered('divergent integral')
    return top - bot


# ----------------------------------------------------------------------------------------------------------------------------- sums

def _bernoulli_plus(n):
    """B_0..B_n with B_1 = +1/2 (the convention of sums of powers)."""
    B = [Fraction(0)] * (n + 1)
    B[0] = Fraction(1)
    for m in range(1, n + 1):
        B[m] = -sum(comb(m + 1, j) * B[j] for j in range(m)) / (m + 1)
    if n >= 1:
        B[1] = Fraction(1, 2)
    return B


def _power_sum_poly(p):
    """Coefficient list (in n) of S_p(n) = sum_{k=0}^{n} k**p, with 0**0 = 1."""
    if p == 0:
        return [Fraction(1), Fraction(1)]
    if p > 60:
        raise NotCovered('power above 60')
    B = _bernoulli_plus(p)
    out = [Fraction(0)] * (p + 2)
    for j in range(p + 1):
        out[p + 1 - j] += Fraction(comb(p + 1, j)) * B[j] / (p + 1)
    return _pn(out)


def _frac_eval(e, env):
    """The Fraction value of e with env (symbol -> Fraction), or None when some atom is not rational arithmetic."""
    total = Fraction(0)
    for mono, c in e.t.items():
        v = c
        for a, x in mono:
            if a.kind == 's':
                if a.name not in env:
                    return None
                b = env[a.name]
            elif a.kind == 'b':
                b = _frac_eval(a.arg, env)
                if b is None:
                    return None
            elif a.kind == 'f' and a.name in ('abs', 'floor'):
                b = _frac_eval(a.arg, env)
                if b is None:
                    return None
                b = abs(b) if a.name == 'abs' else Fraction(b.numerator // b.denominator)
            else:
                return None
            if x.denominator != 1:
                return None
            if b == 0 and x < 0:
                raise BadInput('division by zero')
            v = v * b ** int(x)
        total += v
    return total


def sum_range_value(e, var, lo, hi):
    """The exact sum of e over the integers lo..hi (a Fraction)."""
    if hi < lo:
        return Fraction(0)
    if hi - lo + 1 > MAX_SUM_TERMS:
        raise NotCovered('more than %d terms' % MAX_SUM_TERMS)
    total = Fraction(0)
    for k in range(lo, hi + 1):
        v = _frac_eval(e, {var: Fraction(k)})
        if v is None:
            v = subs(e, var, _const(k)).as_fraction()
            if v is None:
                raise NotCovered('a term is not a rational number')
        total += v
    return total


def _geometric_part(G, P, k):
    """(T) with T(k+1) - T(k) = G(k) P(k) for G(k) = c * r**k (exp/(-1)^k atoms); returns (T, R)."""
    shifted = subs(G, k, sym(k) + 1)
    R = shifted * _pow(G, Fraction(-1))
    if R.has(k):
        raise NotCovered('ratio of consecutive terms is not constant')
    if is_zero(R - 1):
        raise NotCovered('ratio 1')
    d = 0
    coeffs = []
    pp = _to_poly(P, k)
    if pp is None:
        raise NotCovered('polynomial factor expected')
    d = len(pp) - 1
    q = [ZERO] * (d + 1)
    inv = _pow(R - 1, Fraction(-1))
    for j in range(d, -1, -1):
        acc = _const(pp[j]) if j < len(pp) else ZERO
        for i in range(j + 1, d + 1):
            acc = acc - R * q[i] * comb(i, j)
        q[j] = acc * inv
    Q = ZERO
    for j in range(d + 1):
        Q = Q + q[j] * _pow(sym(k), j)
    T = G * Q
    if not is_zero(subs(T, k, sym(k) + 1) - T - G * P):
        raise NotCovered('geometric sum: check failed')
    return T, R


def sum_closed_expr(e, k, lo, hi):
    """Closed form of sum_{k=lo}^{hi} e as an Expr. lo, hi: Expr; hi None for infinity."""
    if not e.has(k):
        if hi is None:
            raise NotCovered('constant summand to infinity')
        return e * (hi - lo + 1)
    lo_q = lo.as_fraction()
    hi_q = hi.as_fraction() if hi is not None else None
    if lo_q is not None and hi_q is not None and lo_q.denominator == 1 and hi_q.denominator == 1:
        return _const(sum_range_value(e, k, int(lo_q), int(hi_q)))
    if hi is not None and (lo.has(k) or hi.has(k)):
        raise NotCovered('bounds contain the summation variable')
    groups = {}
    for mono, c in e.t.items():
        kk = tuple((a, x) for a, x in mono if not a.has(k))
        v = tuple((a, x) for a, x in mono if a.has(k))
        g = groups.setdefault(kk, {})
        g[v] = g.get(v, 0) + c
    total = ZERO
    for kk, vs in groups.items():
        total = total + _sum_group(Expr({v: c for v, c in vs.items() if c}), k, lo, hi) * Expr({kk: Fraction(1)})
    return total


def _sum_group(G, k, lo, hi):
    poly, geo, rat = ZERO, {}, ZERO
    for v, c in G.t.items():
        if all(a.kind == 's' and a.name == k and x.denominator == 1 and x >= 0 for a, x in v):
            poly = poly + Expr({v: c})
        elif all((a.kind == 's' and a.name == k and x.denominator == 1) or (a.kind == 'b' and x.denominator == 1 and _to_poly(a.arg, k) is not None) for a, x in v):
            rat = rat + Expr({v: c})
        else:
            ps = tuple((a, x) for a, x in v if a.kind == 's')
            gs = tuple((a, x) for a, x in v if a.kind in 'em')
            if len(ps) + len(gs) != len(v) or not gs or any(x.denominator != 1 or x < 0 for a, x in ps) or any(x != 1 for a, x in gs):
                raise NotCovered('summand class not covered')
            key = Expr({gs: Fraction(1)})
            geo[key] = geo.get(key, ZERO) + Expr({ps: c})
    out = ZERO
    if rat.t:
        rf = _to_ratfunc(rat, k)
        if rf is None:
            raise NotCovered('rational summand with non-rational coefficients')
        q, terms = _apart_terms(*rf)
        poly = poly + _pexpr(q, k)
        out = out + _telescoping(terms, k, lo, hi)
    if poly.t:
        pp = _to_poly(poly, k)
        if hi is None:
            raise NotCovered('polynomial summand diverges')
        F = ZERO
        for j, c in enumerate(pp):
            if c:
                F = F + _pexpr(_power_sum_poly(j), '_n') * c
        out = out + subs(F, '_n', hi) - subs(F, '_n', lo - 1)
    for Gk, P in geo.items():
        T, R = _geometric_part(Gk, P, k)
        if hi is None:
            ar = abs(_decimal_value(R, {}, 40))
            if ar >= 1:
                raise NotCovered('geometric series does not converge')
            out = out - subs(T, k, lo)
        else:
            out = out + subs(T, k, hi + 1) - subs(T, k, lo)
    return out


def _telescoping(terms, k, lo, hi):
    """Closed form of the sum of c/(k - r)**m terms when they telescope; NotCovered otherwise."""
    classes = {}
    for a, f, m in terms:
        if len(f) != 2:
            raise NotCovered('summand with a non-linear irreducible denominator')
        if len(a) != 1:
            raise NotCovered('summand: unexpected numerator')
        w = -f[0]                                   # the factor is (k + w)... f = x - r with f[0] = -r, so k - r = k + f[0]
        w = f[0]
        classes.setdefault((m, w - (w.numerator // w.denominator)), []).append((w, a[0]))
    lo_q = lo.as_fraction()
    out = ZERO
    for (m, _), items in classes.items():
        if sum(c for _, c in items) != 0:
            raise NotCovered('summand does not telescope (harmonic-type sum)')
        w0 = min(w for w, _ in items)
        for w, c in items:
            s = w - w0
            if s.denominator != 1 or s > 200:
                raise NotCovered('telescoping shift too large')
            s = int(s)
            # poles inside the range
            if lo_q is None:
                if w.denominator == 1:
                    raise NotCovered('pole position against a symbolic lower bound is unknown')
            else:
                r = -w
                if r.denominator == 1 and r >= lo_q:
                    raise NotCovered('a pole inside the range of summation')
            for i in range(s):
                if hi is not None:
                    out = out + _pow(hi + (1 + i) + w0, Fraction(-m)) * c
                out = out - _pow(lo + i + w0, Fraction(-m)) * c
    return out


# ----------------------------------------------------------------------------------------------------------------------------- solving one equation

def replace_atoms(e, repl):
    """Rebuild e with atoms in repl (Atom -> Expr) replaced, descending into arguments."""
    out = ZERO
    for mono, c in e.t.items():
        term = _const(c)
        for a, x in mono:
            term = term * _replace_atom(a, x, repl)
        out = out + term
    return out


def _replace_atom(a, x, repl):
    if a in repl:
        return _pow(repl[a], x)
    if a.kind in 'skr':
        return _atom_expr(a, x)
    arg = replace_atoms(a.arg, repl)
    if arg == a.arg:
        return _atom_expr(a, x)
    if a.kind == 'f':
        base = fn(a.name, arg)
    elif a.kind == 'e':
        base = _exp(arg)
    elif a.kind == 'b':
        base = arg
    else:
        base = _sign_pow(arg)
    return _pow(base, x)


def _decimal_of_const(e, digits=40):
    return _decimal_value(e, {}, digits)


def _exact_sign(e):
    """Sign of a constant Expr: exact for rationals, by 60-digit evaluation otherwise (0 only for the zero Expr)."""
    return _const_sign(e)


def _solve_poly_q(p):
    """Real roots (Exprs) of a polynomial over Q: rational roots and the roots of a quadratic cofactor; NotCovered for a cofactor of degree 3 or more."""
    p = _pn(p)
    if not p:
        raise NotCovered('infinitely many solutions (the equation holds on a whole interval or for every value)')
    roots, rest = _rational_roots(p)
    out = [_const(r) for r, _ in roots]
    if len(rest) == 3:
        c, b, a = rest
        disc = b * b - 4 * a * c
        if disc > 0:
            sq = _pow(_const(disc), Fraction(1, 2))
            out.append((_const(-b) + sq) * (1 / (2 * a)))
            out.append((_const(-b) - sq) * (1 / (2 * a)))
    elif len(rest) > 3:
        raise NotCovered('polynomial of degree 3 or more without rational roots')
    return out


def _solve_rational(F, var):
    rf = _to_ratfunc(F, var)
    if rf is None:
        raise NotCovered('equation is not a rational function of the unknown with rational coefficients')
    return _solve_poly_q(rf[0])


def _const_atoms_only(mono, var):
    return all(not a.has(var) for a, _ in mono)


def _solve_exp(F, var):
    forms = {}
    for mono, c in F.t.items():
        ea = [a for a, x in mono if a.kind == 'e' and a.has(var)]
        others = [(a, x) for a, x in mono if not (a.kind == 'e' and a.has(var))]
        if any(a.has(var) for a, _ in others) or len(ea) > 1:
            raise NotCovered('exp mixed with other functions of the unknown')
        const = Expr({tuple(others): c})
        forms.setdefault(ea[0] if ea else None, ZERO)
        forms[ea[0] if ea else None] = forms[ea[0] if ea else None] + const
    args = [a.arg for a in forms if a is not None]
    alphas = [diff(g, var) for g in args]
    if any((not al.t) or al.has(var) for al in alphas):
        raise NotCovered('exp of a non-linear argument')
    mu = alphas[0]
    qs = []
    for al in alphas:
        r = (al * _pow(mu, Fraction(-1))).as_fraction()
        if r is None:
            raise NotCovered('exponents not rational multiples of each other')
        qs.append(r)
    d = 1
    for q in qs:
        d = d * q.denominator // gcd(d, q.denominator)
    poly = {}
    for (a, ex), q in zip([(a, None) for a in forms if a is not None], qs):
        n = int(q * d)
        beta = a.arg - diff(a.arg, var) * sym(var)
        poly[n] = poly.get(n, ZERO) + forms[a] * _exp(beta)
    if None in forms:
        poly[0] = poly.get(0, ZERO) + forms[None]
    lowest = min(poly)
    coeffs = {n - lowest: c for n, c in poly.items() if c.t}
    if not coeffs:
        raise NotCovered('infinitely many solutions (the equation holds on a whole interval or for every value)')
    deg = max(coeffs)
    ys = []
    if all(c.as_fraction() is not None for c in coeffs.values()):
        ys = _solve_poly_q([coeffs.get(i, ZERO).as_fraction() or Fraction(0) for i in range(deg + 1)])
    elif deg <= 1 and set(coeffs) <= {0, 1}:
        if deg == 0:
            return []
        ys = [-coeffs.get(0, ZERO) * _pow(coeffs[1], Fraction(-1))]
    else:
        raise NotCovered('coefficients are not rational')
    out = []
    for y in ys:
        if _exact_sign(y) > 0:
            out.append(_log(y) * d * _pow(mu, Fraction(-1)))
    return out


def _solve_log(F, var):
    logs, C0 = {}, ZERO
    for mono, c in F.t.items():
        la = [(a, x) for a, x in mono if a.kind == 'f' and a.name == 'log' and a.has(var)]
        if not la:
            if any(a.has(var) for a, _ in mono):
                raise NotCovered('log mixed with other functions of the unknown')
            C0 = C0 + Expr({mono: c})
            continue
        if len(la) != 1 or la[0][1] != 1 or len(mono) != 1:
            raise NotCovered('log mixed with other factors')
        logs[la[0][0]] = logs.get(la[0][0], 0) + c
    items = [(a, c) for a, c in logs.items() if c]
    if not items:
        raise NotCovered('log terms cancel')
    polys = []
    for a, c in items:
        g = _to_poly(a.arg, var)
        if g is None:
            raise NotCovered('log of a non-polynomial argument')
        polys.append((g, c))
    den = 1
    for _, c in polys:
        den = den * c.denominator // gcd(den, c.denominator)
    T = _exp(-C0 * den)                                   # prod g^(c*den) = exp(-C0*den)
    tq = T.as_fraction()
    pos, neg = [Fraction(1)], [Fraction(1)]
    for g, c in polys:
        n = int(c * den)
        if n > 0:
            pos = _pmul(pos, _ppow(g, n))
        else:
            neg = _pmul(neg, _ppow(g, -n))
    if tq is not None:
        cands = _solve_poly_q(_psub(pos, _pscale(neg, tq)))
    elif len(polys) == 1 and len(polys[0][0]) == 2 and int(polys[0][1] * den) == 1:
        g = polys[0][0]
        cands = [(T - _const(g[0])) * (1 / g[1])]
    else:
        raise NotCovered('log equation with an irrational right-hand side')
    out = []
    for x0 in cands:
        ok = True
        for g, _ in polys:
            val = subs(_pexpr(g, var), var, x0)
            if val.is_zero() or _exact_sign(val) <= 0:
                ok = False
        if ok:
            out.append(x0)
    return out


def _solve_abs(F, var):
    absatoms = [a for a in F.atoms() if a.kind == 'f' and a.name == 'abs' and a.arg.has(var)]
    polys = []
    for a in absatoms:
        g = _to_poly(a.arg, var)
        if g is None:
            raise NotCovered('abs of a non-polynomial argument')
        polys.append(g)
    bps = set()
    for g in polys:
        roots, rest = _rational_roots(g)
        for r, _ in roots:
            bps.add(r)
        if len(rest) == 3 and rest[1] ** 2 - 4 * rest[0] * rest[2] < 0:
            continue
        if len(rest) > 1:
            raise NotCovered('abs argument with real irrational zeros')
    bps = sorted(bps)
    bounds = [None] + bps + [None]
    out = []
    for i in range(len(bounds) - 1):
        lo, hi = bounds[i], bounds[i + 1]
        if lo is None and hi is None:
            s = Fraction(0)
        elif lo is None:
            s = hi - 1
        elif hi is None:
            s = lo + 1
        else:
            s = (lo + hi) / 2
        repl = {}
        for a, g in zip(absatoms, polys):
            sign = 1 if _peval(g, s) > 0 else -1
            repl[a] = a.arg * sign
        Fr = replace_atoms(F, repl)
        for x0 in _solve_equation_expr(Fr, var):
            ok = True
            for bound, sgn in ((lo, 1), (hi, -1)):
                if bound is not None:
                    d = x0 - _const(bound)
                    if not d.is_zero() and _exact_sign(d) * sgn < 0:
                        ok = False
            if ok and not any(is_zero(x0 - y) for y in out):
                out.append(x0)
    return out


def _solve_equation_expr(F, var):
    atoms = [a for a in F.atoms() if a.has(var)]
    kinds = {(a.kind, a.name if a.kind == 'f' else None) for a in atoms}
    if ('f', 'abs') in kinds:
        return _solve_abs(F, var)
    if any(k[0] == 'e' for k in kinds):
        return _solve_exp(F, var)
    if ('f', 'log') in kinds:
        return _solve_log(F, var)
    if any(k[0] == 'f' for k in kinds) or any(k[0] == 'm' for k in kinds):
        raise NotCovered('equation with trigonometric or floor terms')
    for mono in F.t:
        for a, x in mono:
            if (a.kind == 's' and a.name == var and x.denominator != 1) or (a.kind == 'b' and a.has(var) and x.denominator != 1):
                raise NotCovered('equation with a radical of the unknown')
    return _solve_rational(F, var)


def solve_expr(lhs, rhs, var):
    """The real solutions of lhs = rhs in var as a list of Exprs sorted by value; verified by substitution."""
    F = lhs - rhs
    if not F.has(var):
        if F.is_zero():
            raise NotCovered('an identity: every value solves it')
        return []
    roots = _solve_equation_expr(F, var)
    keep = []
    for r in roots:
        if any(is_zero(r - y) for y in keep):
            continue
        val = subs(F, var, r)
        if not is_zero(val):
            if val.symbols():
                raise NotCovered('root check failed')
            if abs(_decimal_value(val, {}, 60)) > Decimal(10) ** -35:
                raise NotCovered('root check failed')
        keep.append(r)
    keep.sort(key=lambda r: _decimal_value(r, {}, 40))
    return keep


# ----------------------------------------------------------------------------------------------------------------------------- registered operations

KIND = 'symbolic algebra'


def _sop(name, bound):
    """Register an operation; runaway recursion, huge numbers and a zero division inside it become NotCovered / BadInput, never a crash."""
    def deco(fn):
        @functools.wraps(fn)
        def wrapper(*a, **k):
            try:
                return fn(*a, **k)
            except RecursionError:
                raise NotCovered('too deeply nested')
            except (OverflowError, MemoryError, InvalidOperation):
                raise NotCovered('too large or out of the numeric range')
            except ZeroDivisionError:
                raise BadInput('division by zero')
        return op(name, kind=KIND, bound=bound)(wrapper)
    return deco


_VAR_RE = re.compile(r'^[A-Za-z](?:_[A-Za-z0-9]+)?$')
_INF_WORDS = {'oo': 1, '+oo': 1, 'inf': 1, '+inf': 1, 'infinity': 1, '\\infty': 1, '+\\infty': 1, '-oo': -1, '-inf': -1, '-infinity': -1, '-\\infty': -1}


def _expr(x, e_is_euler=False):
    if isinstance(x, Expr):
        return x
    if isinstance(x, bool):
        raise BadInput('a number or expression text is expected')
    if isinstance(x, (int, Fraction)):
        return _const(x)
    return parse(x, e_is_euler)


def _var(v):
    if not isinstance(v, str) or not _VAR_RE.match(v):
        raise BadInput(f'bad variable name {v!r}')
    return v


def _bound(x):
    """Fraction/Expr bound, or None for +/- infinity given as text (second value: the sign of the infinity)."""
    if isinstance(x, str) and x.strip() in _INF_WORDS:
        return None, _INF_WORDS[x.strip()]
    return _expr(x), 0


@_sop('symbolic.canon', 'text of at most 4000 tokens, at most 4000 terms, integer power of a sum up to 200')
def canon(expr, e_is_euler=False):
    """Canonical text of an expression: multiplied out, like terms collected, prime radicals and exp merged; parse(canon(x)) == parse(x).
    canon('(x+1)^2 - x^2') -> '2*x + 1'"""
    return to_text(_expr(expr, e_is_euler))


@_sop('symbolic.expand', 'as symbolic.canon')
def expand(expr, e_is_euler=False):
    """Expand products and integer powers of sums (the canonical form is the expanded form).
    expand('(x+1)^3') -> 'x^3 + 3*x^2 + 3*x + 1'"""
    return to_text(_expr(expr, e_is_euler))


@_sop('symbolic.substitute', 'as symbolic.canon')
def substitute(expr, var, value, e_is_euler=False):
    """Replace a symbol by an expression and return the canonical form.
    substitute('x^2 + y', 'x', 'y + 1') -> 'y^2 + 3*y + 1'"""
    return to_text(subs(_expr(expr, e_is_euler), _var(var), _expr(value, e_is_euler)))


def _values(values, e_is_euler):
    out = {}
    for k, v in (values or {}).items():
        out[_var(k)] = _expr(v, e_is_euler)
    return out


@_sop('symbolic.evaluate', 'as symbolic.canon; NotCovered unless the value is rational')
def evaluate(expr, values=None, e_is_euler=False):
    """Exact value (Fraction) of an expression whose symbols all have values; irrational results are NotCovered.
    evaluate('x^2 + 1/3', {'x': '1/2'}) -> Fraction(7, 12)"""
    e = _expr(expr, e_is_euler)
    r = subs_many(e, _values(values, e_is_euler))
    if r.symbols():
        raise BadInput('free symbols: ' + ', '.join(sorted(r.symbols())))
    q = r.as_fraction()
    if q is None:
        raise NotCovered('the value is not rational: ' + to_text(r))
    return q


@_sop('symbolic.evaluate_numeric', 'as symbolic.canon; digits up to 200')
def evaluate_numeric(expr, values=None, digits=30, e_is_euler=False):
    """Decimal value with `digits` significant digits (numeric, not exact).
    evaluate_numeric('sin(1)', digits=10) -> Decimal('0.8414709848')"""
    if not isinstance(digits, int) or not 1 <= digits <= 200:
        raise BadInput('digits must be between 1 and 200')
    e = _expr(expr, e_is_euler)
    vals = _values(values, e_is_euler)
    for q in vals.values():
        if q.symbols():
            raise BadInput('values must be constants')
    r = subs_many(e, vals)                         # exact substitution first, so floor(1/3*9) is floor(3), not floor(2.999...)
    if r.symbols():
        raise BadInput('free symbols: ' + ', '.join(sorted(r.symbols())))
    return _decimal_value(r, {}, digits)


@_sop('symbolic.simplify', 'as symbolic.canon; one variable for the reduction to lowest terms')
def simplify(expr, e_is_euler=False):
    """Normal form: canonical expanded form, and with one variable the rational part reduced to lowest terms. No identities beyond the form are applied.
    simplify('(x^2-1)/(x-1)') -> 'x + 1'"""
    e = _expr(expr, e_is_euler)
    r = simplify_expr(e)
    syms = r.symbols()
    if len(syms) == 1 and not r.atoms() - {a for m in r.t for a, _ in m if a.kind in 'sb'}:
        var = next(iter(syms))
        rf = _to_ratfunc(r, var)
        if rf is not None and len(rf[1]) > 1:
            n, d = _integer_form(*rf)
            nt, dt = to_text(_pexpr(n, var)), to_text(_pexpr(d, var))
            return f'({nt})/({dt})' if len(n) > 1 and (len([c for c in n if c]) > 1) else f'{nt}/({dt})'
    return to_text(r)


@_sop('symbolic.rational_form', 'rational function of one variable with rational coefficients, degrees up to the term limit')
def rational_form(expr, var, e_is_euler=False):
    """(numerator, denominator) texts of a rational function in lowest terms: integer coefficients, jointly primitive, positive leading denominator.
    rational_form('1/x + 1/(x+1)', 'x') -> ('2*x + 1', 'x^2 + x')"""
    var = _var(var)
    e = _expr(expr, e_is_euler)
    if e.symbols() - {var}:
        raise NotCovered('other symbols present')
    rf = _to_ratfunc(e, var)
    if rf is None:
        raise NotCovered('not a rational function of the variable with rational coefficients')
    n, d = _integer_form(*rf)
    return to_text(_pexpr(n, var)), to_text(_pexpr(d, var))


@_sop('symbolic.equal', 'as symbolic.canon')
def equal(a, b, e_is_euler=False):
    """True when the two expressions are equal as functions. False only where the form is complete (polynomials, rational functions, radicals); NotCovered when
    function atoms (sin, exp, log, ...) leave the difference non-zero, because an unknown identity might remove it.
    equal('(x+1)^2', 'x^2+2x+1') -> True"""
    d = _expr(a, e_is_euler) - _expr(b, e_is_euler)
    if is_zero(d):
        return True
    if _is_free(d):
        return False
    raise NotCovered('not shown equal; an identity between function atoms may still hold')


@_sop('symbolic.diff', 'as symbolic.canon; order up to 20')
def derivative(expr, var, order=1, e_is_euler=False):
    """Derivative with respect to var (abs and floor: NotCovered where undefined).
    diff('x^3*sin(x)', 'x') -> 'x^3*cos(x) + 3*x^2*sin(x)'"""
    if not isinstance(order, int) or not 0 <= order <= 20:
        raise BadInput('order must be 0..20')
    return to_text(diff(_expr(expr, e_is_euler), _var(var), order))


@_sop('symbolic.apart', 'rational function of one variable; denominator degree up to 40; factors without rational roots must have degree 2 or 3')
def apart(expr, var, e_is_euler=False):
    """Partial fractions over Q; the pieces are checked to add back to the input.
    apart('1/(x^2-1)', 'x') -> '1/(2*(x - 1)) - 1/(2*(x + 1))'"""
    return to_text(apart_expr(_expr(expr, e_is_euler), _var(var)))


@_sop('symbolic.series_coefficients', 'Taylor order up to 40, point rational')
def series_coefficients(expr, var, point=0, order=5, e_is_euler=False):
    """[c_0..c_order] with expr = sum c_k (var - point)^k + O((var - point)^(order+1)); NotCovered when not analytic at the point.
    series_coefficients('exp(x)', 'x', 0, 3) -> ['1', '1', '1/2', '1/6']"""
    pt = _expr(point, e_is_euler).as_fraction()
    if pt is None:
        raise NotCovered('point must be rational')
    return [to_text(c) for c in taylor_coefficients(_expr(expr, e_is_euler), _var(var), pt, order)]


@_sop('symbolic.taylor', 'Taylor order up to 40, point rational')
def taylor(expr, var, point=0, order=5, e_is_euler=False):
    """Taylor polynomial of degree `order` about `point`, expanded in var.
    taylor('cos(x)', 'x', 0, 4) -> '1/24*x^4 - 1/2*x^2 + 1'  (printed in canonical order)"""
    var = _var(var)
    pt = _expr(point, e_is_euler).as_fraction()
    if pt is None:
        raise NotCovered('point must be rational')
    cs = taylor_coefficients(_expr(expr, e_is_euler), var, pt, order)
    out = ZERO
    for k, c in enumerate(cs):
        out = out + c * _pow(sym(var) - _const(pt), k)
    return to_text(out)


def _point(p):
    if isinstance(p, str) and p.strip() in _INF_WORDS:
        return 'oo' if _INF_WORDS[p.strip()] > 0 else '-oo'
    q = _expr(p).as_fraction()
    if q is None:
        raise NotCovered('the point must be rational or infinity')
    return q


@_sop('symbolic.limit', 'series of up to 48 terms; no exp of an unbounded series with positive growth, no x^x')
def limit(expr, var, point, direction='both', e_is_euler=False):
    """Limit by series expansion: text of the value, or 'oo' / '-oo'. point: rational or 'oo' / '-oo'; direction: 'both', '+' or '-' for a finite point.
    limit('sin(x)/x', 'x', 0) -> '1'"""
    if direction not in ('both', '+', '-'):
        raise BadInput('direction must be both, + or -')
    r = limit_expr(_expr(expr, e_is_euler), _var(var), _point(point), direction)
    return r if isinstance(r, str) else to_text(r)


@_sop('symbolic.integrate', 'classes: polynomials, rational functions with irreducible factors of degree <= 2, c*x^k*exp(ax+b), x^k*sin/cos(ax+b), x^k*log(poly), x^k*(ax+b)^r; verified by differentiation')
def integrate(expr, var, e_is_euler=False):
    """An antiderivative (no constant), verified by differentiating it back; log|.| is written log(abs(.)).
    integrate('x*exp(x)', 'x') -> 'x*exp(x) - exp(x)'"""
    return to_text(antiderivative(_expr(expr, e_is_euler), _var(var)))


@_sop('symbolic.integrate_definite', 'as symbolic.integrate; limits rational or oo; no pole or branch point inside the interval')
def integrate_definite(expr, var, lo, hi, e_is_euler=False):
    """Exact definite integral over [lo, hi] (improper at an endpoint or at infinity through limits); NotCovered for a pole inside or a divergent integral.
    integrate_definite('x^2', 'x', 0, 3) -> '9'"""
    def b(x):
        if isinstance(x, str) and x.strip() in _INF_WORDS:
            return None
        v = _expr(x)
        if v.symbols():
            raise NotCovered('limits must be constants')
        return v
    return to_text(definite_integral(_expr(expr, e_is_euler), _var(var), b(lo), b(hi)))


@_sop('symbolic.sum_closed', 'polynomial, geometric (rational ratio) times polynomial, telescoping rational summands; numeric bounds by evaluation up to 20000 terms')
def sum_closed(expr, var, lo, hi, e_is_euler=False):
    """Closed form of sum_{var=lo}^{hi} expr; hi may be 'oo' (convergent geometric or telescoping sums). The symbol in (-1)^k is taken to be an integer.
    sum_closed('k^2', 'k', 1, 'n') -> '1/3*n^3 + 1/2*n^2 + 1/6*n'"""
    var = _var(var)
    lo_e, lo_inf = _bound(lo)
    hi_e, hi_inf = _bound(hi)
    if lo_e is None or hi_inf < 0 or lo_inf:
        raise NotCovered('infinite lower bound')
    return to_text(sum_closed_expr(_expr(expr, e_is_euler), var, lo_e, hi_e))


@_sop('symbolic.sum_range', 'at most 20000 terms; each term must be rational')
def sum_range(expr, var, lo, hi, e_is_euler=False):
    """Exact sum (Fraction) over the integers lo..hi by evaluating every term.
    sum_range('1/(k*(k+1))', 'k', 1, 9) -> Fraction(9, 10)"""
    if not isinstance(lo, int) or not isinstance(hi, int) or isinstance(lo, bool) or isinstance(hi, bool):
        raise BadInput('lo and hi must be integers')
    return sum_range_value(_expr(expr, e_is_euler), _var(var), lo, hi)


@_sop('symbolic.solve', 'real solutions of one equation: rational function (roots rational or quadratic), polynomial in exp(mu*x), linear in logs, or with abs of polynomials with rational zeros')
def solve(equation, var, e_is_euler=False):
    """Real solutions of `lhs = rhs` in var as canonical texts sorted by value; each is checked by substituting it back. [] means no real solution.
    solve('x^2 - 3*x + 2 = 0', 'x') -> ['1', '2']"""
    lhs, rhs = parse_equation(equation, e_is_euler)
    return [to_text(r) for r in solve_expr(lhs, rhs, _var(var))]
