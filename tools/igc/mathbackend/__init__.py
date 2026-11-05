"""MIP math backend: the registry every module of tools/igc/mathbackend registers its operations in, the two errors, and Surd, an exact `p + q*sqrt(d)`.

See docs/plans/2026-11-05-mip-math-backend-coverage.md for the contract. Standard library only: nothing in this package imports a third-party library
(SymPy, mpmath and NetworkX appear only in sim_*.py and test files, as the independent oracle).

    from tools.igc.mathbackend import op, run, registry, NotCovered, BadInput, Surd

    @op('numbers.gcd', kind='number theory', bound='any size')
    def gcd(*xs): ...

    run('numbers.gcd', 12, 18)      # 6
"""
import importlib
from decimal import Decimal, localcontext
from fractions import Fraction
from math import isqrt


class NotCovered(Exception):
    """The operation does not cover this input (or the input is past its bound). The reason is the message; the caller must not guess a value."""


class BadInput(ValueError):
    """The input is malformed or outside the operation's domain (a division by zero, a negative count, ...)."""


_REG = {}
MODULES = ('numbers', 'algebra', 'counting', 'geometry', 'search', 'symbolic')


def op(name, kind, bound=''):
    """Register a function as an operation `name` ('<module>.<name>') of a family `kind`, with the limit it enforces in `bound`."""
    def deco(fn):
        if name in _REG:
            raise ValueError(f'operation {name} registered twice')
        _REG[name] = dict(fn=fn, kind=kind, bound=bound, doc=(fn.__doc__ or '').strip())
        fn.op_name = name
        return fn
    return deco


def load_all():
    """Import every module that exists, so that its operations register. Modules not written yet are skipped."""
    for m in MODULES:
        try:
            importlib.import_module(f'{__name__}.{m}')
        except ModuleNotFoundError as e:
            if e.name != f'{__name__}.{m}':
                raise


def registry():
    load_all()
    return dict(_REG)


def run(name, *args, **kw):
    load_all()
    if name not in _REG:
        raise NotCovered(f'no operation {name}')
    return _REG[name]['fn'](*args, **kw)


def _squarefree(d):
    """(s, r) with d = s*s*r and r square-free, by trial division (d is small in practice)."""
    if d < 0:
        raise BadInput('negative radicand')
    if d == 0:
        return 0, 0
    s, r, p = 1, d, 2
    while p * p <= r:
        if p > 10 ** 7:
            raise NotCovered('radicand too large to make square-free')
        while r % (p * p) == 0:
            r //= p * p
            s *= p
        p += 1 if p == 2 else 2
    return s, r


class Surd:
    """p + q*sqrt(d) with Fractions p, q and a square-free integer d >= 1 (d = 1 or q = 0: a rational). Exact; a mix of different radicands is NotCovered."""
    __slots__ = ('p', 'q', 'd')

    def __init__(self, p=0, q=0, d=1):
        p, q, d = Fraction(p), Fraction(q), int(d)
        if d < 0:
            raise BadInput('negative radicand')
        if q != 0 and d != 1:
            s, r = _squarefree(d)
            q, d = q * s, r
        if d == 1 or q == 0 or d == 0:
            p, q, d = (p + q if d == 1 else p), Fraction(0), 1
        self.p, self.q, self.d = p, q, d

    @staticmethod
    def sqrt(x):
        """The exact square root of a non-negative rational as a Surd."""
        x = Fraction(x)
        if x < 0:
            raise BadInput('square root of a negative number')
        n, m = x.numerator, x.denominator
        s, r = _squarefree(n * m)      # sqrt(n/m) = sqrt(n*m)/m
        return Surd(0, Fraction(s, m), r) if r != 1 else Surd(Fraction(s, m))

    def is_rational(self):
        return self.q == 0

    def _like(self, o):
        if isinstance(o, (int, Fraction)):
            return Surd(o)
        if isinstance(o, Surd):
            return o
        return NotImplemented

    def _same(self, o):
        if self.q != 0 and o.q != 0 and self.d != o.d:
            raise NotCovered('several radicals')
        return self.d if self.q != 0 else o.d

    def __add__(self, o):
        o = self._like(o)
        if o is NotImplemented:
            return o
        d = self._same(o)
        return Surd(self.p + o.p, self.q + o.q, d)
    __radd__ = __add__

    def __neg__(self):
        return Surd(-self.p, -self.q, self.d)

    def __sub__(self, o):
        o = self._like(o)
        return o if o is NotImplemented else self + (-o)

    def __rsub__(self, o):
        o = self._like(o)
        return o if o is NotImplemented else o + (-self)

    def __mul__(self, o):
        o = self._like(o)
        if o is NotImplemented:
            return o
        d = self._same(o)
        return Surd(self.p * o.p + self.q * o.q * d, self.p * o.q + self.q * o.p, d)
    __rmul__ = __mul__

    def conjugate(self):
        return Surd(self.p, -self.q, self.d)

    def __truediv__(self, o):
        o = self._like(o)
        if o is NotImplemented:
            return o
        if o.q == 0:
            if o.p == 0:
                raise BadInput('division by zero')
            return Surd(self.p / o.p, self.q / o.p, self.d)
        n = o.p * o.p - o.q * o.q * o.d
        if n == 0:
            raise BadInput('division by zero')
        return self * o.conjugate() / n

    def __rtruediv__(self, o):
        o = self._like(o)
        return o if o is NotImplemented else o / self

    def sign(self):
        a, b, d = self.p, self.q, self.d
        if b == 0:
            return (a > 0) - (a < 0)
        if a >= 0 and b > 0:
            return 1
        if a <= 0 and b < 0:
            return -1
        left, right = a * a, b * b * d
        if a > 0 and b < 0:
            return (left > right) - (left < right)
        return (right > left) - (right < left)

    def __eq__(self, o):
        o = self._like(o)
        if o is NotImplemented:
            return False
        return (self.p, self.q, self.d if self.q else 1) == (o.p, o.q, o.d if o.q else 1)

    def __hash__(self):
        return hash((self.p, self.q, self.d if self.q else 1))

    def _cmp(self, o):
        o = self._like(o)
        if o is NotImplemented:
            return o
        diff = self - o
        return diff.sign()

    def __lt__(self, o):
        c = self._cmp(o)
        return c if c is NotImplemented else c < 0

    def __le__(self, o):
        c = self._cmp(o)
        return c if c is NotImplemented else c <= 0

    def __gt__(self, o):
        c = self._cmp(o)
        return c if c is NotImplemented else c > 0

    def __ge__(self, o):
        c = self._cmp(o)
        return c if c is NotImplemented else c >= 0

    def to_decimal(self, digits=30):
        with localcontext() as ctx:
            ctx.prec = digits + 10
            v = Decimal(self.p.numerator) / Decimal(self.p.denominator)
            if self.q != 0:
                v += Decimal(self.q.numerator) / Decimal(self.q.denominator) * Decimal(self.d).sqrt()
            ctx.prec = digits
            return +v

    def __float__(self):
        return float(self.to_decimal(20))

    def __str__(self):
        if self.q == 0:
            return str(self.p)
        return f'{self.p} + {self.q}*sqrt({self.d})' if self.p != 0 else f'{self.q}*sqrt({self.d})'

    def __repr__(self):
        return f'Surd({self.p}, {self.q}, {self.d})'
