"""Simulation of tools/igc/mathbackend/numbers.py against independent oracles. No model.

    PYTHONPATH=. python -m tools.igc.mathbackend.sim_numbers --seed 1 --n 60      (needs SymPy: the venv python)

Each registered operation gets `n` cases (edge cases first, then random ones of school and contest sizes). A case is (op, args, oracle, oracle_name), where the oracle is
  - a value computed by SymPy (written from scratch in this module's source: numbers.py copies nothing from SymPy, so SymPy is independent),
  - a value from a brute force or long division written below (itertools-free loops, a different algorithm from the backend's),
  - a `Cert`: a check that the answer is right without recomputing it (factors multiply back and are prime, x*y satisfy the equation, ...), or
  - BAD / REFUSE: the input must be rejected (BadInput) / must be refused (NotCovered).
Report per operation: generated, answered, agree, refused (with the reasons counted), disagree (must be 0), seconds. Written to build-mathbackend/numbers.json.
"""
import argparse
import json
import math
import random
import re
import sys
import time
from fractions import Fraction
from pathlib import Path

import sympy as sp
from sympy.ntheory.digits import digits as sp_digits
from sympy.ntheory.modular import solve_congruence
from sympy.functions.combinatorial.numbers import stirling as sp_stirling
from sympy.solvers.diophantine.diophantine import diop_DN

from tools.igc.mathbackend import BadInput, NotCovered, registry
from tools.igc.mathbackend import run as backend_run
import tools.igc.mathbackend.numbers as backend

MODULE = 'numbers'
ROOT = Path(__file__).resolve().parents[3]
SYMPY_RING = None


class Cert:
    """An oracle that checks an answer instead of recomputing it. check(result) -> True when the result is certified."""
    def __init__(self, check, desc=''):
        self.check, self.desc = check, desc


BAD = object()       # the input is malformed: BadInput expected
REFUSE = object()    # the operation must refuse (NotCovered)


def F(x):
    return Fraction(int(x.p), int(x.q)) if isinstance(x, sp.Rational) else x


def norm(x):
    if isinstance(x, (list, tuple)):
        return tuple(norm(i) for i in x)
    if isinstance(x, dict):
        return tuple(sorted((norm(k), norm(v)) for k, v in x.items()))
    if isinstance(x, sp.Integer):
        return int(x)
    if isinstance(x, sp.Rational):
        return Fraction(int(x.p), int(x.q))
    return x


def rrat(rng, big=False):
    """A random rational, as int, Fraction or "p/q" string."""
    k = rng.choice([1, 1, 2, 3])
    lim = 10 ** (rng.randint(15, 40) if big else rng.randint(1, 4))
    a, b = rng.randint(-lim, lim), rng.randint(1, lim)
    f = Fraction(a, b)
    return rng.choice([f, f, f'{f.numerator}/{f.denominator}' if f.denominator != 1 else str(f.numerator), f.numerator if f.denominator == 1 else f])


def to_sp(x):
    if isinstance(x, str):
        return sp.Rational(x)
    return sp.Rational(x.numerator, x.denominator) if isinstance(x, Fraction) else sp.Integer(x)


def rint(rng, digits_lo=1, digits_hi=6, neg=True):
    v = rng.randrange(10 ** rng.randint(digits_lo, digits_hi))
    return -v if neg and rng.random() < 0.3 else v


GEN = {}


def gen(*names):
    def deco(fn):
        for nm in names:
            GEN[f'{MODULE}.{nm}'] = (lambda f, nm: (lambda rng, n: f(rng, n, nm)))(fn, nm)
        return fn
    return deco


def fill(edges, make, n, rng):
    """edges first, then random cases from make(rng) until there are n."""
    out = list(edges)
    while len(out) < n:
        out.append(make(rng))
    return out[:max(n, len(edges))]


# ---------------------------------------------------------------------------------------------------------------------------------------------------------
# exact arithmetic

@gen('add', 'sub', 'mul', 'div')
def g_arith(rng, n, name):
    fn = {'add': lambda a, b: a + b, 'sub': lambda a, b: a - b, 'mul': lambda a, b: a * b, 'div': lambda a, b: a / b}[name]

    def make(rng):
        a, b = rrat(rng, rng.random() < 0.25), rrat(rng, rng.random() < 0.25)
        if name == 'div' and to_sp(b) == 0:
            return ((a, b), BAD, 'zero divisor')
        return ((a, b), fn(to_sp(a), to_sp(b)), 'SymPy Rational')
    edges = [((0, 0), BAD if name == 'div' else fn(sp.Integer(0), sp.Integer(0)), 'SymPy Rational'),
             (('1/3', '2/6'), BAD if False else fn(sp.Rational(1, 3), sp.Rational(1, 3)), 'SymPy Rational'),
             ((0.5, 1), BAD, 'float refused'), ((True, 1), BAD, 'bool refused'), (('1e999', 1), BAD, 'exponent string refused'),
             (('abc', 1), BAD, 'not a number')]
    return fill(edges, make, n, rng)


@gen('pow_int')
def g_pow_int(rng, n, name):
    def make(rng):
        a = rrat(rng)
        e = rng.randint(-8, 14)
        if to_sp(a) == 0 and e < 0:
            return ((a, e), BAD, '0 to a negative power')
        return ((a, e), to_sp(a) ** e, 'SymPy Rational ** int')
    edges = [((0, 0), sp.Integer(1), 'SymPy'), ((-1, 10 ** 12 + 1), sp.Integer(-1), 'SymPy'), ((1, 10 ** 15), sp.Integer(1), 'SymPy'),
             ((3, 10 ** 8), REFUSE, 'past the 2,000,000-bit bound'), (('2/3', -3), sp.Rational(27, 8), 'SymPy')]
    return fill(edges, make, n, rng)


@gen('pow_rational')
def g_pow_rational(rng, n, name):
    def make(rng):
        s = rng.randint(2, 6)
        base = sp.Rational(rng.randint(1, 12), rng.randint(1, 12))
        r = rng.randint(-5, 7)
        kind = rng.random()
        if kind < 0.55:                                  # exact: (base^s)^(r/s)
            a = base ** s
            e = sp.Rational(r, s)
            return ((F(a), F(e)), base ** r, 'SymPy exact (base**s)**(r/s) = base**r')
        if kind < 0.8:                                    # inexact: a base that is not a perfect s-th power
            a = sp.Rational(rng.choice([2, 3, 5, 6, 7, 10, 11]), rng.choice([1, 1, 3]))
            e = sp.Rational(rng.choice([1, 1, 2, 3]), s)
            v = a ** e
            return ((F(a), F(e)), F(v) if v.is_Rational else REFUSE, 'SymPy a**e is rational? else a refusal')
        a = -sp.Rational(rng.randint(1, 20), rng.randint(1, 5))   # negative base, fractional exponent: refused by design
        e = sp.Rational(1, rng.choice([2, 3, 5]))
        return ((F(a), F(e)), REFUSE, 'negative base with fractional exponent is refused')
    edges = [((8, '2/3'), sp.Integer(4), 'SymPy'), (('9/4', '-1/2'), sp.Rational(2, 3), 'SymPy'), ((0, '1/2'), sp.Integer(0), 'SymPy'),
             ((0, '-1/2'), BAD, '0 to a negative power'), ((16, '1/4'), sp.Integer(2), 'SymPy'), ((2, 3), sp.Integer(8), 'SymPy')]
    return fill(edges, make, n, rng)


@gen('iroot', 'isqrt')
def g_iroot(rng, n, name):
    def make(rng):
        k = 2 if name == 'isqrt' else rng.randint(1, 15)
        v = rng.randrange(10 ** rng.randint(1, 60))
        if rng.random() < 0.3:
            v = rng.randrange(1, 10 ** rng.randint(1, 8)) ** k + rng.choice([-1, 0, 1])
        v = max(v, 0)
        args = (v,) if name == 'isqrt' else (v, k)
        return (args, sp.integer_nthroot(v, k)[0], 'SymPy integer_nthroot')
    edges = [((0,) if name == 'isqrt' else (0, 3), 0, 'SymPy'), ((1,) if name == 'isqrt' else (1, 5), 1, 'SymPy'),
             ((-1,) if name == 'isqrt' else (-1, 3), BAD, 'negative n'), ((10 ** 40,) if name == 'isqrt' else (10 ** 40, 2), 10 ** 20, 'exact')]
    return fill(edges, make, n, rng)


@gen('is_square')
def g_is_square(rng, n, name):
    def make(rng):
        if rng.random() < 0.5:
            a, b = rng.randint(0, 10 ** 6), rng.randint(1, 10 ** 4)
            q = Fraction(a * a, b * b) * rng.choice([1, 1, 1, Fraction(3, 2)])
        else:
            q = Fraction(rng.randint(-100, 10 ** 8), rng.randint(1, 50))
        s = sp.sqrt(sp.Rational(q.numerator, q.denominator)) if q >= 0 else None
        return ((q,), bool(s is not None and s.is_Rational), 'SymPy sqrt(Rational).is_Rational')
    return fill([((0,), True, 'SymPy'), ((-4,), False, 'negative')], make, n, rng)


@gen('perfect_power', 'is_perfect_power')
def g_perfect_power(rng, n, name):
    def make(rng):
        r = rng.random()
        if r < 0.6:
            b = rng.randint(2, 40)
            e = rng.randint(2, 12)
            v = b ** e * rng.choice([1, 1, 1, -1 if e % 2 else 1])
        else:
            v = rng.randint(-10 ** 6, 10 ** 6)
        if abs(v) < 2:
            v = 2
        pp = sp.perfect_power(v)
        if name == 'perfect_power':
            return ((v,), (int(pp[0]), int(pp[1])) if pp else (v, 1), 'SymPy perfect_power')
        return ((v,), bool(pp), 'SymPy perfect_power')
    edges = [((64,), (2, 6) if name == 'perfect_power' else True, 'SymPy'), ((-27,), (-3, 3) if name == 'perfect_power' else True, 'SymPy'),
             ((-16,), (-16, 1) if name == 'perfect_power' else False, 'SymPy')]
    return fill(edges, make, n, rng)


# ---------------------------------------------------------------------------------------------------------------------------------------------------------
# gcd, modular arithmetic

@gen('gcd', 'lcm')
def g_gcd(rng, n, name):
    def make(rng):
        k = rng.randint(1, 5)
        xs = [rng.randint(0, 10 ** rng.randint(1, 25)) for _ in range(k)]
        if rng.random() < 0.5:
            g = rng.randint(1, 10 ** 6)
            xs = [x * g for x in xs]
        o = sp.igcd(0, *xs) if name == 'gcd' else sp.ilcm(1, *xs)
        return (tuple(xs), int(o), 'SymPy igcd/ilcm')
    return fill([((), BAD, 'no arguments'), ((0, 0), 0, 'SymPy'), ((12, 18, 30), 6 if name == 'gcd' else 180, 'SymPy')], make, n, rng)


@gen('xgcd')
def g_xgcd(rng, n, name):
    def make(rng):
        a, b = rint(rng, 1, 30), rint(rng, 1, 30)
        g = int(sp.igcd(a, b))
        return ((a, b), Cert(lambda r, a=a, b=b, g=g: len(r) == 3 and r[0] == g and a * r[1] + b * r[2] == g, 'g = gcd (SymPy) and a*x + b*y = g'), 'certificate')
    return fill([((0, 0), Cert(lambda r: r == (0, 0, 0) or (r[0] == 0), 'g = 0'), 'certificate'), ((240, 46), Cert(lambda r: r[0] == 2 and 240 * r[1] + 46 * r[2] == 2), 'cert')],
                make, n, rng)


@gen('modinv')
def g_modinv(rng, n, name):
    def make(rng):
        m = rng.randint(2, 10 ** rng.randint(1, 20))
        a = rint(rng, 1, 22)
        try:
            return ((a, m), int(sp.mod_inverse(a, m)), 'SymPy mod_inverse')
        except ValueError:
            return ((a, m), (), 'SymPy: no inverse')
    return fill([((5, 1), 0, 'x*a = 0 mod 1'), ((2, 4), (), 'SymPy'), ((3, 7), 5, 'SymPy'), ((3, 0), BAD, 'modulus 0')], make, n, rng)


def pm(a, e, m):
    """Square-and-multiply written here, independent of the backend's call to pow."""
    r, a = 1 % m, a % m
    while e:
        if e & 1:
            r = r * a % m
        a = a * a % m
        e >>= 1
    return r


@gen('powmod')
def g_powmod(rng, n, name):
    def make(rng):
        m = rng.randint(1, 10 ** rng.randint(1, 30))
        a = rint(rng, 1, 30)
        e = rng.randint(-5, 10 ** rng.randint(1, 30)) if rng.random() < 0.9 else rng.randint(0, 5)
        if e >= 0:
            return ((a, e, m), pm(a, e, m), 'square-and-multiply written in the sim')
        try:
            inv = int(sp.mod_inverse(a, m)) if m > 1 else 0
        except ValueError:
            return ((a, e, m), (), 'SymPy: no inverse')
        return ((a, e, m), pm(inv, -e, m) if m > 1 else 0, 'SymPy mod_inverse then square-and-multiply')
    return fill([((2, 100, 10 ** 9 + 7), pm(2, 100, 10 ** 9 + 7), 'sim'), ((0, 0, 5), 1, 'convention 0**0 = 1'), ((5, 3, 0), BAD, 'modulus 0')], make, n, rng)


@gen('crt')
def g_crt(rng, n, name):
    def make(rng):
        k = rng.randint(1, 4)
        coprime = rng.random() < 0.5
        ms = []
        while len(ms) < k:
            m = rng.randint(1, 60 if not coprime else 1000)
            if not coprime or all(math.gcd(m, x) == 1 for x in ms):
                ms.append(m)
        if rng.random() < 0.6:                   # consistent: built from a hidden x
            x = rng.randint(0, 10 ** 6)
            rs = [x % m for m in ms]
        else:
            rs = [rng.randint(0, 80) for _ in ms]
        out = solve_congruence(*zip(rs, ms))
        return ((rs, ms), (int(out[0]), int(out[1])) if out else (), 'SymPy solve_congruence')
    return fill([(([2, 3], [3, 5]), (8, 15), 'SymPy'), (([1, 0], [2, 4]), (), 'SymPy'), (([], []), BAD, 'empty')], make, n, rng)


@gen('lincong')
def g_lincong(rng, n, name):
    def make(rng):
        mod = rng.randint(1, 400)
        a, b = rng.randint(-500, 500), rng.randint(-500, 500)
        sol = [x for x in range(mod) if (a * x - b) % mod == 0]

        def check(r, sol=sol, mod=mod):
            if not sol:
                return r == ()
            x0, st = r
            return sorted(x for x in range(mod) if (x - x0) % st == 0) == sol
        return ((a, b, mod), Cert(check, 'every solution in 0..n-1 by brute force'), 'brute force')
    return fill([((6, 4, 10), Cert(lambda r: r == (4, 5)), 'brute force')], make, n, rng)


@gen('jacobi')
def g_jacobi(rng, n, name):
    def make(rng):
        m = rng.randrange(1, 10 ** rng.randint(1, 20), 2) | 1
        a = rint(rng, 1, 22)
        return ((a, m), int(sp.jacobi_symbol(a, m)), 'SymPy jacobi_symbol')
    return fill([((2, 15), 1, 'SymPy'), ((3, 4), BAD, 'even n'), ((5, 0), BAD, 'n = 0')], make, n, rng)


@gen('legendre')
def g_legendre(rng, n, name):
    def make(rng):
        p = sp.randprime(3, 10 ** rng.randint(2, 25))
        a = rint(rng, 1, 25)
        return ((a, int(p)), int(sp.legendre_symbol(a % p, p)), 'SymPy legendre_symbol')
    return fill([((3, 7), -1, 'SymPy'), ((7, 7), 0, 'SymPy'), ((3, 9), BAD, 'not prime'), ((3, 2), BAD, 'p = 2')], make, n, rng)


@gen('sqrt_mod')
def g_sqrt_mod(rng, n, name):
    def make(rng):
        p = int(sp.randprime(2, 10 ** rng.randint(1, 25)))
        if rng.random() < 0.6:
            a = (rng.randrange(p) ** 2 + rng.choice([0, 0, 0, p])) % (p) if p > 1 else 0
        else:
            a = rng.randrange(p + 5)
        r = sp.sqrt_mod(a, p, all_roots=True)
        return ((a, p), sorted(int(x) for x in r) if r else [], 'SymPy sqrt_mod all_roots')
    return fill([((10, 13), [6, 7], 'SymPy'), ((0, 13), [0], 'SymPy'), ((3, 2), [1], 'p = 2'), ((4, 9), BAD, 'not prime')], make, n, rng)


@gen('sqrt_mod_all')
def g_sqrt_mod_all(rng, n, name):
    def make(rng):
        if rng.random() < 0.6:
            m = rng.randint(1, 3000)
            a = rng.randint(0, 2 * m)
            return ((a, m), [x for x in range(m) if (x * x - a) % m == 0], 'brute force over 0..n-1')
        m = rng.choice([3 ** 8 * 5 ** 3, 7 ** 5 * 11 ** 2, 101 * 103 * 107, 2 ** 5 * 3 ** 6 * 13, 10 ** 9 + 7, 3 ** 15 * 17])
        x = rng.randrange(m)
        a = x * x % m
        r = sp.sqrt_mod(a, m, all_roots=True)
        return ((a, m), sorted(int(v) for v in r), 'SymPy sqrt_mod all_roots (composite)')
    return fill([((1, 8), [1, 3, 5, 7], 'brute'), ((0, 1), [0], 'brute'), ((3, 0), BAD, 'modulus 0')], make, n, rng)


@gen('discrete_log')
def g_dlog(rng, n, name):
    def make(rng):
        if rng.random() < 0.6:
            m = rng.randint(1, 800)
            a, b = rng.randint(0, 2 * m), rng.randint(0, 2 * m)
            xs = [x for x in range(2 * m + 3) if pow(a, x, m) == b % m]
            return ((a, b, m), xs[0] if xs else (), 'brute force over exponents 0..2n+2')
        p = int(sp.randprime(10 ** 3, 10 ** rng.randint(4, 11)))
        g = sp.primitive_root(p)
        x = rng.randrange(p - 1)
        b = pow(g, x, p)
        return ((g, b, p), int(sp.discrete_log(p, b, g)), 'SymPy discrete_log (prime modulus)')
    return fill([((2, 8, 13), 3, 'brute'), ((2, 3, 10 ** 13), REFUSE, 'past the 10**12 bound')], make, n, rng)


@gen('binomial_mod')
def g_binomial_mod(rng, n, name):
    def lucas_small(nn, kk, p):
        r = 1
        while nn or kk:
            a, b = nn % p, kk % p
            if b > a:
                return 0
            r = r * math.comb(a, b) % p
            nn //= p
            kk //= p
        return r

    def make(rng):
        r = rng.random()
        if r < 0.5:
            nn = rng.randint(0, 3000)
            kk = rng.randint(0, nn + 2)
            m = rng.choice([2, 3, 4, 6, 8, 9, 10, 12, 16, 30, 36, 49, 100, 1000, 997, 10 ** 6, 2 ** 10 * 3 ** 5, 10 ** 9 + 7, 998244353, 1001 * 7])
            return ((nn, kk, m), int(sp.binomial(nn, kk)) % m if kk <= nn else 0, 'SymPy binomial then mod (exact)')
        p = rng.choice([2, 3, 5, 7, 11, 13, 97, 101, 997])
        nn = rng.randint(0, 10 ** rng.randint(3, 18))
        kk = rng.randint(0, nn)
        return ((nn, kk, p), lucas_small(nn, kk, p), "Lucas' theorem written in the sim (small prime)")
    return fill([((10 ** 5, 5 * 10 ** 4, 10 ** 9 + 7), math.comb(10 ** 5, 5 * 10 ** 4) % (10 ** 9 + 7), 'exact comb'), ((5, 7, 10), 0, 'k > n'), ((5, 2, 1), 0, 'mod 1')],
                make, n, rng)


# ---------------------------------------------------------------------------------------------------------------------------------------------------------
# primes

CARMICHAEL = [561, 1105, 1729, 2465, 2821, 6601, 8911, 41041, 825265, 321197185, 5394826801, 232250619601, 9746347772161]
STRONG_PSEUDO = [2047, 3215031751, 3825123056546413051, 318665857834031151167461, 3317044064679887385961981, 25326001, 1122004669633]


@gen('is_prime')
def g_is_prime(rng, n, name):
    def make(rng):
        r = rng.random()
        if r < 0.25:
            v = rng.randrange(0, 10 ** rng.randint(1, 8))
        elif r < 0.5:
            v = int(sp.randprime(2, 10 ** rng.randint(3, 30)))
        elif r < 0.7:
            v = int(sp.randprime(2, 10 ** rng.randint(3, 12))) * int(sp.randprime(2, 10 ** rng.randint(3, 12)))
        elif r < 0.85:
            v = 2 ** rng.randint(2, 130) + rng.choice([-1, 1])
        else:
            v = rng.randrange(10 ** 24, 10 ** 40)
        return ((v,), bool(sp.isprime(v)), 'SymPy isprime')
    edges = [((v,), bool(sp.isprime(v)), 'SymPy isprime') for v in CARMICHAEL + STRONG_PSEUDO + [-7, 0, 1, 2, 3, 4, 2 ** 61 - 1, 2 ** 89 - 1, 2 ** 127 - 1]]
    return fill(edges, make, n, rng)


@gen('next_prime', 'prev_prime')
def g_nextprev(rng, n, name):
    def make(rng):
        v = rng.randrange(3, 10 ** rng.randint(1, 30))
        return ((v,), int(sp.nextprime(v) if name == 'next_prime' else sp.prevprime(v)), 'SymPy nextprime/prevprime')
    edges = [((2,), 3, 'SymPy'), ((1,), 2, 'SymPy')] if name == 'next_prime' else [((3,), 2, 'SymPy'), ((2,), BAD, 'no prime below 2'), ((100,), 97, 'SymPy')]
    return fill(edges, make, n, rng)


@gen('primes_upto')
def g_primes_upto(rng, n, name):
    def make(rng):
        v = rng.randint(0, 10 ** rng.randint(1, 5))
        return ((v,), [int(p) for p in sp.primerange(2, v + 1)], 'SymPy primerange')
    return fill([((1,), [], 'SymPy'), ((2,), [2], 'SymPy'), ((10 ** 7 + 1,), REFUSE, 'past the 10**7 bound')], make, n, rng)


@gen('nth_prime')
def g_nth_prime(rng, n, name):
    def make(rng):
        k = rng.randint(1, 10 ** rng.randint(1, 5))
        return ((k,), int(sp.prime(k)), 'SymPy prime')
    return fill([((1,), 2, 'SymPy'), ((1000,), 7919, 'SymPy'), ((0,), BAD, 'k = 0'), ((600001,), REFUSE, 'past the bound')], make, n, rng)


@gen('prime_pi')
def g_prime_pi(rng, n, name):
    def make(rng):
        v = rng.randint(0, 10 ** rng.randint(1, 7))
        return ((v,), int(sp.primepi(v)), 'SymPy primepi')
    edges = [((0,), 0, 'SymPy'), ((1,), 0, 'SymPy'), ((2,), 1, 'SymPy'), ((10 ** 8,), 5761455, 'known table value pi(10^8)'), ((10 ** 9,), 50847534, 'known pi(10^9)')]
    return fill(edges, make, n, rng)


def _prime_cert(n):
    def check(f):
        if not isinstance(f, dict):
            return False
        prod = 1
        for p, e in f.items():
            if e < 1 or not sp.isprime(p):
                return False
            if p < 10 ** 7 and any(p % d == 0 for d in range(2, math.isqrt(p) + 1)):
                return False          # trial division on the small ones
            prod *= p ** e
        return prod == n
    return Cert(check, 'prime factors (SymPy isprime, trial division below 1e7) multiply back to n')


@gen('factorint')
def g_factorint(rng, n, name):
    def make(rng):
        r = rng.random()
        if r < 0.3:
            v = rng.randint(1, 10 ** rng.randint(1, 12))
        elif r < 0.55:
            v = rng.randint(1, 10 ** rng.randint(13, 30))
        elif r < 0.75:
            v = 1
            for _ in range(rng.randint(2, 5)):
                v *= int(sp.randprime(2, 10 ** rng.randint(2, 9)))
        elif r < 0.9:
            v = int(sp.randprime(2, 10 ** rng.randint(10, 15))) * int(sp.randprime(2, 10 ** rng.randint(10, 15)))   # two big primes
        else:
            v = rng.choice([2 ** 64 - 1, 2 ** 67 - 1, 10 ** 18 + 9, 600851475143, 2 ** 100 - 1, 7 ** 40, 10 ** 30 + 57, 3 ** 50 * 7 ** 20])
        want = dict(sp.factorint(v)) if v < 10 ** 22 else None
        if want is not None:
            return ((v,), {int(k): int(e) for k, e in want.items()}, 'SymPy factorint')
        return ((v,), _prime_cert(v), 'certificate')
    return fill([((1,), {}, 'SymPy'), ((360,), {2: 3, 3: 2, 5: 1}, 'SymPy'), ((0,), BAD, 'zero'), ((-6,), BAD, 'negative'), ((2 ** 300 + 1,), REFUSE, 'past 2**240')],
                make, n, rng)


@gen('factorint_partial')
def g_factorint_partial(rng, n, name):
    def make(rng):
        v = int(sp.randprime(2, 10 ** rng.randint(10, 17))) * int(sp.randprime(2, 10 ** rng.randint(10, 17))) * rng.choice([1, 6, 30, 7 ** 4, 1001])
        if rng.random() < 0.5:
            v = rng.randint(1, 10 ** rng.randint(3, 22))

        def check(r, v=v):
            f, left = r
            prod = 1
            for p, e in f.items():
                if not sp.isprime(p) or e < 1:
                    return False
                prod *= p ** e
            for c in left:
                if c <= 1:
                    return False
                prod *= c
            return prod == v
        return ((v,), Cert(check, 'proven primes and leftovers multiply back to n; the primes pass SymPy isprime'), 'certificate')
    return fill([((12,), Cert(lambda r: r == ({2: 2, 3: 1}, [])), 'certificate')], make, n, rng)


@gen('is_squarefree', 'smallest_prime_factor', 'radical')
def g_factor_fns(rng, n, name):
    def make(rng):
        v = rng.randint(2, 10 ** rng.randint(1, 18))
        f = sp.factorint(v)
        if name == 'is_squarefree':
            return ((v,), all(e == 1 for e in f.values()), 'SymPy factorint')
        if name == 'smallest_prime_factor':
            return ((v,), int(min(f)), 'SymPy factorint')
        r = 1
        for p in f:
            r *= int(p)
        return ((v,), r, 'SymPy factorint')
    edges = {'is_squarefree': [((1,), True, 'SymPy'), ((12,), False, 'SymPy'), ((0,), BAD, 'zero')],
             'smallest_prime_factor': [((2,), 2, 'SymPy'), ((1,), BAD, 'n < 2'), ((91,), 7, 'SymPy')],
             'radical': [((1,), 1, 'SymPy'), ((360,), 30, 'SymPy'), ((0,), BAD, 'zero')]}[name]
    return fill(edges, make, n, rng)


# ---------------------------------------------------------------------------------------------------------------------------------------------------------
# number-theoretic functions

@gen('divisors', 'num_divisors', 'sigma', 'totient', 'moebius', 'carmichael')
def g_nt(rng, n, name):
    def make(rng):
        r = rng.random()
        v = rng.randint(1, 10 ** rng.randint(1, 18)) if r < 0.7 else rng.choice([720720, 5040, 2 ** 20, 3 ** 12 * 5 ** 5, 2 * 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23, 10 ** 12, 1])
        if name == 'divisors':
            v = min(v, 10 ** 12) if rng.random() < 0.5 else v
            return ((v,), [int(d) for d in sp.divisors(v)] if v < 10 ** 14 else [int(d) for d in sp.divisors(v)], 'SymPy divisors')
        if name == 'num_divisors':
            return ((v,), int(sp.divisor_count(v)), 'SymPy divisor_count')
        if name == 'sigma':
            k = rng.choice([1, 1, 2, 3, 0])
            return ((v, k), int(sp.divisor_sigma(v, k)), 'SymPy divisor_sigma')
        if name == 'totient':
            return ((v,), int(sp.totient(v)), 'SymPy totient')
        if name == 'moebius':
            return ((v,), int(sp.mobius(v)), 'SymPy mobius')
        return ((v,), int(sp.reduced_totient(v)), 'SymPy reduced_totient')
    edges = [((1,), [1] if name == 'divisors' else 1, 'SymPy'), ((0,), BAD, 'zero')]
    if name == 'sigma':
        edges = [((1,), 1, 'SymPy'), ((0,), BAD, 'zero'), ((28,), 56, 'SymPy')]
    elif name == 'moebius':
        edges = [((1,), 1, 'SymPy'), ((0,), BAD, 'zero'), ((30,), -1, 'SymPy')]
    elif name == 'carmichael':
        edges = [((1,), 1, 'SymPy'), ((2,), 1, 'SymPy'), ((8,), 2, 'SymPy'), ((16,), 4, 'SymPy'), ((0,), BAD, 'zero')]
    elif name == 'totient':
        edges = [((1,), 1, 'SymPy'), ((0,), BAD, 'zero'), ((36,), 12, 'SymPy')]
    return fill(edges, make, n, rng)


@gen('order')
def g_order(rng, n, name):
    def make(rng):
        m = rng.randint(2, 10 ** rng.randint(1, 15))
        a = rng.randint(-5 * m, 5 * m) if rng.random() < 0.3 else rng.randint(0, m)
        if math.gcd(a, m) != 1:
            return ((a, m), (), 'gcd(a, n) > 1')
        return ((a, m), int(sp.n_order(a, m)), 'SymPy n_order')
    return fill([((2, 7), 3, 'SymPy'), ((5, 1), 1, 'order mod 1'), ((2, 4), (), 'gcd')], make, n, rng)


@gen('primitive_root')
def g_primroot(rng, n, name):
    def make(rng):
        r = rng.random()
        if r < 0.4:
            m = int(sp.randprime(3, 10 ** rng.randint(2, 15)))
        elif r < 0.7:
            m = int(sp.randprime(3, 1000)) ** rng.randint(1, 4) * rng.choice([1, 2])
        else:
            m = rng.randint(3, 10 ** rng.randint(2, 10))
        pr = sp.primitive_root(m)
        return ((m,), int(pr) if pr is not None else (), 'SymPy primitive_root (None -> none)')
    return fill([((1,), 0, 'convention'), ((2,), 1, 'convention'), ((8,), (), 'SymPy None'), ((23,), 5, 'SymPy')], make, n, rng)


@gen('valuation')
def g_valuation(rng, n, name):
    def make(rng):
        p = rng.choice([2, 3, 5, 6, 10, 12, 7, 100])
        v = rng.randint(1, 10 ** 12) * p ** rng.randint(0, 20) * rng.choice([1, -1])
        return ((v, p), int(sp.multiplicity(p, v)), 'SymPy multiplicity')
    return fill([((0, 2), BAD, 'valuation of zero'), ((360, 2), 3, 'SymPy'), ((5, 1), BAD, 'p < 2')], make, n, rng)


@gen('factorial_valuation')
def g_fval(rng, n, name):
    def make(rng):
        p = int(sp.randprime(2, 200))
        nn = rng.randint(0, 3000)
        return ((nn, p), int(sp.multiplicity(p, sp.factorial(nn))) if nn else 0, 'SymPy multiplicity in the exact factorial')
    return fill([((100, 5), 24, 'SymPy'), ((0, 2), 0, 'SymPy'), ((10, 4), BAD, 'p not prime')], make, n, rng)


@gen('factorial_trailing_zeros')
def g_ftz(rng, n, name):
    def make(rng):
        b = rng.choice([2, 6, 10, 12, 16, 100, 36, 7, 60, 720, 1000])
        nn = rng.randint(0, 2500)
        f = math.factorial(nn)
        z = 0
        while f % b == 0:
            f //= b
            z += 1
        return ((nn, b), z, 'divide the exact factorial by the base until it stops')
    return fill([((100,), 24, 'known'), ((10, 12), 4, 'exact factorial')], make, n, rng)


# ---------------------------------------------------------------------------------------------------------------------------------------------------------
# combinatorial numbers

@gen('factorial', 'double_factorial', 'catalan', 'bell', 'partition_number')
def g_comb1(rng, n, name):
    hi = {'factorial': 3000, 'double_factorial': 3000, 'catalan': 3000, 'bell': 300, 'partition_number': 1500}[name]
    fn = {'factorial': sp.factorial, 'double_factorial': sp.factorial2, 'catalan': sp.catalan, 'bell': sp.bell, 'partition_number': sp.npartitions}[name]

    def make(rng):
        v = rng.randint(0, hi if rng.random() < 0.3 else 60)
        return ((v,), int(fn(v)), 'SymPy ' + fn.__name__)
    edges = [((0,), int(fn(0)), 'SymPy'), ((1,), int(fn(1)), 'SymPy'), ((10,), int(fn(10)), 'SymPy'), ((-1,), BAD if name != 'double_factorial' else 1, 'domain')]
    if name == 'double_factorial':
        edges.append(((-2,), BAD, 'domain'))
    return fill(edges, make, n, rng)


@gen('binomial')
def g_binomial(rng, n, name):
    def make(rng):
        nn = rng.randint(-60, 5000)
        kk = rng.randint(-3, 80)
        return ((nn, kk), int(sp.binomial(nn, kk)), 'SymPy binomial')
    return fill([((52, 5), 2598960, 'SymPy'), ((-3, 2), 6, 'SymPy'), ((5, 7), 0, 'SymPy'), ((5, -1), 0, 'SymPy')], make, n, rng)


@gen('multinomial')
def g_multinomial(rng, n, name):
    def make(rng):
        ks = [rng.randint(0, 30) for _ in range(rng.randint(1, 6))]
        return (tuple(ks), int(sp.factorial(sum(ks)) / sp.prod([sp.factorial(k) for k in ks])), 'SymPy factorials')
    return fill([((2, 3, 1), 60, 'SymPy'), ((), BAD, 'no parts'), ((2, -1), BAD, 'negative part')], make, n, rng)


@gen('rising_factorial', 'falling_factorial')
def g_rf(rng, n, name):
    fn = sp.rf if name == 'rising_factorial' else sp.ff

    def make(rng):
        x = rrat(rng)
        k = rng.randint(0, 25)
        return ((x, k), F(fn(to_sp(x), k)), 'SymPy rf/ff')
    return fill([((3, 4), F(fn(3, 4)), 'SymPy'), ((5, 0), 1, 'SymPy'), ((5, -1), BAD, 'negative n')], make, n, rng)


def mat_fib(n, m):
    """F(n) mod m by 2x2 matrix power, written in the sim (not fast doubling)."""
    def mm(A, B):
        return [[(A[0][0] * B[0][0] + A[0][1] * B[1][0]) % m, (A[0][0] * B[0][1] + A[0][1] * B[1][1]) % m],
                [(A[1][0] * B[0][0] + A[1][1] * B[1][0]) % m, (A[1][0] * B[0][1] + A[1][1] * B[1][1]) % m]]
    R, M = [[1 % m, 0], [0, 1 % m]], [[1 % m, 1 % m], [1 % m, 0]]
    while n:
        if n & 1:
            R = mm(R, M)
        M = mm(M, M)
        n >>= 1
    return R[0][1]


@gen('fibonacci', 'lucas')
def g_fib(rng, n, name):
    fn = sp.fibonacci if name == 'fibonacci' else sp.lucas

    def make(rng):
        r = rng.random()
        if r < 0.5:
            v = rng.randint(-60, 3000)
            return ((v,), int(fn(v)), 'SymPy ' + name)
        m = rng.randint(1, 10 ** rng.randint(1, 18))
        v = rng.randint(0, 10 ** rng.randint(1, 40))
        want = mat_fib(v, m) if name == 'fibonacci' else (2 * mat_fib(v + 1, m) - mat_fib(v, m)) % m
        return ((v, m), want, '2x2 matrix power written in the sim' + ('' if name == 'fibonacci' else ' (L = 2F(n+1) - F(n))'))
    return fill([((10,), int(fn(10)), 'SymPy'), ((0,), int(fn(0)), 'SymPy'), ((-6,), int(fn(-6)), 'SymPy'), ((-3,), int(fn(-3)), 'SymPy'),
                 ((5, 0), BAD, 'modulus 0'), ((10 ** 7,), REFUSE, 'past the exact bound')], make, n, rng)


@gen('stirling1', 'stirling2')
def g_stirling(rng, n, name):
    kind = 1 if name == 'stirling1' else 2

    def make(rng):
        nn, kk = rng.randint(0, 60), rng.randint(0, 62)
        signed = rng.random() < 0.5 and kind == 1
        args = (nn, kk, signed) if kind == 1 else (nn, kk)
        if kk > nn:
            return (args, 0, 'definition: zero for k > n')
        v = sp_stirling(nn, kk, kind=kind, signed=signed) if kind == 1 else sp_stirling(nn, kk, kind=2)
        return (args, int(v), 'SymPy stirling')
    return fill([((4, 2) if kind == 2 else (4, 2), 11 if kind == 1 else 7, 'SymPy'), ((0, 0), 1, 'SymPy'), ((5, -1), BAD, 'negative k')], make, n, rng)


# ---------------------------------------------------------------------------------------------------------------------------------------------------------
# digits and bases

@gen('digits', 'digit_sum', 'num_digits', 'from_digits', 'digital_root')
def g_digits(rng, n, name):
    def make(rng):
        v = rng.randrange(10 ** rng.randint(1, 70))
        b = rng.choice([2, 3, 5, 7, 8, 10, 10, 12, 16, 36, 60, 1000, 10 ** 9])
        ds = [int(d) for d in sp_digits(v, b)[1:]]
        if name == 'digits':
            return ((v, b), ds, 'SymPy digits')
        if name == 'digit_sum':
            return ((v, b), sum(ds), 'SymPy digits then sum')
        if name == 'num_digits':
            return ((v, b), len(ds), 'SymPy digits then len')
        if name == 'from_digits':
            return ((ds, b), v, 'Horner in the sim')
        r = v
        while r >= b:
            r = sum(int(d) for d in sp_digits(r, b)[1:])
        return ((v, b), r, 'repeated digit sum with SymPy digits')
    edges = {'digits': [((0, 10), [0], 'SymPy'), ((255, 16), [15, 15], 'SymPy'), ((5, 1), BAD, 'base 1'), ((-5, 10), BAD, 'negative')],
             'digit_sum': [((0, 10), 0, 'SymPy'), ((2 ** 100, 10), 115, 'known')],
             'num_digits': [((0, 10), 1, 'convention'), ((2 ** 100, 10), 31, 'known')],
             'from_digits': [(([1, 0, 1], 2), 5, 'Horner'), (([2], 2), BAD, 'digit out of range'), (([], 10), 0, 'empty is 0')],
             'digital_root': [((0, 10), 0, 'convention'), ((987654321, 10), 9, 'known')]}[name]
    return fill(edges, make, n, rng)


@gen('base_convert')
def g_base_convert(rng, n, name):
    sym = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'

    def conv(v, b):
        if v == 0:
            return '0'
        out = []
        while v:
            v, r = divmod(v, b)
            out.append(sym[r])
        return ''.join(reversed(out))

    def make(rng):
        fb, tb = rng.randint(2, 36), rng.randint(2, 36)
        v = rng.randrange(10 ** rng.randint(1, 40))
        s = conv(v, fb)
        if rng.random() < 0.3:
            s = s.lower()
        neg = rng.random() < 0.2 and v != 0
        return (('-' + s if neg else s, fb, tb), ('-' if neg else '') + conv(v, tb), 'sim long division, upper-case digits')
    return fill([(('ff', 16, 2), '11111111', 'known'), (('12', 2, 10), BAD, 'digit 2 in base 2'), (('1', 1, 10), BAD, 'base 1'), (('', 10, 2), BAD, 'empty')], make, n, rng)


@gen('rat_floor', 'rat_ceil', 'rat_trunc', 'rat_round', 'rat_mod', 'frac_part')
def g_ratfns(rng, n, name):
    def make(rng):
        x = rrat(rng, rng.random() < 0.3)
        sx = to_sp(x)
        if name == 'rat_floor':
            return ((x,), int(sp.floor(sx)), 'SymPy floor')
        if name == 'rat_ceil':
            return ((x,), int(sp.ceiling(sx)), 'SymPy ceiling')
        if name == 'rat_trunc':
            return ((x,), int(sp.floor(sx)) if sx >= 0 else int(sp.ceiling(sx)), 'SymPy floor/ceiling by sign')
        if name == 'frac_part':
            return ((x,), F(sx - sp.floor(sx)), 'SymPy x - floor(x)')
        if name == 'rat_mod':
            y = rrat(rng)
            if to_sp(y) == 0:
                return ((x, y), BAD, 'modulo zero')
            return ((x, y), F(sp.Mod(sx, to_sp(y))), 'SymPy Mod')
        mode = rng.choice(['half_away', 'half_up', 'half_even'])
        if rng.random() < 0.4:
            sx = sp.Rational(rng.randint(-200, 200), 2)       # make exact halves common
            x = F(sx)
        half = sp.Rational(1, 2)
        if mode == 'half_up':
            want = sp.floor(sx + half)
        elif mode == 'half_away':
            want = sp.floor(abs(sx) + half) * (1 if sx >= 0 else -1)
        else:
            fl = sp.floor(sx)
            want = fl if (sx - fl) < half else (fl + 1 if (sx - fl) > half else (fl if fl % 2 == 0 else fl + 1))
        return ((x, mode), int(want), 'SymPy floor with the half rule written in the sim')
    edges = {'rat_floor': [(('-7/2',), -4, 'SymPy')], 'rat_ceil': [(('-7/2',), -3, 'SymPy')], 'rat_trunc': [(('-7/2',), -3, 'SymPy')],
             'frac_part': [(('-7/2',), Fraction(1, 2), 'SymPy')], 'rat_mod': [((1, 0), BAD, 'modulo zero'), ((-7, 3), 2, 'SymPy'), ((7, -3), -2, 'SymPy')],
             'rat_round': [(('5/2',), 3, 'half away'), (('-5/2',), -3, 'half away'), (('5/2', 'half_even'), 2, 'half even'), (('1/3', 'sideways'), BAD, 'unknown mode')]}[name]
    return fill(edges, make, n, rng)


# ---------------------------------------------------------------------------------------------------------------------------------------------------------
# continued fractions and decimals

@gen('cf_rational', 'cf_value', 'cf_convergents')
def g_cf(rng, n, name):
    def make(rng):
        x = Fraction(rng.randint(-10 ** 6, 10 ** 6) if rng.random() < 0.7 else rng.randint(-10 ** 25, 10 ** 25), rng.randint(1, 10 ** 6))
        sx = sp.Rational(x.numerator, x.denominator)
        terms = [int(t) for t in sp.continued_fraction(sx)]
        if name == 'cf_rational':
            return ((x,), terms, 'SymPy continued_fraction')
        if name == 'cf_value':
            return ((terms,), x, 'SymPy continued_fraction_reduce / the rational itself')
        return ((x,), [F(c) for c in sp.continued_fraction_convergents(terms)], 'SymPy continued_fraction_convergents')
    edges = {'cf_rational': [(('415/93',), [4, 2, 6, 7], 'SymPy'), ((5,), [5], 'SymPy'), (('-7/2',), [-4, 2], 'SymPy')],
             'cf_value': [(([4, 2, 6, 7],), Fraction(415, 93), 'SymPy'), (([],), BAD, 'empty'), (([1, 0],), BAD, 'zero term')],
             'cf_convergents': [(('415/93',), [4, Fraction(9, 2), Fraction(58, 13), Fraction(415, 93)], 'SymPy'), (([1, 2, 2, 2],), [1, Fraction(3, 2), Fraction(7, 5), Fraction(17, 12)], 'SymPy')]}[name]
    return fill(edges, make, n, rng)


@gen('cf_sqrt', 'cf_sqrt_convergents')
def g_cfsqrt(rng, n, name):
    def make(rng):
        d = rng.randint(2, 10 ** rng.randint(2, 7 if name == 'cf_sqrt' else 12))
        if name == 'cf_sqrt':
            if sp.integer_nthroot(d, 2)[1]:
                return ((d,), (math.isqrt(d), []), 'perfect square: empty period')
            a0, per = sp.continued_fraction_periodic(0, 1, d)[0], sp.continued_fraction_periodic(0, 1, d)[1]
            return ((d,), (int(a0), [int(t) for t in per]), 'SymPy continued_fraction_periodic')
        if sp.integer_nthroot(d, 2)[1]:
            return ((d, 5), BAD, 'perfect square')
        c = rng.randint(1, 12)
        it = sp.continued_fraction_iterator(sp.sqrt(d))
        terms = [next(it) for _ in range(c)]
        return ((d, c), [F(x) for x in sp.continued_fraction_convergents(terms)], 'SymPy continued_fraction_iterator of sqrt(d)')
    edges = {'cf_sqrt': [((23,), (4, [1, 3, 1, 8]), 'SymPy'), ((2,), (1, [2]), 'SymPy'), ((49,), (7, []), 'perfect square'), ((0,), BAD, 'd = 0')],
             'cf_sqrt_convergents': [((2, 4), [1, Fraction(3, 2), Fraction(7, 5), Fraction(17, 12)], 'SymPy'), ((4, 3), BAD, 'perfect square')]}[name]
    return fill(edges, make, n, rng)


@gen('best_approx')
def g_best(rng, n, name):
    def make(rng):
        x = Fraction(rng.randint(-10 ** 6, 10 ** 6), rng.randint(1, 10 ** 6))
        md = rng.randint(1, 80)
        best = min(abs(Fraction(round(x * q), q) - x) for q in range(1, md + 1))
        return ((x, md), Cert(lambda r, x=x, md=md, best=best: Fraction(r).denominator <= md and abs(Fraction(r) - x) == best, 'denominator bound and distance equals the brute-force minimum'),
                'brute force over denominators')
    return fill([(('3141592653/1000000000', 100), Cert(lambda r: r == Fraction(311, 99)), 'brute force'), ((1, 0), BAD, 'max_den 0')], make, n, rng)


def long_division(x, digits):
    """First `digits` digits after the point of |x| by schoolbook division, and the (preperiod, period) found by remainder cycle detection."""
    x = abs(x)
    r = x.numerator % x.denominator
    out, seen = [], {}
    pre = per = None
    i = 0
    while True:
        if r == 0:
            pre, per = i, 0
            break
        if r in seen:
            pre, per = seen[r], i - seen[r]
            break
        seen[r] = i
        r *= 10
        out.append(r // x.denominator)
        r %= x.denominator
        i += 1
    return out, pre, per


def digits_of(x, k):
    x = abs(x)
    r = x.numerator % x.denominator
    out = []
    for _ in range(k):
        r *= 10
        out.append(r // x.denominator)
        r %= x.denominator
    return out


@gen('decimal_expansion')
def g_decexp(rng, n, name):
    def make(rng):
        x = rrat(rng, rng.random() < 0.3)
        q = to_sp(x)
        k = rng.randint(0, 60)
        fx = Fraction(int(q.p), int(q.q))
        ip = abs(fx.numerator) // fx.denominator
        s = str(ip)
        if k:
            s += '.' + ''.join(map(str, digits_of(fx, k)))
        neg = fx < 0 and any(c not in '0.' for c in s)
        return ((x, k), ('-' if neg else '') + s, 'schoolbook long division in the sim')
    return fill([(('1/7', 12), '0.142857142857', 'long division'), ((5, 0), '5', 'long division'), (('1/3', -1), BAD, 'negative digits')], make, n, rng)


@gen('decimal_digit')
def g_decdigit(rng, n, name):
    def make(rng):
        den = rng.choice([7, 13, 17, 19, 23, 97, 101, 6, 12, 14, 28, 999, 1001, 4999, 10007, 65537, 30, 11, 3])
        num = rng.randint(1, 3 * den)
        x = Fraction(num, den)
        _, pre, per = long_division(x, 0)
        k = rng.randint(1, 200) if rng.random() < 0.5 else rng.randint(1, 10 ** rng.randint(3, 18))
        # independent: the digit at position k lies in the cycle when k > pre, so walk the cycle from the long division
        digs, pre, per = long_division(x, 0)
        if per == 0:
            want = digs[k - 1] if k <= len(digs) else 0
        elif k <= pre:
            want = digs[k - 1]
        else:
            want = digs[pre + (k - 1 - pre) % per]
        return ((x, k), want, 'long division with remainder-cycle detection, indexed through the cycle')
    return fill([(('1/7', 2024), '142857'[(2024 - 1) % 6] and int('142857'[(2024 - 1) % 6]), 'cycle'), (('1/7', 0), BAD, 'k = 0')], make, n, rng)


@gen('decimal_period', 'decimal_preperiod', 'repeating_decimal')
def g_period(rng, n, name):
    def make(rng):
        den = rng.choice([rng.randint(1, 3000), rng.randint(1, 3000), rng.randint(3000, 20000)])
        x = Fraction(rng.randint(0, 3 * den), den)
        digs, pre, per = long_division(x, 0)
        if name == 'decimal_period':
            return ((x,), per, 'long division remainder cycle')
        if name == 'decimal_preperiod':
            return ((x,), pre, 'long division remainder cycle')
        ip = x.numerator // x.denominator
        s = ''.join(map(str, digs))
        return ((x,), (ip, s[:pre], s[pre:pre + per]), 'long division remainder cycle')
    edges = {'decimal_period': [(('1/7',), 6, 'known'), (('1/6',), 1, 'known'), (('1/8',), 0, 'known')],
             'decimal_preperiod': [(('1/6',), 1, 'known'), (('1/7',), 0, 'known'), (('1/80',), 4, 'known')],
             'repeating_decimal': [(('1/6',), (0, '1', '6'), 'known'), (('22/7',), (3, '', '142857'), 'known'), (('1/4',), (0, '25', ''), 'known'), (('-1/3',), BAD, 'negative')]}[name]
    return fill(edges, make, n, rng)


@gen('repeating_to_fraction')
def g_rep2frac(rng, n, name):
    def make(rng):
        den = rng.randint(1, 5000)
        x = Fraction(rng.randint(0, 4 * den), den)
        digs, pre, per = long_division(x, 0)
        s = ''.join(map(str, digs))
        return ((x.numerator // x.denominator, s[:pre], s[pre:pre + per]), x, 'round trip from the sim long division')
    return fill([((0, '1', '6'), Fraction(1, 6), 'known'), ((0, '', '9'), Fraction(1), '0.999... = 1'), ((0, 'x', '1'), BAD, 'not digits'), ((1, '', ''), Fraction(1), 'integer')], make, n, rng)


# ---------------------------------------------------------------------------------------------------------------------------------------------------------
# Diophantine

@gen('lin_diophantine')
def g_lindio(rng, n, name):
    def make(rng):
        big = rng.random() < 0.4
        lim = 10 ** (rng.randint(5, 20) if big else rng.randint(1, 2))
        a, b = rng.randint(-lim, lim), rng.randint(-lim, lim)
        if a == 0 and b == 0:
            a = 1
        g = int(sp.igcd(a, b))
        c = rng.randint(-3 * lim, 3 * lim) if rng.random() < 0.3 else g * rng.randint(-lim, lim)

        def check(r, a=a, b=b, c=c, g=g):
            if c % g:
                return r == ()
            x0, y0, dx, dy = r
            if not all(a * (x0 + dx * t) + b * (y0 + dy * t) == c for t in (-2, 0, 1, 7)):
                return False
            if a * dx + b * dy != 0 or math.gcd(dx, dy) != 1:
                return False
            return abs(dx) == abs(b) // g and abs(dy) == abs(a) // g
        return ((a, b, c), Cert(check, 'solves the equation for several t, step is (b/g, -a/g), solvable iff g | c (g from SymPy igcd)'), 'certificate')
    edges = [((3, 5, 7), Cert(lambda r: r == (4, -1, 5, -3)), 'cert'), ((6, 4, 3), Cert(lambda r: r == ()), 'cert'), ((0, 0, 3), Cert(lambda r: r == ()), 'cert'),
             ((0, 0, 0), REFUSE, 'every pair'), ((0, 4, 8), Cert(lambda r: r[1] == 2 and r[0] == 0), 'cert'), ((7, 0, 14), Cert(lambda r: r[0] == 2 and r[1] == 0), 'cert')]
    return fill(edges, make, n, rng)


@gen('lin_diophantine_nonneg_count')
def g_lindio_count(rng, n, name):
    def make(rng):
        a, b = rng.randint(1, 30), rng.randint(1, 30)
        c = rng.randint(0, 1500)
        cnt = sum(1 for x in range(c // a + 1) if (c - a * x) % b == 0)
        return ((a, b, c), cnt, 'brute force over x')
    return fill([((3, 5, 100), 7, 'brute'), ((0, 5, 10), BAD, 'a = 0'), ((3, 5, -1), BAD, 'c < 0')], make, n, rng)


@gen('pell')
def g_pell(rng, n, name):
    def make(rng):
        d = rng.randint(2, rng.choice([100, 1000, 10 ** 4, 10 ** 5]))
        if sp.integer_nthroot(d, 2)[1]:
            return ((d,), (), 'square d: no non-trivial solution')
        s = diop_DN(d, 1)
        return ((d,), (int(s[0][0]), int(s[0][1])), 'SymPy diop_DN(d, 1)')
    return fill([((61,), (1766319049, 226153980), 'SymPy'), ((2,), (3, 2), 'SymPy'), ((4,), (), 'square'), ((0,), BAD, 'd = 0'), ((109,), (158070671986249, 15140424455100), 'SymPy')], make, n, rng)


@gen('pell_negative')
def g_pellneg(rng, n, name):
    def make(rng):
        d = rng.randint(2, rng.choice([100, 1000, 10 ** 4, 10 ** 5]))
        if sp.integer_nthroot(d, 2)[1]:
            return ((d,), (), 'square d')
        s = diop_DN(d, -1)
        return ((d,), (int(s[0][0]), int(s[0][1])) if s else (), 'SymPy diop_DN(d, -1)')
    return fill([((13,), (18, 5), 'SymPy'), ((3,), (), 'SymPy'), ((2,), (1, 1), 'SymPy')], make, n, rng)


@gen('pell_solutions')
def g_pellsol(rng, n, name):
    def make(rng):
        d = rng.choice([2, 3, 5, 6, 7, 8, 10, 11, 13, 14, 15, 17, 19, 21, 23, 29, 31, 41, 61])
        k = rng.randint(1, 6)
        base = diop_DN(d, 1)[0]
        x1, y1 = int(base[0]), int(base[1])
        sols = []
        for j in range(1, k + 1):          # (x1 + y1 sqrt d)^j by sympy's exact expansion
            e = sp.expand((x1 + y1 * sp.sqrt(d)) ** j)
            sols.append((int(e.coeff(sp.sqrt(d), 0)), int(e.coeff(sp.sqrt(d), 1))))
        return ((d, k), sols, 'SymPy exact powers of the fundamental unit')
    return fill([((2, 3), [(3, 2), (17, 12), (99, 70)], 'SymPy'), ((4, 3), [], 'square d'), ((2, 0), BAD, 'count 0')], make, n, rng)


@gen('pell_search')
def g_pellsearch(rng, n, name):
    def make(rng):
        d = rng.randint(1, 40)
        N = rng.randint(-30, 30)
        ymax = rng.randint(0, 120)
        want = []
        for y in range(ymax + 1):
            for x in range(0, 40 * ymax + 300):          # a plain scan over x as well (a different loop from isqrt)
                v = x * x - d * y * y
                if v == N:
                    want.append((x, y))
                if v > N and x * x > d * y * y + 30:
                    break
        return ((d, N, ymax), want, 'double loop over x and y in the sim')
    return fill([((2, -1, 100), [(1, 1), (7, 5), (41, 29)], 'known'), ((0, 1, 5), BAD, 'd = 0')], make, n, rng)


@gen('pythagorean_triples', 'pythagorean_triple_count')
def g_pyth(rng, n, name):
    def make(rng):
        N = rng.randint(0, 250)
        prim = rng.random() < 0.4
        trip = [(a, b, c) for c in range(1, N + 1) for b in range(1, c) for a in range(1, b) if a * a + b * b == c * c]
        if name == 'pythagorean_triple_count':
            return ((N,), len(trip), 'triple loop')
        if prim:
            trip = [t for t in trip if math.gcd(math.gcd(t[0], t[1]), t[2]) == 1]
        return ((N, prim), sorted(trip), 'triple loop')
    return fill([((20,), 6 if name == 'pythagorean_triple_count' else [(3, 4, 5), (5, 12, 13), (6, 8, 10), (8, 15, 17), (9, 12, 15), (12, 16, 20)], 'triple loop')],
                make, n, rng)


# ---------------------------------------------------------------------------------------------------------------------------------------------------------
# sums of squares

def r2_divisors(v):
    """Jacobi's two-square theorem from SymPy's divisor list: 4 * (divisors = 1 mod 4 minus divisors = 3 mod 4). A different route from the backend's factor formula."""
    if v == 0:
        return 1
    return 4 * sum((1 if d % 4 == 1 else -1) for d in sp.divisors(v) if d % 2)


def signed_count(reps):
    tot = 0
    for a, b in reps:
        tot += len({(sa * a, sb * b) for sa in (1, -1) for sb in (1, -1)} | {(sa * b, sb * a) for sa in (1, -1) for sb in (1, -1)})
    return tot


@gen('two_squares', 'is_sum_of_two_squares', 'r2')
def g_two(rng, n, name):
    def make(rng):
        r = rng.random()
        if r < 0.35:
            v = rng.randint(0, 3 * 10 ** rng.randint(1, 5))
        elif r < 0.8:
            ps = [int(sp.randprime(2, 10 ** 4)) for _ in range(rng.randint(1, 4))]
            ps = [p for p in ps if p % 4 != 3] + [p * p for p in ps if p % 4 == 3]
            v = 1
            for p in ps:
                v *= p
            v *= rng.choice([1, 2, 4, 5, 25, 13 * 17, 65 * 85])
        else:
            v = rng.randint(1, 10 ** rng.randint(6, 13))
        small = v <= 300_000
        if small:
            brute = sorted((a, b) for a in range(math.isqrt(v // 2) + 1) for b in [math.isqrt(v - a * a)] if b >= a and a * a + b * b == v)
        if name == 'two_squares':
            if small:
                return ((v,), brute, 'brute force over a')
            want = r2_divisors(v)

            def check(reps, v=v, want=want):
                return (all(0 <= a <= b and a * a + b * b == v for a, b in reps) and len(set(reps)) == len(reps) and list(reps) == sorted(reps)
                        and signed_count(reps) == want)
            return ((v,), Cert(check, 'each pair squares back, distinct, and the signed count equals Jacobi 4(d1 - d3) from SymPy divisors: complete'), 'certificate')
        want = (signed_count(brute) if v else 1) if small else r2_divisors(v)
        if name == 'is_sum_of_two_squares':
            return ((v,), want > 0, 'brute force' if small else 'Jacobi 4(d1 - d3) from SymPy divisors')
        return ((v,), want, 'brute force' if small else 'Jacobi 4(d1 - d3) from SymPy divisors')
    edges = [((50,), [(1, 7), (5, 5)] if name == 'two_squares' else (True if name == 'is_sum_of_two_squares' else 12), 'brute'),
             ((3,), [] if name == 'two_squares' else (False if name == 'is_sum_of_two_squares' else 0), 'brute'),
             ((0,), [(0, 0)] if name == 'two_squares' else (True if name == 'is_sum_of_two_squares' else 1), 'convention'), ((-1,), BAD, 'negative')]
    return fill(edges, make, n, rng)


@gen('is_sum_of_three_squares')
def g_three(rng, n, name):
    def make(rng):
        v = rng.randint(0, 2500)
        squares = {i * i for i in range(51)}
        ok = any((v - a) in {x + y for x in squares for y in squares} for a in squares if a <= v)
        return ((v,), ok, 'brute force over three squares')
    return fill([((7,), False, 'known'), ((0,), True, 'known'), ((28,), False, '4*7'), ((-3,), BAD, 'negative')], make, n, rng)


@gen('r4')
def g_r4(rng, n, name):
    pair = {}
    for x in range(-40, 41):
        for y in range(-40, 41):
            pair[x * x + y * y] = pair.get(x * x + y * y, 0) + 1

    def make(rng):
        v = rng.randint(1, 1500)
        return ((v,), sum(c * pair.get(v - s, 0) for s, c in pair.items() if s <= v), 'brute force: convolution of two-square counts')
    return fill([((1,), 8, 'known'), ((5,), 48, 'known'), ((0,), BAD, 'n = 0')], make, n, rng)


@gen('four_squares')
def g_four(rng, n, name):
    def make(rng):
        v = rng.randint(0, 10 ** rng.randint(1, 18)) if rng.random() < 0.8 else 4 ** rng.randint(1, 12) * rng.choice([7, 15, 23, 31, 1])
        return ((v,), Cert(lambda r, v=v: len(r) == 4 and all(0 <= r[i] <= r[i + 1] for i in range(3)) and sum(x * x for x in r) == v, 'four sorted non-negative squares sum to n'), 'certificate')
    return fill([((7,), Cert(lambda r: r == (1, 1, 1, 2)), 'cert'), ((0,), Cert(lambda r: r == (0, 0, 0, 0)), 'cert'), ((-1,), BAD, 'negative')], make, n, rng)


# ---------------------------------------------------------------------------------------------------------------------------------------------------------
# Frobenius, floor sum

def reach(coins, lim):
    ok = [False] * (lim + 1)
    ok[0] = True
    for v in range(1, lim + 1):
        ok[v] = any(v >= c and ok[v - c] for c in coins)
    return ok


@gen('frobenius', 'frobenius_count')
def g_frob(rng, n, name):
    def make(rng):
        k = rng.randint(2, 4)
        coins = rng.sample(range(2, 45), k)
        g = math.gcd(*coins)
        if g != 1:
            return (tuple(coins), (), 'common factor: infinitely many gaps')
        lim = max(coins) * min(coins) + 50
        ok = reach(coins, lim)
        gaps = [v for v in range(lim + 1) if not ok[v]]
        if name == 'frobenius':
            return (tuple(coins), max(gaps) if gaps else -1, 'dynamic programming over reachable amounts')
        return (tuple(coins), len(gaps), 'dynamic programming over reachable amounts')
    edges = {'frobenius': [((3, 5), 7, 'DP'), ((6, 9, 20), 43, 'DP'), ((4, 6), (), 'gcd'), ((1, 7), -1, 'DP'), ((0, 5), BAD, 'zero coin')],
             'frobenius_count': [((3, 5), 4, 'DP'), ((6, 9, 20), 22 if False else len([v for v in range(2000) if not reach([6, 9, 20], 2000)[v]]), 'DP'), ((1, 7), 0, 'DP')]}[name]
    return fill(edges, make, n, rng)


@gen('floor_sum')
def g_floor_sum(rng, n, name):
    def make(rng):
        if rng.random() < 0.6:
            nn, m, a, b = rng.randint(0, 60), rng.randint(1, 50), rng.randint(-100, 100), rng.randint(-100, 100)
        else:
            nn, m, a, b = rng.randint(0, 20000), rng.randint(1, 10 ** 9), rng.randint(-10 ** 9, 10 ** 9), rng.randint(-10 ** 9, 10 ** 9)
        return ((nn, m, a, b), sum((a * i + b) // m for i in range(nn)), 'direct summation of the floors')
    return fill([((10, 7, 3, 2), sum((3 * i + 2) // 7 for i in range(10)), 'direct'), ((0, 5, 1, 1), 0, 'empty sum'), ((5, 0, 1, 1), BAD, 'm = 0'), ((-1, 5, 1, 1), BAD, 'n < 0')],
                make, n, rng)


# ---------------------------------------------------------------------------------------------------------------------------------------------------------
# runner

def _reason(msg):
    return re.sub(r'\d[\d,_]*', '#', str(msg))[:80]


def check_case(opname, args, oracle):
    """('agree'|'disagree'|'refused', detail)."""
    try:
        res = backend_run(opname, *args)
    except NotCovered as e:
        if oracle is REFUSE:
            return 'refused_expected', _reason(e)
        return 'refused', _reason(e)
    except BadInput as e:
        if oracle is BAD:
            return 'agree', ''
        return 'disagree', f'unexpected BadInput: {e}'
    except Exception as e:                                  # an exception that is neither is a bug in the backend
        return 'disagree', f'crash {type(e).__name__}: {e}'
    if oracle is BAD:
        return 'disagree', f'expected BadInput, got {str(res)[:80]}'
    if oracle is REFUSE:
        return 'disagree', f'expected a refusal, got {str(res)[:80]}'
    if isinstance(oracle, Cert):
        try:
            ok = oracle.check(res)
        except Exception as e:
            return 'disagree', f'certificate crashed {type(e).__name__}: {e} on {str(res)[:80]}'
        return ('agree', '') if ok else ('disagree', f'certificate failed ({oracle.desc}) on {str(res)[:120]}')
    if norm(res) == norm(oracle):
        return 'agree', ''
    return 'disagree', f'got {str(res)[:100]} expected {str(oracle)[:100]}'


def run_sim(seed=1, n=60, only=None, out_dir=None):
    n = max(int(n), 40)
    reg = {k: v for k, v in registry().items() if k.startswith(MODULE + '.')}
    report = {'module': MODULE, 'seed': seed, 'n_per_op': n, 'sympy': sp.__version__, 'ops': {}, 'no_case': []}
    for name in sorted(reg):
        if only and name not in only:
            continue
        if name not in GEN:
            report['no_case'].append(name)
            continue
        rng = random.Random(f'{seed}:{name}')
        t0 = time.time()
        cases = GEN[name](rng, n)
        print(f'  {name}: cases built in {time.time() - t0:.1f} s', file=sys.stderr, flush=True)
        rec = dict(kind=reg[name]['kind'], bound=reg[name]['bound'], generated=len(cases), answered=0, agree=0, refused=0, refused_expected=0, refused_reasons={},
                   disagree=0, disagree_examples=[], oracles={})
        for args, oracle, oname in cases:
            rec['oracles'][oname] = rec['oracles'].get(oname, 0) + 1
            kind, detail = check_case(name, args, oracle)
            if kind == 'agree':
                rec['answered'] += 1
                rec['agree'] += 1
            elif kind == 'refused_expected':
                rec['refused_expected'] += 1
                rec['agree'] += 1
                rec['answered'] += 1
            elif kind == 'refused':
                rec['refused'] += 1
                rec['refused_reasons'][detail] = rec['refused_reasons'].get(detail, 0) + 1
            else:
                rec['answered'] += 1
                rec['disagree'] += 1
                if len(rec['disagree_examples']) < 5:
                    rec['disagree_examples'].append({'args': str(args)[:200], 'detail': detail})
        rec['seconds'] = round(time.time() - t0, 2)
        print(f'  {name}: {rec["generated"]} cases, disagree {rec["disagree"]}, refused {rec["refused"]}, {rec["seconds"]} s', file=sys.stderr, flush=True)
        report['ops'][name] = rec
    ops = report['ops'].values()
    report['totals'] = dict(registered=len(reg), with_cases=len(report['ops']), generated=sum(r['generated'] for r in ops), answered=sum(r['answered'] for r in ops),
                            agree=sum(r['agree'] for r in ops), refused=sum(r['refused'] for r in ops), disagree=sum(r['disagree'] for r in ops))
    report['not_covered'] = backend.NOT_COVERED
    out = Path(out_dir) if out_dir else ROOT / 'build-mathbackend'
    out.mkdir(parents=True, exist_ok=True)
    (out / f'{MODULE}.json').write_text(json.dumps(report, indent=1, ensure_ascii=False, default=str), encoding='utf-8')
    return report


def print_table(report):
    rows = report['ops']
    w = max(len(k) for k in rows) if rows else 10
    print(f"{'operation':<{w}}  gen  answ  agree  refused  disagree  sec")
    for name, r in rows.items():
        print(f"{name:<{w}}  {r['generated']:>3}  {r['answered']:>4}  {r['agree']:>5}  {r['refused']:>7}  {r['disagree']:>8}  {r['seconds']:>5}")
        for why, c in r['refused_reasons'].items():
            print(f'{"":<{w}}    refused x{c}: {why}')
        for ex in r['disagree_examples']:
            print(f'{"":<{w}}    DISAGREE {ex["args"]} -> {ex["detail"]}')
    t = report['totals']
    print(f"\n{t['registered']} operations registered, {t['with_cases']} with cases; generated {t['generated']}, answered {t['answered']}, agree {t['agree']}, "
          f"refused {t['refused']}, disagree {t['disagree']}")
    if report['no_case']:
        print('operations with no case yet:', ', '.join(report['no_case']))


def generate(seed, n):
    """The cases of every operation: a list of (op_name, args, oracle_value, oracle_name)."""
    out = []
    for name in sorted(GEN):
        rng = random.Random(f'{seed}:{name}')
        out += [(name, a, o, on) for a, o, on in GEN[name](rng, max(int(n), 40))]
    return out


def run(seed, n):
    """Run the simulation and return the report dict (also written to build-mathbackend/numbers.json)."""
    return run_sim(seed, n)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--n', type=int, default=60)
    ap.add_argument('--only', nargs='*')
    a = ap.parse_args()
    rep = run_sim(a.seed, a.n, set(a.only) if a.only else None)
    print_table(rep)
    sys.exit(1 if rep['totals']['disagree'] else 0)
