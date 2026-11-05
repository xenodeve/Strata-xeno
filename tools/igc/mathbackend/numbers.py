"""MIP math backend, module `numbers`: exact integer and rational arithmetic, primes and factoring, modular arithmetic, number-theoretic functions,
combinatorial numbers, digits and bases, continued fractions, decimal expansions, Diophantine equations, Pell, sums of squares, Frobenius, floor sum.

Standard library only. Written from the textbook algorithms (nothing copied from another project; see third_party/mathbackend/NOTICES_numbers.md).
Contract: docs/plans/2026-11-05-mip-math-backend-coverage.md. Integers are `int`, rationals `Fraction`, no floats anywhere.

Three kinds of answer:
  a value                  the result;
  `()` or `[]`             a legitimate "there is none" (no inverse, no solution, no primitive root, a non-residue) - the answer is "none", not a refusal;
  NotCovered / BadInput    the operation will not answer (past its bound, or not exactly decidable here) / the input is outside the domain.

    from tools.igc.mathbackend import run
    run('numbers.factorint', 360)        # {2: 3, 3: 2, 5: 1}
"""
import heapq
import math
from fractions import Fraction

from tools.igc.mathbackend import BadInput, NotCovered, op

# What this module does not cover, with the reason (the simulation report and the final report quote it).
NOT_COVERED = {
    'factoring past the bound': 'trial division, Pollard p-1 and Brent rho only; a number whose second-largest prime factor is above about 1e11 is refused (ECM, '
                                'quadratic sieve and NFS would cover it)',
    'primality proof of a large prime': 'above 3.3e24 a prime is proven only by Pocklington with a completely factored part of n-1 above sqrt(n); else refused '
                                       '(the composite verdict is always certain)',
    'general Pell x^2 - d*y^2 = N, all solution classes': 'only a bounded search over y (pell_search) is given; the fundamental solutions of every class need '
                                                         'the LMM algorithm',
    'general quadratic Diophantine equations': 'a*x^2 + b*x*y + c*y^2 + d*x + e*y + f = 0 is not done (needs the full conic classification)',
    'linear Diophantine in more than two unknowns': 'not done; reduce by hand with two-unknown steps',
    'discrete logarithm past n = 1e12': 'baby-step giant-step needs sqrt(n) memory; Pohlig-Hellman for smooth orders is not implemented',
    'partition numbers past n = 10000, Bell past 1500, Stirling past 3000': 'exact big-integer recurrences are quadratic; no Rademacher series',
    'egyptian fractions, quadratic forms, p-adic and algebraic number fields': 'not in the floor list; nothing registered',
}

_BITS = 2_000_000            # the largest result of a power, in bits
_MAX_BITS_FACTOR = 240       # factorint: n below 2**240
_RHO_STEPS = 250_000         # Brent rho steps per try; two tries
_MR_LIMIT = 3317044064679887385961981   # the first 13 prime bases are a proof below this
_MR_BASES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41)


# ----------------------------------------------------------------------------------------------------------------------------------------------------------
# argument checks

def _int(x, name='argument'):
    if isinstance(x, bool):
        raise BadInput(f'{name} must be an integer, not a bool')
    if isinstance(x, int):
        return x
    if isinstance(x, Fraction) and x.denominator == 1:
        return int(x.numerator)
    raise BadInput(f'{name} must be an integer')


def _rat(x, name='argument'):
    if isinstance(x, bool):
        raise BadInput(f'{name} must be a rational number')
    if isinstance(x, (int, Fraction)):
        return Fraction(x)
    if isinstance(x, str):
        s = x.strip()
        if len(s) > 4000 or 'e' in s.lower() or 'n' in s.lower():
            raise BadInput(f'{name}: not a rational number')
        try:
            return Fraction(s)
        except (ValueError, ZeroDivisionError):
            raise BadInput(f'{name}: not a rational number')
    raise BadInput(f'{name} must be an int, a Fraction or a string such as "3/4"')


def _small(n, hi, name):
    if n > hi:
        raise NotCovered(f'{name} too large (bound {hi})')


def _pos(n, name='argument'):
    if n < 1:
        raise BadInput(f'{name} must be positive')


def _nonneg(n, name='argument'):
    if n < 0:
        raise BadInput(f'{name} must not be negative')


def _bits(f):
    return max(f.numerator.bit_length(), f.denominator.bit_length())


# ----------------------------------------------------------------------------------------------------------------------------------------------------------
# exact rational arithmetic, powers and roots

@op('numbers.add', kind='exact arithmetic', bound='any size')
def add(a, b):
    """Exact sum of two rationals (int, Fraction or "p/q"). add(1, "1/2") -> 3/2."""
    return _rat(a) + _rat(b)


@op('numbers.sub', kind='exact arithmetic', bound='any size')
def sub(a, b):
    """Exact difference of two rationals. sub("1/2", "1/3") -> 1/6."""
    return _rat(a) - _rat(b)


@op('numbers.mul', kind='exact arithmetic', bound='any size')
def mul(a, b):
    """Exact product of two rationals. mul("2/3", "9/4") -> 3/2."""
    return _rat(a) * _rat(b)


@op('numbers.div', kind='exact arithmetic', bound='any size')
def div(a, b):
    """Exact quotient of two rationals; a zero divisor is BadInput. div(1, 3) -> 1/3."""
    b = _rat(b)
    if b == 0:
        raise BadInput('division by zero')
    return _rat(a) / b


@op('numbers.pow_int', kind='exact arithmetic', bound='result at most 2,000,000 bits')
def pow_int(a, n):
    """a**n for a rational a and an integer n (negative n inverts). pow_int("2/3", -3) -> 27/8."""
    a, n = _rat(a), _int(n, 'exponent')
    if a == 0:
        if n < 0:
            raise BadInput('0 to a negative power')
        return Fraction(1 if n == 0 else 0)
    if a in (1, -1):
        return Fraction(1) if (a == 1 or n % 2 == 0) else Fraction(-1)
    if _bits(a) * abs(n) > _BITS:
        raise NotCovered('too large')
    return a ** n


def _iroot(n, k):
    """floor(n ** (1/k)) for n >= 0, k >= 1, by integer Newton iteration."""
    if n < 2 or k == 1:
        return n
    if k == 2:
        return math.isqrt(n)
    b = n.bit_length()
    if k >= b:
        return 1
    x = 1 << -(-b // k)        # at least the root
    while True:
        y = ((k - 1) * x + n // x ** (k - 1)) // k
        if y >= x:
            return x
        x = y


@op('numbers.iroot', kind='exact arithmetic', bound='n up to 2,000,000 bits')
def iroot(n, k):
    """floor of the k-th root of n >= 0. iroot(1000, 3) -> 10; iroot(999, 3) -> 9."""
    n, k = _int(n), _int(k, 'k')
    _nonneg(n, 'n')
    _pos(k, 'k')
    _small(n.bit_length(), _BITS, 'n')
    return _iroot(n, k)


@op('numbers.isqrt', kind='exact arithmetic', bound='n up to 2,000,000 bits')
def isqrt_floor(n):
    """floor(sqrt(n)) for n >= 0. isqrt(17) -> 4."""
    n = _int(n)
    _nonneg(n, 'n')
    _small(n.bit_length(), _BITS, 'n')
    return math.isqrt(n)


@op('numbers.is_square', kind='exact arithmetic', bound='n up to 2,000,000 bits')
def is_square(n):
    """True when the integer n (or the rational n) is the square of a rational. is_square(49) -> True; is_square("4/9") -> True."""
    f = _rat(n)
    if f < 0:
        return False
    _small(_bits(f), _BITS, 'n')
    a, b = f.numerator, f.denominator
    return math.isqrt(a) ** 2 == a and math.isqrt(b) ** 2 == b


def _perfect_power(n):
    """(b, e) with b**e == n and e as large as possible (e = 1 when n is no perfect power). n is an integer with |n| >= 2."""
    neg = n < 0
    base, e_total = abs(n), 1
    top = base.bit_length()
    sieve = bytearray([1]) * (top + 1)
    for p in range(2, top + 1):
        if not sieve[p]:
            continue
        for q in range(p * p, top + 1, p):
            sieve[q] = 0
        if neg and p == 2:
            continue
        while True:
            r = _iroot(base, p)
            if r > 1 and r ** p == base:
                base, e_total = r, e_total * p
            else:
                break
    return (-base if neg else base), e_total


@op('numbers.perfect_power', kind='exact arithmetic', bound='|n| up to 20,000 bits')
def perfect_power(n):
    """(b, e) with n = b**e and e maximal; (n, 1) when n is not a perfect power; a negative n only with an odd e. perfect_power(64) -> (2, 6)."""
    n = _int(n)
    if n in (0, 1, -1):
        return (n, 1)
    _small(abs(n).bit_length(), 20000, 'n')
    return _perfect_power(n)


@op('numbers.is_perfect_power', kind='exact arithmetic', bound='|n| up to 20,000 bits')
def is_perfect_power(n):
    """True when n = b**e with e >= 2 (0, 1 and -1 count). is_perfect_power(72) -> False; is_perfect_power(-27) -> True."""
    return perfect_power(n)[1] > 1 or _int(n) in (0, 1, -1)


@op('numbers.pow_rational', kind='exact arithmetic', bound='result at most 2,000,000 bits; root index up to 10**6')
def pow_rational(a, e):
    """a**e for rationals a and e when the result is rational; NotCovered when the root is not exact (no float, no radical) or when a < 0 with a fractional e.
    pow_rational(8, "2/3") -> 4; pow_rational("9/4", "-1/2") -> 2/3."""
    a, e = _rat(a), _rat(e, 'exponent')
    if e.denominator == 1:
        return pow_int(a, e.numerator)
    r, s = e.numerator, e.denominator
    if a < 0:
        raise NotCovered('negative base with a fractional exponent: the real root and the principal value differ')
    if a == 0:
        if r < 0:
            raise BadInput('0 to a negative power')
        return Fraction(0)
    _small(s, 10 ** 6, 'root index')
    _small(_bits(a), _BITS, 'base')
    num, den = _iroot(a.numerator, s), _iroot(a.denominator, s)
    if num ** s != a.numerator or den ** s != a.denominator:
        raise NotCovered('not an exact power')
    return pow_int(Fraction(num, den), r)


# ----------------------------------------------------------------------------------------------------------------------------------------------------------
# gcd, modular arithmetic

@op('numbers.gcd', kind='modular arithmetic', bound='any size')
def gcd(*xs):
    """Greatest common divisor of one or more integers (non-negative). gcd(12, 18, 30) -> 6."""
    if not xs:
        raise BadInput('gcd of nothing')
    return math.gcd(*[_int(x) for x in xs])


@op('numbers.lcm', kind='modular arithmetic', bound='result at most 2,000,000 bits')
def lcm(*xs):
    """Least common multiple of one or more integers (non-negative; 0 if any is 0). lcm(4, 6, 10) -> 60."""
    if not xs:
        raise BadInput('lcm of nothing')
    out = 1
    for x in xs:
        out = math.lcm(out, _int(x))
        _small(out.bit_length(), _BITS, 'lcm')
    return out


def _xgcd(a, b):
    x0, x1, y0, y1 = 1, 0, 0, 1
    while b:
        q = a // b
        a, b = b, a - q * b
        x0, x1 = x1, x0 - q * x1
        y0, y1 = y1, y0 - q * y1
    if a < 0:
        return -a, -x0, -y0
    return a, x0, y0


@op('numbers.xgcd', kind='modular arithmetic', bound='any size')
def xgcd(a, b):
    """(g, x, y) with a*x + b*y = g = gcd(a, b) >= 0. xgcd(240, 46) -> (2, -9, 47)."""
    return _xgcd(_int(a), _int(b))


@op('numbers.modinv', kind='modular arithmetic', bound='any size')
def modinv(a, m):
    """The inverse of a mod m (0 <= x < m), or () when a and m are not coprime. modinv(3, 7) -> 5; modinv(2, 4) -> ()."""
    a, m = _int(a), _int(m)
    _pos(m, 'modulus')
    g, x, _ = _xgcd(a % m, m)
    if g != 1:
        return ()
    return x % m


@op('numbers.powmod', kind='modular arithmetic', bound='any size')
def powmod(a, e, m):
    """a**e mod m (m >= 1; a negative e uses the inverse and gives () when there is none). powmod(2, 100, 1000000007) -> 976371285."""
    a, e, m = _int(a), _int(e, 'exponent'), _int(m)
    _pos(m, 'modulus')
    if e < 0:
        inv = modinv(a, m)
        if inv == ():
            return ()
        return pow(inv, -e, m)
    return pow(a, e, m)


def _crt2(r1, m1, r2, m2):
    g = math.gcd(m1, m2)
    if (r2 - r1) % g:
        return None
    m2g = m2 // g
    t = ((r2 - r1) // g * pow(m1 // g, -1, m2g)) % m2g if m2g > 1 else 0
    lcm_ = m1 // g * m2
    return (r1 + m1 * t) % lcm_, lcm_


@op('numbers.crt', kind='modular arithmetic', bound='any size')
def crt(residues, moduli):
    """Chinese remainder theorem, moduli need not be coprime: (r, M) with M the lcm and x = r (mod M) the solutions, or () when the system is inconsistent.
    crt([2, 3], [3, 5]) -> (8, 15); crt([1, 0], [2, 4]) -> ()."""
    if len(residues) != len(moduli) or not residues:
        raise BadInput('residues and moduli must be lists of the same positive length')
    r, m = 0, 1
    for ri, mi in zip(residues, moduli):
        ri, mi = _int(ri), _int(mi, 'modulus')
        _pos(mi, 'modulus')
        out = _crt2(r, m, ri % mi, mi)
        if out is None:
            return ()
        r, m = out
    return (r, m)


@op('numbers.lincong', kind='modular arithmetic', bound='any size')
def lincong(a, b, n):
    """All solutions of a*x = b (mod n) as (x0, step): x = x0 + step*t, 0 <= x0 < step, or () when there is none. lincong(6, 4, 10) -> (4, 5)."""
    a, b, n = _int(a), _int(b), _int(n)
    _pos(n, 'modulus')
    g = math.gcd(a, n)
    if b % g:
        return ()
    n2 = n // g
    x0 = (b // g) * pow(a // g, -1, n2) % n2 if n2 > 1 else 0
    return (x0, n2)


def _jacobi(a, n):
    a %= n
    t = 1
    while a:
        while a % 2 == 0:
            a //= 2
            if n % 8 in (3, 5):
                t = -t
        a, n = n, a
        if a % 4 == 3 and n % 4 == 3:
            t = -t
        a %= n
    return t if n == 1 else 0


@op('numbers.jacobi', kind='modular arithmetic', bound='any size')
def jacobi(a, n):
    """Jacobi symbol (a/n) for odd n >= 1: -1, 0 or 1. jacobi(2, 15) -> 1."""
    a, n = _int(a), _int(n)
    if n < 1 or n % 2 == 0:
        raise BadInput('n must be odd and positive')
    return _jacobi(a, n)


@op('numbers.legendre', kind='modular arithmetic', bound='p < 3.3e24 or a proven prime')
def legendre(a, p):
    """Legendre symbol (a/p) for an odd prime p: -1, 0 or 1. legendre(3, 7) -> -1."""
    a, p = _int(a), _int(p)
    if p < 3 or p % 2 == 0 or not is_prime(p):
        raise BadInput('p must be an odd prime')
    return _jacobi(a, p)


def _sqrt_prime(a, p):
    a %= p
    if a == 0:
        return [0]
    if p == 2:
        return [a]
    if pow(a, (p - 1) // 2, p) != 1:
        return []
    if p % 4 == 3:
        r = pow(a, (p + 1) // 4, p)
    else:                                   # Tonelli-Shanks
        q, s = p - 1, 0
        while q % 2 == 0:
            q //= 2
            s += 1
        z = 2
        while pow(z, (p - 1) // 2, p) != p - 1:
            z += 1
        m, c, t, r = s, pow(z, q, p), pow(a, q, p), pow(a, (q + 1) // 2, p)
        while t != 1:
            i, tt = 0, t
            while tt != 1:
                tt = tt * tt % p
                i += 1
            b = pow(c, 1 << (m - i - 1), p)
            m, c = i, b * b % p
            t, r = t * c % p, r * b % p
    return sorted({r, p - r})


@op('numbers.sqrt_mod', kind='modular arithmetic', bound='p < 3.3e24 or a proven prime')
def sqrt_mod(a, p):
    """All x in [0, p) with x*x = a (mod p), p prime (Tonelli-Shanks); [] for a non-residue. sqrt_mod(10, 13) -> [6, 7]."""
    a, p = _int(a), _int(p)
    if p < 2 or not is_prime(p):
        raise BadInput('p must be prime')
    return _sqrt_prime(a, p)


@op('numbers.sqrt_mod_all', kind='modular arithmetic', bound='n factors; prime powers up to 200,000 by search, odd ones with p not dividing a by Hensel; at most 10**6 roots')
def sqrt_mod_all(a, n):
    """All x in [0, n) with x*x = a (mod n) for any n >= 1, sorted; [] when none. sqrt_mod_all(1, 8) -> [1, 3, 5, 7]."""
    a, n = _int(a), _int(n)
    _pos(n, 'modulus')
    a %= n
    if n == 1:
        return [0]
    roots, mod = [0], 1
    for p, e in sorted(_factor_dict(n).items()):
        q = p ** e
        if a % p and p != 2:
            r = _sqrt_prime(a, p)
            if not r:
                return []
            cur, pk = r[0], p
            for _ in range(e - 1):
                pk *= p
                cur = (cur - (cur * cur - a) * pow(2 * cur, -1, pk)) % pk
            rq = sorted({cur, q - cur})
        elif q <= 200_000:
            rq = [x for x in range(q) if (x * x - a) % q == 0]
            if not rq:
                return []
        else:
            raise NotCovered('prime power modulus too large for the search')
        if len(roots) * len(rq) > 10 ** 6:
            raise NotCovered('too many roots')
        new = []
        for r1 in roots:
            for r2 in rq:
                new.append(_crt2(r1, mod, r2, q)[0])
        roots, mod = new, mod * q
    return sorted(roots)


@op('numbers.discrete_log', kind='modular arithmetic', bound='n up to 10**12 (baby-step giant-step)')
def discrete_log(a, b, n):
    """The least x >= 0 with a**x = b (mod n), any n >= 1 (a need not be coprime to n); () when there is none. discrete_log(2, 8, 13) -> 3."""
    a, b, n = _int(a), _int(b), _int(n)
    _pos(n, 'modulus')
    _small(n, 10 ** 12, 'modulus')
    a %= n
    b %= n
    if n == 1:
        return 0
    cnt, t = 0, 1
    while True:
        if b == t % n:
            return cnt
        g = math.gcd(a, n)
        if g == 1:
            break
        if b % g:
            return ()
        b //= g
        n //= g
        t = t * (a // g) % n
        cnt += 1
        if n == 1:
            return cnt if b % n == t % n else ()
    m = math.isqrt(n) + 1
    table, cur = {}, b
    for j in range(m):
        table[cur] = j
        cur = cur * a % n
    am, cur = pow(a, m, n), t
    for i in range(1, m + 1):
        cur = cur * am % n
        if cur in table:
            return i * m - table[cur] + cnt
    return ()


@op('numbers.binomial_mod', kind='modular arithmetic', bound='m factors; a prime power at most 10**6, or a larger prime with each base-p digit of k (or n-k) at most 2,000,000; n any size')
def binomial_mod(n, k, m):
    """C(n, k) mod m for huge n, k (Granville's factorial-without-p method per prime power, then CRT). binomial_mod(10**18, 10**9, 1000003) -> a residue."""
    n, k, m = _int(n), _int(k), _int(m)
    _pos(m, 'modulus')
    _nonneg(n, 'n')
    if k < 0 or k > n:
        return 0
    if m == 1:
        return 0
    residues, mods = [], []
    for p, e in _factor_dict(m).items():
        pe = p ** e
        if e == 1 and pe > 10 ** 6:           # a large prime: Lucas' theorem, each digit pair by a short product
            r, nn, kk = 1, n, k
            while nn and kk and r:
                ni, ki = nn % p, kk % p
                nn //= p
                kk //= p
                if ki > ni:
                    r = 0
                    break
                ki = min(ki, ni - ki)
                _small(ki, 2_000_000, 'digit of k')
                num = den = 1
                for i in range(ki):
                    num = num * (ni - i) % p
                    den = den * (i + 1) % p
                r = r * num % p * pow(den, -1, p) % p
            residues.append(r)
            mods.append(pe)
            continue
        _small(pe, 10 ** 6, 'prime power of the modulus')
        tab = [1] * pe
        for i in range(1, pe):
            tab[i] = tab[i - 1] * (i if i % p else 1) % pe

        def f(x):
            res = 1
            while x:
                res = res * pow(tab[pe - 1], x // pe, pe) % pe * tab[x % pe] % pe
                x //= p
            return res

        def digsum(x):
            s = 0
            while x:
                s += x % p
                x //= p
            return s
        v = (digsum(k) + digsum(n - k) - digsum(n)) // (p - 1)    # the exponent of p in C(n, k)
        if v >= e:
            r = 0
        else:
            r = pow(p, v, pe) * f(n) % pe * pow(f(k) * f(n - k) % pe, -1, pe) % pe
        residues.append(r)
        mods.append(pe)
    return crt(residues, mods)[0]


# ----------------------------------------------------------------------------------------------------------------------------------------------------------
# primes

def _sieve(n):
    s = bytearray([1]) * (n + 1)
    s[0:2] = b'\x00\x00'[:min(2, n + 1)]
    for i in range(2, math.isqrt(n) + 1):
        if s[i]:
            s[i * i::i] = bytes(len(range(i * i, n + 1, i)))
    return s


_SMALL = [i for i, v in enumerate(_sieve(5000)) if v]
_SMALL_SET = frozenset(_SMALL)


def _mr(n, bases):
    d, s = n - 1, 0
    while d % 2 == 0:
        d //= 2
        s += 1
    for a in bases:
        a %= n
        if a == 0:
            continue
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(s - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def _lucas_strong(n):
    """Strong Lucas probable-prime test with Selfridge's parameters (n odd, not a square)."""
    if math.isqrt(n) ** 2 == n:
        return False
    D = 5
    while True:
        j = _jacobi(D, n)
        if j == -1:
            break
        if j == 0:
            return False                     # n is large, so a shared factor means composite
        D = -(D + 2) if D > 0 else -D + 2
    P, Q = 1, (1 - D) // 4
    if math.gcd(n, 2 * Q * D) != 1:
        return False
    d, s = n + 1, 0
    while d % 2 == 0:
        d //= 2
        s += 1
    U, V, Qk = 1, P, Q % n
    for bit in bin(d)[3:]:
        U, V, Qk = U * V % n, (V * V - 2 * Qk) % n, Qk * Qk % n
        if bit == '1':
            U, V = (P * U + V) % n, (D * U + P * V) % n
            if U & 1:
                U += n
            if V & 1:
                V += n
            U, V = U >> 1, V >> 1
            Qk = Qk * Q % n
    if U == 0 or V == 0:
        return True
    for _ in range(s - 1):
        V = (V * V - 2 * Qk) % n
        Qk = Qk * Qk % n
        if V == 0:
            return True
    return False


def _bpsw(n):
    """Baillie-PSW for n odd and above the trial-division range: a False is a proof of compositeness; a True is a probable prime."""
    return _mr(n, (2,)) and _lucas_strong(n)


def _is_prime_det(n):
    if n < 2:
        return False
    if n <= _SMALL[-1]:
        return n in _SMALL_SET
    for p in _SMALL[:25]:
        if n % p == 0:
            return False
    return _mr(n, _MR_BASES)


class _Unproven(Exception):
    pass


def _prime_status(c, depth):
    """True for a proven prime, False for a composite, None for a probable prime without proof."""
    if c < _MR_LIMIT:
        return _is_prime_det(c)
    if any(c % p == 0 for p in _SMALL[:100]) or not _bpsw(c):
        return False
    try:
        return _prove(c, depth)
    except _Unproven:
        return None


def _prove(n, depth):
    """Pocklington: n - 1 = F*R, F > sqrt(n) completely factored with proven primes and gcd(F, R) = 1, and for each prime q | F an a with a^(n-1) = 1 and
    gcd(a^((n-1)/q) - 1, n) = 1 proves n prime."""
    if depth > 4:
        raise _Unproven
    m = n - 1
    facs, F, r = {}, 1, m
    for p in _SMALL:
        if r % p == 0:
            while r % p == 0:
                r //= p
                facs[p] = facs.get(p, 0) + 1
                F *= p
    stack = [r]
    while F * F <= n and stack:
        c = stack.pop()
        if c == 1:
            continue
        st = _prime_status(c, depth + 1)
        if st is True:
            facs[c] = facs.get(c, 0) + 1
            F *= c
        elif st is False:
            d = _split(c)
            if d is not None:
                stack += [d, c // d]
    if F * F <= n or math.gcd(F, m // F) != 1:
        raise _Unproven
    for q in facs:
        for a in range(2, 80):
            if pow(a, m, n) == 1 and math.gcd(pow(a, m // q, n) - 1, n) == 1:
                break
        else:
            raise _Unproven
    return True


@op('numbers.is_prime', kind='primes', bound='deterministic Miller-Rabin below 3.3e24; above, composite is certain and a prime is proven by Pocklington or refused; n below 2**2048')
def is_prime(n):
    """True when n is prime. is_prime(2**61 - 1) -> True."""
    n = _int(n)
    if n < _MR_LIMIT:
        return _is_prime_det(n)
    _small(n.bit_length(), 2048, 'n')
    st = _prime_status(n, 0)
    if st is None:
        raise NotCovered('probable prime above 3.3e24 that could not be proven')
    return st


@op('numbers.next_prime', kind='primes', bound='at most 100,000 candidates')
def next_prime(n):
    """The least prime greater than n. next_prime(100) -> 101."""
    n = _int(n)
    if n < 2:
        return 2
    c = n + 1 if n % 2 == 0 else n + 2        # the next odd number above n
    for _ in range(100_000):
        if is_prime(c):
            return c
        c += 2
    raise NotCovered('no prime found within 100,000 candidates')


@op('numbers.prev_prime', kind='primes', bound='at most 100,000 candidates')
def prev_prime(n):
    """The greatest prime less than n; BadInput when n <= 2. prev_prime(100) -> 97."""
    n = _int(n)
    if n <= 2:
        raise BadInput('no prime below 2')
    if n == 3:
        return 2
    c = n - 1 if (n - 1) % 2 else n - 2
    for _ in range(100_000):
        if c < 2:
            return 2
        if is_prime(c):
            return c
        c -= 2
    raise NotCovered('no prime found within 100,000 candidates')


@op('numbers.primes_upto', kind='primes', bound='n up to 10**7')
def primes_upto(n):
    """The primes <= n in order (sieve of Eratosthenes). primes_upto(20) -> [2, 3, 5, 7, 11, 13, 17, 19]."""
    n = _int(n)
    _small(n, 10 ** 7, 'n')
    if n < 2:
        return []
    return [i for i, v in enumerate(_sieve(n)) if v]


@op('numbers.nth_prime', kind='primes', bound='k up to 600,000')
def nth_prime(k):
    """The k-th prime, k >= 1. nth_prime(1000) -> 7919."""
    k = _int(k)
    _pos(k, 'k')
    _small(k, 600_000, 'k')
    if k < 6:
        return (2, 3, 5, 7, 11)[k - 1]
    top = int(k * (math.log(k) + math.log(math.log(k)))) + 10
    s = _sieve(top)
    cnt = 0
    for i, v in enumerate(s):
        if v:
            cnt += 1
            if cnt == k:
                return i
    raise NotCovered('sieve bound too small')


@op('numbers.prime_pi', kind='primes', bound='n up to 10**9 (Lucy-Hedgehog counting)')
def prime_pi(n):
    """The number of primes <= n. prime_pi(100) -> 25."""
    n = _int(n)
    _small(n, 10 ** 9, 'n')
    if n < 2:
        return 0
    r = math.isqrt(n)
    small = [i - 1 for i in range(r + 1)]          # small[i] = count of integers in 2..i
    large = [0] + [n // i - 1 for i in range(1, r + 1)]   # large[i] = count of integers in 2..n//i
    for p in range(2, r + 1):
        if small[p] == small[p - 1]:
            continue
        pc = small[p - 1]
        p2 = p * p
        for i in range(1, min(r, n // p2) + 1):
            d = i * p
            large[i] -= (large[d] if d <= r else small[n // d]) - pc
        for i in range(r, p2 - 1, -1):
            small[i] -= small[i // p] - pc
    return large[1]


# ----------------------------------------------------------------------------------------------------------------------------------------------------------
# factoring

def _rho(n, c, steps):
    """Brent's variant of Pollard rho: a divisor of n in 1 < d < n, or None within `steps` iterations."""
    y, r, q, g, m = 2, 1, 1, 1, 128
    x = ys = 0
    used = 0
    while g == 1:
        x = y
        for _ in range(r):
            y = (y * y + c) % n
        k = 0
        while k < r and g == 1:
            ys = y
            for _ in range(min(m, r - k)):
                y = (y * y + c) % n
                q = q * abs(x - y) % n
            g = math.gcd(q, n)
            k += m
        used += 2 * r
        r *= 2
        if g == 1 and used > steps:
            return None
    if g == n:
        g = 1
        while g == 1:
            ys = (ys * ys + c) % n
            g = math.gcd(abs(x - ys), n)
    return g if g != n else None


def _pm1(n, bound=20000):
    """Pollard p-1, stage 1 only."""
    a = 2
    for i, p in enumerate(_SMALL):
        if p > bound:
            break
        pe = p
        while pe * p <= bound:
            pe *= p
        a = pow(a, pe, n)
        if i % 40 == 39:
            g = math.gcd(a - 1, n)
            if 1 < g < n:
                return g
            if g == n:
                return None
    g = math.gcd(a - 1, n)
    return g if 1 < g < n else None


def _split(n):
    """A proper divisor of the composite n (odd, no factor below 5000, not a perfect power), or None."""
    d = _pm1(n)
    if d:
        return d
    for c in (1, 3):
        d = _rho(n, c, _RHO_STEPS)
        if d:
            return d
    return None


def _factor_all(n):
    """(factors, leftover): the proven prime factors of n >= 1 with exponents, and the list of cofactors this module could not split or prove prime."""
    out, left = {}, []
    for p in _SMALL:
        if p * p > n:
            break
        if n % p == 0:
            e = 0
            while n % p == 0:
                n //= p
                e += 1
            out[p] = out.get(p, 0) + e
    if n == 1:
        return out, left
    stack = [n]
    while stack:
        c = stack.pop()
        if c == 1:
            continue
        if c <= _SMALL[-1] ** 2:
            out[c] = out.get(c, 0) + 1          # no factor below 5000, so a prime
            continue
        st = _prime_status(c, 0)
        if st is True:
            out[c] = out.get(c, 0) + 1
            continue
        if st is None:
            left.append(c)
            continue
        b, e = _perfect_power(c)
        if e > 1:
            stack += [b] * e
            continue
        d = _split(c)
        if d is None:
            left.append(c)
            continue
        stack += [d, c // d]
    return out, left


def _factor_dict(n):
    _pos(n, 'n')
    _small(n.bit_length(), _MAX_BITS_FACTOR, 'n')
    out, left = _factor_all(n)
    if left:
        raise NotCovered('could not factor completely: cofactor ' + str(min(left)) + ' left')
    return out


@op('numbers.factorint', kind='primes', bound='n < 2**240; trial division to 5000, Pollard p-1 (B=20000), Brent rho 2 x 250,000 steps; else refused')
def factorint(n):
    """The prime factorisation of n >= 1 as {prime: exponent}. factorint(360) -> {2: 3, 3: 2, 5: 1}."""
    n = _int(n)
    return dict(sorted(_factor_dict(n).items()))


@op('numbers.factorint_partial', kind='primes', bound='as factorint; never refuses for lack of a factor')
def factorint_partial(n):
    """(factors, leftover): the proven prime factors found and the list of cofactors left unsplit (empty when complete). factorint_partial(12) -> ({2: 2, 3: 1}, [])."""
    n = _int(n)
    _pos(n, 'n')
    _small(n.bit_length(), _MAX_BITS_FACTOR, 'n')
    out, left = _factor_all(n)
    return dict(sorted(out.items())), sorted(left)


@op('numbers.is_squarefree', kind='primes', bound='as factorint')
def is_squarefree(n):
    """True when no square greater than 1 divides n >= 1. is_squarefree(30) -> True."""
    return all(e == 1 for e in _factor_dict(_int(n)).values())


@op('numbers.smallest_prime_factor', kind='primes', bound='as factorint')
def smallest_prime_factor(n):
    """The least prime dividing n >= 2. smallest_prime_factor(91) -> 7."""
    n = _int(n)
    if n < 2:
        raise BadInput('n must be at least 2')
    return min(_factor_dict(n))


@op('numbers.radical', kind='primes', bound='as factorint')
def radical(n):
    """The product of the distinct primes dividing n >= 1. radical(360) -> 30."""
    out = 1
    for p in _factor_dict(_int(n)):
        out *= p
    return out


# ----------------------------------------------------------------------------------------------------------------------------------------------------------
# number-theoretic functions

@op('numbers.divisors', kind='number theory', bound='as factorint; at most 10**6 divisors')
def divisors(n):
    """All positive divisors of n >= 1 in increasing order. divisors(12) -> [1, 2, 3, 4, 6, 12]."""
    f = _factor_dict(_int(n))
    count = 1
    for e in f.values():
        count *= e + 1
    _small(count, 10 ** 6, 'divisor count')
    ds = [1]
    for p, e in f.items():
        ds = [d * p ** k for d in ds for k in range(e + 1)]
    return sorted(ds)


@op('numbers.num_divisors', kind='number theory', bound='as factorint')
def num_divisors(n):
    """The number of positive divisors of n >= 1. num_divisors(360) -> 24."""
    out = 1
    for e in _factor_dict(_int(n)).values():
        out *= e + 1
    return out


@op('numbers.sigma', kind='number theory', bound='as factorint; k in 0..1000')
def sigma(n, k=1):
    """The sum of the k-th powers of the divisors of n >= 1 (k = 1 the sum of divisors). sigma(28) -> 56; sigma(6, 2) -> 50."""
    n, k = _int(n), _int(k, 'k')
    _nonneg(k, 'k')
    _small(k, 1000, 'k')
    out = 1
    for p, e in _factor_dict(n).items():
        out *= (e + 1) if k == 0 else (p ** (k * (e + 1)) - 1) // (p ** k - 1)
    return out


@op('numbers.totient', kind='number theory', bound='as factorint')
def totient(n):
    """Euler's phi: the count of 1 <= k <= n coprime to n >= 1. totient(36) -> 12."""
    out = 1
    for p, e in _factor_dict(_int(n)).items():
        out *= (p - 1) * p ** (e - 1)
    return out


@op('numbers.moebius', kind='number theory', bound='as factorint')
def moebius(n):
    """The Moebius function of n >= 1: 0 if a square divides n, else (-1)**(number of primes). moebius(30) -> -1."""
    f = _factor_dict(_int(n))
    if any(e > 1 for e in f.values()):
        return 0
    return -1 if len(f) % 2 else 1


def _carmichael_of(f):
    out = 1
    for p, e in f.items():
        if p == 2:
            lam = 1 if e == 1 else 2 if e == 2 else 2 ** (e - 2)
        else:
            lam = (p - 1) * p ** (e - 1)
        out = out * lam // math.gcd(out, lam)
    return out


@op('numbers.carmichael', kind='number theory', bound='as factorint')
def carmichael(n):
    """Carmichael's lambda: the exponent of the group of units mod n >= 1. carmichael(15) -> 4; carmichael(8) -> 2."""
    return _carmichael_of(_factor_dict(_int(n)))


@op('numbers.order', kind='number theory', bound='as factorint, applied to n and to p-1 of its primes')
def order(a, n):
    """The multiplicative order of a mod n >= 1, or () when gcd(a, n) > 1. order(2, 7) -> 3."""
    a, n = _int(a), _int(n)
    _pos(n, 'modulus')
    if n == 1:
        return 1
    if math.gcd(a, n) != 1:
        return ()
    f = _factor_dict(n)
    lam = _carmichael_of(f)
    primes = set()
    for p in f:
        primes.add(p)
        if p > 2:
            primes.update(_factor_dict(p - 1))
    primes.add(2)
    for q in sorted(primes):
        while lam % q == 0 and pow(a, lam // q, n) == 1:
            lam //= q
    return lam


@op('numbers.primitive_root', kind='number theory', bound='as factorint')
def primitive_root(n):
    """The least primitive root mod n >= 1, or () when the group of units is not cyclic. primitive_root(23) -> 5; primitive_root(8) -> ()."""
    n = _int(n)
    _pos(n, 'modulus')
    if n <= 2:
        return n - 1
    f = _factor_dict(n)
    odd = [p for p in f if p != 2]
    ok = (len(f) == 1 and 2 not in f) or (len(f) == 1 and f.get(2) == 2) or (len(f) == 2 and f.get(2) == 1 and len(odd) == 1)
    if not ok:
        return ()
    phi = 1
    for p, e in f.items():
        phi *= (p - 1) * p ** (e - 1)
    qs = list(_factor_dict(phi))
    g = 2
    while True:
        if math.gcd(g, n) == 1 and all(pow(g, phi // q, n) != 1 for q in qs):
            return g
        g += 1


@op('numbers.valuation', kind='number theory', bound='any size')
def valuation(n, p):
    """The exponent of the integer p >= 2 in n != 0 (the largest e with p**e dividing n). valuation(360, 2) -> 3."""
    n, p = _int(n), _int(p)
    if n == 0:
        raise BadInput('valuation of 0')
    if p < 2:
        raise BadInput('p must be at least 2')
    e = 0
    while n % p == 0:
        n //= p
        e += 1
    return e


@op('numbers.factorial_valuation', kind='number theory', bound='any size')
def factorial_valuation(n, p):
    """The exponent of the prime p in n! (Legendre's formula). factorial_valuation(100, 5) -> 24."""
    n, p = _int(n), _int(p)
    _nonneg(n, 'n')
    if p < 2 or not is_prime(p):
        raise BadInput('p must be prime')
    e = 0
    while n:
        n //= p
        e += n
    return e


@op('numbers.factorial_trailing_zeros', kind='number theory', bound='base factors as factorint')
def factorial_trailing_zeros(n, base=10):
    """The number of trailing zeros of n! written in the given base. factorial_trailing_zeros(100) -> 24."""
    n, base = _int(n), _int(base)
    _nonneg(n, 'n')
    if base < 2:
        raise BadInput('base must be at least 2')
    return min(factorial_valuation(n, p) // e for p, e in _factor_dict(base).items())


# ----------------------------------------------------------------------------------------------------------------------------------------------------------
# combinatorial numbers

@op('numbers.factorial', kind='combinatorial numbers', bound='n up to 100,000')
def factorial(n):
    """n!. factorial(10) -> 3628800."""
    n = _int(n)
    _nonneg(n, 'n')
    _small(n, 100_000, 'n')
    return math.factorial(n)


@op('numbers.double_factorial', kind='combinatorial numbers', bound='n up to 100,000')
def double_factorial(n):
    """n!! = n (n-2) (n-4) ... ending at 1 or 2; (-1)!! = 0!! = 1. double_factorial(7) -> 105."""
    n = _int(n)
    if n < -1:
        raise BadInput('n must be at least -1')
    _small(n, 100_000, 'n')
    out = 1
    for i in range(n, 1, -2):
        out *= i
    return out


@op('numbers.binomial', kind='combinatorial numbers', bound='n (or k - n - 1 for negative n) up to 10**6')
def binomial(n, k):
    """C(n, k) for any integer n (negative n by the generalised definition) and integer k (0 for k < 0). binomial(52, 5) -> 2598960; binomial(-3, 2) -> 6."""
    n, k = _int(n), _int(k)
    if k < 0:
        return 0
    if n >= 0:
        _small(n, 10 ** 6, 'n')
        return math.comb(n, k) if k <= n else 0
    _small(k - n - 1, 10 ** 6, 'k - n - 1')
    return (-1) ** k * math.comb(k - n - 1, k)


@op('numbers.multinomial', kind='combinatorial numbers', bound='sum of parts up to 100,000')
def multinomial(*ks):
    """(k1 + k2 + ...)! / (k1! k2! ...). multinomial(2, 3, 1) -> 60."""
    ks = [_int(k) for k in ks]
    if not ks:
        raise BadInput('no parts')
    for k in ks:
        _nonneg(k, 'part')
    _small(sum(ks), 100_000, 'sum of parts')
    out, tot = 1, 0
    for k in ks:
        tot += k
        out *= math.comb(tot, k)
    return out


def _rf(x, n, step):
    out = Fraction(1)
    for i in range(n):
        out *= x + step * i
    return out


@op('numbers.rising_factorial', kind='combinatorial numbers', bound='n up to 100,000')
def rising_factorial(x, n):
    """x (x+1) ... (x+n-1) for a rational x and integer n >= 0. rising_factorial(3, 4) -> 360."""
    x, n = _rat(x), _int(n)
    _nonneg(n, 'n')
    _small(n, 100_000, 'n')
    return _rf(x, n, 1)


@op('numbers.falling_factorial', kind='combinatorial numbers', bound='n up to 100,000')
def falling_factorial(x, n):
    """x (x-1) ... (x-n+1) for a rational x and integer n >= 0. falling_factorial(10, 3) -> 720."""
    x, n = _rat(x), _int(n)
    _nonneg(n, 'n')
    _small(n, 100_000, 'n')
    return _rf(x, n, -1)


def _fib_pair(n, m):
    a, b = 0, 1
    for bit in bin(n)[2:]:
        c = a * (2 * b - a)
        d = a * a + b * b
        if m is not None:
            c %= m
            d %= m
        if bit == '1':
            a, b = d, c + d
            if m is not None:
                b %= m
        else:
            a, b = c, d
    return a, b


def _fib_args(n, mod):
    n = _int(n)
    if mod is not None:
        mod = _int(mod, 'modulus')
        _pos(mod, 'modulus')
        _small(abs(n).bit_length(), 100_000, 'n')
    else:
        _small(abs(n), 2_000_000, 'n')
    return n, mod


@op('numbers.fibonacci', kind='combinatorial numbers', bound='|n| up to 2,000,000 exactly; any n below 2**100000 with a modulus')
def fibonacci(n, mod=None):
    """F(n) with F(0)=0, F(1)=1 (negative n by F(-n) = (-1)^(n+1) F(n)), optionally mod m by fast doubling. fibonacci(10) -> 55."""
    n, mod = _fib_args(n, mod)
    a = _fib_pair(abs(n), mod)[0]
    if n < 0 and abs(n) % 2 == 0:
        a = -a
    return a % mod if mod is not None else a


@op('numbers.lucas', kind='combinatorial numbers', bound='|n| up to 2,000,000 exactly; any n below 2**100000 with a modulus')
def lucas(n, mod=None):
    """L(n) with L(0)=2, L(1)=1 (L(-n) = (-1)^n L(n)), optionally mod m. lucas(10) -> 123."""
    n, mod = _fib_args(n, mod)
    a, b = _fib_pair(abs(n), mod)
    v = 2 * b - a
    if n < 0 and abs(n) % 2 == 1:
        v = -v
    return v % mod if mod is not None else v


@op('numbers.catalan', kind='combinatorial numbers', bound='n up to 50,000')
def catalan(n):
    """The n-th Catalan number C(2n, n) / (n + 1). catalan(5) -> 42."""
    n = _int(n)
    _nonneg(n, 'n')
    _small(n, 50_000, 'n')
    return math.comb(2 * n, n) // (n + 1)


@op('numbers.bell', kind='combinatorial numbers', bound='n up to 1500')
def bell(n):
    """The n-th Bell number (set partitions of n elements, Bell triangle). bell(5) -> 52."""
    n = _int(n)
    _nonneg(n, 'n')
    _small(n, 1500, 'n')
    row = [1]
    for _ in range(n):
        nxt = [row[-1]]
        for x in row:
            nxt.append(nxt[-1] + x)
        row = nxt
    return row[0]


@op('numbers.stirling1', kind='combinatorial numbers', bound='n up to 1500')
def stirling1(n, k, signed=False):
    """Stirling number of the first kind: unsigned c(n, k), the permutations of n with k cycles (signed=True gives s(n, k) = (-1)^(n-k) c(n, k)).
    stirling1(4, 2) -> 11."""
    n, k = _int(n), _int(k)
    _nonneg(n, 'n')
    _nonneg(k, 'k')
    _small(n, 1500, 'n')
    if k > n:
        return 0
    poly = [1]
    for i in range(n):
        new = [0] * (len(poly) + 1)
        for j, c in enumerate(poly):
            new[j + 1] += c
            new[j] += i * c
        poly = new
    c = poly[k]
    return -c if signed and (n - k) % 2 else c


@op('numbers.stirling2', kind='combinatorial numbers', bound='n, k up to 3000')
def stirling2(n, k):
    """Stirling number of the second kind: the partitions of an n-set into k non-empty blocks. stirling2(5, 3) -> 25."""
    n, k = _int(n), _int(k)
    _nonneg(n, 'n')
    _nonneg(k, 'k')
    _small(n, 3000, 'n')
    _small(k, 3000, 'k')
    if k > n:
        return 0
    if n == 0:
        return 1
    if k == 0:
        return 0
    total, c = 0, 1                      # sum over j of (-1)^(k-j) C(k, j) j^n, then / k!
    for j in range(k + 1):
        if j:
            c = c * (k - j + 1) // j
        total += (-1) ** (k - j) * c * j ** n
    return total // math.factorial(k)


@op('numbers.partition_number', kind='combinatorial numbers', bound='n up to 10,000')
def partition_number(n):
    """p(n), the number of integer partitions of n (Euler's pentagonal recurrence). partition_number(10) -> 42."""
    n = _int(n)
    _nonneg(n, 'n')
    _small(n, 10_000, 'n')
    p = [1] + [0] * n
    for m in range(1, n + 1):
        s, k = 0, 1
        while True:
            g1 = k * (3 * k - 1) // 2
            if g1 > m:
                break
            sign = 1 if k % 2 else -1
            s += sign * p[m - g1]
            g2 = k * (3 * k + 1) // 2
            if g2 <= m:
                s += sign * p[m - g2]
            k += 1
        p[m] = s
    return p[n]


# ----------------------------------------------------------------------------------------------------------------------------------------------------------
# digits and bases

_DIGITS = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'


def _digits_of(n, b):
    """Most significant first; n >= 0, b >= 2."""
    if n == 0:
        return [0]
    k = max(1, 60 // b.bit_length())
    chunk = b ** k
    out = []
    while n:
        n, r = divmod(n, chunk)
        piece = []
        for _ in range(k):
            r, d = divmod(r, b)
            piece.append(d)
            if r == 0 and n == 0:
                break
        out.extend(piece)
    while len(out) > 1 and out[-1] == 0:
        out.pop()
    out.reverse()
    return out


def _digit_args(n, base):
    n, base = _int(n), _int(base, 'base')
    _nonneg(n, 'n')
    if base < 2:
        raise BadInput('base must be at least 2')
    _small(n.bit_length(), 200_000, 'n')
    return n, base


@op('numbers.digits', kind='digits and bases', bound='n up to 200,000 bits')
def digits(n, base=10):
    """The digits of n >= 0 in the base (>= 2), most significant first. digits(255, 16) -> [15, 15]."""
    n, base = _digit_args(n, base)
    return _digits_of(n, base)


@op('numbers.digit_sum', kind='digits and bases', bound='n up to 200,000 bits')
def digit_sum(n, base=10):
    """The sum of the digits of n >= 0 in the base. digit_sum(2**100) -> 115."""
    n, base = _digit_args(n, base)
    return sum(_digits_of(n, base))


@op('numbers.num_digits', kind='digits and bases', bound='n up to 200,000 bits')
def num_digits(n, base=10):
    """The number of digits of n >= 0 in the base (0 has one). num_digits(2**100) -> 31."""
    n, base = _digit_args(n, base)
    return len(_digits_of(n, base))


@op('numbers.from_digits', kind='digits and bases', bound='at most 200,000 digits')
def from_digits(ds, base=10):
    """The number whose digits (most significant first) are ds in the base; every digit must satisfy 0 <= d < base. from_digits([1, 0, 1], 2) -> 5."""
    base = _int(base, 'base')
    if base < 2:
        raise BadInput('base must be at least 2')
    ds = [_int(d, 'digit') for d in ds]
    _small(len(ds), 200_000, 'digit count')
    n = 0
    for d in ds:
        if not 0 <= d < base:
            raise BadInput('a digit is outside 0..base-1')
        n = n * base + d
    return n


@op('numbers.base_convert', kind='digits and bases', bound='bases 2..36, at most 200,000 bits')
def base_convert(s, from_base, to_base):
    """Convert the digit string s (0-9, A-Z, either case, optional leading '-') between bases 2..36; the result uses upper case. base_convert('ff', 16, 2) -> '11111111'."""
    from_base, to_base = _int(from_base, 'from_base'), _int(to_base, 'to_base')
    if not (2 <= from_base <= 36 and 2 <= to_base <= 36):
        raise BadInput('bases must be in 2..36')
    if not isinstance(s, str):
        raise BadInput('s must be a string')
    t = s.strip()
    neg = t.startswith('-')
    if neg:
        t = t[1:]
    if not t:
        raise BadInput('no digits')
    n = 0
    _small(len(t), 200_000, 'digit count')
    for ch in t.upper():
        v = _DIGITS.find(ch)
        if v < 0 or v >= from_base:
            raise BadInput(f'digit {ch!r} is not valid in base {from_base}')
        n = n * from_base + v
    out = ''.join(_DIGITS[d] for d in _digits_of(n, to_base))
    return '-' + out if neg and n else out


@op('numbers.digital_root', kind='digits and bases', bound='n up to 200,000 bits')
def digital_root(n, base=10):
    """Repeated digit sum until one digit is left. digital_root(987654321) -> 9."""
    n, base = _digit_args(n, base)
    return 0 if n == 0 else 1 + (n - 1) % (base - 1)


# ----------------------------------------------------------------------------------------------------------------------------------------------------------
# floor, ceiling, rounding, modulo of rationals

@op('numbers.rat_floor', kind='exact arithmetic', bound='any size')
def rat_floor(x):
    """The greatest integer <= x. rat_floor("-7/2") -> -4."""
    return math.floor(_rat(x))


@op('numbers.rat_ceil', kind='exact arithmetic', bound='any size')
def rat_ceil(x):
    """The least integer >= x. rat_ceil("-7/2") -> -3."""
    return math.ceil(_rat(x))


@op('numbers.rat_trunc', kind='exact arithmetic', bound='any size')
def rat_trunc(x):
    """x rounded toward zero. rat_trunc("-7/2") -> -3."""
    return math.trunc(_rat(x))


@op('numbers.rat_round', kind='exact arithmetic', bound='any size')
def rat_round(x, mode='half_away'):
    """x rounded to an integer; mode 'half_away' (halves go away from zero, the school rule), 'half_even' or 'half_up' (halves go up). rat_round("5/2") -> 3."""
    x = _rat(x)
    fl = math.floor(x)
    d = x - fl
    if d < Fraction(1, 2):
        return fl
    if d > Fraction(1, 2):
        return fl + 1
    if mode == 'half_away':
        return fl + 1 if x > 0 else fl
    if mode == 'half_up':
        return fl + 1
    if mode == 'half_even':
        return fl if fl % 2 == 0 else fl + 1
    raise BadInput("mode must be 'half_away', 'half_up' or 'half_even'")


@op('numbers.rat_mod', kind='exact arithmetic', bound='any size')
def rat_mod(a, b):
    """a - b*floor(a/b) for rationals, b != 0: the result has the sign of b. rat_mod("7/2", 1) -> 1/2; rat_mod(-7, 3) -> 2."""
    a, b = _rat(a), _rat(b)
    if b == 0:
        raise BadInput('modulo zero')
    return a - b * math.floor(a / b)


@op('numbers.frac_part', kind='exact arithmetic', bound='any size')
def frac_part(x):
    """x - floor(x), in [0, 1). frac_part("-7/2") -> 1/2."""
    x = _rat(x)
    return x - math.floor(x)


# ----------------------------------------------------------------------------------------------------------------------------------------------------------
# continued fractions

def _cf(x):
    a, b = x.numerator, x.denominator
    out = []
    while b:
        q = a // b
        out.append(q)
        a, b = b, a - q * b
    return out


def _cf_value(terms):
    h0, h1, k0, k1 = 1, 0, 0, 1
    for a in terms:
        h0, h1 = a * h0 + h1, h0
        k0, k1 = a * k0 + k1, k0
    return Fraction(h0, k0)


@op('numbers.cf_rational', kind='continued fractions', bound='numerator and denominator up to 2,000,000 bits')
def cf_rational(x):
    """The simple continued fraction [a0; a1, ...] of a rational (a0 = floor, last term > 1 unless it is the only one). cf_rational("415/93") -> [4, 2, 6, 7]."""
    x = _rat(x)
    _small(_bits(x), _BITS, 'x')
    return _cf(x)


@op('numbers.cf_value', kind='continued fractions', bound='at most 100,000 terms')
def cf_value(terms):
    """The rational [a0; a1, ..., an] (a0 any integer, the others positive). cf_value([4, 2, 6, 7]) -> 415/93."""
    terms = [_int(t) for t in terms]
    if not terms or any(t < 1 for t in terms[1:]):
        raise BadInput('terms must be a non-empty list with positive terms after the first')
    _small(len(terms), 100_000, 'term count')
    return _cf_value(terms)


@op('numbers.cf_convergents', kind='continued fractions', bound='at most 100,000 terms')
def cf_convergents(x):
    """The convergents of a rational or of a list of terms [a0, a1, ...]. cf_convergents("415/93") -> [4, 9/2, 58/13, 415/93]."""
    terms = [_int(t) for t in x] if isinstance(x, (list, tuple)) else _cf(_rat(x))
    if not terms or any(t < 1 for t in terms[1:]):
        raise BadInput('terms must be a non-empty list with positive terms after the first')
    _small(len(terms), 100_000, 'term count')
    out = []
    h0, h1, k0, k1 = 1, 0, 0, 1
    for a in terms:
        h0, h1 = a * h0 + h1, h0
        k0, k1 = a * k0 + k1, k0
        out.append(Fraction(h0, k0))
    return out


def _cf_sqrt(d):
    a0 = math.isqrt(d)
    if a0 * a0 == d:
        return a0, []
    m, dd, a, period = 0, 1, a0, []
    while True:
        m = dd * a - m
        dd = (d - m * m) // dd
        a = (a0 + m) // dd
        period.append(a)
        if a == 2 * a0:
            return a0, period
        if len(period) > 2_000_000:
            raise NotCovered('period too long')


@op('numbers.cf_sqrt', kind='continued fractions', bound='d up to 10**18, period up to 2,000,000')
def cf_sqrt(d):
    """(a0, period): sqrt(d) = [a0; period repeated] for a non-square d >= 1; the period is [] when d is a square. cf_sqrt(23) -> (4, [1, 3, 1, 8])."""
    d = _int(d)
    _pos(d, 'd')
    _small(d, 10 ** 18, 'd')
    return _cf_sqrt(d)


@op('numbers.cf_sqrt_convergents', kind='continued fractions', bound='d up to 10**18, count up to 100,000')
def cf_sqrt_convergents(d, count):
    """The first `count` convergents of sqrt(d) for a non-square d. cf_sqrt_convergents(2, 4) -> [1, 3/2, 7/5, 17/12]."""
    d, count = _int(d), _int(count)
    _pos(d, 'd')
    _pos(count, 'count')
    _small(d, 10 ** 18, 'd')
    _small(count, 100_000, 'count')
    a0, period = _cf_sqrt(d)
    if not period:
        raise BadInput('d is a perfect square')
    out, h0, h1, k0, k1 = [], 1, 0, 0, 1
    for i in range(count):
        a = a0 if i == 0 else period[(i - 1) % len(period)]
        h0, h1 = a * h0 + h1, h0
        k0, k1 = a * k0 + k1, k0
        out.append(Fraction(h0, k0))
    return out


@op('numbers.best_approx', kind='continued fractions', bound='any size')
def best_approx(x, max_den):
    """The rational closest to x with denominator at most max_den (from the convergents and semiconvergents). best_approx("3141592653/1000000000", 100) -> 311/99."""
    x = _rat(x)
    max_den = _int(max_den)
    _pos(max_den, 'max_den')
    return x.limit_denominator(max_den)


# ----------------------------------------------------------------------------------------------------------------------------------------------------------
# decimal expansion of a rational

def _dstr(k):
    """str(k) for k >= 0 without the interpreter's limit on digits of an int-to-str conversion."""
    return ''.join(map(str, _digits_of(k, 10)))


def _split_decimal(den):
    """(a, b, rest): den = 2**a * 5**b * rest."""
    a = b = 0
    while den % 2 == 0:
        den //= 2
        a += 1
    while den % 5 == 0:
        den //= 5
        b += 1
    return a, b, den


@op('numbers.decimal_expansion', kind='decimal expansion', bound='at most 100,000 digits')
def decimal_expansion(x, digits=20):
    """x written in decimal with exactly `digits` digits after the point, cut (not rounded) toward zero. decimal_expansion("1/7", 12) -> '0.142857142857'."""
    x = _rat(x)
    digits = _int(digits)
    _nonneg(digits, 'digits')
    _small(digits, 100_000, 'digits')
    neg = x < 0
    x = abs(x)
    scaled = x.numerator * 10 ** digits // x.denominator
    ip, fp = divmod(scaled, 10 ** digits)
    s = _dstr(ip)
    if digits:
        s += '.' + _dstr(fp).rjust(digits, '0')
    return '-' + s if neg and scaled else s


@op('numbers.decimal_digit', kind='decimal expansion', bound='position up to 10**30')
def decimal_digit(x, k):
    """The k-th digit after the decimal point (k >= 1) of |x|, found by modular exponentiation without writing the expansion. decimal_digit("1/7", 2024) -> 4."""
    x = abs(_rat(x))
    k = _int(k)
    _pos(k, 'k')
    _small(k, 10 ** 30, 'k')
    rem = x.numerator * pow(10, k - 1, x.denominator) % x.denominator    # the fractional part, scaled, has remainder rem / den
    return (10 * rem) // x.denominator


@op('numbers.decimal_period', kind='decimal expansion', bound='denominator factors as factorint')
def decimal_period(x):
    """The length of the repeating block of the decimal expansion of x (0 when it terminates). decimal_period("1/7") -> 6; decimal_period("1/6") -> 1."""
    x = _rat(x)
    _, _, rest = _split_decimal(x.denominator)
    if rest == 1:
        return 0
    return order(10, rest)


@op('numbers.decimal_preperiod', kind='decimal expansion', bound='any size')
def decimal_preperiod(x):
    """The number of digits after the point before the repeating block starts. decimal_preperiod("1/6") -> 1."""
    x = _rat(x)
    a, b, _ = _split_decimal(x.denominator)
    return max(a, b)


@op('numbers.repeating_decimal', kind='decimal expansion', bound='period up to 1,000,000 digits')
def repeating_decimal(x):
    """(integer_part, pre, rep) for x >= 0: x = integer_part . pre (rep repeated); rep is '' for a terminating decimal. repeating_decimal("1/6") -> (0, '1', '6')."""
    x = _rat(x)
    if x < 0:
        raise BadInput('x must not be negative')
    ip = math.floor(x)
    f = x - ip
    num, den = f.numerator, f.denominator
    pre_len = decimal_preperiod(f)
    per = decimal_period(f)
    _small(per, 1_000_000, 'period')
    _small(pre_len, 1_000_000, 'pre-period')
    total = pre_len + per
    scaled = num * 10 ** total // den if den > 1 else 0

    def block(k, width):
        return _dstr(k).rjust(width, '0') if width else ''
    pre_val = num * 10 ** pre_len // den if den > 1 else 0
    pre = block(pre_val, pre_len)
    rep = block(scaled - pre_val * 10 ** per, per) if per else ''
    return (ip, pre, rep)


@op('numbers.repeating_to_fraction', kind='decimal expansion', bound='at most 100,000 digits')
def repeating_to_fraction(integer_part, pre, rep):
    """The fraction equal to integer_part . pre (rep repeated), digit strings pre and rep. repeating_to_fraction(0, '1', '6') -> 1/6; (0, '', '9') -> 1."""
    ip = _int(integer_part)
    for s in (pre, rep):
        if not isinstance(s, str) or (s and not s.isdigit()):
            raise BadInput('pre and rep must be digit strings')
    _small(len(pre) + len(rep), 100_000, 'digit count')
    p = int(pre) if pre else 0
    base = Fraction(p, 10 ** len(pre))
    if rep:
        base += Fraction(int(rep), (10 ** len(rep) - 1) * 10 ** len(pre))
    return abs(ip) + base if ip >= 0 else ip - base


# ----------------------------------------------------------------------------------------------------------------------------------------------------------
# Diophantine equations

@op('numbers.lin_diophantine', kind='Diophantine', bound='any size')
def lin_diophantine(a, b, c):
    """All integer solutions of a*x + b*y = c as (x0, y0, dx, dy): x = x0 + dx*t, y = y0 + dy*t for every integer t, with dx = b/g, dy = -a/g, g = gcd(a, b) and
    x0 the least non-negative value (when b != 0); () when there is none. lin_diophantine(3, 5, 7) -> (4, -1, 5, -3). a = b = 0 with c = 0 is NotCovered
    (every pair)."""
    a, b, c = _int(a), _int(b), _int(c)
    if a == 0 and b == 0:
        if c == 0:
            raise NotCovered('every pair of integers is a solution')
        return ()
    g, x, y = _xgcd(a, b)
    if c % g:
        return ()
    k = c // g
    x0, y0 = x * k, y * k
    dx, dy = b // g, -(a // g)
    if dx:
        x0 %= abs(dx)
        y0 = (c - a * x0) // b
    else:
        y0 %= abs(dy)
        x0 = (c - b * y0) // a
    return (x0, y0, dx, dy)


@op('numbers.lin_diophantine_nonneg_count', kind='Diophantine', bound='any size')
def lin_diophantine_nonneg_count(a, b, c):
    """The number of solutions of a*x + b*y = c in non-negative integers, a, b > 0, c >= 0. lin_diophantine_nonneg_count(3, 5, 100) -> 7."""
    a, b, c = _int(a), _int(b), _int(c)
    _pos(a, 'a')
    _pos(b, 'b')
    _nonneg(c, 'c')
    sol = lin_diophantine(a, b, c)
    if sol == ():
        return 0
    x0, _, dx, _ = sol
    top = c // a
    return 0 if x0 > top else (top - x0) // dx + 1


@op('numbers.pell', kind='Diophantine', bound='d up to 10**18; continued-fraction period up to 20,000')
def pell(d):
    """The fundamental solution (x, y), y > 0, of x**2 - d*y**2 = 1 for a non-square d >= 2; () for a square d (only the trivial solution). pell(61) -> (1766319049, 226153980)."""
    d = _int(d)
    _pos(d, 'd')
    _small(d, 10 ** 18, 'd')
    a0, period = _cf_sqrt(d)
    if not period:
        return ()
    _small(len(period), 20_000, 'period')
    h0, h1, k0, k1 = a0, 1, 1, 0
    i = 0
    while h0 * h0 - d * k0 * k0 != 1:
        a = period[i % len(period)]
        h0, h1 = a * h0 + h1, h0
        k0, k1 = a * k0 + k1, k0
        i += 1
    return (h0, k0)


@op('numbers.pell_negative', kind='Diophantine', bound='d up to 10**18; continued-fraction period up to 20,000')
def pell_negative(d):
    """The least positive solution (x, y) of x**2 - d*y**2 = -1 for a non-square d, or () when it has none (the period of sqrt(d) is even). pell_negative(13) -> (18, 5)."""
    d = _int(d)
    _pos(d, 'd')
    _small(d, 10 ** 18, 'd')
    a0, period = _cf_sqrt(d)
    if not period or len(period) % 2 == 0:
        return ()
    _small(len(period), 20_000, 'period')
    h0, h1, k0, k1 = a0, 1, 1, 0
    for i in range(len(period) - 1):
        a = period[i]
        h0, h1 = a * h0 + h1, h0
        k0, k1 = a * k0 + k1, k0
    return (h0, k0)


@op('numbers.pell_solutions', kind='Diophantine', bound='count up to 1000, as pell')
def pell_solutions(d, count):
    """The first `count` positive solutions of x**2 - d*y**2 = 1 in increasing order. pell_solutions(2, 3) -> [(3, 2), (17, 12), (99, 70)]."""
    count = _int(count)
    _pos(count, 'count')
    _small(count, 1000, 'count')
    base = pell(d)
    if base == ():
        return []
    d = _int(d)
    x1, y1 = base
    out, x, y = [], x1, y1
    for _ in range(count):
        out.append((x, y))
        x, y = x1 * x + d * y1 * y, x1 * y + y1 * x
    return out


@op('numbers.pell_search', kind='Diophantine', bound='y up to 10**6')
def pell_search(d, n, ymax):
    """Every (x, y) with x >= 0, 0 <= y <= ymax and x**2 - d*y**2 = n (a search; for the same equation with any n). pell_search(2, -1, 100) -> [(1, 1), (7, 5), (41, 29)]."""
    d, n, ymax = _int(d), _int(n), _int(ymax)
    _pos(d, 'd')
    _nonneg(ymax, 'ymax')
    _small(ymax, 10 ** 6, 'ymax')
    out = []
    for y in range(ymax + 1):
        t = n + d * y * y
        if t >= 0:
            x = math.isqrt(t)
            if x * x == t:
                out.append((x, y))
    return out


# ----------------------------------------------------------------------------------------------------------------------------------------------------------
# sums of squares

def _two_squares_ok(f):
    return all(e % 2 == 0 for q, e in f.items() if q % 4 == 3)


@op('numbers.is_sum_of_two_squares', kind='sums of squares', bound='as factorint')
def is_sum_of_two_squares(n):
    """True when n >= 0 is a sum of two integer squares (every prime 3 mod 4 occurs to an even power). is_sum_of_two_squares(50) -> True."""
    n = _int(n)
    _nonneg(n, 'n')
    return n == 0 or _two_squares_ok(_factor_dict(n))


@op('numbers.r2', kind='sums of squares', bound='as factorint')
def r2(n):
    """The number of ordered pairs of integers (x, y), signs included, with x*x + y*y = n >= 0 (Jacobi: 4 (d1(n) - d3(n))). r2(25) -> 12."""
    n = _int(n)
    _nonneg(n, 'n')
    if n == 0:
        return 1
    f = _factor_dict(n)
    if not _two_squares_ok(f):
        return 0
    out = 4
    for p, e in f.items():
        if p % 4 == 1:
            out *= e + 1
    return out


def _gauss_prime(p):
    """(x, y) with x*x + y*y = p for a prime p = 1 (mod 4), by Hermite-Serret: the first remainders below sqrt(p) in Euclid's algorithm on (p, sqrt(-1))."""
    r = _sqrt_prime(p - 1, p)[0]
    a, b = p, r
    while b * b > p:
        a, b = b, a % b
    return b, math.isqrt(p - b * b)


@op('numbers.two_squares', kind='sums of squares', bound='as factorint; at most 100,000 representations')
def two_squares(n):
    """Every (a, b) with 0 <= a <= b and a*a + b*b = n >= 0, sorted; [] when there is none. two_squares(50) -> [(1, 7), (5, 5)]."""
    n = _int(n)
    _nonneg(n, 'n')
    if n == 0:
        return [(0, 0)]
    f = _factor_dict(n)
    if not _two_squares_ok(f):
        return []
    total = 1
    for p, e in f.items():
        if p % 4 == 1:
            total *= e + 1
    _small(total, 100_000, 'number of representations')
    ints = [(1, 0)]
    for p, e in f.items():
        if p == 2:
            for _ in range(e):
                ints = [(a - b, a + b) for a, b in ints]       # times (1 + i)
        elif p % 4 == 3:
            ints = [(a * p ** (e // 2), b * p ** (e // 2)) for a, b in ints]
        else:
            x, y = _gauss_prime(p)
            options = []
            for k in range(e + 1):                             # (x + yi)^k (x - yi)^(e-k)
                zr, zi = 1, 0
                for j in range(e):
                    wr, wi = (x, y) if j < k else (x, -y)
                    zr, zi = zr * wr - zi * wi, zr * wi + zi * wr
                options.append((zr, zi))
            ints = [(a * c - b * d, a * d + b * c) for a, b in ints for c, d in options]
    return sorted({tuple(sorted((abs(a), abs(b)))) for a, b in ints})


@op('numbers.is_sum_of_three_squares', kind='sums of squares', bound='any size')
def is_sum_of_three_squares(n):
    """True when n >= 0 is a sum of three integer squares: not of the form 4**a * (8b + 7) (Legendre). is_sum_of_three_squares(7) -> False."""
    n = _int(n)
    _nonneg(n, 'n')
    while n and n % 4 == 0:
        n //= 4
    return n % 8 != 7


@op('numbers.r4', kind='sums of squares', bound='as factorint')
def r4(n):
    """The number of integer 4-tuples with x1**2 + x2**2 + x3**2 + x4**2 = n >= 1 (Jacobi: 8 times the sum of the divisors not divisible by 4). r4(5) -> 48."""
    n = _int(n)
    _pos(n, 'n')
    s = 0
    for d in divisors(n):
        if d % 4:
            s += d
    return 8 * s


@op('numbers.four_squares', kind='sums of squares', bound='n up to 10**18; at most 200,000 candidates')
def four_squares(n):
    """(a, b, c, d) with 0 <= a <= b <= c <= d and a*a + b*b + c*c + d*d = n >= 0 (Lagrange; found by search with an exact two-squares test). four_squares(7) -> (1, 1, 1, 2)."""
    n = _int(n)
    _nonneg(n, 'n')
    _small(n, 10 ** 18, 'n')
    if n == 0:
        return (0, 0, 0, 0)
    shift = 0
    while n % 4 == 0:                      # a representation of 4m is twice one of m
        n //= 4
        shift += 1
    tries = 0
    d = math.isqrt(n)
    while d >= 0:
        r = n - d * d
        c = math.isqrt(r)
        while c >= 0:
            tries += 1
            if tries > 200_000:
                raise NotCovered('search bound reached')
            try:
                if is_sum_of_two_squares(r - c * c):
                    a, b = two_squares(r - c * c)[0]
                    return tuple(x << shift for x in sorted((a, b, c, d)))
            except NotCovered:
                pass                       # this remainder could not be factored; another one may be
            c -= 1
        d -= 1
    raise NotCovered('no representation found')


# ----------------------------------------------------------------------------------------------------------------------------------------------------------
# Frobenius, floor sum, Pythagorean triples

def _frobenius_data(coins):
    coins = sorted({_int(c, 'coin') for c in coins})
    if not coins:
        raise BadInput('no coins')
    if coins[0] < 1:
        raise BadInput('coins must be positive')
    g = 0
    for c in coins:
        g = math.gcd(g, c)
    return coins, g


@op('numbers.frobenius', kind='Frobenius', bound='two coins any size; more coins: the least coin up to 100,000')
def frobenius(*coins):
    """The largest integer that is not a non-negative combination of the coins (-1 when 1 is a coin), or () when the coins share a factor (infinitely many gaps).
    frobenius(3, 5) -> 7; frobenius(6, 9, 20) -> 43."""
    coins, g = _frobenius_data(coins)
    if g != 1:
        return ()
    if coins[0] == 1:
        return -1
    if len(coins) == 2:
        return coins[0] * coins[1] - coins[0] - coins[1]
    _small(coins[0], 100_000, 'least coin')
    _small(len(coins), 50, 'coin count')
    return max(_residue_dist(coins)) - coins[0]


def _residue_dist(coins):
    m = coins[0]
    dist = [None] * m
    dist[0] = 0
    heap = [(0, 0)]
    while heap:
        d, r = heapq.heappop(heap)
        if d > dist[r]:
            continue
        for c in coins[1:]:
            nd = d + c
            nr = nd % m
            if dist[nr] is None or nd < dist[nr]:
                dist[nr] = nd
                heapq.heappush(heap, (nd, nr))
    return dist


@op('numbers.frobenius_count', kind='Frobenius', bound='two coins any size; more coins: the least coin up to 100,000')
def frobenius_count(*coins):
    """How many positive integers are not a non-negative combination of the coins (Sylvester: (a-1)(b-1)/2 for two coprime coins); () when the coins share a factor.
    frobenius_count(3, 5) -> 4."""
    coins, g = _frobenius_data(coins)
    if g != 1:
        return ()
    if coins[0] == 1:
        return 0
    if len(coins) == 2:
        return (coins[0] - 1) * (coins[1] - 1) // 2
    _small(coins[0], 100_000, 'least coin')
    _small(len(coins), 50, 'coin count')
    m = coins[0]
    return sum((d - r) // m for r, d in enumerate(_residue_dist(coins)))


@op('numbers.floor_sum', kind='floor sum', bound='n, m, a, b up to 2,000,000 bits')
def floor_sum(n, m, a, b):
    """The sum of floor((a*i + b) / m) for i = 0 .. n-1, with n >= 0, m >= 1 and any integers a, b (Euclid-like, O(log m)). floor_sum(10, 7, 3, 2) -> 21."""
    n, m, a, b = _int(n), _int(m), _int(a), _int(b)
    _nonneg(n, 'n')
    _pos(m, 'm')
    for v in (n, m, a, b):
        _small(v.bit_length(), _BITS, 'argument')
    ans = 0
    while True:
        ans += (n - 1) * n // 2 * (a // m)
        a %= m
        ans += n * (b // m)
        b %= m
        y_max = a * n + b
        if y_max < m:
            break
        n, b = y_max // m, y_max % m
        m, a = a, m
    return ans


@op('numbers.pythagorean_triples', kind='Diophantine', bound='hypotenuse up to 10**6, at most 100,000 triples')
def pythagorean_triples(max_c, primitive=False):
    """Every (a, b, c) with a < b, a*a + b*b = c*c and c <= max_c (only the primitive ones when primitive=True), sorted. pythagorean_triples(20) -> [(3, 4, 5), (5, 12, 13), (6, 8, 10), (8, 15, 17), (9, 12, 15), (12, 16, 20)]."""
    max_c = _int(max_c)
    _nonneg(max_c, 'max_c')
    _small(max_c, 10 ** 6, 'max_c')
    out = []
    m = 2
    while m * m + 1 <= max_c:
        for n in range(2 if m % 2 else 1, m, 2):          # opposite parity to m
            if math.gcd(m, n) != 1:
                continue
            c = m * m + n * n
            if c > max_c:
                break
            a, b = m * m - n * n, 2 * m * n
            if a > b:
                a, b = b, a
            for k in (range(1, max_c // c + 1) if not primitive else (1,)):
                out.append((a * k, b * k, c * k))
                _small(len(out), 100_000, 'triples')
        m += 1
    return sorted(out)


@op('numbers.pythagorean_triple_count', kind='Diophantine', bound='hypotenuse up to 10**7')
def pythagorean_triple_count(max_c):
    """The number of integer right triangles (a < b) with hypotenuse <= max_c. pythagorean_triple_count(100) -> 52."""
    max_c = _int(max_c)
    _nonneg(max_c, 'max_c')
    _small(max_c, 10 ** 7, 'max_c')
    total = 0
    m = 2
    while m * m + 1 <= max_c:
        for n in range(2 if m % 2 else 1, m, 2):
            if math.gcd(m, n) != 1:
                continue
            c = m * m + n * n
            if c > max_c:
                break
            total += max_c // c
        m += 1
    return total
