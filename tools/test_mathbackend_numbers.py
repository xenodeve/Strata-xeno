"""tools/igc/mathbackend/numbers.py unit tests. python tools/test_mathbackend_numbers.py (standard library only, no model, no network).

The expected values are known results (OEIS-style tables, textbook examples) or are checked by brute force or by a certificate (the answer is verified, not
recomputed by the same code). The independent oracle runs are in sim_numbers.py."""
import math
import random
import sys
import unittest
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.igc.mathbackend import BadInput, NotCovered, registry, run
import tools.igc.mathbackend.numbers as N


def trial_prime(n):
    if n < 2:
        return False
    return all(n % d for d in range(2, math.isqrt(n) + 1))


class Arithmetic(unittest.TestCase):
    def test_rationals(self):
        self.assertEqual(run('numbers.add', 1, '1/2'), Fraction(3, 2))
        self.assertEqual(run('numbers.div', 1, 3), Fraction(1, 3))
        self.assertEqual(run('numbers.mul', '2/3', '9/4'), Fraction(3, 2))
        with self.assertRaises(BadInput):
            run('numbers.div', 1, 0)
        with self.assertRaises(BadInput):
            run('numbers.add', 0.5, 1)
        with self.assertRaises(BadInput):
            run('numbers.add', '1e999999999', 1)
        with self.assertRaises(BadInput):
            run('numbers.add', True, 1)

    def test_powers(self):
        self.assertEqual(run('numbers.pow_int', '2/3', -3), Fraction(27, 8))
        self.assertEqual(run('numbers.pow_int', -2, 3), -8)
        self.assertEqual(run('numbers.pow_int', 1, 10 ** 12), 1)
        self.assertEqual(run('numbers.pow_int', -1, 10 ** 12 + 1), -1)
        with self.assertRaises(BadInput):
            run('numbers.pow_int', 0, -1)
        with self.assertRaises(NotCovered):
            run('numbers.pow_int', 3, 10 ** 8)
        self.assertEqual(run('numbers.pow_rational', 8, '2/3'), 4)
        self.assertEqual(run('numbers.pow_rational', '9/4', '-1/2'), Fraction(2, 3))
        with self.assertRaises(NotCovered):
            run('numbers.pow_rational', 2, '1/2')
        with self.assertRaises(NotCovered):
            run('numbers.pow_rational', -8, '1/3')

    def test_roots(self):
        self.assertEqual(run('numbers.iroot', 1000, 3), 10)
        self.assertEqual(run('numbers.iroot', 999, 3), 9)
        rng = random.Random(1)
        for _ in range(300):
            n, k = rng.randrange(10 ** rng.randint(1, 60)), rng.randint(1, 12)
            r = run('numbers.iroot', n, k)
            self.assertTrue(r ** k <= n < (r + 1) ** k)
        self.assertEqual(run('numbers.perfect_power', 64), (2, 6))
        self.assertEqual(run('numbers.perfect_power', -27), (-3, 3))
        self.assertEqual(run('numbers.perfect_power', -16), (-16, 1))
        self.assertEqual(run('numbers.perfect_power', 72), (72, 1))
        self.assertEqual(run('numbers.perfect_power', 2 ** 12 * 3 ** 6), (2 ** 2 * 3, 6))
        for n in range(2, 3000):
            b, e = run('numbers.perfect_power', n)
            self.assertEqual(b ** e, n)
            top = max([e2 for e2 in range(1, 13) if round(n ** (1 / e2)) ** e2 == n] or [1])
            self.assertEqual(e, top)
        self.assertTrue(run('numbers.is_square', '4/9'))
        self.assertFalse(run('numbers.is_square', -4))


class Modular(unittest.TestCase):
    def test_gcd_lcm_xgcd(self):
        self.assertEqual(run('numbers.gcd', 12, 18, 30), 6)
        self.assertEqual(run('numbers.lcm', 4, 6, 10), 60)
        self.assertEqual(run('numbers.gcd', 0, 0), 0)
        rng = random.Random(2)
        for _ in range(200):
            a, b = rng.randint(-10 ** 9, 10 ** 9), rng.randint(-10 ** 9, 10 ** 9)
            g, x, y = run('numbers.xgcd', a, b)
            self.assertEqual(g, math.gcd(a, b))
            self.assertEqual(a * x + b * y, g)

    def test_inverse_pow_crt(self):
        self.assertEqual(run('numbers.modinv', 3, 7), 5)
        self.assertEqual(run('numbers.modinv', 2, 4), ())
        self.assertEqual(run('numbers.modinv', 5, 1), 0)
        self.assertEqual(run('numbers.powmod', 2, 100, 10 ** 9 + 7), pow(2, 100, 10 ** 9 + 7))
        self.assertEqual(run('numbers.powmod', 3, -1, 7), 5)
        self.assertEqual(run('numbers.powmod', 2, -1, 4), ())
        self.assertEqual(run('numbers.crt', [2, 3], [3, 5]), (8, 15))
        self.assertEqual(run('numbers.crt', [1, 0], [2, 4]), ())
        r, m = run('numbers.crt', [3, 5, 3], [6, 8, 9])
        self.assertEqual(m, 72)
        self.assertEqual((r % 6, r % 8, r % 9), (3, 5, 3))
        rng = random.Random(3)
        for _ in range(200):                      # brute force over the lcm
            ms = [rng.randint(1, 12) for _ in range(3)]
            rs = [rng.randint(0, 20) for _ in range(3)]
            lcm_ = math.lcm(*ms)
            sol = [x for x in range(lcm_) if all((x - r) % m == 0 for r, m in zip(rs, ms))]
            out = run('numbers.crt', rs, ms)
            if sol:
                self.assertEqual(out, (sol[0], lcm_))
                self.assertEqual(len(sol), 1)
            else:
                self.assertEqual(out, ())

    def test_lincong(self):
        self.assertEqual(run('numbers.lincong', 6, 4, 10), (4, 5))
        for a in range(0, 13):
            for b in range(0, 13):
                for n in range(1, 13):
                    sol = [x for x in range(n) if (a * x - b) % n == 0]
                    out = run('numbers.lincong', a, b, n)
                    if not sol:
                        self.assertEqual(out, ())
                    else:
                        x0, step = out
                        self.assertEqual(sorted(x for x in range(n) if (x - x0) % step == 0), sol)

    def test_symbols_roots(self):
        self.assertEqual(run('numbers.jacobi', 2, 15), 1)
        self.assertEqual(run('numbers.legendre', 3, 7), -1)
        for p in (3, 5, 7, 11, 13, 101):
            for a in range(-p, 2 * p):
                self.assertEqual(run('numbers.legendre', a, p), 0 if a % p == 0 else (1 if any((x * x - a) % p == 0 for x in range(p)) else -1))
        for n in range(1, 60, 2):
            for a in range(-5, 40):
                j = run('numbers.jacobi', a, n)
                prod = 1
                for p, e in run('numbers.factorint', n).items():
                    prod *= run('numbers.legendre', a, p) ** e if p > 2 else 1
                self.assertEqual(j, prod)
        self.assertEqual(run('numbers.sqrt_mod', 10, 13), [6, 7])
        for p in (2, 3, 5, 7, 13, 17, 29, 41, 97, 193, 257):
            for a in range(p + 3):
                self.assertEqual(run('numbers.sqrt_mod', a, p), [x for x in range(p) if (x * x - a) % p == 0])
        big = 2 ** 61 - 1
        for a in (4, 123456789 ** 2 % big):
            for r in run('numbers.sqrt_mod', a, big):
                self.assertEqual(r * r % big, a % big)
        p = 10 ** 18 + 9                           # p = 1 mod 8: Tonelli-Shanks proper
        self.assertTrue(run('numbers.is_prime', p))
        for r in run('numbers.sqrt_mod', 17 ** 2, p):
            self.assertEqual(r * r % p, 289)

    def test_sqrt_mod_all(self):
        for n in range(1, 90):
            for a in range(0, 2 * n, 3):
                self.assertEqual(run('numbers.sqrt_mod_all', a, n), [x for x in range(n) if (x * x - a) % n == 0], (a, n))
        out = run('numbers.sqrt_mod_all', 4, 3 ** 12 * 5 ** 3)
        self.assertTrue(all((x * x - 4) % (3 ** 12 * 5 ** 3) == 0 for x in out) and len(out) == 4)

    def test_discrete_log(self):
        self.assertEqual(run('numbers.discrete_log', 2, 8, 13), 3)
        for n in range(1, 60):
            for a in range(0, n + 3):
                for b in range(0, n + 1, 3):
                    xs = [x for x in range(2 * n + 2) if pow(a, x, n) == b % n]
                    out = run('numbers.discrete_log', a, b, n)
                    self.assertEqual(out, xs[0] if xs else (), (a, b, n))

    def test_binomial_mod(self):
        for m in (1, 2, 4, 6, 8, 9, 12, 30, 97, 100, 1000):
            for n in range(0, 40):
                for k in range(0, n + 2):
                    self.assertEqual(run('numbers.binomial_mod', n, k, m), math.comb(n, k) % m if k <= n else 0, (n, k, m))
        self.assertEqual(run('numbers.binomial_mod', 10 ** 5, 5 * 10 ** 4, 10 ** 9 + 7), math.comb(10 ** 5, 5 * 10 ** 4) % (10 ** 9 + 7))


class Primes(unittest.TestCase):
    def test_small_primality_exhaustive(self):
        for n in range(-5, 5000):
            self.assertEqual(run('numbers.is_prime', n), trial_prime(n), n)

    def test_pseudoprimes(self):
        for n in (561, 1105, 1729, 2047, 3215031751, 3825123056546413051, 318665857834031151167461, 3317044064679887385961981,
                  2 ** 61 - 1, 2 ** 89 - 1, 2 ** 127 - 1):
            expect = n in (2 ** 61 - 1, 2 ** 89 - 1, 2 ** 127 - 1)
            self.assertEqual(run('numbers.is_prime', n), expect, n)

    def test_bpsw_agrees_with_deterministic(self):
        rng = random.Random(4)
        for _ in range(400):
            n = rng.randrange(3, 10 ** rng.randint(6, 24)) | 1
            if n < 5000:
                continue
            self.assertEqual(N._bpsw(n), N._is_prime_det(n), n)

    def test_large_primes_proof_and_refusal(self):
        self.assertTrue(run('numbers.is_prime', 2 ** 127 - 1))
        self.assertFalse(run('numbers.is_prime', (2 ** 89 - 1) * (2 ** 61 - 1)))
        # a prime above 3.3e24 with p-1 = 2 * (a prime near 2**100): proven by Pocklington with the cofactor proven in turn
        q = N.next_prime(2 ** 100)
        for k in range(1, 200):
            if N._bpsw(2 * k * q + 1):
                p = 2 * k * q + 1
                break
        self.assertTrue(run('numbers.is_prime', p))

    def test_next_prev(self):
        self.assertEqual(run('numbers.next_prime', 100), 101)
        self.assertEqual(run('numbers.next_prime', 101), 103)
        self.assertEqual(run('numbers.next_prime', -3), 2)
        self.assertEqual(run('numbers.next_prime', 2), 3)
        self.assertEqual(run('numbers.prev_prime', 100), 97)
        self.assertEqual(run('numbers.prev_prime', 3), 2)
        with self.assertRaises(BadInput):
            run('numbers.prev_prime', 2)
        for n in range(1, 400):
            self.assertEqual(run('numbers.next_prime', n), next(m for m in range(n + 1, n + 100) if trial_prime(m)))
            if n > 2:
                self.assertEqual(run('numbers.prev_prime', n), next(m for m in range(n - 1, 0, -1) if trial_prime(m)))

    def test_counting(self):
        self.assertEqual(run('numbers.primes_upto', 20), [2, 3, 5, 7, 11, 13, 17, 19])
        self.assertEqual(run('numbers.primes_upto', 1), [])
        self.assertEqual(run('numbers.nth_prime', 1000), 7919)
        self.assertEqual(run('numbers.nth_prime', 1), 2)
        self.assertEqual(run('numbers.prime_pi', 100), 25)
        self.assertEqual(run('numbers.prime_pi', 10 ** 6), 78498)
        self.assertEqual(run('numbers.prime_pi', 10 ** 8), 5761455)
        for n in list(range(0, 300)) + [997, 1000, 12345]:
            self.assertEqual(run('numbers.prime_pi', n), sum(trial_prime(m) for m in range(n + 1)))

    def test_factorint_certificate(self):
        rng = random.Random(5)
        cases = [1, 2, 360, 2 ** 64 - 1, 2 ** 67 - 1, 10 ** 18 + 9, 600851475143, 999999999999999989 * 3, 2 ** 100 - 1, 7 ** 40]
        cases += [rng.randrange(2, 10 ** rng.randint(3, 30)) for _ in range(60)]
        for n in cases:
            try:
                f = run('numbers.factorint', n)
            except NotCovered:
                self.assertGreater(n, 10 ** 15)    # a refusal is allowed only for a number with two big factors
                continue
            prod = 1
            for p, e in f.items():
                self.assertTrue(e >= 1 and run('numbers.is_prime', p))
                if p < 10 ** 12:
                    self.assertTrue(trial_prime(p))
                prod *= p ** e
            self.assertEqual(prod, n)
        self.assertEqual(run('numbers.factorint', 360), {2: 3, 3: 2, 5: 1})
        self.assertEqual(run('numbers.factorint', 1), {})

    def test_factorint_refuses_hard_semiprime(self):
        p, q = N.next_prime(10 ** 15), N.next_prime(3 * 10 ** 15)
        with self.assertRaises(NotCovered):
            run('numbers.factorint', p * q)
        part, left = run('numbers.factorint_partial', 6 * p * q)
        self.assertEqual(part, {2: 1, 3: 1})
        self.assertEqual(left, [p * q])

    def test_misc(self):
        self.assertTrue(run('numbers.is_squarefree', 30))
        self.assertFalse(run('numbers.is_squarefree', 12))
        self.assertEqual(run('numbers.smallest_prime_factor', 91), 7)
        self.assertEqual(run('numbers.radical', 360), 30)
        with self.assertRaises(BadInput):
            run('numbers.factorint', 0)
        with self.assertRaises(NotCovered):
            run('numbers.factorint', 2 ** 300 + 1)


class NumberTheory(unittest.TestCase):
    def test_divisor_functions_brute(self):
        for n in range(1, 400):
            ds = [d for d in range(1, n + 1) if n % d == 0]
            self.assertEqual(run('numbers.divisors', n), ds)
            self.assertEqual(run('numbers.num_divisors', n), len(ds))
            self.assertEqual(run('numbers.sigma', n), sum(ds))
            self.assertEqual(run('numbers.sigma', n, 2), sum(d * d for d in ds))
            self.assertEqual(run('numbers.sigma', n, 0), len(ds))
            self.assertEqual(run('numbers.totient', n), sum(math.gcd(k, n) == 1 for k in range(1, n + 1)))

    def test_moebius_carmichael_order_primitive(self):
        mu = {1: 1, 2: -1, 4: 0, 6: 1, 30: -1, 12: 0}
        for n, v in mu.items():
            self.assertEqual(run('numbers.moebius', n), v)
        self.assertEqual(run('numbers.carmichael', 15), 4)
        self.assertEqual(run('numbers.carmichael', 8), 2)
        self.assertEqual(run('numbers.carmichael', 16), 4)
        for n in range(1, 200):
            units = [a for a in range(1, n + 1) if math.gcd(a, n) == 1] if n > 1 else [0]
            def brute_order(a):
                if n == 1:
                    return 1
                k, x = 1, a % n
                while x != 1:
                    x = x * a % n
                    k += 1
                return k
            orders = {a: brute_order(a) for a in units}
            self.assertEqual(run('numbers.carmichael', n), math.lcm(*orders.values()), n)
            for a in range(0, n + 2):
                got = run('numbers.order', a, n)
                if n == 1:
                    self.assertEqual(got, 1)
                elif math.gcd(a, n) != 1:
                    self.assertEqual(got, ())
                else:
                    self.assertEqual(got, orders[a % n], (a, n))
            cyc = [g for g in units if orders[g] == len(units)] if n > 2 else []
            pr = run('numbers.primitive_root', n)
            if n <= 2:
                self.assertEqual(pr, n - 1)
            elif cyc:
                self.assertEqual(pr, min(cyc), n)
            else:
                self.assertEqual(pr, (), n)

    def test_valuations(self):
        self.assertEqual(run('numbers.valuation', 360, 2), 3)
        self.assertEqual(run('numbers.valuation', -360, 6), 2)
        self.assertEqual(run('numbers.factorial_valuation', 100, 5), 24)
        self.assertEqual(run('numbers.factorial_trailing_zeros', 100), 24)
        self.assertEqual(run('numbers.factorial_trailing_zeros', 10, 12), 4)
        for n in range(0, 60):
            f = math.factorial(n)
            for b in (2, 6, 10, 12, 16, 100):
                z = 0
                while f % b ** (z + 1) == 0:
                    z += 1
                self.assertEqual(run('numbers.factorial_trailing_zeros', n, b), z, (n, b))


class Combinatorial(unittest.TestCase):
    def test_known_values(self):
        self.assertEqual(run('numbers.factorial', 10), 3628800)
        self.assertEqual(run('numbers.double_factorial', 7), 105)
        self.assertEqual(run('numbers.double_factorial', 8), 384)
        self.assertEqual(run('numbers.double_factorial', -1), 1)
        self.assertEqual(run('numbers.binomial', 52, 5), 2598960)
        self.assertEqual(run('numbers.binomial', -3, 2), 6)
        self.assertEqual(run('numbers.binomial', 5, 7), 0)
        self.assertEqual(run('numbers.binomial', 5, -1), 0)
        self.assertEqual(run('numbers.multinomial', 2, 3, 1), 60)
        self.assertEqual(run('numbers.rising_factorial', 3, 4), 360)
        self.assertEqual(run('numbers.falling_factorial', 10, 3), 720)
        self.assertEqual(run('numbers.rising_factorial', '1/2', 2), Fraction(3, 4))
        self.assertEqual([run('numbers.fibonacci', n) for n in range(11)], [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55])
        self.assertEqual(run('numbers.fibonacci', -6), -8)
        self.assertEqual(run('numbers.fibonacci', -7), 13)
        self.assertEqual([run('numbers.lucas', n) for n in range(8)], [2, 1, 3, 4, 7, 11, 18, 29])
        self.assertEqual(run('numbers.lucas', -3), -4)
        self.assertEqual(run('numbers.fibonacci', 10 ** 18, 10 ** 9 + 7), 209783453)
        self.assertEqual([run('numbers.catalan', n) for n in range(9)], [1, 1, 2, 5, 14, 42, 132, 429, 1430])
        self.assertEqual([run('numbers.bell', n) for n in range(9)], [1, 1, 2, 5, 15, 52, 203, 877, 4140])
        self.assertEqual([run('numbers.partition_number', n) for n in range(13)], [1, 1, 2, 3, 5, 7, 11, 15, 22, 30, 42, 56, 77])
        self.assertEqual(run('numbers.partition_number', 100), 190569292)
        self.assertEqual(run('numbers.stirling2', 5, 3), 25)
        self.assertEqual(run('numbers.stirling1', 4, 2), 11)
        self.assertEqual(run('numbers.stirling1', 4, 3, signed=True), -6)

    def test_fibonacci_mod_matches_exact(self):
        for n in list(range(0, 80)) + [1000, 4321]:
            for m in (1, 2, 7, 10, 1000, 10 ** 9 + 7):
                self.assertEqual(run('numbers.fibonacci', n, m), run('numbers.fibonacci', n) % m)
                self.assertEqual(run('numbers.lucas', n, m), run('numbers.lucas', n) % m)

    def test_stirling_bell_brute(self):
        S2 = {(0, 0): 1}
        S1 = {(0, 0): 1}
        for n in range(1, 13):
            for k in range(0, n + 1):
                S2[n, k] = k * S2.get((n - 1, k), 0) + S2.get((n - 1, k - 1), 0)
                S1[n, k] = (n - 1) * S1.get((n - 1, k), 0) + S1.get((n - 1, k - 1), 0)
        for n in range(0, 13):
            for k in range(0, 15):
                self.assertEqual(run('numbers.stirling2', n, k), S2.get((n, k), 0), (n, k))
                self.assertEqual(run('numbers.stirling1', n, k), S1.get((n, k), 0), (n, k))
        for n in range(0, 12):
            self.assertEqual(run('numbers.bell', n), sum(S2.get((n, k), 0) for k in range(n + 1)))

    def test_partition_brute(self):
        for n in range(0, 40):
            ways = [1] + [0] * n
            for part in range(1, n + 1):
                for s in range(part, n + 1):
                    ways[s] += ways[s - part]
            self.assertEqual(run('numbers.partition_number', n), ways[n])

    def test_bounds(self):
        with self.assertRaises(NotCovered):
            run('numbers.factorial', 10 ** 7)
        with self.assertRaises(BadInput):
            run('numbers.factorial', -1)
        with self.assertRaises(NotCovered):
            run('numbers.bell', 5000)


class Digits(unittest.TestCase):
    def test_digits(self):
        self.assertEqual(run('numbers.digits', 255, 16), [15, 15])
        self.assertEqual(run('numbers.digits', 0), [0])
        self.assertEqual(run('numbers.digit_sum', 2 ** 100), 115)
        self.assertEqual(run('numbers.num_digits', 2 ** 100), 31)
        self.assertEqual(run('numbers.from_digits', [1, 0, 1], 2), 5)
        rng = random.Random(6)
        for _ in range(300):
            n, b = rng.randrange(10 ** rng.randint(1, 80)), rng.choice([2, 3, 7, 10, 16, 36, 1000, 10 ** 12])
            ds = run('numbers.digits', n, b)
            self.assertEqual(run('numbers.from_digits', ds, b), n)
            self.assertTrue(all(0 <= d < b for d in ds) and (ds[0] != 0 or ds == [0]))
        big = 7 ** 30000
        self.assertEqual(len(run('numbers.digits', big)), math.floor(30000 * math.log10(7)) + 1)
        self.assertEqual(run('numbers.from_digits', run('numbers.digits', big)), big)

    def test_base_convert(self):
        self.assertEqual(run('numbers.base_convert', 'ff', 16, 2), '11111111')
        self.assertEqual(run('numbers.base_convert', '-101', 2, 10), '-5')
        self.assertEqual(run('numbers.base_convert', '0', 2, 16), '0')
        for _ in range(100):
            n = random.randrange(10 ** 20)
            for a, b in ((10, 2), (10, 36), (2, 16), (36, 7)):
                s = run('numbers.base_convert', str(n), 10, a) if a != 10 else str(n)
                self.assertEqual(int(run('numbers.base_convert', s, a, b), b), n)
        with self.assertRaises(BadInput):
            run('numbers.base_convert', '12', 2, 10)
        self.assertEqual(run('numbers.digital_root', 987654321), 9)
        self.assertEqual(run('numbers.digital_root', 0), 0)
        for n in range(0, 500):
            r = n
            while r >= 10:
                r = sum(map(int, str(r)))
            self.assertEqual(run('numbers.digital_root', n), r)

    def test_rounding(self):
        self.assertEqual(run('numbers.rat_floor', '-7/2'), -4)
        self.assertEqual(run('numbers.rat_ceil', '-7/2'), -3)
        self.assertEqual(run('numbers.rat_trunc', '-7/2'), -3)
        self.assertEqual(run('numbers.rat_round', '5/2'), 3)
        self.assertEqual(run('numbers.rat_round', '-5/2'), -3)
        self.assertEqual(run('numbers.rat_round', '5/2', 'half_even'), 2)
        self.assertEqual(run('numbers.rat_round', '-5/2', 'half_up'), -2)
        self.assertEqual(run('numbers.rat_round', '7/3'), 2)
        self.assertEqual(run('numbers.rat_mod', '7/2', 1), Fraction(1, 2))
        self.assertEqual(run('numbers.rat_mod', -7, 3), 2)
        self.assertEqual(run('numbers.rat_mod', 7, -3), -2)
        self.assertEqual(run('numbers.frac_part', '-7/2'), Fraction(1, 2))
        with self.assertRaises(BadInput):
            run('numbers.rat_mod', 1, 0)


class ContinuedFractionsAndDecimals(unittest.TestCase):
    def test_cf(self):
        self.assertEqual(run('numbers.cf_rational', '415/93'), [4, 2, 6, 7])
        self.assertEqual(run('numbers.cf_rational', '-7/2'), [-4, 2])
        self.assertEqual(run('numbers.cf_rational', 5), [5])
        self.assertEqual(run('numbers.cf_value', [4, 2, 6, 7]), Fraction(415, 93))
        self.assertEqual(run('numbers.cf_convergents', '415/93'), [4, Fraction(9, 2), Fraction(58, 13), Fraction(415, 93)])
        self.assertEqual(run('numbers.cf_convergents', [1, 2, 2, 2]), [1, Fraction(3, 2), Fraction(7, 5), Fraction(17, 12)])
        rng = random.Random(7)
        for _ in range(200):
            x = Fraction(rng.randint(-10 ** 6, 10 ** 6), rng.randint(1, 10 ** 6))
            self.assertEqual(run('numbers.cf_value', run('numbers.cf_rational', x)), x)
        self.assertEqual(run('numbers.cf_sqrt', 23), (4, [1, 3, 1, 8]))
        self.assertEqual(run('numbers.cf_sqrt', 2), (1, [2]))
        self.assertEqual(run('numbers.cf_sqrt', 49), (7, []))
        self.assertEqual(run('numbers.cf_sqrt_convergents', 2, 4), [1, Fraction(3, 2), Fraction(7, 5), Fraction(17, 12)])
        for d in range(2, 200):
            if math.isqrt(d) ** 2 == d:
                continue
            for h in run('numbers.cf_sqrt_convergents', d, 12):
                # a convergent h of sqrt(d) is within 1/q^2 of it
                self.assertLess(abs(h.numerator ** 2 - d * h.denominator ** 2), 2 * math.isqrt(d) + 2)
        self.assertEqual(run('numbers.best_approx', '3141592653/1000000000', 100), Fraction(311, 99))

    def test_best_approx_brute(self):
        rng = random.Random(8)
        for _ in range(200):
            x = Fraction(rng.randint(1, 10 ** 5), rng.randint(1, 10 ** 5))
            md = rng.randint(1, 60)
            got = run('numbers.best_approx', x, md)
            self.assertLessEqual(got.denominator, md)
            best = min(abs(Fraction(round(x * q), q) - x) for q in range(1, md + 1))
            self.assertEqual(abs(got - x), best)

    def long_division(self, x, digits):
        n, d = x.numerator, x.denominator
        out, r = [], n % d
        for _ in range(digits):
            r *= 10
            out.append(r // d)
            r %= d
        return ''.join(map(str, out))

    def test_decimals(self):
        self.assertEqual(run('numbers.decimal_expansion', '1/7', 12), '0.142857142857')
        self.assertEqual(run('numbers.decimal_expansion', '-1/8', 5), '-0.12500')
        self.assertEqual(run('numbers.decimal_expansion', 5, 0), '5')
        self.assertEqual(run('numbers.decimal_expansion', '2/3', 0), '0')
        self.assertEqual(run('numbers.decimal_period', '1/7'), 6)
        self.assertEqual(run('numbers.decimal_period', '1/6'), 1)
        self.assertEqual(run('numbers.decimal_period', '1/8'), 0)
        self.assertEqual(run('numbers.decimal_preperiod', '1/6'), 1)
        self.assertEqual(run('numbers.repeating_decimal', '1/6'), (0, '1', '6'))
        self.assertEqual(run('numbers.repeating_decimal', '22/7'), (3, '', '142857'))
        self.assertEqual(run('numbers.repeating_decimal', '1/4'), (0, '25', ''))
        self.assertEqual(run('numbers.repeating_to_fraction', 0, '1', '6'), Fraction(1, 6))
        self.assertEqual(run('numbers.repeating_to_fraction', 0, '', '9'), 1)
        rng = random.Random(9)
        for _ in range(150):
            x = Fraction(rng.randint(0, 5000), rng.randint(1, 400))
            ip, pre, rep = run('numbers.repeating_decimal', x)
            self.assertEqual(run('numbers.repeating_to_fraction', ip, pre, rep), x)
            self.assertEqual(len(rep), run('numbers.decimal_period', x))
            self.assertEqual(len(pre), run('numbers.decimal_preperiod', x))
            f = x - math.floor(x)
            ld = self.long_division(f, 60)
            self.assertEqual(ld, (pre + rep * 60)[:60] if rep else (pre + '0' * 60)[:60])
            self.assertEqual(run('numbers.decimal_expansion', x, 30)[-30:], self.long_division(f, 30))
            for k in (1, 2, 5, 31, 59):
                self.assertEqual(run('numbers.decimal_digit', x, k), int(ld[k - 1]))
        self.assertEqual(run('numbers.decimal_digit', '1/7', 10 ** 18), int(('142857'[(10 ** 18 - 1) % 6])))
        ex = run('numbers.decimal_expansion', '1/3', 50000)
        self.assertEqual(len(ex), 50002)


class Diophantine(unittest.TestCase):
    def test_linear(self):
        self.assertEqual(run('numbers.lin_diophantine', 3, 5, 7), (4, -1, 5, -3))
        self.assertEqual(run('numbers.lin_diophantine', 6, 4, 3), ())
        self.assertEqual(run('numbers.lin_diophantine', 0, 4, 8), (0, 2, 1, 0))
        self.assertEqual(run('numbers.lin_diophantine', 0, 0, 3), ())
        with self.assertRaises(NotCovered):
            run('numbers.lin_diophantine', 0, 0, 0)
        for a in range(-9, 10):
            for b in range(-9, 10):
                for c in range(-12, 13):
                    if a == 0 and b == 0:
                        continue
                    sol = run('numbers.lin_diophantine', a, b, c)
                    brute = [(x, y) for x in range(-40, 41) for y in range(-40, 41) if a * x + b * y == c]
                    if not sol:
                        self.assertEqual(brute, [], (a, b, c))
                        continue
                    x0, y0, dx, dy = sol
                    g = math.gcd(a, b)
                    self.assertEqual((abs(dx), abs(dy)), (abs(b) // g, abs(a) // g))
                    self.assertEqual(a * dx + b * dy, 0)
                    mine = {(x0 + dx * t, y0 + dy * t) for t in range(-80, 81)}
                    self.assertTrue(set(brute) <= mine, (a, b, c))
                    for t in range(3):
                        self.assertEqual(a * (x0 + dx * t) + b * (y0 + dy * t), c)

    def test_nonneg_count(self):
        self.assertEqual(run('numbers.lin_diophantine_nonneg_count', 3, 5, 100), 7)
        for a in range(1, 10):
            for b in range(1, 10):
                for c in range(0, 60):
                    brute = sum(1 for x in range(c + 1) for y in range(c + 1) if a * x + b * y == c)
                    self.assertEqual(run('numbers.lin_diophantine_nonneg_count', a, b, c), brute, (a, b, c))

    def test_pell(self):
        self.assertEqual(run('numbers.pell', 61), (1766319049, 226153980))
        self.assertEqual(run('numbers.pell', 2), (3, 2))
        self.assertEqual(run('numbers.pell', 4), ())
        self.assertEqual(run('numbers.pell_negative', 13), (18, 5))
        self.assertEqual(run('numbers.pell_negative', 3), ())
        self.assertEqual(run('numbers.pell_solutions', 2, 3), [(3, 2), (17, 12), (99, 70)])
        for d in range(2, 400):
            if math.isqrt(d) ** 2 == d:
                continue
            x, y = run('numbers.pell', d)
            self.assertEqual(x * x - d * y * y, 1)
            self.assertTrue(x > 0 and y > 0)
            if y < 20000:      # the fundamental solution has the least y: nothing smaller
                for yy in range(1, y):
                    t = 1 + d * yy * yy
                    self.assertNotEqual(math.isqrt(t) ** 2, t)
            neg = run('numbers.pell_negative', d)
            if neg:
                self.assertEqual(neg[0] ** 2 - d * neg[1] ** 2, -1)
            else:
                self.assertEqual(run('numbers.pell_search', d, -1, 3000), [])
        self.assertEqual(run('numbers.pell_search', 2, -1, 100), [(1, 1), (7, 5), (41, 29)])

    def test_pythagorean(self):
        self.assertEqual(run('numbers.pythagorean_triples', 20), [(3, 4, 5), (5, 12, 13), (6, 8, 10), (8, 15, 17), (9, 12, 15), (12, 16, 20)])
        self.assertEqual(run('numbers.pythagorean_triples', 30, True), [(3, 4, 5), (5, 12, 13), (7, 24, 25), (8, 15, 17), (20, 21, 29)])
        for N_ in (0, 5, 50, 130):
            brute = sorted((a, b, c) for c in range(1, N_ + 1) for b in range(1, c) for a in range(1, b) if a * a + b * b == c * c)
            self.assertEqual(run('numbers.pythagorean_triples', N_), brute)
            self.assertEqual(run('numbers.pythagorean_triple_count', N_), len(brute))
        self.assertEqual(run('numbers.pythagorean_triple_count', 100), 52)


class SumsOfSquares(unittest.TestCase):
    def test_two_squares(self):
        self.assertEqual(run('numbers.two_squares', 50), [(1, 7), (5, 5)])
        self.assertEqual(run('numbers.two_squares', 0), [(0, 0)])
        self.assertEqual(run('numbers.two_squares', 3), [])
        for n in range(0, 1500):
            brute = sorted((a, b) for a in range(0, 40) for b in range(a, 40) if a * a + b * b == n)
            self.assertEqual(run('numbers.two_squares', n), brute, n)
            self.assertEqual(run('numbers.is_sum_of_two_squares', n), bool(brute))
            ordered = sum(1 for x in range(-40, 41) for y in range(-40, 41) if x * x + y * y == n)
            self.assertEqual(run('numbers.r2', n), ordered, n)
        # large: certificate (each pair squares back), and the count equals the divisor-formula count
        n = 5 ** 3 * 13 * 17 * 29 * 37 * 2 ** 5 * 3 ** 4
        reps = run('numbers.two_squares', n)
        self.assertTrue(all(0 <= a <= b and a * a + b * b == n for a, b in reps))
        self.assertEqual(len(set(reps)), len(reps))

    def test_two_squares_count_matches_r2(self):
        for n in range(1, 3000, 7):
            reps = run('numbers.two_squares', n)
            total = 0
            for a, b in reps:                         # signs and order
                total += len({(sa * a, sb * b) for sa in (1, -1) for sb in (1, -1)} | {(sa * b, sb * a) for sa in (1, -1) for sb in (1, -1)})
            self.assertEqual(total, run('numbers.r2', n), n)

    def test_three_four_squares(self):
        for n in range(0, 300):
            has = any(a * a + b * b + c * c == n for a in range(18) for b in range(18) for c in range(18))
            self.assertEqual(run('numbers.is_sum_of_three_squares', n), has, n)
        for n in range(1, 120):
            brute = sum(1 for a in range(-11, 12) for b in range(-11, 12) for c in range(-11, 12) for d in range(-11, 12) if a * a + b * b + c * c + d * d == n)
            self.assertEqual(run('numbers.r4', n), brute, n)
        for n in list(range(0, 200)) + [10 ** 15 + 7, 4 ** 20 * 7, 123456789012345678, 8 * 10 ** 17 + 7]:
            a, b, c, d = run('numbers.four_squares', n)
            self.assertEqual(a * a + b * b + c * c + d * d, n)
            self.assertTrue(0 <= a <= b <= c <= d)


class FrobeniusFloorSum(unittest.TestCase):
    def test_frobenius(self):
        self.assertEqual(run('numbers.frobenius', 3, 5), 7)
        self.assertEqual(run('numbers.frobenius', 6, 9, 20), 43)
        self.assertEqual(run('numbers.frobenius', 4, 6), ())
        self.assertEqual(run('numbers.frobenius', 1, 7), -1)
        self.assertEqual(run('numbers.frobenius_count', 3, 5), 4)
        rng = random.Random(10)
        for _ in range(150):
            coins = sorted(rng.sample(range(2, 30), rng.randint(2, 4)))
            lim = 3000
            rep = [False] * (lim + 1)
            rep[0] = True
            for v in range(1, lim + 1):
                rep[v] = any(v >= c and rep[v - c] for c in coins)
            gaps = [v for v in range(lim + 1) if not rep[v]]
            if math.gcd(*coins) != 1:
                self.assertEqual(run('numbers.frobenius', *coins), ())
                continue
            self.assertEqual(run('numbers.frobenius', *coins), max(gaps) if gaps else -1, coins)
            self.assertEqual(run('numbers.frobenius_count', *coins), len(gaps), coins)

    def test_floor_sum(self):
        self.assertEqual(run('numbers.floor_sum', 10, 7, 3, 2), sum((3 * i + 2) // 7 for i in range(10)))
        rng = random.Random(11)
        for _ in range(500):
            n, m, a, b = rng.randint(0, 40), rng.randint(1, 30), rng.randint(-60, 60), rng.randint(-60, 60)
            self.assertEqual(run('numbers.floor_sum', n, m, a, b), sum((a * i + b) // m for i in range(n)), (n, m, a, b))
        # a large case against the identity floor_sum(n) = floor_sum(n-1, b+a) + floor(b/m) (shifting the start by one term)
        n, m, a, b = 10 ** 9, 998244353, 123456789, 987654321
        self.assertEqual(run('numbers.floor_sum', n, m, a, b), b // m + run('numbers.floor_sum', n - 1, m, a, a + b))


class Registry(unittest.TestCase):
    def test_every_operation_is_documented_with_a_bound(self):
        mine = {k: v for k, v in registry().items() if k.startswith('numbers.')}
        self.assertGreaterEqual(len(mine), 100)
        for name, rec in mine.items():
            self.assertTrue(rec['doc'], name)
            self.assertTrue(rec['bound'], name)
            self.assertTrue(rec['kind'], name)

    def test_only_standard_library_imports(self):
        src = (Path(__file__).resolve().parent / 'igc' / 'mathbackend' / 'numbers.py').read_text(encoding='utf-8')
        mods = {line.split()[1].split('.')[0] for line in src.splitlines() if line.startswith(('import ', 'from '))}
        self.assertEqual(mods, {'heapq', 'math', 'fractions', 'tools'})


if __name__ == '__main__':
    unittest.main()
