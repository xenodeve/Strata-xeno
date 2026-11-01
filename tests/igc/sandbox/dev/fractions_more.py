from fractions import Fraction as F
print(F(), F(5), F(6, 8), F(-1, 2), F(1, -2), F(-1, -2), F(0, 5), F('1/3'), F(' 2/4 '), F('0.5'), F('-1.25'), F('1e2'), F('1e-2'), F('2'), F('.5'), F('1_000'), F(0.25), F(0.1), F(F(1, 2)), F(F(1, 2), F(1, 3)), F(True), F(1.0))
a, b = F(1, 2), F(1, 3)
print(a + b, a - b, a * b, a / b, a // b, a % b, -a, +a, abs(-a), a ** 2, a ** -2, a ** 0, F(4, 9) ** 0.5, a ** F(2), 2 ** F(2), 2 ** F(-1), 2 ** F(1, 2), a + 1, 1 + a, a - 1, 1 - a, a * 2, 2 * a, a / 2, 2 / a, a // 1, 7 // a, a % 1, 7 % a, a + 0.5, 0.5 + a, a * 0.5, a / 0.5, a ** 2.0, divmod(F(7, 2), F(1, 3)), divmod(7, F(1, 3)), divmod(F(7, 2), 2))
print(a == b, a != b, a < b, a > b, a <= a, a >= b, a == 0.5, a == 1, F(2) == 2, 2 == F(2), a < 1, 1 > a, a < 0.75, a > 0.25, 0.75 > a, F(1, 3) == 1 / 3, float('inf') > a, a < float('inf'), a > float('-inf'), a == float('nan'), a < float('nan'), hash(F(1, 2)) == hash(0.5), hash(F(2)) == hash(2), hash(F(1, 3)) == hash(F(2, 6)), {F(1, 2): 1}[0.5], {0.5: 1}[F(1, 2)], F(1, 2) in [0.5], 0.5 in {F(1, 2)})
print(float(a), int(F(7, 2)), int(F(-7, 2)), bool(F(0)), bool(a), round(F(5, 2)), round(F(7, 2)), round(F(-5, 2)), round(F(1, 3), 2), round(F(1234), -2), round(F(5, 3), 0), str(a), repr(a), str(F(2)), repr(F(2)), f'{a}', '%s' % a, a.numerator, a.denominator, F(5).numerator, F(5).denominator, a.limit_denominator(1), F(3.141592653589793).limit_denominator(100), F(1, 3).limit_denominator(1000), a.as_integer_ratio(), sum([F(1, 2), F(1, 3)]), max(a, b), sorted([a, b, 1, 0.1]), min(F(1, 2), 0.25), sum([F(1, 2)] * 2), [a, b].index(b), isinstance(a, F), type(a).__name__, type(a + b).__name__, type(a + 1).__name__, type(a + 0.5).__name__, type(a ** 2).__name__, type(a // b).__name__)
import math
print(math.floor(F(7, 2)), math.ceil(F(7, 2)), math.trunc(F(-7, 2)), math.floor(F(-7, 2)), math.ceil(F(-7, 2)), math.sqrt(F(1, 4)), math.isqrt(9), F(10 ** 30, 3), F(1, 10 ** 30) + F(1, 10 ** 30), float(F(10 ** 400, 10 ** 399)), F(2 ** 100, 3 ** 50) > 1)
for bad in [lambda: F(1, 0), lambda: F('a'), lambda: F('1/0'), lambda: F(''), lambda: F(None), lambda: a / 0, lambda: a // 0, lambda: a % 0, lambda: F(0) ** -1, lambda: F(float('nan')), lambda: F(float('inf')), lambda: a + 'x', lambda: 'x' + a, lambda: a < 'x', lambda: F(1.5, 2), lambda: a.limit_denominator(0), lambda: f'{a:.2f}']:
    try:
        print(bad())
    except (ZeroDivisionError, ValueError, TypeError, OverflowError) as e:
        print(type(e).__name__, e)
