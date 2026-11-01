from fractions import Fraction as F
rate = F(5, 100)
p = F(1000)
for y in range(3):
    p *= 1 + rate
print(p, float(p), round(float(p), 2), round(p, 2), f'{float(p):,.2f}', p.limit_denominator(10))
h = sum(F(1, k) for k in range(1, 11))
print(h, h.numerator, h.denominator, float(h), sum(F(1, 2 ** k) for k in range(20)), F(3, 4) ** 3, 1 - F(1, 3) - F(1, 6), max(F(1, 3), F(2, 7)), sorted([F(1, 2), F(1, 3), F(3, 4)]))
