from fractions import Fraction
s = sum(Fraction(1, n) for n in range(1, 21))
print(s, float(s), s.numerator % 1000, s.denominator % 1000)
