from fractions import Fraction
try:
    print(format(Fraction(1, 2), '.2f'))
except TypeError as e:
    print(e)
