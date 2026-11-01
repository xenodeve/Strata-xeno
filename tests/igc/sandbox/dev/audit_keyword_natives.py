import math
print(round(number=1.256, ndigits=1), round(1.256, ndigits=2), round(number=2), pow(base=2, exp=3), pow(2, exp=3), pow(2, 3, mod=5), pow(base=2, exp=3, mod=5), math.isclose(a=1, b=1), math.isclose(1, b=1.0), math.isclose(1, 2, rel_tol=1.0))
for fn in [lambda: round(), lambda: pow(2), lambda: pow(), lambda: math.isclose(1), lambda: round(1, 2, 3), lambda: round(x=1), lambda: pow(1, 2, z=3), lambda: 'aaa'.replace('a', 'b', count=1), lambda: 'a'.split(sep=',', x=1)]:
    try:
        print(fn())
    except TypeError as e:
        print(type(e).__name__)
