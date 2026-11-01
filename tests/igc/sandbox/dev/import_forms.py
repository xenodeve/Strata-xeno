import math as m, itertools as it
from math import floor as fl, pi
from collections import (Counter, deque,)
import functools
import typing
print(m.floor(1.5), list(it.islice(it.count(), 2)), fl(2.5), pi, Counter('a'), deque([1]), functools.reduce(lambda a, b: a + b, [1, 2]))
def f():
    import math
    from itertools import chain as c
    return math.ceil(1.2), list(c([1]))
print(f())
if True:
    import fractions
print(fractions.Fraction(1, 2))
try:
    import surely_missing_module as np
except ImportError as e:
    print(type(e).__name__, e)
try:
    from surely_missing_module import path
except ModuleNotFoundError as e:
    print(e)
