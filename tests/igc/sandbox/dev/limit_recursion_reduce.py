# expect: RecursionLimit
from functools import reduce
def f(a, b):
    return reduce(f, [a, b + 1])
f(0, 0)
