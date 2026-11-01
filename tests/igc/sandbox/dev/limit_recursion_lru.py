# expect: RecursionLimit
from functools import lru_cache
@lru_cache(None)
def f(n):
    return f(n + 1)
f(0)
