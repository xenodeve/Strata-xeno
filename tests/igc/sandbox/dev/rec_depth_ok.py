def f(n):
    return 0 if n == 0 else 1 + f(n - 1)
print(f(150))
def g(n):
    return 0 if n == 0 else 1 + max(g(n - 1) for _ in [1])
print(g(50))
from functools import lru_cache
@lru_cache(None)
def h(n):
    return 0 if n == 0 else 1 + h(n - 1)
print(h(150))
def k(n):
    return 0 if n == 0 else 1 + sum(map(k, [n - 1]))
print(k(60))
def s(n):
    return 0 if n == 0 else 1 + sorted([n - 1], key=s)[0] * 0 + s(n - 1) if False else (0 if n == 0 else 1 + s(n - 1))
print(s(100))
