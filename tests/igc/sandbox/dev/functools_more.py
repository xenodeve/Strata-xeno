from functools import reduce, lru_cache, cache
import functools
print(reduce(lambda a, b: a + b, 'abc'), reduce(lambda a, b: a + b, [1], 2), reduce(lambda a, b: a + b, [5]), reduce(max, [1, 3, 2]), reduce(lambda a, b: a * b, range(1, 6), 1), functools.reduce(lambda a, b: b, iter([1, 2])))
try:
    reduce(lambda a, b: a, [])
except TypeError as e:
    print(e)
calls = []
@lru_cache(maxsize=2)
def f(x):
    calls.append(x)
    return x * 2
print(f(1), f(2), f(1), f(3), f(2), f(1), calls)
f.cache_clear()
print(f(1), calls)
@lru_cache
def g(a, b=2, *c, d=3):
    calls.append('g')
    return a + b + d
print(g(1), g(1), g(1, 2), g(a=1), g(1, b=2), g(1, d=3), g(1, 2, 3))
@functools.lru_cache(None)
def h(n):
    return 1 if n < 2 else n * h(n - 1)
print(h(20), h(10))
@functools.cache
def k(n):
    return n
print(k(1), k(1))
@lru_cache(maxsize=0)
def z(x):
    calls.append('z')
    return x
print(z(1), z(1), calls.count('z'))
try:
    f([1])
except TypeError as e:
    print(e)
@lru_cache()
def w(x=1):
    calls.append('w')
    return x
print(w(), w(), w(1), calls.count('w'))
@cache
def fibo(n):
    return n if n < 2 else fibo(n - 1) + fibo(n - 2)
print(fibo(150))
