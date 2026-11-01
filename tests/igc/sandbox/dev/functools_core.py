from functools import reduce, lru_cache, cache
print(reduce(lambda a, b: a * b, [1, 2, 3, 4]), reduce(lambda a, b: a + b, [], 0))
@lru_cache(maxsize=None)
def fib(n):
    return n if n < 2 else fib(n - 1) + fib(n - 2)
print(fib(90))
@cache
def sq(x):
    print('compute', x)
    return x * x
print(sq(3), sq(3))
