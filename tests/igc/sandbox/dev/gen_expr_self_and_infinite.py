from itertools import count
print(next(x for x in count() if x * x > 50))
print(any(x > 3 for x in count()), all(x < 3 for x in count()))
g = (x for x in count())
print(next(g), next(g))
def primes():
    return (n for n in count(2) if all(n % d for d in range(2, n)))
p = primes()
print([next(p) for _ in range(5)])
try:
    g2 = (next(g2) for _ in [1])
    next(g2)
except ValueError as e:
    print(e)
