def is_prime(n):
    return n > 1 and all(n % d for d in range(2, int(n ** 0.5) + 1))
def divisors(n):
    return sorted(d for i in range(1, int(n ** 0.5) + 1) if n % i == 0 for d in {i, n // i})
def totient(n):
    return sum(1 for k in range(1, n + 1) if __import_gcd(n, k) == 1)
from math import gcd as __import_gcd
print([n for n in range(50) if is_prime(n)], divisors(360), totient(36), [n for n in range(2, 500) if sum(divisors(n)[:-1]) == n], pow(2, 10 ** 18, 10 ** 9 + 7), pow(3, 10 ** 100, 97))
