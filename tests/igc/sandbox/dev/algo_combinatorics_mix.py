from itertools import permutations, combinations, product
from math import comb, factorial
print(sum(1 for p in permutations(range(5)) if all(p[i] != i for i in range(5))), sum(sum(c) == 10 for c in combinations(range(1, 8), 3)), sum(1 for a, b, c in product(range(1, 7), repeat=3) if a + b + c == 10), comb(10, 3), factorial(10) // (factorial(3) * factorial(7)))
