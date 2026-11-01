# expect: MemoryLimit
from itertools import product
print(next(product([1], repeat=10 ** 12)))
