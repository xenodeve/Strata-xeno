# expect: RecursionLimit
from itertools import chain
g = [1]
for i in range(100000):
    g = chain(g)
print(list(g))
