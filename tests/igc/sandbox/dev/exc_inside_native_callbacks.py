def bad(x):
    raise ValueError('in key %d' % x)
for fn in [lambda: sorted([1, 2], key=bad), lambda: max([1], key=bad), lambda: list(map(bad, [1])), lambda: list(filter(bad, [2])), lambda: sum(bad(x) for x in [3]), lambda: ''.join(bad(x) for x in [4]), lambda: dict((bad(x), 1) for x in [5]), lambda: any(bad(x) for x in [6]), lambda: [1, 2].sort(key=bad)]:
    try:
        print(fn())
    except ValueError as e:
        print(e)
from functools import reduce, lru_cache
from collections import defaultdict
try:
    reduce(lambda a, b: bad(b), [1, 2])
except ValueError as e:
    print(e)
d = defaultdict(lambda: bad(9))
try:
    d['k']
except ValueError as e:
    print(e, len(d))
@lru_cache(None)
def c(x):
    if x == 0:
        raise KeyError('zero')
    return x
for v in [0, 0, 1, 1]:
    try:
        print(c(v))
    except KeyError as e:
        print(e)
def deep(n):
    if n == 0:
        raise ValueError('bottom')
    return sum(deep(k) for k in [n - 1])
try:
    deep(40)
except ValueError as e:
    print(e)
def deep2(n):
    if n == 0:
        raise ValueError('bottom2')
    return max(map(deep2, [n - 1]))
try:
    deep2(40)
except ValueError as e:
    print(e)
def recover(n):
    try:
        return deep(n)
    except ValueError:
        return recover(n - 1) if n > 35 else 'recovered'
print(recover(40))
