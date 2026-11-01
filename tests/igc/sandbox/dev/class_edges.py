from functools import lru_cache, reduce
from heapq import heappush, heappop
from collections import defaultdict, deque


class C:
    def __init__(self, a, b=2, *rest, c=3, **more):
        self.a, self.b = a, b
        self.rest = rest
        self.c = c
        self.more = more

    def add(self, x, y=1):
        return x + y + self.a

    def many(self, *xs, **kw):
        return len(xs), sorted(kw)

    def key(self, v):
        return -v


c = C(1, 5, 6, 7, c=8, d=9)
print(c.a, c.b, c.rest, c.c, c.more)
print(c.add(1), c.add(1, 2), c.add(y=3, x=4), c.add(*[1, 2]), c.add(**{'x': 1, 'y': 5}))
print(c.many(1, 2, k=3), c.many(*[1, 2, 3], **{'z': 1}))
print(sorted([3, 1, 2], key=c.key), max([3, 1, 2], key=c.key), reduce(lambda a, b: a + c.add(b), [1, 2], 0))
print(list(filter(c.key, [0, 1])), list(map(c.add, [1, 2], [3, 4])))
print(issubclass(C, object), issubclass(C, C), issubclass(int, object))
print(c == 1, 1 == c, c != 1, c == None, c == 'x', c == [c], [c] == [c], (c,) == (c,), c in [c], c in (1, 2), c.add == 1, 1 == c.add, C == 1, 1 == C, C == C, C != int)
print(c.add == c.add, c.add != c.key, C(1).add == c.add, hash(c.add) == hash(c.add), {c.add: 1}[c.add], len({c.add, c.add, c.key}))
print(hash(C) == hash(C), {C: 1}[C], hash(c) == hash(c), type(c) == C, type(c) != int, type(C) == type, type(C) is type)


class Heap:
    def __init__(self):
        self.h = []

    def push(self, p, item):
        heappush(self.h, (p, len(self.h), item))

    def pop(self):
        return heappop(self.h)[2]


h = Heap()
h.push(2, C(2))
h.push(1, C(1))
print(h.pop().a, h.pop().a)
try:
    heappush(h.h, (1, C(1)))
    heappush(h.h, (1, C(1)))
except TypeError:
    print('TypeError')

groups = defaultdict(list)
for i in range(3):
    groups[i % 2].append(C(i))
print({k: [o.a for o in v] for k, v in groups.items()})
q = deque([C(1), C(2)])
q.rotate(1)
print(q[0].a, [o.a for o in q])


class Bad:
    def __init__(self, x):
        if x < 0:
            raise ValueError('negative')
        self.x = x


for v in (1, -1):
    try:
        print(Bad(v).x)
    except ValueError as e:
        print('ValueError', e)
    finally:
        print('done')
try:
    Bad()
except TypeError:
    print('TypeError')
try:
    Bad(1, 2)
except TypeError:
    print('TypeError')
try:
    Bad(1).x.y
except AttributeError:
    print('AttributeError')
try:
    c.nope += 1
except AttributeError:
    print('AttributeError')
try:
    del c.nope
except AttributeError:
    print('AttributeError')


class Cycle:
    def __init__(self):
        self.me = self
        self.others = [self, {'k': self}]


a, b = Cycle(), Cycle()
a.o, b.o = b, a
print(a.o.o is a, a.me.me.me is a, a.others[1]['k'] is a)
n = None
for i in range(50000):
    node = Bad(i)
    node.next = n
    n = node
print(n.x, n.next.x)
n = None
print('freed')


class Memo:
    def __init__(self):
        self.count = 0

    @lru_cache(None)
    def f(self, n):
        self.count += 1
        return n if n < 2 else self.f(n - 1) + self.f(n - 2)


m1, m2 = Memo(), Memo()
print(m1.f(20), m1.count, m2.f(10), m2.count, m1.f(20), m1.count)
g = m1.f
print(g(5), m1.count)


class Shadow:
    def __init__(self):
        self.max = 3
        self.len = lambda x: 'mine'

    def min(self, a, b):
        return max(a, b)

    def sum(self):
        return min(self.max, 2), self.len(1), len([1])


s = Shadow()
print(s.min(1, 2), s.sum(), min(1, 2), sum([1]))


def outer():
    total = 0

    class Acc:
        def add(self, n):
            nonlocal total
            total += n
            return total

    x = Acc()
    x.add(2)
    return x.add(3), total


print(outer())


class Twice:
    def v(self):
        return 1

    def v(self):
        return 2


print(Twice().v())
t = Twice()
t.v = 5
print(t.v)
del t.v
print(t.v())
print([i for i in range(3) if Twice().v() > i], {k: Twice().v() for k in 'ab'}, f'{Twice().v()}', f'{c.a + 1:03d}')
lst = [C(i) for i in range(3)]
lst[1].a += 10
lst[2].rest += (1,)
for o in lst:
    o.a *= 2
print([o.a for o in lst], lst[2].rest)
c.items = [1]
c.items[0] += 1
c.d = {'a': 1}
c.d['a'] += 1
c.d['b'] = 2
print(c.items, c.d)
x, c.a, [y, c.b] = 1, 2, [3, 4]
print(x, c.a, y, c.b)
for c.a in range(3):
    pass
print(c.a)
print((lambda self: self.a)(c), callable(C), callable(c.add), isinstance(c.add, object), isinstance(C, type), isinstance(C, object), isinstance(c, (int, C)))
