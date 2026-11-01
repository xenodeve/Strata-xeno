from functools import lru_cache
from typing import List


class Solution:
    """A LeetCode form: helpers, recursion through self, nested functions, state on self."""

    def __init__(self, base=10):
        self.base = base
        self.calls = 0
        self.seen = []

    def digits(self, n):
        self.calls += 1
        return [] if n == 0 else self.digits(n // self.base) + [n % self.base]

    @lru_cache(maxsize=None)
    def fib(self, n):
        self.calls += 1
        return n if n < 2 else self.fib(n - 1) + self.fib(n - 2)

    def run(self, nums: List[int], *, scale=2) -> int:
        total = 0

        def walk(i):
            nonlocal total
            if i == len(nums):
                return
            total += nums[i] * scale
            self.seen.append(nums[i])
            walk(i + 1)

        walk(0)
        return total

    def squares(self, n):
        return (i * i * self.base for i in range(n))


s = Solution()
print(s.digits(1234), s.calls)
print(s.fib(30), s.calls)
print(s.run([1, 2, 3], scale=3), s.seen)
print(list(s.squares(4)), sum(s.squares(3)))

t = Solution(base=2)
print(t.digits(10), t.calls, s.calls)
print(isinstance(t, Solution), isinstance(t, object), isinstance(1, Solution), type(t) is Solution, type(t) == type(s))
print(Solution.__name__, type(s).__name__, callable(Solution), callable(s.run), callable(s))


class Empty(object):
    pass


class Doc:
    "only a docstring"


e = Empty()
e.x = 1
e.x += 2
e.y: int = 5
e.items = [e.x, e.y]
print(e.x, e.y, e.items)
del e.y
try:
    print(e.y)
except AttributeError as err:
    print('AttributeError', err)
try:
    Empty(1)
except TypeError as err:
    print('TypeError', err)
print(isinstance(Doc(), Doc), isinstance(Doc(), Empty))


class Dsu:
    def __init__(self, n):
        self.parent = list(range(n))
        self.size = [1] * n

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a == b:
            return False
        if self.size[a] < self.size[b]:
            a, b = b, a
        self.parent[b] = a
        self.size[a] += self.size[b]
        return True


d = Dsu(6)
print([d.union(a, b) for a, b in [(0, 1), (2, 3), (1, 3), (0, 2)]], d.find(3) == d.find(0), max(d.size))


class Node:
    def __init__(self, value, next=None):
        self.value = value
        self.next = next


head = None
for i in range(5):
    head = Node(i, head)
out = []
while head:
    out.append(head.value)
    head = head.next
print(out)

a, b = Node(1), Node(1)
print(a == a, a == b, a != b, a is a, {a: 1, b: 2}[b], len({a, b, a}))
print(sorted([Node(3), Node(1), Node(2)], key=lambda n: n.value)[0].value)
fn = Dsu(3).find
print(fn(2), list(map(Dsu(3).find, [0, 1])), (lambda f: f(1))(Dsu(2).find))
m = Dsu(2)
print(m.find == m.find, m.find != m.union)
m.find = lambda x: x + 100
print(m.find(1), Dsu.find(Dsu(2), 1))


def make():
    class Local:
        def __init__(self):
            self.v = limit

        def get(self):
            return self.v + limit

    limit = 7
    return Local().get()


print(make())


class Rec:
    def down(self, n):
        return 0 if n == 0 else 1 + self.down(n - 1)


print(Rec().down(150))
try:
    Rec().down()
except TypeError as err:
    print('TypeError', err)
try:
    Rec().nope()
except AttributeError as err:
    print('AttributeError', err)
try:
    Rec.nope
except AttributeError as err:
    print('AttributeError', err)
try:
    Rec() < Rec()
except TypeError:
    print('TypeError')
try:
    len(Rec())
except TypeError as err:
    print('TypeError', err)


class Init:
    def __init__(self):
        return 1


try:
    Init()
except TypeError as err:
    print('TypeError', err)
print(bool(Rec()), not Rec())
