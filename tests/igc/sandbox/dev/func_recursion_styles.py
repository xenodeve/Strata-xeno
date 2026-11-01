def ack(m, n):
    return n + 1 if m == 0 else ack(m - 1, 1) if n == 0 else ack(m - 1, ack(m, n - 1))
print(ack(2, 2))
def flat(x):
    return [y for z in x for y in (flat(z) if isinstance(z, list) else [z])]
print(flat([1, [2, [3, [4]]]]))
def perms(s):
    return [s] if len(s) <= 1 else [c + p for i, c in enumerate(s) for p in perms(s[:i] + s[i + 1:])]
print(perms('abc'))
def depth(t):
    return 1 + max((depth(c) for c in t), default=0)
print(depth([[], [[]], [[[]]]]))
def tree_sum(t):
    return t if isinstance(t, int) else sum(map(tree_sum, t))
print(tree_sum([1, [2, 3], [[4]]]))
def collatz(n, steps=0):
    return steps if n == 1 else collatz(n // 2 if n % 2 == 0 else 3 * n + 1, steps + 1)
print(collatz(27 if False else 9))
def sum_digits(n):
    return n if n < 10 else n % 10 + sum_digits(n // 10)
print(sum_digits(10 ** 50 - 1))
def binsearch(a, x, lo=0, hi=None):
    hi = len(a) if hi is None else hi
    if lo >= hi:
        return -1
    mid = (lo + hi) // 2
    return mid if a[mid] == x else binsearch(a, x, mid + 1, hi) if a[mid] < x else binsearch(a, x, lo, mid)
print(binsearch(list(range(100)), 37), binsearch([], 1))
