import heapq
import bisect
from heapq import heappush, heappop, heapify, heapreplace, heappushpop, nlargest, nsmallest
from bisect import bisect_left, bisect_right, insort, insort_left, insort_right


def numbers(seed, count, top):
    out = []
    x = seed
    for _ in range(count):
        x = (x * 1103515245 + 12345) % 2147483648
        out.append(x % top)
    return out


for seed in range(1, 40):
    items = numbers(seed, 60, 50)
    h = []
    for x in items:
        heappush(h, x)
    print(h)
    print([heappop(h) for _ in range(30)], h)
    g = list(items)
    heapify(g)
    print(g)
    print(heapreplace(g, 7), heappushpop(g, 3), heappushpop(g, 90), g)
    print(nlargest(5, items), nsmallest(5, items), nlargest(3, items, key=lambda v: -v), nsmallest(3, items, key=lambda v: (v % 7, v)))

pairs = []
for i, x in enumerate(numbers(5, 40, 9)):
    heapq.heappush(pairs, (x, i, str(i)))
print([heapq.heappop(pairs) for _ in range(40)])
print(heappushpop([], 4))

for seed in range(1, 30):
    a = sorted(numbers(seed, 50, 30))
    t = tuple(a)
    for x in numbers(seed + 100, 30, 34):
        print(bisect_left(a, x), bisect_right(a, x), bisect.bisect(a, x), bisect_left(t, x, 5), bisect_right(a, x, 3, 20), bisect_left(a, x, hi=10), bisect_right(a, x, lo=2))
    b = []
    for x in numbers(seed, 30, 12):
        insort(b, x)
    print(b)
    c = []
    for x in numbers(seed, 30, 12):
        insort_left(c, x)
    d = [(1, 'a')]
    insort_right(d, (1, 'a'))
    insort_left(d, (0, 'z'))
    print(c, d)

words = sorted(['pear', 'fig', 'apple', 'kiwi'], key=len)
print(bisect_left(words, 4, key=len), bisect_right(words, 4, key=len))
insort(words, 'plum', key=len)
insort_left(words, 'date', key=len)
print(words)

for bad in (lambda: heappop([]), lambda: heapreplace([], 1), lambda: heappush((), 1), lambda: heapify(()), lambda: bisect_left([1], 1, -1), lambda: insort((1,), 2), lambda: heappush([1], 'a')):
    try:
        bad()
    except Exception as e:
        print(type(e).__name__)
