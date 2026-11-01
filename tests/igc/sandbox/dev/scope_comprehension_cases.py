x = 5
print([x for _ in range(2)], [y for y in range(x)])
def f():
    n = 3
    return [i * n for i in range(n)], {i: n for i in range(1)}, {n for i in range(1)}, sum(n for i in range(2))
print(f())
print([[j for j in range(i)] for i in range(3)])
print([(x, y) for x in range(2) for y in range(x, 2) if x != y])
print([x for x in [1, 2, 3] if x > 1 if x < 3])
print([a + b for a, b in [(1, 2)]], [a for a, *b in [(1, 2)]], {a: b for a, b in zip('ab', [1, 2])})
fs = [lambda: i for i in range(2)]
print([f2() for f2 in fs])
gs = [(lambda i=i: i)() for i in range(2)]
print(gs)
def g():
    data = [1, 2]
    return [d + e for d in data for e in data]
print(g())
i = 'outer'
_ = [i for i in range(2)]
print(i)
try:
    [undefined_name for q in [1]]
except NameError as e:
    print(e)
print([x for x in range(3)][::-1], len([0 for _ in range(10)]), [[]for _ in range(2)])
print(list(x * 2 for x in range(2)), tuple(x for x in 'ab'), set(x for x in [1]), dict((x, 1) for x in 'a'), sorted(x for x in [2, 1]), max(x for x in [1, 2]), ''.join(c for c in 'ab'))
