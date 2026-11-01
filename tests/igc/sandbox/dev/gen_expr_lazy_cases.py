log = []
def t(x):
    log.append(x)
    return x
g = (t(x) for x in [1, 2, 3])
print(log, next(g), log)
print(list(g), log)
print(list(g))
n = 0
h = (n + i for i in range(2))
n = 10
print(list(h))
src = [1, 2]
k = (x for x in src)
src = [9]
print(list(k))
m = [1, 2]
k2 = (x for x in m)
m.append(3)
print(list(k2))
def first():
    return next(x for x in range(5) if x > 2)
print(first())
print(next((x for x in [] ), 'none'))
print(any(x == 2 for x in [1, 2, 1 / 1]), 2 in (x for x in [1, 2]), 5 in (x for x in [1]))
c = (x for x in range(3))
print(1 in c, list(c))
try:
    (x for x in 5)
except TypeError as e:
    print(e)
bad = (1 / 0 for _ in [1])
try:
    next(bad)
except ZeroDivisionError as e:
    print('zde', e)
print(list(bad))
sq = (x * x for x in range(3))
print(sum(sq), sum(sq))
nested = (y for x in [[1], [2, 3]] for y in x if y > 1)
print(list(nested))
it = iter(x for x in 'ab')
print(next(it), list(it))
print(sorted((x for x in range(3)), reverse=True), list(zip((x for x in 'ab'), (y for y in [1, 2]))), list(map(str, (x for x in [1]))), dict(enumerate(x for x in 'a')))
