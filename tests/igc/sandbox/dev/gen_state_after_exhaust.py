g = (x for x in [1])
print(list(g), list(g), next(g, 'd'))
def f():
    return (i for i in range(3))
a, b = f(), f()
print(next(a), next(b), next(a), list(b), list(a))
h = (x for x in [1, 2, 3])
print(next(h))
del h
k = [(x for x in [i]) for i in range(2)]
print([next(g2) for g2 in k])
print(sum(x for x in (y for y in (z for z in [1, 2]))))
