# expect: RecursionLimit
g = iter([1])
for i in range(100000):
    g = (x for x in g)
print(list(g))
