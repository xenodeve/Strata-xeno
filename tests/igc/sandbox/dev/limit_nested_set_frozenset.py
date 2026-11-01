# expect: RecursionLimit
d = frozenset()
for i in range(100000):
    d = frozenset([d])
print(d)
