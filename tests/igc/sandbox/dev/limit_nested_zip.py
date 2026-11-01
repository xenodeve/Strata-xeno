# expect: RecursionLimit
g = [1]
for i in range(100000):
    g = zip(g)
print(list(g))
