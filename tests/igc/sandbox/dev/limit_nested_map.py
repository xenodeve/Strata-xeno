# expect: RecursionLimit
g = [1]
for i in range(100000):
    g = map(abs, g)
print(list(g))
