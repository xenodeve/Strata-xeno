# expect: RecursionLimit
d = {}
for i in range(100000):
    d = {'k': d}
print(d)
