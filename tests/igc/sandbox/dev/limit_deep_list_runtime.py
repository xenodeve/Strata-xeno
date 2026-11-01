# expect: RecursionLimit
a = []
for i in range(100000):
    a = [a]
print(a)
