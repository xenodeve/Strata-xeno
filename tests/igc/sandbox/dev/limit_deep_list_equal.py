# expect: RecursionLimit
a = []
b = []
for i in range(100000):
    a = [a]
    b = [b]
print(a == b)
