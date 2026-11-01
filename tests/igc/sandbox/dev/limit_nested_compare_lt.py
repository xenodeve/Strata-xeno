# expect: RecursionLimit
a = []
b = [1]
for i in range(100000):
    a = [a]
    b = [b]
print(a < b)
