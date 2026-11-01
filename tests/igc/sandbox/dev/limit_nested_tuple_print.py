# expect: RecursionLimit
d = ()
for i in range(100000):
    d = (d,)
print(d)
