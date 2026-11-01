# expect: Unsupported
a = [3, 'x', 1]
try:
    a.sort()
except TypeError:
    print(a)
