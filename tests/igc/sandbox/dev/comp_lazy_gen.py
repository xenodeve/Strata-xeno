def f(x):
    print('f', x)
    return x > 0
print(any(f(x) for x in [1, 2, 3]))
print(all(f(x) for x in [0, 1]))
g = (1 / x for x in [1, 0])
print(next(g))
try:
    print(next(g))
except ZeroDivisionError as e:
    print('caught', e)
