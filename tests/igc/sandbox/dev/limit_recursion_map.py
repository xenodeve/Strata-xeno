# expect: RecursionLimit
def f(n):
    return list(map(f, [n + 1]))
f(0)
