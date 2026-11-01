# expect: RecursionLimit
def f(n):
    return [f(n + 1) for _ in [1]]
f(0)
