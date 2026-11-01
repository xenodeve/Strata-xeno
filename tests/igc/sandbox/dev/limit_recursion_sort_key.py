# expect: RecursionLimit
def f(n):
    return sorted([n], key=f)
f(0)
