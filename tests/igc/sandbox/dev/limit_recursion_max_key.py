# expect: RecursionLimit
def f(n):
    return max([n], key=lambda x: f(x + 1))
f(0)
