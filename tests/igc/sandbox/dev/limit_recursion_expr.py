# expect: RecursionLimit
def f(n):
    return 1 + (2 * (3 - f(n + 1)))
print(f(0))
