# expect: RecursionLimit
def a(n):
    return b(n)
def b(n):
    return a(n)
a(0)
