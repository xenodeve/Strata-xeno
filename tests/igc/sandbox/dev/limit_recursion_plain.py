# expect: RecursionLimit
def f():
    f()
f()
