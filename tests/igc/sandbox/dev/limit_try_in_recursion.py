# expect: RecursionLimit
def f():
    try:
        f()
    except ValueError:
        pass
f()
