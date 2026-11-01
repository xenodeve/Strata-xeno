# expect: RecursionLimit
def f():
    try:
        f()
    finally:
        pass
f()
