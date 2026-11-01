def f():
    def g():
        nonlocal x
    return g
x = 1
