def outer():
    v = 1
    def mid():
        def inner():
            return v
        return inner
    v = 2
    return mid()()
print(outer())
def counters():
    fs = []
    for i in range(3):
        def f(i=i):
            return i
        fs.append(f)
    return [f() for f in fs]
print(counters())
def late():
    fs = []
    for i in range(3):
        fs.append(lambda: i)
    i = 10
    return [f() for f in fs]
print(late())
def nl():
    a = 1
    def b():
        nonlocal a
        a += 1
        def c():
            nonlocal a
            a *= 10
        c()
    b()
    return a
print(nl())
def unbound_cell():
    def inner():
        return later
    try:
        inner()
    except NameError as e:
        print(type(e).__name__, e)
    later = 1
    return inner()
print(unbound_cell())
def param_cell(x):
    def g():
        return x
    x += 1
    return g()
print(param_cell(1))
def del_cell():
    y = 1
    def g():
        return y
    del y
    try:
        return g()
    except NameError as e:
        return str(e)
print(del_cell())
def recursive_inner(n):
    def go(k):
        return 1 if k == 0 else k * go(k - 1)
    return go(n)
print(recursive_inner(5))
def global_in_nested():
    def h():
        global gv
        gv = 5
    h()
    return gv
print(global_in_nested(), gv)
