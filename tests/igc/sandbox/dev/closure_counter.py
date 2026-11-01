def make():
    n = 0
    def inc(k=1):
        nonlocal n
        n += k
        return n
    def get():
        return n
    return inc, get
inc, get = make()
inc(); inc(5)
print(get())
print([f() for f in [lambda i=i: i for i in range(3)]])
print([f() for f in [lambda: i for i in range(3)]])
