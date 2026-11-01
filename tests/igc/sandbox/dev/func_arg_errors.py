def a(x, y):
    pass
def b(x, y=1):
    pass
def c(*, k):
    pass
def d():
    pass
def e(x, *, k, m):
    pass
def outer():
    def inner(p):
        pass
    return inner
g = lambda q: q
for fn in [lambda: a(), lambda: a(1), lambda: a(1, 2, 3), lambda: b(), lambda: b(1, 2, 3), lambda: b(1, z=2), lambda: b(1, x=2), lambda: c(), lambda: c(1, k=2), lambda: d(1), lambda: d(k=1), lambda: e(1), lambda: e(1, k=1), lambda: e(), lambda: outer()(), lambda: outer()(1, 2), lambda: g(), lambda: a(*[1, 2, 3]), lambda: a(**{'x': 1}), lambda: a(1, **{'x': 2, 'y': 3}), lambda: a(**{'x': 1}, **{'x': 2}), lambda: a(**5), lambda: a(*5), lambda: a(1, 2, **{1: 2}), lambda: (1)(), lambda: 'a'(), lambda: None(), lambda: [1]()]:
    try:
        print(fn())
    except TypeError as e2:
        print(e2)
