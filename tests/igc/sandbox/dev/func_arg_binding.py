def f(a, b=1, *c, d, e=2, **g):
    return a, b, c, d, e, g
print(f(1, d=2), f(1, 2, 3, 4, d=5), f(1, d=2, e=3, z=4), f(a=1, b=2, d=3), f(*[1, 2], d=3), f(1, **{'d': 2}), f(*[1], **{'d': 2, 'x': 3}))
def k(a, *, b):
    return a, b
print(k(1, b=2))
def v(*args, **kw):
    return args, kw
print(v(), v(1), v(a=1), v(1, 2, a=1, b=2), v(*'ab'), v(*[1], *[2], 3), v(**{'a': 1}, **{'b': 2}), v(1, *[2], c=3, **{'d': 4}))
def d(a=[], b={}, c=()):
    a.append(1)
    return a, b, c
print(d(), d())
def default_eval():
    print('evaluated')
    return 1
def e(x=default_eval()):
    return x
print(e(), e())
n = 1
def late(x=n):
    return x
n = 2
print(late())
print((lambda: 1)(), (lambda x: x)(1), (lambda x, y=2: x + y)(1), (lambda *a: a)(1, 2), (lambda **k: k)(a=1), (lambda x, *y, z=1, **w: (x, y, z, w))(1, 2, z=3, q=4))
def noreturn():
    pass
def retempty():
    return
def retmany():
    return 1, 2
print(noreturn(), retempty(), retmany())
def doc():
    """docstring"""
    'also a string'
    ...
    return 5
print(doc())
