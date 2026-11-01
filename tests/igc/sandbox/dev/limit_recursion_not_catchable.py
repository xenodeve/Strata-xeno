# expect: RecursionLimit
def f():
    return f()
try:
    f()
except RecursionError:
    print('caught')
