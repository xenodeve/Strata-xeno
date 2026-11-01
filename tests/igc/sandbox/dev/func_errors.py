def f(a, b):
    return a + b
for call in [lambda: f(1), lambda: f(1, 2, 3), lambda: f(1, c=2), lambda: f(1, a=2)]:
    try:
        call()
    except TypeError as e:
        print('TypeError', e)
