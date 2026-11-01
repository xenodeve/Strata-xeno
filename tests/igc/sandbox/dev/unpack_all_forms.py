a, b = 1, 2; print(a, b)
a, b = b, a; print(a, b)
[a, b] = [3, 4]; print(a, b)
(a, b) = 'xy'; print(a, b)
a, (b, c) = 1, (2, 3); print(a, b, c)
a, *b = 1, 2, 3; print(a, b)
*a, b = 1, 2, 3; print(a, b)
a, *b, c = 1, 2, 3, 4; print(a, b, c)
a, *b, c = 1, 2; print(a, b, c)
*a, = 1, 2; print(a)
[*a] = 'ab'; print(a)
a, *b = 'hello'; print(a, b)
a, b = {1: 2, 3: 4}; print(a, b)
a, b = iter([5, 6]); print(a, b)
a, b = (x * 2 for x in [1, 2]); print(a, b)
x = y = z = 0; print(x, y, z)
l = [0, 0]; i = 0; i, l[i] = 1, 5; print(i, l)
d = {}; d['a'], d['b'] = 1, 2; print(d)
for (p, q), r in [((1, 2), 3)]:
    print(p, q, r)
for i, (j, k) in enumerate([(1, 2)]):
    print(i, j, k)
for a, *b in [(1, 2, 3)]:
    print(a, b)
for bad in ["a, b = 1", "a, b = 1, 2, 3", "a, b, c = 1, 2", "a, b = [1]", "a, *b, c = [1]", "a, b = iter([1])", "a, b = iter([1, 2, 3])", "a, b = None", "a, b = 5.5"]:
    pass
def try_unpack(v, n):
    try:
        if n == 2:
            a, b = v
        elif n == 3:
            a, b, c = v
        else:
            a, *b, c = v
        print('ok')
    except (ValueError, TypeError) as e:
        print(type(e).__name__, e)
for v, n in [(1, 2), ((1, 2, 3), 2), ((1, 2), 3), ([1], 2), ([1], 0), (iter([1]), 2), (iter([1, 2, 3]), 2), (None, 2), (5.5, 2), (None, 0), ('abc', 2)]:
    try_unpack(v, n)
