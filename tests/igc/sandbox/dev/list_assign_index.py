a = [0] * 5
a[0] = 1; a[-1] = 2; a[1:3] = [7]; print(a)
a[1:1] = [8, 9]; print(a)
a[:] = 'xy'; print(a)
a[len(a):] = [1]; print(a)
a[0:2] = (); print(a)
b = list(range(6)); b[::2] = 'abc'; print(b); b[::-1] = range(6); print(b)
del b[0]; del b[-1]; print(b); del b[:]; print(b)
c = list(range(10)); del c[1:3]; print(c); del c[::2]; print(c); del c[::-1]; print(c)
d = [1, 2, 3]; d[5:7] = [4]; print(d); d[-100:0] = [0]; print(d); d[2:1] = [9]; print(d)
x = [1]
try:
    x[5] = 1
except IndexError as e:
    print(e)
try:
    del x[5]
except IndexError as e:
    print(e)
try:
    x['a'] = 1
except TypeError as e:
    print(e)
try:
    x[::2] = [1, 2]
except ValueError as e:
    print(e)
try:
    x[0:1] = 5
except TypeError as e:
    print(e)
try:
    x['a']
except TypeError as e:
    print(e)
try:
    (1,)[0] = 2
except TypeError as e:
    print(e)
try:
    del (1,)[0]
except TypeError as e:
    print(e)
