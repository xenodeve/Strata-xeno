a, b, c = 1, 2, 3
del a, b
del (c)
l = [1, 2, 3]; d = {'a': 1}
del l[0], d['a']
print(l, d)
x, y = 1, 2
del (x, y)
p, q = 1, 2
del [p, q]
m = [[1, 2], [3]]
del m[0][1]
print(m)
try:
    print(x)
except NameError as e:
    print(e)
