a = [1, 2, 3, 4]
for x in a:
    if x % 2:
        a.remove(x)
print(a)
b = [1, 2]
for i, x in enumerate(b):
    if i < 2:
        b.append(x * 2)
print(b)
c = [1, 2, 3]
it = iter(c)
next(it); c.clear()
print(list(it))
d = [3, 2, 1]
for x in d:
    d.sort()
print(d)
e = [1, 2, 3]
print([e.pop() for _ in e], e)
f = [1, 2, 3]
r = reversed(f)
print(next(r)); f.clear(); print(list(r))
s = 'abc'
for ch in s:
    s += ch
print(s)
g = {1, 2, 3}
for x in list(g):
    g.discard(x)
print(g)
