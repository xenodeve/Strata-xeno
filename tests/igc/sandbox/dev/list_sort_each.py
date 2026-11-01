a = [3, 1, 2]; a.sort(); print(a); a.sort(reverse=True); print(a)
b = ['bb', 'a', 'ccc']; b.sort(key=len); print(b); b.sort(key=len, reverse=True); print(b)
c = [(2, 'x'), (1, 'y'), (2, 'a')]; c.sort(); print(c); c.sort(key=lambda t: t[0]); print(c)
d = [1.5, 1, True, 0]; d.sort(); print(d)
e = []; e.sort(); print(e)
f = [[3], [1, 2], []]; f.sort(); print(f)
g = list(range(50, 0, -1)); g.sort(); print(g[:3], g[-1])
print(a.sort(), sorted(a))
h = [1, 2, 3]
try:
    h.sort(key=lambda x: (h.append(x), x)[1])
except ValueError as e2:
    print(e2)
