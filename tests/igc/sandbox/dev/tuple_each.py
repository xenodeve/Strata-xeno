t = (1, 2, 3, 2)
print(t, t[0], t[-1], t[1:], t[::2], len(t), t.count(2), t.index(2), t.index(2, 2), 2 in t, 5 in t, t + (4,), t * 2, () + (), (1,) * 0, (1, 2) < (1, 3), (1,) == (1,), (1, 2) != (2, 1), hash((1, 2)) == hash((1, 2)), tuple(t), sum(t), max(t), sorted(t), list(t))
x = 1, 2
y = 1,
z = ()
print(x, y, z, (1), ((1,),), (1, (2, (3,))))
a, b = x; print(a, b)
d = {(1, 2): 'k'}; print(d[(1, 2)], d[1, 2])
print(t.index(9) if False else 'skip')
for bad in [lambda: t.index(9), lambda: (1,) + [2], lambda: (1,) * 'a', lambda: ([],) in {}, lambda: {([],): 1}, lambda: (1,) < 1]:
    try:
        print(bad())
    except (ValueError, TypeError) as e:
        print(type(e).__name__, e)
