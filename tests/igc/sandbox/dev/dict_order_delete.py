d = {}
for i in range(20):
    d[i] = i
for i in range(0, 20, 2):
    del d[i]
d[0] = 'back'
print(list(d), len(d))
e = {'x': 1}
e['x'] = 2
print(e)
f = {1: 'a'}
f[1.0] = 'b'
f[True] = 'c'
print(f, list(f))
g = {}
g[(1, 2)] = 1
g[frozenset([1])] = 2
g[None] = 3
g[1.5] = 4
g['s'] = 5
g[2 ** 100] = 6
print(g[(1, 2)], g[frozenset({1})], g[None], g[1.5], g['s'], g[2 ** 100], len(g))
big = {i: str(i) for i in range(2000)}
print(len(big), big[1999], sum(big), list(big)[:3])
for i in range(1990):
    del big[i]
print(big, len(big))
