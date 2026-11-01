from collections import defaultdict
d = defaultdict(int)
d['a'] += 1; d['b']; print(d, d['a'], len(d), 'c' in d, d.get('c'), len(d), sorted(d), dict(d), d.default_factory)
l = defaultdict(list); l[1].append(2); l[1].append(3); print(l, l[1], l[2], len(l))
s = defaultdict(set); s['a'].add(1); print(s)
f = defaultdict(lambda: 'x'); print(f['k'], list(f.items()))
n = defaultdict(); print(n, n.default_factory)
try:
    n['a']
except KeyError as e:
    print('KeyError', e)
g = defaultdict(str, a='x'); print(g, g['b'] == '', g)
h = defaultdict(int, {'a': 1}); print(h)
i = defaultdict(int, [('a', 1)]); print(i)
j = defaultdict(lambda: defaultdict(int)); j['a']['b'] += 1; print(j['a']['b'], len(j))
print(isinstance(d, dict), type(d).__name__, d == {'a': 1, 'b': 0}, d.copy() == d, type(d.copy()).__name__, d.pop('a'), d.setdefault('z', 9), d)
try:
    defaultdict(5)
except TypeError as e:
    print(e)
