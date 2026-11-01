d = {'a': 1, 'b': 2}
d['c'] = 3
print(d, d.get('a'), d.get('z'), d.get('z', 0), list(d.keys()), list(d.values()), list(d.items()), 'a' in d)
print(d.pop('a'), d.pop('z', None), d.setdefault('x', 9), d.setdefault('b', 0), d)
d.update({'q': 1}); d.update(r=2); d.update([('s', 3)]); print(d)
del d['q']; print(d, len(d))
e = {1: 'a', True: 'b', 1.0: 'c'}; print(e)
d2 = {}; d2['z'] = 1; d2['y'] = 2; del d2['z']; d2['z'] = 3; print(d2)
