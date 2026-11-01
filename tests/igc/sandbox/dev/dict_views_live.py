d = {'a': 1, 'b': 2}
k, v, i = d.keys(), d.values(), d.items()
d['c'] = 3
print(k, v, i, len(k), 'a' in k, 1 in v, ('a', 1) in i, ('a', 2) in i, list(reversed(k)), list(reversed(v)), list(reversed(i)))
print(sorted(k & {'a', 'z'}), sorted(k | {'z'}), sorted(k - {'a'}), sorted(k ^ {'a', 'z'}), k == {'a', 'b', 'c'}, k >= {'a'}, k <= {'a'}, sorted(i & {('a', 1)}), sorted(d.keys() & d.keys()))
print(list(map(lambda p: p[0], d.items())), [x for x, y in d.items() if y > 1], max(d, key=d.get), min(d.values()), sum(d.values()), sorted(d.items(), key=lambda kv: -kv[1]))
