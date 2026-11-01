d = {'a': 1, 'b': 2, 'c': 3}
print(list(d), list(d.keys()), list(d.values()), list(d.items()), d.keys(), d.values(), d.items(), len(d), 'a' in d, 1 in d, 'a' not in d)
print(d.get('a'), d.get('z'), d.get('z', 5), d.pop('a'), d.pop('z', 0), d.popitem(), d, d.setdefault('y'), d.setdefault('b', 5), d)
e = d.copy(); e['q'] = 1; print(d, e)
d.clear(); print(d)
print(dict.fromkeys('ab'), dict.fromkeys('ab', 0), {}.fromkeys([1]), dict.fromkeys(range(2), []))
print({'a': 1} == {'a': 1}, {'a': 1, 'b': 2} == {'b': 2, 'a': 1}, {'a': 1} != {'a': 2}, {1: 1} == {1.0: 1.0}, {} == {}, {'a': 1} | {'b': 2}, {'a': 1} | {'a': 2})
x = {'a': 1}; x |= {'b': 2}; x |= [('c', 3)]; print(x)
print({**{'a': 1}, 'b': 2, **{'c': 3, 'a': 9}}, dict(**{'a': 1}, b=2), {i: i * i for i in range(3)}, {k: v for k, v in [('a', 1)]})
for bad in [lambda: {}['x'], lambda: {}.pop('x'), lambda: {}.popitem(), lambda: {'a': 1} < {'a': 2}, lambda: {[]: 1}, lambda: {}[[]], lambda: {} + {}, lambda: dict.fromkeys(5), lambda: {**1}, lambda: {}.get(), lambda: {}.nosuch]:
    try:
        print(bad())
    except (KeyError, TypeError, AttributeError) as e:
        print(type(e).__name__, e)
try:
    del d['nope']
except KeyError as e:
    print(repr(e))
