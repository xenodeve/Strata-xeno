d = {'a': 1, 'b': 2}
print(sorted(d.keys() | ['z']), sorted(d.keys() & ('a', 'q')), sorted(d.keys() - ['a']), sorted(d.keys() ^ ['a', 'q']), sorted(['a'] & d.keys()) if False else 0, sorted(['z'] | d.keys()) if False else 0, sorted({'q'} | d.keys()), sorted({'a'} & d.keys()), sorted(d.items() | [('z', 1)]), sorted(d.items() & [('a', 1)]), sorted(d.keys() | 'xy'), type(d.keys() | set()).__name__)
print(sorted(['z'] | d.keys()), sorted(('a',) & d.keys()), sorted('ab' - d.keys()))
for fn in [lambda: d.values() | d.values(), lambda: d.values() & [1], lambda: d.keys() | 5, lambda: {1} | [1], lambda: [1] | {1}]:
    try:
        print(sorted(fn()))
    except TypeError as e:
        print(e)
