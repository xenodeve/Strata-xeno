print(sum([]), sum([1.5, 2]), sum((1, 2)), sum({1: 'a', 2: 'b'}), sum([True, True]), sum(x * 2 for x in range(3)), sum([1], 10), sum([], 5), sum(range(5), start=2))
print(min(1, 2.5), max(1, 1.0), max(1.0, 1), min('b', 'a'), max([1, 2], [1, 3]), max((1, 'a'), (1, 'b')), min([3, 1], key=lambda x: x), max('a', 'bb', key=len), min([], default=None), max({'a': 1, 'b': 0}), max(1, 2, 3, key=lambda x: -x))
for fn in [lambda: min([]), lambda: max(()), lambda: min(1), lambda: sum(['a']), lambda: sum(['a'], ''), lambda: max(1, 2, default=3), lambda: max([1, 'a'])]:
    try:
        print(fn())
    except (ValueError, TypeError) as e:
        print(type(e).__name__, e)
