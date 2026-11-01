a = []
push = a.append
push(1); push(2)
print(a)
up = 'abc'.upper
print(up(), list(map('x{}'.format, [1, 2])), sorted(['b', 'A'], key=str.lower), list(map(str.strip, [' a '])), max({'a': 1, 'b': 2}, key={'a': 1, 'b': 2}.get), list(map({'a': 1}.get, ['a', 'z'])), list(filter('abc'.count, ['a', 'z'])), str.upper('x'), list.count([1, 1], 1), dict.get({'a': 1}, 'a'), int.bit_length(5), ''.join(map(chr, [97])), sum(map(len, ['a', 'bb'])), list(map(int.bit_length, [1, 2])), set.union({1}, {2}) == {1, 2}, tuple.count((1,), 1), str.join(',', ['a']), str.split('a b'), list.index([1], 1))
for fn in [lambda: str.upper(1), lambda: list.append(1, 2), lambda: dict.get(1, 2), lambda: set.add(1, 2), lambda: 'a'.upper(1), lambda: [].append(), lambda: [].append(1, 2), lambda: {}.get(), lambda: {}.get(1, 2, 3), lambda: ''.split(1, 2, 3), lambda: [].pop(1, 2), lambda: len(), lambda: len(1, 2), lambda: abs(), lambda: ord(), lambda: isinstance(1), lambda: range(1, 2, 3, 4), lambda: divmod(1), lambda: repr()]:
    try:
        print(fn())
    except TypeError as e:
        print(e)
