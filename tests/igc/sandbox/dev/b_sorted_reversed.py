print(sorted('hello'), sorted((3, 1)), sorted({2, 1}), sorted({'b': 1, 'a': 2}), sorted([1, 2.5, True]), sorted(['b', 'a', 'C']), sorted([[2], [1, 5], [1]]))
print(sorted([('a', 2), ('b', 1)], key=lambda p: p[1]), sorted(['aa', 'b', 'ccc'], key=len, reverse=True), sorted([1, -2, 3], key=abs), sorted(['x', 'y'], reverse=True), sorted('bca', key=None))
s = [(1, 'a'), (0, 'b'), (1, 'c'), (0, 'd')]
print(sorted(s, key=lambda x: x[0]), sorted(s, key=lambda x: x[0], reverse=True))
print(list(reversed('abc')), list(reversed((1, 2))), list(reversed([1, 2])), list(reversed({'a': 1, 'b': 2})), list(reversed(range(0, 6, 2))))
try:
    sorted([1, 'a'])
except TypeError as e:
    print(e)
try:
    reversed({1})
except TypeError as e:
    print(e)
