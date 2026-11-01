a = set(range(0, 30, 3))
b = set(range(0, 30, 5))
print(sorted(a & b), sorted(a - b)[:3], len(a | b), sorted(a ^ b)[-2:], {x for x in a if x % 2} == {3, 9, 15, 21, 27}, len({frozenset([i, j]) for i in range(4) for j in range(4)}), len(set('mississippi')), sorted(set('mississippi'), key='mississippi'.index))
print(sorted({(x, y) for x in range(2) for y in range(2) if x <= y}), {1, 2} <= {1, 2, 3}, not {1} & {2}, sum({1, 2, 2, 3}), max({-1, -2}), sorted({'b', 'a'}), len({1.0, 1, True}), len({0, False, 0.0, -0.0}), {None}, {()}, {(1,)})
