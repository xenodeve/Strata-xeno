a = [1, 2]
b = a
a += [3]; print(a, b, a is b)
a = a + [4]; print(a, b, a is b)
c = [1]; c *= 3; print(c)
d = [1]; e = d; d *= 2; print(e)
f = []; f += 'ab'; f += (1,); f += {2: 3}; print(f)
print([1, 2] == [1, 2], [1, 2] != [1, 2], [1] < [2], [1] < [1, 0], [] < [1], [2] > [1, 5], [1, 'a'] == [1, 'a'], [1.0] == [1], [[1]] == [[1]], [1, 2] <= [1, 2], [1] >= [1])
print([0] * 0, [0] * -1, [[]] * 2, 2 * [1], [1, 2] * 2, ['a'] * 3, len([None] * 1000))
g = [[]] * 2; g[0].append(1); print(g)
for bad in [lambda: [1] + 1, lambda: [1] + (2,), lambda: [1] * 'a', lambda: [1] * 1.5, lambda: [1] < 1, lambda: [1] < ['a'], lambda: [1] - [1]]:
    try:
        print(bad())
    except TypeError as e:
        print(e)
