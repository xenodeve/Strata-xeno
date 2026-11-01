from itertools import permutations
print(list(permutations([])), list(permutations([1])), list(permutations('ab')), list(permutations('abc', 2)), list(permutations('abc', 0)), list(permutations('ab', 3)), len(list(permutations(range(5)))), list(permutations(range(4), 3))[:5], list(permutations(range(4), 3))[-1], list(permutations([1, 1])))
g = permutations((x for x in [1, 2]))
print(next(g), next(g), next(g, 'end'))
for bad in [lambda: permutations(5), lambda: permutations([1], -1), lambda: permutations([1], 'a')]:
    try:
        print(bad())
    except (TypeError, ValueError) as e:
        print(type(e).__name__, e)
