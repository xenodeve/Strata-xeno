from itertools import combinations, combinations_with_replacement as cwr
print(list(combinations([], 0)), list(combinations('abc', 0)), list(combinations('abc', 1)), list(combinations('abcd', 2)), list(combinations('abc', 3)), list(combinations('abc', 4)), len(list(combinations(range(10), 5))), list(combinations([1, 1, 2], 2)))
print(list(cwr('ab', 0)), list(cwr('ab', 1)), list(cwr('ab', 2)), list(cwr('abc', 2)), list(cwr('a', 3)), list(cwr('', 0)), list(cwr('', 1)), len(list(cwr(range(4), 3))))
for bad in [lambda: combinations([1]), lambda: combinations([1], -1), lambda: cwr([1], -1)]:
    try:
        print(bad())
    except (TypeError, ValueError) as e:
        print(type(e).__name__)
