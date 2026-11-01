from itertools import permutations, combinations, combinations_with_replacement, product, accumulate, chain, count, islice
print(list(permutations([1, 2, 3], 2)), list(combinations('abc', 2)), list(combinations_with_replacement([1, 2], 2)))
print(list(product([1, 2], 'ab')), list(product([0, 1], repeat=2)), list(accumulate([1, 2, 3])), list(chain([1], [2, 3])), list(islice(count(5), 3)), list(islice(count(0, 2), 1, 4)))
