nums = range(1, 21)
evens = (n for n in nums if n % 2 == 0)
squares = (n * n for n in evens)
small = (n for n in squares if n < 200)
print(sum(small), list(small))
from itertools import islice, count, accumulate, chain, product, combinations
tri = accumulate(count(1))
print(list(islice(tri, 6)), list(islice((x for x in count() if x % 7 == 3), 3)), list(chain.from_iterable(zip('ab', 'cd'))), sum(1 for p in product('ab', repeat=4) if p.count('a') == 2), max(combinations([3, 1, 4, 1, 5], 2), key=sum))
