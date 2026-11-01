from itertools import product, accumulate, chain, count, islice
print(list(product()), list(product([])), list(product([1], [])), list(product('ab')), list(product('ab', repeat=0)), list(product('ab', repeat=2)), list(product([1, 2], [3], repeat=2)), len(list(product(range(3), repeat=3))))
print(list(accumulate([])), list(accumulate([1, 2, 3], lambda a, b: a * b)), list(accumulate([1, 2], initial=10)), list(accumulate([], initial=1)), list(accumulate([3, 1, 2], max)), list(accumulate(['a', 'b'])), list(accumulate([1.5, 2])), list(accumulate([1, 2], func=lambda a, b: a - b)))
print(list(chain()), list(chain([], [])), list(chain('ab', [1], (2,))), list(chain.from_iterable([[1], [2, 3]])), list(chain.from_iterable(['ab', 'c'])), list(chain(*[[1], [2]])))
print(list(islice('abcdef', 2)), list(islice('abcdef', 2, 4)), list(islice('abcdef', 0, None, 2)), list(islice('abcdef', 1, None)), list(islice('abc', 0)), list(islice('abc', 10)), list(islice('abcdef', 1, 5, 2)), list(islice(count(), 3, 5)), list(islice(count(1.5), 2)), list(islice(count(0, 0.5), 3)), list(islice(count(10, -2), 3)), list(islice(count(start=2, step=3), 2)), list(islice('abc', None)), list(islice('abc', None, None, None)))
c = count(); print(next(c), next(c), next(c))
it = iter(range(10)); print(list(islice(it, 2)), next(it), list(islice(it, 1, 3)), next(it))
for bad in [lambda: list(islice('a', -1)), lambda: list(islice('a', 0, -1)), lambda: list(islice('a', 0, 1, 0)), lambda: list(islice('a', 'b')), lambda: list(chain(1)), lambda: list(accumulate([1, 'a'])), lambda: count('a'), lambda: list(product(5))]:
    try:
        print(bad())
    except (TypeError, ValueError) as e:
        print(type(e).__name__, e)
