print(list(), tuple(), dict(), list('ab'), list((1,)), list({1: 2}), list(range(2)), tuple('ab'), tuple([1]), tuple({1: 2}), dict([('a', 1)]), dict({'a': 1}, b=2), dict(a=1), dict(zip('ab', [1, 2])), sorted(set('aab')), frozenset([1]) == frozenset([1]), set(frozenset([1])) == {1}, dict([[1, 2]]), dict(['ab']))
for bad in [lambda: dict([1]), lambda: dict([(1, 2, 3)]), lambda: list(1), lambda: tuple(None), lambda: set(5), lambda: set([[]]), lambda: {[]: 1}, lambda: {[]}, lambda: dict(1)]:
    try:
        print(bad())
    except (TypeError, ValueError) as e:
        print(type(e).__name__, e)
