for fn in [lambda: [0] * 10 ** 100, lambda: 'a' * 10 ** 100, lambda: (1,) * 10 ** 100, lambda: 10 ** 100 * [0], lambda: [0][10 ** 100], lambda: 'a'[10 ** 100], lambda: (1,)[-10 ** 100]]:
    try:
        print(len(fn()))
    except (OverflowError, IndexError) as e:
        print(type(e).__name__, e)
print([0] * -10 ** 100, 'a' * 0)
