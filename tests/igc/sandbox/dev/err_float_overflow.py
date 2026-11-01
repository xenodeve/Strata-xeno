print(1e308 * 10, -1e308 * 10, 1e308 + 1e308)
for fn in [lambda: 2.0 ** 10000, lambda: float(10 ** 400), lambda: 10 ** 400 / 1.0, lambda: int(float('inf')), lambda: int(float('nan')), lambda: round(float('inf')), lambda: 10.0 ** 400, lambda: 10 ** 400 * 1.0]:
    try:
        print(fn())
    except (OverflowError, ValueError) as e:
        print(type(e).__name__, e)
