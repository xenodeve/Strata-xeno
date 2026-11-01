print(pow(2, 3), pow(2, -1), pow(2.0, 3), pow(2, 0.5), pow(-2, 3), pow(0, 0), pow(2, 10, 7), pow(-2, 3, 5), pow(2, 3, -5), pow(3, -1, 7), pow(2, 100, 3), pow(10 ** 30, 10 ** 30, 97), pow(True, 2), pow(2, True))
for bad in [lambda: pow(2, 3, 0), lambda: pow(2, -1, 4), lambda: pow(2.0, 3, 5), lambda: pow('a', 2), lambda: 0.0 ** -1, lambda: pow(0, -1)]:
    try:
        print(bad())
    except (TypeError, ValueError, ZeroDivisionError) as e:
        print(type(e).__name__, e)
