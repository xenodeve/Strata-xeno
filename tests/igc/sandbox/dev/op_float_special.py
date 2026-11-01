inf, nan = float('inf'), float('nan')
print(inf + 1, inf - inf, inf * 0, inf / inf, -inf, nan + 1, nan == nan, nan != nan, nan < 1, nan > 1, inf > 1e308, -inf < -1e308, inf == inf, 1 / inf, -1 / inf, 0.0 == -0.0, 1e308 + 1e308, 2.0 ** 0.5, 1e-320 * 1e-320, 5e-324 / 2, abs(-0.0), abs(-inf), abs(nan), min(1.0, 2.0), max(0.0, -0.0), 0.1 + 0.2 == 0.3, round(0.1 + 0.2, 10) == 0.3, int(1e15), int(-1e15), int(1.9999999999999998), float(2 ** 53 + 1), float(-2 ** 53 - 1), float(2 ** 1023), 2 ** 1023 * 2.0 if False else 0, 1e22 == 10 ** 22, 1e23 == 10 ** 23, 2 ** 53 + 1.0, 10 ** 22 / 10 ** 21, 10 ** 400 / 10 ** 399, 1 / 10 ** 400, -1 / 10 ** 400, 3 / 10 ** 323, 10 ** 308 / 0.1 if False else 0, int(5e-324))
print(inf ** 0, nan ** 0, 1 ** nan, 1 ** inf, inf ** -1, (-inf) ** 3, (-inf) ** 2, (-inf) ** -3, 0.5 ** inf, 2 ** inf, 0.5 ** -inf, 2 ** -inf, (-1.0) ** inf, 0.0 ** 0.0, 0.0 ** 1, (-0.0) ** 3, (-0.0) ** 2, 1.0 ** 1e308)
for bad in [lambda: inf % 1, lambda: int(inf), lambda: int(nan), lambda: 1 % inf, lambda: -1 % inf, lambda: inf // 1, lambda: (-8.0) ** (1 / 3) if False else 0, lambda: 10.0 ** 400, lambda: 1e300 ** 2, lambda: (-1e300) ** 3, lambda: round(inf), lambda: float(10 ** 309), lambda: 10 ** 309 + 1.0, lambda: 10 ** 309 < inf, lambda: 10 ** 309 > 1e308, lambda: 10 ** 309 == inf, lambda: -10 ** 309 < -inf]:
    try:
        print(bad())
    except (ValueError, OverflowError) as e:
        print(type(e).__name__, e)
