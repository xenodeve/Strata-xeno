print(round(0.5), round(1.5), round(2.5), round(-0.5), round(-1.5), round(2.4), round(2.6), round(1e20), round(5), round(True), round(1.25, 1), round(1.35, 1), round(2.675, 2), round(1.005, 2), round(0.125, 2), round(0.375, 2), round(1234.5678, 2), round(1234.5678, -1), round(1234.5678, -2), round(-1234.5678, -3), round(5.5, 0), round(1e300, 2), round(1e-300, 2), round(1.5, 400), round(1e300, -400), round(5e-324, 400))
print(round(123, 1), round(123, 0), round(123, -1), round(125, -1), round(135, -1), round(-125, -1), round(1234, -5), round(5, -1), round(15, -1), round(25, -1), round(10 ** 30 + 5, -1), type(round(1.5)).__name__, type(round(1.5, 0)).__name__, type(round(1, 1)).__name__)
for bad in [lambda: round('a'), lambda: round(float('nan')), lambda: round(float('inf')), lambda: round(1.5, 'a'), lambda: round(None)]:
    try:
        bad()
    except (TypeError, ValueError, OverflowError) as e:
        print(type(e).__name__, e)
