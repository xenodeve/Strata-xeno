for v in [0.1, 1/3, 1e22, 1e21, 1e16, 1e15, 1.5e-5, 0.0001, 0.00001, 123456789.125, 2.5, -0.0, 1e300*10, -1e300*10, 5e-324, 1.7976931348623157e308, 100.0, 3.0e0]:
    print(v, repr(v), str(v))
print(float('nan'), float('inf'), float('-inf'), float('1e5'), float(' 2.5 '), float(3))
