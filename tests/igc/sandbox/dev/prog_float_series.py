import math
pi4 = sum((-1) ** k / (2 * k + 1) for k in range(2000))
e = sum(1 / math.factorial(k) for k in range(20))
s = 0.0
for k in range(1, 1000):
    s += 1 / k ** 2
print(pi4 * 4, e, s, math.sqrt(s * 6), abs(e - math.e) < 1e-12, round(pi4 * 4, 3), f'{e:.10f}', 0.1 * 3, 1.1 + 2.2, 3.3 * 3, 1e16 + 1, 2 ** 0.5 * 2 ** 0.5, 100 * 1.1, 1 / 3 + 1 / 3 + 1 / 3)
