p, r = 1000.0, 0.05
for y in range(1, 6):
    p *= 1 + r
    print(y, round(p, 2), f'{p:,.2f}')
print(1000 * (1 + 0.05) ** 5, sum(1000 * 1.05 ** i for i in range(5)))
