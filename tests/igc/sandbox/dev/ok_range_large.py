r = range(0, 2 ** 60, 2 ** 40)
print(len(r), r[5], r[-1], 2 ** 40 in r, 3 in r, r.index(2 ** 41), list(r[:2]))
print(len(range(10 ** 18)), range(10 ** 18)[10 ** 17], 10 ** 18 - 1 in range(10 ** 18), 10 ** 30 in range(5))
