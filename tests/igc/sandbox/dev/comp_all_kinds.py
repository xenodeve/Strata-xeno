print([x * x for x in range(6) if x % 2])
print({k: v for k, v in zip('abc', range(3))})
print(sorted({x % 3 for x in range(10)}))
print(sum(x for x in range(5)))
print([(i, j) for i in range(3) for j in range(i)])
x = 'outer'
print([x for x in 'ab'], x)
print(any(x > 2 for x in [1, 2, 3]), all(x > 2 for x in [1, 2, 3]))
