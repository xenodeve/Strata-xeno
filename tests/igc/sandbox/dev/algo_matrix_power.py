def mul(a, b):
    return [[sum(x * y for x, y in zip(r, c)) % 1000007 for c in zip(*b)] for r in a]
def power(m, n):
    r = [[1, 0], [0, 1]]
    while n:
        if n & 1:
            r = mul(r, m)
        m = mul(m, m)
        n >>= 1
    return r
print(power([[1, 1], [1, 0]], 90))
