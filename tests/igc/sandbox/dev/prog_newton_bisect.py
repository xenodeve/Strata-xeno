def newton(f, df, x, n=20):
    for _ in range(n):
        x -= f(x) / df(x)
    return x
def bisect(f, lo, hi, n=60):
    for _ in range(n):
        mid = (lo + hi) / 2
        if f(lo) * f(mid) <= 0:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2
print(newton(lambda x: x * x - 2, lambda x: 2 * x, 1.0), bisect(lambda x: x ** 3 - x - 2, 1, 2), newton(lambda x: x ** 3 - 27, lambda x: 3 * x * x, 5.0))
