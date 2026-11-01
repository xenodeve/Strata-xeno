def expand(n, d, k):
    out = str(n // d) + '.'
    n %= d
    for _ in range(k):
        n *= 10
        out += str(n // d)
        n %= d
    return out
print(expand(1, 7, 30), expand(22, 7, 10))
