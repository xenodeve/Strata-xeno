def sieve(n):
    is_p = [True] * (n + 1)
    is_p[0:2] = [False, False]
    for i in range(2, int(n ** 0.5) + 1):
        if is_p[i]:
            is_p[i * i::i] = [False] * len(is_p[i * i::i])
    return [i for i, p in enumerate(is_p) if p]
print(sieve(100), len(sieve(1000)))
