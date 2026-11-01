def popcount(n):
    c = 0
    while n:
        n &= n - 1
        c += 1
    return c
print([popcount(n) for n in (0, 1, 255, 2 ** 40 - 1)], [bin(n).count('1') for n in (7, 8)], 5 ^ 3 ^ 3, 1 << 10, (1 << 10) - 1, 0b1010 | 0b0101, ~0, -1 & 0xFF, -256 >> 4, hex(0xFF ^ 0x0F), format(10, '08b'), int('1010', 2), [i for i in range(16) if i & (i - 1) == 0])
subsets = [[x for j, x in enumerate('abc') if i >> j & 1] for i in range(8)]
print(subsets)
