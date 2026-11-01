def to_roman(n):
    out = ''
    for v, s in zip((1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1), 'M CM D CD C XC L XL X IX V IV I'.split()):
        while n >= v:
            out += s
            n -= v
    return out
def from_roman(s):
    m = dict(zip('IVXLCDM', (1, 5, 10, 50, 100, 500, 1000)))
    return sum(-m[a] if m[a] < m[b] else m[a] for a, b in zip(s, s[1:])) + m[s[-1]]
print([to_roman(n) for n in (4, 9, 14, 40, 90, 400, 1994, 2026)], all(from_roman(to_roman(i)) == i for i in range(1, 400)))
