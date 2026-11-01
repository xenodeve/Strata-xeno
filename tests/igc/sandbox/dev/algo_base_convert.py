def to_base(n, b):
    if n == 0:
        return '0'
    digits = ''
    while n:
        digits = '0123456789abcdef'[n % b] + digits
        n //= b
    return digits
print(to_base(255, 2), to_base(255, 16), to_base(0, 7), int(to_base(12345, 7), 7), sum(int(d) for d in str(2 ** 100)))
