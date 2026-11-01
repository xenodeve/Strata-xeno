a = 2 ** 200
b = 3 ** 100
print(a, b, a + b, a - b, a * b, a // b, a % b, -a // b, -a % b, a // -b, a % -b)
print(divmod(a, b), divmod(-a, b), pow(2, 1000, 1000007), pow(3, -1, 7), abs(-a))
print(a >> 100, a << 3, -a >> 100, a & b, a | b, a ^ b, ~a, -a & b, -a | b, -a ^ -b)
print(a == 2 ** 200, a != b, a < b, a > b, hash(a) == hash(2 ** 200), bin(5), hex(-255), oct(8), hex(a))
print(10 ** 30 / 3, 2 ** 64 / 2 ** 63, float(2 ** 100), int(1e30), int(-3.9), 7 // 2, -7 // 2, 7 % -2, -7 % 2)
