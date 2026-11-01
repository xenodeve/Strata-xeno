print(sum(int(c) for c in str(2 ** 1000)), len(str(3 ** 1000)), sum(i ** i for i in range(1, 11)) % 10 ** 10, 2 ** 127 - 1, (2 ** 61 - 1) * (2 ** 89 - 1) % 1000003, str(2 ** 64)[-3:], [i for i in range(1, 30) if 2 ** i % 7 == 1][:3], int(str(10 ** 30)[::-1]), 10 ** 25 // 7, 10 ** 25 % 7)
import math
print(math.factorial(30) // math.factorial(28), math.comb(40, 20), math.isqrt(2 * 10 ** 40), math.gcd(2 ** 100 - 1, 2 ** 75 - 1), math.lcm(*range(1, 21)), len(str(math.factorial(100))), sum(map(int, str(math.factorial(100)))), math.log10(2 ** 100), math.log2(10 ** 30))
