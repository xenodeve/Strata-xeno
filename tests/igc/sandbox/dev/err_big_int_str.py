try:
    int('9' * 5000)
except ValueError as e:
    print('ValueError', str(e)[:30])
x = 10 ** 5000
print(x % 7, len(bin(x)))
try:
    print(x)
except ValueError as e:
    print('ValueError', str(e)[:30])
print(len(str(10 ** 4299)))
