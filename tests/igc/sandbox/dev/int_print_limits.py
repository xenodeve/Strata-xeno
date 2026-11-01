print(len(str(10 ** 4299)), str(10 ** 4299)[:3], str(-10 ** 4299)[:4], int('1' * 4300) % 7, int('-' + '1' * 4300) % 7, len(repr(2 ** 14000)), len(hex(2 ** 20000)), len(bin(2 ** 20000)), int('f' * 5000, 16) % 7, int('1' * 5000, 2) % 7)
for fn in [lambda: str(10 ** 4300), lambda: repr(10 ** 4300), lambda: print(10 ** 4300), lambda: f'{10 ** 4300}', lambda: '%d' % 10 ** 4300, lambda: int('1' * 4301), lambda: int('1' * 4301, 10), lambda: int('1' * 4301, 3), lambda: [10 ** 4300]]:
    try:
        print(str(fn())[:5])
    except ValueError as e:
        print(str(e))
