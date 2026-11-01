print(int(), int(5), int(-5.9), int(5.9), int('12'), int('-12'), int('+12'), int('  12\n'), int('1_000'), int('0'), int(True), int('11', 2), int('zz', 36), int('0x1f', 16), int('0b11', 0), int('0o7', 0), int('0x1f', 0), int('1f', 16), int('Z', 36))
for args in [('x',), ('',), (' ',), ('1.5',), ('1__0',), ('_1',), ('1_',), ('12', 2), ('9', 8), ('0x', 16), ('0b2', 0), ('010', 0), ('1', 1), ('1', 37)]:
    try:
        print(int(*args))
    except ValueError as e:
        print('ValueError', e)
for args in [(None,), ([],), (5, 2), (1.5, 2)]:
    try:
        print(int(*args))
    except TypeError as e:
        print('TypeError', e)
