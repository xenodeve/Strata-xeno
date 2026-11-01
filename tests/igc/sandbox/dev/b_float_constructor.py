print(float(), float(1), float(True), float('1'), float('1.'), float('.5'), float('1e3'), float('1E3'), float('-1.5e-3'), float('+inf'), float('-Infinity'), float('NaN'), float('1_000.5'), float('  2 \n'), float('1e400'), float('-1e400'), float('1e-400'))
for s in ['', ' ', 'x', '1.2.3', '1e', 'e5', '.', '1_', '_1', '1__0', 'in', 'infinit', '0x10', '--1']:
    try:
        print(float(s))
    except ValueError as e:
        print(e)
for v in [None, [], {}]:
    try:
        float(v)
    except TypeError as e:
        print(e)
