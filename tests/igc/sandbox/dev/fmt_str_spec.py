s = 'abc'
print(f'[{s}] [{s:5}] [{s:<5}] [{s:>5}] [{s:^5}] [{s:*^7}] [{s:.2}] [{s:5.1}] [{s:^6.2}] [{s!r:>7}] [{s!s}] [{s!a}] [{s:s}] [{s:05}]')
for spec in ['d', 'x', '+', ',', '#', '=5', '%']:
    try:
        print(format(s, spec))
    except ValueError as e:
        print(e)
