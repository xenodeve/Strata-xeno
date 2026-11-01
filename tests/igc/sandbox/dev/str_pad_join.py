print('a'.center(5), 'ab'.center(5), 'a'.center(4), 'ab'.center(4), 'a'.center(5, '*'), 'abc'.center(2), repr('a'.ljust(3)), repr('a'.rjust(3)), 'a'.ljust(3, '-'), 'a'.rjust(3, '-'), 'abc'.ljust(1), 'abc'.zfill(5), '42'.zfill(1), '+1'.zfill(4), '-1'.zfill(4), ''.zfill(2), 'a\tb'.expandtabs(), 'a\tb'.expandtabs(4), '\t'.expandtabs(0), 'ab\tc\nd\te'.expandtabs(3), repr('a\tb'.expandtabs(tabsize=2)))
print(','.join(['a', 'b']), ''.join([]), ','.join(['a']), ','.join('abc'), '-'.join(('x', 'y')), ''.join({'a': 1}), ' '.join(str(i) for i in range(3)), ', '.join(map(str, [1, 2])), ''.join(sorted(set('aab'))))
for bad in [lambda: ','.join([1]), lambda: ','.join(['a', 1]), lambda: ','.join(5), lambda: 'a'.center(5, 'ab'), lambda: 'a'.ljust(5, ''), lambda: 'a'.zfill('x')]:
    try:
        print(bad())
    except TypeError as e:
        print(type(e).__name__, e)
