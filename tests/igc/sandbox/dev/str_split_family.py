print('a b  c'.split(), '  a  '.split(), ''.split(), '   '.split(), 'a,b'.split(','), ',a,'.split(','), ''.split(','), 'a,b,c'.split(',', 1), 'a,b,c'.split(',', 0), 'a b c'.split(None, 1), 'a b c'.split(maxsplit=1), 'a,b'.split(sep=','), 'a b'.split(' ', -1), 'aXXbXXc'.split('XX'), '  a b  '.split(None, 1), '\ta\nb\r'.split())
print('a b c'.rsplit(), 'a b c'.rsplit(None, 1), 'a,b,c'.rsplit(',', 1), 'a,b,c'.rsplit(',', 0), '  a b  '.rsplit(None, 1), 'a b'.rsplit(maxsplit=0), ''.rsplit(), ''.rsplit(','))
print('a\nb'.splitlines(), 'a\r\nb\rc\n'.splitlines(), 'a\nb\n'.splitlines(True), ''.splitlines(), '\n'.splitlines(), 'a\x0bb\x0cc\x1cd\x1de\x1ef\x85g h i'.splitlines(), 'x\n\ny'.splitlines(keepends=True))
print('a-b-c'.partition('-'), 'abc'.partition('-'), 'a-b-c'.rpartition('-'), 'abc'.rpartition('-'), ''.partition('x'), '-'.partition('-'))
for bad in [lambda: 'a'.split(''), lambda: 'a'.rsplit(''), lambda: 'a'.partition(''), lambda: 'a'.split(1), lambda: 'a'.split(',', 'x')]:
    try:
        bad()
    except (ValueError, TypeError) as e:
        print(type(e).__name__, e)
