s = '  Hello, World  '
print(repr(s.strip()), repr(s.lstrip()), repr(s.rstrip()), s.split(), 'a,b,,c'.split(','), 'a b c'.split(' ', 1), 'x'.join(['1', '2']))
print('hello'.replace('l', 'L'), 'hello'.replace('l', 'L', 1), 'hello'.find('l'), 'hello'.find('z'), 'hello'.index('e'), 'hello'.count('l'))
print('Hello'.startswith('He'), 'Hello'.endswith('lo'), 'abc'.upper(), 'ABC'.lower(), '123'.isdigit(), 'abc'.isalpha(), 'a1'.isalnum(), ' \t'.isspace(), 'hello world'.title(), 'hello'.capitalize(), '42'.zfill(5), '-42'.zfill(5))
