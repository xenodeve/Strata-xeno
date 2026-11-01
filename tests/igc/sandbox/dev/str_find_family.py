s = 'hello world hello'
print(s.find('hello'), s.find('hello', 1), s.find('hello', 1, 5), s.find('z'), s.find(''), s.find('', 100), s.find('', 17), s.find('o', -5), s.find('o', 0, -100), s.rfind('hello'), s.rfind('hello', 0, 10), s.rfind('z'), s.rfind(''), s.rfind('o', -5, -1), s.index('world'), s.rindex('hello'), s.count('l'), s.count('hello'), s.count(''), s.count('l', 5), s.count('l', 3, 5), ''.count(''), 'aaa'.count('aa'), s.count('', 3, 5), s.count('', 20))
print(s.startswith('hello'), s.startswith('world'), s.startswith('world', 6), s.startswith(('x', 'hello')), s.startswith(''), s.startswith('', 100), s.endswith('hello'), s.endswith('world', 0, 11), s.endswith(('x', 'lo')), s.endswith(''), s.startswith('hello', -5), s.endswith('hello', -5))
for bad in [lambda: s.index('z'), lambda: s.rindex('z'), lambda: s.find(1), lambda: s.count(None), lambda: s.startswith(1), lambda: s.startswith((1,)), lambda: s.endswith(1)]:
    try:
        bad()
    except (ValueError, TypeError) as e:
        print(type(e).__name__, e)
