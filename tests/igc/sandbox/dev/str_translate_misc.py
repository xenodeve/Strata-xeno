t = str.maketrans('abc', 'xyz')
print('aabbcc'.translate(t), 'abc'.translate(str.maketrans({'a': 'X', 'b': None})), 'abc'.translate(str.maketrans('a', 'b', 'c')), 'abc'.translate({97: 'Q', 98: None, 99: 100}), sorted(t.items()))
print('abc' < 'abd', 'abc' < 'ab', 'Z' < 'a', '' < 'a', 'a' == 'a', 'a' != 'b', 'abc' >= 'abc', 'é' > 'z', 'ab' * 0, 'ab' * -1, 'a' in 'abc', '' in 'abc', 'abc' in 'abc', 'abcd' in 'abc')
for bad in [lambda: 'a' + 1, lambda: 1 + 'a', lambda: 'a' * 'b', lambda: 'a' * 1.5, lambda: 1 in 'a', lambda: 'a' - 'b', lambda: 'a' < 1, lambda: -'a', lambda: 'a'[1.5], lambda: 'a'['b']]:
    try:
        print(bad())
    except TypeError as e:
        print(e)
s = 'abc'
try:
    s[0] = 'x'
except TypeError as e:
    print(e)
try:
    del s[0]
except TypeError as e:
    print(e)
