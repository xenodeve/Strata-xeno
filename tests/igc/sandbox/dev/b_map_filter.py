print(list(map(int, ['1', '2'])), list(map(len, ['a', 'bb'])), list(map(pow, [2, 3], [3, 2])), list(map(lambda x: x + 1, range(3))), list(map(str.upper, ['a'])), list(map(max, [1, 3], [2, 2])))
print(list(filter(str.isdigit, ['1', 'a'])), list(filter(None, ['', 'a', 0, 1, None, []])), list(filter(bool, [0, 1])))
m = map(print, [1, 2])
print('lazy')
print(list(m))
log = []
def f(x):
    log.append(x)
    return x
it = map(f, [1, 2, 3])
print(next(it), log)
print(next(it), log)
