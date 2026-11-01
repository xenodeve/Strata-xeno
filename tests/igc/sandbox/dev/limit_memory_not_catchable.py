# expect: MemoryLimit
try:
    x = 'a' * 10 ** 10
except MemoryError:
    print('caught')
