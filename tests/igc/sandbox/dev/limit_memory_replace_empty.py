# expect: MemoryLimit
s = 'a' * 10000000
print(len(s.replace('', 'bbbbbbbbbb')))
