# expect: MemoryLimit
s = ['a' * 1000000] * 100
print(len(''.join(s)))
