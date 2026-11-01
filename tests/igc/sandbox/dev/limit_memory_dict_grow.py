# expect: MemoryLimit
d = {}
i = 0
while True:
    d[i] = [i] * 100
    i += 1
