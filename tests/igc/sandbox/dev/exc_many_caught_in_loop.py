n = 0
for i in range(2000):
    try:
        {}[i]
    except KeyError:
        n += 1
    try:
        [][i]
    except IndexError:
        n += 1
    try:
        1 / 0
    except ZeroDivisionError:
        n += 1
print(n)
