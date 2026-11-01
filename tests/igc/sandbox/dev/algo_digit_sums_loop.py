total = 0
for n in range(1, 2000):
    if sum(map(int, str(n))) % 7 == 0:
        total += n
print(total)
