cnt = 0
for a in range(1, 40):
    for b in range(a, 40):
        c2 = a * a + b * b
        c = int(c2 ** 0.5)
        if c * c == c2 and c < 40:
            cnt += 1
print(cnt)
