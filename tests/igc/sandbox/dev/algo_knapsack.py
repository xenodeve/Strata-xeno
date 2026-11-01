items = [(12, 4), (2, 2), (1, 1), (1, 2), (4, 10)]
best = [0] * 16
for w, v in items:
    for c in range(15, w - 1, -1):
        best[c] = max(best[c], best[c - w] + v)
print(best[15])
