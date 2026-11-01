def step(live):
    from collections import Counter
    n = Counter((x + dx, y + dy) for x, y in live for dx in (-1, 0, 1) for dy in (-1, 0, 1) if dx or dy)
    return {c for c, k in n.items() if k == 3 or (k == 2 and c in live)}
glider = {(1, 0), (2, 1), (0, 2), (1, 2), (2, 2)}
g = glider
for _ in range(4):
    g = step(g)
print(sorted(g), sorted((x + 1, y + 1) for x, y in glider) == sorted(g), len(step({(0, 0), (1, 0), (2, 0)})))
