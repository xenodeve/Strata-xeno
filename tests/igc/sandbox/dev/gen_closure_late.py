def make():
    n = 1
    g = (x + n for x in [1, 2])
    n = 10
    return g
print(list(make()))
def nested():
    out = []
    for i in range(3):
        out.append(sum(j * i for j in range(3)))
    return out
print(nested())
print([list(j for j in range(i)) for i in range(3)])
print({k: sum(1 for _ in range(k)) for k in range(3)})
