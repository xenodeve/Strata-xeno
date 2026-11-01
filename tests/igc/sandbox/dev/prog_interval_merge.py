def merge(iv):
    out = []
    for s, e in sorted(iv):
        if out and s <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], e))
        else:
            out.append((s, e))
    return out
print(merge([(1, 3), (8, 10), (2, 6), (15, 18), (17, 20)]), merge([]), merge([(1, 1)]))
