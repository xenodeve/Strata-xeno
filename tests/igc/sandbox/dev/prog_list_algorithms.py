def bubble(a):
    a = a[:]
    for i in range(len(a)):
        for j in range(len(a) - i - 1):
            if a[j] > a[j + 1]:
                a[j], a[j + 1] = a[j + 1], a[j]
    return a
def quick(a):
    return a if len(a) < 2 else quick([x for x in a[1:] if x < a[0]]) + [a[0]] + quick([x for x in a[1:] if x >= a[0]])
def merge_sort(a):
    if len(a) < 2:
        return a
    m = len(a) // 2
    l, r = merge_sort(a[:m]), merge_sort(a[m:])
    out, i, j = [], 0, 0
    while i < len(l) and j < len(r):
        if l[i] <= r[j]:
            out.append(l[i]); i += 1
        else:
            out.append(r[j]); j += 1
    return out + l[i:] + r[j:]
data = [5, 2, 9, 1, 5, 6, -3, 0]
print(bubble(data), quick(data), merge_sort(data), sorted(data), data)
