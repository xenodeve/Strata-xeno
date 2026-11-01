d = {1: 1, 2: 2}
try:
    for k in d:
        d[k + 10] = 0
except RuntimeError as e:
    print(e)
s = {1, 2}
try:
    for k in s:
        s.add(k + 10)
except RuntimeError as e:
    print(e)
a = [1, 2, 3]
for x in a:
    if x == 1:
        a.remove(x)
    print(x)
a = [1, 2]
for x in a:
    if len(a) < 5:
        a.append(x)
print(a)
