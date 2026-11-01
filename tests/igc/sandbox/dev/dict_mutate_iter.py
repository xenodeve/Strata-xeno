d = {1: 1, 2: 2}
try:
    for k in d:
        del d[k]
except RuntimeError as e:
    print(e)
print(d)
d = {1: 1, 2: 2}
for k in d:
    d[k] = 5
print(d)
try:
    for k, v in d.items():
        d[k + 100] = v
except RuntimeError as e:
    print(e)
try:
    for v in d.values():
        d.clear()
except RuntimeError as e:
    print(e)
for k in list({1: 1}):
    pass
print(k)
