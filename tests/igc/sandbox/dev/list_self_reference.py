a = [1]
a.append(a)
print(a)
b = {}
b['self'] = b
print(b)
c = [[]]
c[0].append(c)
print(c)
t = ([],)
t[0].append(t)
print(t)
