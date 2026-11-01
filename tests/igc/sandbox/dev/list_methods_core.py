a = [3, 1, 2]
a.append(4); a.extend([5, 6]); a.insert(1, 9); print(a)
print(a.pop(), a.pop(0), a); a.remove(1); print(a); a.sort(); print(a); a.reverse(); print(a)
print(a.index(9), a.count(2), a.copy() == a, a.copy() is a)
b = sorted(['pear', 'fig', 'apple'], key=len, reverse=True); print(b)
b.sort(key=lambda s: s[-1]); print(b)
