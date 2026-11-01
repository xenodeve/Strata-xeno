a = {1, 2, 3}; b = {2, 3, 4}
print(sorted(a | b), sorted(a & b), sorted(a - b), sorted(a ^ b), a <= b, {1} < a, a >= {1}, 2 in a)
a.add(9); a.discard(1); a.discard(100); a.remove(2); print(sorted(a))
print(sorted(a.union(b)), sorted(a.intersection(b)), sorted(a.difference(b)), len(set()), set() == set())
print(set(), repr(set()))
