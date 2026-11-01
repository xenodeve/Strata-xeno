a = {1, 2, 3}
print(len(a), 1 in a, 5 in a, 5 not in a, sorted(a), sum(a), min(a), max(a), a == {3, 2, 1}, a != {1}, set() == frozenset(), {1, 1, 1} == {1}, {1, 1.0, True} == {1}, len({1, 1.0, True}), {(1, 2)} == {(1, 2)})
b = a.copy(); b.add(4); print(sorted(a), sorted(b), a.issubset(b), b.issuperset(a), a.isdisjoint({9}), a.isdisjoint({1}), a.issubset([1, 2, 3, 4]), a.issuperset([1]))
print(sorted(a.union([9], (8,))), sorted(a.intersection([1, 2], {2})), sorted(a.difference([1], {2})), sorted(a.symmetric_difference([1, 9])), sorted(a.union()), sorted(a.intersection()))
c = {1, 2, 3}; c.update([4], {5}); print(sorted(c)); c.intersection_update([1, 2, 9]); print(sorted(c)); c.difference_update([1]); print(sorted(c)); c.symmetric_difference_update([2, 3]); print(sorted(c)); c.clear(); print(c)
d = {1, 2}; d |= {3}; d &= {1, 3}; d -= {1}; d ^= {3, 4}; print(d)
e = {1}; f = e; e |= {2}; print(sorted(f)); e = e | {3}; print(sorted(f))
print({1} < {1, 2}, {1} <= {1}, {1, 2} > {1}, {1} >= {1}, {1} < {2}, {1} < {1}, frozenset([1]) < {1, 2}, sorted(frozenset([1]) | {2}), type(frozenset([1]) | {2}).__name__, type({2} | frozenset([1])).__name__)
s = {1}; print(s.pop(), s)
for bad in [lambda: set().pop(), lambda: {1}.remove(2), lambda: {[]}, lambda: {1}.add([]), lambda: {1} | [2], lambda: {1} + {2}, lambda: {1}[0], lambda: {1} < [1], lambda: [] in {1}, lambda: frozenset([1]).add(2)]:
    try:
        print(bad())
    except (KeyError, TypeError, AttributeError) as e:
        print(type(e).__name__, e)
