print(list(range(5)), list(range(2, 5)), list(range(0, 10, 3)), list(range(10, 0, -3)), list(range(5, 2)), list(range(0)))
print(range(5), range(1, 5), range(1, 5, 2), range(0, -5, -1))
r = range(0, 20, 5)
print(r[1], r[-1], r[1:3], len(r), 5 in r, 6 in r, 5.0 in r, 'x' in r, r.index(10), r.count(10), r.start, r.stop, r.step)
print(range(3) == range(3), range(0) == range(5, 2), list(reversed(range(3))), sum(range(101)), max(range(4)), min(range(1, 4)))
for bad in [lambda: range(1.5), lambda: range(0, 1, 0), lambda: range(), lambda: range('a'), lambda: range(3)[5]]:
    try:
        bad()
    except (TypeError, ValueError, IndexError) as e:
        print(type(e).__name__, e)
