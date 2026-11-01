a = [1, 2, 3]
print(a.append(4), a, a.extend('ab'), a, a.extend((5,)), a, a.extend(range(2)), a, a.extend({9: 1}), a)
a.insert(0, 'z'); a.insert(-1, 'y'); a.insert(100, 'x'); a.insert(-100, 'w'); print(a)
print(a.pop(), a.pop(0), a.pop(-2), a.pop(1), a)
a.clear(); print(a, len(a))
b = [1, 2, 1, 2]
b.remove(2); print(b, b.index(2), b.index(1, 1), b.index(1, -1), b.count(1), b.count(5))
b.reverse(); print(b)
c = b.copy(); c.append(0); print(b, c)
for bad in [lambda: [].pop(), lambda: [1].pop(5), lambda: [1].remove(2), lambda: [1].index(2), lambda: [1].pop('a'), lambda: [1].insert('a', 1), lambda: [].extend(5), lambda: [1].index(1, 'a')]:
    try:
        print(bad())
    except (IndexError, ValueError, TypeError) as e:
        print(type(e).__name__, e)
