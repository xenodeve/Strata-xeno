from collections import deque
q = deque()
print(q, len(q), bool(q), list(q))
q.append(1); q.append(2); q.appendleft(0); print(q, q[0], q[-1], q[1], len(q), 1 in q, 5 in q, list(reversed(q)), sum(q), q == deque([0, 1, 2]), q != deque(), q.count(1), q.index(2))
q.extend([3, 4]); q.extendleft([-1, -2]); print(q); print(q.pop(), q.popleft(), q)
q.rotate(1); print(q); q.rotate(-2); print(q); q.rotate(); print(q); q.rotate(100); print(q)
q.reverse(); print(q); q.remove(1); print(q); q.insert(1, 9); print(q); q[0] = 7; print(q); del q[0]; print(q)
c = q.copy(); c.append(5); print(q, c); q.clear(); print(q)
m = deque([1, 2, 3], 2); print(m, m.maxlen); m.append(4); print(m); m.appendleft(0); print(m); m.extend([7, 8, 9]); print(m)
n = deque(maxlen=0); n.append(1); print(n)
print(deque('abc'), deque(range(3)), deque((1,)), deque(iter([1])), deque([1]) + deque([2]), deque(maxlen=3).maxlen, deque().maxlen)
o = deque([1]); o += deque([2]); print(o)
for bad in [lambda: deque().pop(), lambda: deque().popleft(), lambda: deque([1])[5], lambda: deque([1]).remove(2), lambda: deque([1]).index(2), lambda: deque([1])['a'], lambda: deque([1], 1).insert(0, 2), lambda: deque([1], -1), lambda: hash(deque())]:
    try:
        print(bad())
    except (IndexError, ValueError, TypeError) as e:
        print(type(e).__name__, e)
r = deque([1, 2])
try:
    for x in r:
        r.append(x)
except RuntimeError as e:
    print(e)
