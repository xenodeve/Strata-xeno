a = [1, 2]; a.append(a); print(a, len(a), a[2][2][0])
d = {}; d['a'] = d; d['b'] = [d]; print(d)
t = ([],); t[0].append(t); print(t)
l1 = []; l2 = [l1]; l1.append(l2); print(l1, l2)
print(a == a, a[2] is a, d == d, repr(a) == str(a), [a] == [a])
s = str(a); print(s, len(s))
q = [[]]; q[0].append(q[0]); print(q)
from collections import deque
dq = deque(); dq.append(dq); print(dq)
