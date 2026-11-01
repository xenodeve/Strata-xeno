from collections import Counter, defaultdict, deque
c = Counter('abracadabra'); print(c.most_common(2), c['a'], c['z'], sum(c.values()))
d = defaultdict(list); d['a'].append(1); d['b']; print(dict(d), sorted(d))
q = deque([1, 2, 3], maxlen=3); q.append(4); q.appendleft(0); print(list(q), q.pop(), q.popleft(), len(q))
