from collections import defaultdict, deque
edges = [('a', 'b'), ('a', 'c'), ('b', 'd'), ('c', 'd'), ('d', 'e')]
g, indeg = defaultdict(list), defaultdict(int)
for u, v in edges:
    g[u].append(v)
    indeg[v] += 1
    indeg[u] += 0
q = deque(sorted(n for n in indeg if indeg[n] == 0))
order = []
while q:
    n = q.popleft()
    order.append(n)
    for m in g[n]:
        indeg[m] -= 1
        if indeg[m] == 0:
            q.append(m)
print(order, len(order) == len(indeg), dict(g), dict(indeg))
def dfs(n, seen=None):
    seen = set() if seen is None else seen
    seen.add(n)
    for m in g[n]:
        if m not in seen:
            dfs(m, seen)
    return seen
print(sorted(dfs('b')), sorted(dfs('a')))
