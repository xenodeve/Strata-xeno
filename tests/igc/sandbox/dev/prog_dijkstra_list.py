def dijkstra(adj, s):
    dist = {n: float('inf') for n in adj}
    dist[s] = 0
    todo = set(adj)
    while todo:
        u = min(todo, key=lambda n: (dist[n], n))
        todo.remove(u)
        for v, w in adj[u]:
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
    return dist
adj = {'a': [('b', 7), ('c', 9), ('f', 14)], 'b': [('a', 7), ('c', 10), ('d', 15)], 'c': [('a', 9), ('b', 10), ('d', 11), ('f', 2)], 'd': [('b', 15), ('c', 11), ('e', 6)], 'e': [('d', 6), ('f', 9)], 'f': [('a', 14), ('c', 2), ('e', 9)], 'z': []}
print(dijkstra(adj, 'a'))
