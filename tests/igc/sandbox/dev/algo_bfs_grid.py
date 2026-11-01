from collections import deque
grid = ['..#.', '..#.', '....', '#...']
def bfs(start, goal):
    q = deque([(start, 0)])
    seen = {start}
    while q:
        (r, c), d = q.popleft()
        if (r, c) == goal:
            return d
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < 4 and 0 <= nc < 4 and grid[nr][nc] == '.' and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append(((nr, nc), d + 1))
    return -1
print(bfs((0, 0), (0, 3)))
