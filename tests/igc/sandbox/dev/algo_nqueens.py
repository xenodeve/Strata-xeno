def solve(n, row=0, cols=(), d1=(), d2=()):
    if row == n:
        return 1
    return sum(solve(n, row + 1, cols + (c,), d1 + (row + c,), d2 + (row - c,)) for c in range(n) if c not in cols and row + c not in d1 and row - c not in d2)
print([solve(i) for i in range(1, 7)])
