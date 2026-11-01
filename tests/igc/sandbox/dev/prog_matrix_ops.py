A = [[1, 2], [3, 4]]
B = [[5, 6], [7, 8]]
T = [list(r) for r in zip(*A)]
P = [[sum(a * b for a, b in zip(row, col)) for col in zip(*B)] for row in A]
det = A[0][0] * A[1][1] - A[0][1] * A[1][0]
print(T, P, det, [[A[i][j] + B[i][j] for j in range(2)] for i in range(2)], sum(map(sum, A)), max(map(max, B)), [row[::-1] for row in A[::-1]])
I = [[1 if i == j else 0 for j in range(3)] for i in range(3)]
print(I, sum(I[i][i] for i in range(3)))
for row in P:
    print(' '.join(f'{x:4d}' for x in row))
