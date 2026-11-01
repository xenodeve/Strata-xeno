def two_sum(a, t):
    seen = {}
    for i, x in enumerate(a):
        if t - x in seen:
            return seen[t - x], i
        seen[x] = i
print(two_sum([2, 7, 11, 15], 9), two_sum([3, 3], 6), two_sum([1], 5))
