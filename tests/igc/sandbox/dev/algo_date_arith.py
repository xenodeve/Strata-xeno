def leap(y):
    return y % 4 == 0 and (y % 100 != 0 or y % 400 == 0)
def days(y, m):
    return [31, 29 if leap(y) else 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31][m - 1]
def weekday(y, m, d):
    t = [0, 3, 2, 5, 0, 3, 5, 1, 4, 6, 2, 4]
    y -= m < 3
    return (y + y // 4 - y // 100 + y // 400 + t[m - 1] + d) % 7
print(sum(days(2024, m) for m in range(1, 13)), weekday(2026, 11, 1), [y for y in range(1896, 1912) if leap(y)])
