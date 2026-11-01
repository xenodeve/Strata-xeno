data = [2, 4, 4, 4, 5, 5, 7, 9]
n = len(data)
mean = sum(data) / n
var = sum((x - mean) ** 2 for x in data) / n
s = sorted(data)
median = (s[n // 2 - 1] + s[n // 2]) / 2
print(mean, var, var ** 0.5, median, max(set(data), key=data.count))
