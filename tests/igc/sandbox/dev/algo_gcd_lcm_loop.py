from math import gcd
total = 0
for a in range(1, 30):
    for b in range(a, 30):
        if gcd(a, b) == 1:
            total += a * b
print(total)
