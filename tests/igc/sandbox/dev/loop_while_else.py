i = 0
while i < 5:
    i += 1
    if i == 3:
        continue
    if i == 7:
        break
    print(i)
else:
    print('done', i)
n = 10
while n:
    n -= 3
    if n < 0:
        break
else:
    print('no')
print(n)
