pairs = [(1, 2), (3, 4), (5, 6)]
total = 0
for a, b in pairs:
    total += a * b
else:
    print(total)
for i, (a, b) in enumerate(pairs):
    if a == 3:
        print('found', i, b)
        break
else:
    print('none')
for ch in 'héllo':
    print(ch, ord(ch))
