a = list(range(10))
print(a[2:5], a[:3], a[7:], a[-3:], a[::2], a[::-1], a[8:2:-2], a[100:], a[-100:2])
s = 'hello world'
print(s[1:4], s[::-1], s[-5:], s[::3], s[20:], s[-1])
t = (1, 2, 3, 4)
print(t[1:3], t[::-1], t[-1])
a[2:4] = [20, 30, 40]
print(a)
a[::2] = [0] * len(a[::2])
print(a)
del a[1:3]
del a[::3]
print(a)
print(range(10)[2:8:2], range(10)[::-1], list(range(10)[-3:]))
