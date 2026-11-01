s = 'abcdef'
print(s[0], s[-6], s[1:1], s[3:1], s[::2], s[1::2], s[::-2], s[-2:], s[:-2], s[100:200], s[-100:-50], s[5:0:-1], s[5:-100:-1], s[None:None], s[None:2:None], s[::1], s[-1::-1], s[2:100:2], s[True:], s[False])
for bad in [lambda: s[6], lambda: s[-7], lambda: s[::0], lambda: s[1.5:], lambda: s['a':]]:
    try:
        print(bad())
    except (IndexError, ValueError, TypeError) as e:
        print(type(e).__name__, e)
