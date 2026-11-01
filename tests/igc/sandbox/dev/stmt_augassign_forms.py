a = 1; a += 1; a -= 3; a *= -4; a /= 2; print(a)
b = 7; b //= 2; b %= 2; b **= 3; print(b)
c = 6; c &= 3; c |= 8; c ^= 1; c <<= 2; c >>= 1; print(c)
s = 'a'; s += 'b'; s *= 2; print(s)
t = (1,); t += (2,); print(t)
l = [[1]]; l[0] += [2]; l[0][0] += 5; print(l)
d = {'a': 1}; d['a'] += 1; d['b'] = d.get('b', 0) + 1; print(d)
v = [1, 2]; i = 0; v[i] += 10; print(v)
calls = []
def idx():
    calls.append(1)
    return 0
v[idx()] += 1; print(v, calls)
tt = ([],)
try:
    tt[0] += [1]
except TypeError as e:
    print(e, tt)
u = None
try:
    u += 1
except TypeError as e:
    print(e)
try:
    undefined_aug += 1
except NameError as e:
    print(e)
