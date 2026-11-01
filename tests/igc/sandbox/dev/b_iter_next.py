it = iter([1, 2])
print(next(it), next(it), next(it, 'done'))
try:
    next(it)
except StopIteration as e:
    print('stop', repr(e), e.args)
try:
    next([1])
except TypeError as e:
    print(e)
try:
    iter(5)
except TypeError as e:
    print(e)
s = iter('ab')
d = iter({'x': 1})
t = iter((1,))
print(next(s), next(d), next(t), list(s), list(d))
it = iter([1, 2, 3])
print(next(it), list(it), list(it))
it = iter(range(3))
print(iter(it) is it, sum(it), sum(it))
