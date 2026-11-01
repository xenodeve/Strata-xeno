print(list(enumerate([])), list(enumerate('ab')), list(enumerate('ab', 5)), list(enumerate('ab', start=-1)))
print(list(zip()), list(zip([1])), list(zip([1, 2], [3])), dict(zip('ab', [1, 2])), list(zip(*[[1, 2], [3, 4]])))
a = iter([1, 2, 3, 4])
print(list(zip(a, a)))
a = iter([1, 2, 3])
b = iter([1])
print(list(zip(a, b)), list(a))
print(list(zip([1, 2], [3, 4], strict=True)))
try:
    list(zip([1, 2], [3], strict=True))
except ValueError as e:
    print(e)
