x = [1]
del x[0]
print(x)
a = 1
del a
try:
    a
except NameError:
    print('gone')
try:
    assert 1 == 2, 'nope'
except AssertionError as e:
    print('AE', e)
assert True
