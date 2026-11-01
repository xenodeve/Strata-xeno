try:
    raise ValueError('a')
except ValueError as err:
    saved = err
try:
    print(err)
except NameError as n:
    print(n)
print(saved)
err = 'value'
try:
    raise ValueError
except ValueError as err:
    pass
try:
    print(err)
except NameError:
    print('deleted again')
def f():
    try:
        raise KeyError
    except KeyError as e:
        return 'r'
print(f())
for i in range(2):
    try:
        raise ValueError
    except ValueError as e:
        if i == 0:
            continue
        break
try:
    e
except NameError:
    print('gone after break')
try:
    try:
        raise ValueError('first')
    except ValueError as q:
        raise KeyError('second')
except KeyError:
    try:
        q
    except NameError:
        print('q gone')
