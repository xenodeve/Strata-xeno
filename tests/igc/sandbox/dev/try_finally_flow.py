def a():
    try:
        return 'try'
    finally:
        print('fin a')
def b():
    try:
        raise ValueError('x')
    except ValueError:
        return 'except'
    finally:
        print('fin b')
def c():
    try:
        return 'try'
    finally:
        return 'finally wins'
def d():
    for i in range(3):
        try:
            if i == 1:
                continue
            if i == 2:
                break
            print('body', i)
        finally:
            print('fin', i)
    return i
def e():
    try:
        try:
            raise KeyError('k')
        finally:
            print('inner')
    except KeyError as err:
        print('outer', err)
def f():
    try:
        raise ValueError('lost')
    finally:
        return 'swallowed'
def g():
    while True:
        try:
            raise ValueError
        finally:
            break
    return 'broke out'
def h():
    for i in range(2):
        try:
            pass
        finally:
            continue
    return i
def i():
    x = 1
    try:
        return x
    finally:
        x = 2
def j():
    try:
        return 'a'
    finally:
        try:
            return 'b'
        finally:
            print('nested')
def k():
    for x in [1, 2]:
        for y in [3, 4]:
            try:
                return (x, y)
            finally:
                print('k fin')
print(a(), b(), c(), d(), e(), f(), g(), h(), i(), j(), k())
