def f(x):
    out = []
    try:
        out.append('try')
        if x == 1:
            raise ValueError
        if x == 2:
            raise TypeError
        if x == 3:
            return out + ['ret']
    except ValueError:
        out.append('value')
    else:
        out.append('else')
    try:
        try:
            if x == 2:
                raise TypeError
        except KeyError:
            out.append('no')
    except TypeError:
        out.append('type')
    return out
for i in range(4):
    try:
        print(f(i))
    except TypeError:
        print('escaped')
