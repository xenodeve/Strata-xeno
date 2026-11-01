try:
    try:
        raise KeyError('k')
    except KeyError as e:
        print(repr(e), e, e.args)
        raise
except Exception as e2:
    print('outer', type(e2).__name__)
try:
    e
except NameError as n:
    print(n)
for i in range(3):
    try:
        if i == 1:
            continue
        print(i)
    finally:
        print('f', i)
