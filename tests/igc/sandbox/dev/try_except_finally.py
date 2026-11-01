def f(x):
    try:
        if x == 0:
            raise ValueError('zero')
        if x == 1:
            return 'ret'
        return 1 / (x - 2)
    except ValueError as e:
        print('caught', e)
        return 'ex'
    except (ZeroDivisionError, TypeError) as e:
        print(type(e).__name__, e)
    else:
        print('else')
    finally:
        print('finally')
    return 'end'
for i in range(4):
    print(f(i))
