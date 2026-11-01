for v in [5, 1.0, None, True, len]:
    try:
        print(len(v))
    except TypeError as e:
        print(e)
print(len(''), len('abc'), len([]), len(()), len({}), len(set()), len(range(0)), len(range(3, 10, 2)), len(range(10, 0, -3)))
