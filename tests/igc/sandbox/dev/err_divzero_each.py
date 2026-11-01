for fn in [lambda: 1 / 0, lambda: 1 // 0, lambda: 1 % 0, lambda: 1.0 / 0, lambda: 1.0 // 0, lambda: 1.0 % 0, lambda: divmod(1, 0), lambda: divmod(1.0, 0), lambda: 0 ** -1, lambda: 0.0 ** -1, lambda: 2 ** 200 / 0, lambda: 2 ** 200 // 0]:
    try:
        fn()
    except ZeroDivisionError as e:
        print(e)
