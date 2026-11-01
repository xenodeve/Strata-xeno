for kw in [dict(sep=1), dict(end=2), dict(bad=3)]:
    try:
        print('x', **kw)
    except TypeError as e:
        print('TypeError', e)
