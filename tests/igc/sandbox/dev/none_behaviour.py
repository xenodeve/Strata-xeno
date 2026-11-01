print(None, repr(None), str(None), bool(None), None == None, None != None, None == 0, None == '', [None] == [None], {None: 1}[None], type(None), [None].count(None), None in [None], sorted([1, 2], key=lambda x: None) if False else 0)
for fn in [lambda: None + 1, lambda: None < 1, lambda: None[0], lambda: len(None), lambda: iter(None), lambda: None(), lambda: int(None), lambda: float(None), lambda: None.x, lambda: abs(None), lambda: sum(None), lambda: sorted(None), lambda: max(None), lambda: range(None), lambda: list(None), lambda: 'a'.join(None)]:
    try:
        print(fn())
    except (TypeError, AttributeError) as e:
        print(type(e).__name__, e)
