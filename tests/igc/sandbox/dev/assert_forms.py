assert True, 'no'
assert 1
assert [1]
for v, m in [(0, 'zero'), ('', None), ([], [1, 2]), (None, 5)]:
    try:
        if m is None:
            assert v
        else:
            assert v, m
    except AssertionError as e:
        print(repr(e), e.args)
