# expect: Unsupported
try:
    next(iter([]))
except StopIteration as e:
    print(e.value)
