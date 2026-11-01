print(None is None, None is not None, (lambda: None)() is None, [] is not None, 1 is not None, 'a' is None, True is True, False is False, True is not False, int is int, type(1) is int, [].sort() is None)
a = [1]
b = a
c = [1]
print(a is b, a is c, a is not c, {} is {}, set() is set())
