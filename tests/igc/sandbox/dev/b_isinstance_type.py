print(isinstance(1, int), isinstance(1, float), isinstance(1.0, float), isinstance(True, bool), isinstance(True, int), isinstance('a', str), isinstance([], list), isinstance((), tuple), isinstance({}, dict), isinstance(set(), set), isinstance(frozenset(), frozenset), isinstance(range(1), range), isinstance(None, type(None)), isinstance(1, (str, int)), isinstance(1, (str, (float, int))), isinstance(int, type), isinstance(ValueError('x'), Exception), isinstance(ValueError('x'), ValueError), isinstance(KeyError(), LookupError), isinstance(1, object), isinstance(len, object))
print(type(1), type(1.0), type('a'), type(None), type([]), type(()), type({}), type(set()), type(True), type(range(1)), type(int), type(1).__name__, type('').__name__, type(None).__name__, type([]) == list, type(1) is int, type(1) is not str, type(ValueError('x')).__name__)
try:
    isinstance(1, 'a')
except TypeError as e:
    print(e)
