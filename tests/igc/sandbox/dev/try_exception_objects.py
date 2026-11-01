try:
    raise ValueError('a', 1)
except ValueError as e:
    print(e, repr(e), e.args, str(e), type(e), type(e).__name__, isinstance(e, Exception), len(e.args))
try:
    raise ValueError
except ValueError as e:
    print(repr(e), str(e) == '', e.args)
try:
    raise ValueError()
except Exception as e:
    print(repr(e))
e = KeyError('k')
print(repr(e), str(e), KeyError(), str(KeyError()), KeyError(1, 2), repr(Exception('x')), str(Exception(1)), repr(Exception(None)), str(Exception(None)), repr(TypeError([1])), repr(ZeroDivisionError('z')), AssertionError('a').args, ValueError('m').args[0])
for exc in [ValueError, TypeError, KeyError, IndexError, ZeroDivisionError, OverflowError, AttributeError, NameError, RuntimeError, NotImplementedError, StopIteration, AssertionError, ImportError, ModuleNotFoundError, LookupError, ArithmeticError, UnboundLocalError, RecursionError, MemoryError, OSError, SyntaxError, IndentationError, TabError, UnicodeError, EOFError]:
    try:
        raise exc('m')
    except Exception as e:
        print(type(e).__name__, isinstance(e, LookupError), isinstance(e, ArithmeticError), str(e))
print(issubclass(ZeroDivisionError, ArithmeticError), issubclass(KeyError, LookupError), issubclass(KeyError, Exception), issubclass(bool, int), issubclass(int, bool), issubclass(Exception, BaseException), issubclass(ModuleNotFoundError, ImportError), issubclass(ValueError, (TypeError, ValueError)))
