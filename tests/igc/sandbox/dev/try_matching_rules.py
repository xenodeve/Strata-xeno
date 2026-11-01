def f(e):
    try:
        raise e
    except LookupError:
        return 'lookup'
    except (ArithmeticError, TypeError):
        return 'arith or type'
    except Exception:
        return 'exception'
print(f(KeyError()), f(IndexError()), f(ZeroDivisionError()), f(TypeError()), f(ValueError()), f(ValueError('x')), f(KeyError))
try:
    try:
        raise ValueError('inner')
    except KeyError:
        print('no')
except ValueError as e:
    print('propagated', e)
try:
    try:
        raise ValueError
    except ValueError:
        raise TypeError('from handler')
except TypeError as e:
    print(e)
try:
    try:
        raise ValueError('v')
    except ValueError as e:
        raise TypeError('t') from e
except TypeError as e:
    print(e)
try:
    raise ValueError('n') from None
except ValueError as e:
    print(e)
try:
    try:
        1 / 0
    except:
        print('bare')
        raise
except ZeroDivisionError as e:
    print(e)
try:
    try:
        raise ValueError
    except 'notatype':
        pass
except TypeError as e:
    print(e)
try:
    try:
        raise ValueError
    except (ValueError, 5):
        print('matched first')
except TypeError as e:
    print('te', e)
try:
    raise 5
except TypeError as e:
    print(e)
try:
    raise ValueError from 5
except TypeError as e:
    print(e)
try:
    raise
except RuntimeError as e:
    print(e)
def reraise_in_call():
    raise
try:
    try:
        raise KeyError('kk')
    except KeyError:
        reraise_in_call()
except KeyError as e:
    print('via call', e)
