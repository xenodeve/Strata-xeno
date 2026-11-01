def parse_num(s):
    for kind in (int, float):
        try:
            return kind(s)
        except ValueError:
            pass
    return None
print([parse_num(x) for x in ['1', '2.5', 'x', '', '1e3', ' 7 ', '0x10', '1_000']])
def safe_div(a, b):
    try:
        return a / b
    except ZeroDivisionError:
        return float('inf')
    except TypeError as e:
        return str(e)
    finally:
        pass
print(safe_div(1, 2), safe_div(1, 0), safe_div('a', 1))
def validate(x):
    if not isinstance(x, int):
        raise TypeError(f'expected int, got {type(x).__name__}')
    if x < 0:
        raise ValueError(f'negative: {x}')
    return x
for v in [1, -1, 'a', 2.5, None, True]:
    try:
        print(validate(v))
    except (TypeError, ValueError) as e:
        print(type(e).__name__ + ':', e)
