for fn in [lambda: (1).x, lambda: 'a'.x, lambda: [].x, lambda: {}.x, lambda: (1,).x, lambda: {1}.x, lambda: None.x, lambda: 1.5.x, lambda: True.x, lambda: range(1).x, lambda: int.x, lambda: str.x, lambda: len.x, lambda: ValueError.x, lambda: ValueError('a').x]:
    try:
        print(fn())
    except AttributeError as e:
        print(e)
print((1).real, (1).imag, (1).numerator, (1).denominator, 1.5.real, 1.5.imag, True.real, (5).bit_length(), (5).bit_count(), (-5).bit_length(), (0).bit_length(), (2 ** 70).bit_length(), (5).conjugate(), 1.5.conjugate(), 1.5.is_integer(), 2.0.is_integer(), (2.5).as_integer_ratio(), (5).as_integer_ratio(), 0.1.as_integer_ratio(), float('inf').is_integer(), float('nan').is_integer(), 'a'.upper, [].append) if False else None
print((1).real, (1).imag, (1).numerator, (1).denominator, 1.5.real, 1.5.imag, True.real, (5).bit_length(), (5).bit_count(), (-5).bit_length(), (0).bit_length(), (2 ** 70).bit_length(), (5).conjugate(), 1.5.conjugate(), 1.5.is_integer(), 2.0.is_integer(), (2.5).as_integer_ratio(), (5).as_integer_ratio(), 0.1.as_integer_ratio(), float('inf').is_integer(), float('nan').is_integer())
