for e in [Exception(), Exception('a'), Exception('a', 'b'), Exception(1), Exception(None), Exception([1]), KeyError(), KeyError('a'), KeyError('a', 'b'), KeyError(1), IndexError('x'), OSError('x'), StopIteration(5), SystemExit(1), ValueError(ValueError('n'))]:
    print(repr(str(e)), repr(e), e.args)
print(f'{ValueError("a")}', '%s' % KeyError('k'), '{}'.format(TypeError('t')), str(ValueError('x')) + '!', [ValueError('in list')], (KeyError('k'),), {'e': IndexError()})
