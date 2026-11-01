try:
    raise BaseException('b')
except Exception:
    print('no')
except BaseException as e:
    print(repr(e))
raise GeneratorExit
