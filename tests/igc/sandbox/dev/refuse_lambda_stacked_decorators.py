# expect: Unsupported
from functools import cache
@cache
@cache
def f():
    pass
