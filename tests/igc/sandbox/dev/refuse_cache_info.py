# expect: Unsupported
from functools import cache
@cache
def f():
    return 1
print(f.cache_info())
