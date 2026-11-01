from functools import lru_cache
@lru_cache(maxsize=None)
def count(n, open_=0):
    if n == 0:
        return 1 if open_ == 0 else 0
    return count(n - 1, open_ + 1) + (count(n - 1, open_ - 1) if open_ else 0)
print(count(10), [count(2 * i) for i in range(8)])
