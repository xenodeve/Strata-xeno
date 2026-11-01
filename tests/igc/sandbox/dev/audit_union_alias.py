from typing import List, Optional
A = int | None
B = None | int
C = List[int] | None
D = int | str
def f(x: A = None) -> B:
    return x
print(f(), f(1))
for fn in [lambda: None | None, lambda: int | 1, lambda: 1 | int, lambda: 'a' | int]:
    try:
        print(fn())
    except TypeError as e:
        print(e)
