import typing
from typing import List, Dict, Set, Tuple, Optional, Union, Any, Callable, Iterable, Iterator, Sequence, Mapping, TypeVar
T = TypeVar if False else None
Vec = List[int]
Grid = List[List[Optional[int]]]
def f(a: Vec, b: 'Grid' = None, *c: int, d: Dict[str, Any] = {}, **e: float) -> Optional[Union[int, str]]:
    x: int = 1
    y: List[int] = []
    z: Tuple[int, ...]
    return x
print(f([]), typing.TYPE_CHECKING)
def g(fn: Callable[[int], int], it: Iterable[int]) -> Iterator[int]:
    return map(fn, it)
print(list(g(lambda x: x + 1, [1])))
h: 'int' = 5
print(h)
def k(a: list[int], b: dict[str, int], c: tuple[int, ...], d: int | None = None) -> set[int]:
    return {1}
print(k([], {}, ()))
