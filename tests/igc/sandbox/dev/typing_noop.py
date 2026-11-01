from typing import List, Dict, Tuple, Optional, Any
def f(xs: List[int], d: Dict[str, int] = {}, n: Optional[int] = None) -> Tuple[int, Any]:
    y: int = len(xs)
    return y, n
print(f([1, 2]))
