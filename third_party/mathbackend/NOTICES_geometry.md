# Copied code: geometry (tools/igc/mathbackend/geometry.py)

Nothing is copied into the geometry module. Every function was written from the mathematics (the formula is named in its docstring); no file of SymPy, mpmath,
CGAL or any other project was opened while writing it, and none of their code is in `geometry.py`. SymPy 1.14.0 (BSD-3) and mpmath 1.3.0 (BSD-3) are used only
in `sim_geometry.py` as the independent oracle, and are imported there, not vendored.

| project | version | licence | files taken | backend function | changes |
|---|---|---|---|---|---|
| (none) | - | - | - | - | - |

Published results used as test values (numbers, not code): OEIS A006561 and the Poonen-Rubinstein formula (SIAM J. Discrete Math. 11 (1998) 135-156) for the
interior intersection points of the diagonals of a regular n-gon; AIME 2024 I Problem 6 (315 rectangles in a regular dodecagon); Gauss circle counts.
