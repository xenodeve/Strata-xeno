# Notices for tools/igc/mathbackend/search.py

Rows for the copied-code table of `third_party/mathbackend/README.md` (project, version, licence, files taken, backend function, what was changed).

| project | version | licence | files taken | backend function | changes |
|---|---|---|---|---|---|
| (none) | - | - | - | - | No third-party code was copied into `search.py`. |

`search.py` is written for this repository from the textbook algorithms. Every algorithm below is a published method; the code is our own, written without a copy of any library
open, and imports only the Python standard library and `tools.igc.mathbackend`.

| algorithm | where | published method |
|---|---|---|
| DPLL with unit propagation, pure literals, model counting | `sat_solve`, `sat_count` | Davis, Putnam, Logemann, Loveland (1960-62) |
| backtracking constraint search with all-different | `csp_*` | textbook |
| subset sum, knapsack, change-making DP | `subset_sum`, `knapsack`, `coin_change_*` | textbook DP |
| two-phase simplex with Bland's rule over `fractions.Fraction` | `lp_solve` | Dantzig; Bland (1977) |
| branch and bound on the exact LP relaxation | `ilp_solve` | Land and Doig (1960) |
| Dijkstra, Bellman-Ford, Floyd-Warshall | `shortest_path`, `all_pairs_shortest` | textbook |
| Edmonds-Karp | `max_flow` | Edmonds and Karp (1972) |
| Kruskal | `mst` | Kruskal (1956) |
| Hungarian method with potentials | `assignment` | Kuhn (1955), Munkres (1957) |
| tanh-sinh quadrature | `integrate` | Takahasi and Mori (1974) |
| Neville polynomial extrapolation, Richardson, Wynn epsilon | `derivative`, `limit`, `series_sum` | Neville (1934), Richardson (1911), Wynn (1956) |
| Cohen-Villegas-Zagier acceleration of alternating series | `series_sum` | Cohen, Rodriguez Villegas, Zagier, Experimental Mathematics 9 (2000) |
| Gragg-Bulirsch-Stoer extrapolation | `ode_ivp` | Gragg (1965), Bulirsch and Stoer (1966) |
| Machin's formula for pi, Taylor series for sin, cos, atan, the all-positive erf series | decimal functions | textbook |

Not opened and not copied: SymPy 1.14.0 (`sympy/logic/algorithms/dpll2.py`, `sympy/solvers/simplex.py`) and mpmath 1.3.0 (`mpmath/calculus/`), both BSD-3. SymPy and mpmath are
used in `sim_search.py` only, as the independent oracle; because nothing was taken from them, they stay independent oracles for every
operation.
