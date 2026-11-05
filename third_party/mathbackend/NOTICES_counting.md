# NOTICES for `tools/igc/mathbackend/counting.py`

Rows of the copied-code table (README.md) for the `counting` module.

| project | version | licence | files taken | backend function | changes |
|---|---|---|---|---|---|
| (none) | - | - | - | - | No third-party code is copied into `counting.py`. Every operation is written from its definition (Burnside's lemma, the hook-length and hook-content formulas, the matrix-tree theorem with Bareiss elimination, Aho-Corasick, broken-profile DP, inclusion-exclusion, Gauss-Jordan over `Fraction`, the Li-Conway martingale/automaton for coin patterns). |

Not copied, only used as an independent oracle in `sim_counting.py` (the simulation, never imported by the backend): SymPy 1.14.0 (BSD-3-Clause, Copyright (c) SymPy Development Team),
for `bell`, `stirling`, `catalan`, `harmonic`, `subfactorial`, `partition`, `totient`, `Matrix` linear algebra, `interpolate`, `summation`, `fibonacci`, `lucas`. Because nothing from SymPy is
inside `counting.py`, SymPy stays an independent oracle for every operation. No licence text is stored under `third_party/mathbackend/` for this module because no code is taken.
