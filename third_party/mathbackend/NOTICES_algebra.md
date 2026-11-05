# Copied code: module `algebra` (tools/igc/mathbackend/algebra.py)

Nothing is copied. Every function in `algebra.py` was written from the textbook algorithm (Euclid and Yun for gcd and square-free decomposition, Sturm's theorem,
Cantor-Zassenhaus and Hensel lifting with Zassenhaus recombination for factoring, Bareiss, Faddeev-LeVerrier, Berlekamp-Massey, Newton's identities, Kitamasa),
without the source of SymPy, mpmath or any other project open. The backend file imports the Python standard library and `tools.igc.mathbackend` only.

SymPy 1.14.0 (BSD-3-Clause) and mpmath 1.3.0 (BSD-3-Clause) are imported only by `sim_algebra.py` and are used there as independent oracles (they are not part of
the backend and none of their code is in it). So there are no rows for the table in `third_party/mathbackend/README.md` and no `LICENSE` folder for this module.

| project | version | licence | files taken | backend function | changes |
|---|---|---|---|---|---|
| (none) | | | | | |
