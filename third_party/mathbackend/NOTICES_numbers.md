# Copied code: module `numbers` (tools/igc/mathbackend/numbers.py)

| project | version | licence | files taken | backend function | changes |
|---|---|---|---|---|---|
| (none) | - | - | - | - | - |

Nothing is copied from another project in `numbers.py`. Every function was written from the textbook algorithm (Miller-Rabin, Baillie-PSW, Pollard p-1 and
Brent rho, Tonelli-Shanks, Pocklington, extended Euclid and CRT, Hermite-Serret for sums of two squares, continued-fraction Pell, Dijkstra on residues for the
Frobenius number, the Euclid-like floor sum, Granville's factorial method for binomials mod prime powers, Euler's pentagonal recurrence), with no source open.

Because nothing is copied, SymPy (BSD-3, 1.14.0) is an independent oracle for this module in `sim_numbers.py`: it is imported there and in no backend file.
So `third_party/mathbackend/<project>/LICENSE` has no entry from this module.
