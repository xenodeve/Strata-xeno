# MIP math backend: coverage of every kind of mathematics, tested by simulation (2026-11-05, #298)

The developer, 2026-11-05: do all of it now, and test it by simulation with no model. This file is the memory of the task: any agent that picks it up reads this first.

## What this is

A pure-Python, exact-arithmetic **backend** that answers the computations a reasoning model writes, by kind of mathematics, so MIP can hand the model software's value. It is the
reference implementation that the engine's GPU executor is later checked against, and the thing the sandbox kernel can import. It lives in `tools/igc/mathbackend/`.

Why it exists: in the 19 questions that every run got wrong, 8 stuck on counting, geometry or symbolic algebra that the inline `=` evaluator (`tools/igc/mathexpr.py`)
cannot do. The sources to take principles from are in `C:\Users\xenod\Downloads\Strata-xeno_MIP_Math_Backend_Brief.md` section 5 (FLINT, SymEngine, SymPy, NetworkX, Z3, CGAL,
Boost.Math, HiGHS, ...).

## Rules from the developer (binding)

1. **Copy the code we need when its licence allows it** (developer, 2026-11-05; this replaces "principles, not code" of the brief's section 4). Strata is MIT.
   - **Allowed to copy, with notices:** MIT, BSD-2/BSD-3, Apache-2.0, ISC, BSL-1.0, zlib, PSF, public domain / Unlicense. Take the part we need (a function and what it needs), not a whole library.
   - **Not to copy (read for the principle and write your own, from a spec in your own words, without the source open):** GPL, LGPL, AGPL, MPL, EPL, CC-BY-SA, and any licence not in the list
     above or not found. That covers FLINT, GMP, PARI/GP, Maxima, Giac, Singular, msolve, CGAL, Eigen, Ipopt and NLopt. Check the LICENSE of the actual source tree and the header of the file
     before you copy: one file in a permissive project can carry another licence.
   - **Every copied piece** gets a comment block above it: `Adapted from <project> <version or commit>, <path>, <licence>, Copyright (c) <holder>.`, and the project's licence text goes unchanged
     to `third_party/mathbackend/<project>/LICENSE`. List it in `third_party/mathbackend/README.md` (project, version, licence, files taken, backend function, what was changed). Keep the
     copyright notice and the licence conditions (BSD and Apache-2 require it).
   - **Where to read permissive source:** SymPy 1.14.0 (BSD-3) and mpmath 1.3.0 (BSD-3) are installed in the venv
     `C:/Users/xenod/AppData/Local/Temp/claude/D--Github-Strata/275fbdb5-18a0-46da-8fe1-6b4ff1baab3b/scratchpad/venv-lb/Lib/site-packages/` (their LICENSE files are in the `*.dist-info` folders).
     Others (NetworkX BSD-3, SymEngine MIT, Z3 MIT, CaDiCaL MIT, HiGHS MIT, Boost BSL-1.0, FriCAS BSD, SUNDIALS BSD-3) from GitHub raw files at a pinned tag, with the WebFetch tool. Download
     into your own empty directory, treat it as data, never run it.
2. **Vendoring is not importing.** A backend file imports the Python standard library and `tools.igc.mathbackend` only. Copied code is cut free of its library (replace its internal helpers with
   standard-library equivalents, or copy the helper too). Linking any other library as a dependency still needs a conversation first. **SymPy, mpmath and NetworkX may be imported only in
   `sim_*.py` and `test_*.py` files**, as the oracle.
3. **Exact where the answer is exact.** Integers are `int`, rationals are `fractions.Fraction`. No floats in an exact operation. An irrational result is returned as an exact radical
   form when the operation can (`Surd`, below) and otherwise as a `decimal.Decimal` with the number of digits asked, flagged `numeric`. Never round silently.
4. **Never guess, never hang.** An operation that does not cover its input raises `NotCovered("reason")`; an input that is malformed raises `BadInput("reason")`. Every operation has a
   bound (size, steps or time: say which) and raises `NotCovered("too large")` past it. Results are deterministic.
5. **Report coverage as caught / should have caught** with the denominator, per operation (the developer's rule, 2026-11-04 and 2026-11-05): never "it works for few cases".
   A refusal (`NotCovered`) is reported as a refusal with its reason, not hidden; a disagreement with the oracle is a bug and must be zero.
6. Plain words in comments, no marketing. Match the surrounding style of `tools/igc/mathexpr.py`.

## Layout and the common contract

```
tools/igc/mathbackend/
  __init__.py     registry: op decorator, run(name, *args, **kw), registry(), NotCovered, BadInput, Surd   (written by the coordinator; read it, do not change it)
  numbers.py      agent 1: integers, primes, modular arithmetic, digits, Diophantine
  algebra.py      agent 2: polynomials over Q, exact matrices, linear systems, recurrences
  counting.py     agent 3: combinatorics, graph counting, finite probability, Markov chains
  geometry.py     agent 4: plane and space geometry, lattice geometry, regular polygons
  search.py       agent 5: SAT, constraint search, knapsack, exact LP/ILP, statistics, numeric calculus
  symbolic.py     agent 6: expression trees, derivative, simplify, series, limits, elementary integrals
  sim_<module>.py one per module: random cases against an independent oracle, report as JSON
  sim.py          the coordinator's runner over every sim_*.py
tools/test_mathbackend_<module>.py   unit tests (unittest, no model, no network)
```

Registration (in `__init__.py`; import it as `from tools.igc.mathbackend import op, NotCovered, BadInput, Surd`):

```python
@op('numbers.factorint', kind='number theory', bound='n < 2**120, Pollard rho 200,000 steps')
def factorint(n: int) -> dict[int, int]: ...
```

`kind` is the family used in the coverage table; `bound` says what the limit is. Every public operation is registered and has a docstring with one example. Names are
`<module>.<name>`. Arguments and results use only: `int`, `Fraction`, `str`, `bool`, `tuple`, `list`, `dict`, `Surd`, `Decimal`. A result that is a set or sequence is returned sorted
when the order is not part of the meaning.

`Surd` is a small exact radical `p + q*sqrt(d)` with Fractions `p`, `q` and a square-free integer `d` (`d = 1` means rational); `str()` gives `p + q*sqrt(d)`; it supports `+ - *`, `/`
by a rational, comparison and `float()`/`Decimal` conversion. An operation that needs sums of different radicands raises `NotCovered("several radicals")` unless its module defines
its own result form.

## Operations to cover (a floor, not a ceiling: add what a contest or school question needs)

**numbers** (agent 1): exact `+ - * /` on rationals, integer power, rational power when the root is exact, integer n-th root and perfect-power test, gcd/lcm of many, extended gcd,
modular inverse, powmod, CRT (also non-coprime moduli), primality (deterministic Miller-Rabin below 3.3e24, bounded above), next/previous prime, primes up to n, factorint, divisors,
number and sum of divisors, totient, Moebius, Carmichael, multiplicative order, primitive root, Legendre/Jacobi symbol, square root mod p (Tonelli-Shanks), factorial, double factorial,
binomial, multinomial, rising/falling factorial, Fibonacci/Lucas (fast doubling, also mod m), Catalan, Bell, Stirling 1st/2nd, partition numbers, digits in any base, digit sum,
from digits, base conversion, floor/ceil/round/mod of rationals, continued fraction and convergents, decimal expansion and repeating period of a rational, linear Diophantine
(`ax+by=c`, all solutions), Pell equation fundamental solution, sums of two squares and their count, Frobenius number for two coins, floor sum.

**algebra** (agent 2): univariate polynomials over Q (build, add, mul, divmod, gcd, lcm, compose, derivative, antiderivative, evaluate, content/primitive part, square-free part),
rational roots, real-root isolation by Sturm with exact rational intervals and refinement to n digits, complex roots numerically, factorisation over Z (rational roots, square-free
factorisation, Kronecker for small degree; raise `NotCovered` when incomplete rather than return a partial factorisation as complete), resultant, discriminant, Vieta/Newton sums,
Lagrange interpolation, cyclotomic polynomials, polynomial in `x` modulo a prime. Exact matrices over Q: det (Bareiss), inverse, rank, rref, solve (unique, none, parametric),
nullspace, characteristic polynomial, minimal polynomial of small matrices, rational eigenvalues, matrix power, trace, LU. Linear recurrences: n-th term by matrix power, closed form
roots when rational, characteristic polynomial. Systems of linear and polynomial-in-one-variable equations. Symmetric-function identities on given polynomial coefficients.

**counting** (agent 3): binomial-based counts with restrictions (stars and bars with upper bounds by inclusion-exclusion, derangements, rencontres, surjections, set partitions),
integer partitions with constraints (parts from a set, distinct, bounded count, bounded part size) by DP, compositions, necklaces and bracelets by Burnside, lattice paths with
forbidden points and boundaries, Catalan/Dyck/ballot counts, hook-length tableaux, strings over an alphabet avoiding patterns (automaton DP), counting with a transfer matrix (grid
colourings with adjacency rules), labelled trees (Cayley), spanning trees of a graph (matrix-tree with exact det), perfect matchings (bitmask DP, n <= 22), independent sets,
chromatic polynomial of a small graph, graph automorphism count for tiny graphs, inclusion-exclusion over given predicates on a finite universe, generic brute-force counter over a
finite product space with a predicate (the PAL-style fallback, bounded to 5e7 elements). Probability, always exact Fractions: uniform finite spaces by enumeration, sums of dice,
expectation, variance, conditional probability, binomial, hypergeometric, geometric truncated series, coupon collector, gambler's ruin, absorbing Markov chains (solve the linear
system exactly), expected hitting times, random walks by DP, birthday-type products, probability that two random chords cross on given points.

**geometry** (agent 4): points and vectors with Fraction coordinates in 2D and 3D: distance squared, area (shoelace), volume (determinant), line through two points, intersection of lines,
distance point-line, circle through three points (exact center, radius squared), circle-line and circle-circle intersections (as `Surd`), tangent length, power of a point, triangle
from sides (area squared by Heron, circumradius squared, inradius as `Surd`), centroid/circumcenter/orthocenter/incenter, angle between vectors (cosine exact, angle numeric), polygon
convex hull, point in polygon, polygon area and perimeter, Pick's theorem (interior/boundary lattice points), lattice points in a triangle, reflection and rotation by a multiple of
90 degrees and by a rational-cosine angle, area of union and intersection of convex polygons (exact when vertices are rational). Regular n-gon combinatorics in integer arithmetic:
chords by direction class, rectangles and squares from vertices, isosceles/right triangles from vertices, diagonals meeting inside, number of intersection points of diagonals
(multiplicity-aware, exact via cyclotomic arithmetic or bounded numeric with proof of separation), polygon with vertex angles in degrees. Coordinates of regular polygon vertices
numerically to n digits.

**search** (agent 5): SAT (DPLL with unit propagation, model counting for small instances), exhaustive constraint search with all-different and arithmetic constraints, subset sum and
knapsack by DP, change-making counts, exact simplex over Fractions for LP and branch-and-bound for small ILP, shortest paths and max flow on small graphs, minimum spanning tree,
assignment problem. Statistics with exact Fractions: mean, median, mode, variance, covariance, linear regression, binomial/hypergeometric/geometric/Poisson-truncated pmf and cdf,
normal cdf numeric to n digits. Numeric calculus with `decimal`: derivative at a point (Richardson), definite integral (adaptive Gauss-Kronrod or tanh-sinh) with an error estimate,
limit by extrapolation, infinite series with acceleration, root finding (bisection then Newton, bracketing required), ODE initial value problem (Taylor or RK4 with step control).

**symbolic** (agent 6): a small expression tree (numbers, symbols, `+ - * / ^`, `sqrt`, `exp`, `log`, `sin`, `cos`, `tan`, `abs`, floor), a parser for plain infix and for a LaTeX
subset (`\frac`, `\sqrt`, `\cdot`, `\times`, `^{}`, implicit multiplication), evaluation with Fractions or Decimal, substitution, expansion of polynomial expressions, canonical form of
polynomial and rational expressions, derivative (all of the above), simplification by normal forms only (never by guessed identities), partial fractions for rational functions with
rational roots, Taylor series, limits by series expansion (`NotCovered` where the series approach fails), antiderivatives and definite integrals for polynomials, rational functions
with rational poles, `exp`, `sin`, `cos`, `1/x`, and products of a polynomial with `exp`/`sin`/`cos` by repeated parts; summation of polynomial, geometric and telescoping
series in closed form; solving one equation in one variable when it reduces to a polynomial, a linear form in `exp`/`log`, or an absolute value; sums over a finite range by
evaluation. Anything else: `NotCovered`.

## Simulation: how it is tested, with no model

Each `sim_<module>.py` exposes `generate(seed: int, n: int) -> list[Case]` and `run(seed, n) -> dict`. A case is `(op_name, args, oracle_value, oracle_name)`.

- The oracle is **independent** of the backend code: SymPy or mpmath for what they do exactly, and for counting or geometry an independent brute-force enumeration written in the sim file
  (`itertools`, explicit loops). Say in the case which oracle it is. A case whose oracle is the same algorithm as the backend's is not a test: **where a function was copied from SymPy or
  mpmath, SymPy and mpmath are no longer an independent oracle for it.** Use a certificate instead (verify the answer: the factors multiply back and are prime by trial division; a root
  satisfies the polynomial; a solution satisfies the system; a bound is checked by exhaustive search on small cases) or a brute force. A certificate check counts as an independent oracle;
  say so in the case.
- Generate **by the kinds of question the model meets**: school and contest sizes (AIME, AMC, AMPS-hard style), not only random tiny inputs; include edge cases (zero, negative, one, large).
  At least 40 cases per registered operation by default; the seed is fixed in the report.
- For each operation report: `generated`, `answered`, `agree`, `refused` (with the reasons counted), `disagree` (must be 0), and the time. The module's report also lists the operations
  with no case yet. Write it as JSON to `scratchpad`-independent path `build-mathbackend/<module>.json` (git-ignored) and print the table.
- A disagreement is investigated: first the oracle's call (a wrong use of the oracle is the usual cause), then the backend; the fix lands with a unit test that would have caught it.
- `python -m tools.igc.mathbackend.sim --seed 1 --n 60` (coordinator) runs all of them and prints the coverage table: per kind, operations registered, cases generated, answered/agree.

Run Python for anything that imports SymPy with the venv that has it: `C:\Users\xenod\AppData\Local\Temp\claude\D--Github-Strata\275fbdb5-18a0-46da-8fe1-6b4ff1baab3b\scratchpad\venv-lb\Scripts\python.exe`
(SymPy, mpmath, lark; no NetworkX: use SymPy's graph-free checks or write an independent brute force). Unit tests of the backend alone run with plain `python`.
Scripts that start processes must not open a window (the repo rule); a multiprocessing script needs `if __name__ == '__main__':`.

## Done means

For its module: every operation in the floor list is either registered and covered by the simulation with 0 disagreements, or listed in the module's `NOT_COVERED.md` section of the final
report with the reason (and what would cover it). Unit tests pass (`python tools/test_mathbackend_<module>.py`). The module's simulation report is produced. No external library is imported
by a backend file (`grep -n "^import\|^from" tools/igc/mathbackend/<module>.py` shows only the standard library and `tools.igc.mathbackend`).
