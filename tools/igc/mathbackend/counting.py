"""MIP math backend, module `counting`: exact counting and exact finite probability. Standard library only; every operation is registered as `counting.<name>`.

Written from the definitions (no third-party code is copied; see third_party/mathbackend/NOTICES_counting.md). Integers are `int`, probabilities and expectations are `Fraction`.
Every operation has a bound and raises NotCovered("too large: ...") past it; a malformed input raises BadInput.

Families (the `kind` of each operation):
  combinatorics   stars and bars, derangements, rencontres, surjections, set partitions, 12-fold way, permutation statistics, anagrams, subsets with a sum
  partitions      integer partitions with constraints, compositions, hook-length tableaux
  symmetry        necklaces and bracelets (Burnside), colourings up to a permutation group
  paths           lattice paths, Dyck paths, ballot, Motzkin, Schroeder, Narayana
  strings         strings avoiding / containing patterns (Aho-Corasick DP), transfer-matrix walks, grid colourings, domino tilings, digit constraints
  graphs          spanning trees (matrix-tree), labelled trees (Cayley), perfect matchings, matchings, independent sets, chromatic polynomial, automorphisms
  inclusion-exclusion   union / complement sizes, integers divisible by given numbers, integers coprime to m
  brute force     a bounded counter over a finite space with a predicate in a small expression language
  probability     uniform spaces, dice, distributions, Bayes, binomial, hypergeometric, geometric, coupon collector, gambler's ruin, Markov chains, patterns in a coin sequence,
                  random walks, birthday, chords

Predicates are strings in a small language (comparisons, `and or not`, `+ - * // % **` with a constant exponent, `& | ^`, tuple indexing and a few functions:
abs min max sum len gcd lcm isqrt comb sorted set tuple is_prime is_square dsum cnt). The text is parsed with `ast` and every node is checked against a whitelist
before anything is compiled, so nothing else can run. Variables are x0, x1, ... (or `names`) and `v`, the whole tuple. `/` is refused: use `//` and `%`.
"""
import ast
import itertools
import keyword
import math
from fractions import Fraction
from functools import lru_cache
from math import comb, factorial, gcd

from tools.igc.mathbackend import BadInput, NotCovered, op

BRUTE_LIMIT = 5 * 10 ** 7        # elements a brute-force enumeration may visit
WORK_LIMIT = 10 ** 7         # elementary steps a DP may take


# ---------------------------------------------------------------- input checks

def _int(x, name):
    if isinstance(x, bool) or not isinstance(x, int):
        raise BadInput(f'{name} must be an integer')
    return x


def _nat(x, name, hi=None):
    _int(x, name)
    if x < 0:
        raise BadInput(f'{name} must be >= 0')
    if hi is not None and x > hi:
        raise NotCovered(f'too large: {name} > {hi}')
    return x


def _frac(x, name):
    if isinstance(x, float):
        raise BadInput(f'{name}: a float is not exact, give an int, a Fraction or "a/b"')
    try:
        return Fraction(x)
    except (ValueError, TypeError, ZeroDivisionError):
        raise BadInput(f'{name} is not a rational number')


def _prob(x, name):
    p = _frac(x, name)
    if not 0 <= p <= 1:
        raise BadInput(f'{name} must be between 0 and 1')
    return p


def _work(w, what):
    if w > WORK_LIMIT:
        raise NotCovered(f'too large: {what} needs about {w} steps (limit {WORK_LIMIT})')


def _factor_small(n):
    """Prime factorisation of n (n <= 10**13) by trial division."""
    f, p = {}, 2
    while p * p <= n:
        while n % p == 0:
            f[p] = f.get(p, 0) + 1
            n //= p
        p += 1 if p == 2 else 2
    if n > 1:
        f[n] = f.get(n, 0) + 1
    return f


def _divisors(n):
    ds = [1]
    for p, e in _factor_small(n).items():
        ds = [d * p ** i for d in ds for i in range(e + 1)]
    return sorted(ds)


def _phi(n):
    r = n
    for p in _factor_small(n):
        r = r // p * (p - 1)
    return r


def _multinomial(parts):
    total, r = 0, 1
    for a in parts:
        total += a
        r *= comb(total, a)
    return r


def _graph(n, edges, maxn, multi=False):
    """n vertices 0..n-1 and a list of edges (u, v); returns (n, sorted edges with u < v). Self-loops and (unless multi) repeated edges are BadInput."""
    n = _nat(n, 'n', maxn)
    if not isinstance(edges, (list, tuple)):
        raise BadInput('edges must be a list of pairs')
    out, seen = [], set()
    for e in edges:
        if not isinstance(e, (list, tuple)) or len(e) != 2:
            raise BadInput('an edge is a pair (u, v)')
        u, v = _int(e[0], 'vertex'), _int(e[1], 'vertex')
        if not (0 <= u < n and 0 <= v < n):
            raise BadInput(f'edge ({u}, {v}) has a vertex outside 0..{n - 1}')
        if u == v:
            raise BadInput('a self-loop is not allowed')
        e = (min(u, v), max(u, v))
        if e in seen and not multi:
            raise BadInput(f'edge {e} repeated')
        seen.add(e)
        out.append(e)
    return n, sorted(out)


def _adj_masks(n, edges):
    adj = [0] * n
    for u, v in edges:
        adj[u] |= 1 << v
        adj[v] |= 1 << u
    return adj


# ---------------------------------------------------------------- combinatorics with restrictions

def _bound_list(b, k, name, default):
    if b is None:
        return [default] * k
    if isinstance(b, int) and not isinstance(b, bool):
        return [b] * k
    if isinstance(b, (list, tuple)) and len(b) == k:
        return [default if x is None else _int(x, name) for x in b]
    raise BadInput(f'{name} must be an integer or a list of {k} integers')


@op('counting.stars_and_bars', kind='combinatorics', bound='sum after shifting <= 10**6 without upper bounds; (sum+1)*k <= 1e7 with upper bounds')
def stars_and_bars(n, k, upper=None, lower=0):
    """Number of integer solutions of x1+...+xk = n with lower <= xi <= upper (each a number or a list of k, None = unbounded).
    stars_and_bars(10, 3) = 66; stars_and_bars(10, 3, upper=4) = 6."""
    n, k = _int(n, 'n'), _nat(k, 'k', 10 ** 6)
    lo = _bound_list(lower, k, 'lower', 0)
    n2 = n - sum(lo)
    if n2 < 0:
        return 0
    if upper is None or (isinstance(upper, (list, tuple)) and all(u is None for u in upper)):
        if k == 0:
            return int(n2 == 0)
        _nat(n2, 'n', 10 ** 6)
        return comb(n2 + k - 1, k - 1)
    up = _bound_list(upper, k, 'upper', n2 + max(lo, default=0))
    u = [a - b for a, b in zip(up, lo)]
    if any(x < 0 for x in u):
        return 0
    if k == 0:
        return int(n2 == 0)
    _work((n2 + 1) * k, 'stars_and_bars')
    dp = [0] * (n2 + 1)
    dp[0] = 1
    for ui in u:
        pre = [0]
        for x in dp:
            pre.append(pre[-1] + x)
        dp = [pre[s + 1] - pre[max(0, s - ui)] for s in range(n2 + 1)]
    return dp[n2]


@op('counting.derangements', kind='combinatorics', bound='n <= 5000')
def derangements(n):
    """Permutations of n with no fixed point. derangements(4) = 9."""
    n = _nat(n, 'n', 5000)
    a, b = 1, 0           # D(0), D(1)
    if n == 0:
        return 1
    for i in range(2, n + 1):
        a, b = b, (i - 1) * (a + b)
    return b


@op('counting.rencontres', kind='combinatorics', bound='n <= 5000')
def rencontres(n, k):
    """Permutations of n with exactly k fixed points. rencontres(4, 1) = 8."""
    n, k = _nat(n, 'n', 5000), _nat(k, 'k')
    if k > n:
        return 0
    return comb(n, k) * derangements(n - k)


@op('counting.surjections', kind='combinatorics', bound='k <= 2000, n <= 5000')
def surjections(n, k):
    """Functions from an n-set onto a k-set (inclusion-exclusion). surjections(4, 2) = 14."""
    n, k = _nat(n, 'n', 5000), _nat(k, 'k', 2000)
    if k > n:
        return 0
    return sum((-1) ** j * comb(k, j) * (k - j) ** n for j in range(k + 1))


@op('counting.set_partitions', kind='combinatorics', bound='n <= 1500; with block_sizes (n+1)*(k+1)*len(sizes) <= 1e7')
def set_partitions(n, k=None, block_sizes=None):
    """Partitions of an n-set into k blocks (k=None: any number: the Bell number), optionally with every block size in block_sizes.
    set_partitions(5, 2) = 15; set_partitions(5) = 52; set_partitions(6, block_sizes=[2, 3]) = 25."""
    n = _nat(n, 'n', 1500)
    if k is not None:
        k = _nat(k, 'k')
    if block_sizes is not None:
        sizes = sorted({_nat(s, 'block size') for s in block_sizes if s})
        if any(s < 1 for s in sizes) or len(sizes) != len(set(block_sizes)):
            raise BadInput('block sizes must be positive and distinct')
        kk = n if k is None else k
        if k is not None and k > n:
            return 0
        _work((n + 1) * (kk + 1) * max(1, len(sizes)), 'set_partitions')
        # a[m][j]: partitions of an m-set into j blocks; the block of the smallest element has size s
        a = [[0] * (kk + 1) for _ in range(n + 1)]
        a[0][0] = 1
        for m in range(1, n + 1):
            for s in sizes:
                if s > m:
                    break
                c = comb(m - 1, s - 1)
                prev = a[m - s]
                row = a[m]
                for j in range(1, kk + 1):
                    if prev[j - 1]:
                        row[j] += c * prev[j - 1]
        return sum(a[n]) if k is None else a[n][k]
    if k is not None and k > n:
        return 0
    row = _stirling_row(n)
    return sum(row) if k is None else (row[k] if k < len(row) else 0)


def _stirling_row(n):
    """Stirling numbers of the second kind S(n, 0..n)."""
    row = [1]
    for m in range(1, n + 1):
        new = [0] * (m + 1)
        for j in range(1, m + 1):
            new[j] = (j * row[j] if j < len(row) else 0) + row[j - 1]
        row = new
    return row


@op('counting.twelvefold_way', kind='combinatorics', bound='n <= 1500; partitions n <= 3000')
def twelvefold_way(n, k, balls='labelled', boxes='labelled', rule='any'):
    """Ways to put n balls into k boxes. balls and boxes are 'labelled' or 'identical'; rule is 'any', 'injective' (at most one per box) or 'surjective' (no empty box).
    twelvefold_way(4, 2, 'labelled', 'labelled', 'surjective') = 14; twelvefold_way(4, 2, 'identical', 'identical', 'any') = 3."""
    n, k = _nat(n, 'n', 3000), _nat(k, 'k', 3000)
    if balls not in ('labelled', 'identical') or boxes not in ('labelled', 'identical') or rule not in ('any', 'injective', 'surjective'):
        raise BadInput("balls/boxes: 'labelled' or 'identical'; rule: 'any', 'injective', 'surjective'")
    if balls == 'labelled' and boxes == 'labelled':
        if rule == 'any':
            if n * k.bit_length() > 10 ** 7:
                raise NotCovered('too large: k**n has more than 1e7 bits')
            return k ** n
        if rule == 'injective':
            return math.perm(k, n) if n <= k else 0
        return surjections(n, k)
    if balls == 'identical' and boxes == 'labelled':
        if rule == 'any':
            return comb(n + k - 1, n) if k > 0 else int(n == 0)
        if rule == 'injective':
            return comb(k, n)
        return comb(n - 1, k - 1) if n >= 1 and k >= 1 else int(n == 0 and k == 0)
    if balls == 'labelled' and boxes == 'identical':
        if rule == 'any':
            if n > 1500:
                _too_large('n', 1500)
            return sum(_stirling_row(n)[:k + 1])
        if rule == 'injective':
            return int(n <= k)
        return set_partitions(n, k)
    if rule == 'any':
        return partition_count(n, max_parts=k)
    if rule == 'injective':
        return int(n <= k)
    return partition_count(n, num_parts=k)


def _too_large(name, hi):
    raise NotCovered(f'too large: {name} > {hi}')


@op('counting.permutations_with_cycle_type', kind='combinatorics', bound='n <= 2000')
def permutations_with_cycle_type(cycle_type):
    """Permutations of n = sum(cycle_type) whose cycle lengths are exactly cycle_type (a list). permutations_with_cycle_type([2, 2, 1]) = 15."""
    if not isinstance(cycle_type, (list, tuple)):
        raise BadInput('cycle_type is a list of positive integers')
    ct = [_nat(c, 'cycle length') for c in cycle_type]
    if any(c < 1 for c in ct):
        raise BadInput('cycle lengths are positive')
    n = _nat(sum(ct), 'n', 2000)
    den = 1
    for length in set(ct):
        m = ct.count(length)
        den *= length ** m * factorial(m)
    return factorial(n) // den


@op('counting.eulerian', kind='combinatorics', bound='n <= 1000')
def eulerian(n, k):
    """Permutations of n with exactly k descents. eulerian(4, 1) = 11."""
    n, k = _nat(n, 'n', 1000), _nat(k, 'k')
    if n == 0:
        return int(k == 0)
    if k >= n:
        return 0
    row = [1]
    for m in range(2, n + 1):
        new = [0] * m
        for j in range(m):
            a = (j + 1) * row[j] if j < len(row) else 0
            b = (m - j) * row[j - 1] if 1 <= j <= len(row) else 0
            new[j] = a + b
        row = new
    return row[k] if k < len(row) else 0


@op('counting.mahonian', kind='combinatorics', bound='n * k <= 1e7')
def mahonian(n, k):
    """Permutations of n with exactly k inversions. mahonian(4, 3) = 6."""
    n, k = _nat(n, 'n', 2000), _nat(k, 'k')
    if k > n * (n - 1) // 2:
        return 0
    _work(n * (k + 1), 'mahonian')
    dp = [0] * (k + 1)
    dp[0] = 1
    for i in range(2, n + 1):
        pre = [0]
        for x in dp:
            pre.append(pre[-1] + x)
        dp = [pre[s + 1] - pre[max(0, s - (i - 1))] for s in range(k + 1)]
    return dp[k]


@op('counting.anagram_count', kind='combinatorics', bound='len(word) <= 400; states * letters**2 <= 1e7')
def anagram_count(word, no_adjacent_equal=False):
    """Distinct arrangements of the letters of a word, optionally with no two equal letters side by side.
    anagram_count('MISSISSIPPI') = 34650; anagram_count('MISSISSIPPI', True) = 2016."""
    if not isinstance(word, str):
        raise BadInput('word must be a string')
    if len(word) > 400:
        raise NotCovered('too large: word longer than 400')
    cs = tuple(sorted(word.count(c) for c in set(word)))
    if not no_adjacent_equal:
        return _multinomial(cs)
    states = 1
    for c in cs:
        states *= c + 1
    _work(states * len(cs) * len(cs), 'anagram_count')

    @lru_cache(maxsize=None)
    def go(rem, last):
        if not any(rem):
            return 1
        t = 0
        for i, r in enumerate(rem):
            if r and i != last:
                t += go(rem[:i] + (r - 1,) + rem[i + 1:], i)
        return t
    try:
        return go(cs, -1)
    finally:
        go.cache_clear()


@op('counting.count_subsets_with_sum', kind='combinatorics', bound='len(values) <= 3000; distinct states <= 3e6')
def count_subsets_with_sum(values, target=None, size=None, mod=None, rem=0):
    """Subsets (by position) of a list of integers whose sum equals target, or whose sum % mod == rem; optionally only subsets with `size` elements.
    count_subsets_with_sum([1, 2, 3, 4, 5], target=6) = 3; count_subsets_with_sum(list(range(1, 11)), mod=5, rem=0) = 208."""
    if not isinstance(values, (list, tuple)):
        raise BadInput('values must be a list of integers')
    vals = [_int(v, 'value') for v in values]
    if len(vals) > 3000:
        raise NotCovered('too large: more than 3000 values')
    if (target is None) == (mod is None):
        raise BadInput('give exactly one of target and mod')
    if mod is not None:
        mod = _nat(mod, 'mod', 10 ** 6)
        if mod < 1:
            raise BadInput('mod must be positive')
        rem = _int(rem, 'rem') % mod
    else:
        _int(target, 'target')
    if size is not None:
        _nat(size, 'size')
    dp = {(0, 0): 1}
    for v in vals:
        new = dict(dp)
        for (s, c), w in dp.items():
            if size is not None and c >= size:
                continue
            s2 = (s + v) % mod if mod else s + v
            key = (s2, c + 1 if size is not None else 0)
            new[key] = new.get(key, 0) + w
        dp = new
        if len(dp) > 3 * 10 ** 6:
            raise NotCovered('too large: more than 3e6 distinct (sum, size) states')
    goal = rem if mod is not None else target
    return sum(w for (s, c), w in dp.items() if s == goal and (size is None or c == size))


# ---------------------------------------------------------------- partitions, compositions, tableaux

@op('counting.partition_count', kind='partitions', bound='n <= 3000; sizes*n <= 1e7 (with a part count: sizes*n*parts <= 1e7)')
def partition_count(n, parts=None, distinct=False, num_parts=None, max_parts=None, max_part=None, min_part=None):
    """Integer partitions of n. parts: allowed part sizes (default all); distinct: no repeated part; num_parts: exactly that many parts; max_parts: at most that many;
    max_part / min_part: bounds on the part sizes. partition_count(10) = 42; partition_count(10, distinct=True) = 10; partition_count(10, num_parts=3) = 8;
    partition_count(20, parts=[1, 5, 10]) = 9."""
    n = _nat(n, 'n', 3000)
    if num_parts is not None and max_parts is not None:
        raise BadInput('give num_parts or max_parts, not both')
    K = None
    if num_parts is not None:
        K = _nat(num_parts, 'num_parts')
    if max_parts is not None:
        K = _nat(max_parts, 'max_parts')
    exact = num_parts is not None
    plain = parts is None and not distinct and max_part is None and min_part is None
    if plain and K is not None:
        # a partition with at most K parts <-> a partition with parts <= K (conjugation); exactly K parts: take 1 from each
        if exact:
            if K == 0:
                return int(n == 0)
            return partition_count(n - K, max_part=K) if n >= K else 0
        return partition_count(n, max_part=K) if K > 0 else int(n == 0)
    sizes = range(1, n + 1) if parts is None else sorted({_nat(p, 'part') for p in parts})
    if any(p < 1 for p in sizes):
        raise BadInput('parts must be positive')
    lo = 1 if min_part is None else _nat(min_part, 'min_part')
    hi = n if max_part is None else _nat(max_part, 'max_part')
    sizes = [p for p in sizes if lo <= p <= min(hi, n)]
    if K is None:
        _work(len(sizes) * (n + 1), 'partition_count')
        dp = [0] * (n + 1)
        dp[0] = 1
        for p in sizes:
            if distinct:
                for s in range(n, p - 1, -1):
                    dp[s] += dp[s - p]
            else:
                for s in range(p, n + 1):
                    dp[s] += dp[s - p]
        return dp[n]
    K = min(K, n)
    if exact and num_parts > n:
        return 0
    _work(len(sizes) * (n + 1) * (K + 1), 'partition_count')
    dp = [[0] * (n + 1) for _ in range(K + 1)]
    dp[0][0] = 1
    for p in sizes:
        if distinct:
            for c in range(K, 0, -1):
                src, dst = dp[c - 1], dp[c]
                for s in range(n, p - 1, -1):
                    if src[s - p]:
                        dst[s] += src[s - p]
        else:
            for c in range(1, K + 1):
                src, dst = dp[c - 1], dp[c]
                for s in range(p, n + 1):
                    if src[s - p]:
                        dst[s] += src[s - p]
    if exact:
        return dp[min(num_parts, K)][n] if num_parts <= K else 0
    return sum(dp[c][n] for c in range(K + 1))


@op('counting.compositions_count', kind='partitions', bound='n * sizes <= 1e7 (with k: k * n * sizes <= 1e7)')
def compositions_count(n, k=None, parts=None, min_part=1, max_part=None):
    """Ordered sums (compositions) of n into k parts (k=None: any number), each part in `parts` or in [min_part, max_part].
    compositions_count(5) = 16; compositions_count(10, 3) = 36; compositions_count(10, parts=[1, 2]) = 89."""
    n = _nat(n, 'n', 200000)
    if parts is not None:
        sizes = sorted({_nat(p, 'part') for p in parts})
        if any(p < 1 for p in sizes):
            raise BadInput('parts must be positive')
        sizes = [p for p in sizes if p <= n]
    else:
        lo = _nat(min_part, 'min_part')
        if k is not None:
            return stars_and_bars(n, _nat(k, 'k'), upper=None if max_part is None else _nat(max_part, 'max_part'), lower=lo)
        hi = n if max_part is None else min(_nat(max_part, 'max_part'), n)
        if lo == 0:
            raise BadInput('a part of size 0 gives infinitely many compositions')
        sizes = list(range(lo, hi + 1))
    if k is None:
        _work(n * max(1, len(sizes)), 'compositions_count')
        f = [0] * (n + 1)
        f[0] = 1
        for s in range(1, n + 1):
            f[s] = sum(f[s - p] for p in sizes if p <= s)
        return f[n]
    k = _nat(k, 'k', 100000)
    _work(k * (n + 1) * max(1, len(sizes)), 'compositions_count')
    f = [0] * (n + 1)
    f[0] = 1
    for _ in range(k):
        g = [0] * (n + 1)
        for s in range(n + 1):
            if f[s]:
                for p in sizes:
                    if s + p <= n:
                        g[s + p] += f[s]
        f = g
    return f[n]


def _shape(shape):
    if not isinstance(shape, (list, tuple)) or not shape:
        raise BadInput('shape is a non-empty list of row lengths')
    sh = [_nat(x, 'row length') for x in shape]
    if any(x < 1 for x in sh) or any(sh[i] < sh[i + 1] for i in range(len(sh) - 1)):
        raise BadInput('shape must be positive and non-increasing')
    if sum(sh) > 2000:
        raise NotCovered('too large: more than 2000 cells')
    return sh


def _hooks(sh):
    cols = [sum(1 for r in sh if r > j) for j in range(sh[0])]
    return [[sh[i] - j - 1 + cols[j] - i - 1 + 1 for j in range(sh[i])] for i in range(len(sh))]


@op('counting.standard_young_tableaux', kind='partitions', bound='at most 2000 cells')
def standard_young_tableaux(shape):
    """Standard Young tableaux of a shape (hook-length formula). standard_young_tableaux([3, 2]) = 5."""
    sh = _shape(shape)
    den = 1
    for row in _hooks(sh):
        for h in row:
            den *= h
    return factorial(sum(sh)) // den


@op('counting.semistandard_young_tableaux', kind='partitions', bound='at most 2000 cells')
def semistandard_young_tableaux(shape, k):
    """Semistandard Young tableaux of a shape with entries 1..k (hook-content formula). semistandard_young_tableaux([2, 1], 3) = 8."""
    sh = _shape(shape)
    k = _nat(k, 'k', 10 ** 6)
    r = Fraction(1)
    for i, row in enumerate(_hooks(sh)):
        for j, h in enumerate(row):
            r *= Fraction(k + j - i, h)
    assert r.denominator == 1
    return int(r)


# ---------------------------------------------------------------- necklaces, bracelets, Burnside

def _counts(counts):
    if not isinstance(counts, (list, tuple)) or not counts:
        raise BadInput('counts is a non-empty list')
    cs = [_nat(c, 'count') for c in counts]
    return cs


def _fixed_content(cycles, counts):
    """Colourings fixed by a permutation with the given cycle lengths (a list) that use colour i exactly counts[i] times: the cycles are coloured, each cycle one colour."""
    k = len(counts)
    if sum(counts) != sum(cycles):
        return 0
    if len(set(cycles)) == 1:
        L, m = cycles[0], len(cycles)
        if any(c % L for c in counts):
            return 0
        return _multinomial([c // L for c in counts])
    states = 1
    for c in counts:
        states *= c + 1
    _work(states * len(cycles) * k, 'fixed-content DP')
    cur = {tuple(counts): 1}
    for L in sorted(cycles, reverse=True):
        nxt = {}
        for rem, w in cur.items():
            for i in range(k):
                if rem[i] >= L:
                    key = rem[:i] + (rem[i] - L,) + rem[i + 1:]
                    nxt[key] = nxt.get(key, 0) + w
        cur = nxt
    return cur.get(tuple([0] * k), 0)


@op('counting.necklace_count', kind='symmetry', bound='n <= 5000; with counts: <= 8 colours and states*cycles*colours <= 1e7')
def necklace_count(n=None, k=None, counts=None, bracelet=False):
    """Colourings of n beads on a circle up to rotation (a necklace) or up to rotation and reflection (bracelet = True), with k colours, or with exactly counts[i] beads of
    colour i (then n and k come from counts). necklace_count(6, 2) = 14; necklace_count(6, 2, bracelet=True) = 13; necklace_count(counts=[2, 2, 2]) = 16."""
    if counts is not None:
        cs = _counts(counts)
        if len(cs) > 8:
            raise NotCovered('too large: more than 8 colours with fixed counts')
        n = _nat(sum(cs), 'n', 5000)
        if n == 0:
            return 1
    else:
        n, k = _nat(n, 'n', 5000), _nat(k, 'k', 10 ** 6)
        if n == 0:
            return 1
    total, group = 0, 0
    for g in _divisors(n):          # rotations whose cycles are n/g beads long: phi(n/g) of them have g cycles
        mult = _phi(n // g)
        group += mult
        total += mult * (_fixed_content([n // g] * g, cs) if counts is not None else k ** g)
    if bracelet:
        for kind_, mult in _reflections(n):
            group += mult
            if counts is not None:
                total += mult * _fixed_content(kind_, cs)
            else:
                total += mult * k ** len(kind_)
    assert total % group == 0
    return total // group


def _reflections(n):
    """The reflections of an n-gon as (cycle lengths, how many): n odd: n with one fixed bead; n even: n/2 through two beads, n/2 through two edges."""
    if n == 1:
        return [([1], 1)]
    if n % 2:
        return [([1] + [2] * ((n - 1) // 2), n)]
    return [([1, 1] + [2] * ((n - 2) // 2), n // 2), ([2] * (n // 2), n // 2)]


@op('counting.orbit_count_colorings', kind='symmetry', bound='group <= 200000 elements, n_points <= 64')
def orbit_count_colorings(n_points, generators, k=None, counts=None):
    """Colourings of n_points points with k colours (or exactly counts[i] of colour i) up to the group generated by `generators` (permutations of 0..n_points-1, as lists).
    The 8 vertices of a cube under its rotation group, 2 colours, give 23 (see the unit test). orbit_count_colorings(4, [[1, 2, 3, 0]], 2) = 6."""
    n = _nat(n_points, 'n_points', 64)
    if not isinstance(generators, (list, tuple)) or not generators:
        raise BadInput('generators is a non-empty list of permutations')
    gens = []
    for g in generators:
        if not isinstance(g, (list, tuple)) or sorted(g) != list(range(n)):
            raise BadInput('a generator must be a permutation of 0..n_points-1')
        gens.append(tuple(g))
    if (k is None) == (counts is None):
        raise BadInput('give exactly one of k and counts')
    if counts is not None:
        cs = _counts(counts)
        if sum(cs) != n:
            raise BadInput('counts must add up to n_points')
        if len(cs) > 8:
            raise NotCovered('too large: more than 8 colours with fixed counts')
    else:
        k = _nat(k, 'k', 10 ** 6)
    ident = tuple(range(n))
    seen = {ident}
    frontier = [ident]
    while frontier:
        nxt = []
        for h in frontier:
            for g in gens:
                c = tuple(g[h[i]] for i in range(n))
                if c not in seen:
                    seen.add(c)
                    nxt.append(c)
                    if len(seen) > 200000:
                        raise NotCovered('too large: group has more than 200000 elements')
        frontier = nxt
    types = {}
    for h in seen:
        done, cyc = [False] * n, []
        for i in range(n):
            if not done[i]:
                m, j = 0, i
                while not done[j]:
                    done[j] = True
                    j = h[j]
                    m += 1
                cyc.append(m)
        key = tuple(sorted(cyc))
        types[key] = types.get(key, 0) + 1
    total = 0
    for cyc, mult in types.items():
        total += mult * (_fixed_content(list(cyc), cs) if counts is not None else k ** len(cyc))
    assert total % len(seen) == 0
    return total // len(seen)


# ---------------------------------------------------------------- lattice paths, Dyck, ballot

@op('counting.lattice_paths', kind='paths', bound='region <= 4e6 cells')
def lattice_paths(end, start=(0, 0), steps=((1, 0), (0, 1)), forbidden=(), region=None, constraint=None):
    """Paths from start to end using the given steps, never on a forbidden point, staying in region (xmin, ymin, xmax, ymax), and optionally on one side of the diagonal:
    constraint in 'y<=x', 'y<x', 'y>=x', 'y>x'. Steps must all go forward in x (or straight up) or all forward in y; with a negative step component give the region.
    lattice_paths((4, 4)) = 70; lattice_paths((4, 4), constraint='y<=x') = 14; lattice_paths((6, 0), steps=[(1, 1), (1, -1)], region=(0, 0, 6, 3)) = 5."""
    def pt(p, name):
        if not isinstance(p, (list, tuple)) or len(p) != 2:
            raise BadInput(f'{name} is a pair')
        return _int(p[0], name), _int(p[1], name)
    sx, sy = pt(start, 'start')
    ex, ey = pt(end, 'end')
    st = [pt(s, 'step') for s in steps]
    if not st or any(s == (0, 0) for s in st):
        raise BadInput('steps must be non-empty and non-zero')
    if constraint not in (None, 'y<=x', 'y<x', 'y>=x', 'y>x'):
        raise BadInput("constraint: None, 'y<=x', 'y<x', 'y>=x' or 'y>x'")
    if all(dx > 0 or (dx == 0 and dy > 0) for dx, dy in st):
        order = 'xy'
    elif all(dy > 0 or (dy == 0 and dx > 0) for dx, dy in st):
        order = 'yx'
    else:
        raise NotCovered('steps not ordered: a cycle is possible')
    if region is None:
        if any(dx < 0 or dy < 0 for dx, dy in st):
            raise NotCovered('a negative step needs a region')
        region = (min(sx, ex), min(sy, ey), max(sx, ex), max(sy, ey))
    if not isinstance(region, (list, tuple)) or len(region) != 4:
        raise BadInput('region is (xmin, ymin, xmax, ymax)')
    x0, y0, x1, y1 = (_int(r, 'region') for r in region)
    if x1 < x0 or y1 < y0:
        return 0
    cells = (x1 - x0 + 1) * (y1 - y0 + 1)
    if cells > 4 * 10 ** 6:
        raise NotCovered('too large: region has more than 4e6 cells')
    forb = {pt(f, 'forbidden') for f in forbidden}

    def ok(x, y):
        if (x, y) in forb:
            return False
        if constraint == 'y<=x':
            return y <= x
        if constraint == 'y<x':
            return y < x
        if constraint == 'y>=x':
            return y >= x
        if constraint == 'y>x':
            return y > x
        return True
    inside = lambda x, y: x0 <= x <= x1 and y0 <= y <= y1
    if not (inside(sx, sy) and inside(ex, ey) and ok(sx, sy) and ok(ex, ey)):
        return 0
    w = [[0] * (y1 - y0 + 1) for _ in range(x1 - x0 + 1)]
    outer = range(x0, x1 + 1) if order == 'xy' else range(y0, y1 + 1)
    inner = range(y0, y1 + 1) if order == 'xy' else range(x0, x1 + 1)
    for a in outer:
        for b in inner:
            x, y = (a, b) if order == 'xy' else (b, a)
            if (x, y) == (sx, sy):
                w[x - x0][y - y0] = 1
                continue
            if not ok(x, y):
                continue
            t = 0
            for dx, dy in st:
                px, py = x - dx, y - dy
                if inside(px, py):
                    t += w[px - x0][py - y0]
            w[x - x0][y - y0] = t
    return w[ex - x0][ey - y0]


@op('counting.catalan', kind='paths', bound='n <= 100000')
def catalan(n):
    """Catalan number C(2n, n)/(n+1). catalan(5) = 42."""
    n = _nat(n, 'n', 100000)
    return comb(2 * n, n) // (n + 1)


@op('counting.dyck_paths', kind='paths', bound='n <= 100000; with max_height n * max_height <= 1e7')
def dyck_paths(n, max_height=None):
    """Up/down paths of 2n steps from height 0 to 0 never below 0 (balanced bracket strings of n pairs), optionally never above max_height. dyck_paths(4) = 14; dyck_paths(4, 2) = 8."""
    n = _nat(n, 'n', 100000)
    if max_height is None:
        return catalan(n)
    h = _nat(max_height, 'max_height')
    h = min(h, n)
    _work(2 * n * (h + 1), 'dyck_paths')
    f = [0] * (h + 1)
    f[0] = 1
    for _ in range(2 * n):
        g = [0] * (h + 1)
        for i, x in enumerate(f):
            if x:
                if i + 1 <= h:
                    g[i + 1] += x
                if i >= 1:
                    g[i - 1] += x
        f = g
    return f[0]


@op('counting.ballot', kind='paths', bound='a + b <= 200000')
def ballot(a, b, strict=True):
    """Orders of a votes for A and b votes for B in which A is strictly ahead after every vote (strict) or never behind (strict=False). ballot(3, 1) = 2; ballot(3, 3, False) = 5."""
    a, b = _nat(a, 'a', 200000), _nat(b, 'b', 200000)
    if a + b == 0 and strict:
        raise BadInput('no votes: "strictly ahead" is not defined')
    if a < b or (strict and a == b):
        return 0
    if strict:
        return (a - b) * comb(a + b, a) // (a + b)
    return (a - b + 1) * comb(a + b, b) // (a + 1)


@op('counting.motzkin', kind='paths', bound='n <= 3000')
def motzkin(n):
    """Motzkin number: paths of n steps (up, level, down) from 0 to 0 never below 0. motzkin(4) = 9."""
    n = _nat(n, 'n', 3000)
    m = [1, 1]
    for i in range(2, n + 1):
        m.append(((2 * i + 1) * m[i - 1] + (3 * i - 3) * m[i - 2]) // (i + 2))
    return m[n]


@op('counting.schroeder', kind='paths', bound='n <= 3000')
def schroeder(n, small=False):
    """Large Schroeder number (paths with steps (1,1), (1,-1), (2,0) from (0,0) to (2n,0) never below the axis); small=True gives the little Schroeder number. schroeder(3) = 22."""
    n = _nat(n, 'n', 3000)
    s = [1, 2]
    for i in range(2, n + 1):
        s.append((3 * (2 * i - 1) * s[i - 1] - (i - 2) * s[i - 2]) // (i + 1))
    if small:
        if n == 0:
            return 1
        return s[n] // 2
    return s[n]


@op('counting.narayana', kind='paths', bound='n <= 100000')
def narayana(n, k):
    """Dyck paths of n pairs with exactly k peaks. narayana(4, 2) = 6."""
    n, k = _nat(n, 'n', 100000), _nat(k, 'k')
    if n == 0:
        return int(k == 0)
    if not 1 <= k <= n:
        return 0
    return comb(n, k) * comb(n, k - 1) // n


# ---------------------------------------------------------------- strings, transfer matrices, grids

_ALPHA = '0123456789abcdefghijklmnopqrstuvwxyz'


def _alphabet(a):
    if isinstance(a, int) and not isinstance(a, bool):
        if not 1 <= a <= 36:
            raise BadInput('alphabet size must be 1..36')
        return _ALPHA[:a]
    if isinstance(a, str) and a and len(set(a)) == len(a) and len(a) <= 36:
        return a
    raise BadInput('alphabet is a size or a string of distinct characters')


@op('counting.strings_with_patterns', kind='strings', bound='states * 2**len(contain) * alphabet * length <= 1e7, at most 10 contain patterns')
def strings_with_patterns(alphabet, length, avoid=(), contain=()):
    """Strings of the given length over an alphabet (a size 1..36 using 0-9a-z, or a string of distinct characters) that contain none of the substrings in `avoid` and every
    substring in `contain` (Aho-Corasick automaton DP). strings_with_patterns('ab', 5, avoid=['aa']) = 13; strings_with_patterns('01', 4, contain=['11']) = 8."""
    alpha = _alphabet(alphabet)
    n = _nat(length, 'length', 10 ** 6)
    avoid = [p for p in avoid]
    contain = [p for p in contain]
    for p in avoid + contain:
        if not isinstance(p, str) or not p:
            raise BadInput('a pattern is a non-empty string')
    if len(contain) > 10:
        raise NotCovered('too large: more than 10 required patterns')
    a_set = set(alpha)
    if any(set(p) - a_set for p in contain):
        return 0
    avoid = [p for p in avoid if not (set(p) - a_set)]
    pats = avoid + contain
    # trie
    nxt, fail, out = [{}], [0], [0]
    for pi, p in enumerate(pats):
        s = 0
        for ch in p:
            if ch not in nxt[s]:
                nxt.append({})
                fail.append(0)
                out.append(0)
                nxt[s][ch] = len(nxt) - 1
            s = nxt[s][ch]
        out[s] |= 1 << pi
    queue = []
    for ch, t in nxt[0].items():
        queue.append(t)
    qi = 0
    while qi < len(queue):
        s = queue[qi]
        qi += 1
        for ch, t in nxt[s].items():
            f = fail[s]
            while f and ch not in nxt[f]:
                f = fail[f]
            fail[t] = nxt[f][ch] if ch in nxt[f] and nxt[f][ch] != t else 0
            out[t] |= out[fail[t]]
            queue.append(t)
    ns = len(nxt)
    delta = [[0] * len(alpha) for _ in range(ns)]
    order = [0] + queue          # BFS order: fail[s] is shallower, so already complete
    for s in order:
        for ci, ch in enumerate(alpha):
            if ch in nxt[s]:
                delta[s][ci] = nxt[s][ch]
            else:
                delta[s][ci] = delta[fail[s]][ci] if s else 0
    na = len(avoid)
    amask = (1 << na) - 1
    nc = len(contain)
    _work(ns * (1 << nc) * len(alpha) * max(n, 1), 'strings_with_patterns')
    if out[0] & amask:
        return 0
    cur = {(0, 0): 1}
    for _ in range(n):
        new = {}
        for (s, m), w in cur.items():
            for ci in range(len(alpha)):
                t = delta[s][ci]
                o = out[t]
                if o & amask:
                    continue
                key = (t, m | (o >> na))
                new[key] = new.get(key, 0) + w
        cur = new
    full = (1 << nc) - 1
    return sum(w for (s, m), w in cur.items() if m == full)


@op('counting.transfer_matrix_walks', kind='strings', bound='matrix <= 80 x 80, length <= 10**6')
def transfer_matrix_walks(matrix, length, start=None, end=None):
    """Walks of `length` steps in a weighted digraph given by its square matrix (entries int or Fraction): the sum over start/end states (None = all) of the product of weights.
    transfer_matrix_walks([[1, 1], [1, 0]], 10, 0, 0) = 89."""
    if not isinstance(matrix, (list, tuple)) or not matrix:
        raise BadInput('matrix is a non-empty list of rows')
    m = len(matrix)
    if m > 80:
        raise NotCovered('too large: matrix larger than 80')
    A = []
    for r in matrix:
        if not isinstance(r, (list, tuple)) or len(r) != m:
            raise BadInput('matrix must be square')
        A.append([_frac(x, 'entry') if not isinstance(x, int) else x for x in r])
    n = _nat(length, 'length', 10 ** 6)
    for nm, v in (('start', start), ('end', end)):
        if v is not None and not (isinstance(v, int) and not isinstance(v, bool) and 0 <= v < m):
            raise BadInput(f'{nm} is a state index')
    vec = [1 if (start is None or i == start) else 0 for i in range(m)]
    if n * m * m <= 4 * 10 ** 6:
        for _ in range(n):
            vec = [sum(vec[i] * A[i][j] for i in range(m) if vec[i]) for j in range(m)]
    else:
        def mul(X, Y):
            Yt = list(zip(*Y))
            return [[sum(a * b for a, b in zip(row, col)) for col in Yt] for row in X]
        P = A
        e = n
        while e:
            if e & 1:
                vec = [sum(vec[i] * P[i][j] for i in range(m)) for j in range(m)]
            e >>= 1
            if e:
                P = mul(P, P)
    return sum(vec) if end is None else vec[end]


def _matrix01(M, k, name):
    if M is None:
        return None
    if not isinstance(M, (list, tuple)) or len(M) != k or any(not isinstance(r, (list, tuple)) or len(r) != k for r in M):
        raise BadInput(f'{name} must be a {k} x {k} matrix of 0/1')
    return [[bool(x) for x in r] for r in M]


@op('counting.grid_colorings', kind='strings', bound='colours**min(rows, cols) <= 3e5 and cells * states * colours <= 1e7')
def grid_colorings(rows, cols, k, allowed=None, allowed_v=None):
    """Colourings of a rows x cols grid with k colours where each pair of neighbours must be allowed. allowed[a][b] says colour a on the left may be next to colour b on the right
    (default: the two colours differ); allowed_v is the same for up/down (default: allowed). Broken-profile DP.
    grid_colorings(2, 2, 3) = 18; grid_colorings(3, 3, 3) = 246."""
    r, c, k = _nat(rows, 'rows', 10 ** 6), _nat(cols, 'cols', 10 ** 6), _nat(k, 'k', 100)
    h = _matrix01(allowed, k, 'allowed')
    v = _matrix01(allowed_v, k, 'allowed_v')
    if h is None:
        h = [[a != b for b in range(k)] for a in range(k)]
    if v is None:
        v = h if allowed is not None else [[a != b for b in range(k)] for a in range(k)]
    if r == 0 or c == 0:
        return 1
    if r > c:
        r, c, h, v = c, r, v, h        # the profile is the shorter side; swap the roles of horizontal and vertical
    _nat(r, 'shorter side', 12)
    states = k ** r
    if states > 3 * 10 ** 5:
        raise NotCovered('too large: colours**shorter side > 3e5')
    _work(r * c * states * k, 'grid_colorings')
    # cells in column-major order; the profile holds the last r cells: profile[0] is the cell to the left, profile[-1] the cell above
    cur = {(): 1}
    for col in range(c):
        for row in range(r):
            new = {}
            for prof, w in cur.items():
                left = prof[0] if col > 0 else None
                up = prof[-1] if row > 0 else None
                for a in range(k):
                    if left is not None and not h[left][a]:
                        continue
                    if up is not None and not v[up][a]:
                        continue
                    key = (prof + (a,))[1:] if len(prof) == r else prof + (a,)
                    new[key] = new.get(key, 0) + w
            cur = new
    return sum(cur.values())


@op('counting.domino_tilings', kind='strings', bound='min(rows, cols) <= 14, cells * 2**short <= 1e7')
def domino_tilings(rows, cols, blocked=()):
    """Tilings of a rows x cols board (minus the `blocked` cells (row, col)) by dominoes. domino_tilings(8, 8) = 12988816; domino_tilings(2, 5) = 8."""
    r, c = _nat(rows, 'rows', 1000), _nat(cols, 'cols', 1000)
    bl = set()
    for b in blocked:
        if not isinstance(b, (list, tuple)) or len(b) != 2 or not (0 <= b[0] < r and 0 <= b[1] < c):
            raise BadInput('a blocked cell is (row, col) inside the board')
        bl.add((b[0], b[1]))
    if r > c:
        r, c = c, r
        bl = {(y, x) for x, y in bl}
    if r > 14:
        raise NotCovered('too large: shorter side > 14')
    _work(r * c * (1 << r) * 2, 'domino_tilings')
    if r == 0 or c == 0:
        return 1
    # column-major cells; mask bit i set = the cell at row i of the 'current column or next column slot' is already covered
    cur = {0: 1}
    for col in range(c):
        for row in range(r):
            new = {}
            bit = 1 << row
            blocked_here = (row, col) in bl
            for mask, w in cur.items():
                if mask & bit:                         # covered by a domino that started to the left
                    new[mask ^ bit] = new.get(mask ^ bit, 0) + w
                    continue
                if blocked_here:
                    new[mask] = new.get(mask, 0) + w
                    continue
                # horizontal domino: covers (row, col+1): set the bit for the next column
                if col + 1 < c and (row, col + 1) not in bl:
                    new[mask | bit] = new.get(mask | bit, 0) + w
                # vertical domino: covers (row+1, col): must be free now
                if row + 1 < r and not (mask & (bit << 1)) and (row + 1, col) not in bl:
                    m2 = mask | (bit << 1)
                    new[m2] = new.get(m2, 0) + w
            cur = new
    return cur.get(0, 0)


@op('counting.count_digit_constraints', kind='strings', bound='hi < 10**60, base <= 16, mod <= 2000; states <= 4e6')
def count_digit_constraints(lo, hi, base=10, digit_sum=None, forbidden_digits=(), distinct=False, no_adjacent_equal=False, nondecreasing=False, increasing=False, mod=None, rem=0):
    """Integers n in [lo, hi] (lo >= 1) whose digits in `base` satisfy: digit sum in digit_sum (a number or a list), no digit in forbidden_digits, all digits distinct, no two
    equal neighbours, digits non-decreasing / strictly increasing, and n % mod == rem. count_digit_constraints(1, 1000, digit_sum=10) = 63 (digit sum 10 up to 1000);
    count_digit_constraints(1, 10**6, forbidden_digits=[7]) = 531441."""
    lo, hi = _nat(lo, 'lo'), _nat(hi, 'hi', 10 ** 60)
    base = _nat(base, 'base', 16)
    if base < 2 or lo < 1:
        raise BadInput('base >= 2 and lo >= 1')
    sums = None
    if digit_sum is not None:
        sums = frozenset([digit_sum] if isinstance(digit_sum, int) else [_int(s, 'digit_sum') for s in digit_sum])
    forb = frozenset(_nat(d, 'digit') for d in forbidden_digits)
    if mod is not None:
        mod = _nat(mod, 'mod', 2000)
        if mod < 1:
            raise BadInput('mod must be positive')
        rem = _int(rem, 'rem') % mod
    if hi < lo:
        return 0

    def upto(N):
        if N < 1:
            return 0
        ds = []
        x = N
        while x:
            ds.append(x % base)
            x //= base
        ds.reverse()
        L = len(ds)
        visited = [0]

        @lru_cache(maxsize=None)
        def go(pos, tight, started, s, used, last, r):
            visited[0] += 1
            if visited[0] > 4 * 10 ** 6:
                raise NotCovered('too large: more than 4e6 digit-DP states')
            if pos == L:
                if not started:
                    return 0
                if sums is not None and s not in sums:
                    return 0
                if mod is not None and r != rem:
                    return 0
                return 1
            top = ds[pos] if tight else base - 1
            t = 0
            for d in range(top + 1):
                nt = tight and d == top
                if not started and d == 0:
                    t += go(pos + 1, nt, False, 0, 0, -1, 0)
                    continue
                if d in forb:
                    continue
                if distinct and used >> d & 1:
                    continue
                if no_adjacent_equal and d == last:
                    continue
                if nondecreasing and last != -1 and d < last:
                    continue
                if increasing and last != -1 and d <= last:
                    continue
                t += go(pos + 1, nt, True, s + d if sums is not None else 0, (used | 1 << d) if distinct else 0,
                        d if (no_adjacent_equal or nondecreasing or increasing) else -1, (r * base + d) % mod if mod is not None else 0)
            return t
        try:
            return go(0, True, False, 0, 0, -1, 0)
        finally:
            go.cache_clear()
    return upto(hi) - upto(lo - 1)


# ---------------------------------------------------------------- graphs

def _det_int(M):
    """Exact determinant of an integer matrix (Bareiss fraction-free elimination)."""
    n = len(M)
    if n == 0:
        return 1
    M = [row[:] for row in M]
    sign, prev = 1, 1
    for k in range(n - 1):
        if M[k][k] == 0:
            for i in range(k + 1, n):
                if M[i][k] != 0:
                    M[k], M[i] = M[i], M[k]
                    sign = -sign
                    break
            else:
                return 0
        for i in range(k + 1, n):
            for j in range(k + 1, n):
                M[i][j] = (M[i][j] * M[k][k] - M[i][k] * M[k][j]) // prev
        prev = M[k][k]
    return sign * M[n - 1][n - 1]


@op('counting.spanning_trees', kind='graphs', bound='n <= 80 vertices')
def spanning_trees(n, edges, multi=False):
    """Number of spanning trees of a graph on vertices 0..n-1 (matrix-tree theorem, exact determinant). multi=True lets an edge repeat (parallel edges).
    spanning_trees(4, [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]) = 16."""
    n, es = _graph(n, edges, 80, multi)
    if n == 0:
        return 0
    L = [[0] * n for _ in range(n)]
    for u, v in es:
        L[u][u] += 1
        L[v][v] += 1
        L[u][v] -= 1
        L[v][u] -= 1
    return _det_int([row[1:] for row in L[1:]])


@op('counting.labelled_trees', kind='graphs', bound='n <= 100000')
def labelled_trees(n, degrees=None):
    """Labelled trees on n vertices (Cayley: n**(n-2)); with degrees (the degree of each vertex) only the trees with that degree sequence. labelled_trees(5) = 125; labelled_trees(4, [1, 1, 2, 2]) = 2."""
    n = _nat(n, 'n', 100000)
    if degrees is not None:
        d = [_nat(x, 'degree') for x in degrees]
        if len(d) != n:
            raise BadInput('degrees needs one entry per vertex')
        if n == 1:
            return int(d == [0])
        if n == 2:
            return int(d == [1, 1])
        if any(x < 1 for x in d) or sum(d) != 2 * (n - 1):
            return 0
        den = 1
        for x in d:
            den *= factorial(x - 1)
        return factorial(n - 2) // den
    if n == 0:
        return 0
    if n <= 2:
        return 1
    return n ** (n - 2)


@op('counting.rooted_forests', kind='graphs', bound='n <= 100000')
def rooted_forests(n, k):
    """Forests on n labelled vertices made of k trees whose roots are k given vertices: k * n**(n-k-1). rooted_forests(4, 2) = 8."""
    n, k = _nat(n, 'n', 100000), _nat(k, 'k')
    if k > n or k == 0:
        return 0
    if k == n:
        return 1
    return k * n ** (n - k - 1)


@op('counting.perfect_matchings', kind='graphs', bound='n <= 22 (edges given), 3e6 DP states; complete graph n <= 2000')
def perfect_matchings(n, edges=None):
    """Perfect matchings of a graph on 0..n-1 (edges=None: the complete graph, (n-1)!!). perfect_matchings(6) = 15; perfect_matchings(4, [(0,1),(1,2),(2,3),(3,0)]) = 2."""
    n = _nat(n, 'n', 2000)
    if n % 2:
        return 0
    if edges is None:
        r = 1
        for i in range(1, n, 2):
            r *= i
        return r
    n, es = _graph(n, edges, 22)
    adj = _adj_masks(n, es)
    full = (1 << n) - 1
    memo = {full: 1}

    def go(mask):
        if mask in memo:
            return memo[mask]
        if len(memo) > 3 * 10 ** 6:
            raise NotCovered('too large: more than 3e6 DP states')
        free = ~mask & full
        v = (free & -free).bit_length() - 1
        cand = adj[v] & free & ~(1 << v)
        t = 0
        while cand:
            b = cand & -cand
            t += go(mask | (1 << v) | b)
            cand ^= b
        memo[mask] = t
        return t
    return go(0)


@op('counting.matchings_count', kind='graphs', bound='n <= 22, 3e6 DP states')
def matchings_count(n, edges, size=None):
    """Matchings (sets of disjoint edges, the empty one included) of a graph; with `size` only those with that many edges. matchings_count(3, [(0,1),(1,2),(0,2)]) = 4."""
    n, es = _graph(n, edges, 22)
    adj = _adj_masks(n, es)
    if size is not None:
        _nat(size, 'size')
    memo = {}

    def go(free):
        """polynomial (list by matching size) of matchings among the vertices in `free`"""
        if free == 0:
            return (1,)
        if free in memo:
            return memo[free]
        if len(memo) > 3 * 10 ** 6:
            raise NotCovered('too large: more than 3e6 DP states')
        v = (free & -free).bit_length() - 1
        rest = free ^ (1 << v)
        res = list(go(rest))                 # v unmatched
        cand = adj[v] & rest
        while cand:
            b = cand & -cand
            sub = go(rest ^ b)
            if len(res) < len(sub) + 1:
                res.extend([0] * (len(sub) + 1 - len(res)))
            for i, x in enumerate(sub):
                res[i + 1] += x
            cand ^= b
        memo[free] = tuple(res)
        return memo[free]
    poly = go((1 << n) - 1)
    if size is None:
        return sum(poly)
    return poly[size] if size < len(poly) else 0


@op('counting.permutations_allowed', kind='graphs', bound='n <= 20, 3e6 DP states')
def permutations_allowed(allowed):
    """Permutations p of 0..n-1 with allowed[i][p(i)] true for every i (the permanent of a 0/1 matrix; perfect matchings of a bipartite graph).
    permutations_allowed([[1,1,0],[1,0,1],[0,1,1]]) = 2."""
    if not isinstance(allowed, (list, tuple)):
        raise BadInput('allowed is a square 0/1 matrix')
    n = len(allowed)
    if n > 20:
        raise NotCovered('too large: n > 20')
    rows = []
    for r in allowed:
        if not isinstance(r, (list, tuple)) or len(r) != n:
            raise BadInput('allowed must be square')
        rows.append(sum(1 << j for j, x in enumerate(r) if x))
    cur = {0: 1}
    for i in range(n):
        new = {}
        for mask, w in cur.items():
            cand = rows[i] & ~mask
            while cand:
                b = cand & -cand
                new[mask | b] = new.get(mask | b, 0) + w
                cand ^= b
        cur = new
        if len(cur) > 3 * 10 ** 6:
            raise NotCovered('too large: more than 3e6 DP states')
    return cur.get((1 << n) - 1, 0) if n else 1


@op('counting.independent_sets', kind='graphs', bound='n <= 48, 2e6 branching nodes')
def independent_sets(n, edges, size=None):
    """Independent sets (no two adjacent vertices, the empty set included) of a graph; with `size` only those with that many vertices; size='poly' gives the whole
    independence polynomial as a list indexed by size. independent_sets(4, [(0,1),(1,2),(2,3)]) = 8."""
    n, es = _graph(n, edges, 48)
    adj = _adj_masks(n, es)
    memo = {}
    nodes = [0]

    def mul(a, b):
        r = [0] * (len(a) + len(b) - 1)
        for i, x in enumerate(a):
            if x:
                for j, y in enumerate(b):
                    r[i + j] += x * y
        return r

    def add(a, b):
        if len(a) < len(b):
            a, b = b, a
        r = list(a)
        for i, x in enumerate(b):
            r[i] += x
        return r

    def components(mask):
        comps = []
        while mask:
            s = mask & -mask
            comp, frontier = s, s
            while frontier:
                v = (frontier & -frontier).bit_length() - 1
                frontier &= frontier - 1
                new = adj[v] & mask & ~comp
                comp |= new
                frontier |= new
            comps.append(comp)
            mask &= ~comp
        return comps

    def poly(mask):
        if mask == 0:
            return [1]
        if mask in memo:
            return memo[mask]
        nodes[0] += 1
        if nodes[0] > 2 * 10 ** 6:
            raise NotCovered('too large: more than 2e6 branching nodes')
        comps = components(mask)
        if len(comps) > 1:
            r = [1]
            for c in comps:
                r = mul(r, poly(c))
        else:
            best, bd = -1, -1
            m = mask
            while m:
                v = (m & -m).bit_length() - 1
                m &= m - 1
                d = bin(adj[v] & mask).count('1')
                if d > bd:
                    best, bd = v, d
            if bd == 0:
                r = [comb(bin(mask).count('1'), i) for i in range(bin(mask).count('1') + 1)]
            else:
                without = poly(mask & ~(1 << best))
                with_ = poly(mask & ~(1 << best) & ~adj[best])
                r = add(without, [0] + with_)
        memo[mask] = r
        return r
    p = poly((1 << n) - 1)
    if size is None:
        return sum(p)
    if size == 'poly':
        return list(p)
    _nat(size, 'size')
    return p[size] if size < len(p) else 0


def _components(n, edges):
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    seen, comps = [False] * n, []
    for s in range(n):
        if not seen[s]:
            seen[s] = True
            comp, stack = [s], [s]
            while stack:
                x = stack.pop()
                for y in adj[x]:
                    if not seen[y]:
                        seen[y] = True
                        comp.append(y)
                        stack.append(y)
            comps.append(sorted(comp))
    return comps


@op('counting.chromatic_polynomial', kind='graphs', bound='n <= 13 vertices')
def chromatic_polynomial(n, edges):
    """Chromatic polynomial P(k) (number of proper k-colourings) as a list of integer coefficients, lowest power first. A triangle: [0, 2, -3, 1] (k^3 - 3k^2 + 2k).
    chromatic_polynomial(3, [(0,1),(1,2),(0,2)]) = [0, 2, -3, 1]."""
    n, es = _graph(n, edges, 13)
    if n == 0:
        return [1]
    adj = _adj_masks(n, es)
    full = (1 << n) - 1
    ind = [True] * (1 << n)
    for mask in range(1, 1 << n):
        low = (mask & -mask).bit_length() - 1
        rest = mask & (mask - 1)
        ind[mask] = ind[rest] and not (adj[low] & rest)
    dp = [None] * (1 << n)       # dp[mask][j]: partitions of mask into j independent blocks
    dp[0] = [1]
    for mask in range(1, 1 << n):
        low = mask & -mask
        rest = mask ^ low
        res = [0] * (bin(mask).count('1') + 1)
        sub = rest
        while True:
            block = sub | low
            if ind[block]:
                prev = dp[mask ^ block]
                for j, x in enumerate(prev):
                    if x:
                        res[j + 1] += x
            if sub == 0:
                break
            sub = (sub - 1) & rest
        while len(res) > 1 and res[-1] == 0:
            res.pop()
        dp[mask] = res
    a = dp[full]
    # sum_j a_j * k(k-1)...(k-j+1)
    result = [0] * (n + 1)
    fall = [1]
    for j in range(0, n + 1):
        if j < len(a) and a[j]:
            for i, c in enumerate(fall):
                result[i] += a[j] * c
        new = [0] * (len(fall) + 1)       # multiply by (k - j)
        for i, c in enumerate(fall):
            new[i + 1] += c
            new[i] -= j * c
        fall = new
    while len(result) > 1 and result[-1] == 0:
        result.pop()
    return result


@op('counting.colorings_count', kind='graphs', bound='n <= 100000; trees and cycles any size, other components: <= 13 vertices by the chromatic polynomial, <= 40 vertices by backtracking (3e6 nodes)')
def colorings_count(n, edges, k):
    """Proper colourings of a graph with k colours (adjacent vertices differ). Trees and cycles by formula, small components by the chromatic polynomial, the rest by backtracking.
    colorings_count(5, [(0,1),(1,2),(2,3),(3,4),(4,0)], 3) = 30."""
    n, es = _graph(n, edges, 100000)
    k = _nat(k, 'k', 10 ** 6)
    total = 1
    for comp in _components(n, es):
        idx = {v: i for i, v in enumerate(comp)}
        ce = [(idx[u], idx[v]) for u, v in es if u in idx]
        s, m = len(comp), len(ce)
        if s == 1:
            total *= k
        elif m == s - 1:
            total *= k * (k - 1) ** (s - 1)
        elif m == s and all(d == 2 for d in _degrees(s, ce)):
            total *= (k - 1) ** s + (-1) ** s * (k - 1)
        elif s <= 13:
            poly = chromatic_polynomial(s, ce)
            total *= sum(c * k ** i for i, c in enumerate(poly))
        else:
            total *= _backtrack_colorings(s, ce, k)
        if total == 0:
            return 0
    return total


def _degrees(n, edges):
    d = [0] * n
    for u, v in edges:
        d[u] += 1
        d[v] += 1
    return d


def _backtrack_colorings(n, edges, k):
    if n > 40:
        raise NotCovered('too large: component with more than 40 vertices and cycles')
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    order, seen = [0], {0}
    for x in order:                     # BFS order keeps the colouring connected
        for y in sorted(adj[x]):
            if y not in seen:
                seen.add(y)
                order.append(y)
    pos = {v: i for i, v in enumerate(order)}
    back = [[pos[y] for y in adj[v] if pos[y] < pos[v]] for v in order]
    col = [0] * n
    nodes = [0]

    def go(i):
        if i == n:
            return 1
        nodes[0] += 1
        if nodes[0] > 3 * 10 ** 6:
            raise NotCovered('too large: more than 3e6 backtracking nodes')
        t = 0
        used = {col[j] for j in back[i]}
        for c in range(k):
            if c not in used:
                col[i] = c
                t += go(i + 1)
        return t
    if k == 0:
        return 0
    col[0] = 0
    return k * go(1)           # by symmetry the first vertex's colour is free


@op('counting.graph_automorphisms', kind='graphs', bound='n <= 16, 2e6 search nodes')
def graph_automorphisms(n, edges):
    """Size of the automorphism group of a small graph (vertex permutations that keep the edge set). graph_automorphisms(4, [(0,1),(1,2),(2,3),(3,0)]) = 8."""
    n, es = _graph(n, edges, 16)
    adj = _adj_masks(n, es)
    deg = [bin(a).count('1') for a in adj]
    image = [-1] * n
    used = 0
    nodes = [0]

    def go(i, used):
        if i == n:
            return 1
        nodes[0] += 1
        if nodes[0] > 2 * 10 ** 6:
            raise NotCovered('too large: more than 2e6 search nodes')
        t = 0
        for c in range(n):
            if used >> c & 1 or deg[c] != deg[i]:
                continue
            good = True
            for j in range(i):
                if (adj[i] >> j & 1) != (adj[c] >> image[j] & 1):
                    good = False
                    break
            if good:
                image[i] = c
                t += go(i + 1, used | 1 << c)
        return t
    return go(0, 0)


# ---------------------------------------------------------------- inclusion-exclusion

@op('counting.union_size', kind='inclusion-exclusion', bound='at most 2e6 intersection entries')
def union_size(intersections):
    """Size of the union of sets from the sizes of their intersections: a dict {tuple of set indices: size of that intersection}; the answer is
    the sum of (-1)**(len(key)+1) * size. union_size({(0,): 10, (1,): 8, (0, 1): 3}) = 15."""
    if not isinstance(intersections, dict) or len(intersections) > 2 * 10 ** 6:
        raise BadInput('intersections is a dict of tuple -> size')
    t = 0
    for key, sz in intersections.items():
        if not isinstance(key, (tuple, list)) or not key or len(set(key)) != len(key):
            raise BadInput('a key is a non-empty tuple of distinct set indices')
        t += (-1) ** (len(key) + 1) * _int(sz, 'size')
    return t


@op('counting.count_divisible', kind='inclusion-exclusion', bound='at most 20 divisors')
def count_divisible(n, divisors, exactly=None, lo=1):
    """Integers in [lo, n] divisible by at least one of the divisors (exactly=None), by none of them (exactly=0), by all of them (exactly=len), or by exactly `exactly`
    of them (inclusion-exclusion over lcm's). count_divisible(100, [2, 3, 5]) = 74; count_divisible(100, [2, 3, 5], exactly=0) = 26."""
    n, lo = _int(n, 'n'), _int(lo, 'lo')
    ds = [_int(d, 'divisor') for d in divisors]
    if any(d < 1 for d in ds):
        raise BadInput('divisors must be positive')
    m = len(ds)
    if m > 20:
        raise NotCovered('too large: more than 20 divisors')
    if exactly is not None:
        _nat(exactly, 'exactly')
    if n < lo:
        return 0

    # S[t] = sum over t-subsets of the number of multiples of their lcm in [lo, n]
    S = [0] * (m + 1)
    lcms = [1] * (1 << m)
    for mask in range(1 << m):
        if mask:
            low = (mask & -mask).bit_length() - 1
            prev = lcms[mask & (mask - 1)]
            lcms[mask] = prev * ds[low] // gcd(prev, ds[low])
        l = lcms[mask]
        S[bin(mask).count('1')] += n // l - (lo - 1) // l
    if exactly is None:
        return sum((-1) ** (t + 1) * S[t] for t in range(1, m + 1))
    if exactly > m:
        return 0
    return sum((-1) ** (t - exactly) * comb(t, exactly) * S[t] for t in range(exactly, m + 1))


@op('counting.count_coprime', kind='inclusion-exclusion', bound='m <= 10**13')
def count_coprime(n, m, lo=1):
    """Integers in [lo, n] coprime to m (inclusion-exclusion over the primes of m). count_coprime(100, 30) = 26."""
    n, lo, m = _int(n, 'n'), _int(lo, 'lo'), _nat(m, 'm', 10 ** 13)
    if m < 1:
        raise BadInput('m must be positive')
    if n < lo:
        return 0
    ps = list(_factor_small(m))
    t = 0
    for mask in range(1 << len(ps)):
        d, sgn = 1, 1
        for i, p in enumerate(ps):
            if mask >> i & 1:
                d *= p
                sgn = -sgn
        t += sgn * (n // d - (lo - 1) // d)
    return t


# ---------------------------------------------------------------- predicate language and brute force

_FUNCS = {'abs': abs, 'min': min, 'max': max, 'sum': sum, 'len': len, 'gcd': math.gcd, 'lcm': math.lcm, 'isqrt': math.isqrt,
          'sorted': sorted, 'set': set, 'tuple': tuple}


def _mul(a, b):
    """* in a predicate is for integers only (a tuple times a huge integer would be a memory bomb)."""
    if isinstance(a, int) and isinstance(b, int):
        return a * b
    raise TypeError('* needs two integers')


def _comb(n, k):
    if not (isinstance(n, int) and isinstance(k, int)) or n > 5000:
        raise ValueError('comb: integers with n <= 5000')
    return math.comb(n, k)


@lru_cache(maxsize=1 << 20)
def _is_prime(x):
    if x > 10 ** 9:
        raise ValueError('is_prime: x <= 10**9')
    if x < 2:
        return False
    if x < 4:
        return True
    if x % 2 == 0:
        return False
    i = 3
    while i * i <= x:
        if x % i == 0:
            return False
        i += 2
    return True


def _is_square(x):
    return x >= 0 and math.isqrt(x) ** 2 == x


def _dsum(x):
    return sum(map(int, str(abs(x))))


def _cnt(v, a):
    return sum(1 for y in v if y == a)


_FUNCS.update(comb=_comb, is_prime=_is_prime, is_square=_is_square, dsum=_dsum, cnt=_cnt)
_OKNODES = tuple(getattr(ast, n) for n in (
    'Expression', 'BoolOp', 'And', 'Or', 'BinOp', 'Add', 'Sub', 'Mult', 'FloorDiv', 'Mod', 'Pow', 'BitAnd', 'BitOr', 'BitXor', 'UnaryOp', 'Not', 'USub', 'UAdd', 'Invert',
    'Compare', 'Eq', 'NotEq', 'Lt', 'LtE', 'Gt', 'GtE', 'In', 'NotIn', 'IfExp', 'Call', 'Name', 'Load', 'Constant', 'Tuple', 'List', 'Subscript', 'Slice'))


def _compile(expr, names):
    """expr (text) -> function of the variables `names` (checked against the whitelist)."""
    if not isinstance(expr, str) or not expr.strip():
        raise BadInput('a predicate is a non-empty string')
    if len(expr) > 600:
        raise BadInput('predicate longer than 600 characters')
    if '/' in expr.replace('//', ''):
        raise BadInput("'/' is not allowed: use // and %")
    try:
        tree = ast.parse(expr.strip(), mode='eval')
    except SyntaxError as e:
        raise BadInput(f'predicate does not parse: {e.msg}')
    pows = 0
    for node in ast.walk(tree):
        if not isinstance(node, _OKNODES):
            raise BadInput(f'predicate uses {type(node).__name__}, which the language does not allow')
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or (isinstance(node.value, int) and abs(node.value) <= 10 ** 18):
                continue
            raise BadInput('only integer constants up to 10**18 are allowed')
        if isinstance(node, ast.Name):
            if node.id not in names and node.id not in _FUNCS:
                raise BadInput(f'unknown name {node.id}')
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in _FUNCS or node.keywords:
                raise BadInput('only the listed functions may be called, with positional arguments')
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
            pows += 1
            e = node.right
            if pows > 3 or not (isinstance(e, ast.Constant) and isinstance(e.value, int) and 0 <= e.value <= 32):
                raise BadInput('** needs a constant exponent 0..32 (at most 3 per predicate)')
    src = f"lambda {','.join(names)}: ({expr.strip()})"
    tree = _MulChecked().visit(ast.parse(src, mode='eval'))
    ast.fix_missing_locations(tree)
    return eval(compile(tree, '<predicate>', 'eval'), {'__builtins__': {}, '_mul': _mul, **_FUNCS})


class _MulChecked(ast.NodeTransformer):
    def visit_BinOp(self, node):
        self.generic_visit(node)
        if isinstance(node.op, ast.Mult):
            return ast.Call(func=ast.Name(id='_mul', ctx=ast.Load()), args=[node.left, node.right], keywords=[])
        return node


def _domain(spec):
    if isinstance(spec, str):
        try:
            a, b = spec.split('..')
            a, b = int(a), int(b)
        except ValueError:
            raise BadInput("a domain is 'lo..hi' or a list of integers")
        if b - a + 1 > 10 ** 7:
            raise NotCovered('too large: domain has more than 1e7 values')
        return list(range(a, b + 1))
    if isinstance(spec, (list, tuple)) and spec:
        return [_int(x, 'domain value') for x in spec]
    raise BadInput("a domain is 'lo..hi' or a non-empty list of integers")


def _space(space, domains, r):
    """(iterator of tuples, size, arity or None) of a finite space."""
    if space == 'product':
        if not isinstance(domains, (list, tuple)) or not domains:
            raise BadInput('product: domains is a non-empty list of domains')
        ds = [_domain(d) for d in domains]
        size = 1
        for d in ds:
            size *= len(d)
        if size > BRUTE_LIMIT:
            raise NotCovered(f'too large: {size} elements (limit {BRUTE_LIMIT})')
        return itertools.product(*ds), size, len(ds)
    items = _domain(domains)
    n = len(items)
    if space == 'subsets':
        if n > 25:
            raise NotCovered('too large: subsets of more than 25 items')
        return itertools.chain.from_iterable(itertools.combinations(items, j) for j in range(n + 1)), 2 ** n, None
    if r is None:
        r = n
    r = _nat(r, 'r')
    if space == 'permutations':
        size = math.perm(n, r)
        it = itertools.permutations(items, r)
    elif space == 'combinations':
        size = comb(n, r)
        it = itertools.combinations(items, r)
    elif space == 'multisets':
        size = comb(n + r - 1, r) if n + r else 1
        it = itertools.combinations_with_replacement(items, r)
    else:
        raise BadInput("space: 'product', 'permutations', 'combinations', 'multisets' or 'subsets'")
    if size > BRUTE_LIMIT:
        raise NotCovered(f'too large: {size} elements (limit {BRUTE_LIMIT})')
    return it, size, r


def _predicate(expr, arity, names):
    if arity is None:
        if names:
            raise BadInput('subsets have no fixed arity: use v only')
        f = _compile(expr, ['v'])
        return lambda t: f(t)
    if names is None:
        names = [f'x{i}' for i in range(arity)]
    if len(names) != arity or len(set(names)) != arity:
        raise BadInput(f'names must be {arity} distinct identifiers')
    for nm in names:
        if not isinstance(nm, str) or not nm.isidentifier() or keyword.iskeyword(nm) or nm in _FUNCS or nm == 'v':
            raise BadInput(f'bad variable name {nm!r}')
    f = _compile(expr, list(names) + ['v'])
    return lambda t: f(*t, t)


def _guard(fn, *a):
    try:
        return fn(*a)
    except (ZeroDivisionError, TypeError, ValueError, IndexError, OverflowError) as e:
        raise BadInput(f'predicate failed on an element: {type(e).__name__}: {e}')


@op('counting.brute_count', kind='brute force', bound='5e7 elements')
def brute_count(domains, predicate, space='product', r=None, names=None):
    """Number of elements of a finite space satisfying a predicate. space 'product': domains is a list of domains ('lo..hi' or a list of ints), one variable each;
    'permutations' / 'combinations' / 'multisets' (r items chosen from one domain) and 'subsets' (all subsets of one domain, variable v only). The predicate is text.
    brute_count(['1..6', '1..6'], 'x0 + x1 == 7') = 6; brute_count('1..4', 'x0 < x1', 'combinations', 2) = 6."""
    it, size, arity = _space(space, domains, r)
    f = _predicate(predicate, arity, names)
    return _guard(lambda: sum(1 for t in it if f(t)))


@op('counting.predicate_histogram', kind='brute force', bound='5e7 elements, 12 predicates')
def predicate_histogram(domains, predicates, space='product', r=None, names=None):
    """For a list of predicates: {j: number of elements satisfying exactly j of them} (inclusion-exclusion questions: j = 0 is 'none', j = len is 'all').
    predicate_histogram(['1..30'], ['x0 % 2 == 0', 'x0 % 3 == 0']) = {0: 10, 1: 15, 2: 5}."""
    if not isinstance(predicates, (list, tuple)) or not predicates or len(predicates) > 12:
        raise BadInput('predicates is a list of 1..12 predicate texts')
    it, size, arity = _space(space, domains, r)
    fs = [_predicate(p, arity, names) for p in predicates]
    hist = {j: 0 for j in range(len(fs) + 1)}

    def run():
        for t in it:
            hist[sum(1 for f in fs if f(t))] += 1
    _guard(run)
    return hist


@op('counting.prob_uniform', kind='probability', bound='5e7 elements')
def prob_uniform(domains, event, given=None, space='product', r=None, names=None):
    """Probability of `event` when an element of a finite space is chosen uniformly, optionally given another event (conditional probability). Exact Fraction.
    prob_uniform(['1..6', '1..6'], 'x0 + x1 == 7') = 1/6; prob_uniform(['1..6', '1..6'], 'x0 + x1 == 7', given='x0 == 3') = 1/6."""
    it, size, arity = _space(space, domains, r)
    fe = _predicate(event, arity, names)
    fg = _predicate(given, arity, names) if given is not None else None

    def run():
        e = g = 0
        for t in it:
            if fg is None or fg(t):
                g += 1
                if fe(t):
                    e += 1
        return e, g
    e, g = _guard(run)
    if g == 0:
        raise BadInput('the conditioning event never happens')
    return Fraction(e, g)


@op('counting.expectation_uniform', kind='probability', bound='5e7 elements')
def expectation_uniform(domains, expr, space='product', r=None, names=None):
    """Mean and variance of an integer-valued expression over a uniformly chosen element of a finite space: {'mean': Fraction, 'variance': Fraction, 'count': int}.
    expectation_uniform(['1..6', '1..6'], 'max(x0, x1)') gives mean 161/36."""
    it, size, arity = _space(space, domains, r)
    f = _predicate(expr, arity, names)

    def run():
        n = s = s2 = 0
        for t in it:
            x = f(t)
            n += 1
            s += x
            s2 += x * x
        return n, s, s2
    n, s, s2 = _guard(run)
    if n == 0:
        raise BadInput('the space is empty')
    mean = Fraction(s, n)
    return {'mean': mean, 'variance': Fraction(s2, n) - mean * mean, 'count': n}


# ---------------------------------------------------------------- dice and distributions

@op('counting.dice_sum_distribution', kind='probability', bound='dice * max sum * faces <= 1e7')
def dice_sum_distribution(n, sides=6, faces=None):
    """Number of ways to get each total with n dice with `sides` faces 1..sides, or with the given faces (a list, which may repeat a value): {total: ways}.
    dice_sum_distribution(2) = {2: 1, 3: 2, ..., 12: 1}; dice_sum_distribution(2, faces=[0, 1]) = {0: 1, 1: 2, 2: 1}."""
    n = _nat(n, 'n', 5000)
    fc = list(range(1, _nat(sides, 'sides', 10000) + 1)) if faces is None else [_int(f, 'face') for f in faces]
    if not fc:
        raise BadInput('a die needs at least one face')
    lo, hi = min(fc), max(fc)
    _work(n * (n * max(abs(lo), abs(hi)) + 1) * len(fc), 'dice_sum_distribution')
    cur = {0: 1}
    for _ in range(n):
        new = {}
        for s, w in cur.items():
            for f in fc:
                new[s + f] = new.get(s + f, 0) + w
        cur = new
    return dict(sorted(cur.items()))


@op('counting.dice_sum_probability', kind='probability', bound='as dice_sum_distribution')
def dice_sum_probability(n, sides, lo, hi=None, faces=None):
    """Probability that the total of n fair dice is lo (hi=None) or between lo and hi inclusive. dice_sum_probability(2, 6, 7) = 1/6; dice_sum_probability(3, 6, 10, 11) = 1/4."""
    dist = dice_sum_distribution(n, sides, faces)
    lo = _int(lo, 'lo')
    hi = lo if hi is None else _int(hi, 'hi')
    total = sum(dist.values())
    return Fraction(sum(w for s, w in dist.items() if lo <= s <= hi), total)


@op('counting.dice_extreme_distribution', kind='probability', bound='n <= 5000, sides <= 10000')
def dice_extreme_distribution(n, sides, stat='max'):
    """Ways for the maximum (or minimum) of n dice with faces 1..sides to be each value: {value: ways}. dice_extreme_distribution(2, 3, 'max') = {1: 1, 2: 3, 3: 5}."""
    n, s = _nat(n, 'n', 5000), _nat(sides, 'sides', 10000)
    if stat not in ('max', 'min'):
        raise BadInput("stat is 'max' or 'min'")
    out = {}
    for v in range(1, s + 1):
        if stat == 'max':
            out[v] = v ** n - (v - 1) ** n
        else:
            out[v] = (s - v + 1) ** n - (s - v) ** n
    return out


@op('counting.dist_stats', kind='probability', bound='100000 values')
def dist_stats(pairs):
    """Mean and variance of a distribution given as (value, weight) pairs (weights are normalised): {'mean': Fraction, 'variance': Fraction, 'total_weight': Fraction}.
    dist_stats([(1, 1), (2, 1), (3, 1)]) gives mean 2, variance 2/3."""
    if not isinstance(pairs, (list, tuple)) or not pairs or len(pairs) > 100000:
        raise BadInput('pairs is a non-empty list of (value, weight)')
    tw = s = s2 = Fraction(0)
    for p in pairs:
        if not isinstance(p, (list, tuple)) or len(p) != 2:
            raise BadInput('each entry is (value, weight)')
        x, w = _frac(p[0], 'value'), _frac(p[1], 'weight')
        if w < 0:
            raise BadInput('weights must be >= 0')
        tw += w
        s += w * x
        s2 += w * x * x
    if tw == 0:
        raise BadInput('all weights are zero')
    mean = s / tw
    return {'mean': mean, 'variance': s2 / tw - mean * mean, 'total_weight': tw}


@op('counting.bayes_posterior', kind='probability', bound='10000 hypotheses')
def bayes_posterior(priors, likelihoods):
    """Posterior probability of each hypothesis given the evidence: priors[i] * likelihoods[i] normalised (Fractions). bayes_posterior([1/2, 1/2], [1/4, 1/8]) = [2/3, 1/3]."""
    if not isinstance(priors, (list, tuple)) or len(priors) != len(likelihoods) or not priors or len(priors) > 10000:
        raise BadInput('priors and likelihoods are lists of the same non-zero length')
    pr = [_prob(p, 'prior') for p in priors]
    li = [_prob(p, 'likelihood') for p in likelihoods]
    if sum(pr) != 1:
        raise BadInput('priors must add up to 1')
    joint = [a * b for a, b in zip(pr, li)]
    tot = sum(joint)
    if tot == 0:
        raise BadInput('the evidence has probability 0')
    return [j / tot for j in joint]


@op('counting.binomial_probability', kind='probability', bound='n <= 20000')
def binomial_probability(n, p, lo, hi=None):
    """P(X = lo) (hi=None) or P(lo <= X <= hi) for X ~ Binomial(n, p), p a Fraction/"a/b". binomial_probability(10, "1/2", 5) = 63/256."""
    n, p = _nat(n, 'n', 20000), _prob(p, 'p')
    lo = _int(lo, 'lo')
    hi = lo if hi is None else _int(hi, 'hi')
    a, b = p.numerator, p.denominator
    num = sum(comb(n, k) * a ** k * (b - a) ** (n - k) for k in range(max(lo, 0), min(hi, n) + 1))
    return Fraction(num, b ** n)


@op('counting.hypergeometric_probability', kind='probability', bound='N <= 10**6')
def hypergeometric_probability(N, K, n, lo, hi=None):
    """Draw n from N items of which K are marked, without replacement: P(marked drawn = lo) or P(lo <= marked <= hi). hypergeometric_probability(52, 13, 5, 2) = 9139/33320."""
    N, K, n = _nat(N, 'N', 10 ** 6), _nat(K, 'K'), _nat(n, 'n')
    if K > N or n > N:
        raise BadInput('K and n must not exceed N')
    lo = _int(lo, 'lo')
    hi = lo if hi is None else _int(hi, 'hi')
    num = sum(comb(K, k) * comb(N - K, n - k) for k in range(max(lo, 0), min(hi, n, K) + 1) if n - k <= N - K)
    return Fraction(num, comb(N, n))


@op('counting.multivariate_hypergeometric', kind='probability', bound='total <= 10**6, 50 kinds')
def multivariate_hypergeometric(counts, drawn):
    """Probability of drawing exactly drawn[i] items of kind i when sum(drawn) items are taken without replacement from counts[i] items of each kind.
    multivariate_hypergeometric([4, 4, 4], [2, 1, 1]) = 32/165."""
    if not isinstance(counts, (list, tuple)) or len(counts) != len(drawn) or not counts or len(counts) > 50:
        raise BadInput('counts and drawn are lists of the same length')
    c = [_nat(x, 'count') for x in counts]
    d = [_nat(x, 'drawn') for x in drawn]
    if sum(c) > 10 ** 6:
        raise NotCovered('too large: total > 1e6')
    if any(x > y for x, y in zip(d, c)):
        return Fraction(0)
    num = 1
    for x, y in zip(c, d):
        num *= comb(x, y)
    return Fraction(num, comb(sum(c), sum(d)))


@op('counting.geometric_distribution', kind='probability', bound='n <= 10**6')
def geometric_distribution(p, n):
    """Number of trials until the first success, success probability p: {'pmf': P(T = n), 'cdf': P(T <= n), 'tail': P(T > n), 'mean': 1/p, 'variance': (1-p)/p**2}.
    geometric_distribution("1/6", 3) gives pmf 25/216."""
    p, n = _prob(p, 'p'), _nat(n, 'n', 10 ** 6)
    if p == 0:
        raise BadInput('p must be positive')
    q = 1 - p
    tail = q ** n
    return {'pmf': p * q ** (n - 1) if n >= 1 else Fraction(0), 'cdf': 1 - tail, 'tail': tail, 'mean': 1 / p, 'variance': q / (p * p)}


@op('counting.coupon_collector', kind='probability', bound='probs: <= 16 coupons; equal probabilities: n <= 100000')
def coupon_collector(n=None, m=None, probs=None):
    """Draws needed to collect coupons: {'mean': Fraction, 'variance': Fraction}. With n equally likely coupons: until m distinct ones (m=None: all n) are seen; with probs (a list of
    probabilities): until all are seen. coupon_collector(6) gives mean 147/10; coupon_collector(probs=["1/2", "1/2"]) gives mean 3."""
    if probs is not None:
        ps = [_prob(x, 'prob') for x in probs]
        if len(ps) > 16:
            raise NotCovered('too large: more than 16 coupons with given probabilities')
        if not ps or sum(ps) != 1 or any(x == 0 for x in ps):
            raise BadInput('probs: positive probabilities adding up to 1')
        mean = e2 = Fraction(0)
        for mask in range(1, 1 << len(ps)):
            P = sum(ps[i] for i in range(len(ps)) if mask >> i & 1)
            sgn = 1 if bin(mask).count('1') % 2 else -1
            mean += sgn / P
            e2 += sgn * (2 - P) / (P * P)
        return {'mean': mean, 'variance': e2 - mean * mean}
    n = _nat(n, 'n', 100000)
    m = n if m is None else _nat(m, 'm', n)
    mean = var = Fraction(0)
    for i in range(m):
        p = Fraction(n - i, n)
        mean += 1 / p
        var += (1 - p) / (p * p)
    return {'mean': mean, 'variance': var}


@op('counting.coupon_collector_probability', kind='probability', bound='n <= 2000, draws <= 20000')
def coupon_collector_probability(n, draws):
    """Probability that n equally likely coupons are all seen within `draws` draws (inclusion-exclusion). coupon_collector_probability(2, 3) = 3/4."""
    n, t = _nat(n, 'n', 2000), _nat(draws, 'draws', 20000)
    if n == 0:
        return Fraction(1)
    num = sum((-1) ** j * comb(n, j) * (n - j) ** t for j in range(n + 1))
    return Fraction(num, n ** t)


@op('counting.gamblers_ruin', kind='probability', bound='N <= 2000')
def gamblers_ruin(i, N, p):
    """A gambler with i units wins 1 with probability p, loses 1 otherwise, and stops at 0 or N: {'win': P(reach N), 'ruin': P(reach 0), 'expected_duration': Fraction}.
    gamblers_ruin(3, 10, "1/2") gives win 3/10 and expected duration 21."""
    i, N, p = _nat(i, 'i'), _nat(N, 'N', 2000), _frac(p, 'p')
    if not 0 < p < 1 or i > N or N < 1:
        raise BadInput('0 < p < 1 and 0 <= i <= N, N >= 1')
    q = 1 - p
    if i == 0 or i == N:
        return {'win': Fraction(int(i == N)), 'ruin': Fraction(int(i == 0)), 'expected_duration': Fraction(0)}
    if p == q:
        win = Fraction(i, N)
        dur = Fraction(i * (N - i))
    else:
        r = q / p
        win = (1 - r ** i) / (1 - r ** N)
        dur = Fraction(i) / (q - p) - Fraction(N) / (q - p) * win
    return {'win': win, 'ruin': 1 - win, 'expected_duration': dur}


@op('counting.birthday_probability', kind='probability', bound='n <= 10**5')
def birthday_probability(n, days):
    """n people with independent uniform birthdays among `days`: {'distinct': P(all different), 'collision': P(some two share)}. birthday_probability(23, 365) has collision just over 1/2."""
    n, d = _nat(n, 'n', 10 ** 5), _nat(days, 'days', 10 ** 9)
    if d < 1:
        raise BadInput('days must be positive')
    if n > d:
        return {'distinct': Fraction(0), 'collision': Fraction(1)}
    num = den = 1
    for i in range(n):
        num *= d - i
        den *= d
    p = Fraction(num, den)
    return {'distinct': p, 'collision': 1 - p}


# ---------------------------------------------------------------- Markov chains

def _stochastic(P, maxn=60):
    if not isinstance(P, (list, tuple)) or not P or len(P) > maxn:
        raise NotCovered(f'too large: more than {maxn} states') if isinstance(P, (list, tuple)) and len(P) > maxn else BadInput('P is a non-empty square matrix')
    n = len(P)
    M = []
    for r in P:
        if not isinstance(r, (list, tuple)) or len(r) != n:
            raise BadInput('P must be square')
        row = [_frac(x, 'entry') for x in r]
        if any(x < 0 for x in row) or sum(row) != 1:
            raise BadInput('each row must be non-negative and add up to 1')
        M.append(row)
    return M


def _solve(A, B):
    """Solve A X = B exactly (Gauss-Jordan over Fractions). A is n x n, B is n x m; NotCovered if A is singular."""
    n = len(A)
    M = [list(A[i]) + list(B[i]) for i in range(n)]
    for c in range(n):
        piv = next((r for r in range(c, n) if M[r][c] != 0), None)
        if piv is None:
            raise NotCovered('singular system')
        M[c], M[piv] = M[piv], M[c]
        inv = 1 / M[c][c]
        M[c] = [x * inv for x in M[c]]
        for r in range(n):
            if r != c and M[r][c] != 0:
                f = M[r][c]
                M[r] = [a - f * b for a, b in zip(M[r], M[c])]
    return [row[n:] for row in M]


def _reach(P):
    n = len(P)
    R = [[P[i][j] != 0 or i == j for j in range(n)] for i in range(n)]
    for k in range(n):
        for i in range(n):
            if R[i][k]:
                for j in range(n):
                    if R[k][j]:
                        R[i][j] = True
    return R


@op('counting.markov_absorption', kind='probability', bound='60 states')
def markov_absorption(P):
    """Absorbing Markov chain (state i is absorbing when P[i][i] = 1): for every transient state the probability of ending in each absorbing state and the expected number of steps,
    solved exactly. Returns {'absorbing': [...], 'transient': [...], 'probabilities': {i: {a: p}}, 'expected_steps': {i: steps}}. NotCovered when some transient state never
    reaches an absorbing one. markov_absorption([[1,0,0],[1/2,0,1/2],[0,0,1]]) gives from state 1: 1/2 to each, 1 step."""
    return _absorption(_stochastic(P))


def _absorption(M):
    n = len(M)
    ab = [i for i in range(n) if M[i][i] == 1]
    tr = [i for i in range(n) if i not in ab]
    if not ab:
        raise NotCovered('no absorbing state')
    R = _reach(M)
    if any(not any(R[i][a] for a in ab) for i in tr):
        raise NotCovered('a transient state never reaches an absorbing state')
    if not tr:
        return {'absorbing': ab, 'transient': [], 'probabilities': {}, 'expected_steps': {}}
    A = [[(1 if i == j else 0) - M[ti][tj] for j, tj in enumerate(tr)] for i, ti in enumerate(tr)]
    B = [[M[ti][a] for a in ab] + [Fraction(1)] for ti in tr]
    X = _solve(A, B)
    return {'absorbing': ab, 'transient': tr,
            'probabilities': {ti: {a: X[i][j] for j, a in enumerate(ab)} for i, ti in enumerate(tr)},
            'expected_steps': {ti: X[i][len(ab)] for i, ti in enumerate(tr)}}


@op('counting.markov_hitting_time', kind='probability', bound='60 states')
def markov_hitting_time(P, targets, start=None):
    """Expected number of steps to first reach any state in `targets`: a dict {state: Fraction or 'inf'} (the string 'inf' when the chain can get stuck away from the targets),
    or just the value for `start`. markov_hitting_time([[1/2, 1/2], [1/2, 1/2]], [1], 0) = 2."""
    M = _stochastic(P)
    n = len(M)
    tg = sorted({_nat(t, 'target', n - 1) for t in targets})
    if not tg:
        raise BadInput('targets must not be empty')
    rest = [i for i in range(n) if i not in tg]
    R = _reach(M)
    # a state has a finite expected time iff every state it can reach (without passing a target) can still reach a target
    stuck = {i for i in rest if not any(R[i][t] for t in tg)}
    infinite = set(stuck)
    changed = True
    while changed:
        changed = False
        for i in rest:
            if i not in infinite and any(M[i][j] != 0 and j in infinite for j in rest):
                infinite.add(i)
                changed = True
    fin = [i for i in rest if i not in infinite]
    h = {t: Fraction(0) for t in tg}
    for i in infinite:
        h[i] = 'inf'
    if fin:
        A = [[(1 if i == j else 0) - M[i][j] for j in fin] for i in fin]
        X = _solve(A, [[Fraction(1)] for _ in fin])
        for k, i in enumerate(fin):
            h[i] = X[k][0]
    if start is not None:
        return h[_nat(start, 'start', n - 1)]
    return dict(sorted(h.items()))


@op('counting.markov_stationary', kind='probability', bound='60 states')
def markov_stationary(P):
    """The stationary distribution (a list of Fractions) of a chain with exactly one closed class of states (so it is unique). markov_stationary([[1/2, 1/2], [1/4, 3/4]]) = [1/3, 2/3]."""
    M = _stochastic(P)
    n = len(M)
    R = _reach(M)
    closed = []
    seen = set()
    for i in range(n):
        if i in seen:
            continue
        cls = [j for j in range(n) if R[i][j] and R[j][i]]
        seen.update(cls)
        if all(not (R[c][j] and not R[j][c]) for c in cls for j in range(n)):
            closed.append(cls)
    if len(closed) != 1:
        raise NotCovered('the stationary distribution is not unique (several closed classes)')
    cls = closed[0]
    k = len(cls)
    # pi (P - I) = 0 on the class, with sum pi = 1: replace the last equation by the normalisation
    A = [[M[cls[j]][cls[i]] - (1 if i == j else 0) for j in range(k)] for i in range(k)]
    A[-1] = [Fraction(1)] * k
    B = [[Fraction(0)] for _ in range(k)]
    B[-1] = [Fraction(1)]
    X = _solve(A, B)
    pi = [Fraction(0)] * n
    for i, c in enumerate(cls):
        pi[c] = X[i][0]
    return pi


@op('counting.markov_distribution', kind='probability', bound='steps * states**2 <= 1e7')
def markov_distribution(P, start, steps):
    """Distribution after `steps` steps from a state index (or from a list of probabilities). markov_distribution([[0,1],[1,0]], 0, 3) = [0, 1]."""
    M = _stochastic(P)
    n = len(M)
    t = _nat(steps, 'steps', 10 ** 6)
    _work(t * n * n, 'markov_distribution')
    if isinstance(start, int) and not isinstance(start, bool):
        if not 0 <= start < n:
            raise BadInput('start is a state index')
        v = [Fraction(int(i == start)) for i in range(n)]
    else:
        v = [_prob(x, 'start') for x in start]
        if len(v) != n or sum(v) != 1:
            raise BadInput('start must be a distribution over the states')
    for _ in range(t):
        v = [sum(v[i] * M[i][j] for i in range(n) if v[i]) for j in range(n)]
    return v


def _pattern_chain(patterns, alpha, probs):
    """The Aho-Corasick chain that stops when any pattern first ends: states 0..ns-1 (stop states absorbing)."""
    for p in patterns:
        if not isinstance(p, str) or not p or set(p) - set(alpha):
            raise BadInput('a pattern is a non-empty string over the alphabet')
    for i, p in enumerate(patterns):
        for j, q in enumerate(patterns):
            if i != j and p in q:
                raise NotCovered('a pattern contains another: the first to appear is not defined')
    nxt, fail, term = [{}], [0], [-1]
    for pi, p in enumerate(patterns):
        s = 0
        for ch in p:
            if ch not in nxt[s]:
                nxt.append({})
                fail.append(0)
                term.append(-1)
                nxt[s][ch] = len(nxt) - 1
            s = nxt[s][ch]
        term[s] = pi
    queue = list(nxt[0].values())
    qi = 0
    while qi < len(queue):
        s = queue[qi]
        qi += 1
        for ch, t in nxt[s].items():
            f = fail[s]
            while f and ch not in nxt[f]:
                f = fail[f]
            fail[t] = nxt[f][ch] if ch in nxt[f] and nxt[f][ch] != t else 0
            queue.append(t)
    ns = len(nxt)
    delta = [[0] * len(alpha) for _ in range(ns)]
    for s in [0] + queue:
        for ci, ch in enumerate(alpha):
            delta[s][ci] = nxt[s][ch] if ch in nxt[s] else (delta[fail[s]][ci] if s else 0)
    P = [[Fraction(0)] * ns for _ in range(ns)]
    for s in range(ns):
        if term[s] >= 0:
            P[s][s] = Fraction(1)
            continue
        for ci in range(len(alpha)):
            P[s][delta[s][ci]] += probs[ci]
    return P, term


@op('counting.pattern_race', kind='probability', bound='patterns total length <= 60')
def pattern_race(patterns, alphabet='HT', probs=None):
    """Independent random letters from an alphabet (equally likely unless probs is given): the probability that each pattern is the first to appear, and the expected number of
    letters until one does: {'probabilities': [...], 'expected_length': Fraction}. pattern_race(['HH', 'TH'], 'HT') has probabilities [1/4, 3/4]."""
    alpha = _alphabet(alphabet)
    pats = list(patterns)
    if not pats or sum(len(p) for p in pats if isinstance(p, str)) > 60:
        raise NotCovered('too large: patterns longer than 60 in total (or none)')
    pr = [Fraction(1, len(alpha))] * len(alpha) if probs is None else [_prob(x, 'prob') for x in probs]
    if len(pr) != len(alpha) or sum(pr) != 1:
        raise BadInput('probs: one probability per letter, adding up to 1')
    P, term = _pattern_chain(pats, alpha, pr)
    res = _absorption(P)
    ab = res['absorbing']
    if 0 in ab:
        raise BadInput('the empty start state is absorbing')
    out = [Fraction(0)] * len(pats)
    for a in ab:
        out[term[a]] = res['probabilities'][0][a]
    return {'probabilities': out, 'expected_length': res['expected_steps'][0]}


@op('counting.expected_wait_pattern', kind='probability', bound='pattern length <= 1000')
def expected_wait_pattern(pattern, alphabet='HT'):
    """Expected number of independent equally likely letters until a pattern first appears (sum of k**i over the i where the first i letters equal the last i letters).
    expected_wait_pattern('HH') = 6; expected_wait_pattern('HT') = 4."""
    alpha = _alphabet(alphabet)
    if not isinstance(pattern, str) or not pattern or set(pattern) - set(alpha) or len(pattern) > 1000:
        raise BadInput('a pattern is a non-empty string over the alphabet')
    k = len(alpha)
    return sum(k ** i for i in range(1, len(pattern) + 1) if pattern[:i] == pattern[-i:])


# ---------------------------------------------------------------- random walks, chords

@op('counting.walk_distribution', kind='probability', bound='states * steps * moves <= 1e7')
def walk_distribution(steps, n, start=None, absorbing=(), lo=None, hi=None):
    """Random walk of n moves. steps: list of (move, weight) where a move is an int (1-D) or a tuple of ints (several dimensions); weights are normalised.
    Positions in `absorbing` stop the walk; in 1-D lo / hi are absorbing walls. Returns {position: probability} (Fractions, sorted).
    walk_distribution([(1, 1), (-1, 1)], 4) gives position 0 with probability 3/8."""
    if not isinstance(steps, (list, tuple)) or not steps:
        raise BadInput('steps is a non-empty list of (move, weight)')
    moves, weights = [], []
    for s in steps:
        if not isinstance(s, (list, tuple)) or len(s) != 2:
            raise BadInput('a step is (move, weight)')
        mv, w = s
        mv = (mv,) if isinstance(mv, int) and not isinstance(mv, bool) else tuple(_int(x, 'move') for x in mv)
        w = _frac(w, 'weight')
        if w < 0:
            raise BadInput('weights must be >= 0')
        moves.append(mv)
        weights.append(w)
    dim = len(moves[0])
    if any(len(m) != dim for m in moves):
        raise BadInput('all moves need the same dimension')
    tw = sum(weights)
    if tw == 0:
        raise BadInput('all weights are zero')
    probs = [w / tw for w in weights]
    n = _nat(n, 'n', 10 ** 5)
    st = tuple([0] * dim) if start is None else ((start,) if isinstance(start, int) else tuple(start))
    if len(st) != dim:
        raise BadInput('start has the wrong dimension')
    stops = {(a,) if isinstance(a, int) else tuple(a) for a in absorbing}
    cur = {st: Fraction(1)}
    for _ in range(n):
        new = {}
        _work(len(cur) * len(moves), 'walk_distribution')
        for pos, w in cur.items():
            if pos in stops or (dim == 1 and ((lo is not None and pos[0] <= lo) or (hi is not None and pos[0] >= hi))):
                new[pos] = new.get(pos, 0) + w
                continue
            for mv, p in zip(moves, probs):
                if p:
                    q = tuple(a + b for a, b in zip(pos, mv))
                    new[q] = new.get(q, 0) + w * p
        cur = new
        if len(cur) > 2 * 10 ** 6:
            raise NotCovered('too large: more than 2e6 positions')
    out = {(k[0] if dim == 1 else k): v for k, v in sorted(cur.items())}
    return out


def _cross(a, b, c, d):
    lo, hi = min(a, b), max(a, b)
    if len({a, b, c, d}) < 4:
        return False
    return (lo < c < hi) != (lo < d < hi)


@op('counting.chords_cross', kind='probability', bound='any')
def chords_cross(a, b, c, d):
    """Do the chords (a, b) and (c, d) of a circle cross? The numbers are the positions of the endpoints around the circle (distinct numbers = distinct points; chords that share
    an endpoint do not cross). chords_cross(1, 3, 2, 4) = True."""
    return _cross(*(_int(x, 'position') for x in (a, b, c, d)))


@op('counting.crossing_pairs', kind='probability', bound='2000 chords')
def crossing_pairs(chords):
    """Number of pairs of chords (given as pairs of positions around a circle) that cross in the interior. crossing_pairs([(1,3),(2,4),(5,6)]) = 1."""
    if not isinstance(chords, (list, tuple)) or len(chords) > 2000:
        raise NotCovered('too large: more than 2000 chords')
    cs = []
    for ch in chords:
        if not isinstance(ch, (list, tuple)) or len(ch) != 2:
            raise BadInput('a chord is a pair of positions')
        cs.append((_int(ch[0], 'position'), _int(ch[1], 'position')))
    return sum(1 for i in range(len(cs)) for j in range(i + 1, len(cs)) if _cross(*cs[i], *cs[j]))


@op('counting.chord_crossing_probability', kind='probability', bound='n_points <= 10**6')
def chord_crossing_probability(model, n_points=None):
    """Probability that two random chords of a circle cross. Models:
    'four_points': four independent uniform points (continuous), or four distinct points among n_points paired at random: 1/3.
    'two_distinct_chords': two different chords chosen among all C(n,2) chords through n points: C(n,4) / C(C(n,2), 2).
    'independent_chords': two chords each chosen uniformly (independently) among the C(n,2): 2 C(n,4) / C(n,2)**2.
    'independent_points': each chord from two independent uniform picks among n points (a repeated point makes no crossing): n(n-1)(n-2)(n-3) / (3 n**4).
    chord_crossing_probability('four_points') = 1/3; chord_crossing_probability('two_distinct_chords', 6) = 1/7."""
    if model == 'four_points':
        if n_points is not None and _nat(n_points, 'n_points') < 4:
            raise BadInput('need at least 4 points')
        return Fraction(1, 3)
    n = _nat(n_points, 'n_points', 10 ** 6)
    if n < 1:
        raise BadInput('n_points must be positive')
    if model == 'two_distinct_chords':
        c = comb(n, 2)
        if c < 2:
            raise BadInput('need at least two chords')
        return Fraction(comb(n, 4), comb(c, 2))
    if model == 'independent_chords':
        if n < 2:
            raise BadInput('need at least one chord')
        return Fraction(2 * comb(n, 4), comb(n, 2) ** 2)
    if model == 'independent_points':
        return Fraction(n * (n - 1) * (n - 2) * (n - 3), 3 * n ** 4)
    raise BadInput("model: 'four_points', 'two_distinct_chords', 'independent_chords' or 'independent_points'")


@op('counting.matching_crossings', kind='probability', bound='m <= 2000')
def matching_crossings(m):
    """A uniformly random perfect matching of 2m points on a circle: {'matchings': (2m-1)!!, 'non_crossing': Catalan(m), 'probability_non_crossing': Fraction,
    'expected_crossings': C(m,2)/3}. matching_crossings(3) has 15 matchings, 5 non-crossing."""
    m = _nat(m, 'm', 100000)
    if m > 2000:
        raise NotCovered('too large: m > 2000')
    tot = 1
    for i in range(1, 2 * m, 2):
        tot *= i
    cat = catalan(m)
    return {'matchings': tot, 'non_crossing': cat, 'probability_non_crossing': Fraction(cat, tot), 'expected_crossings': Fraction(comb(m, 2), 3)}


NOT_COVERED = {
    'chromatic polynomial of graphs with more than 13 vertices': 'the subset DP is O(3**n); a larger graph needs deletion-contraction with isomorphism hashing',
    'perfect matchings counted with Pfaffians (planar graphs, large n)': 'needs a planar orientation and an exact Pfaffian; only the bitmask DP (n <= 22) is registered',
    'general graph isomorphism / orbit counting of graphs': 'only automorphism counts of tiny graphs by backtracking are registered',
    'counting with a predicate given as program text beyond the small expression language': 'no code is executed; use brute_count with the predicate language',
    'probability of continuous geometric events (Buffon, random triangles on a circle)': 'not a finite exact count; belongs to geometry/calculus with a model of the randomness',
    'unbounded or infinite-state Markov chains': 'only finite chains of at most 60 states, solved exactly',
}
