"""Simulation of tools/igc/mathbackend/counting.py: random cases against an independent oracle, no model. See docs/plans/2026-11-05-mip-math-backend-coverage.md.

    PYTHONPATH=. PYTHONUTF8=1 <venv python with sympy> -m tools.igc.mathbackend.sim_counting --seed 1 --n 60

generate(seed, n) -> list of Case (n cases for every operation); run(seed, n) -> dict (also written to build-mathbackend/counting.json).
A Case is (op_name, args, oracle_value, oracle_name) plus a kwargs dict. The oracle of a case is named in it:
  brute          an explicit enumeration written here with itertools / loops, sharing no code with the backend
  closed form    an identity or formula written here (the formula is not the backend's algorithm)
  sympy          SymPy 1.14 (nothing is copied from SymPy into counting.py, so it is independent)
  certificate    the answer is substituted back into the defining equations
  expected refusal   the case is past the operation's bound and the backend must raise NotCovered
"""
import argparse
import itertools
import json
import math
import random
import time
from collections import namedtuple
from fractions import Fraction
from pathlib import Path

import sympy
from sympy import Matrix, Rational
from sympy.functions.combinatorial.numbers import bell, catalan as sp_catalan, harmonic, stirling, subfactorial
from sympy.functions.combinatorial.numbers import partition as npartitions, totient

from tools.igc.mathbackend import BadInput, NotCovered
from tools.igc.mathbackend import counting as C

Case = namedtuple('Case', 'op args oracle oracle_name kwargs')


class Refuse:
    """The oracle's value: the operation must refuse (NotCovered)."""
    def __init__(self, why=''):
        self.why = why


def mk(op, args, val, name, **kw):
    return Case(op, tuple(args), val, name, kw)


def fr(x):
    """sympy Rational / int -> Fraction / int."""
    x = sympy.nsimplify(x) if not isinstance(x, (int, Rational)) else x
    if isinstance(x, int):
        return x
    return int(x) if x.q == 1 else Fraction(int(x.p), int(x.q))


def sp_to_fr(x):
    x = Rational(x)
    return Fraction(int(x.p), int(x.q))


# ---------------------------------------------------------------- small shared helpers (oracle side)

def rand_graph(rng, n, p=None, maxm=None):
    p = rng.choice([0.25, 0.4, 0.6, 0.8]) if p is None else p
    es = [(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p]
    if maxm is not None and len(es) > maxm:
        es = sorted(rng.sample(es, maxm))
    return es


def uf_find(par, x):
    while par[x] != x:
        par[x] = par[par[x]]
        x = par[x]
    return x


def is_tree(n, es):
    if len(es) != n - 1:
        return False
    par = list(range(n))
    for u, v in es:
        a, b = uf_find(par, u), uf_find(par, v)
        if a == b:
            return False
        par[a] = b
    return True


def lagrange_coeffs(vals):
    """Coefficients (low power first, ints) of the polynomial of degree <= len(vals)-1 through (k, vals[k])."""
    x = sympy.symbols('x')
    poly = sympy.interpolate(list(zip(range(len(vals)), vals)), x)
    cs = sympy.Poly(poly, x).all_coeffs()[::-1]
    return [int(c) for c in cs]


def set_partitions_list(n):
    out = []

    def go(i, blocks):
        if i == n:
            out.append([list(b) for b in blocks])
            return
        for b in blocks:
            b.append(i)
            go(i + 1, blocks)
            b.pop()
        blocks.append([i])
        go(i + 1, blocks)
        blocks.pop()
    go(0, [])
    return out


# ---------------------------------------------------------------- generators, one per operation

def g_stars_and_bars(rng, n):
    cs = []
    for i in range(n):
        if i % 2 == 0:
            k = rng.randint(1, 5)
            N = rng.randint(0, 10)
            kind = rng.choice(['none', 'scalar', 'list'])
            low = rng.choice([0, 0, 1, [rng.randint(0, 2) for _ in range(k)]])
            up = None if kind == 'none' else (rng.randint(0, 6) if kind == 'scalar' else [rng.choice([None, rng.randint(0, 6)]) for _ in range(k)])
            lows = [low] * k if isinstance(low, int) else low
            ups = [N] * k if up is None else ([up] * k if isinstance(up, int) else [N if u is None else u for u in up])
            cnt = sum(1 for t in itertools.product(*[range(l, u + 1) for l, u in zip(lows, ups)]) if sum(t) == N)
            cs.append(mk('counting.stars_and_bars', (N, k), cnt, 'brute', upper=up, lower=low))
        else:
            k, N, u = rng.randint(2, 12), rng.randint(0, 200), rng.randint(1, 30)
            val = sum((-1) ** j * math.comb(k, j) * math.comb(N - j * (u + 1) + k - 1, k - 1) for j in range(k + 1) if N - j * (u + 1) >= 0)
            if rng.random() < 0.3:
                cs.append(mk('counting.stars_and_bars', (N, k), math.comb(N + k - 1, k - 1), 'closed form C(n+k-1,k-1)'))
            else:
                cs.append(mk('counting.stars_and_bars', (N, k), val, 'closed form inclusion-exclusion', upper=u))
    return cs


def g_derangements(rng, n):
    cs = []
    for i in range(n):
        if i < 9:
            m = i
            val = sum(1 for p in itertools.permutations(range(m)) if all(p[j] != j for j in range(m)))
            cs.append(mk('counting.derangements', (m,), val, 'brute'))
        else:
            m = rng.randint(9, 120)
            val = sum(Fraction((-1) ** k, math.factorial(k)) for k in range(m + 1)) * math.factorial(m)
            cs.append(mk('counting.derangements', (m,), int(val), 'closed form n! sum (-1)^k/k!'))
    return cs


def g_rencontres(rng, n):
    cs = []
    for i in range(n):
        if i % 2 == 0:
            m, k = rng.randint(0, 7), rng.randint(0, 8)
            val = sum(1 for p in itertools.permutations(range(m)) if sum(1 for j in range(m) if p[j] == j) == k)
            cs.append(mk('counting.rencontres', (m, k), val, 'brute'))
        else:
            m = rng.randint(8, 80)
            k = rng.randint(0, m)
            val = Fraction(math.factorial(m), math.factorial(k)) * sum(Fraction((-1) ** j, math.factorial(j)) for j in range(m - k + 1))
            cs.append(mk('counting.rencontres', (m, k), int(val), 'closed form n!/k! sum (-1)^j/j!'))
    return cs


def g_surjections(rng, n):
    cs = []
    for i in range(n):
        if i % 2 == 0:
            m, k = rng.randint(0, 6), rng.randint(0, 4)
            val = sum(1 for f in itertools.product(range(k), repeat=m) if len(set(f)) == k)
            cs.append(mk('counting.surjections', (m, k), val, 'brute'))
        else:
            m, k = rng.randint(1, 40), rng.randint(1, 15)
            val = int(math.factorial(k) * stirling(m, k, kind=2)) if k <= m else 0
            cs.append(mk('counting.surjections', (m, k), val, 'sympy k! * stirling2'))
    return cs


def g_set_partitions(rng, n):
    cs = []
    lists = {m: set_partitions_list(m) for m in range(0, 8)}
    for i in range(n):
        r = i % 4
        if r == 0:
            m = rng.randint(0, 7)
            cs.append(mk('counting.set_partitions', (m,), len(lists[m]), 'brute'))
        elif r == 1:
            m, k = rng.randint(0, 7), rng.randint(0, 8)
            cs.append(mk('counting.set_partitions', (m, k), sum(1 for p in lists[m] if len(p) == k), 'brute'))
        elif r == 2:
            m = rng.randint(1, 7)
            sizes = sorted(rng.sample(range(1, 5), rng.randint(1, 3)))
            k = rng.choice([None, rng.randint(1, 4)])
            val = sum(1 for p in lists[m] if all(len(b) in sizes for b in p) and (k is None or len(p) == k))
            cs.append(mk('counting.set_partitions', (m, k), val, 'brute', block_sizes=sizes))
        else:
            m = rng.randint(8, 60)
            if rng.random() < 0.5:
                cs.append(mk('counting.set_partitions', (m,), int(bell(m)), 'sympy bell'))
            else:
                k = rng.randint(1, m)
                cs.append(mk('counting.set_partitions', (m, k), int(stirling(m, k, kind=2)), 'sympy stirling2'))
    return cs


def g_twelvefold(rng, n):
    cs = []
    for i in range(n):
        m, k = rng.randint(0, 5), rng.randint(0, 4)
        balls, boxes = rng.choice(['labelled', 'identical']), rng.choice(['labelled', 'identical'])
        rule = rng.choice(['any', 'injective', 'surjective'])
        seen = set()
        for f in itertools.product(range(k), repeat=m):
            occ = tuple(f.count(b) for b in range(k))
            if rule == 'injective' and max(occ, default=0) > 1:
                continue
            if rule == 'surjective' and min(occ, default=1) < 1 and k > 0:
                continue
            if rule == 'surjective' and k == 0 and m > 0:
                continue
            if balls == 'labelled' and boxes == 'labelled':
                key = f
            elif balls == 'identical' and boxes == 'labelled':
                key = occ
            elif balls == 'labelled':
                key = frozenset(frozenset(j for j in range(m) if f[j] == b) for b in range(k) if b in f)
            else:
                key = tuple(sorted(occ))
            seen.add(key)
        cs.append(mk('counting.twelvefold_way', (m, k, balls, boxes, rule), len(seen), 'brute enumeration of all functions, classes by canonical form'))
    return cs


def g_cycle_type(rng, n):
    cs = []
    for i in range(n):
        m = rng.randint(1, 7)
        ct, left = [], m
        while left:
            c = rng.randint(1, left)
            ct.append(c)
            left -= c
        want = sorted(ct)
        cnt = 0
        for p in itertools.permutations(range(m)):
            seen, cyc = [False] * m, []
            for s in range(m):
                if not seen[s]:
                    L, j = 0, s
                    while not seen[j]:
                        seen[j] = True
                        j = p[j]
                        L += 1
                    cyc.append(L)
            cnt += sorted(cyc) == want
        cs.append(mk('counting.permutations_with_cycle_type', (ct,), cnt, 'brute'))
    return cs


def g_eulerian(rng, n):
    cs = []
    for i in range(n):
        if i % 2 == 0:
            m, k = rng.randint(1, 7), rng.randint(0, 7)
            val = sum(1 for p in itertools.permutations(range(m)) if sum(1 for j in range(m - 1) if p[j] > p[j + 1]) == k)
            cs.append(mk('counting.eulerian', (m, k), val, 'brute'))
        else:
            m = rng.randint(2, 40)
            k = rng.randint(0, m - 1)
            val = sum((-1) ** j * math.comb(m + 1, j) * (k + 1 - j) ** m for j in range(k + 2))
            cs.append(mk('counting.eulerian', (m, k), val, 'closed form explicit sum'))
    return cs


def g_mahonian(rng, n):
    cs = []
    q = sympy.symbols('q')
    for i in range(n):
        if i % 2 == 0:
            m, k = rng.randint(1, 7), rng.randint(0, 22)
            val = sum(1 for p in itertools.permutations(range(m)) if sum(1 for a in range(m) for b in range(a + 1, m) if p[a] > p[b]) == k)
            cs.append(mk('counting.mahonian', (m, k), val, 'brute'))
        else:
            m = rng.randint(2, 12)
            k = rng.randint(0, m * (m - 1) // 2 + 2)
            poly = sympy.expand(sympy.prod([sum(q ** j for j in range(i2)) for i2 in range(1, m + 1)]))
            cs.append(mk('counting.mahonian', (m, k), int(sympy.Poly(poly, q).coeff_monomial(q ** k)), 'sympy product of (1+q+..+q^(i-1))'))
    return cs


def g_anagram(rng, n):
    cs = []
    for i in range(n):
        if i % 3 != 2:
            L = rng.randint(1, 8)
            w = ''.join(rng.choice('ABCD'[:rng.randint(1, 4)]) for _ in range(L))
            adj = rng.random() < 0.5
            perms = set(itertools.permutations(w))
            val = sum(1 for p in perms if not adj or all(p[j] != p[j + 1] for j in range(len(p) - 1)))
            cs.append(mk('counting.anagram_count', (w, adj), val, 'brute set of permutations'))
        else:
            L = rng.randint(9, 40)
            w = ''.join(rng.choice('ABCDE') for _ in range(L))
            den = 1
            for ch in set(w):
                den *= math.factorial(w.count(ch))
            cs.append(mk('counting.anagram_count', (w,), math.factorial(L) // den, 'closed form n!/prod m_i!'))
    return cs


def g_subset_sum(rng, n):
    cs = []
    for i in range(n):
        L = rng.randint(0, 12)
        vals = [rng.randint(-3, 9) for _ in range(L)]
        size = rng.choice([None, None, rng.randint(0, 5)])
        if rng.random() < 0.5:
            target = rng.randint(-3, 25)
            val = sum(1 for m in range(1 << L) if sum(vals[j] for j in range(L) if m >> j & 1) == target and (size is None or bin(m).count('1') == size))
            cs.append(mk('counting.count_subsets_with_sum', (vals,), val, 'brute subsets', target=target, size=size))
        else:
            md, rem = rng.randint(1, 7), rng.randint(0, 9)
            val = sum(1 for m in range(1 << L) if sum(vals[j] for j in range(L) if m >> j & 1) % md == rem % md and (size is None or bin(m).count('1') == size))
            cs.append(mk('counting.count_subsets_with_sum', (vals,), val, 'brute subsets', mod=md, rem=rem, size=size))
    return cs


def partitions_brute(n, maxpart=None):
    """All partitions of n as non-increasing tuples."""
    out = []

    def go(rem, mx, cur):
        if rem == 0:
            out.append(tuple(cur))
            return
        for p in range(min(rem, mx), 0, -1):
            cur.append(p)
            go(rem - p, p, cur)
            cur.pop()
    go(n, n if maxpart is None else maxpart, [])
    return out


def g_partition_count(rng, n):
    cs = []
    allp = {m: partitions_brute(m) for m in range(0, 26)}
    for i in range(n):
        if i % 5 == 4:
            m = rng.randint(30, 400)
            cs.append(mk('counting.partition_count', (m,), int(npartitions(m)), 'sympy npartitions'))
            continue
        m = rng.randint(0, 25)
        kw = {}
        pred = []
        if rng.random() < 0.3:
            ps = sorted(rng.sample(range(1, 9), rng.randint(1, 4)))
            kw['parts'] = ps
            pred.append(lambda p, ps=ps: all(x in ps for x in p))
        if rng.random() < 0.3:
            kw['distinct'] = True
            pred.append(lambda p: len(set(p)) == len(p))
        r = rng.random()
        if r < 0.25:
            k = rng.randint(0, 6)
            kw['num_parts'] = k
            pred.append(lambda p, k=k: len(p) == k)
        elif r < 0.5:
            k = rng.randint(0, 6)
            kw['max_parts'] = k
            pred.append(lambda p, k=k: len(p) <= k)
        if rng.random() < 0.25:
            mp = rng.randint(1, 8)
            kw['max_part'] = mp
            pred.append(lambda p, mp=mp: all(x <= mp for x in p))
        if rng.random() < 0.2:
            mn = rng.randint(1, 4)
            kw['min_part'] = mn
            pred.append(lambda p, mn=mn: all(x >= mn for x in p))
        val = sum(1 for p in allp[m] if all(f(p) for f in pred))
        cs.append(mk('counting.partition_count', (m,), val, 'brute enumeration of all partitions', **kw))
    return cs


def g_compositions(rng, n):
    cs = []

    def comps(m, parts):
        if m == 0:
            yield ()
            return
        for p in parts:
            if p <= m:
                for rest in comps(m - p, parts):
                    yield (p,) + rest
    for i in range(n):
        if i % 4 == 3:
            m = rng.randint(1, 60)
            r = rng.random()
            if r < 0.34:
                cs.append(mk('counting.compositions_count', (m,), 2 ** (m - 1), 'closed form 2^(n-1)'))
            elif r < 0.67:
                k = rng.randint(1, m)
                cs.append(mk('counting.compositions_count', (m, k), math.comb(m - 1, k - 1), 'closed form C(n-1,k-1)'))
            else:
                cs.append(mk('counting.compositions_count', (m,), int(sympy.fibonacci(m + 1)), 'sympy Fibonacci (parts 1 and 2)', parts=[1, 2]))
            continue
        m = rng.randint(0, 13)
        r = rng.random()
        if r < 0.4:
            parts = sorted(rng.sample(range(1, 7), rng.randint(1, 3)))
            k = rng.choice([None, rng.randint(0, 5)])
            val = sum(1 for c in comps(m, parts) if k is None or len(c) == k)
            cs.append(mk('counting.compositions_count', (m, k), val, 'brute enumeration', parts=parts))
        else:
            lo, hi = rng.randint(1, 3), rng.choice([None, rng.randint(2, 6)])
            if hi is not None and hi < lo:
                hi = lo
            k = rng.choice([None, rng.randint(1, 5)])
            parts = list(range(lo, (hi if hi is not None else m) + 1))
            val = sum(1 for c in comps(m, parts) if k is None or len(c) == k)
            cs.append(mk('counting.compositions_count', (m, k), val, 'brute enumeration', min_part=lo, max_part=hi))
    return cs


def shapes_upto(n):
    out = []
    for m in range(1, n + 1):
        out.extend(partitions_brute(m))
    return out


def syt_rec(shape, memo={}):
    shape = tuple(x for x in shape if x)
    if not shape:
        return 1
    if shape in memo:
        return memo[shape]
    t = 0
    for i in range(len(shape)):
        if i == len(shape) - 1 or shape[i] > shape[i + 1]:
            s = list(shape)
            s[i] -= 1
            t += syt_rec(tuple(s))
    memo[shape] = t
    return t


def ssyt_rec(shape, k, memo={}):
    shape = tuple(x for x in shape if x)
    if not shape:
        return 1
    if k == 0 or len(shape) > k:
        return 0
    key = (shape, k)
    if key in memo:
        return memo[key]
    ranges = []
    for i, x in enumerate(shape):
        lo = shape[i + 1] if i + 1 < len(shape) else 0
        ranges.append(range(lo, x + 1))
    t = 0
    for mu in itertools.product(*ranges):
        t += ssyt_rec(mu, k - 1)
    memo[key] = t
    return t


def weyl_dim(shape, k):
    lam = list(shape) + [0] * (k - len(shape))
    r = Fraction(1)
    for i in range(k):
        for j in range(i + 1, k):
            r *= Fraction(lam[i] - lam[j] + j - i, j - i)
    return int(r)


def g_syt(rng, n):
    shs = shapes_upto(14)
    cs = []
    for i in range(n):
        sh = list(rng.choice(shs))
        cs.append(mk('counting.standard_young_tableaux', (sh,), syt_rec(tuple(sh)), 'brute recurrence (remove the largest entry from a corner)'))
    return cs


def g_ssyt(rng, n):
    shs = shapes_upto(7)
    cs = []
    for i in range(n):
        if i % 3 != 2:
            sh = list(rng.choice(shs))
            k = rng.randint(1, 4)
            cs.append(mk('counting.semistandard_young_tableaux', (sh, k), ssyt_rec(tuple(sh), k), 'brute recurrence (remove a horizontal strip)'))
        else:
            sh = list(rng.choice(shapes_upto(12)))
            k = rng.randint(len(sh), len(sh) + 6)
            cs.append(mk('counting.semistandard_young_tableaux', (sh, k), weyl_dim(sh, k), 'closed form Weyl dimension formula'))
    return cs


def necklace_brute(n, k, bracelet, counts=None):
    seen = set()
    if counts is not None:
        base = [c for c, m in enumerate(counts) for _ in range(m)]
        it = set(itertools.permutations(base))
    else:
        it = itertools.product(range(k), repeat=n)
    for w in it:
        forms = [w[i:] + w[:i] for i in range(n)]
        if bracelet:
            r = w[::-1]
            forms += [r[i:] + r[:i] for i in range(n)]
        seen.add(min(forms))
    return len(seen)


def g_necklace(rng, n):
    cs = []
    for i in range(n):
        if i % 3 == 2:
            counts = [rng.randint(0, 3) for _ in range(rng.randint(2, 4))]
            L = sum(counts)
            if L == 0:
                counts[0] = 2
                L = 2
            br = rng.random() < 0.5
            cs.append(mk('counting.necklace_count', (), necklace_brute(L, len(counts), br, counts), 'brute canonical form', counts=counts, bracelet=br))
        else:
            L, k = rng.randint(1, 8), rng.randint(1, 3)
            br = rng.random() < 0.5
            cs.append(mk('counting.necklace_count', (L, k), necklace_brute(L, k, br), 'brute canonical form', bracelet=br))
    return cs


def orbits_by_union_find(npts, gens, colorings):
    par = {c: c for c in colorings}

    def find(x):
        while par[x] != x:
            par[x] = par[par[x]]
            x = par[x]
        return x
    for c in colorings:
        for g in gens:
            d = tuple(c[g[j]] for j in range(npts))
            if d in par:
                a, b = find(c), find(d)
                if a != b:
                    par[a] = b
    return len({find(c) for c in colorings})


def cube_vertex_gens():
    verts = [(x, y, z) for x in (0, 1) for y in (0, 1) for z in (0, 1)]
    idx = {v: i for i, v in enumerate(verts)}
    rz = [idx[(1 - y, x, z)] for x, y, z in verts]
    rx = [idx[(x, 1 - z, y)] for x, y, z in verts]
    return [rz, rx]


def cube_face_gens():
    faces = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
    idx = {f: i for i, f in enumerate(faces)}
    rz = [idx[(-b, a, c)] for a, b, c in faces]
    rx = [idx[(a, -c, b)] for a, b, c in faces]
    return [rz, rx]


def g_orbit(rng, n):
    cs = []
    groups = []
    for m in range(3, 9):
        groups.append((f'cyclic {m}', m, [[(j + 1) % m for j in range(m)]]))
        groups.append((f'dihedral {m}', m, [[(j + 1) % m for j in range(m)], [(-j) % m for j in range(m)]]))
    groups.append(('cube vertices', 8, cube_vertex_gens()))
    groups.append(('cube faces', 6, cube_face_gens()))
    groups.append(('symmetric 5', 5, [[1, 2, 3, 4, 0], [1, 0, 2, 3, 4]]))
    groups.append(('klein-ish swaps 6', 6, [[1, 0, 2, 3, 4, 5], [0, 1, 3, 2, 4, 5], [0, 1, 2, 3, 5, 4]]))
    for i in range(n):
        name, m, gens = rng.choice(groups)
        if i % 2 == 0 or m > 8:
            k = rng.randint(1, 3 if m <= 8 else 2)
            cols = list(itertools.product(range(k), repeat=m))
            cs.append(mk('counting.orbit_count_colorings', (m, gens, k), orbits_by_union_find(m, gens, cols), f'brute orbits by union-find ({name})'))
        else:
            counts = [rng.randint(0, 3) for _ in range(rng.randint(2, 3))]
            counts[0] += max(0, m - sum(counts))
            if sum(counts) != m:
                counts = [m - 1, 1]
            base = [c for c, t in enumerate(counts) for _ in range(t)]
            cols = list(set(itertools.permutations(base)))
            cs.append(mk('counting.orbit_count_colorings', (m, gens, None, counts), orbits_by_union_find(m, gens, cols), f'brute orbits by union-find ({name}, fixed content)'))
    return cs


def lattice_dfs(start, end, steps, forb, region, constraint):
    x0, y0, x1, y1 = region

    def ok(x, y):
        if not (x0 <= x <= x1 and y0 <= y <= y1) or (x, y) in forb:
            return False
        return {None: True, 'y<=x': y <= x, 'y<x': y < x, 'y>=x': y >= x, 'y>x': y > x}[constraint]

    def go(p):
        if not ok(*p):
            return 0
        if p == end:
            return 1
        return sum(go((p[0] + dx, p[1] + dy)) for dx, dy in steps)
    return go(start)


def g_lattice(rng, n):
    cs = []
    for i in range(n):
        r = i % 5
        if r == 4:
            a, b = rng.randint(0, 60), rng.randint(0, 60)
            cs.append(mk('counting.lattice_paths', ((a, b),), math.comb(a + b, a), 'closed form C(a+b,a)'))
            continue
        ex, ey = rng.randint(0, 5), rng.randint(0, 5)
        steps = rng.choice([((1, 0), (0, 1)), ((1, 0), (0, 1), (1, 1)), ((1, 0), (0, 1)), ((2, 0), (0, 1), (1, 1)), ((1, 1), (1, -1))])
        sx, sy = rng.choice([(0, 0), (0, 0), (1, 0), (0, 2)])
        if steps == ((1, 1), (1, -1)):
            ex, ey = rng.randint(1, 8), rng.randint(-2, 3)
            region = (0, -3, 8, 3)
        else:
            region = None
            if rng.random() < 0.3:
                region = (0, 0, ex, ey)
        forb = [(rng.randint(0, 5), rng.randint(0, 5)) for _ in range(rng.choice([0, 0, 1, 2, 3]))]
        constraint = rng.choice([None, None, 'y<=x', 'y<x', 'y>=x', 'y>x'])
        eff = region if region else (min(sx, ex), min(sy, ey), max(sx, ex), max(sy, ey))
        val = lattice_dfs((sx, sy), (ex, ey), steps, set(forb), eff, constraint)
        cs.append(mk('counting.lattice_paths', ((ex, ey),), val, 'brute depth-first enumeration', start=(sx, sy), steps=list(steps), forbidden=forb, region=region, constraint=constraint))
    return cs


def dyck_brute(n, h=None):
    t = 0
    for seq in itertools.product((1, -1), repeat=2 * n):
        s, ok = 0, True
        for x in seq:
            s += x
            if s < 0 or (h is not None and s > h):
                ok = False
                break
        t += ok and s == 0
    return t


def g_catalan(rng, n):
    cs = []
    for i in range(n):
        if i < 8:
            cs.append(mk('counting.catalan', (i,), dyck_brute(i), 'brute Dyck words'))
        else:
            m = rng.randint(8, 300)
            cs.append(mk('counting.catalan', (m,), int(sp_catalan(m)), 'sympy catalan'))
    return cs


def g_dyck(rng, n):
    cs = []
    for i in range(n):
        m = rng.randint(0, 8)
        h = rng.choice([None, rng.randint(0, 8)])
        cs.append(mk('counting.dyck_paths', (m, h), dyck_brute(m, h), 'brute sequences'))
    return cs


def ballot_dp(a, b, strict):
    memo = {}

    def go(i, j):
        if i < j or (strict and i == j and i + j > 0):
            return 0
        if i == 0 and j == 0:
            return 1
        if (i, j) in memo:
            return memo[(i, j)]
        r = (go(i - 1, j) if i > 0 else 0) + (go(i, j - 1) if j > 0 else 0)
        memo[(i, j)] = r
        return r
    return go(a, b)


def g_ballot(rng, n):
    cs = []
    for i in range(n):
        if i % 2 == 0:
            a, b = rng.randint(0, 7), rng.randint(0, 7)
            strict = rng.random() < 0.5
            if a + b == 0 and strict:
                strict = False
            t = 0
            for seq in itertools.product((1, -1), repeat=a + b):
                if seq.count(1) != a:
                    continue
                s, ok = 0, True
                for x in seq:
                    s += x
                    if s < 0 or (strict and s == 0):
                        ok = False
                        break
                t += ok
            cs.append(mk('counting.ballot', (a, b, strict), t, 'brute sequences'))
        else:
            a, b = rng.randint(1, 60), rng.randint(0, 60)
            strict = rng.random() < 0.5
            cs.append(mk('counting.ballot', (a, b, strict), ballot_dp(a, b, strict), 'brute DP over (a, b) positions'))
    return cs


def g_motzkin(rng, n):
    cs = []
    for i in range(n):
        if i < 12:
            t = 0
            for seq in itertools.product((1, 0, -1), repeat=i):
                s, ok = 0, True
                for x in seq:
                    s += x
                    if s < 0:
                        ok = False
                        break
                t += ok and s == 0
            cs.append(mk('counting.motzkin', (i,), t, 'brute sequences'))
        else:
            m = rng.randint(12, 200)
            val = sum(math.comb(m, 2 * k) * int(sp_catalan(k)) for k in range(m // 2 + 1))
            cs.append(mk('counting.motzkin', (m,), val, 'closed form sum C(n,2k) Catalan(k)'))
    return cs


def schroeder_brute(n):
    # paths of total width 2n from height 0 to 0, steps U, D (width 1) and H (width 2, level), never below 0
    def go(x, h):
        if x == 2 * n:
            return int(h == 0)
        t = go(x + 1, h + 1)
        if h > 0:
            t += go(x + 1, h - 1)
        if x + 2 <= 2 * n:
            t += go(x + 2, h)
        return t
    return go(0, 0)


def g_schroeder(rng, n):
    cs = []
    for i in range(n):
        if i < 7:
            cs.append(mk('counting.schroeder', (i,), schroeder_brute(i), 'brute step sequences'))
        elif i % 2 == 0:
            m = rng.randint(7, 200)
            cs.append(mk('counting.schroeder', (m,), sum(Fraction(math.comb(m, k) * math.comb(m + k, k), k + 1) for k in range(m + 1)) and int(sum(Fraction(math.comb(m, k) * math.comb(m + k, k), k + 1) for k in range(m + 1))), 'closed form sum C(n,k)C(n+k,k)/(k+1)'))
        else:
            m = rng.randint(1, 200)
            v = int(sum(Fraction(math.comb(m, k) * math.comb(m + k, k), k + 1) for k in range(m + 1))) // 2
            cs.append(mk('counting.schroeder', (m, True), v, 'closed form half of the large number'))
    return cs


def g_narayana(rng, n):
    cs = []
    for i in range(n):
        m, k = rng.randint(0, 9), rng.randint(0, 10)
        t = 0
        for seq in itertools.product((1, -1), repeat=2 * m):
            s, ok = 0, True
            for x in seq:
                s += x
                if s < 0:
                    ok = False
                    break
            if ok and s == 0:
                peaks = sum(1 for j in range(2 * m - 1) if seq[j] == 1 and seq[j + 1] == -1)
                t += peaks == k
        cs.append(mk('counting.narayana', (m, k), t, 'brute Dyck words counting peaks'))
    return cs


def g_strings(rng, n):
    cs = []
    for i in range(n):
        k = rng.randint(2, 3)
        alpha = 'abc'[:k]
        L = rng.randint(0, 8)
        pats = [''.join(rng.choice(alpha) for _ in range(rng.randint(1, 3))) for _ in range(rng.randint(0, 3))]
        cont = [''.join(rng.choice(alpha) for _ in range(rng.randint(1, 2))) for _ in range(rng.choice([0, 0, 1, 2]))]
        t = 0
        for tup in itertools.product(alpha, repeat=L):
            s = ''.join(tup)
            if all(p not in s for p in pats) and all(c in s for c in cont):
                t += 1
        if rng.random() < 0.5:
            cs.append(mk('counting.strings_with_patterns', (alpha, L), t, 'brute enumeration of all strings', avoid=pats, contain=cont))
        else:      # an integer alphabet size means the characters 0, 1, 2: translate the patterns the same way
            tr = str.maketrans('abc', '012')
            cs.append(mk('counting.strings_with_patterns', (k, L), t, 'brute enumeration of all strings', avoid=[p.translate(tr) for p in pats], contain=[p.translate(tr) for p in cont]))
    return cs


def g_transfer(rng, n):
    cs = []
    for i in range(n):
        m = rng.randint(1, 4)
        A = [[rng.randint(0, 2) for _ in range(m)] for _ in range(m)]
        L = rng.randint(0, 14)
        st = rng.choice([None, rng.randrange(m)])
        en = rng.choice([None, rng.randrange(m)])
        P = Matrix(A) ** L
        val = sum(P[a, b] for a in (range(m) if st is None else [st]) for b in (range(m) if en is None else [en]))
        cs.append(mk('counting.transfer_matrix_walks', (A, L, st, en), int(val), 'sympy matrix power'))
    return cs


def grid_brute(r, c, k, h, v):
    t = 0
    for col in itertools.product(range(k), repeat=r * c):
        ok = True
        for y in range(r):
            for x in range(c):
                a = col[y * c + x]
                if x + 1 < c and not h[a][col[y * c + x + 1]]:
                    ok = False
                if y + 1 < r and not v[a][col[(y + 1) * c + x]]:
                    ok = False
        t += ok
    return t


def g_grid(rng, n):
    cs = []
    for i in range(n):
        if i % 6 == 5:
            c = rng.randint(1, 30)
            k = rng.randint(2, 6)
            if rng.random() < 0.5:
                cs.append(mk('counting.grid_colorings', (1, c, k), k * (k - 1) ** (c - 1), 'closed form k(k-1)^(c-1)'))
            else:
                cs.append(mk('counting.grid_colorings', (2, c, k), k * (k - 1) * (k * k - 3 * k + 3) ** (c - 1), 'closed form ladder k(k-1)(k^2-3k+3)^(c-1)'))
            continue
        r, c, k = rng.randint(1, 3), rng.randint(1, 4), rng.randint(1, 3)
        while r * c > 9:
            c -= 1
        proper = [[a != b for b in range(k)] for a in range(k)]
        mode = rng.choice(['proper', 'proper', 'random', 'two'])
        if mode == 'proper':
            h = v = proper
            kw = {}
        elif mode == 'random':
            h = [[rng.random() < 0.6 for _ in range(k)] for _ in range(k)]
            v = h
            kw = {'allowed': [[int(x) for x in row] for row in h]}
        else:
            h = [[rng.random() < 0.6 for _ in range(k)] for _ in range(k)]
            v = [[rng.random() < 0.6 for _ in range(k)] for _ in range(k)]
            kw = {'allowed': [[int(x) for x in row] for row in h], 'allowed_v': [[int(x) for x in row] for row in v]}
        cs.append(mk('counting.grid_colorings', (r, c, k), grid_brute(r, c, k, h, v), 'brute enumeration of all colourings', **kw))
    return cs


def domino_brute(r, c, blocked):
    filled = [[(y, x) in blocked for x in range(c)] for y in range(r)]

    def go():
        for y in range(r):
            for x in range(c):
                if not filled[y][x]:
                    t = 0
                    filled[y][x] = True
                    if x + 1 < c and not filled[y][x + 1]:
                        filled[y][x + 1] = True
                        t += go()
                        filled[y][x + 1] = False
                    if y + 1 < r and not filled[y + 1][x]:
                        filled[y + 1][x] = True
                        t += go()
                        filled[y + 1][x] = False
                    filled[y][x] = False
                    return t
        return 1
    return go()


def g_domino(rng, n):
    cs = []
    for i in range(n):
        if i % 5 == 4:
            m = rng.randint(1, 40)
            cs.append(mk('counting.domino_tilings', (2, m), int(sympy.fibonacci(m + 1)), 'sympy Fibonacci (2 x m board)'))
            continue
        r, c = rng.randint(1, 5), rng.randint(1, 5)
        bl = [(rng.randrange(r), rng.randrange(c)) for _ in range(rng.choice([0, 0, 1, 2]))]
        cs.append(mk('counting.domino_tilings', (r, c), domino_brute(r, c, set(bl)), 'brute backtracking on the first empty cell', blocked=sorted(set(bl))))
    cs[0] = mk('counting.domino_tilings', (8, 8), 12988816, 'known value (chessboard tilings, Fischer 1961)')
    return cs


def g_digits(rng, n):
    cs = []
    for i in range(n):
        base = rng.choice([10, 10, 10, 2, 8])
        hi = rng.randint(1, 20000)
        lo = rng.randint(1, hi)
        kw = {}
        preds = []
        if rng.random() < 0.4:
            ds = [rng.randint(1, 20) for _ in range(rng.randint(1, 2))]
            kw['digit_sum'] = ds if len(ds) > 1 else ds[0]
            preds.append(lambda dg, ds=ds: sum(dg) in ds)
        if rng.random() < 0.3:
            fd = sorted(rng.sample(range(base), rng.randint(1, min(3, base - 1))))
            kw['forbidden_digits'] = fd
            preds.append(lambda dg, fd=fd: not any(d in fd for d in dg))
        if rng.random() < 0.2:
            kw['distinct'] = True
            preds.append(lambda dg: len(set(dg)) == len(dg))
        if rng.random() < 0.2:
            kw['no_adjacent_equal'] = True
            preds.append(lambda dg: all(dg[j] != dg[j + 1] for j in range(len(dg) - 1)))
        if rng.random() < 0.15:
            kw['nondecreasing'] = True
            preds.append(lambda dg: all(dg[j] <= dg[j + 1] for j in range(len(dg) - 1)))
        if rng.random() < 0.15:
            kw['increasing'] = True
            preds.append(lambda dg: all(dg[j] < dg[j + 1] for j in range(len(dg) - 1)))
        if rng.random() < 0.3:
            md = rng.randint(2, 13)
            rm = rng.randint(0, md - 1)
            kw['mod'], kw['rem'] = md, rm
            preds.append(lambda dg, md=md, rm=rm: int(''.join(map(str, dg)), base) % md == rm)
        t = 0
        for x in range(lo, hi + 1):
            dg, y = [], x
            while y:
                dg.append(y % base)
                y //= base
            dg.reverse()
            if all(p(dg) for p in preds):
                t += 1
        cs.append(mk('counting.count_digit_constraints', (lo, hi, base), t, 'brute loop over every integer', **kw))
    return cs


def g_spanning(rng, n):
    cs = []
    for i in range(n):
        r = i % 6
        if r == 5:
            kind = rng.choice(['K', 'Kmn', 'C', 'W'])
            if kind == 'K':
                m = rng.randint(2, 12)
                cs.append(mk('counting.spanning_trees', (m, [(u, v) for u in range(m) for v in range(u + 1, m)]), m ** (m - 2), 'closed form Cayley n^(n-2)'))
            elif kind == 'Kmn':
                a, b = rng.randint(1, 6), rng.randint(1, 6)
                es = [(u, a + v) for u in range(a) for v in range(b)]
                cs.append(mk('counting.spanning_trees', (a + b, es), a ** (b - 1) * b ** (a - 1), 'closed form m^(n-1) n^(m-1)'))
            elif kind == 'C':
                m = rng.randint(3, 30)
                cs.append(mk('counting.spanning_trees', (m, [(j, (j + 1) % m) if j + 1 < m else (0, j) for j in range(m)]), m, 'closed form n'))
            else:
                m = rng.randint(3, 20)         # wheel: hub 0 and a rim cycle of m vertices
                es = [(0, j) for j in range(1, m + 1)] + [(j, j + 1) for j in range(1, m)] + [(1, m)]
                cs.append(mk('counting.spanning_trees', (m + 1, es), int(sympy.lucas(2 * m)) - 2, 'sympy Lucas: wheel has L(2n)-2 spanning trees'))
            continue
        m = rng.randint(1, 7)
        es = rand_graph(rng, m, maxm=12)
        multi = rng.random() < 0.25 and es
        if multi:
            es = es + [rng.choice(es) for _ in range(rng.randint(1, 2))]
        t = sum(1 for sub in itertools.combinations(es, m - 1) if is_tree(m, list(sub)))
        cs.append(mk('counting.spanning_trees', (m, es), t, 'brute enumeration of (n-1)-edge subsets', multi=bool(multi)))
    return cs


def g_labelled(rng, n):
    cs = []
    trees = {}
    for m in range(1, 7):
        allv = [(u, v) for u in range(m) for v in range(u + 1, m)]
        trees[m] = [sub for sub in itertools.combinations(allv, m - 1) if is_tree(m, list(sub))]
    for i in range(n):
        m = rng.randint(1, 6)
        if i % 2 == 0:
            cs.append(mk('counting.labelled_trees', (m,), len(trees[m]), 'brute enumeration of all trees'))
        else:
            d = [rng.randint(0, 4) for _ in range(m)]
            if rng.random() < 0.7:
                # a degree sequence of a real tree
                t = rng.choice(trees[m])
                d = [0] * m
                for u, v in t:
                    d[u] += 1
                    d[v] += 1
            val = 0
            for t in trees[m]:
                dd = [0] * m
                for u, v in t:
                    dd[u] += 1
                    dd[v] += 1
                val += dd == d
            cs.append(mk('counting.labelled_trees', (m, d), val, 'brute enumeration of all trees by degree sequence'))
    return cs


def g_forests(rng, n):
    cs = []
    for i in range(n):
        m, k = rng.randint(1, 6), rng.randint(0, 7)
        allv = [(u, v) for u in range(m) for v in range(u + 1, m)]
        t = 0
        if 1 <= k <= m:
            for sub in itertools.combinations(allv, m - k):
                par = list(range(m))
                ok = True
                for u, v in sub:
                    a, b = uf_find(par, u), uf_find(par, v)
                    if a == b:
                        ok = False
                        break
                    par[a] = b
                if ok and len({uf_find(par, r) for r in range(k)}) == k:
                    t += 1
        cs.append(mk('counting.rooted_forests', (m, k), t, 'brute enumeration of forests with the roots in different trees'))
    return cs


def pm_brute(n, es):
    t = 0
    for sub in itertools.combinations(es, n // 2):
        vs = [x for e in sub for x in e]
        t += len(set(vs)) == n
    return t


def g_perfect_matchings(rng, n):
    cs = []
    for i in range(n):
        if i % 6 == 5:
            m = rng.randint(0, 40)
            d = 1
            for j in range(m - 1, 0, -2):
                d *= j
            cs.append(mk('counting.perfect_matchings', (m,), d if m % 2 == 0 else 0, 'closed form (n-1)!!'))
            continue
        m = rng.choice([0, 2, 4, 4, 6, 6, 8])
        es = rand_graph(rng, m, maxm=14)
        cs.append(mk('counting.perfect_matchings', (m, es), pm_brute(m, es), 'brute enumeration of (n/2)-edge subsets'))
    return cs


def g_matchings(rng, n):
    cs = []
    for i in range(n):
        m = rng.randint(0, 8)
        es = rand_graph(rng, m, maxm=13)
        size = rng.choice([None, rng.randint(0, 4)])
        t = 0
        for mask in range(1 << len(es)):
            sub = [es[j] for j in range(len(es)) if mask >> j & 1]
            vs = [x for e in sub for x in e]
            if len(set(vs)) == len(vs) and (size is None or len(sub) == size):
                t += 1
        cs.append(mk('counting.matchings_count', (m, es, size), t, 'brute enumeration of all edge subsets'))
    return cs


def g_perm_allowed(rng, n):
    cs = []
    for i in range(n):
        if i % 5 == 4:
            m = rng.randint(1, 12)
            A = [[int(i2 != j) for j in range(m)] for i2 in range(m)]
            cs.append(mk('counting.permutations_allowed', (A,), int(subfactorial(m)), 'sympy subfactorial (derangement matrix)'))
            continue
        m = rng.randint(0, 7)
        A = [[int(rng.random() < 0.55) for _ in range(m)] for _ in range(m)]
        t = sum(1 for p in itertools.permutations(range(m)) if all(A[j][p[j]] for j in range(m)))
        cs.append(mk('counting.permutations_allowed', (A,), t, 'brute permutations'))
    return cs


def g_independent(rng, n):
    cs = []
    for i in range(n):
        if i % 5 == 4:
            m = rng.randint(1, 45)
            kind = rng.choice(['path', 'cycle', 'complete'])
            if kind == 'path':
                cs.append(mk('counting.independent_sets', (m, [(j, j + 1) for j in range(m - 1)]), int(sympy.fibonacci(m + 2)), 'sympy Fibonacci F(n+2)'))
            elif kind == 'cycle' and m >= 3:
                cs.append(mk('counting.independent_sets', (m, [(j, j + 1) for j in range(m - 1)] + [(0, m - 1)]), int(sympy.lucas(m)), 'sympy Lucas L(n)'))
            else:
                m = min(m, 20)
                cs.append(mk('counting.independent_sets', (m, [(u, v) for u in range(m) for v in range(u + 1, m)]), m + 1, 'closed form n+1'))
            continue
        m = rng.randint(0, 11)
        es = rand_graph(rng, m)
        adj = [0] * m
        for u, v in es:
            adj[u] |= 1 << v
            adj[v] |= 1 << u
        counts = {}
        for mask in range(1 << m):
            if all(not (adj[j] & mask) for j in range(m) if mask >> j & 1):
                s = bin(mask).count('1')
                counts[s] = counts.get(s, 0) + 1
        mode = rng.choice(['total', 'size', 'poly'])
        if mode == 'total':
            cs.append(mk('counting.independent_sets', (m, es), sum(counts.values()), 'brute enumeration of all vertex subsets'))
        elif mode == 'size':
            s = rng.randint(0, 5)
            cs.append(mk('counting.independent_sets', (m, es, s), counts.get(s, 0), 'brute enumeration of all vertex subsets'))
        else:
            poly = [counts.get(s, 0) for s in range(max(counts) + 1)]
            cs.append(mk('counting.independent_sets', (m, es, 'poly'), poly, 'brute enumeration of all vertex subsets'))
    return cs


def proper_colorings_brute(m, es, k):
    return sum(1 for col in itertools.product(range(k), repeat=m) if all(col[u] != col[v] for u, v in es))


def g_chromatic(rng, n):
    cs = []
    for i in range(n):
        if i % 6 == 5:
            m = rng.randint(1, 12)
            x = sympy.symbols('x')
            poly = sympy.Poly(sympy.ff(x, m), x).all_coeffs()[::-1]
            cs.append(mk('counting.chromatic_polynomial', (m, [(u, v) for u in range(m) for v in range(u + 1, m)]), [int(c) for c in poly], 'sympy falling factorial (complete graph)'))
            continue
        m = rng.randint(1, 6)
        es = rand_graph(rng, m)
        vals = [proper_colorings_brute(m, es, k) for k in range(m + 1)]
        cs.append(mk('counting.chromatic_polynomial', (m, es), lagrange_coeffs(vals), 'brute colourings at k = 0..n, interpolated'))
    return cs


def g_colorings(rng, n):
    cs = []
    for i in range(n):
        if i % 6 == 5:
            m, k = rng.randint(3, 40), rng.randint(0, 5)
            tr = (sympy.ones(k, k) - sympy.eye(k)) ** m
            val = int(tr.trace()) if k else 0
            cs.append(mk('counting.colorings_count', (m, [(j, (j + 1) % m) if j + 1 < m else (0, j) for j in range(m)], k), val, 'sympy trace of (J-I)^n (closed walks)'))
            continue
        m, k = rng.randint(1, 7), rng.randint(0, 4)
        es = rand_graph(rng, m)
        cs.append(mk('counting.colorings_count', (m, es, k), proper_colorings_brute(m, es, k), 'brute enumeration of all colourings'))
    return cs


def g_automorphisms(rng, n):
    cs = []
    known = [(10, [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0), (0, 5), (1, 6), (2, 7), (3, 8), (4, 9), (5, 7), (7, 9), (9, 6), (6, 8), (8, 5)], 120, 'known: Petersen graph has 120 automorphisms')]
    for i in range(n):
        if i < len(known):
            m, es, v, nm = known[i]
            cs.append(mk('counting.graph_automorphisms', (m, es), v, nm))
            continue
        m = rng.randint(1, 7)
        es = rand_graph(rng, m)
        eset = set(es)
        t = 0
        for p in itertools.permutations(range(m)):
            if all((min(p[u], p[v]), max(p[u], p[v])) in eset for u, v in es):
                t += 1
        cs.append(mk('counting.graph_automorphisms', (m, es), t, 'brute enumeration of all permutations'))
    return cs


def g_union(rng, n):
    cs = []
    for i in range(n):
        U = rng.randint(1, 30)
        k = rng.randint(1, 5)
        sets = [set(x for x in range(U) if rng.random() < 0.4) for _ in range(k)]
        d = {}
        for r in range(1, k + 1):
            for sub in itertools.combinations(range(k), r):
                d[sub] = len(set.intersection(*[sets[j] for j in sub]))
        cs.append(mk('counting.union_size', (d,), len(set().union(*sets)), 'brute set union'))
    return cs


def g_divisible(rng, n):
    cs = []
    for i in range(n):
        N = rng.randint(1, 3000)
        lo = rng.choice([1, 1, rng.randint(1, N)])
        ds = [rng.randint(1, 20) for _ in range(rng.randint(1, 5))]
        ex = rng.choice([None, 0, 1, 2, len(ds), len(ds) + 1])
        cnt = 0
        for x in range(lo, N + 1):
            c = sum(1 for d in ds if x % d == 0)
            cnt += (c >= 1) if ex is None else (c == ex)
        cs.append(mk('counting.count_divisible', (N, ds, ex, lo), cnt, 'brute loop'))
    return cs


def g_coprime(rng, n):
    cs = []
    for i in range(n):
        if i % 4 == 3:
            m = rng.randint(1, 10 ** 9)
            cs.append(mk('counting.count_coprime', (m, m), int(totient(m)), 'sympy totient (phi(m) = integers in [1,m] coprime to m)'))
            continue
        m = rng.randint(1, 600)
        N = rng.randint(1, 2500)
        lo = rng.choice([1, rng.randint(1, N)])
        cs.append(mk('counting.count_coprime', (N, m, lo), sum(1 for x in range(lo, N + 1) if math.gcd(x, m) == 1), 'brute gcd loop'))
    return cs


# Predicate templates: (text, arity or None, python twin). The twins are plain Python written here.
PRED3 = [
    ('x0 + x1 == 7', lambda a, b, c: a + b == 7),
    ('x0 * x1 % 5 == 0', lambda a, b, c: a * b % 5 == 0),
    ('x0 < x1 and x1 < x2', lambda a, b, c: a < b < c),
    ('gcd(x0, x1) == 1', lambda a, b, c: math.gcd(a, b) == 1),
    ('abs(x0 - x1) <= 2 or x2 == 1', lambda a, b, c: abs(a - b) <= 2 or c == 1),
    ('max(x0, x1, x2) == 5', lambda a, b, c: max(a, b, c) == 5),
    ('is_prime(x0 + x1 + x2)', lambda a, b, c: C._is_prime(a + b + c) if False else (lambda s: s > 1 and all(s % d for d in range(2, int(s ** .5) + 1)))(a + b + c)),
    ('len(set(v)) == 3', lambda a, b, c: len({a, b, c}) == 3),
    ('sum(v) % 3 == 0', lambda a, b, c: (a + b + c) % 3 == 0),
    ('x0 ** 2 + x1 ** 2 == x2 ** 2', lambda a, b, c: a * a + b * b == c * c),
    ('x0 + x1 > x2 and x1 + x2 > x0 and x0 + x2 > x1', lambda a, b, c: a + b > c and b + c > a and a + c > b),
    ('not (x0 == x1) and (x2 // 2) % 2 == 1', lambda a, b, c: a != b and (c // 2) % 2 == 1),
    ('(x0 & x1) == 0', lambda a, b, c: a & b == 0),
    ('(x0 ^ x1) == x2', lambda a, b, c: a ^ b == c),
    ('x0 if x1 > 2 else x2 == 3', lambda a, b, c: bool(a) if b > 2 else c == 3),
    ('sorted(v)[1] == 3', lambda a, b, c: sorted([a, b, c])[1] == 3),
    ('x0 in (1, 3, 5) and x1 not in (2, 4)', lambda a, b, c: a in (1, 3, 5) and b not in (2, 4)),
    ('dsum(x0 * x1) == x2', lambda a, b, c: sum(map(int, str(a * b))) == c),
    ('is_square(x0 + x1 * x2)', lambda a, b, c: math.isqrt(a + b * c) ** 2 == a + b * c),
    ('cnt(v, 2) == 1', lambda a, b, c: [a, b, c].count(2) == 1),
]


def space_cases(rng):
    """(domains, space, r, arity, pyiter): a random finite space and the plain-Python list of tuples in it."""
    sp = rng.choice(['product', 'product', 'permutations', 'combinations', 'multisets'])
    if sp == 'product':
        ds = [list(range(rng.randint(0, 2), rng.randint(3, 6))) for _ in range(3)]
        dom = [f'{d[0]}..{d[-1]}' if rng.random() < 0.5 else d for d in ds]
        return dom, sp, None, 3, list(itertools.product(*ds))
    items = list(range(1, rng.randint(4, 7)))
    dom = f'{items[0]}..{items[-1]}'
    if sp == 'permutations':
        return dom, sp, 3, 3, list(itertools.permutations(items, 3))
    if sp == 'combinations':
        return dom, sp, 3, 3, list(itertools.combinations(items, 3))
    return dom, sp, 3, 3, list(itertools.combinations_with_replacement(items, 3))


def g_brute(rng, n):
    cs = []
    for i in range(n):
        if i % 8 == 7:
            items = list(range(1, rng.randint(3, 10)))
            tgt = rng.randint(3, 15)
            val = sum(1 for m in range(1 << len(items)) if sum(items[j] for j in range(len(items)) if m >> j & 1) == tgt)
            cs.append(mk('counting.brute_count', (f'{items[0]}..{items[-1]}', 'sum(v) == %d' % tgt), val, 'plain Python subset loop', space='subsets'))
            continue
        dom, sp, r, ar, tuples = space_cases(rng)
        text, twin = rng.choice(PRED3)
        val = sum(1 for t in tuples if twin(*t))
        cs.append(mk('counting.brute_count', (dom, text), val, 'plain Python loop with an independent lambda', space=sp, r=r))
    return cs


def g_hist(rng, n):
    cs = []
    for i in range(n):
        dom, sp, r, ar, tuples = space_cases(rng)
        picks = rng.sample(PRED3, rng.randint(1, 3))
        hist = {j: 0 for j in range(len(picks) + 1)}
        for t in tuples:
            hist[sum(1 for _, tw in picks if tw(*t))] += 1
        cs.append(mk('counting.predicate_histogram', (dom, [p for p, _ in picks]), hist, 'plain Python loop', space=sp, r=r))
    return cs


def g_prob(rng, n):
    cs = []
    for i in range(n):
        if i % 6 == 5:
            a, b = rng.randint(1, 6), rng.randint(1, 6)
            tw = rng.choice([(f'x0 + x1 == {a + b}', lambda x, y: x + y == a + b), ('x0 == x1', lambda x, y: x == y), ('x0 * x1 >= 12', lambda x, y: x * y >= 12)])
            val = Fraction(sum(1 for x in range(1, 7) for y in range(1, 7) if tw[1](x, y)), 36)
            cs.append(mk('counting.prob_uniform', (['1..6', '1..6'], tw[0]), val, 'plain Python loop over two dice'))
            continue
        dom, sp, r, ar, tuples = space_cases(rng)
        (e, te), (g, tg) = rng.sample(PRED3, 2)
        den = [t for t in tuples if tg(*t)]
        if not den:
            cs.append(mk('counting.prob_uniform', (dom, e), Fraction(sum(1 for t in tuples if te(*t)), len(tuples)), 'plain Python loop', space=sp, r=r))
            continue
        if rng.random() < 0.5:
            val = Fraction(sum(1 for t in den if te(*t)), len(den))
            cs.append(mk('counting.prob_uniform', (dom, e), val, 'plain Python loop (conditional)', given=g, space=sp, r=r))
        else:
            cs.append(mk('counting.prob_uniform', (dom, e), Fraction(sum(1 for t in tuples if te(*t)), len(tuples)), 'plain Python loop', space=sp, r=r))
    return cs


EXPRS = [
    ('x0 + x1 + x2', lambda a, b, c: a + b + c),
    ('max(x0, x1, x2)', lambda a, b, c: max(a, b, c)),
    ('x0 * x1 - x2', lambda a, b, c: a * b - c),
    ('(x0 + x1) % 4', lambda a, b, c: (a + b) % 4),
    ('len(set(v))', lambda a, b, c: len({a, b, c})),
    ('abs(x0 - x2)', lambda a, b, c: abs(a - c)),
    ('gcd(x0, gcd(x1, x2))', lambda a, b, c: math.gcd(a, math.gcd(b, c))),
    ('min(x0, x1) * x2', lambda a, b, c: min(a, b) * c),
]


def g_expect(rng, n):
    cs = []
    for i in range(n):
        dom, sp, r, ar, tuples = space_cases(rng)
        text, tw = rng.choice(EXPRS)
        vals = [tw(*t) for t in tuples]
        N = len(vals)
        mean = Fraction(sum(vals), N)
        var = sum((Fraction(v) - mean) ** 2 for v in vals) / N
        cs.append(mk('counting.expectation_uniform', (dom, text), {'mean': mean, 'variance': var, 'count': N}, 'plain Python loop, variance as mean of squared deviations', space=sp, r=r))
    return cs


def g_dice_dist(rng, n):
    cs = []
    for i in range(n):
        if i % 4 == 3:
            m, s = rng.randint(1, 12), rng.randint(2, 12)
            ways = {}
            for t in range(m, m * s + 1):
                ways[t] = sum((-1) ** j * math.comb(m, j) * math.comb(t - j * s - 1, m - 1) for j in range(m + 1) if t - j * s - 1 >= m - 1)
            cs.append(mk('counting.dice_sum_distribution', (m, s), ways, 'closed form inclusion-exclusion'))
            continue
        m, s = rng.randint(0, 4), rng.randint(1, 6)
        faces = None
        if rng.random() < 0.3:
            faces = [rng.randint(-2, 4) for _ in range(rng.randint(1, 5))]
        fc = list(range(1, s + 1)) if faces is None else faces
        ways = {}
        for t in itertools.product(fc, repeat=m):
            ways[sum(t)] = ways.get(sum(t), 0) + 1
        cs.append(mk('counting.dice_sum_distribution', (m, s), dict(sorted(ways.items())), 'brute enumeration', faces=faces))
    return cs


def g_dice_prob(rng, n):
    cs = []
    for i in range(n):
        m, s = rng.randint(1, 5), rng.randint(2, 8)
        if s ** m > 20000:
            m = 3
        lo = rng.randint(m, m * s)
        hi = rng.choice([None, rng.randint(lo, m * s + 2)])
        tot = good = 0
        for t in itertools.product(range(1, s + 1), repeat=m):
            tot += 1
            good += (lo <= sum(t) <= (lo if hi is None else hi))
        cs.append(mk('counting.dice_sum_probability', (m, s, lo, hi), Fraction(good, tot), 'brute enumeration'))
    return cs


def g_dice_ext(rng, n):
    cs = []
    for i in range(n):
        m, s = rng.randint(1, 4), rng.randint(1, 6)
        stat = rng.choice(['max', 'min'])
        out = {v: 0 for v in range(1, s + 1)}
        for t in itertools.product(range(1, s + 1), repeat=m):
            out[max(t) if stat == 'max' else min(t)] += 1
        cs.append(mk('counting.dice_extreme_distribution', (m, s, stat), out, 'brute enumeration'))
    return cs


def g_dist_stats(rng, n):
    cs = []
    for i in range(n):
        pairs = [(rng.randint(-5, 10), rng.randint(0, 6)) for _ in range(rng.randint(1, 8))]
        if sum(w for _, w in pairs) == 0:
            pairs[0] = (pairs[0][0], 1)
        tw = Fraction(sum(w for _, w in pairs))
        mean = sum(Fraction(x * w) for x, w in pairs) / tw
        var = sum(w * (x - mean) ** 2 for x, w in pairs) / tw
        cs.append(mk('counting.dist_stats', (pairs,), {'mean': mean, 'variance': var, 'total_weight': tw}, 'closed form variance as weighted mean squared deviation'))
    return cs


def g_bayes(rng, n):
    cs = []
    for i in range(n):
        k = rng.randint(1, 5)
        a = [rng.randint(1, 6) for _ in range(k)]
        D = sum(a)
        E = rng.randint(2, 9)
        b = [rng.randint(0, E) for _ in range(k)]
        if all(x == 0 for x in b):
            b[0] = 1
        joint = [x * y for x, y in zip(a, b)]
        tot = sum(joint)
        val = [Fraction(j, tot) for j in joint]
        cs.append(mk('counting.bayes_posterior', ([Fraction(x, D) for x in a], [Fraction(y, E) for y in b]), val, 'brute counts in a sample space of D*E equally likely points'))
    return cs


def g_binom(rng, n):
    cs = []
    for i in range(n):
        if i % 3 == 2:
            m = rng.randint(1, 60)
            p = Fraction(rng.randint(0, 12), 12)
            lo = rng.randint(0, m)
            hi = rng.choice([None, rng.randint(lo, m)])
            # probability DP: distribution of the number of successes after each trial
            dist = [Fraction(1)]
            for _ in range(m):
                nd = [Fraction(0)] * (len(dist) + 1)
                for j, x in enumerate(dist):
                    nd[j] += x * (1 - p)
                    nd[j + 1] += x * p
                dist = nd
            hh = lo if hi is None else hi
            cs.append(mk('counting.binomial_probability', (m, p, lo, hi), sum(dist[lo:hh + 1]), 'probability DP over trials'))
            continue
        m = rng.randint(0, 9)
        p = Fraction(rng.randint(0, 6), 6)
        lo = rng.randint(0, m)
        hi = rng.choice([None, rng.randint(lo, m)])
        hh = lo if hi is None else hi
        tot = Fraction(0)
        for seq in itertools.product((0, 1), repeat=m):
            if lo <= sum(seq) <= hh:
                tot += p ** sum(seq) * (1 - p) ** (m - sum(seq))
        cs.append(mk('counting.binomial_probability', (m, p, lo, hi), tot, 'brute enumeration of all success/failure sequences'))
    return cs


def g_hyper(rng, n):
    cs = []
    for i in range(n):
        if i % 3 == 2:
            N = rng.randint(5, 60)
            K = rng.randint(0, N)
            m = rng.randint(0, N)
            k = rng.randint(0, m)
            # draw one at a time: dist[j] = P(j marked among the draws so far)
            dist = {0: Fraction(1)}
            M, Kk = N, K
            for t in range(m):
                nd = {}
                for j, x in dist.items():
                    marked_left = K - j
                    left = N - t
                    if marked_left > 0:
                        nd[j + 1] = nd.get(j + 1, 0) + x * Fraction(marked_left, left)
                    if left - marked_left > 0:
                        nd[j] = nd.get(j, 0) + x * Fraction(left - marked_left, left)
                dist = nd
            cs.append(mk('counting.hypergeometric_probability', (N, K, m, k), dist.get(k, Fraction(0)), 'sequential-draw probability DP'))
            continue
        N = rng.randint(1, 12)
        K = rng.randint(0, N)
        m = rng.randint(0, min(N, 6))
        lo = rng.randint(0, m)
        hi = rng.choice([None, rng.randint(lo, m)])
        hh = lo if hi is None else hi
        good = tot = 0
        for comb_ in itertools.combinations(range(N), m):
            tot += 1
            good += lo <= sum(1 for x in comb_ if x < K) <= hh
        cs.append(mk('counting.hypergeometric_probability', (N, K, m, lo, hi), Fraction(good, tot), 'brute enumeration of all draws'))
    return cs


def g_mhyper(rng, n):
    cs = []
    for i in range(n):
        kinds = rng.randint(1, 3)
        counts = [rng.randint(0, 4) for _ in range(kinds)]
        drawn = [rng.randint(0, c + 1) for c in counts]
        N, m = sum(counts), sum(drawn)
        if m > N:
            drawn = [0] * kinds
            m = 0
        labels = [k for k, c in enumerate(counts) for _ in range(c)]
        good = tot = 0
        for comb_ in itertools.combinations(range(N), m):
            tot += 1
            got = [0] * kinds
            for x in comb_:
                got[labels[x]] += 1
            good += got == drawn
        cs.append(mk('counting.multivariate_hypergeometric', (counts, drawn), Fraction(good, tot), 'brute enumeration of all draws'))
    return cs


def g_geom(rng, n):
    cs = []
    cache = {}
    t = sympy.symbols('t', integer=True, positive=True)
    for i in range(n):
        p = Fraction(rng.randint(1, 9), rng.randint(9, 12))
        if p > 1:
            p = Fraction(1, 2)
        m = rng.randint(0, 25)
        q = 1 - p
        pmf = p * q ** (m - 1) if m >= 1 else Fraction(0)
        cdf = sum(p * q ** (j - 1) for j in range(1, m + 1))
        if p == 1:
            cache[p] = (Fraction(1), Fraction(1))      # sympy mishandles 0**0 in the series: T = 1 for sure
        if p not in cache:
            P = Rational(p.numerator, p.denominator)
            cache[p] = (sp_to_fr(sympy.summation(t * P * (1 - P) ** (t - 1), (t, 1, sympy.oo))), sp_to_fr(sympy.summation(t ** 2 * P * (1 - P) ** (t - 1), (t, 1, sympy.oo))))
        mean, m2 = cache[p]
        cs.append(mk('counting.geometric_distribution', (p, m), {'pmf': pmf, 'cdf': cdf, 'tail': 1 - cdf, 'mean': mean, 'variance': m2 - mean ** 2}, 'series sums for pmf/cdf, sympy infinite sums for mean/variance'))
    return cs


def coupon_chain(ps):
    """Mean and variance of the collection time by the Markov chain on collected subsets (solved with sympy)."""
    n = len(ps)
    states = list(range(1 << n))
    full = (1 << n) - 1
    tr = [s for s in states if s != full]
    idx = {s: i for i, s in enumerate(tr)}
    Q = sympy.zeros(len(tr), len(tr))
    for s in tr:
        for c in range(n):
            t = s | (1 << c)
            if t != full:
                Q[idx[s], idx[t]] += Rational(ps[c].numerator, ps[c].denominator)
    I = sympy.eye(len(tr))
    N = (I - Q).inv()
    one = sympy.ones(len(tr), 1)
    tvec = N * one
    var = (2 * N - I) * tvec - Matrix([x ** 2 for x in tvec])
    return sp_to_fr(tvec[idx[0]]), sp_to_fr(var[idx[0]])


def g_coupon(rng, n):
    cs = []
    for i in range(n):
        r = i % 3
        if r == 0:
            k = rng.randint(1, 4)
            w = [rng.randint(1, 5) for _ in range(k)]
            ps = [Fraction(x, sum(w)) for x in w]
            mean, var = coupon_chain(ps)
            cs.append(mk('counting.coupon_collector', (), {'mean': mean, 'variance': var}, 'sympy: Markov chain on collected subsets, (2N-I)t - t^2', probs=ps))
        elif r == 1:
            m = rng.randint(1, 40)
            mean = m * harmonic(m)
            var = m * m * sum(Rational(1, j * j) for j in range(1, m + 1)) - m * harmonic(m)
            cs.append(mk('counting.coupon_collector', (m,), {'mean': sp_to_fr(mean), 'variance': sp_to_fr(var)}, 'closed form n H_n and n^2 H_n^(2) - n H_n'))
        else:
            m = rng.randint(1, 30)
            j = rng.randint(1, m)
            # time to see j distinct of m: chain on the number collected
            mean = sum(Fraction(m, m - a) for a in range(j))
            var = sum(Fraction(m * a, (m - a) ** 2) for a in range(j))
            cs.append(mk('counting.coupon_collector', (m, j), {'mean': mean, 'variance': var}, 'closed form sum of geometric means and variances (p = (n-a)/n)'))
    return cs


def g_coupon_prob(rng, n):
    cs = []
    for i in range(n):
        if i % 2 == 0:
            m, d = rng.randint(1, 4), rng.randint(0, 7)
            good = sum(1 for seq in itertools.product(range(m), repeat=d) if len(set(seq)) == m)
            cs.append(mk('counting.coupon_collector_probability', (m, d), Fraction(good, m ** d), 'brute enumeration of all draw sequences'))
        else:
            m, d = rng.randint(1, 15), rng.randint(1, 60)
            val = Fraction(math.factorial(m) * int(stirling(d, m, kind=2)), m ** d) if d >= m else Fraction(0)
            cs.append(mk('counting.coupon_collector_probability', (m, d), val, 'sympy n! S(t,n) / n^t'))
    return cs


def g_gambler(rng, n):
    cs = []
    for i in range(n):
        N = rng.randint(2, 12)
        j = rng.randint(0, N)
        p = Fraction(rng.randint(1, 5), 6) if rng.random() < 0.7 else Fraction(1, 2)
        P = Rational(p.numerator, p.denominator)
        Q = 1 - P
        # linear systems h_i = P h_{i+1} + Q h_{i-1}, h_0 = 0, h_N = 1 ;  d_i = 1 + P d_{i+1} + Q d_{i-1}, d_0 = d_N = 0
        A = sympy.zeros(N + 1, N + 1)
        b1 = sympy.zeros(N + 1, 1)
        b2 = sympy.zeros(N + 1, 1)
        A[0, 0] = 1
        A[N, N] = 1
        b1[N] = 1
        for s in range(1, N):
            A[s, s], A[s, s + 1], A[s, s - 1] = 1, -P, -Q
            b2[s] = 1
        win = sp_to_fr(A.LUsolve(b1)[j])
        dur = sp_to_fr(A.LUsolve(b2)[j])
        cs.append(mk('counting.gamblers_ruin', (j, N, p), {'win': win, 'ruin': 1 - win, 'expected_duration': dur}, 'sympy: the two linear systems solved exactly'))
    return cs


def g_birthday(rng, n):
    cs = []
    for i in range(n):
        if i % 2 == 0:
            d, m = rng.randint(1, 6), rng.randint(0, 5)
            good = sum(1 for t in itertools.product(range(d), repeat=m) if len(set(t)) == m)
            cs.append(mk('counting.birthday_probability', (m, d), {'distinct': Fraction(good, d ** m), 'collision': 1 - Fraction(good, d ** m)}, 'brute enumeration'))
        else:
            d, m = rng.randint(2, 400), rng.randint(0, 60)
            dist = Fraction(int(sympy.ff(d, m)), d ** m) if m <= d else Fraction(0)
            cs.append(mk('counting.birthday_probability', (m, d), {'distinct': dist, 'collision': 1 - dist}, 'sympy falling factorial / d^n'))
    return cs


def rand_chain(rng, n, absorbing=0, den=6):
    P = []
    ab = set(rng.sample(range(n), absorbing)) if absorbing else set()
    for i in range(n):
        if i in ab:
            P.append([Fraction(int(i == j)) for j in range(n)])
            continue
        w = [rng.randint(0, 3) if rng.random() < 0.7 else 0 for _ in range(n)]
        if sum(w) == 0:
            w[rng.randrange(n)] = 1
        P.append([Fraction(x, sum(w)) for x in w])
    return P, ab


def to_sp(P):
    return Matrix([[Rational(x.numerator, x.denominator) for x in row] for row in P])


def g_absorption(rng, n):
    cs = []
    for i in range(n):
        m = rng.randint(2, 6)
        P, _ab = rand_chain(rng, m, absorbing=rng.randint(1, min(2, m - 1)))
        ab = {s_ for s_ in range(m) if P[s_][s_] == 1}      # a row that happens to be the identity row is absorbing too
        abl = sorted(ab)
        tr = [s for s in range(m) if s not in ab]
        R = to_sp(P)
        # refusal when a transient state cannot reach an absorbing one: decide by graph search written here
        reach = {s: {s} for s in range(m)}
        for s in range(m):
            stack = [s]
            while stack:
                x = stack.pop()
                for y in range(m):
                    if P[x][y] != 0 and y not in reach[s]:
                        reach[s].add(y)
                        stack.append(y)
        if any(not (reach[s] & ab) for s in tr):
            cs.append(mk('counting.markov_absorption', (P,), Refuse('transient state never absorbed'), 'expected refusal (a closed non-absorbing class)'))
            continue
        Qm = R.extract(tr, tr)
        Rm = R.extract(tr, abl)
        Nm = (sympy.eye(len(tr)) - Qm).inv()
        B = Nm * Rm
        t = Nm * sympy.ones(len(tr), 1)
        val = {'absorbing': abl, 'transient': tr,
               'probabilities': {s: {a: sp_to_fr(B[ii, jj]) for jj, a in enumerate(abl)} for ii, s in enumerate(tr)},
               'expected_steps': {s: sp_to_fr(t[ii]) for ii, s in enumerate(tr)}}
        cs.append(mk('counting.markov_absorption', (P,), val, 'sympy fundamental matrix (I-Q)^-1 (certificate: B = QB + R and t = 1 + Qt hold for the solution)'))
    return cs


def g_hitting(rng, n):
    cs = []
    for i in range(n):
        m = rng.randint(2, 6)
        P, _ = rand_chain(rng, m)
        tg = sorted(rng.sample(range(m), rng.randint(1, min(2, m - 1))))
        rest = [s for s in range(m) if s not in tg]
        # which states reach a target (search written here)
        def reach_set(s):
            seen, stack = {s}, [s]
            while stack:
                x = stack.pop()
                if x in tg:
                    continue
                for y in range(m):
                    if P[x][y] != 0 and y not in seen:
                        seen.add(y)
                        stack.append(y)
            return seen
        inf = set()
        for s in rest:
            rs = reach_set(s)
            if any(not (reach_set(u) & set(tg)) for u in rs if u not in tg):
                inf.add(s)
        fin = [s for s in rest if s not in inf]
        val = {t: Fraction(0) for t in tg}
        for s in inf:
            val[s] = 'inf'
        if fin:
            R = to_sp(P)
            A = sympy.eye(len(fin)) - R.extract(fin, fin)
            h = A.LUsolve(sympy.ones(len(fin), 1))
            for ii, s in enumerate(fin):
                val[s] = sp_to_fr(h[ii])
        cs.append(mk('counting.markov_hitting_time', (P, tg), dict(sorted(val.items())), 'sympy linear solve; infinite states by a reachability search written here'))
    return cs


def g_stationary(rng, n):
    cs = []
    for i in range(n):
        m = rng.randint(2, 6)
        P = [[Fraction(rng.randint(1, 4)) for _ in range(m)] for _ in range(m)]
        P = [[x / sum(row) for x in row] for row in P]
        R = to_sp(P)
        A = (R.T - sympy.eye(m))
        A[m - 1, :] = sympy.ones(1, m)
        b = sympy.zeros(m, 1)
        b[m - 1] = 1
        pi = A.LUsolve(b)
        val = [sp_to_fr(x) for x in pi]
        # certificate: pi P = pi, sum = 1, pi > 0
        assert sum(val) == 1 and all(sum(val[a] * P[a][c] for a in range(m)) == val[c] for c in range(m))
        cs.append(mk('counting.markov_stationary', (P,), val, 'sympy linear solve (certificate: pi P = pi checked here)'))
    return cs


def g_mdist(rng, n):
    cs = []
    for i in range(n):
        m = rng.randint(2, 5)
        P, _ = rand_chain(rng, m)
        steps = rng.randint(0, 12)
        s = rng.randrange(m)
        v = sympy.zeros(1, m)
        v[s] = 1
        out = v * to_sp(P) ** steps
        cs.append(mk('counting.markov_distribution', (P, s, steps), [sp_to_fr(x) for x in out], 'sympy matrix power'))
    return cs


def li_conway(pats, k):
    """Win probabilities and expected stopping time by the Li-Conway martingale equations (alpha-ary alphabet, equal probabilities)."""
    m = len(pats)

    def corr(a, b):   # sum of k^i over i with suffix of a of length i == prefix of b of length i
        return sum(k ** i for i in range(1, min(len(a), len(b)) + 1) if a[-i:] == b[:i])
    ps = sympy.symbols(f'p0:{m}')
    E = sympy.symbols('E')
    eqs = [sympy.Eq(E, sum(ps[j] * corr(pats[j], pats[i]) for j in range(m))) for i in range(m)] + [sympy.Eq(sum(ps), 1)]
    sol = sympy.solve(eqs, list(ps) + [E], dict=True)[0]
    return [sp_to_fr(sol[p]) for p in ps], sp_to_fr(sol[E])


def g_pattern_race(rng, n):
    cs = []
    for i in range(n):
        k = rng.choice([2, 2, 3])
        alpha = 'HT' if k == 2 else 'abc'
        m = rng.randint(1, 3)
        pats = []
        for _ in range(m):
            pats.append(''.join(rng.choice(alpha) for _ in range(rng.randint(1, 4))))
        pats = list(dict.fromkeys(pats))
        for _try in range(6):          # most cases should be valid sets: no pattern inside another
            if not any(p != q and p in q for p in pats for q in pats):
                break
            pats = list(dict.fromkeys(''.join(rng.choice(alpha) for _ in range(rng.randint(1, 4))) for _ in range(m)))
        if any(p != q and p in q for p in pats for q in pats):
            cs.append(mk('counting.pattern_race', (pats, alpha), Refuse('a pattern contains another'), 'expected refusal'))
            continue
        pr, e = li_conway(pats, k)
        cs.append(mk('counting.pattern_race', (pats, alpha), {'probabilities': pr, 'expected_length': e}, 'sympy: Li-Conway martingale equations (no automaton)'))
    return cs


def kmp_wait(pat, alpha):
    k = len(alpha)
    L = len(pat)
    # states 0..L-1 = longest matched prefix; reaching L stops
    def step(s, ch):
        t = pat[:s] + ch
        while t and not pat.startswith(t):
            t = t[1:]
        return len(t)
    E = sympy.symbols(f'e0:{L}')
    eqs = []
    for s in range(L):
        rhs = 1
        for ch in alpha:
            t = step(s, ch)
            if t < L:
                rhs += Rational(1, k) * E[t]
        eqs.append(sympy.Eq(E[s], rhs))
    sol = sympy.solve(eqs, list(E), dict=True)[0]
    return sp_to_fr(sol[E[0]])


def g_wait(rng, n):
    cs = []
    for i in range(n):
        k = rng.choice([2, 2, 3])
        alpha = 'HT' if k == 2 else 'abc'
        pat = ''.join(rng.choice(alpha) for _ in range(rng.randint(1, 5)))
        cs.append(mk('counting.expected_wait_pattern', (pat, alpha), kmp_wait(pat, alpha), 'sympy: KMP automaton chain solved exactly'))
    cs[0] = mk('counting.expected_wait_pattern', ('HHH', 'HT'), 14, 'known value (three heads in a row: 14 flips)')
    return cs


def walk_brute(moves, probs, n, start, stops, lo=None, hi=None):
    dim = len(start)
    out = {}
    for seq in itertools.product(range(len(moves)), repeat=n):
        pos = start
        w = Fraction(1)
        for j in seq:
            w *= probs[j]
            stop = pos in stops or (dim == 1 and ((lo is not None and pos[0] <= lo) or (hi is not None and pos[0] >= hi)))
            if not stop:
                pos = tuple(a + b for a, b in zip(pos, moves[j]))
        out[pos] = out.get(pos, 0) + w
    return {k: v for k, v in sorted(out.items()) if v != 0}


def g_walk(rng, n):
    cs = []
    for i in range(n):
        dim = rng.choice([1, 1, 2])
        pool = [(1,), (-1,), (2,), (0,)] if dim == 1 else [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1)]
        mv = rng.sample(pool, rng.randint(2, min(4, len(pool))))
        wts = [rng.randint(1, 4) for _ in mv]
        probs = [Fraction(w, sum(wts)) for w in wts]
        L = rng.randint(0, 6)
        stops, lo, hi = set(), None, None
        kw = {}
        if dim == 1 and rng.random() < 0.5:
            lo, hi = -rng.randint(1, 3), rng.randint(1, 3)
            kw = {'lo': lo, 'hi': hi}
        elif rng.random() < 0.3:
            stop = tuple(rng.randint(-1, 2) for _ in range(dim))
            stops = {stop}
            kw = {'absorbing': [list(stop) if dim > 1 else stop[0]]}
        steps = [((m[0] if dim == 1 else m), w) for m, w in zip(mv, wts)]
        val = walk_brute(mv, probs, L, tuple([0] * dim), stops, lo, hi)
        val = {(k[0] if dim == 1 else k): v for k, v in val.items()}
        cs.append(mk('counting.walk_distribution', (steps, L), val, 'brute enumeration of all move sequences', **kw))
    return cs


def circle_cross(N, a, b, c, d):
    """Do chords ab and cd of a regular N-gon cross? Segment intersection on the actual coordinates."""
    if len({a, b, c, d}) < 4:
        return False
    pt = lambda j: (math.cos(2 * math.pi * j / N), math.sin(2 * math.pi * j / N))

    def orient(p, q, r):
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])
    A, B, Cc, D = pt(a), pt(b), pt(c), pt(d)
    return orient(A, B, Cc) * orient(A, B, D) < 0 and orient(Cc, D, A) * orient(Cc, D, B) < 0


def g_chords_cross(rng, n):
    cs = []
    for i in range(n):
        N = rng.randint(4, 14)
        a, b, c, d = (rng.randrange(N) for _ in range(4))
        cs.append(mk('counting.chords_cross', (a, b, c, d), circle_cross(N, a, b, c, d), f'coordinates of a regular {N}-gon and a segment intersection test'))
    return cs


def g_crossing_pairs(rng, n):
    cs = []
    for i in range(n):
        N = rng.randint(4, 12)
        ch = []
        for _ in range(rng.randint(0, 10)):
            a, b = rng.randrange(N), rng.randrange(N)
            if a != b:
                ch.append((a, b))
        t = sum(1 for x in range(len(ch)) for y in range(x + 1, len(ch)) if circle_cross(N, *ch[x], *ch[y]))
        cs.append(mk('counting.crossing_pairs', (ch,), t, f'coordinates of a regular {N}-gon and segment intersection tests'))
    return cs


def g_chord_prob(rng, n):
    cs = []
    for i in range(n):
        model = ['four_points', 'two_distinct_chords', 'independent_chords', 'independent_points'][i % 4]
        N = rng.randint(4, 9)
        chords = list(itertools.combinations(range(N), 2))
        if model == 'four_points':
            good = tot = 0
            for q in itertools.combinations(range(N), 4):
                a, b, c, d = q
                for pairing in (((a, b), (c, d)), ((a, c), (b, d)), ((a, d), (b, c))):
                    tot += 1
                    good += circle_cross(N, *pairing[0], *pairing[1])
            cs.append(mk('counting.chord_crossing_probability', (model, N), Fraction(good, tot), 'brute: all 4-point sets and all pairings, real coordinates'))
        elif model == 'two_distinct_chords':
            good = sum(1 for x in range(len(chords)) for y in range(x + 1, len(chords)) if circle_cross(N, *chords[x], *chords[y]))
            tot = len(chords) * (len(chords) - 1) // 2
            cs.append(mk('counting.chord_crossing_probability', (model, N), Fraction(good, tot), 'brute: all chord pairs, real coordinates'))
        elif model == 'independent_chords':
            good = sum(1 for x in chords for y in chords if circle_cross(N, *x, *y))
            cs.append(mk('counting.chord_crossing_probability', (model, N), Fraction(good, len(chords) ** 2), 'brute: all ordered chord pairs, real coordinates'))
        else:
            N = min(N, 8)
            good = sum(1 for t in itertools.product(range(N), repeat=4) if circle_cross(N, *t))
            cs.append(mk('counting.chord_crossing_probability', (model, N), Fraction(good, N ** 4), 'brute: all 4-tuples of points, real coordinates'))
    cs[0] = mk('counting.chord_crossing_probability', ('four_points',), Fraction(1, 3), 'known value (four uniform points on a circle)')
    return cs


def matchings_all(pts):
    if not pts:
        yield []
        return
    a = pts[0]
    for j in range(1, len(pts)):
        rest = pts[1:j] + pts[j + 1:]
        for m in matchings_all(rest):
            yield [(a, pts[j])] + m


def g_matching_crossings(rng, n):
    cs = []
    for i in range(n):
        m = i % 6
        ms = list(matchings_all(list(range(2 * m))))
        noncross = 0
        cr = 0
        for mt in ms:
            c = sum(1 for x in range(len(mt)) for y in range(x + 1, len(mt)) if circle_cross(max(2 * m, 4), *mt[x], *mt[y]))
            cr += c
            noncross += c == 0
        tot = len(ms)
        cs.append(mk('counting.matching_crossings', (m,), {'matchings': tot, 'non_crossing': noncross, 'probability_non_crossing': Fraction(noncross, tot), 'expected_crossings': Fraction(cr, tot)},
                     'brute enumeration of all perfect matchings, real coordinates'))
    return cs



def named_graphs():
    petersen = (10, [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0), (0, 5), (1, 6), (2, 7), (3, 8), (4, 9), (5, 7), (7, 9), (9, 6), (6, 8), (8, 5)])
    cube = (8, [(u, v) for u in range(8) for v in range(u + 1, 8) if bin(u ^ v).count('1') == 1])
    k33 = (6, [(u, 3 + v) for u in range(3) for v in range(3)])
    octa = (6, [(u, v) for u in range(6) for v in range(u + 1, 6) if v - u != 3])
    prism = (6, [(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3), (0, 3), (1, 4), (2, 5)])
    wheel5 = (6, [(0, j) for j in range(1, 6)] + [(j, j % 5 + 1) for j in range(1, 6)])
    return [('Petersen', petersen), ('cube Q3', cube), ('K33', k33), ('octahedron', octa), ('triangular prism', prism), ('wheel W5', wheel5)]


def g_named(rng, n):
    """Structured graphs of the kind a contest names, with brute-force oracles (not random graphs)."""
    cs = []
    for name, (m, es) in named_graphs():
        eset = set(es)
        trees = sum(1 for sub in itertools.combinations(es, m - 1) if is_tree(m, list(sub)))
        cs.append(mk('counting.spanning_trees', (m, es), trees, f'brute enumeration ({name})'))
        cs.append(mk('counting.perfect_matchings', (m, es), pm_brute(m, es), f'brute enumeration ({name})'))
        counts = {}
        adj = [0] * m
        for u, v in es:
            adj[u] |= 1 << v
            adj[v] |= 1 << u
        for mask in range(1 << m):
            if all(not (adj[j] & mask) for j in range(m) if mask >> j & 1):
                counts[bin(mask).count('1')] = counts.get(bin(mask).count('1'), 0) + 1
        cs.append(mk('counting.independent_sets', (m, es, 'poly'), [counts.get(s, 0) for s in range(max(counts) + 1)], f'brute enumeration ({name})'))
        for k in (2, 3, 4):
            cs.append(mk('counting.colorings_count', (m, es, k), proper_colorings_brute(m, es, k), f'brute enumeration ({name})'))
        mt = 0
        for mask in range(1 << len(es)):
            sub = [es[j] for j in range(len(es)) if mask >> j & 1]
            vs = [x for e in sub for x in e]
            mt += len(set(vs)) == len(vs)
        cs.append(mk('counting.matchings_count', (m, es), mt, f'brute enumeration ({name})'))
    for name, (m, es), val in (('Petersen', named_graphs()[0][1], 120), ('cube Q3', named_graphs()[1][1], 48), ('octahedron', named_graphs()[3][1], 48), ('triangular prism', named_graphs()[4][1], 12)):
        cs.append(mk('counting.graph_automorphisms', (m, es), val, f'known value ({name}: order of the automorphism group)'))
    return cs


def g_refusals(rng, n):
    """Cases past the bounds: the backend must say NotCovered, not compute and not hang."""
    big_edges = [(u, v) for u in range(30) for v in range(u + 1, 30)]
    cases = [
        ('counting.surjections', (10 ** 6, 5), {}),
        ('counting.derangements', (10 ** 6,), {}),
        ('counting.set_partitions', (5000,), {}),
        ('counting.partition_count', (10 ** 6,), {}),
        ('counting.catalan', (10 ** 7,), {}),
        ('counting.necklace_count', (10 ** 6, 3), {}),
        ('counting.chromatic_polynomial', (20, [(0, 1)]), {}),
        ('counting.perfect_matchings', (30, big_edges), {}),
        ('counting.independent_sets', (60, [(0, 1)]), {}),
        ('counting.spanning_trees', (200, [(0, 1)]), {}),
        ('counting.brute_count', (['1..10000', '1..10000'], 'x0 < x1'), {}),
        ('counting.matchings_count', (30, [(0, 1)]), {}),
        ('counting.permutations_allowed', ([[1] * 25 for _ in range(25)],), {}),
        ('counting.grid_colorings', (30, 30, 4), {}),
        ('counting.domino_tilings', (30, 30), {}),
        ('counting.strings_with_patterns', (36, 10 ** 6), {'avoid': ['abc'] * 3}),
        ('counting.count_digit_constraints', (1, 10 ** 70), {}),
        ('counting.coupon_collector', (), {'probs': [Fraction(1, 25)] * 25}),
        ('counting.anagram_count', ('ab' * 300,), {}),
        ('counting.mahonian', (2000, 10 ** 6), {}),
        ('counting.markov_distribution', ([[Fraction(1, 2), Fraction(1, 2)], [Fraction(1, 2), Fraction(1, 2)]], 0, 10 ** 8), {}),
    ]
    return [mk(op, args, Refuse('past the bound'), 'expected refusal', **kw) for op, args, kw in cases]


GENERATORS = [
    g_stars_and_bars, g_derangements, g_rencontres, g_surjections, g_set_partitions, g_twelvefold, g_cycle_type, g_eulerian, g_mahonian, g_anagram, g_subset_sum,
    g_partition_count, g_compositions, g_syt, g_ssyt, g_necklace, g_orbit, g_lattice, g_catalan, g_dyck, g_ballot, g_motzkin, g_schroeder, g_narayana,
    g_strings, g_transfer, g_grid, g_domino, g_digits, g_spanning, g_labelled, g_forests, g_perfect_matchings, g_matchings, g_perm_allowed, g_independent, g_chromatic, g_colorings,
    g_automorphisms, g_union, g_divisible, g_coprime, g_brute, g_hist, g_prob, g_expect, g_dice_dist, g_dice_prob, g_dice_ext, g_dist_stats, g_bayes, g_binom, g_hyper,
    g_mhyper, g_geom, g_coupon, g_coupon_prob, g_gambler, g_birthday, g_absorption, g_hitting, g_stationary, g_mdist, g_pattern_race, g_wait, g_walk,
    g_chords_cross, g_crossing_pairs, g_chord_prob, g_matching_crossings, g_named,
]


def registered():
    return sorted(f.op_name for f in vars(C).values() if hasattr(f, 'op_name') and f.op_name.startswith('counting.'))


def generate(seed, n=60):
    """n cases for every operation, from a fixed seed; the refusal cases (past the bounds) come last."""
    rng = random.Random(seed)
    cases = []
    for g in GENERATORS:
        cases.extend(g(random.Random(rng.random()), n))
    cases.extend(g_refusals(rng, n))
    return cases


def _same(a, b):
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    return a == b and type(a) in (type(b), int, Fraction, bool) or (isinstance(a, (int, Fraction)) and isinstance(b, (int, Fraction)) and a == b)


def run(seed=1, n=60, out='build-mathbackend/counting.json'):
    t0 = time.time()
    cases = generate(seed, n)
    ops = registered()
    rows = {o: dict(generated=0, answered=0, agree=0, refused=0, refusal_reasons={}, expected_refusals=0, expected_refusals_ok=0, disagree=0, seconds=0.0, oracles={}, disagreements=[]) for o in ops}
    for c in cases:
        row = rows[c.op]
        row['generated'] += 1
        row['oracles'][c.oracle_name] = row['oracles'].get(c.oracle_name, 0) + 1
        fn = getattr(C, c.op.split('.', 1)[1])
        t1 = time.time()
        expect_refusal = isinstance(c.oracle, Refuse)
        if expect_refusal:
            row['expected_refusals'] += 1
        try:
            got = fn(*c.args, **c.kwargs)
        except NotCovered as e:
            reason = str(e).split(':')[0][:60]
            row['seconds'] += time.time() - t1
            if expect_refusal:
                row['expected_refusals_ok'] += 1
            else:
                row['refused'] += 1
                row['refusal_reasons'][reason] = row['refusal_reasons'].get(reason, 0) + 1
            continue
        except BadInput as e:
            row['seconds'] += time.time() - t1
            row['disagree'] += 1
            row['disagreements'].append(dict(args=repr(c.args)[:200], kwargs=repr(c.kwargs)[:100], oracle=repr(c.oracle)[:200], backend=f'BadInput: {e}'))
            continue
        except Exception as e:     # a crash is a disagreement
            row['seconds'] += time.time() - t1
            row['disagree'] += 1
            row['disagreements'].append(dict(args=repr(c.args)[:200], kwargs=repr(c.kwargs)[:100], oracle=repr(c.oracle)[:200], backend=f'{type(e).__name__}: {e}'))
            continue
        row['seconds'] += time.time() - t1
        row['answered'] += 1
        if expect_refusal:
            row['disagree'] += 1
            row['disagreements'].append(dict(args=repr(c.args)[:200], kwargs=repr(c.kwargs)[:100], oracle='must refuse', backend=repr(got)[:100]))
        elif _same(got, c.oracle):
            row['agree'] += 1
        else:
            row['disagree'] += 1
            row['disagreements'].append(dict(args=repr(c.args)[:200], kwargs=repr(c.kwargs)[:100], oracle=repr(c.oracle)[:200], backend=repr(got)[:200]))
    no_cases = [o for o in ops if rows[o]['generated'] == 0]
    total = dict(operations=len(ops), generated=sum(r['generated'] for r in rows.values()), answered=sum(r['answered'] for r in rows.values()),
                 agree=sum(r['agree'] for r in rows.values()), refused=sum(r['refused'] for r in rows.values()),
                 expected_refusals=sum(r['expected_refusals'] for r in rows.values()), expected_refusals_ok=sum(r['expected_refusals_ok'] for r in rows.values()),
                 disagree=sum(r['disagree'] for r in rows.values()))
    report = dict(module='counting', seed=seed, n_per_operation=n, seconds=round(time.time() - t0, 1), total=total, operations_without_cases=no_cases, per_operation=rows,
                  not_covered=C.NOT_COVERED)
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=1, default=str), encoding='utf-8')
    return report


def table(report):
    lines = [f"{'operation':44} {'gen':>4} {'ans':>4} {'agree':>5} {'refused':>7} {'exp.ref':>7} {'disagree':>8} {'sec':>6}"]
    for o, r in report['per_operation'].items():
        lines.append(f"{o:44} {r['generated']:4d} {r['answered']:4d} {r['agree']:5d} {r['refused']:7d} {str(r['expected_refusals_ok']) + '/' + str(r['expected_refusals']):>7} {r['disagree']:8d} {r['seconds']:6.2f}")
    t = report['total']
    lines.append(f"{'TOTAL':44} {t['generated']:4d} {t['answered']:4d} {t['agree']:5d} {t['refused']:7d} {str(t['expected_refusals_ok']) + '/' + str(t['expected_refusals']):>7} {t['disagree']:8d} {report['seconds']:6.1f}")
    return '\n'.join(lines)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--n', type=int, default=60)
    ap.add_argument('--out', default='build-mathbackend/counting.json')
    a = ap.parse_args()
    rep = run(a.seed, a.n, a.out)
    print(table(rep))
    print('operations without cases:', rep['operations_without_cases'])
    for o, r in rep['per_operation'].items():
        for d in r['disagreements'][:3]:
            print('DISAGREE', o, d)
    print('written', a.out)
