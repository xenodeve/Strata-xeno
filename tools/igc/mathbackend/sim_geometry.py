"""Simulation of tools/igc/mathbackend/geometry.py with no model: random cases against an independent oracle, per operation.

    PYTHONPATH=. PYTHONUTF8=1 <venv-with-sympy>/python -m tools.igc.mathbackend.sim_geometry --seed 1 --n 60

(run it with -m: started as a script, the folder of numbers.py would shadow the standard library module of that name)

`generate(seed, n)` returns a list of cases (op_name, args, oracle_value, oracle_name); `run(seed, n)` runs them and returns (and writes to
build-mathbackend/geometry.json) per operation: generated, answered, agree, refused (by reason), refused_expected, disagree, seconds.

The backend code was written from the mathematics, nothing in it is copied from SymPy or mpmath, so both stay independent oracles. The oracle of each case is named:
  sympy ...        the SymPy geometry object (Point, Line, Circle, Triangle, Polygon, Plane) answers it exactly; a result with a radical is compared at 50 digits
  mpmath/float     coordinates computed with mpmath (40 digits) or doubles; a regular n-gon is enumerated by brute force with these (no integer direction classes)
  brute            explicit enumeration written here (lattice points by bounding box and exact signs, convex polygons by half-plane vertex enumeration, ...)
  certificate      the answer is verified by an exact identity (a point satisfies the circle and the line equation in Surd arithmetic, a sphere is equidistant from its points)
  literature       a published value or formula (OEIS A006561 and the Poonen-Rubinstein formula for the diagonal intersections, the AIME 2024 I/6 answer 315, Gauss circle counts)
Where an oracle expects the backend to refuse (concentric circles, overlapping polylines, an angle outside the table) the case says so (REFUSE); a refusal the oracle did
not expect is counted under `refused`, not `agree`."""
import argparse
import json
import math
import random
import sys
import time
from decimal import Decimal
from fractions import Fraction
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import mpmath as mp                    # noqa: E402  (oracle)
import sympy as sp                     # noqa: E402  (oracle)
from sympy.geometry import (Circle, Line, Line3D, Plane, Point, Point3D, Polygon, RegularPolygon, Segment, Triangle, convex_hull)   # noqa: E402

from tools.igc.mathbackend import BadInput, NotCovered, Surd, registry, run as backend_run   # noqa: E402

mp.mp.dps = 50
REFUSE = 'REFUSE'
BADIN = 'BADINPUT'
LIT = {'name': 'literature'}


# ----------------------------------------------------------------------------------------------------------------------------- helpers

def S(f):
    f = Fraction(f)
    return sp.Rational(f.numerator, f.denominator)


def Fr(x):
    x = sp.nsimplify(x) if not isinstance(x, (sp.Rational, sp.Integer)) else x
    if not x.is_Rational:
        raise ValueError(f'not rational: {x}')
    return Fraction(int(x.p), int(x.q))


def SPt(p):
    return Point(*[S(c) for c in p])


def FrPt(pt):
    return tuple(Fr(c) for c in pt.args)


def MF(f):
    f = Fraction(f)
    return mp.mpf(f.numerator) / f.denominator


def SM(s):
    return MF(s.p) + MF(s.q) * mp.sqrt(s.d) if s.q else MF(s.p)


def N50(expr):
    return mp.mpf(str(sp.N(expr, 55)))


def close(a, b, tol=mp.mpf(10) ** -35):
    a, b = mp.mpf(a), mp.mpf(b)
    return abs(a - b) <= tol * max(1, abs(a), abs(b))


def rq(rng, lo=-9, hi=9, rat=0.35):
    n = rng.randint(lo, hi)
    return Fraction(n, rng.choice((2, 3, 4, 5, 6, 7))) if rng.random() < rat else Fraction(n)


def rp(rng, dim=2, lo=-9, hi=9, rat=0.35):
    return tuple(rq(rng, lo, hi, rat) for _ in range(dim))


def ip(rng, dim=2, lo=-6, hi=6):
    return tuple(Fraction(rng.randint(lo, hi)) for _ in range(dim))


def cross2(u, v):
    return u[0] * v[1] - u[1] * v[0]


def sub(p, q):
    return tuple(a - b for a, b in zip(p, q))


def surd_ok(s):
    return isinstance(s, Surd)


def spoly(pts):
    return Polygon(*[SPt(p) for p in pts])


def signed_area_sp(poly):
    return Fr(poly.area)


def star_poly(rng, kmin=3, kmax=8, R=7, integer=True):
    """A random simple polygon star-shaped about its integer centre: (vertices, centre, sympy polygon). Fan triangles (centre, v_i, v_i+1) all have positive area."""
    for _ in range(200):
        k = rng.randint(kmin, kmax)
        c = (Fraction(rng.randint(-2, 2)), Fraction(rng.randint(-2, 2)))
        pts = list({(c[0] + rng.randint(-R, R), c[1] + rng.randint(-R, R)) for _ in range(k)})
        pts = [p for p in pts if p != c]
        if len(pts) < 3:
            continue
        pts.sort(key=lambda p: math.atan2(float(p[1] - c[1]), float(p[0] - c[0])))
        n = len(pts)
        if any(cross2(sub(pts[i], c), sub(pts[(i + 1) % n], c)) <= 0 for i in range(n)):
            continue
        try:
            poly = spoly(pts)
        except Exception:
            continue
        if len(poly.vertices) != n:
            continue
        return pts, c, poly
    raise RuntimeError('no star polygon')


# ----------------------------------------------------------------------------------------------------------------------------- generators
# Each generator returns a list of (args, oracle, oracle_name). The checker of an operation is in CHECK (default: ==).

GEN = {}
CHECK = {}


def gen(name):
    def deco(fn):
        GEN['geometry.' + name] = fn
        return fn
    return deco


def check(name):
    def deco(fn):
        CHECK['geometry.' + name] = fn
        return fn
    return deco


@gen('dist2')
def g_dist2(rng, k):
    out = []
    for _ in range(k):
        d = rng.choice((2, 3))
        p, q = rp(rng, d), rp(rng, d)
        out.append(((p, q), Fr(Point(*map(S, p)).distance(Point(*map(S, q))) ** 2), 'sympy Point.distance squared'))
    return out


@gen('dist')
def g_dist(rng, k):
    out = []
    for _ in range(k):
        d = rng.choice((2, 3))
        p, q = rp(rng, d, rat=0.2), rp(rng, d, rat=0.2)
        dd = Point(*map(S, p)).distance(Point(*map(S, q)))
        out.append(((p, q), (str(sp.N(dd, 55)), sum((a - b) ** 2 for a, b in zip(p, q))), 'sympy distance at 55 digits + certificate Surd^2 = dist2'))
    return out


@check('dist')
def c_dist(args, res, orc):
    return surd_ok(res) and res * res == Surd(orc[1]) and res.sign() >= 0 and close(SM(res), mp.mpf(orc[0]))


@gen('midpoint')
def g_midpoint(rng, k):
    out = []
    for _ in range(k):
        d = rng.choice((2, 3))
        p, q = rp(rng, d), rp(rng, d)
        out.append(((p, q), FrPt(Point(*map(S, p)).midpoint(Point(*map(S, q)))), 'sympy Point.midpoint'))
    return out


@gen('dot')
def g_dot(rng, k):
    out = []
    for _ in range(k):
        d = rng.choice((2, 3))
        u, v = rp(rng, d), rp(rng, d)
        out.append(((u, v), Fr(sp.Matrix([S(a) for a in u]).dot(sp.Matrix([S(a) for a in v]))), 'sympy Matrix.dot'))
    return out


@gen('cross')
def g_cross(rng, k):
    out = []
    for _ in range(k):
        d = rng.choice((2, 3))
        u, v = rp(rng, d), rp(rng, d)
        if d == 2:
            orc = Fr(sp.Matrix([[S(u[0]), S(u[1])], [S(v[0]), S(v[1])]]).det())
        else:
            orc = tuple(Fr(c) for c in sp.Matrix([S(a) for a in u]).cross(sp.Matrix([S(a) for a in v])))
        out.append(((u, v), orc, 'sympy Matrix.det / Matrix.cross'))
    return out


def on_line(rng, p, d, t_lo=-5, t_hi=5):
    t = Fraction(rng.randint(t_lo, t_hi), rng.choice((1, 1, 2, 3)))
    return tuple(a + t * b for a, b in zip(p, d))


@gen('collinear')
def g_collinear(rng, k):
    out = []
    for i in range(k):
        dim = rng.choice((2, 3))
        n = rng.randint(3, 5)
        if i % 2 == 0:
            p0, d = rp(rng, dim, rat=0.2), rp(rng, dim, -3, 3, rat=0.2)
            pts = [on_line(rng, p0, d) for _ in range(n)]
            if rng.random() < 0.4:
                j = rng.randrange(n)
                pts[j] = tuple(c + (1 if idx == rng.randrange(dim) else 0) for idx, c in enumerate(pts[j]))
        else:
            pts = [rp(rng, dim, rat=0.2) for _ in range(n)]
        if len(set(pts)) < 2:
            continue
        out.append((tuple(pts), bool(Point.is_collinear(*[SPt(p) for p in pts])), 'sympy Point.is_collinear'))
    return out


@gen('coplanar')
def g_coplanar(rng, k):
    out = []
    for i in range(k):
        n = rng.randint(4, 6)
        if i % 2 == 0:
            p0, u, v = rp(rng, 3, rat=0.2), rp(rng, 3, -3, 3, rat=0.2), rp(rng, 3, -3, 3, rat=0.2)
            pts = [tuple(p0[c] + rng.randint(-4, 4) * u[c] + Fraction(rng.randint(-4, 4), rng.choice((1, 2))) * v[c] for c in range(3)) for _ in range(n)]
            if rng.random() < 0.4:
                j = rng.randrange(n)
                pts[j] = (pts[j][0] + 1, pts[j][1], pts[j][2])
        else:
            pts = [rp(rng, 3, rat=0.2) for _ in range(n)]
        out.append((tuple(pts), bool(Point3D.are_coplanar(*[SPt(p) for p in pts])), 'sympy Point3D.are_coplanar'))
    return out


CIRCLE_PTS = {r2: [(a, b) for a in range(-15, 16) for b in range(-15, 16) if a * a + b * b == r2] for r2 in (25, 65, 85, 125, 169, 205, 325, 50, 100)}


@gen('concyclic')
def g_concyclic(rng, k):
    out = []
    tries = 0
    while len(out) < k and tries < 5 * k:
        tries += 1
        if len(out) % 2 == 0:
            r2 = rng.choice(list(CIRCLE_PTS))
            c = ip(rng)
            pts = [(c[0] + a, c[1] + b) for a, b in rng.sample(CIRCLE_PTS[r2], 4)]
        else:
            pts = [rp(rng, 2, -6, 6, rat=0.2) for _ in range(4)]
        spts = [SPt(p) for p in pts]
        if len(set(pts)) < 4 or Point.is_collinear(*spts):
            continue
        out.append((tuple(pts), bool(Point.is_concyclic(*spts)), 'sympy Point.is_concyclic'))
    return out


@gen('line_through')
def g_line_through(rng, k):
    out = []
    while len(out) < k:
        p, q = rp(rng), rp(rng)
        if p == q:
            continue
        a, b, c = Line(SPt(p), SPt(q)).coefficients
        out.append(((p, q), (Fr(a), Fr(b), Fr(c)), 'sympy Line.coefficients (ax+by+c=0) + normal-form certificate'))
    return out


@check('line_through')
def c_line_through(args, res, orc):
    p, q = args
    a, b, c = res
    if not all(isinstance(v, int) for v in res):
        return False
    from math import gcd
    if gcd(gcd(abs(a), abs(b)), abs(c)) != 1:
        return False
    first = next(v for v in (a, b, c) if v)
    if first < 0 or (a == 0 and b == 0):
        return False
    if a * p[0] + b * p[1] != c or a * q[0] + b * q[1] != c:
        return False
    sa, sb, sc = orc
    return a * sb == b * sa and a * sc == -c * sa and b * sc == -c * sb


@gen('line_intersection')
def g_line_intersection(rng, k):
    out = []
    while len(out) < k:
        p1, p2 = rp(rng, 2, -6, 6), rp(rng, 2, -6, 6)
        mode = rng.choice(('rand', 'rand', 'par', 'same'))
        if mode == 'rand':
            p3, p4 = rp(rng, 2, -6, 6), rp(rng, 2, -6, 6)
        else:
            d = sub(p2, p1)
            base = p1 if mode == 'same' else tuple(a + b for a, b in zip(p1, rp(rng, 2, -4, 4)))
            p3, p4 = on_line(rng, base, d), on_line(rng, base, d)
        if p1 == p2 or p3 == p4:
            continue
        r = Line(SPt(p1), SPt(p2)).intersection(Line(SPt(p3), SPt(p4)))
        if not r:
            orc = 'parallel'
        elif isinstance(r[0], Line):
            orc = 'coincident'
        else:
            orc = FrPt(r[0])
        out.append(((p1, p2, p3, p4), orc, 'sympy Line.intersection'))
    return out


@gen('segment_intersection')
def g_segment_intersection(rng, k):
    out = []
    while len(out) < k:
        a, b, c, d = (ip(rng, 2, 0, 5) for _ in range(4))
        if rng.random() < 0.25:                   # collinear / overlapping pairs
            dd = ip(rng, 2, -2, 2)
            if dd == (0, 0):
                continue
            a = ip(rng, 2, 0, 3)
            b = on_line(rng, a, dd, 1, 3)
            c = on_line(rng, a, dd, -1, 3)
            d = on_line(rng, a, dd, 0, 4)
        if a == b or c == d:
            if rng.random() < 0.7:
                continue
            seg = (a, b, c, d)
            if a == b and c != d:
                orc = [a] if Segment(SPt(c), SPt(d)).contains(SPt(a)) else []
            elif c == d and a != b:
                orc = [c] if Segment(SPt(a), SPt(b)).contains(SPt(c)) else []
            else:
                orc = [a] if a == c else []
            out.append((seg, orc, 'sympy Segment.contains (degenerate segment)'))
            continue
        r = Segment(SPt(a), SPt(b)).intersection(Segment(SPt(c), SPt(d)))
        orc = []
        if r:
            if isinstance(r[0], Segment):
                orc = sorted([FrPt(r[0].p1), FrPt(r[0].p2)])
            else:
                orc = [FrPt(r[0])]
        out.append(((a, b, c, d), orc, 'sympy Segment.intersection'))
    return out


@check('segment_intersection')
def c_segint(args, res, orc):
    return sorted(res) == orc


def random_polyline(rng, graph):
    n = rng.randint(2, 5)
    if graph:
        xs = sorted(rng.sample(range(-4, 8), n))
        return [(Fraction(x), Fraction(rng.randint(-4, 6))) for x in xs]
    return [ip(rng, 2, -4, 6) for _ in range(n)]


@gen('polyline_intersections')
def g_polyline(rng, k):
    out = []
    while len(out) < k:
        graph = rng.random() < 0.6
        P, Q = random_polyline(rng, graph), random_polyline(rng, graph)
        if graph and rng.random() < 0.3:           # |x - a| style: V shapes
            a = rng.randint(0, 3)
            P = [(Fraction(x), abs(Fraction(x - a))) for x in range(-2, 7, 3)]
        found, overlap = set(), False
        for i in range(len(P) - 1):
            for j in range(len(Q) - 1):
                if P[i] == P[i + 1] or Q[j] == Q[j + 1]:
                    continue
                for e in Segment(SPt(P[i]), SPt(P[i + 1])).intersection(Segment(SPt(Q[j]), SPt(Q[j + 1]))):
                    if isinstance(e, Segment):
                        overlap = True
                    else:
                        found.add(FrPt(e))
        if any(P[i] == P[i + 1] for i in range(len(P) - 1)) or any(Q[j] == Q[j + 1] for j in range(len(Q) - 1)):
            continue
        out.append(((P, Q), REFUSE if overlap else sorted(found), 'sympy Segment.intersection over every segment pair'))
    return out


def line_dist_sq(p, a, b):
    d = Line(SPt(a), SPt(b)).distance(SPt(p))
    return Fr(d ** 2)


@gen('dist_point_line2')
def g_dpl2(rng, k):
    out = []
    while len(out) < k:
        dim = rng.choice((2, 3))
        p, a, b = rp(rng, dim), rp(rng, dim), rp(rng, dim)
        if a == b:
            continue
        L = Line(SPt(a), SPt(b))
        out.append(((p, a, b), Fr(L.distance(SPt(p)) ** 2), 'sympy Line.distance squared'))
    return out


@gen('dist_point_line')
def g_dpl(rng, k):
    out = []
    while len(out) < k:
        dim = rng.choice((2, 3))
        p, a, b = rp(rng, dim, rat=0.2), rp(rng, dim, rat=0.2), rp(rng, dim, rat=0.2)
        if a == b:
            continue
        L = Line(SPt(a), SPt(b))
        d = L.distance(SPt(p))
        out.append(((p, a, b), (str(sp.N(d, 55)), Fr(d ** 2)), 'sympy Line.distance at 55 digits + certificate Surd^2'))
    return out


@check('dist_point_line')
def c_dpl(args, res, orc):
    return surd_ok(res) and res * res == Surd(orc[1]) and res.sign() >= 0 and close(SM(res), mp.mpf(orc[0]))


@gen('circle_through3')
def g_c3(rng, k):
    out = []
    while len(out) < k:
        p, q, r = rp(rng, 2, -7, 7), rp(rng, 2, -7, 7), rp(rng, 2, -7, 7)
        if len({p, q, r}) < 3:
            continue
        if Point.is_collinear(SPt(p), SPt(q), SPt(r)):
            if len(out) % 9 == 0:
                out.append(((p, q, r), BADIN, 'sympy Point.is_collinear'))
            continue
        t = Triangle(SPt(p), SPt(q), SPt(r))
        out.append(((p, q, r), (FrPt(t.circumcenter), Fr(t.circumradius ** 2)), 'sympy Triangle.circumcenter / circumradius'))
    return out


def surd_pt_ok(pt, c, r2, p, q):
    """Certificate: the Surd point is on the circle and on the line pq, exactly."""
    x, y = pt
    lhs = (x - c[0]) * (x - c[0]) + (y - c[1]) * (y - c[1])
    if lhs != Surd(r2):
        return False
    if p is None:
        return True
    return (x - p[0]) * (q[1] - p[1]) == (y - p[1]) * (q[0] - p[0])


def points_close(res, orc):
    res = sorted((SM(x), SM(y)) for x, y in res)
    orc = sorted((mp.mpf(a), mp.mpf(b)) for a, b in orc)
    return len(res) == len(orc) and all(close(a, c) and close(b, d) for (a, b), (c, d) in zip(res, orc))


@gen('circle_line_intersection')
def g_cli(rng, k):
    out = []
    while len(out) < k:
        c = rp(rng, 2, -4, 4, rat=0.2)
        mode = rng.choice(('rand', 'rand', 'lattice', 'tangent'))
        if mode == 'lattice':
            r2 = rng.choice([25, 65, 85, 50, 100, 125])
            cc = ip(rng, 2, -3, 3)
            pa, pb = rng.sample(CIRCLE_PTS[r2], 2)
            c, p, q, rr = cc, (cc[0] + pa[0], cc[1] + pa[1]), (cc[0] + pb[0], cc[1] + pb[1]), Fraction(r2)
        else:
            p, q = rp(rng, 2, -6, 6), rp(rng, 2, -6, 6)
            if p == q:
                continue
            rr = Fraction(rng.randint(1, 40), rng.choice((1, 1, 2, 3)))
            if mode == 'tangent':
                rr = line_dist_sq(c, p, q)
                if rr == 0:
                    continue
        try:
            inter = Circle(SPt(c), sp.sqrt(S(rr))).intersection(Line(SPt(p), SPt(q)))
        except Exception:
            continue
        orc = [(N50(a), N50(b)) for a, b in (pt.args for pt in inter)]
        out.append(((c, rr, p, q), (orc, c, rr, p, q), 'sympy Circle.intersection(Line) at 55 digits + certificate (Surd equations)'))
    return out


@check('circle_line_intersection')
def c_cli(args, res, orc):
    o, c, r2, p, q = orc
    return points_close(res, o) and all(surd_pt_ok(pt, c, r2, p, q) for pt in res)


@gen('circle_circle_intersection')
def g_cci(rng, k):
    out = []
    while len(out) < k:
        c1 = rp(rng, 2, -4, 4, rat=0.2)
        c2 = tuple(a + b for a, b in zip(c1, rp(rng, 2, -4, 4, rat=0.2)))
        if rng.random() < 0.08:
            c2 = c1
        if rng.random() < 0.3:                      # tangent externally with integer radii
            R1, R2 = rng.randint(1, 4), rng.randint(1, 4)
            c1, c2 = (Fraction(0), Fraction(0)), (Fraction(R1 + R2), Fraction(0))
            r1, r2 = Fraction(R1 * R1), Fraction(R2 * R2)
        else:
            r1, r2 = Fraction(rng.randint(1, 30), rng.choice((1, 1, 2))), Fraction(rng.randint(1, 30), rng.choice((1, 1, 2)))
        if c1 == c2:
            out.append(((c1, r1, c2, r2), REFUSE, 'concentric circles are refused'))
            continue
        try:
            inter = Circle(SPt(c1), sp.sqrt(S(r1))).intersection(Circle(SPt(c2), sp.sqrt(S(r2))))
        except Exception:
            continue
        orc = [(N50(a), N50(b)) for a, b in (pt.args for pt in inter)]
        out.append(((c1, r1, c2, r2), (orc, c1, r1, c2, r2), 'sympy Circle.intersection(Circle) at 55 digits + certificate (both circle equations)'))
    return out


@check('circle_circle_intersection')
def c_cci(args, res, orc):
    o, c1, r1, c2, r2 = orc
    return points_close(res, o) and all(surd_pt_ok(pt, c1, r1, None, None) and surd_pt_ok(pt, c2, r2, None, None) for pt in res)


@gen('circle_circle_relation')
def g_ccr(rng, k):
    out = []
    while len(out) < k:
        if rng.random() < 0.6:
            R1, R2 = rng.randint(1, 8), rng.randint(1, 8)
            a, b, h = rng.choice([(3, 4, 5), (5, 12, 13), (8, 15, 17), (0, 1, 1), (1, 0, 1), (6, 8, 10)])
            m = rng.randint(1, 3)
            d = rng.choice([R1 + R2, abs(R1 - R2), R1 + R2 + 1, max(1, abs(R1 - R2) - 1), h * m, abs(R1 - R2) + 1])
            if d == 0:
                continue
            c1, c2 = ip(rng, 2, -3, 3), None
            c2 = (c1[0] + d * Fraction(a, h), c1[1] + d * Fraction(b, h)) if h else None
            if c2 is None or any(x.denominator != 1 for x in c2):
                continue
            r1, r2 = Fraction(R1 * R1), Fraction(R2 * R2)
        else:
            c1, c2 = rp(rng, 2, -5, 5, rat=0.2), rp(rng, 2, -5, 5, rat=0.2)
            r1, r2 = Fraction(rng.randint(1, 30), rng.choice((1, 2))), Fraction(rng.randint(1, 30), rng.choice((1, 2)))
        d = mp.sqrt(sum((MF(x) - MF(y)) ** 2 for x, y in zip(c1, c2)))
        R1, R2 = mp.sqrt(MF(r1)), mp.sqrt(MF(r2))
        tol = mp.mpf(10) ** -40
        if c1 == c2 and r1 == r2:
            orc = 'identical'
        elif abs(d - (R1 + R2)) < tol:
            orc = 'external tangent'
        elif d > R1 + R2:
            orc = 'separate'
        elif abs(d - abs(R1 - R2)) < tol:
            orc = 'internal tangent'
        elif d < abs(R1 - R2):
            orc = 'contained'
        else:
            orc = 'intersecting'
        out.append(((c1, r1, c2, r2), orc, 'mpmath 50 digits: d against R1 + R2 and |R1 - R2|'))
    return out


@gen('circle_circle_overlap_area_numeric')
def g_overlap(rng, k):
    out = []
    while len(out) < k:
        c1 = rp(rng, 2, -4, 4, 0.2)
        c2 = tuple(a + b for a, b in zip(c1, rp(rng, 2, -5, 5, 0.2)))
        if c1 == c2:
            continue
        r1, r2 = Fraction(rng.randint(1, 30), rng.choice((1, 2))), Fraction(rng.randint(1, 30), rng.choice((1, 2)))
        dg = rng.choice((3, 10, 25))
        d = mp.sqrt(sum((MF(a) - MF(b)) ** 2 for a, b in zip(c1, c2)))
        R1, R2 = mp.sqrt(MF(r1)), mp.sqrt(MF(r2))
        # oracle: integrate the chord length 2 min(sqrt(R1^2 - x^2), sqrt(R2^2 - (x - d)^2)) along the line of centres (no inverse cosine in the formula)
        if d >= R1 + R2:
            area = mp.mpf(0)
        elif d <= abs(R1 - R2):
            area = mp.pi * min(R1, R2) ** 2
        else:
            lo, hi = max(-R1, d - R2), min(R1, d + R2)
            x0 = (d * d + R1 * R1 - R2 * R2) / (2 * d)
            f = lambda x: 2 * min(mp.sqrt(max(R1 * R1 - x * x, 0)), mp.sqrt(max(R2 * R2 - (x - d) ** 2, 0)))
            area = mp.quad(f, [lo, x0, hi])
        out.append(((c1, r1, c2, r2, dg), (str(area), dg), 'mpmath quad of the chord length along the line of centres, split where the two arcs cross'))
    return out


@check('circle_circle_overlap_area_numeric')
def c_overlap(args, res, orc):
    return isinstance(res, Decimal) and abs(mp.mpf(str(res)) - mp.mpf(orc[0])) <= mp.mpf(10) ** -min(orc[1], 14) * mp.mpf('1.0001')


@gen('power_of_point')
def g_pop(rng, k):
    out = []
    for _ in range(k):
        p, c, r2 = rp(rng, 2), rp(rng, 2), Fraction(rng.randint(0, 30), rng.choice((1, 2, 3)))
        out.append(((p, c, r2), Fr(SPt(p).distance(SPt(c)) ** 2 - S(r2)), 'sympy Point.distance squared minus r2'))
    return out


@gen('tangent_length')
def g_tl(rng, k):
    out = []
    for _ in range(k):
        p, c = rp(rng, 2, rat=0.2), rp(rng, 2, rat=0.2)
        r2 = Fraction(rng.randint(1, 30), rng.choice((1, 2)))
        pw = Fr(SPt(p).distance(SPt(c)) ** 2 - S(r2))
        out.append(((p, c, r2), BADIN if pw < 0 else (str(sp.N(sp.sqrt(S(pw)), 55)), pw), 'sympy sqrt of the power at 55 digits + certificate Surd^2'))
    return out


@check('tangent_length')
def c_tl(args, res, orc):
    return surd_ok(res) and res * res == Surd(orc[1]) and res.sign() >= 0 and close(SM(res), mp.mpf(orc[0]))


@gen('triangle_area')
def g_ta(rng, k):
    out = []
    for _ in range(k):
        p, q, r = rp(rng), rp(rng), rp(rng)
        try:
            a = abs(Fr(Triangle(SPt(p), SPt(q), SPt(r)).area))
        except Exception:
            a = Fraction(0)                              # collinear: sympy refuses a Triangle, the area is 0
        out.append(((p, q, r), a, 'sympy Triangle.area'))
    return out


@gen('triangle_area_sq')
def g_tas(rng, k):
    out = []
    for _ in range(k):
        dim = rng.choice((2, 3))
        p, q, r = rp(rng, dim), rp(rng, dim), rp(rng, dim)
        a2, b2, c2 = (sum((x - y) ** 2 for x, y in zip(u, v)) for u, v in ((q, r), (p, r), (p, q)))
        out.append(((p, q, r), (2 * (a2 * b2 + b2 * c2 + c2 * a2) - (a2 * a2 + b2 * b2 + c2 * c2)) / 16, 'Heron on the squared sides (a different formula)'))
    return out


@gen('triangle_from_sides')
def g_tfs(rng, k):
    out = []
    while len(out) < k:
        a, b, c = (Fraction(rng.randint(1, 20), rng.choice((1, 1, 2, 3))) for _ in range(3))
        if rng.random() < 0.3:
            a, b, c = rng.choice([(3, 4, 5), (5, 12, 13), (13, 14, 15), (7, 15, 20), (9, 10, 17)])
            a, b, c = Fraction(a), Fraction(b), Fraction(c)
        if a + b <= c or a + c <= b or b + c <= a:
            if len(out) % 8 == 0:
                out.append(((a, b, c), BADIN, 'triangle inequality'))
            continue
        t = Triangle(sss=(S(a), S(b), S(c)))
        orc = dict(area=N50(t.area), cr=N50(t.circumradius), ir=N50(t.inradius), a=a, b=b, c=c)
        out.append(((a, b, c), orc, 'sympy Triangle(sss=) (placed by coordinates, area by shoelace) at 55 digits'))
    return out


@check('triangle_from_sides')
def c_tfs(args, res, o):
    a, b, c = o['a'], o['b'], o['c']
    s = (a + b + c) / 2
    return (close(SM(res['area']), o['area']) and res['area'] * res['area'] == Surd(res['area_sq']) and res['semiperimeter'] == s
            and close(SM(res['circumradius']), o['cr']) and close(SM(res['inradius']), o['ir'])
            and res['circumradius'] * res['circumradius'] == Surd(res['circumradius_sq']) and res['inradius'] * Surd(s) == res['area'])


@gen('triangle_from_sides_sq')
def g_tfss(rng, k):
    out = []
    while len(out) < k:
        a2, b2, c2 = (Fraction(rng.randint(1, 40), rng.choice((1, 1, 2))) for _ in range(3))
        kk = 2 * (a2 * b2 + b2 * c2 + c2 * a2) - (a2 * a2 + b2 * b2 + c2 * c2)
        if kk <= 0:
            if len(out) % 8 == 0:
                out.append(((a2, b2, c2), BADIN, 'Heron: 16 area^2 <= 0'))
            continue
        t = Triangle(sss=(sp.sqrt(S(a2)), sp.sqrt(S(b2)), sp.sqrt(S(c2))))
        out.append(((a2, b2, c2), (N50(t.area ** 2), N50(t.circumradius ** 2)), 'sympy Triangle(sss=) with sqrt sides at 55 digits'))
    return out


@check('triangle_from_sides_sq')
def c_tfss(args, res, orc):
    return close(MF(res['area_sq']), orc[0]) and close(MF(res['circumradius_sq']), orc[1])


@gen('circumradius_sq')
def g_crs(rng, k):
    out = []
    while len(out) < k:
        dim = rng.choice((2, 3))
        p, q, r = rp(rng, dim), rp(rng, dim), rp(rng, dim)
        u, v = sub(q, p), sub(r, p)
        if dim == 2:
            cr2 = cross2(u, v) ** 2
        else:
            cr2 = sum(c * c for c in (u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]))
        if cr2 == 0:
            if len(out) % 8 == 0:
                out.append(((p, q, r), BADIN, 'collinear'))
            continue
        a, b, c = (mp.sqrt(sum((MF(x) - MF(y)) ** 2 for x, y in zip(s, t))) for s, t in ((q, r), (p, r), (p, q)))
        area = mp.sqrt(MF(cr2)) / 2
        R = a * b * c / (4 * area)
        out.append(((p, q, r), str(R * R), 'mpmath R = abc/(4 area) (side lengths and the area by square roots)'))
    return out


@check('circumradius_sq')
def c_crs(args, res, orc):
    return close(MF(res), mp.mpf(orc))


@gen('inradius')
def g_ir(rng, k):
    out = []
    while len(out) < k:
        dim = rng.choice((2, 2, 3))
        if rng.random() < 0.5 and dim == 2:
            a, b = rng.choice([(3, 4), (5, 12), (8, 15), (6, 8), (7, 24)])
            m = rng.choice((1, 2))
            p, q, r = (Fraction(0), Fraction(0)), (Fraction(a * m), Fraction(0)), (Fraction(0), Fraction(b * m))
        else:
            p, q, r = rp(rng, dim, rat=0.2), rp(rng, dim, rat=0.2), rp(rng, dim, rat=0.2)
        u, v = sub(q, p), sub(r, p)
        cr2 = cross2(u, v) ** 2 if dim == 2 else sum(c * c for c in (u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]))
        if cr2 == 0:
            continue
        a, b, c = (mp.sqrt(sum((MF(x) - MF(y)) ** 2 for x, y in zip(s, t))) for s, t in ((q, r), (p, r), (p, q)))
        area = mp.sqrt(MF(cr2)) / 2
        out.append(((p, q, r), (str(2 * area / (a + b + c)), cr2), 'mpmath r = 2 area / perimeter (+ certificate r * perimeter = 2 area when answered)'))
    return out


@check('inradius')
def c_ir(args, res, orc):
    return surd_ok(res) and close(SM(res), mp.mpf(orc[0]))


@gen('circumradius')
def g_cr(rng, k):
    out = []
    while len(out) < k:
        p, q, r = rand_triangle(rng)
        t = Triangle(SPt(p), SPt(q), SPt(r))
        out.append(((p, q, r), (str(sp.N(t.circumradius, 55)), Fr(t.circumradius ** 2)), 'sympy Triangle.circumradius at 55 digits + certificate Surd^2 = R^2'))
    return out


@check('circumradius')
def c_cr_(args, res, orc):
    return surd_ok(res) and res * res == Surd(orc[1]) and res.sign() > 0 and close(SM(res), mp.mpf(orc[0]))


@gen('inradius_numeric')
def g_irn(rng, k):
    out = []
    while len(out) < k:
        p, q, r = rand_triangle(rng)
        dg = rng.choice((3, 12, 30))
        out.append(((p, q, r, dg), (str(sp.N(Triangle(SPt(p), SPt(q), SPt(r)).inradius, 70)), dg), 'sympy Triangle.inradius at 70 digits'))
    return out


@check('inradius_numeric')
def c_irn(args, res, orc):
    return isinstance(res, Decimal) and abs(mp.mpf(str(res)) - mp.mpf(orc[0])) <= mp.mpf(10) ** -orc[1] * mp.mpf('1.0001')


@gen('incenter_numeric')
def g_icn(rng, k):
    out = []
    while len(out) < k:
        p, q, r = rand_triangle(rng)
        dg = rng.choice((3, 12, 30))
        ic = Triangle(SPt(p), SPt(q), SPt(r)).incenter
        out.append(((p, q, r, dg), ((str(sp.N(ic.x, 70)), str(sp.N(ic.y, 70))), dg), 'sympy Triangle.incenter at 70 digits'))
    return out


@check('incenter_numeric')
def c_icn(args, res, orc):
    return len(res) == 2 and all(isinstance(x, Decimal) and abs(mp.mpf(str(x)) - mp.mpf(o)) <= mp.mpf(10) ** -orc[1] * mp.mpf('1.0001') for x, o in zip(res, orc[0]))


@gen('centroid')
def g_centroid(rng, k):
    out = []
    for _ in range(k):
        dim = rng.choice((2, 3))
        pts = [rp(rng, dim) for _ in range(rng.randint(1, 5))]
        m = sp.Matrix([[S(c) for c in p] for p in pts])
        orc = tuple(Fr(sum(m[:, i]) / len(pts)) for i in range(dim))
        out.append((tuple(pts), orc, 'sympy Matrix column mean'))
    return out


def rand_triangle(rng, rat=0.2):
    while True:
        p, q, r = rp(rng, 2, -8, 8, rat), rp(rng, 2, -8, 8, rat), rp(rng, 2, -8, 8, rat)
        if len({p, q, r}) == 3 and not Point.is_collinear(SPt(p), SPt(q), SPt(r)):
            return p, q, r


@gen('circumcenter')
def g_cc(rng, k):
    out = []
    for _ in range(k):
        t = rand_triangle(rng)
        out.append((t, FrPt(Triangle(*[SPt(x) for x in t]).circumcenter), 'sympy Triangle.circumcenter'))
    return out


@gen('orthocenter')
def g_oc(rng, k):
    out = []
    for _ in range(k):
        t = rand_triangle(rng)
        out.append((t, FrPt(Triangle(*[SPt(x) for x in t]).orthocenter), 'sympy Triangle.orthocenter'))
    return out


@gen('incenter')
def g_ic(rng, k):
    out = []
    for _ in range(k):
        t = rand_triangle(rng) if rng.random() < 0.5 else ((Fraction(0), Fraction(0)), (Fraction(4), Fraction(0)), (Fraction(0), Fraction(3)))
        if rng.random() < 0.3:
            t = ((Fraction(0), Fraction(0)), (Fraction(2 * rng.randint(1, 4)), Fraction(0)), (Fraction(0), Fraction(2 * rng.randint(1, 4))))
        ic = Triangle(*[SPt(x) for x in t]).incenter
        out.append((t, (N50(ic.x), N50(ic.y)), 'sympy Triangle.incenter at 55 digits'))
    return out


@check('incenter')
def c_ic(args, res, orc):
    return all(surd_ok(c) for c in res) and close(SM(res[0]), orc[0]) and close(SM(res[1]), orc[1])


def rand_vec_pair(rng):
    dim = rng.choice((2, 3))
    while True:
        if dim == 3 and rng.random() < 0.5:
            u = tuple(Fraction(rng.randint(-2, 2)) for _ in range(3))
            v = tuple(Fraction(rng.randint(-2, 2)) for _ in range(3))
        else:
            u, v = rp(rng, dim, -6, 6, rat=0.2), rp(rng, dim, -6, 6, rat=0.2)
        if any(u) and any(v):
            return u, v


def mp_angle_deg(u, v):
    """Angle in degrees by atan2(|u x v|, u.v), the square of |u x v| from the exact Lagrange identity (acos loses half the digits near 0 and 180)."""
    uu = sum(Fraction(a) ** 2 for a in u)
    vv = sum(Fraction(a) ** 2 for a in v)
    d = sum(Fraction(a) * Fraction(b) for a, b in zip(u, v))
    return mp.atan2(mp.sqrt(MF(uu * vv - d * d)), MF(d)) * 180 / mp.pi


@gen('angle_cos')
def g_acos(rng, k):
    out = []
    for _ in range(k):
        u, v = rand_vec_pair(rng)
        uu, vv, d = sum(a * a for a in u), sum(a * a for a in v), sum(a * b for a, b in zip(u, v))
        out.append(((u, v), (str(MF(d) / mp.sqrt(MF(uu) * MF(vv))), d, uu * vv), 'mpmath dot / (|u||v|) + certificate cos^2 = dot^2/(|u|^2|v|^2)'))
    return out


@check('angle_cos')
def c_acos(args, res, orc):
    return surd_ok(res) and close(SM(res), mp.mpf(orc[0])) and res * res == Surd(Fraction(orc[1] ** 2, 1) / orc[2]) and (res.sign() == (orc[1] > 0) - (orc[1] < 0))


TABLE_DEG = (0, 30, 45, 60, 90, 120, 135, 150, 180)


@gen('angle_degrees_exact')
def g_adx(rng, k):
    out = []
    while len(out) < k:
        if len(out) % 2 == 0:                  # structured pairs that do hit the table (2D: 0, 45, 90, 135, 180; 3D: 60, 120)
            sc = lambda: Fraction(rng.choice((1, 2, 3, 5)), rng.choice((1, 1, 2)))
            kind = rng.choice(('0', '45', '90', '135', '180', '60', '120'))
            if kind in ('0', '180'):
                u = rp(rng, 2, -5, 5, 0.1)
                t = sc() * (1 if kind == '0' else -1)
                v = tuple(t * c for c in u)
            elif kind in ('45', '135'):
                a, b = sc(), sc()
                u = (a, Fraction(0))
                v = (b * (1 if kind == '45' else -1), b)
            elif kind == '90':
                a, b = rq(rng, -5, 5, 0.1), rq(rng, 1, 5, 0.1)
                u, v = (a, b), (-b * sc(), a * sc())
            else:
                u, v = (Fraction(1), Fraction(1), Fraction(0)), (Fraction(1), Fraction(0), Fraction(1))
                if kind == '120':
                    v = (-1, 0, -1)
                perm = rng.sample(range(3), 3)
                sg = [rng.choice((1, -1)) for _ in range(3)]
                u = tuple(Fraction(sg[i] * u[perm[i]]) for i in range(3))
                v = tuple(Fraction(sg[i] * v[perm[i]]) for i in range(3))
            if not any(u) or not any(v):
                continue
        else:
            u, v = rand_vec_pair(rng)
        ang = mp_angle_deg(u, v)
        hit = next((t for t in TABLE_DEG if abs(ang - t) < mp.mpf(10) ** -35), None)
        out.append(((u, v), REFUSE if hit is None else hit, 'mpmath acos in degrees, matched to the table within 1e-35'))
    return out


@gen('angle_degrees_numeric')
def g_adn(rng, k):
    out = []
    for _ in range(k):
        u, v = rand_vec_pair(rng)
        dg = rng.choice((2, 5, 10, 20, 35))
        out.append(((u, v, dg), (str(mp_angle_deg(u, v)), dg), 'mpmath acos in degrees'))
    return out


@check('angle_degrees_numeric')
def c_adn(args, res, orc):
    return isinstance(res, Decimal) and abs(mp.mpf(str(res)) - mp.mpf(orc[0])) <= mp.mpf(10) ** -orc[1] * mp.mpf('1.0001')


@gen('triangle_angles_numeric')
def g_tan(rng, k):
    out = []
    for _ in range(k):
        dim = rng.choice((2, 3))
        while True:
            p, q, r = rp(rng, dim, -6, 6, 0.2), rp(rng, dim, -6, 6, 0.2), rp(rng, dim, -6, 6, 0.2)
            if len({p, q, r}) == 3:
                break
        dg = rng.choice((3, 8, 20))
        orc = (tuple(str(mp_angle_deg(sub(b, a), sub(c, a))) for a, b, c in ((p, q, r), (q, p, r), (r, p, q))), dg)
        out.append(((p, q, r, dg), orc, 'mpmath acos at each vertex'))
    return out


@check('triangle_angles_numeric')
def c_tan(args, res, orc):
    return len(res) == 3 and all(isinstance(x, Decimal) and abs(mp.mpf(str(x)) - mp.mpf(o)) <= mp.mpf(10) ** -orc[1] * mp.mpf('1.0001') for x, o in zip(res, orc[0]))


def unit(deg):
    t = MF(deg) * mp.pi / 180
    return (mp.cos(t), mp.sin(t))


def vang(u, v):
    return mp.acos(max(-1, min(1, (u[0] * v[0] + u[1] * v[1]) / (mp.sqrt(u[0] ** 2 + u[1] ** 2) * mp.sqrt(v[0] ** 2 + v[1] ** 2))))) * 180 / mp.pi


def vsub(p, q):
    return (p[0] - q[0], p[1] - q[1])


@gen('circle_angle')
def g_circle_angle(rng, k):
    out = []
    while len(out) < k:
        kind = rng.choice(('inscribed', 'tangent_chord', 'chords_inside', 'outside'))
        cut = sorted(rng.sample(range(1, 360), 3))
        arcs = [cut[0], cut[1] - cut[0], cut[2] - cut[1], 360 - cut[2]]        # consecutive arcs between four points A, B, C, D around the circle
        pos = [0, cut[0], cut[1], cut[2]]
        A, B, C, D = (unit(a) for a in pos)
        if kind == 'inscribed':
            # vertex D sees the arc AB that does not contain it (D lies after C)
            orc = vang(vsub(A, D), vsub(B, D))
            args = (kind, arcs[0])
        elif kind == 'tangent_chord':
            t = (-mp.sin(0), mp.cos(0))
            ab = rng.randint(1, 359)
            Bp = unit(ab)
            orc = vang(t, vsub(Bp, (mp.mpf(1), mp.mpf(0))))
            args = (kind, ab)
        elif kind == 'chords_inside':
            # chords AC and BD cross at P; the angle APB faces the arc AB, its vertical angle faces CD
            den = (A[0] - C[0]) * (B[1] - D[1]) - (A[1] - C[1]) * (B[0] - D[0])
            a_, b_ = A[0] * C[1] - A[1] * C[0], B[0] * D[1] - B[1] * D[0]
            P = ((a_ * (B[0] - D[0]) - (A[0] - C[0]) * b_) / den, (a_ * (B[1] - D[1]) - (A[1] - C[1]) * b_) / den)
            orc = vang(vsub(A, P), vsub(B, P))
            args = (kind, arcs[0], arcs[2])
        else:
            # lines AB and DC meet at P outside; the wedge facing the far arc DA and the near arc BC is the angle BPC
            den = (A[0] - B[0]) * (C[1] - D[1]) - (A[1] - B[1]) * (C[0] - D[0])
            if abs(den) < 1e-20:
                continue
            a_, b_ = A[0] * B[1] - A[1] * B[0], C[0] * D[1] - C[1] * D[0]
            P = ((a_ * (C[0] - D[0]) - (A[0] - B[0]) * b_) / den, (a_ * (C[1] - D[1]) - (A[1] - B[1]) * b_) / den)
            if mp.sqrt(P[0] ** 2 + P[1] ** 2) <= 1 + 1e-20:
                continue
            far, near = arcs[3], arcs[1]
            # P is beyond the side of the smaller of the arcs BC, DA
            if far < near:
                far, near = near, far
                orc = vang(vsub(A, P), vsub(D, P))        # then the near arc is DA and the wedge is APD
            else:
                orc = vang(vsub(B, P), vsub(C, P))
            args = (kind, far, near)
        out.append((args, str(orc), 'mpmath: the angle measured between vectors from points placed on the unit circle at the given arcs'))
    return out


@check('circle_angle')
def c_circle_angle(args, res, orc):
    return close(MF(res), mp.mpf(orc), mp.mpf(10) ** -30)


# --- polygons

def rand_points(rng, n, lo=-6, hi=6):
    return [ip(rng, 2, lo, hi) if rng.random() < 0.7 else rp(rng, 2, lo, hi, 0.5) for _ in range(n)]


@gen('convex_hull')
def g_hull(rng, k):
    out = []
    while len(out) < k:
        pts = rand_points(rng, rng.randint(1, 12))
        if rng.random() < 0.2:                     # collinear set
            d = ip(rng, 2, -2, 2)
            pts = [on_line(rng, ip(rng), d) for _ in range(rng.randint(2, 5))]
        h = convex_hull(*[SPt(p) for p in pts])
        if isinstance(h, Polygon):
            orc = sorted(FrPt(v) for v in h.vertices)
        elif isinstance(h, Segment):
            orc = sorted([FrPt(h.p1), FrPt(h.p2)])
        else:
            orc = [FrPt(h)]
        out.append(((tuple(pts),), orc, 'sympy convex_hull'))
    return out


@check('convex_hull')
def c_hull(args, res, orc):
    if sorted(res) != orc:
        return False
    if len(res) >= 3:
        n = len(res)
        return sum(res[i][0] * res[(i + 1) % n][1] - res[i][1] * res[(i + 1) % n][0] for i in range(n)) > 0 and res[0] == min(res)
    return True


def rand_poly(rng, kmax=8):
    pts, c, poly = star_poly(rng, 3, kmax)
    if rng.random() < 0.5:
        pts = pts[::-1]
    return pts, c, poly


def sympy_signed_area(pts):
    return Fr(spoly(pts).area)


@gen('polygon_signed_area')
def g_psa(rng, k):
    out = []
    for _ in range(k):
        pts, c, poly = rand_poly(rng)
        out.append(((pts,), sympy_signed_area(pts), 'sympy Polygon.area (signed)'))
    return out


@gen('polygon_area')
def g_pa(rng, k):
    out = []
    for _ in range(k):
        pts, c, poly = rand_poly(rng)
        out.append(((pts,), abs(sympy_signed_area(pts)), 'sympy Polygon.area'))
    return out


def radical_atoms(expr):
    return {a.base for a in expr.atoms(sp.Pow) if a.exp == sp.Rational(1, 2)}


@gen('polygon_perimeter_terms')
def g_pter(rng, k):
    out = []
    for _ in range(k):
        pts, c, poly = rand_poly(rng, 6)
        out.append(((pts,), str(sp.N(poly.perimeter, 55)), 'sympy Polygon.perimeter at 55 digits'))
    return out


@check('polygon_perimeter_terms')
def c_pter(args, res, orc):
    ds = [d for d, _c in res]
    sqfree = all(d == 1 or (d > 1 and all(d % (p * p) for p in range(2, int(d ** 0.5) + 1))) for d in ds)
    return sqfree and ds == sorted(ds) and close(sum(MF(c) * mp.sqrt(d) for d, c in res), mp.mpf(orc))


@gen('polygon_perimeter')
def g_pper(rng, k):
    out = []
    for _ in range(k):
        pts, c, poly = rand_poly(rng, 5) if rng.random() < 0.7 else ([(Fraction(0), Fraction(0)), (Fraction(3), Fraction(0)), (Fraction(3), Fraction(4)), (Fraction(0), Fraction(4))], None, None)
        per = spoly(pts).perimeter
        nrad = len(radical_atoms(per))
        out.append(((pts,), REFUSE if nrad > 1 else str(sp.N(per, 55)), 'sympy Polygon.perimeter (more than one radical atom: the backend must refuse)'))
    return out


@check('polygon_perimeter')
def c_pper(args, res, orc):
    return surd_ok(res) and close(SM(res), mp.mpf(orc))


@gen('polygon_perimeter_numeric')
def g_ppn(rng, k):
    out = []
    for _ in range(k):
        pts, c, poly = rand_poly(rng, 6)
        dg = rng.choice((3, 10, 25))
        out.append(((pts, dg), (str(sp.N(poly.perimeter, 70)), dg), 'sympy Polygon.perimeter at 70 digits'))
    return out


@check('polygon_perimeter_numeric')
def c_ppn(args, res, orc):
    return isinstance(res, Decimal) and abs(mp.mpf(str(res)) - mp.mpf(orc[0])) <= mp.mpf(10) ** -orc[1] * mp.mpf('1.0001')


@gen('polygon_centroid')
def g_pcen(rng, k):
    out = []
    for _ in range(k):
        pts, c, poly = rand_poly(rng)
        out.append(((pts,), FrPt(poly.centroid), 'sympy Polygon.centroid'))
    return out


def brute_segments_meet(a, b, c, d):
    """Closed segments meet (orientation tests, written here independently of the backend)."""
    def orient(p, q, r):
        v = (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])
        return (v > 0) - (v < 0)

    def within(p, q, r):
        return min(p[0], q[0]) <= r[0] <= max(p[0], q[0]) and min(p[1], q[1]) <= r[1] <= max(p[1], q[1])
    o1, o2, o3, o4 = orient(a, b, c), orient(a, b, d), orient(c, d, a), orient(c, d, b)
    if o1 != o2 and o3 != o4:
        return True
    return (o1 == 0 and within(a, b, c)) or (o2 == 0 and within(a, b, d)) or (o3 == 0 and within(c, d, a)) or (o4 == 0 and within(c, d, b))


def brute_simple(poly):
    n = len(poly)
    if len(set(poly)) != n:
        return False
    for i in range(n):
        for j in range(i + 1, n):
            a, b, c, d = poly[i], poly[(i + 1) % n], poly[j], poly[(j + 1) % n]
            if j == i + 1 or (i == 0 and j == n - 1):
                # adjacent edges may share only the common vertex: reject when the far end of one lies on the other (a fold-back)
                shared = b if j == i + 1 else a
                far1 = a if shared == b else b
                far2 = d if shared == c else c
                if brute_segments_meet(far1, shared, far2, shared) and (
                        (far2[0] - shared[0]) * (far1[1] - shared[1]) - (far2[1] - shared[1]) * (far1[0] - shared[0]) == 0
                        and (far2[0] - shared[0]) * (far1[0] - shared[0]) + (far2[1] - shared[1]) * (far1[1] - shared[1]) > 0):
                    return False
            elif brute_segments_meet(a, b, c, d):
                return False
    return True


@gen('is_simple_polygon')
def g_isp(rng, k):
    out = []
    while len(out) < k:
        n = rng.randint(3, 7)
        pts = [ip(rng, 2, -4, 4) for _ in range(n)]
        if rng.random() < 0.4:
            pts, _c, _p = rand_poly(rng, 7)
        out.append(((pts,), brute_simple(pts), 'brute: orientation tests on every pair of edges'))
    return out


@gen('is_convex')
def g_iscx(rng, k):
    out = []
    while len(out) < k:
        mode = rng.choice(('hull', 'star', 'rand'))
        if mode == 'hull':
            pts = rand_points(rng, rng.randint(3, 9))
            h = convex_hull(*[SPt(p) for p in pts])
            if not isinstance(h, Polygon):
                continue
            pts = [FrPt(v) for v in h.vertices]
            if rng.random() < 0.5:                 # add a collinear vertex on an edge
                i = rng.randrange(len(pts))
                m = tuple((a + b) / 2 for a, b in zip(pts[i], pts[(i + 1) % len(pts)]))
                pts.insert(i + 1, m)
        elif mode == 'star':
            pts, _c, _p = rand_poly(rng, 8)
        else:
            pts = [ip(rng, 2, -4, 4) for _ in range(rng.randint(3, 6))]
        try:
            poly = spoly(pts)
            orc = bool(poly.is_convex())
        except Exception:
            orc = False                            # sympy refuses a polygon with intersecting sides: not simple, so not convex
        if not brute_simple(pts):                  # sympy drops collinear vertices first, so a fold-back like (2,2),(2,-4),(2,-1) passes it: not simple here
            orc = False
        out.append(((pts,), orc, 'sympy Polygon.is_convex (polygon construction fails for intersecting sides) and the orientation-test simplicity check below'))
    return out


@gen('point_in_polygon')
def g_pip(rng, k):
    out = []
    while len(out) < k:
        pts, c, poly = rand_poly(rng, 8)
        n = len(pts)
        mode = rng.choice(('rand', 'rand', 'vertex', 'edge'))
        if mode == 'vertex':
            pt = rng.choice(pts)
        elif mode == 'edge':
            i = rng.randrange(n)
            t = Fraction(rng.randint(1, 3), 4)
            pt = tuple(a + t * (b - a) for a, b in zip(pts[i], pts[(i + 1) % n]))
        else:
            pt = rp(rng, 2, -8, 8, 0.4)
        sp_pt = SPt(pt)
        if any(s.contains(sp_pt) for s in poly.sides):
            orc = 'boundary'
        else:
            orc = 'inside' if poly.encloses_point(sp_pt) else 'outside'
        out.append(((pt, pts), orc, 'sympy Segment.contains for the boundary, Polygon.encloses_point for the inside'))
    return out


# --- lattice

def lattice_brute_poly(pts):
    """(area2, boundary, interior) of an integer polygon by enumerating the bounding box: a point is on the boundary when it lies on an edge (exact signs), inside when the
    angles it sees the edges under add up to 2 pi (winding number by atan2, floats; the points are small integers, so the sum is 0 or +-2 pi to 1e-9)."""
    xs, ys = [int(p[0]) for p in pts], [int(p[1]) for p in pts]
    n = len(pts)
    edges = [((int(pts[i][0]), int(pts[i][1])), (int(pts[(i + 1) % n][0]), int(pts[(i + 1) % n][1]))) for i in range(n)]
    B = I = 0
    for x in range(min(xs), max(xs) + 1):
        for y in range(min(ys), max(ys) + 1):
            if any((b[0] - a[0]) * (y - a[1]) == (b[1] - a[1]) * (x - a[0]) and min(a[0], b[0]) <= x <= max(a[0], b[0]) and min(a[1], b[1]) <= y <= max(a[1], b[1])
                   for a, b in edges):
                B += 1
                continue
            tot = 0.0
            for a, b in edges:
                t1 = math.atan2(a[1] - y, a[0] - x)
                t2 = math.atan2(b[1] - y, b[0] - x)
                d = t2 - t1
                while d > math.pi:
                    d -= 2 * math.pi
                while d < -math.pi:
                    d += 2 * math.pi
                tot += d
            I += abs(tot) > math.pi
    return abs(Fr(spoly(pts).area)) * 2, B, I


@gen('lattice_polygon_stats')
def g_lps(rng, k):
    out = []
    while len(out) < k:
        pts, _c, _p = star_poly(rng, 3, 6, R=5)
        a2, B, I = lattice_brute_poly(pts)
        out.append(((pts,), {'area2': int(a2), 'boundary': B, 'interior': I, 'area': a2 / 2}, 'brute: bounding box, exact edge tests for the boundary, winding number (atan2 sum) for the inside; area by sympy'))
    return out


@gen('pick_area')
def g_pick(rng, k):
    out = []
    while len(out) < k:
        pts, _c, _p = star_poly(rng, 3, 6, R=5)
        a2, B, I = lattice_brute_poly(pts)
        out.append(((I, B), a2 / 2, 'sympy Polygon.area for a polygon whose I and B were counted by brute force'))
    return out


@gen('lattice_points_on_segment')
def g_lpos(rng, k):
    out = []
    for _ in range(k):
        p, q = ip(rng, 2, -8, 8), ip(rng, 2, -8, 8)
        n = sum(1 for x in range(int(min(p[0], q[0])), int(max(p[0], q[0])) + 1) for y in range(int(min(p[1], q[1])), int(max(p[1], q[1])) + 1)
                if (q[0] - p[0]) * (y - p[1]) == (q[1] - p[1]) * (x - p[0]))
        out.append(((p, q), n, 'brute: bounding box and exact collinearity'))
    return out


def inside_convex(pt, hull, strict):
    n = len(hull)
    for i in range(n):
        v = cross2(sub(hull[(i + 1) % n], hull[i]), sub(pt, hull[i]))
        if v < 0 or (strict and v == 0):
            return False
    return True


def brute_lattice_convex(hull):
    xs, ys = [p[0] for p in hull], [p[1] for p in hull]
    tot = inter = 0
    for x in range(math.floor(min(xs)), math.ceil(max(xs)) + 1):
        for y in range(math.floor(min(ys)), math.ceil(max(ys)) + 1):
            pt = (Fraction(x), Fraction(y))
            if inside_convex(pt, hull, False):
                tot += 1
                inter += inside_convex(pt, hull, True)
    return {'total': tot, 'interior': inter, 'boundary': tot - inter}


def ccw_hull_sympy(pts):
    h = convex_hull(*[SPt(p) for p in pts])
    if not isinstance(h, Polygon):
        return None
    v = [FrPt(x) for x in h.vertices]
    n = len(v)
    if sum(v[i][0] * v[(i + 1) % n][1] - v[i][1] * v[(i + 1) % n][0] for i in range(n)) < 0:
        v = v[::-1]
    return v


@gen('lattice_points_in_triangle')
def g_lpt(rng, k):
    out = []
    while len(out) < k:
        rational = rng.random() < 0.4
        tri = tuple(rp(rng, 2, -8, 8, 0.8) if rational else ip(rng, 2, -9, 9) for _ in range(3))
        h = ccw_hull_sympy(tri)
        if h is None or len(h) != 3:
            continue
        out.append((tri, brute_lattice_convex(h), 'brute: bounding box, exact half-plane signs'))
    return out


@gen('lattice_points_in_convex_polygon')
def g_lpcp(rng, k):
    out = []
    while len(out) < k:
        rational = rng.random() < 0.3
        pts = [rp(rng, 2, -7, 7, 0.8) if rational else ip(rng, 2, -8, 8) for _ in range(rng.randint(3, 9))]
        h = ccw_hull_sympy(pts)
        if h is None:
            continue
        # present the hull in a random rotation and orientation
        v = list(h)
        r = rng.randrange(len(v))
        v = v[r:] + v[:r]
        if rng.random() < 0.5:
            v = v[::-1]
        out.append(((v,), brute_lattice_convex(h), 'brute: bounding box, exact half-plane signs'))
    return out


@gen('lattice_points_in_circle')
def g_lpc(rng, k):
    out = []
    lit = [((100,), 317, 'literature: Gauss circle N(10) = 317'), ((10000,), 31417, 'literature: Gauss circle N(100) = 31417'),
           ((10 ** 6,), 3141549, 'literature: Gauss circle N(1000) = 3141549')]
    out.extend(lit)
    while len(out) < k:
        r2 = Fraction(rng.randint(0, 400), rng.choice((1, 1, 2, 3)))
        c = ip(rng, 2, -3, 3)
        strict = rng.random() < 0.4
        n = sum(1 for x in range(-30, 31) for y in range(-30, 31) if ((Fraction(x) - 0) ** 2 + (Fraction(y)) ** 2 < r2 if strict else Fraction(x) ** 2 + Fraction(y) ** 2 <= r2))
        out.append(((r2, c, strict), n, 'brute: every integer point of a 61 x 61 box, exact comparison (centre shift is a translation of the box)'))
    return out


# --- reflection and rotation

@gen('reflect_point_line')
def g_rpl(rng, k):
    out = []
    while len(out) < k:
        p, a, b = rp(rng, 2), rp(rng, 2), rp(rng, 2)
        if a == b:
            continue
        out.append(((p, a, b), FrPt(SPt(p).reflect(Line(SPt(a), SPt(b)))), 'sympy Point.reflect(Line)'))
    return out


@gen('reflect_points_line')
def g_rspl(rng, k):
    out = []
    while len(out) < k:
        pts, a, b = [rp(rng, 2) for _ in range(rng.randint(1, 4))], rp(rng, 2), rp(rng, 2)
        if a == b:
            continue
        L = Line(SPt(a), SPt(b))
        out.append(((pts, a, b), [FrPt(SPt(p).reflect(L)) for p in pts], 'sympy Point.reflect(Line)'))
    return out


@gen('reflect_point_point')
def g_rpp(rng, k):
    out = []
    for _ in range(k):
        dim = rng.choice((2, 3))
        p, c = rp(rng, dim), rp(rng, dim)
        out.append(((p, c), FrPt(SPt(p).scale(-1, -1, SPt(c)) if dim == 2 else Point(*[2 * S(y) - S(x) for x, y in zip(p, c)])), 'sympy Point.scale(-1, -1, about c)'))
    return out


@gen('rotate_point_90')
def g_r90(rng, k):
    out = []
    for _ in range(k):
        p, c, q = rp(rng), rp(rng), rng.randint(-5, 5)
        out.append(((p, c, q), FrPt(SPt(p).rotate(q * sp.pi / 2, SPt(c))), 'sympy Point.rotate(q pi/2, about c)'))
    return out


@gen('rotate_points_90')
def g_rs90(rng, k):
    out = []
    for _ in range(k):
        pts, c, q = [rp(rng) for _ in range(rng.randint(1, 4))], rp(rng), rng.randint(-5, 5)
        out.append(((pts, c, q), [FrPt(SPt(p).rotate(q * sp.pi / 2, SPt(c))) for p in pts], 'sympy Point.rotate(q pi/2, about c)'))
    return out


TRIPLES = [(3, 4, 5), (5, 12, 13), (8, 15, 17), (7, 24, 25), (1, 0, 1), (0, 1, 1)]


@gen('rotate_point_cs')
def g_rcs(rng, k):
    out = []
    for i in range(k):
        p, c = rp(rng), rp(rng)
        a, b, h = rng.choice(TRIPLES)
        co, si = Fraction(a * rng.choice((1, -1)), h), Fraction(b * rng.choice((1, -1)), h)
        if i % 10 == 9:
            si = si + Fraction(1, 7)
            out.append(((p, c, co, si), BADIN, 'cos^2 + sin^2 != 1'))
            continue
        m = sp.Matrix([[S(co), -S(si)], [S(si), S(co)]]) * sp.Matrix([S(p[0] - c[0]), S(p[1] - c[1])])
        out.append(((p, c, co, si), (Fr(m[0]) + c[0], Fr(m[1]) + c[1]), 'sympy Matrix product'))
    return out


@gen('rotate_point_cos')
def g_rcos(rng, k):
    out = []
    for _ in range(k):
        p, c = rp(rng, 2, -6, 6, 0.2), rp(rng, 2, -6, 6, 0.2)
        co = Fraction(rng.randint(-8, 8), rng.choice((2, 3, 5, 7, 8, 9)))
        if abs(co) > 1:
            co = Fraction(1, 3)
        dr = rng.choice((1, -1))
        th = mp.acos(MF(co)) * dr
        d = [MF(p[0] - c[0]), MF(p[1] - c[1])]
        x = MF(c[0]) + mp.cos(th) * d[0] - mp.sin(th) * d[1]
        y = MF(c[1]) + mp.sin(th) * d[0] + mp.cos(th) * d[1]
        out.append(((p, c, co, dr), (str(x), str(y)), 'mpmath rotation by dir * acos(cos) at 50 digits'))
    return out


@check('rotate_point_cos')
def c_rcos(args, res, orc):
    return all(surd_ok(v) for v in res) and close(SM(res[0]), mp.mpf(orc[0])) and close(SM(res[1]), mp.mpf(orc[1]))


# --- 3D

@gen('tetrahedron_volume')
def g_tv(rng, k):
    out = []
    for _ in range(k):
        pts = [rp(rng, 3) for _ in range(4)]
        m = sp.Matrix([[S(pts[i][c] - pts[0][c]) for c in range(3)] for i in range(1, 4)])
        out.append((tuple(pts), abs(Fr(m.det())) / 6, 'sympy Matrix.det / 6'))
    return out


def norm_plane(coeffs):
    from math import gcd
    den = 1
    for c in coeffs:
        den = den * c.denominator // gcd(den, c.denominator)
    v = [int(c * den) for c in coeffs]
    g = 0
    for x in v:
        g = gcd(g, abs(x))
    v = [x // g for x in v]
    f = next(x for x in v if x)
    return tuple(-x for x in v) if f < 0 else tuple(v)


@gen('plane_through')
def g_pt(rng, k):
    out = []
    while len(out) < k:
        p, q, r = rp(rng, 3), rp(rng, 3), rp(rng, 3)
        if Point.is_collinear(SPt(p), SPt(q), SPt(r)):
            if len(out) % 9 == 0:
                out.append(((p, q, r), BADIN, 'sympy Point.is_collinear'))
            continue
        pl = Plane(SPt(p), SPt(q), SPt(r))
        n = [Fr(c) for c in pl.normal_vector]
        d = sum(a * b for a, b in zip(n, p))
        out.append(((p, q, r), norm_plane(n + [d]), 'sympy Plane.normal_vector, then normalised here'))
    return out


@gen('dist_point_plane2')
def g_dpp(rng, k):
    out = []
    while len(out) < k:
        pt, p, q, r = rp(rng, 3), rp(rng, 3), rp(rng, 3), rp(rng, 3)
        if Point.is_collinear(SPt(p), SPt(q), SPt(r)):
            continue
        out.append(((pt, p, q, r), Fr(Plane(SPt(p), SPt(q), SPt(r)).distance(SPt(pt)) ** 2), 'sympy Plane.distance squared'))
    return out


@gen('line_plane_intersection')
def g_lpi(rng, k):
    out = []
    while len(out) < k:
        a, b, p, q, r = (rp(rng, 3, -5, 5) for _ in range(5))
        if a == b or Point.is_collinear(SPt(p), SPt(q), SPt(r)):
            continue
        if rng.random() < 0.3:                     # parallel to the plane
            n = [Fr(c) for c in Plane(SPt(p), SPt(q), SPt(r)).normal_vector]
            d = sub(q, p)
            b = tuple(x + y for x, y in zip(a, d))
        pl = Plane(SPt(p), SPt(q), SPt(r))
        res = pl.intersection(Line3D(SPt(a), SPt(b)))
        if not res:
            orc = 'parallel'
        elif isinstance(res[0], Line3D):
            orc = 'in plane'
        else:
            orc = FrPt(res[0])
        out.append(((a, b, p, q, r), orc, 'sympy Plane.intersection(Line3D)'))
    return out


@gen('skew_lines_dist2')
def g_sld(rng, k):
    out = []
    t, s = sp.symbols('t s')
    while len(out) < k:
        p1, p2, p3, p4 = (rp(rng, 3, -5, 5) for _ in range(4))
        if p1 == p2 or p3 == p4:
            continue
        d1, d2 = sub(p2, p1), sub(p4, p3)
        if all(c == 0 for c in (d1[1] * d2[2] - d1[2] * d2[1], d1[2] * d2[0] - d1[0] * d2[2], d1[0] * d2[1] - d1[1] * d2[0])):
            continue                               # parallel: not the case this oracle covers
        v = sp.Matrix([S(p1[i]) + t * S(d1[i]) - S(p3[i]) - s * S(d2[i]) for i in range(3)])
        f = (v.T * v)[0]
        sol = sp.solve([sp.diff(f, t), sp.diff(f, s)], [t, s], dict=True)[0]
        out.append(((p1, p2, p3, p4), Fr(f.subs(sol)), 'sympy: minimise |p1 + t d1 - p3 - s d2|^2 over t, s (calculus)'))
    return out


@gen('sphere_through4')
def g_s4(rng, k):
    out = []
    xs = sp.symbols('x y z')
    while len(out) < k:
        pts = [rp(rng, 3, -5, 5) for _ in range(4)]
        if Point3D.are_coplanar(*[SPt(p) for p in pts]):
            if len(out) % 9 == 0:
                out.append((tuple(pts), BADIN, 'sympy Point3D.are_coplanar'))
            continue
        eqs = [sum((xs[i] - S(p[i])) ** 2 for i in range(3)) - sum((xs[i] - S(pts[0][i])) ** 2 for i in range(3)) for p in pts[1:]]
        sol = sp.solve(eqs, xs, dict=True)[0]
        c = tuple(Fr(sol[v]) for v in xs)
        r2 = sum((a - b) ** 2 for a, b in zip(c, pts[0]))
        out.append((tuple(pts), (c, r2), 'sympy solve of the three difference equations (linear), centre then radius'))
    return out


@check('sphere_through4')
def c_s4(args, res, orc):
    c, r2 = res
    return res == orc and all(sum((x - y) ** 2 for x, y in zip(c, p)) == r2 for p in args)


# --- unions and intersections of polygons

def halfplanes(poly_ccw):
    n = len(poly_ccw)
    hp = []
    for i in range(n):
        a, b = poly_ccw[i], poly_ccw[(i + 1) % n]
        e = sub(b, a)
        # cross(e, p - a) >= 0  <=>  -e_y*x + e_x*y >= -e_y*a_x + e_x*a_y
        hp.append((-e[1], e[0], -e[1] * a[0] + e[0] * a[1]))
    return hp


def hp_inter_area(polys_ccw):
    """Area of the intersection of convex counter-clockwise polygons: every vertex is the meeting of two bounding lines that satisfies all half-planes."""
    hps = [h for p in polys_ccw for h in halfplanes(p)]
    verts = set()
    for i in range(len(hps)):
        for j in range(i + 1, len(hps)):
            a1, b1, c1 = hps[i]
            a2, b2, c2 = hps[j]
            den = a1 * b2 - a2 * b1
            if den == 0:
                continue
            x, y = (c1 * b2 - c2 * b1) / den, (a1 * c2 - a2 * c1) / den
            if all(a * x + b * y >= c for a, b, c in hps):
                verts.add((x, y))
    if len(verts) < 3:
        return Fraction(0)
    cx = sum(v[0] for v in verts) / len(verts)
    cy = sum(v[1] for v in verts) / len(verts)
    ring = sorted(verts, key=lambda v: math.atan2(float(v[1] - cy), float(v[0] - cx)))      # the vertices of a convex polygon, in angular order about an inner point
    return abs(sum(ring[i][0] * ring[(i + 1) % len(ring)][1] - ring[i][1] * ring[(i + 1) % len(ring)][0] for i in range(len(ring)))) / 2


def fan(pts, c):
    n = len(pts)
    return [[c, pts[i], pts[(i + 1) % n]] for i in range(n)]


def inter_area_stars(stars):
    """Area of the common part of star-shaped polygons (given as lists of fan triangles): the sum over tuples of triangles of the convex intersections."""
    from itertools import product
    return sum((hp_inter_area(list(tup)) for tup in product(*stars)), Fraction(0))


def reflect_star(pts, c, a, b):
    L = Line(SPt(a), SPt(b))
    rp_ = [FrPt(SPt(p).reflect(L)) for p in pts][::-1]       # reversal keeps the order counter-clockwise
    rc = FrPt(SPt(c).reflect(L))
    return rp_, rc


@gen('polygons_union_area')
def g_pua(rng, k):
    out = []
    while len(out) < k:
        mode = rng.choice(('rect', 'rect', 'lshape', 'stars', 'reflect', 'convex3'))
        if mode == 'rect':
            rects = []
            for _ in range(rng.randint(2, 4)):
                x0, y0 = rng.randint(0, 6), rng.randint(0, 6)
                w, h = rng.randint(1, 5), rng.randint(1, 5)
                rects.append([(Fraction(x0), Fraction(y0)), (Fraction(x0 + w), Fraction(y0)), (Fraction(x0 + w), Fraction(y0 + h)), (Fraction(x0), Fraction(y0 + h))])
            cells = {(i, j) for r in rects for i in range(int(r[0][0]), int(r[2][0])) for j in range(int(r[0][1]), int(r[2][1]))}
            out.append(((rects,), Fraction(len(cells)), 'brute: unit cells covered by any rectangle'))
        elif mode == 'lshape':
            a, b = rng.randint(3, 8), rng.randint(3, 8)
            c, d = rng.randint(1, a - 1), rng.randint(1, b - 1)
            L = [(0, 0), (a, 0), (a, d), (c, d), (c, b), (0, b)]
            L = [(Fraction(x), Fraction(y)) for x, y in L]
            x0, y0, w, h = rng.randint(0, 3), rng.randint(0, 3), rng.randint(2, 6), rng.randint(2, 6)
            R = [(Fraction(x0), Fraction(y0)), (Fraction(x0 + w), Fraction(y0)), (Fraction(x0 + w), Fraction(y0 + h)), (Fraction(x0), Fraction(y0 + h))]
            cells = set()
            for i in range(0, 12):
                for j in range(0, 12):
                    cp = (Fraction(2 * i + 1, 2), Fraction(2 * j + 1, 2))
                    in_l = (cp[0] < a and cp[1] < d) or (cp[0] < c and cp[1] < b)          # the L is the union of two rectangles
                    in_r = x0 < cp[0] < x0 + w and y0 < cp[1] < y0 + h
                    if in_l or in_r:
                        cells.add((i, j))
            out.append((([L, R],), Fraction(len(cells)), 'brute: unit cells whose centre is in the L-shape (two rectangles) or in the rectangle'))
        elif mode in ('stars', 'reflect'):
            pts, c, poly = star_poly(rng, 3, 6, R=5)
            if mode == 'reflect':
                a, b = ip(rng, 2, -3, 3), ip(rng, 2, -3, 3)
                if a == b:
                    continue
                pts2, c2 = reflect_star(pts, c, a, b)
                shapes = [(pts, c), (pts2, c2)]
            else:
                p2, c2, _poly2 = star_poly(rng, 3, 6, R=5)
                shapes = [(pts, c), (p2, c2)]
                if rng.random() < 0.4:
                    p3, c3, _p3 = star_poly(rng, 3, 5, R=5)
                    shapes.append((p3, c3))
            fans = [fan(p, cc) for p, cc in shapes]
            areas = [abs(sympy_signed_area(p)) for p, _cc in shapes]
            total = sum(areas, Fraction(0))
            for r in range(2, len(shapes) + 1):
                for sub_ in combinations(range(len(shapes)), r):
                    total += (-1) ** (r + 1) * inter_area_stars([fans[i] for i in sub_])
            out.append((([p for p, _cc in shapes],), total, 'brute: inclusion-exclusion over fan triangles, each convex intersection by half-plane vertex enumeration'))
        else:
            tris = []
            for _ in range(3):
                t = ccw_hull_sympy([rp(rng, 2, -5, 5, 0.3) for _ in range(5)])
                if t is not None:
                    tris.append(t)
            if len(tris) < 2:
                continue
            total = Fraction(0)
            for r in range(1, len(tris) + 1):
                for sub_ in combinations(range(len(tris)), r):
                    total += (-1) ** (r + 1) * hp_inter_area([tris[i] for i in sub_])
            out.append(((tris,), total, 'brute: inclusion-exclusion, each convex intersection by half-plane vertex enumeration'))
    return out


@gen('polygons_intersection_area')
def g_pia(rng, k):
    out = []
    while len(out) < k:
        mode = rng.choice(('rect', 'stars', 'convex'))
        if mode == 'rect':
            rects = []
            for _ in range(rng.randint(2, 3)):
                x0, y0 = rng.randint(0, 4), rng.randint(0, 4)
                w, h = rng.randint(2, 6), rng.randint(2, 6)
                rects.append([(Fraction(x0), Fraction(y0)), (Fraction(x0 + w), Fraction(y0)), (Fraction(x0 + w), Fraction(y0 + h)), (Fraction(x0), Fraction(y0 + h))])
            cells = None
            for r in rects:
                s = {(i, j) for i in range(int(r[0][0]), int(r[2][0])) for j in range(int(r[0][1]), int(r[2][1]))}
                cells = s if cells is None else cells & s
            out.append(((rects,), Fraction(len(cells)), 'brute: unit cells covered by every rectangle'))
        elif mode == 'stars':
            shapes = []
            for _ in range(rng.randint(2, 3)):
                p, c, _poly = star_poly(rng, 3, 6, R=5)
                shapes.append((p, c))
            out.append((([p for p, _c in shapes],), inter_area_stars([fan(p, c) for p, c in shapes]), 'brute: sum over fan-triangle tuples of half-plane intersections'))
        else:
            tris = [ccw_hull_sympy([rp(rng, 2, -5, 5, 0.3) for _ in range(5)]) for _ in range(rng.randint(2, 3))]
            if any(t is None for t in tris):
                continue
            out.append(((tris,), hp_inter_area(tris), 'brute: half-plane vertex enumeration'))
    return out


@gen('convex_polygon_intersection')
def g_cpi(rng, k):
    out = []
    while len(out) < k:
        a = ccw_hull_sympy([rp(rng, 2, -5, 5, 0.3) for _ in range(rng.randint(3, 7))])
        b = ccw_hull_sympy([rp(rng, 2, -5, 5, 0.3) for _ in range(rng.randint(3, 7))])
        if a is None or b is None:
            continue
        hps = halfplanes(a) + halfplanes(b)
        verts = set()
        for i in range(len(hps)):
            for j in range(i + 1, len(hps)):
                a1, b1, c1 = hps[i]
                a2, b2, c2 = hps[j]
                den = a1 * b2 - a2 * b1
                if den == 0:
                    continue
                x, y = (c1 * b2 - c2 * b1) / den, (a1 * c2 - a2 * c1) / den
                if all(p * x + q * y >= c for p, q, c in hps):
                    verts.add((x, y))
        if rng.random() < 0.5:
            a = a[::-1]
        out.append(((a, b), sorted(verts), 'brute: vertex set from pairs of bounding lines that satisfy every half-plane'))
    return out


@check('convex_polygon_intersection')
def c_cpi(args, res, orc):
    # the vertex set equals the oracle's extreme vertices (collinear points on edges are dropped by the backend)
    if not orc:
        return res == []
    h = convex_hull(*[SPt(v) for v in orc])
    if isinstance(h, Polygon):
        want = sorted(FrPt(v) for v in h.vertices)
    elif isinstance(h, Segment):
        want = sorted([FrPt(h.p1), FrPt(h.p2)])
    else:
        want = [FrPt(h)]
    return sorted(res) == want


# --- regular n-gon (brute force with coordinates; no integer direction classes)

def ngon_coords(n):
    if n <= 12:
        return [(mp.cos(2 * mp.pi * k / n), mp.sin(2 * mp.pi * k / n)) for k in range(n)], mp.mpf(10) ** -30, 'mpmath 50 digits'
    return [(math.cos(2 * math.pi * k / n), math.sin(2 * math.pi * k / n)) for k in range(n)], 1e-9, 'float64'


def cluster1(vals, tol, wrap=None):
    """Group sorted real numbers whose consecutive gaps are below tol; wrap (a period) joins the last group to the first. Returns lists of indices."""
    order = sorted(range(len(vals)), key=lambda i: vals[i])
    groups = [[order[0]]]
    for a, b in zip(order, order[1:]):
        if vals[b] - vals[a] < tol:
            groups[-1].append(b)
        else:
            groups.append([b])
    if wrap is not None and len(groups) > 1 and vals[order[0]] + wrap - vals[order[-1]] < tol:
        groups[0] = groups[-1] + groups[0]
        groups.pop()
    return groups


def ngon_dir_groups(n):
    P, tol, _name = ngon_coords(n)
    chords = list(combinations(range(n), 2))
    pi = mp.pi if n <= 12 else math.pi
    atan2 = mp.atan2 if n <= 12 else math.atan2
    ang = [atan2(P[j][1] - P[i][1], P[j][0] - P[i][0]) % pi for i, j in chords]
    groups = cluster1(ang, tol * 10, wrap=pi)
    return chords, ang, groups, pi


@gen('regular_ngon_basic')
def g_basic(rng, k):
    out = []
    for n in range(3, 3 + k):
        rpoly = RegularPolygon(Point(0, 0), 1, n)
        inter = Fr(sp.nsimplify(rpoly.interior_angle * 180 / sp.pi))
        central = Fr(sp.nsimplify((sp.pi - rpoly.interior_angle) * 180 / sp.pi))
        diags = sum(1 for i, j in combinations(range(n), 2) if (j - i) % n not in (1, n - 1))
        diam = 0
        P, tol, _nm = ngon_coords(n)
        for i, j in combinations(range(n), 2):
            if abs(P[i][0] + P[j][0]) < tol and abs(P[i][1] + P[j][1]) < tol:
                diam += 1
        out.append(((n,), {'sides': n, 'diagonals': diags, 'chords': n * (n - 1) // 2, 'interior_angle_degrees': inter, 'central_angle_degrees': central, 'diameters': diam},
                    'sympy RegularPolygon angles; diagonals and diameters by brute enumeration of coordinates'))
    return out


@gen('regular_ngon_chords_by_direction')
def g_cbd(rng, k):
    out = []
    for n in range(3, 3 + k):
        chords, ang, groups, pi = ngon_dir_groups(n)
        classes, ok = {}, True
        for g in groups:
            residues = {(chords[i][0] + chords[i][1]) % n for i in g}
            if len(residues) != 1:
                ok = False
            classes[next(iter(residues))] = len(g)
        # perpendicular pairs by angle difference pi/2
        reps = {}
        for g in groups:
            reps[(chords[g[0]][0] + chords[g[0]][1]) % n] = ang[g[0]]
        tol = 1e-8 if n > 12 else mp.mpf(10) ** -25
        perp = sorted({tuple(sorted((a, b))) for a in reps for b in reps if a < b and abs(abs(reps[a] - reps[b]) - pi / 2) < tol})
        out.append(((n,), {'classes': classes, 'perpendicular_pairs': perp, 'chords': n * (n - 1) // 2, 'consistent': ok},
                    f'brute: chord directions from {ngon_coords(n)[2]} coordinates, grouped by angle'))
    return out


@check('regular_ngon_chords_by_direction')
def c_cbd(args, res, orc):
    return orc['consistent'] and res['classes'] == orc['classes'] and sorted(res['perpendicular_pairs']) == orc['perpendicular_pairs'] and res['chords'] == orc['chords']


@gen('regular_ngon_chords_by_length')
def g_cbl(rng, k):
    out = []
    for n in range(3, 3 + k):
        P, tol, nm = ngon_coords(n)
        chords = list(combinations(range(n), 2))
        d2 = [(P[i][0] - P[j][0]) ** 2 + (P[i][1] - P[j][1]) ** 2 for i, j in chords]
        groups = cluster1(d2, tol * 10)
        res = {}
        for g in groups:
            steps = {min((chords[i][1] - chords[i][0]) % n, (chords[i][0] - chords[i][1]) % n) for i in g}
            res[next(iter(steps)) if len(steps) == 1 else -1] = len(g)
        out.append(((n,), res, f'brute: chord lengths from {nm} coordinates, grouped by length'))
    return out


def classify_triangles(n):
    P, tol, nm = ngon_coords(n)
    t = dict(total=0, acute=0, right=0, obtuse=0, isosceles=0, equilateral=0, right_isosceles=0)

    def d2(a, b):
        return (P[a][0] - P[b][0]) ** 2 + (P[a][1] - P[b][1]) ** 2

    def dot(a, b, c):
        return (P[b][0] - P[a][0]) * (P[c][0] - P[a][0]) + (P[b][1] - P[a][1]) * (P[c][1] - P[a][1])
    for i, j, k in combinations(range(n), 3):
        t['total'] += 1
        ds = [dot(i, j, k), dot(j, i, k), dot(k, i, j)]
        right = any(abs(x) < tol for x in ds)
        obtuse = any(x < -tol for x in ds)
        t['right'] += right
        t['obtuse'] += obtuse
        t['acute'] += not right and not obtuse
        a, b, c = d2(j, k), d2(i, k), d2(i, j)
        iso = abs(a - b) < tol or abs(b - c) < tol or abs(a - c) < tol
        t['isosceles'] += iso
        t['equilateral'] += abs(a - b) < tol and abs(b - c) < tol
        t['right_isosceles'] += iso and right
    t['scalene'] = t['total'] - t['isosceles']
    return t, nm


@gen('regular_ngon_triangles')
def g_tri(rng, k):
    out = []
    for n in range(3, 3 + k):
        t, nm = classify_triangles(n)
        out.append(((n,), t, f'brute: every triple of vertices from {nm} coordinates, angles by dot products'))
    return out


def classify_quads(n):
    P, tol, nm = ngon_coords(n)
    q = dict(total=0, with_parallel_pair=0, rectangles=0, squares=0, kites=0)

    def v(a, b):
        return (P[b][0] - P[a][0], P[b][1] - P[a][1])

    def d2(a, b):
        x, y = v(a, b)
        return x * x + y * y
    for i, j, k, l in combinations(range(n), 4):
        q['total'] += 1
        s1, s2, s3, s4 = v(i, j), v(j, k), v(k, l), v(l, i)
        p1, p2 = abs(cross2(s1, s3)) < tol, abs(cross2(s2, s4)) < tol
        q['with_parallel_pair'] += p1 or p2
        rect = p1 and p2
        q['rectangles'] += rect
        e = [d2(i, j), d2(j, k), d2(k, l), d2(l, i)]
        q['squares'] += rect and abs(e[0] - e[1]) < tol
        q['kites'] += (abs(e[0] - e[1]) < tol and abs(e[2] - e[3]) < tol) or (abs(e[1] - e[2]) < tol and abs(e[3] - e[0]) < tol)
    q['exactly_one_parallel_pair'] = q['with_parallel_pair'] - q['rectangles']
    return q, nm


@gen('regular_ngon_quadrilaterals')
def g_quad(rng, k):
    out = []
    for n in range(4, min(4 + k, 30)):
        q, nm = classify_quads(n)
        out.append(((n,), q, f'brute: every 4 vertices from {nm} coordinates, parallel sides by cross products'))
    return out


@gen('regular_ngon_vertex_rectangles')
def g_vrect(rng, k):
    out = []
    for n in range(3, min(3 + k, 31)):
        q, nm = classify_quads(n)
        out.append(((n,), {'rectangles': q['rectangles'], 'squares': q['squares']}, f'brute: every 4 vertices from {nm} coordinates'))
    return out


@gen('regular_ngon_angle')
def g_angle(rng, k):
    out = []
    while len(out) < k:
        n = rng.randint(3, 60)
        i, j, kk = rng.sample(range(n), 3)
        P = [(mp.cos(2 * mp.pi * t / n), mp.sin(2 * mp.pi * t / n)) for t in (i, j, kk)]
        u, v = (P[0][0] - P[1][0], P[0][1] - P[1][1]), (P[2][0] - P[1][0], P[2][1] - P[1][1])
        ang = abs(mp.atan2(u[0] * v[1] - u[1] * v[0], u[0] * v[0] + u[1] * v[1])) * 180 / mp.pi
        out.append(((n, i, j, kk), str(ang), 'mpmath: angle between the vectors from vertex j'))
    return out


@check('regular_ngon_angle')
def c_angle(args, res, orc):
    return close(MF(res), mp.mpf(orc))


@gen('regular_ngon_chord_angle')
def g_cangle(rng, k):
    out = []
    while len(out) < k:
        n = rng.randint(3, 60)
        a, b = rng.sample(range(n), 2)
        c, d = rng.sample(range(n), 2)
        P = [(mp.cos(2 * mp.pi * t / n), mp.sin(2 * mp.pi * t / n)) for t in (a, b, c, d)]
        u, v = (P[1][0] - P[0][0], P[1][1] - P[0][1]), (P[3][0] - P[2][0], P[3][1] - P[2][1])
        ang = abs(mp.atan2(u[0] * v[1] - u[1] * v[0], u[0] * v[0] + u[1] * v[1])) * 180 / mp.pi      # atan2, not acos: acos loses half the digits near 0 and 180
        ang = min(ang, 180 - ang)
        out.append(((n, (a, b), (c, d)), str(ang), 'mpmath: acute angle between the two chord vectors'))
    return out


@check('regular_ngon_chord_angle')
def c_cangle(args, res, orc):
    return close(MF(res), mp.mpf(orc), mp.mpf(10) ** -30)


@gen('regular_ngon_vertices_numeric')
def g_vn(rng, k):
    out = []
    for _ in range(k):
        n, dg = rng.randint(3, 40), rng.choice((3, 10, 30))
        rad = Fraction(rng.randint(1, 20), rng.choice((1, 2, 3)))
        orc = [(str(MF(rad) * mp.cos(2 * mp.pi * t / n)), str(MF(rad) * mp.sin(2 * mp.pi * t / n))) for t in range(n)]
        out.append(((n, dg, rad), (orc, dg), 'mpmath cos/sin at 50 digits'))
    return out


@check('regular_ngon_vertices_numeric')
def c_vn(args, res, orc):
    o, dg = orc
    return len(res) == len(o) and all(isinstance(x, Decimal) and abs(mp.mpf(str(x)) - mp.mpf(a)) <= mp.mpf(10) ** -dg * mp.mpf('1.0001') for pr, po in zip(res, o) for x, a in zip(pr, po))


@gen('regular_ngon_area_exact')
def g_ae(rng, k):
    out = []
    for i in range(k):
        n = [3, 4, 5, 6, 7, 8, 12, 10][i % 8]
        r2 = Fraction(rng.randint(1, 30), rng.choice((1, 2, 3)))
        if n in (5, 7, 10):
            out.append(((n, r2), REFUSE, 'nested radicals / cubic irrationality: no Surd'))
        else:
            out.append(((n, r2), str(n * MF(r2) / 2 * mp.sin(2 * mp.pi / n)), 'mpmath (n/2) R^2 sin(2 pi / n)'))
    return out


@check('regular_ngon_area_exact')
def c_ae(args, res, orc):
    return surd_ok(res) and close(SM(res), mp.mpf(orc))


@gen('regular_ngon_side_numeric')
def g_sn(rng, k):
    out = []
    for _ in range(k):
        n, dg = rng.randint(3, 200), rng.choice((3, 10, 30))
        rad = Fraction(rng.randint(1, 20), rng.choice((1, 2, 3)))
        out.append(((n, rad, dg), (str(2 * MF(rad) * mp.sin(mp.pi / n)), dg), 'mpmath 2 R sin(pi/n)'))
    return out


@check('regular_ngon_side_numeric')
def c_sn(args, res, orc):
    return isinstance(res, Decimal) and abs(mp.mpf(str(res)) - mp.mpf(orc[0])) <= mp.mpf(10) ** -orc[1] * mp.mpf('1.0001')


def pr_formula(n):
    """Poonen-Rubinstein (SIAM J. Discrete Math 1998): the number of interior intersection points of the diagonals of a regular n-gon."""
    d = lambda m: 1 if n % m == 0 else 0
    from math import comb
    v = Fraction(comb(n, 4))
    v += Fraction(-5 * n ** 3 + 45 * n ** 2 - 70 * n + 24, 24) * d(2)
    v -= Fraction(3 * n, 2) * d(4)
    v += Fraction(-45 * n ** 2 + 262 * n, 6) * d(6)
    v += 42 * n * d(12) + 60 * n * d(18) + 35 * n * d(24) - 38 * n * d(30) - 82 * n * d(42) - 330 * n * d(60) - 144 * n * d(84) - 96 * n * d(90) - 144 * n * d(120) - 96 * n * d(210)
    return int(v)


def brute_diag(n):
    """Points where diagonals meet, with multiplicity, regions: coordinates only (line intersections by Cramer, clustering by distance)."""
    P, tol, nm = ngon_coords(n)
    pts = []
    for i, j, k, l in combinations(range(n), 4):
        # diagonals i-k and j-l cross
        x1, y1, x2, y2 = P[i][0], P[i][1], P[k][0], P[k][1]
        x3, y3, x4, y4 = P[j][0], P[j][1], P[l][0], P[l][1]
        den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        a, b = x1 * y2 - y1 * x2, x3 * y4 - y3 * x4
        pts.append(((a * (x3 - x4) - (x1 - x2) * b) / den, (a * (y3 - y4) - (y1 - y2) * b) / den))
    if not pts:
        return {'pairs': 0, 'points': 0, 'by_multiplicity': {}, 'regions': 1}, nm
    xs = [p[0] for p in pts]
    gx = cluster1(xs, tol * 10)
    clusters = []
    for g in gx:
        ys = [pts[i][1] for i in g]
        for h in cluster1(ys, tol * 10):
            clusters.append([g[t] for t in h])
    rep = [pts[c[0]] for c in clusters]
    # regions by Euler: V = n + points, E = n + sum over diagonals (1 + distinct points on the diagonal), F inside = E - V + 1
    E = n
    for i, j in combinations(range(n), 2):
        if (j - i) % n in (1, n - 1):
            continue
        cnt = 0
        for (x, y) in rep:
            if abs((P[j][0] - P[i][0]) * (y - P[i][1]) - (P[j][1] - P[i][1]) * (x - P[i][0])) < tol * 10:
                cnt += 1
        E += 1 + cnt
    V = n + len(rep)
    sizes = sorted(len(c) for c in clusters)
    by_m = {}
    for s in sizes:
        m = (1 + math.isqrt(1 + 8 * s)) // 2
        by_m[m] = by_m.get(m, 0) + 1
    return {'pairs': math.comb(n, 4), 'points': len(rep), 'by_multiplicity': dict(sorted(by_m.items())), 'regions': E - V + 1}, nm


@gen('regular_ngon_diagonal_intersections')
def g_diag(rng, k):
    out = []
    for n in range(3, 3 + min(k, 58)):
        if n <= 22:
            orc, nm = brute_diag(n)
            out.append(((n,), orc, f'brute: line intersections from {nm} coordinates, clustered by distance; regions by Euler on the same points'))
        else:
            out.append(((n,), {'points': pr_formula(n), 'pairs': math.comb(n, 4)}, 'literature: Poonen-Rubinstein formula (validated against the brute force for n <= 22 in this run)'))
    return out


@check('regular_ngon_diagonal_intersections')
def c_diag(args, res, orc):
    return all(res[key] == val for key, val in orc.items())


def brute_rect_on_lines(n):
    """Rectangles inside the regular n-gon with sides on chords, floats: lines as (normal, offset) from coordinates, corners by Cramer, inside by the n half-planes."""
    P, tol, nm = ngon_coords(n)
    tol = 1e-9
    chords = list(combinations(range(n), 2))
    lines = []
    for i, j in chords:
        dx, dy = P[j][0] - P[i][0], P[j][1] - P[i][1]
        L = math.hypot(dx, dy)
        nx, ny = -dy / L, dx / L
        if ny < -1e-12 or (abs(ny) < 1e-12 and nx < 0):
            nx, ny = -nx, -ny
        lines.append((nx, ny, nx * P[i][0] + ny * P[i][1]))
    apothem = math.cos(math.pi / n)
    mids = [(math.cos(math.pi * (2 * t + 1) / n), math.sin(math.pi * (2 * t + 1) / n)) for t in range(n)]

    def corner(a, b):
        (n1x, n1y, c1), (n2x, n2y, c2) = lines[a], lines[b]
        den = n1x * n2y - n1y * n2x
        return ((c1 * n2y - c2 * n1y) / den, (n1x * c2 - n2x * c1) / den)

    def inside(pt):
        return all(pt[0] * m[0] + pt[1] * m[1] <= apothem + tol for m in mids)
    # parallel groups by (nx, ny)
    m = len(lines)
    perp = [[b for b in range(m) if abs(lines[a][0] * lines[b][0] + lines[a][1] * lines[b][1]) < tol] for a in range(m)]
    par = [[b for b in range(m) if b != a and abs(lines[a][0] * lines[b][1] - lines[a][1] * lines[b][0]) < tol] for a in range(m)]
    ok = {}
    for a in range(m):
        for b in perp[a]:
            ok[(a, b)] = inside(corner(a, b))
    rect = squares = 0
    for a in range(m):
        for b in par[a]:
            if b < a:
                continue
            common = [c for c in perp[a] if c in set(perp[b]) and ok[(a, c)] and ok[(b, c)]]
            for ci in range(len(common)):
                for cj in range(ci + 1, len(common)):
                    c, d = common[ci], common[cj]
                    if abs(lines[c][0] * lines[d][1] - lines[c][1] * lines[d][0]) >= tol:
                        continue               # c, d must be parallel to each other (they are both perpendicular to a)
                    rect += 1
                    w = abs(lines[a][2] - lines[b][2])
                    h = abs(lines[c][2] - lines[d][2])
                    squares += abs(w - h) < tol
    return {'rectangles': rect // 2, 'squares': squares // 2}, nm      # each rectangle is met twice: as (L pair, M pair) and as (M pair, L pair)


@gen('regular_ngon_rectangles_on_lines')
def g_rol(rng, k):
    out = []
    for n in range(3, 3 + min(k, 46)):
        if n % 2:
            out.append(((n,), {'rectangles': 0, 'squares': 0}, 'argument: no two chord directions of an odd n-gon are perpendicular'))
        elif n <= 48:
            orc, nm = brute_rect_on_lines(n)
            out.append(((n,), orc, f'brute: every pair of parallel chord lines and perpendicular pair, corners by Cramer ({nm}), inside by the n half-planes'))
        else:
            out.append(((n,), None, 'no oracle beyond n = 48'))
    out = [c for c in out if c[1] is not None]
    out.append(((12,), {'rectangles': 315}, 'literature: AIME 2024 I Problem 6 answer 315'))
    return out


# ----------------------------------------------------------------------------------------------------------------------------- runner

def generate(seed, n, only=None):
    """All cases: (op_name, args, oracle_value, oracle_name). n is the number of random cases per operation (finite-domain operations give their whole domain).
    `only` (a set of operation names) skips the other generators without changing the cases of the ones kept."""
    rng = random.Random(seed)
    cases = []
    for name, fn in GEN.items():
        sub_rng = random.Random(rng.randrange(1 << 60))
        if only and name not in only:
            continue
        for args, orc, oname in fn(sub_rng, n):
            cases.append((name, args, orc, oname))
    return cases


def agree(name, args, res, orc):
    chk = CHECK.get(name)
    if chk is not None:
        return bool(chk(args, res, orc))
    if isinstance(orc, dict) and isinstance(res, dict):
        return all(k in res and res[k] == v for k, v in orc.items())
    return res == orc


def run(seed=1, n=60, write=True, only=None):
    t0 = time.time()
    cases = generate(seed, n, only)
    ops = {name: dict(generated=0, answered=0, agree=0, refused=0, refused_expected=0, bad_input_expected=0, disagree=0, reasons={}, seconds=0.0, oracles=set(), failures=[])
           for name in registry() if name.startswith('geometry.')}
    for name, args, orc, oname in cases:
        if only and name not in only:
            continue
        st = ops[name]
        st['generated'] += 1
        st['oracles'].add(oname.split(' (')[0].split(':')[0] if False else oname)
        t = time.time()
        try:
            res = backend_run(name, *args)
            st['answered'] += 1
            if orc in (REFUSE, BADIN):
                st['disagree'] += 1
                st['failures'].append((args, repr(res)[:200], f'oracle expected {orc}'))
            elif agree(name, args, res, orc):
                st['agree'] += 1
            else:
                st['disagree'] += 1
                st['failures'].append((args, repr(res)[:300], repr(orc)[:300]))
        except NotCovered as e:
            st['refused'] += 1
            st['reasons'][str(e)] = st['reasons'].get(str(e), 0) + 1
            if orc == REFUSE:
                st['refused_expected'] += 1
        except BadInput as e:
            if orc == BADIN:
                st['bad_input_expected'] += 1
            else:
                st['disagree'] += 1
                st['failures'].append((args, f'BadInput {e}', repr(orc)[:300]))
        except Exception as e:                       # a crash is a disagreement
            st['disagree'] += 1
            st['failures'].append((args, f'{type(e).__name__}: {e}', repr(orc)[:300]))
        st['seconds'] += time.time() - t
    report = {'module': 'geometry', 'seed': seed, 'n': n, 'seconds': round(time.time() - t0, 1), 'ops': {}, 'no_case': sorted(k for k, v in ops.items() if v['generated'] == 0)}
    for k, v in sorted(ops.items()):
        v = dict(v)
        v['oracles'] = sorted(v['oracles'])[:3]
        v['seconds'] = round(v['seconds'], 2)
        v['failures'] = [[repr(a)[:200], b, c] for a, b, c in v['failures'][:5]]
        report['ops'][k] = v
    if write:
        out = ROOT / 'build-mathbackend'
        out.mkdir(exist_ok=True)
        (out / 'geometry.json').write_text(json.dumps(report, indent=1, default=str), encoding='utf-8')
    return report


def table(report):
    lines = [f"{'operation':46s} {'gen':>4s} {'ans':>4s} {'agree':>5s} {'refused':>7s} {'(exp)':>5s} {'bad(exp)':>8s} {'DISAGREE':>8s} {'sec':>6s}"]
    tot = dict(generated=0, answered=0, agree=0, refused=0, refused_expected=0, bad_input_expected=0, disagree=0)
    for k, v in report['ops'].items():
        lines.append(f"{k:46s} {v['generated']:4d} {v['answered']:4d} {v['agree']:5d} {v['refused']:7d} {v['refused_expected']:5d} {v['bad_input_expected']:8d} {v['disagree']:8d} {v['seconds']:6.2f}")
        for key in tot:
            tot[key] += v[key]
    lines.append(f"{'TOTAL':46s} {tot['generated']:4d} {tot['answered']:4d} {tot['agree']:5d} {tot['refused']:7d} {tot['refused_expected']:5d} {tot['bad_input_expected']:8d} {tot['disagree']:8d} {report['seconds']:6.1f}")
    if report['no_case']:
        lines.append('operations with no case: ' + ', '.join(report['no_case']))
    return '\n'.join(lines)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--n', type=int, default=60)
    ap.add_argument('--only', nargs='*')
    a = ap.parse_args()
    rep = run(a.seed, a.n, only=set(a.only) if a.only else None)
    print(table(rep))
    for k, v in rep['ops'].items():
        for f in v['failures']:
            print('FAIL', k, f)
        for r, c in v['reasons'].items():
            print('refused', k, c, r)
