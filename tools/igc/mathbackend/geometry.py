"""MIP math backend, geometry: points and vectors with exact coordinates, lines, circles, triangles, polygons, lattice points, reflections and rotations,
areas of unions and intersections of polygons, and the combinatorics of a regular n-gon in integer arithmetic.

Standard library only. A point is a tuple of 2 or 3 numbers (int, Fraction, Decimal and 'a/b' strings are read exactly; a float is refused, it is not an exact
number). Coordinates and lengths squared come back as `Fraction`; a length, a radius or a coordinate that needs a square root comes back as a `Surd`
(`p + q*sqrt(d)`), and an angle as exact degrees when the cosine is a simple radical, otherwise as a `Decimal` by an operation whose name ends in `_numeric`.

Nothing here is copied from another project: every function is written from the mathematics (the formula is named in its docstring), so SymPy and mpmath
stay independent oracles for `sim_geometry.py`.

Two exactness devices are used by the regular n-gon operations:
  * A sum of roots of unity with integer coefficients, v = sum c_k w^k (w = exp(2 pi i / m)), is an algebraic integer of Z[w]. All its conjugates have
    absolute value at most L = sum |c_k|, and the norm of a non-zero v is an integer, so |v| >= L^-(phi(m)-1). Evaluating v at more bits than that bound
    needs proves "v = 0" or "v != 0 and this is its sign" with no guess (`_cyclo`).
  * Counting the distinct points where diagonals meet and the rectangles whose sides lie on diagonals use that test, with a floating-point pre-filter that is
    only ever used to say "different" (its error is far below the gap it needs).

Bounds are stated in each operation's `bound`; past one the operation raises NotCovered('too large')."""
import cmath
import math
from decimal import Decimal, ROUND_HALF_EVEN, getcontext, localcontext
from fractions import Fraction
from math import comb, gcd, isqrt

from tools.igc.mathbackend import BadInput, NotCovered, Surd, op

# Operations that are deliberately not registered, each with the reason and what would cover it (read by the report and the tests).
NOT_COVERED = {
    'circle_circle_overlap_area (exact)': 'the lens area holds pi and inverse cosines of algebraic numbers, with no result form in the package: only circle_circle_overlap_area_numeric is registered',
    'inradius / incenter exact for a triangle whose sides have several radicals': 'the value lies in Q(sqrt a, sqrt b, sqrt c), not a single Surd: inradius and incenter refuse, and inradius_numeric / incenter_numeric answer',
    'angle chasing from a described diagram': 'circle_angle, regular_ngon_angle, regular_ngon_chord_angle and the coordinate angles cover the arcs-and-chords theorems; deriving the angles from a word description needs a proof engine, not a calculator',
    'intersections of piecewise curves with arcs (circle pieces)': 'polyline_intersections is exact for piecewise-linear curves; a circular piece needs Surd points per piece and sorting several radicals',
    'regular_ngon_vertices_exact': 'the exact coordinates are elements of Q(zeta_n) of degree phi(n); there is no result form for them in the package (Surd has one radical)',
    'regular_ngon_diagonal_intersections n > 60': 'one crossing pair per 4 vertices is kept in memory and each coinciding point costs one cyclotomic test; n = 60 takes seconds, n = 100 (3.9 million pairs) is left out of the bound',
    'regular_ngon_rectangles_on_lines n > 48': 'the corners grow as n^3 and the simulation checks the count only up to n = 48',
    'triangle_circumcenter_3d': 'a 3D triangle needs its plane; not written',
    'segment_intersection_3d': 'two 3D segments meet only on a measure-zero case; not written',
    'polygon_union_area with curved edges': 'circular arcs give pi and several radicals; no exact result form in the package',
    'union of non-simple polygons': 'self-intersecting inputs are read with the even-odd rule and documented as such; a winding-number rule is not offered',
}


# ----------------------------------------------------------------------------------------------------------------------------- reading numbers

def _q(x):
    if isinstance(x, bool):
        raise BadInput('a bool is no coordinate')
    if isinstance(x, (int, Fraction)):
        return Fraction(x)
    if isinstance(x, float):
        raise BadInput('a float is not exact; pass an int, a Fraction or a string such as "1/3"')
    if isinstance(x, (Decimal, str)):
        try:
            return Fraction(x)
        except (ValueError, ZeroDivisionError, ArithmeticError):
            raise BadInput(f'not a number: {x!r}')
    raise BadInput(f'not a number: {x!r}')


def _pt(p, dim=None):
    try:
        t = tuple(_q(c) for c in p)
    except TypeError:
        raise BadInput('a point is a tuple of 2 or 3 numbers')
    if len(t) not in (2, 3) or (dim is not None and len(t) != dim):
        raise BadInput('a point has 2 or 3 coordinates' if dim is None else f'a point with {dim} coordinates is needed')
    return t


def _pts(ps, lo=1, hi=5000, dim=None):
    ps = list(ps)
    if len(ps) > hi:
        raise NotCovered(f'too large: more than {hi} points')
    out = [_pt(p, dim) for p in ps]
    if len(out) < lo:
        raise BadInput(f'at least {lo} points are needed')
    if out and len({len(p) for p in out}) != 1:
        raise BadInput('points of different dimensions')
    return out


def _same_dim(*ps):
    if len({len(p) for p in ps}) != 1:
        raise BadInput('points of different dimensions')


def _int(n, lo, hi, what='n'):
    if isinstance(n, bool) or not isinstance(n, int):
        raise BadInput(f'{what} must be an integer')
    if n < lo:
        raise BadInput(f'{what} must be at least {lo}')
    if n > hi:
        raise NotCovered(f'too large: {what} > {hi}')
    return n


def _sub(p, q):
    return tuple(a - b for a, b in zip(p, q))


def _add(p, q):
    return tuple(a + b for a, b in zip(p, q))


def _scale(p, t):
    return tuple(a * t for a in p)


def _dot(u, v):
    return sum(a * b for a, b in zip(u, v))


def _cross2(u, v):
    return u[0] * v[1] - u[1] * v[0]


def _cross3(u, v):
    return (u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0])


def _cross_sq(u, v):
    """|u x v|^2 in 2D or 3D (Lagrange: |u|^2 |v|^2 - (u.v)^2)."""
    return _dot(u, u) * _dot(v, v) - _dot(u, v) ** 2


def _digits(d):
    if isinstance(d, bool) or not isinstance(d, int) or d < 0:
        raise BadInput('digits must be a non-negative integer')
    if d > 2000:
        raise NotCovered('too large: more than 2000 digits')
    return d


# ----------------------------------------------------------------------------------------------------------------------------- Decimal trigonometry

_PI = {}


def _pi(prec):
    """pi to prec digits by Machin's formula 16 atan(1/5) - 4 atan(1/239)."""
    for p, v in _PI.items():
        if p >= prec:
            return v
    with localcontext() as ctx:
        ctx.prec = prec + 12
        eps = Decimal(10) ** -(prec + 10)

        def atan_inv(k):
            x = Decimal(1) / k
            x2 = x * x
            term, s, n = x, x, 1
            while abs(term) > eps:
                term *= -x2
                n += 2
                s += term / n
            return s
        v = 16 * atan_inv(5) - 4 * atan_inv(239)
    _PI[prec] = v
    return v


def _sincos(x):
    """(sin x, cos x) at the current context precision, by the Taylor series after reduction to [-pi, pi]."""
    ctx = getcontext()
    pi = _pi(ctx.prec + 5)
    two = 2 * pi
    x = x - two * ((x / two).to_integral_value())
    if x > pi:
        x -= two
    elif x < -pi:
        x += two
    eps = Decimal(10) ** -(ctx.prec + 2)
    x2 = x * x
    s, c = x, Decimal(1)
    ts, tc = x, Decimal(1)
    n = 1
    while abs(ts) > eps or abs(tc) > eps:
        tc = -tc * x2 / ((n) * (n + 1))
        ts = -ts * x2 / ((n + 1) * (n + 2))
        n += 2
        c += tc
        s += ts
    return s, c


def _atan(x):
    """arctan of a Decimal at the current precision: halve the angle twice with x/(1+sqrt(1+x^2)), then the Taylor series."""
    ctx = getcontext()
    if x == 0:
        return Decimal(0)
    if x < 0:
        return -_atan(-x)
    if x > 1:
        return _pi(ctx.prec + 5) / 2 - _atan(1 / x)
    for _ in range(3):
        x = x / (1 + (1 + x * x).sqrt())
    eps = Decimal(10) ** -(ctx.prec + 2)
    x2 = x * x
    term, s, n = x, x, 1
    while abs(term) > eps:
        term *= -x2
        n += 2
        s += term / n
    return 8 * s


def _atan2(y, x):
    ctx = getcontext()
    pi = _pi(ctx.prec + 5)
    if x > 0:
        return _atan(y / x)
    if x < 0:
        return _atan(y / x) + (pi if y >= 0 else -pi)
    if y > 0:
        return pi / 2
    if y < 0:
        return -pi / 2
    return Decimal(0)


def _dec(f):
    return Decimal(f.numerator) / Decimal(f.denominator)


def _round(x, digits):
    with localcontext() as ctx:
        ctx.prec = max(60, digits + 40 + max(0, x.adjusted()))
        r = x.quantize(Decimal(1).scaleb(-digits), rounding=ROUND_HALF_EVEN)
        return abs(r) if r == 0 else r       # no negative zero


# ----------------------------------------------------------------------------------------------------------------------------- points, vectors, lines

@op('geometry.dist2', kind='plane geometry', bound='any size')
def dist2(p, q):
    """Squared distance between two points of the same dimension. dist2((0,0),(3,4)) -> 25"""
    p, q = _pt(p), _pt(q)
    _same_dim(p, q)
    d = _sub(p, q)
    return _dot(d, d)


@op('geometry.dist', kind='plane geometry', bound='squared distance with a square-free part found by trial division to 10^7')
def dist(p, q):
    """Distance as an exact Surd. dist((0,0),(1,1)) -> sqrt(2)"""
    return Surd.sqrt(dist2(p, q))


@op('geometry.midpoint', kind='plane geometry', bound='any size')
def midpoint(p, q):
    """Midpoint of two points. midpoint((0,0),(1,3)) -> (1/2, 3/2)"""
    p, q = _pt(p), _pt(q)
    _same_dim(p, q)
    return tuple((a + b) / 2 for a, b in zip(p, q))


@op('geometry.dot', kind='plane geometry', bound='any size')
def dot(u, v):
    """Dot product of two vectors. dot((1,2),(3,4)) -> 11"""
    u, v = _pt(u), _pt(v)
    _same_dim(u, v)
    return _dot(u, v)


@op('geometry.cross', kind='plane geometry', bound='any size')
def cross(u, v):
    """Cross product: the scalar u_x v_y - u_y v_x in 2D, the vector in 3D. cross((1,0),(0,1)) -> 1"""
    u, v = _pt(u), _pt(v)
    _same_dim(u, v)
    return _cross2(u, v) if len(u) == 2 else _cross3(u, v)


@op('geometry.collinear', kind='plane geometry', bound='any size')
def collinear(*points):
    """True when all the points (2D or 3D, at least 3) lie on one line. collinear((0,0),(1,1),(2,2)) -> True"""
    ps = _pts(points, 3)
    p0 = ps[0]
    base = next((q for q in ps[1:] if q != p0), None)
    if base is None:
        return True
    u = _sub(base, p0)
    for q in ps:
        w = _sub(q, p0)
        if _cross_sq(u, w) != 0:
            return False
    return True


@op('geometry.coplanar', kind='space geometry', bound='any size')
def coplanar(*points):
    """True when 3D points lie in one plane (at least 4). coplanar((0,0,0),(1,0,0),(0,1,0),(5,7,0)) -> True"""
    ps = _pts(points, 4, dim=3)
    p0 = ps[0]
    normal = None
    for i in range(1, len(ps)):
        for j in range(i + 1, len(ps)):
            c = _cross3(_sub(ps[i], p0), _sub(ps[j], p0))
            if any(c):
                normal = c
                break
        if normal:
            break
    if normal is None:
        return True
    return all(_dot(normal, _sub(q, p0)) == 0 for q in ps)


@op('geometry.concyclic', kind='circles', bound='any size')
def concyclic(p, q, r, s):
    """True when four plane points lie on one circle or line (the 4x4 determinant of (x, y, x^2+y^2, 1) is zero). concyclic((1,0),(0,1),(-1,0),(0,-1)) -> True"""
    pts = [_pt(t, 2) for t in (p, q, r, s)]
    rows = [[x, y, x * x + y * y, Fraction(1)] for x, y in pts]
    return _det(rows) == 0


def _det(m):
    """Determinant of a square matrix of Fractions by fraction-free elimination (small sizes)."""
    m = [list(r) for r in m]
    n = len(m)
    sign, det = 1, Fraction(1)
    for c in range(n):
        piv = next((r for r in range(c, n) if m[r][c] != 0), None)
        if piv is None:
            return Fraction(0)
        if piv != c:
            m[c], m[piv] = m[piv], m[c]
            sign = -sign
        det *= m[c][c]
        for r in range(c + 1, n):
            f = m[r][c] / m[c][c]
            if f:
                for k in range(c, n):
                    m[r][k] -= f * m[c][k]
    return sign * det


def _primitive(coeffs):
    """Scale rational coefficients to coprime integers; the first non-zero one positive."""
    den = 1
    for c in coeffs:
        den = den * c.denominator // gcd(den, c.denominator)
    ints = [int(c * den) for c in coeffs]
    g = 0
    for v in ints:
        g = gcd(g, abs(v))
    if g > 1:
        ints = [v // g for v in ints]
    first = next((v for v in ints if v), 0)
    if first < 0:
        ints = [-v for v in ints]
    return tuple(ints)


@op('geometry.line_through', kind='plane geometry', bound='any size')
def line_through(p, q):
    """The line through two plane points as integers (a, b, c) with ax + by = c, coprime, first non-zero coefficient positive. line_through((0,0),(1,1)) -> (1, -1, 0)"""
    p, q = _pt(p, 2), _pt(q, 2)
    if p == q:
        raise BadInput('the two points are the same')
    a, b = q[1] - p[1], p[0] - q[0]
    c = a * p[0] + b * p[1]
    return _primitive([a, b, c])


def _line_inter(p1, p2, p3, p4):
    r, s = _sub(p2, p1), _sub(p4, p3)
    den = _cross2(r, s)
    qp = _sub(p3, p1)
    if den == 0:
        return 'coincident' if _cross2(qp, r) == 0 else 'parallel'
    t = _cross2(qp, s) / den
    return (p1[0] + t * r[0], p1[1] + t * r[1])


@op('geometry.line_intersection', kind='plane geometry', bound='any size')
def line_intersection(p1, p2, p3, p4):
    """Intersection of line p1p2 with line p3p4: a point, or the string 'parallel' or 'coincident'. line_intersection((0,0),(2,2),(0,2),(2,0)) -> (1, 1)"""
    p1, p2, p3, p4 = (_pt(t, 2) for t in (p1, p2, p3, p4))
    if p1 == p2 or p3 == p4:
        raise BadInput('a line needs two different points')
    return _line_inter(p1, p2, p3, p4)


def _on_segment(p, a, b):
    if _cross2(_sub(b, a), _sub(p, a)) != 0:
        return False
    return min(a[0], b[0]) <= p[0] <= max(a[0], b[0]) and min(a[1], b[1]) <= p[1] <= max(a[1], b[1])


def _seg_inter(a, b, c, d):
    """Points common to segments ab and cd: [] , [point], or [first, last] of a shared piece, exact."""
    if a == b:
        return [a] if _on_segment(a, c, d) else []
    if c == d:
        return [c] if _on_segment(c, a, b) else []
    r, s = _sub(b, a), _sub(d, c)
    den = _cross2(r, s)
    qp = _sub(c, a)
    if den != 0:
        t = _cross2(qp, s) / den
        u = _cross2(qp, r) / den
        if 0 <= t <= 1 and 0 <= u <= 1:
            return [(a[0] + t * r[0], a[1] + t * r[1])]
        return []
    if _cross2(qp, r) != 0:
        return []
    rr = _dot(r, r)
    t0 = _dot(qp, r) / rr
    t1 = t0 + _dot(s, r) / rr
    lo, hi = max(Fraction(0), min(t0, t1)), min(Fraction(1), max(t0, t1))
    if lo > hi:
        return []
    p_lo = (a[0] + lo * r[0], a[1] + lo * r[1])
    if lo == hi:
        return [p_lo]
    return [p_lo, (a[0] + hi * r[0], a[1] + hi * r[1])]


@op('geometry.segment_intersection', kind='plane geometry', bound='any size')
def segment_intersection(a, b, c, d):
    """Points shared by segments ab and cd: [] (disjoint), [P] (one point), or [P, Q] (an overlap from P to Q). segment_intersection((0,0),(2,2),(0,2),(2,0)) -> [(1, 1)]"""
    return _seg_inter(*(_pt(t, 2) for t in (a, b, c, d)))


@op('geometry.polyline_intersections', kind='plane geometry', bound='segments(P) * segments(Q) <= 10^6; an overlap along a segment is NotCovered')
def polyline_intersections(P, Q):
    """The distinct points where two piecewise-linear curves (lists of vertices, consecutive vertices joined) meet, sorted. An overlap along a piece (infinitely many
    points) is NotCovered. Graphs of piecewise-linear functions are polylines. polyline_intersections([(0,0),(2,2)], [(0,2),(2,0)]) -> [(1, 1)]"""
    P, Q = _pts(P, 2), _pts(Q, 2)
    if (len(P) - 1) * (len(Q) - 1) > 10 ** 6:
        raise NotCovered('too large: more than 10^6 segment pairs')
    found = set()
    for i in range(len(P) - 1):
        a, b = P[i], P[i + 1]
        for j in range(len(Q) - 1):
            c, d = Q[j], Q[j + 1]
            if max(a[0], b[0]) < min(c[0], d[0]) or max(c[0], d[0]) < min(a[0], b[0]):
                continue
            if max(a[1], b[1]) < min(c[1], d[1]) or max(c[1], d[1]) < min(a[1], b[1]):
                continue
            pts = _seg_inter(a, b, c, d)
            if len(pts) == 2:
                raise NotCovered('the curves overlap along a segment: infinitely many common points')
            found.update(pts)
    return sorted(found)


@op('geometry.dist_point_line2', kind='plane geometry', bound='any size')
def dist_point_line2(p, a, b):
    """Squared distance from point p to the line through a and b (2D or 3D) = |(b-a) x (p-a)|^2 / |b-a|^2. dist_point_line2((0,2),(0,0),(1,0)) -> 4"""
    p, a, b = _pt(p), _pt(a), _pt(b)
    _same_dim(p, a, b)
    if a == b:
        raise BadInput('a line needs two different points')
    u, w = _sub(b, a), _sub(p, a)
    return _cross_sq(u, w) / _dot(u, u)


@op('geometry.dist_point_line', kind='plane geometry', bound='as dist')
def dist_point_line(p, a, b):
    """Distance from a point to a line as a Surd. dist_point_line((0,1),(0,0),(1,1)) -> sqrt(2)/2 = 1/2*sqrt(2)"""
    return Surd.sqrt(dist_point_line2(p, a, b))


# ----------------------------------------------------------------------------------------------------------------------------- circles

@op('geometry.circle_through3', kind='circles', bound='any size')
def circle_through3(p, q, r):
    """Centre and squared radius of the circle through three plane points, exact: ((cx, cy), r2). Collinear points are BadInput. circle_through3((1,0),(0,1),(-1,0)) -> ((0, 0), 1)"""
    p, q, r = _pt(p, 2), _pt(q, 2), _pt(r, 2)
    b, c = _sub(q, p), _sub(r, p)
    d = 2 * _cross2(b, c)
    if d == 0:
        raise BadInput('the three points are collinear')
    b2, c2 = _dot(b, b), _dot(c, c)
    ux = (c[1] * b2 - b[1] * c2) / d
    uy = (b[0] * c2 - c[0] * b2) / d
    return ((p[0] + ux, p[1] + uy), ux * ux + uy * uy)


def _sorted_points(pts):
    """Sort Surd points exactly by x then y (all share one radicand, or are rational)."""
    def lt(a, b):
        for u, v in zip(a, b):
            if u != v:
                return u < v
        return False
    pts = list(pts)
    for i in range(1, len(pts)):
        j = i
        while j > 0 and lt(pts[j], pts[j - 1]):
            pts[j], pts[j - 1] = pts[j - 1], pts[j]
            j -= 1
    return pts


@op('geometry.circle_line_intersection', kind='circles', bound='any size; the radicand is made square-free by trial division to 10^7')
def circle_line_intersection(center, r2, p, q):
    """Points where the circle (centre, squared radius r2) meets the line through p and q, as pairs of Surd, sorted by x then y: [] (miss), one (tangent) or two.
    circle_line_intersection((0,0), 2, (0,0), (1,0)) -> [(-sqrt(2), 0), (sqrt(2), 0)]"""
    c, p, q = _pt(center, 2), _pt(p, 2), _pt(q, 2)
    r2 = _q(r2)
    if r2 < 0:
        raise BadInput('a negative squared radius')
    if p == q:
        raise BadInput('a line needs two different points')
    d = _sub(q, p)
    f = _sub(p, c)
    a = _dot(d, d)
    hb = _dot(f, d)
    disc = hb * hb - a * (_dot(f, f) - r2)       # a quarter of the quadratic's discriminant
    if disc < 0:
        return []
    mid = (p[0] - hb * d[0] / a, p[1] - hb * d[1] / a)   # the foot of the centre on the line
    if disc == 0:
        return [(Surd(mid[0]), Surd(mid[1]))]
    root = Surd.sqrt(disc)
    pts = [(Surd(mid[0]) + root * (d[0] / a) * sg, Surd(mid[1]) + root * (d[1] / a) * sg) for sg in (1, -1)]
    return _sorted_points(pts)


def _radical_line(c1, r1, c2, r2):
    """(a, b, k): the line a x + b y = k through the common points of two circles (subtract the equations)."""
    a, b = 2 * (c2[0] - c1[0]), 2 * (c2[1] - c1[1])
    k = r1 - r2 - _dot(c1, c1) + _dot(c2, c2)
    return a, b, k


@op('geometry.circle_circle_intersection', kind='circles', bound='any size; the radicand is made square-free by trial division to 10^7')
def circle_circle_intersection(c1, r1sq, c2, r2sq):
    """Points common to two circles as pairs of Surd: [], one (tangent) or two. Concentric circles are NotCovered (none, or all of them).
    circle_circle_intersection((0,0),1,(1,0),1) -> [(1/2, -1/2*sqrt(3)), (1/2, 1/2*sqrt(3))]"""
    c1, c2 = _pt(c1, 2), _pt(c2, 2)
    r1, r2 = _q(r1sq), _q(r2sq)
    if r1 < 0 or r2 < 0:
        raise BadInput('a negative squared radius')
    if c1 == c2:
        raise NotCovered('concentric circles: none or all points are common')
    a, b, k = _radical_line(c1, r1, c2, r2)
    n2 = a * a + b * b
    t = (k - (a * c1[0] + b * c1[1])) / n2           # from the centre c1 along the normal (a, b) to the line
    foot = (c1[0] + a * t, c1[1] + b * t)
    # two points on the radical line: the foot and the foot plus its direction (-b, a)
    return circle_line_intersection(c1, r1, foot, (foot[0] - b, foot[1] + a))


@op('geometry.circle_circle_relation', kind='circles', bound='any size')
def circle_circle_relation(c1, r1sq, c2, r2sq):
    """How two circles lie, decided exactly on squares: 'separate', 'external tangent', 'intersecting', 'internal tangent', 'contained', 'identical'.
    circle_circle_relation((0,0),1,(2,0),1) -> 'external tangent'"""
    c1, c2 = _pt(c1, 2), _pt(c2, 2)
    r1, r2 = _q(r1sq), _q(r2sq)
    if r1 <= 0 or r2 <= 0:
        raise BadInput('a squared radius must be positive')
    d2 = dist2(c1, c2)
    if d2 == 0 and r1 == r2:
        return 'identical'
    # compare d with R1 + R2 and |R1 - R2| through squares: (R1+R2)^2 = r1 + r2 + 2 sqrt(r1 r2)
    s = d2 - r1 - r2                              # d^2 vs (R1+R2)^2 : d > R1+R2  <=>  s > 2 sqrt(r1 r2)
    p = 4 * r1 * r2

    def cmp_sq(x):                                 # sign of x - 2 sqrt(r1 r2) -> compare x with sqrt(p)
        if x < 0:
            return -1
        return (x * x > p) - (x * x < p)
    outer = cmp_sq(s)                              # d vs R1+R2
    inner = cmp_sq(-(d2 - r1 - r2))                # (R1-R2)^2 = r1 + r2 - 2 sqrt(r1 r2) vs d^2 : -(d2 - r1 - r2) = r1 + r2 - d2
    if outer > 0:
        return 'separate'
    if outer == 0:
        return 'external tangent'
    # d < R1 + R2: compare d with |R1 - R2|:  d > |R1-R2| <=> d2 - r1 - r2 > -2 sqrt(r1 r2) <=> (r1 + r2 - d2) < 2 sqrt(r1 r2)
    if inner < 0:
        return 'intersecting'
    return 'internal tangent' if inner == 0 else 'contained'


@op('geometry.circle_circle_overlap_area_numeric', kind='circles', bound='digits <= 2000')
def circle_circle_overlap_area_numeric(c1, r1sq, c2, r2sq, digits=20):
    """Area common to two discs as a Decimal to `digits` places (numeric: it holds pi and inverse cosines). With d the distance of the centres, K = 4 d^2 r1^2 -
    (d^2 + r1^2 - r2^2)^2 (exact, rational; r1, r2 are the squared radii) and half-angles a = atan2(sqrt K, d^2 + r1^2 - r2^2), b likewise, the lens is
    R1^2 a + R2^2 b - sqrt(K)/2; disjoint discs give 0 and nested ones pi times the smaller radius squared.
    circle_circle_overlap_area_numeric((0,0),1,(1,0),1, 6) -> Decimal('1.228370')"""
    c1, c2 = _pt(c1, 2), _pt(c2, 2)
    q1, q2 = _q(r1sq), _q(r2sq)
    digits = _digits(digits)
    rel = circle_circle_relation(c1, q1, c2, q2)
    with localcontext() as ctx:
        ctx.prec = digits + 30
        if rel in ('separate', 'external tangent'):
            return _round(Decimal(0), digits)
        if rel in ('identical', 'contained', 'internal tangent'):
            return _round(_pi(ctx.prec + 5) * _dec(min(q1, q2)), digits)
        d2 = dist2(c1, c2)
        k = 4 * d2 * q1 - (d2 + q1 - q2) ** 2
        rk = _dec(k).sqrt()
        a = _atan2(rk, _dec(d2 + q1 - q2))
        b = _atan2(rk, _dec(d2 + q2 - q1))
        return _round(_dec(q1) * a + _dec(q2) * b - rk / 2, digits)


@op('geometry.power_of_point', kind='circles', bound='any size')
def power_of_point(p, center, r2):
    """Power of point p with respect to the circle: |p - centre|^2 - r^2 (negative inside). power_of_point((3,0),(0,0),1) -> 8"""
    p, c = _pt(p, 2), _pt(center, 2)
    return dist2(p, c) - _q(r2)


@op('geometry.tangent_length', kind='circles', bound='as Surd.sqrt')
def tangent_length(p, center, r2):
    """Length of the tangent from p to the circle as a Surd = sqrt(power); a point inside the circle is BadInput. tangent_length((3,0),(0,0),1) -> 2*sqrt(2)"""
    pw = power_of_point(p, center, r2)
    if pw < 0:
        raise BadInput('the point is inside the circle: no tangent')
    return Surd.sqrt(pw)


@op('geometry.circle_angle', kind='circles', bound='arcs in degrees (rational), 0 <= arc <= 360')
def circle_angle(kind, arc1, arc2=0):
    """Angle chasing on a circle, from the arcs the angle faces (degrees, exact):
    'inscribed'      angle at a point of the circle facing the arc arc1 (not containing the vertex)      = arc1 / 2
    'tangent_chord'  angle between a tangent and a chord at the point of tangency, facing arc1            = arc1 / 2
    'chords_inside'  angle between two chords that meet inside, facing arc1 (and its vertical angle arc2) = (arc1 + arc2) / 2
    'outside'        angle at a point outside between two secants or tangents, far arc arc1, near arc2   = (arc1 - arc2) / 2   (arc1 >= arc2; a tangent has near arc 0 and joins both)
    circle_angle('chords_inside', 70, 50) -> 60"""
    a1, a2 = _q(arc1), _q(arc2)
    if not (0 <= a1 <= 360 and 0 <= a2 <= 360):
        raise BadInput('an arc is between 0 and 360 degrees')
    if kind in ('inscribed', 'tangent_chord'):
        return a1 / 2
    if kind == 'chords_inside':
        if a1 + a2 > 360:
            raise BadInput('the two arcs together exceed 360 degrees')
        return (a1 + a2) / 2
    if kind == 'outside':
        if a1 < a2:
            raise BadInput('the far arc must not be smaller than the near arc')
        if a1 + a2 > 360:
            raise BadInput('the two arcs together exceed 360 degrees')
        return (a1 - a2) / 2
    raise BadInput("kind is 'inscribed', 'tangent_chord', 'chords_inside' or 'outside'")


# ----------------------------------------------------------------------------------------------------------------------------- triangles

@op('geometry.triangle_area', kind='triangles', bound='any size')
def triangle_area(p, q, r):
    """Area of a plane triangle from its vertices (half the absolute cross product). triangle_area((0,0),(4,0),(0,3)) -> 6"""
    p, q, r = _pt(p, 2), _pt(q, 2), _pt(r, 2)
    return abs(_cross2(_sub(q, p), _sub(r, p))) / 2


@op('geometry.triangle_area_sq', kind='triangles', bound='any size')
def triangle_area_sq(p, q, r):
    """Square of the area of a triangle in 2D or 3D, exact: |u x v|^2 / 4. triangle_area_sq((0,0,0),(1,0,0),(0,1,0)) -> 1/4"""
    p, q, r = _pt(p), _pt(q), _pt(r)
    _same_dim(p, q, r)
    return _cross_sq(_sub(q, p), _sub(r, p)) / 4


@op('geometry.triangle_from_sides', kind='triangles', bound='any size; rational sides')
def triangle_from_sides(a, b, c):
    """Heron's formula for rational sides: dict with area_sq = s(s-a)(s-b)(s-c), area (Surd), semiperimeter, circumradius_sq = (abc)^2/(16 area_sq),
    circumradius = abc/(4 area) and inradius = area/s (Surd). A degenerate triangle is BadInput.
    triangle_from_sides(3,4,5)['area'] -> 6, ['inradius'] -> 1"""
    a, b, c = _q(a), _q(b), _q(c)
    if min(a, b, c) <= 0 or a + b <= c or a + c <= b or b + c <= a:
        raise BadInput('the sides do not form a triangle')
    s = (a + b + c) / 2
    a2 = s * (s - a) * (s - b) * (s - c)
    area = Surd.sqrt(a2)
    return {'semiperimeter': s, 'area_sq': a2, 'area': area, 'circumradius_sq': (a * b * c) ** 2 / (16 * a2),
            'circumradius': (a * b * c) / (4 * area), 'inradius': area / s}


@op('geometry.triangle_from_sides_sq', kind='triangles', bound='any size')
def triangle_from_sides_sq(a2, b2, c2):
    """From the squares of the sides (so sides such as sqrt(2) are allowed): 16*area^2 = 2(a2 b2 + b2 c2 + c2 a2) - (a2^2 + b2^2 + c2^2); returns
    dict(area_sq, circumradius_sq = a2 b2 c2 / (16 area_sq)). A degenerate triangle is BadInput. triangle_from_sides_sq(1,1,2)['area_sq'] -> 1/4; (2,2,8) is degenerate"""
    a2, b2, c2 = _q(a2), _q(b2), _q(c2)
    if min(a2, b2, c2) <= 0:
        raise BadInput('a squared side must be positive')
    k = 2 * (a2 * b2 + b2 * c2 + c2 * a2) - (a2 * a2 + b2 * b2 + c2 * c2)
    if k <= 0:
        raise BadInput('the sides do not form a triangle')
    area_sq = k / 16
    return {'area_sq': area_sq, 'circumradius_sq': a2 * b2 * c2 / k}


@op('geometry.circumradius_sq', kind='triangles', bound='any size')
def circumradius_sq(p, q, r):
    """Squared circumradius of a triangle in 2D or 3D = a^2 b^2 c^2 / (16 area^2). circumradius_sq((0,0),(4,0),(0,3)) -> 25/4"""
    p, q, r = _pt(p), _pt(q), _pt(r)
    _same_dim(p, q, r)
    k = 4 * _cross_sq(_sub(q, p), _sub(r, p))      # 16 area^2
    if k == 0:
        raise BadInput('collinear points')
    return dist2(q, r) * dist2(p, r) * dist2(p, q) / k


@op('geometry.inradius', kind='triangles', bound='needs sides with one common radical, else NotCovered')
def inradius(p, q, r):
    """Inradius = 2 area / perimeter as a Surd (NotCovered when the perimeter has several radicals). inradius((0,0),(4,0),(0,3)) -> 1"""
    p, q, r = _pt(p), _pt(q), _pt(r)
    _same_dim(p, q, r)
    k = _cross_sq(_sub(q, p), _sub(r, p))
    if k == 0:
        raise BadInput('collinear points')
    perim = Surd.sqrt(dist2(q, r)) + Surd.sqrt(dist2(p, r)) + Surd.sqrt(dist2(p, q))
    return Surd.sqrt(k) / perim            # 2*area = sqrt(|u x v|^2)


@op('geometry.circumradius', kind='triangles', bound='as Surd.sqrt')
def circumradius(p, q, r):
    """Circumradius of a triangle in 2D or 3D as a Surd = sqrt(circumradius_sq). circumradius((0,0),(4,0),(0,3)) -> 5/2"""
    return Surd.sqrt(circumradius_sq(p, q, r))


@op('geometry.inradius_numeric', kind='triangles', bound='digits <= 2000')
def inradius_numeric(p, q, r, digits=20):
    """Inradius 2 area / perimeter of a triangle in 2D or 3D as a Decimal to `digits` places (numeric); the exact `inradius` refuses when the sides have several
    radicals. inradius_numeric((0,0),(1,1),(4,0), 6) -> Decimal('0.466391')"""
    p, q, r = _pt(p), _pt(q), _pt(r)
    _same_dim(p, q, r)
    digits = _digits(digits)
    k = _cross_sq(_sub(q, p), _sub(r, p))
    if k == 0:
        raise BadInput('collinear points')
    with localcontext() as ctx:
        ctx.prec = digits + 30
        per = sum((_dec(d2).sqrt() for d2 in (dist2(q, r), dist2(p, r), dist2(p, q))), Decimal(0))
        return _round(_dec(k).sqrt() / per, digits)


@op('geometry.incenter_numeric', kind='triangles', bound='digits <= 2000')
def incenter_numeric(p, q, r, digits=20):
    """Incentre of a plane triangle as a pair of Decimals to `digits` places (numeric), for the cases `incenter` refuses (several radicals).
    incenter_numeric((0,0),(4,0),(0,3), 3) -> (Decimal('1.000'), Decimal('1.000'))"""
    p, q, r = _pt(p, 2), _pt(q, 2), _pt(r, 2)
    digits = _digits(digits)
    if _cross2(_sub(q, p), _sub(r, p)) == 0:
        raise BadInput('collinear points')
    with localcontext() as ctx:
        ctx.prec = digits + 30
        a, b, c = _dec(dist2(q, r)).sqrt(), _dec(dist2(p, r)).sqrt(), _dec(dist2(p, q)).sqrt()
        tot = a + b + c
        return tuple(_round((a * _dec(p[i]) + b * _dec(q[i]) + c * _dec(r[i])) / tot, digits) for i in range(2))


@op('geometry.centroid', kind='triangles', bound='any size')
def centroid(*points):
    """Mean of the points (the centroid of a triangle when given its three vertices), 2D or 3D. centroid((0,0),(3,0),(0,3)) -> (1, 1)"""
    ps = _pts(points, 1)
    n = len(ps)
    return tuple(sum(p[i] for p in ps) / n for i in range(len(ps[0])))


@op('geometry.circumcenter', kind='triangles', bound='plane triangle')
def circumcenter(p, q, r):
    """Circumcentre of a plane triangle, exact. circumcenter((0,0),(4,0),(0,3)) -> (2, 3/2)"""
    return circle_through3(p, q, r)[0]


@op('geometry.orthocenter', kind='triangles', bound='plane triangle')
def orthocenter(p, q, r):
    """Orthocentre of a plane triangle = A + B + C - 2 O (Euler line), exact. orthocenter((0,0),(4,0),(0,3)) -> (0, 0)"""
    p, q, r = _pt(p, 2), _pt(q, 2), _pt(r, 2)
    o = circle_through3(p, q, r)[0]
    return tuple(p[i] + q[i] + r[i] - 2 * o[i] for i in range(2))


@op('geometry.incenter', kind='triangles', bound='sides with one common radical, else NotCovered')
def incenter(p, q, r):
    """Incentre = (a A + b B + c C) / (a + b + c) with a = |BC| etc., as a pair of Surd. incenter((0,0),(4,0),(0,3)) -> (1, 1)"""
    p, q, r = _pt(p, 2), _pt(q, 2), _pt(r, 2)
    if _cross2(_sub(q, p), _sub(r, p)) == 0:
        raise BadInput('collinear points')
    a, b, c = Surd.sqrt(dist2(q, r)), Surd.sqrt(dist2(p, r)), Surd.sqrt(dist2(p, q))
    tot = a + b + c
    return tuple((a * p[i] + b * q[i] + c * r[i]) / tot for i in range(2))


# ----------------------------------------------------------------------------------------------------------------------------- angles

@op('geometry.angle_cos', kind='plane geometry', bound='as Surd.sqrt')
def angle_cos(u, v):
    """Cosine of the angle between two vectors as a Surd = u.v / sqrt(|u|^2 |v|^2). angle_cos((1,0),(1,1)) -> 1/2*sqrt(2)"""
    u, v = _pt(u), _pt(v)
    _same_dim(u, v)
    uu, vv = _dot(u, u), _dot(v, v)
    if uu == 0 or vv == 0:
        raise BadInput('a zero vector has no direction')
    return Surd(_dot(u, v)) / Surd.sqrt(uu * vv)


_KNOWN_COS = [(Surd(1), 0), (Surd(0, Fraction(1, 2), 3), 30), (Surd(0, Fraction(1, 2), 2), 45), (Surd(Fraction(1, 2)), 60), (Surd(0), 90),
              (Surd(Fraction(-1, 2)), 120), (Surd(0, Fraction(-1, 2), 2), 135), (Surd(0, Fraction(-1, 2), 3), 150), (Surd(-1), 180)]


@op('geometry.angle_degrees_exact', kind='plane geometry', bound='angles 0, 30, 45, 60, 90, 120, 135, 150, 180 degrees only')
def angle_degrees_exact(u, v):
    """The angle between two vectors in whole degrees when its cosine is 0, +-1, +-1/2, +-sqrt(2)/2 or +-sqrt(3)/2; NotCovered otherwise.
    angle_degrees_exact((1,0),(1,1)) -> 45"""
    c = angle_cos(u, v)
    for k, deg in _KNOWN_COS:
        if c == k:
            return deg
    raise NotCovered('the angle is not 0, 30, 45, 60, 90, 120, 135, 150 or 180 degrees')


@op('geometry.angle_degrees_numeric', kind='plane geometry', bound='digits <= 2000')
def angle_degrees_numeric(u, v, digits=20):
    """The angle between two vectors in degrees to `digits` decimal places, a Decimal (numeric): atan2(|u x v|, u.v) 180/pi.
    angle_degrees_numeric((1,0),(1,1), 5) -> Decimal('45.00000')"""
    u, v = _pt(u), _pt(v)
    _same_dim(u, v)
    digits = _digits(digits)
    uu, vv = _dot(u, u), _dot(v, v)
    if uu == 0 or vv == 0:
        raise BadInput('a zero vector has no direction')
    with localcontext() as ctx:
        ctx.prec = digits + 30
        cs = _cross_sq(u, v)
        cr = _dec(cs).sqrt()
        ang = _atan2(cr, _dec(_dot(u, v))) * 180 / _pi(ctx.prec + 5)
        return _round(ang, digits)


@op('geometry.triangle_angles_numeric', kind='triangles', bound='digits <= 2000')
def triangle_angles_numeric(p, q, r, digits=20):
    """The three angles of a triangle (at p, q, r) in degrees, Decimals (numeric), to `digits` places; they are computed separately (no angle is 180 - the others).
    triangle_angles_numeric((0,0),(1,0),(0,1), 3) -> (Decimal('90.000'), Decimal('45.000'), Decimal('45.000'))"""
    p, q, r = _pt(p), _pt(q), _pt(r)
    _same_dim(p, q, r)
    return (angle_degrees_numeric(_sub(q, p), _sub(r, p), digits), angle_degrees_numeric(_sub(p, q), _sub(r, q), digits),
            angle_degrees_numeric(_sub(p, r), _sub(q, r), digits))


# ----------------------------------------------------------------------------------------------------------------------------- polygons

def _signed2(poly):
    s = Fraction(0)
    n = len(poly)
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        s += a[0] * b[1] - a[1] * b[0]
    return s


def _hull(points):
    pts = sorted(set(points))
    if len(pts) <= 2:
        return pts

    def half(seq):
        h = []
        for p in seq:
            while len(h) >= 2 and _cross2(_sub(h[-1], h[-2]), _sub(p, h[-2])) <= 0:
                h.pop()
            h.append(p)
        return h
    lower, upper = half(pts), half(reversed(pts))
    return lower[:-1] + upper[:-1]


@op('geometry.convex_hull', kind='polygons', bound='5000 points')
def convex_hull(points):
    """Convex hull of plane points (Andrew's monotone chain, exact): the extreme vertices counter-clockwise from the lowest-leftmost, no collinear point kept.
    convex_hull([(0,0),(2,0),(1,1),(2,2),(0,2)]) -> [(0, 0), (2, 0), (2, 2), (0, 2)]"""
    return _hull(_pts(points, 1, dim=2))


@op('geometry.polygon_signed_area', kind='polygons', bound='5000 vertices')
def polygon_signed_area(poly):
    """Signed area by the shoelace formula (positive for counter-clockwise order), exact. polygon_signed_area([(0,0),(1,0),(1,1),(0,1)]) -> 1"""
    return _signed2(_pts(poly, 3, dim=2)) / 2


@op('geometry.polygon_area', kind='polygons', bound='5000 vertices; a simple polygon (a self-crossing one gives the signed sum)')
def polygon_area(poly):
    """Area of a simple polygon by the shoelace formula, exact. polygon_area([(0,0),(4,0),(0,3)]) -> 6"""
    return abs(_signed2(_pts(poly, 3, dim=2))) / 2


def _side_terms(poly):
    terms = {}
    n = len(poly)
    for i in range(n):
        s = Surd.sqrt(dist2(poly[i], poly[(i + 1) % n]))
        key = s.d if s.q != 0 else 1
        terms[key] = terms.get(key, Fraction(0)) + (s.q if s.q != 0 else s.p)
        if s.q != 0 and s.p != 0:
            terms[1] = terms.get(1, Fraction(0)) + s.p
    return {k: v for k, v in terms.items() if v != 0}


@op('geometry.polygon_perimeter_terms', kind='polygons', bound='5000 vertices')
def polygon_perimeter_terms(poly):
    """The perimeter as exact terms [(d, c), ...] meaning the sum of c*sqrt(d) (d = 1 is the rational part), sorted by d; this form holds any number of radicals.
    polygon_perimeter_terms([(0,0),(1,1),(2,0),(1,-3)]) -> [(2, 2), (10, 2)]"""
    poly = _pts(poly, 3, dim=2)
    return sorted(_side_terms(poly).items())


@op('geometry.polygon_perimeter', kind='polygons', bound='5000 vertices; at most one radical, else NotCovered')
def polygon_perimeter(poly):
    """Perimeter as a Surd when the sides share one radical; NotCovered('several radicals') otherwise (use polygon_perimeter_terms). polygon_perimeter([(0,0),(3,0),(3,4)]) -> 12"""
    terms = polygon_perimeter_terms(poly)
    irr = [(d, c) for d, c in terms if d != 1]
    if len(irr) > 1:
        raise NotCovered('several radicals')
    rat = next((c for d, c in terms if d == 1), Fraction(0))
    if not irr:
        return Surd(rat)
    return Surd(rat, irr[0][1], irr[0][0])


@op('geometry.polygon_perimeter_numeric', kind='polygons', bound='5000 vertices; digits <= 2000')
def polygon_perimeter_numeric(poly, digits=20):
    """Perimeter as a Decimal (numeric) to `digits` places, from the exact terms. polygon_perimeter_numeric([(0,0),(1,1),(2,0)], 4) -> Decimal('2.8284')"""
    digits = _digits(digits)
    terms = polygon_perimeter_terms(poly)
    with localcontext() as ctx:
        ctx.prec = digits + 30 + 10
        total = sum((_dec(c) * (Decimal(d).sqrt() if d != 1 else Decimal(1)) for d, c in terms), Decimal(0))
        return _round(total, digits)


@op('geometry.polygon_centroid', kind='polygons', bound='5000 vertices; simple polygon of non-zero area')
def polygon_centroid(poly):
    """Centroid (centre of mass) of a simple polygon's area, exact: sum (x_i + x_j) cross_ij / (6 A). polygon_centroid([(0,0),(2,0),(2,2),(0,2)]) -> (1, 1)"""
    poly = _pts(poly, 3, dim=2)
    a2 = _signed2(poly)
    if a2 == 0:
        raise BadInput('zero area')
    cx = cy = Fraction(0)
    n = len(poly)
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        w = a[0] * b[1] - a[1] * b[0]
        cx += (a[0] + b[0]) * w
        cy += (a[1] + b[1]) * w
    return (cx / (3 * a2), cy / (3 * a2))


def _is_simple(poly):
    n = len(poly)
    if len(set(poly)) != n:
        return False
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        for j in range(i + 1, n):
            c, d = poly[j], poly[(j + 1) % n]
            adjacent = (j == i + 1) or (i == 0 and j == n - 1)
            pts = _seg_inter(a, b, c, d)
            if adjacent:
                shared = b if j == i + 1 else a
                if pts != [shared]:
                    return False
            elif pts:
                return False
    return True


@op('geometry.is_simple_polygon', kind='polygons', bound='500 vertices (all pairs of edges)')
def is_simple_polygon(poly):
    """True when no two edges meet except consecutive ones at their shared vertex, exact. is_simple_polygon([(0,0),(2,2),(2,0),(0,2)]) -> False (a bow-tie)"""
    return _is_simple(_pts(poly, 3, 500, dim=2))


@op('geometry.is_convex', kind='polygons', bound='500 vertices')
def is_convex(poly):
    """True when the polygon is simple and all its turns have one sign (collinear vertices allowed). is_convex([(0,0),(2,0),(2,2),(0,2)]) -> True"""
    poly = _pts(poly, 3, 500, dim=2)
    if not _is_simple(poly):
        return False
    n = len(poly)
    signs = set()
    for i in range(n):
        c = _cross2(_sub(poly[(i + 1) % n], poly[i]), _sub(poly[(i + 2) % n], poly[(i + 1) % n]))
        if c:
            signs.add(c > 0)
    return len(signs) <= 1


@op('geometry.point_in_polygon', kind='polygons', bound='5000 vertices; simple polygon')
def point_in_polygon(pt, poly):
    """'inside', 'boundary' or 'outside' for a point against a simple polygon (any shape), exact: boundary by collinear-and-between, inside by a ray count with the
    half-open rule. point_in_polygon((1,1), [(0,0),(2,0),(2,2),(0,2)]) -> 'inside'"""
    pt = _pt(pt, 2)
    poly = _pts(poly, 3, dim=2)
    n = len(poly)
    for i in range(n):
        if _on_segment(pt, poly[i], poly[(i + 1) % n]):
            return 'boundary'
    inside = False
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        if (a[1] > pt[1]) != (b[1] > pt[1]):
            x = a[0] + (pt[1] - a[1]) * (b[0] - a[0]) / (b[1] - a[1])
            if pt[0] < x:
                inside = not inside
    return 'inside' if inside else 'outside'


# ----------------------------------------------------------------------------------------------------------------------------- lattice points

def _int_pts(poly):
    out = []
    for p in poly:
        if any(c.denominator != 1 for c in p):
            raise BadInput('integer vertices are needed')
        out.append((int(p[0]), int(p[1])))
    return out


@op('geometry.pick_area', kind='lattice geometry', bound='any size')
def pick_area(interior, boundary):
    """Pick's theorem: area = I + B/2 - 1 of a lattice polygon. pick_area(3, 8) -> 6"""
    if isinstance(interior, bool) or isinstance(boundary, bool) or not isinstance(interior, int) or not isinstance(boundary, int) or interior < 0 or boundary < 3:
        raise BadInput('interior >= 0 and boundary >= 3 integers')
    return Fraction(interior) + Fraction(boundary, 2) - 1


@op('geometry.lattice_points_on_segment', kind='lattice geometry', bound='any size')
def lattice_points_on_segment(p, q):
    """Number of lattice points on the closed segment between integer points = gcd(|dx|, |dy|) + 1. lattice_points_on_segment((0,0),(4,6)) -> 3"""
    (a,), (b,) = _int_pts([_pt(p, 2)]), _int_pts([_pt(q, 2)])
    return gcd(abs(a[0] - b[0]), abs(a[1] - b[1])) + 1


@op('geometry.lattice_polygon_stats', kind='lattice geometry', bound='500 vertices; integer vertices; a simple polygon (checked)')
def lattice_polygon_stats(poly):
    """Pick's theorem for a simple polygon with integer vertices: dict(area2 = twice the area, boundary = B, interior = I = (2A - B + 2)/2, area).
    lattice_polygon_stats([(0,0),(4,0),(0,4)]) -> area2 16, boundary 12, interior 3"""
    poly = _pts(poly, 3, 500, dim=2)
    ip = _int_pts(poly)
    if not _is_simple(poly):
        raise BadInput('not a simple polygon (self-crossing, repeated vertex or zero area)')
    a2 = abs(sum(ip[i][0] * ip[(i + 1) % len(ip)][1] - ip[i][1] * ip[(i + 1) % len(ip)][0] for i in range(len(ip))))
    b = sum(gcd(abs(ip[i][0] - ip[(i + 1) % len(ip)][0]), abs(ip[i][1] - ip[(i + 1) % len(ip)][1])) for i in range(len(ip)))
    i_pts = (a2 - b + 2) // 2
    return {'area2': a2, 'boundary': b, 'interior': i_pts, 'area': Fraction(a2, 2)}


def _lattice_rows(hull):
    """(closed, interior) lattice point counts of a convex polygon with rational vertices (counter-clockwise, positive area), by scanning the integer rows y = k."""
    ymin = min(p[1] for p in hull)
    ymax = max(p[1] for p in hull)
    k0, k1 = math.ceil(ymin), math.floor(ymax)
    if (k1 - k0 + 1) * len(hull) > 5 * 10 ** 6:
        raise NotCovered('too large: more than 5*10^6 row-edge steps')
    closed = interior = 0
    n = len(hull)
    for k in range(k0, k1 + 1):
        lo = hi = None
        for i in range(n):
            a, b = hull[i], hull[(i + 1) % n]
            if a[1] == b[1]:
                if a[1] == k:
                    xs = (a[0], b[0])
                else:
                    continue
            elif min(a[1], b[1]) <= k <= max(a[1], b[1]):
                xs = (a[0] + (b[0] - a[0]) * (k - a[1]) / (b[1] - a[1]),)
            else:
                continue
            for x in xs:
                lo = x if lo is None or x < lo else lo
                hi = x if hi is None or x > hi else hi
        if lo is None:
            continue
        closed += max(0, math.floor(hi) - math.ceil(lo) + 1)
        if ymin < k < ymax:
            interior += max(0, math.ceil(hi) - 1 - (math.floor(lo) + 1) + 1)
    return closed, interior


@op('geometry.lattice_points_in_convex_polygon', kind='lattice geometry', bound='rows * edges <= 5*10^6; convex polygon of positive area, rational vertices')
def lattice_points_in_convex_polygon(poly):
    """Lattice points of a convex polygon with rational vertices, by scanning integer rows with exact ceilings and floors: dict(total, interior, boundary).
    lattice_points_in_convex_polygon([(0,0),(4,0),(0,4)]) -> total 15, interior 3, boundary 12"""
    poly = _pts(poly, 3, 500, dim=2)
    hull = _hull(poly)
    if len(hull) < 3:
        raise BadInput('the polygon has zero area')
    if not is_convex(poly):
        raise NotCovered('the polygon is not convex')
    c, i = _lattice_rows(hull)
    return {'total': c, 'interior': i, 'boundary': c - i}


@op('geometry.lattice_points_in_triangle', kind='lattice geometry', bound='integer vertices: any size (Pick); rational vertices: rows <= 5*10^6')
def lattice_points_in_triangle(p, q, r):
    """Lattice points in a closed triangle: dict(total, interior, boundary). Integer vertices use Pick's theorem; rational ones scan rows exactly.
    lattice_points_in_triangle((0,0),(10,0),(0,10)) -> total 66, interior 36, boundary 30"""
    p, q, r = _pt(p, 2), _pt(q, 2), _pt(r, 2)
    if _cross2(_sub(q, p), _sub(r, p)) == 0:
        raise BadInput('collinear vertices: zero area')
    if all(c.denominator == 1 for t in (p, q, r) for c in t):
        ip = [(int(t[0]), int(t[1])) for t in (p, q, r)]
        a2 = abs((ip[1][0] - ip[0][0]) * (ip[2][1] - ip[0][1]) - (ip[1][1] - ip[0][1]) * (ip[2][0] - ip[0][0]))
        b = sum(gcd(abs(ip[i][0] - ip[(i + 1) % 3][0]), abs(ip[i][1] - ip[(i + 1) % 3][1])) for i in range(3))
        i_pts = (a2 - b + 2) // 2
        return {'total': i_pts + b, 'interior': i_pts, 'boundary': b}
    hull = _hull([p, q, r])
    c, i = _lattice_rows(hull)
    return {'total': c, 'interior': i, 'boundary': c - i}


@op('geometry.lattice_points_in_circle', kind='lattice geometry', bound='isqrt(r2) <= 2*10^6; integer centre')
def lattice_points_in_circle(r2, center=(0, 0), strict=False):
    """Number of integer points (x, y) with (x-cx)^2 + (y-cy)^2 <= r2 (< r2 when strict), for an integer centre and a rational r2 >= 0 (Gauss circle count).
    lattice_points_in_circle(25) -> 81"""
    r2 = _q(r2)
    c = _pt(center, 2)
    if r2 < 0:
        raise BadInput('a negative squared radius')
    if any(v.denominator != 1 for v in c):
        raise BadInput('the centre must be an integer point')
    fl = math.floor(r2) if not strict else math.ceil(r2) - 1      # dx^2 + dy^2 are integers: <= floor(r2), or <= ceil(r2) - 1 when strict
    if fl < 0:
        return 0
    m = isqrt(fl)
    if m > 2 * 10 ** 6:
        raise NotCovered('too large: radius above 2*10^6')
    total = 0
    for dx in range(-m, m + 1):
        total += 2 * isqrt(fl - dx * dx) + 1
    return total


# ----------------------------------------------------------------------------------------------------------------------------- reflections and rotations

@op('geometry.reflect_point_line', kind='transformations', bound='any size')
def reflect_point_line(p, a, b):
    """Reflection of a plane point in the line through a and b, exact (2 foot - p). reflect_point_line((0,2),(0,0),(1,0)) -> (0, -2)"""
    p, a, b = _pt(p, 2), _pt(a, 2), _pt(b, 2)
    if a == b:
        raise BadInput('a line needs two different points')
    d = _sub(b, a)
    t = _dot(_sub(p, a), d) / _dot(d, d)
    foot = (a[0] + t * d[0], a[1] + t * d[1])
    return (2 * foot[0] - p[0], 2 * foot[1] - p[1])


@op('geometry.reflect_points_line', kind='transformations', bound='5000 points')
def reflect_points_line(points, a, b):
    """Reflect every point of a list in the line through a and b. reflect_points_line([(0,2),(1,3)], (0,0), (1,0)) -> [(0, -2), (1, -3)]"""
    return [reflect_point_line(p, a, b) for p in _pts(points, 1, dim=2)]


@op('geometry.reflect_point_point', kind='transformations', bound='any size')
def reflect_point_point(p, c):
    """Reflection of p in the point c (2c - p), 2D or 3D. reflect_point_point((1,2),(0,0)) -> (-1, -2)"""
    p, c = _pt(p), _pt(c)
    _same_dim(p, c)
    return tuple(2 * y - x for x, y in zip(p, c))


@op('geometry.rotate_point_90', kind='transformations', bound='any size')
def rotate_point_90(p, center, quarter_turns):
    """Rotation of p about the centre by quarter_turns * 90 degrees counter-clockwise, exact. rotate_point_90((1,0),(0,0),1) -> (0, 1)"""
    p, c = _pt(p, 2), _pt(center, 2)
    if isinstance(quarter_turns, bool) or not isinstance(quarter_turns, int):
        raise BadInput('quarter_turns must be an integer')
    d = _sub(p, c)
    for _ in range(quarter_turns % 4):
        d = (-d[1], d[0])
    return (c[0] + d[0], c[1] + d[1])


@op('geometry.rotate_points_90', kind='transformations', bound='5000 points')
def rotate_points_90(points, center, quarter_turns):
    """Rotate every point of a list about the centre by quarter_turns * 90 degrees. rotate_points_90([(1,0),(2,0)], (0,0), 2) -> [(-1, 0), (-2, 0)]"""
    return [rotate_point_90(p, center, quarter_turns) for p in _pts(points, 1, dim=2)]


@op('geometry.rotate_point_cs', kind='transformations', bound='any size')
def rotate_point_cs(p, center, cos, sin):
    """Rotation about the centre by an angle with rational cosine and sine (cos^2 + sin^2 = 1 is checked), exact. rotate_point_cs((5,0),(0,0),'3/5','4/5') -> (3, 4)"""
    p, c = _pt(p, 2), _pt(center, 2)
    co, si = _q(cos), _q(sin)
    if co * co + si * si != 1:
        raise BadInput('cos^2 + sin^2 must be 1')
    d = _sub(p, c)
    return (c[0] + co * d[0] - si * d[1], c[1] + si * d[0] + co * d[1])


@op('geometry.rotate_point_cos', kind='transformations', bound='as Surd.sqrt')
def rotate_point_cos(p, center, cos, direction=1):
    """Rotation by the angle whose rational cosine is `cos` (counter-clockwise when direction = 1, clockwise when -1); the sine is sqrt(1 - cos^2), so the result is a
    pair of Surd. rotate_point_cos((2,0),(0,0),'1/2') -> (1, sqrt(3))"""
    p, c = _pt(p, 2), _pt(center, 2)
    co = _q(cos)
    if abs(co) > 1:
        raise BadInput('a cosine is between -1 and 1')
    if direction not in (1, -1):
        raise BadInput('direction is 1 or -1')
    si = Surd.sqrt(1 - co * co) * direction
    d = _sub(p, c)
    return (Surd(c[0] + co * d[0]) - si * d[1], Surd(c[1] + co * d[1]) + si * d[0])


# ----------------------------------------------------------------------------------------------------------------------------- 3D

@op('geometry.tetrahedron_volume', kind='space geometry', bound='any size')
def tetrahedron_volume(p, q, r, s):
    """Volume of the tetrahedron pqrs = |det(q-p, r-p, s-p)| / 6, exact. tetrahedron_volume((0,0,0),(1,0,0),(0,1,0),(0,0,1)) -> 1/6"""
    p, q, r, s = (_pt(t, 3) for t in (p, q, r, s))
    u, v, w = _sub(q, p), _sub(r, p), _sub(s, p)
    return abs(_dot(u, _cross3(v, w))) / 6


@op('geometry.plane_through', kind='space geometry', bound='any size')
def plane_through(p, q, r):
    """The plane through three non-collinear 3D points as integers (a, b, c, d) with ax + by + cz = d, coprime, first non-zero coefficient positive.
    plane_through((1,0,0),(0,1,0),(0,0,1)) -> (1, 1, 1, 1)"""
    p, q, r = _pt(p, 3), _pt(q, 3), _pt(r, 3)
    n = _cross3(_sub(q, p), _sub(r, p))
    if not any(n):
        raise BadInput('collinear points')
    return _primitive(list(n) + [_dot(n, p)])


@op('geometry.dist_point_plane2', kind='space geometry', bound='any size')
def dist_point_plane2(pt, p, q, r):
    """Squared distance from a 3D point to the plane through p, q, r = (n.(x-p))^2 / n.n. dist_point_plane2((0,0,3),(0,0,0),(1,0,0),(0,1,0)) -> 9"""
    pt, p, q, r = (_pt(t, 3) for t in (pt, p, q, r))
    n = _cross3(_sub(q, p), _sub(r, p))
    if not any(n):
        raise BadInput('collinear points')
    return _dot(n, _sub(pt, p)) ** 2 / _dot(n, n)


@op('geometry.line_plane_intersection', kind='space geometry', bound='any size')
def line_plane_intersection(a, b, p, q, r):
    """Where the line through a and b meets the plane through p, q, r: a point, or 'parallel' or 'in plane'. line_plane_intersection((0,0,1),(0,0,2),(0,0,0),(1,0,0),(0,1,0)) -> (0, 0, 0)"""
    a, b, p, q, r = (_pt(t, 3) for t in (a, b, p, q, r))
    if a == b:
        raise BadInput('a line needs two different points')
    n = _cross3(_sub(q, p), _sub(r, p))
    if not any(n):
        raise BadInput('collinear points')
    d = _sub(b, a)
    den = _dot(n, d)
    num = _dot(n, _sub(p, a))
    if den == 0:
        return 'in plane' if num == 0 else 'parallel'
    t = num / den
    return tuple(a[i] + t * d[i] for i in range(3))


@op('geometry.skew_lines_dist2', kind='space geometry', bound='any size')
def skew_lines_dist2(p1, p2, p3, p4):
    """Squared distance between the lines p1p2 and p3p4 in 3D = ((p3-p1).(d1 x d2))^2 / |d1 x d2|^2 (parallel lines: point-line distance). skew_lines_dist2((0,0,0),(1,0,0),(0,0,1),(0,1,1)) -> 1"""
    p1, p2, p3, p4 = (_pt(t, 3) for t in (p1, p2, p3, p4))
    if p1 == p2 or p3 == p4:
        raise BadInput('a line needs two different points')
    d1, d2 = _sub(p2, p1), _sub(p4, p3)
    n = _cross3(d1, d2)
    if not any(n):
        return dist_point_line2(p3, p1, p2)
    return _dot(_sub(p3, p1), n) ** 2 / _dot(n, n)


@op('geometry.sphere_through4', kind='space geometry', bound='any size')
def sphere_through4(p, q, r, s):
    """Centre and squared radius of the sphere through four non-coplanar 3D points, exact (3x3 Cramer): ((x, y, z), r2). sphere_through4((1,0,0),(-1,0,0),(0,1,0),(0,0,1)) -> ((0, 0, 0), 1)"""
    p, q, r, s = (_pt(t, 3) for t in (p, q, r, s))
    rows = [[2 * (t[i] - p[i]) for i in range(3)] for t in (q, r, s)]
    rhs = [_dot(t, t) - _dot(p, p) for t in (q, r, s)]
    d = _det(rows)
    if d == 0:
        raise BadInput('the four points are coplanar')
    sol = []
    for col in range(3):
        m = [list(row) for row in rows]
        for k in range(3):
            m[k][col] = rhs[k]
        sol.append(_det(m) / d)
    c = tuple(sol)
    return (c, dist2(c, p))


# ----------------------------------------------------------------------------------------------------------------------------- union and intersection of polygons

def _ccw(poly):
    return poly if _signed2(poly) > 0 else poly[::-1]


def _clip(subject, clip):
    """Sutherland-Hodgman: the convex polygon `subject` cut by the convex counter-clockwise polygon `clip`, exact."""
    out = list(subject)
    n = len(clip)
    for i in range(n):
        a, b = clip[i], clip[(i + 1) % n]
        e = _sub(b, a)

        def side(p):
            return _cross2(e, _sub(p, a))
        inp, out = out, []
        if not inp:
            break
        s = inp[-1]
        for t in inp:
            st, ss = side(t), side(s)
            if st >= 0:
                if ss < 0:
                    u = ss / (ss - st)
                    out.append((s[0] + u * (t[0] - s[0]), s[1] + u * (t[1] - s[1])))
                out.append(t)
            elif ss >= 0:
                u = ss / (ss - st)
                out.append((s[0] + u * (t[0] - s[0]), s[1] + u * (t[1] - s[1])))
            s = t
    return out


@op('geometry.convex_polygon_intersection', kind='polygons', bound='500 vertices each; both convex (checked)')
def convex_polygon_intersection(P, Q):
    """The intersection of two convex polygons as a vertex list (counter-clockwise, no collinear point; [] if empty; one or two points if they only touch), exact
    (Sutherland-Hodgman). convex_polygon_intersection([(0,0),(2,0),(2,2),(0,2)], [(1,1),(3,1),(3,3),(1,3)]) -> [(1, 1), (2, 1), (2, 2), (1, 2)]"""
    P, Q = _pts(P, 3, 500, dim=2), _pts(Q, 3, 500, dim=2)
    if not is_convex(P) or not is_convex(Q):
        raise NotCovered('both polygons must be convex')
    if _signed2(P) == 0 or _signed2(Q) == 0:
        raise BadInput('zero area')
    return _hull(_clip(_ccw(P), _ccw(Q)))


def _intervals_at(edges, x):
    """The even-odd intervals of one polygon's cross-section at abscissa x (edges: (x1, y1, x2, y2) with x1 < x2)."""
    ys = sorted(y1 + (y2 - y1) * (x - x1) / (x2 - x1) for x1, y1, x2, y2 in edges if x1 < x < x2)
    return [(ys[i], ys[i + 1]) for i in range(0, len(ys) - 1, 2)]


def _union_len(ivs):
    ivs = sorted(i for l in ivs for i in l)
    total, cur = Fraction(0), None
    for lo, hi in ivs:
        if cur is None or lo > cur[1]:
            if cur is not None:
                total += cur[1] - cur[0]
            cur = [lo, hi]
        elif hi > cur[1]:
            cur[1] = hi
    return total + (cur[1] - cur[0] if cur is not None else 0)


def _inter_len(lists):
    cur = lists[0]
    for nxt in lists[1:]:
        out = []
        for a_lo, a_hi in cur:
            for b_lo, b_hi in nxt:
                lo, hi = max(a_lo, b_lo), min(a_hi, b_hi)
                if lo < hi:
                    out.append((lo, hi))
        cur = out
        if not cur:
            return Fraction(0)
    return sum((hi - lo for lo, hi in cur), Fraction(0))


def _slab_area(polys, mode):
    """Area of the union (mode 'union') or common intersection (mode 'inter') of simple polygons by vertical slabs: between consecutive abscissae of vertices and of
    edge crossings the order of the edges is fixed, so the cross-section length is linear and the midpoint gives the slab's exact area."""
    all_edges, per_poly = [], []
    xs = set()
    for poly in polys:
        es = []
        n = len(poly)
        for i in range(n):
            a, b = poly[i], poly[(i + 1) % n]
            xs.add(a[0])
            if a[0] == b[0]:
                continue
            if a[0] > b[0]:
                a, b = b, a
            es.append((a[0], a[1], b[0], b[1]))
        per_poly.append(es)
        all_edges.extend(es)
    if len(all_edges) > 300:
        raise NotCovered('too large: more than 300 edges')
    for i in range(len(all_edges)):
        x1, y1, x2, y2 = all_edges[i]
        for j in range(i + 1, len(all_edges)):
            u1, v1, u2, v2 = all_edges[j]
            if x2 < u1 or u2 < x1:
                continue
            for pt in _seg_inter((x1, y1), (x2, y2), (u1, v1), (u2, v2)):
                xs.add(pt[0])
    xs = sorted(xs)
    area = Fraction(0)
    for k in range(len(xs) - 1):
        xl, xr = xs[k], xs[k + 1]
        xm = (xl + xr) / 2
        lists = [_intervals_at(es, xm) for es in per_poly]
        length = _union_len(lists) if mode == 'union' else _inter_len(lists)
        area += length * (xr - xl)
    return area


@op('geometry.polygons_union_area', kind='polygons', bound='300 edges in all; simple polygons (a self-crossing one is read with the even-odd rule)')
def polygons_union_area(polygons):
    """Exact area of the union of simple polygons (convex or not, rational vertices) by vertical slabs. A reflected copy of a polygon is one more polygon in the list.
    polygons_union_area([[(0,0),(2,0),(2,2),(0,2)], [(1,1),(3,1),(3,3),(1,3)]]) -> 7"""
    polys = [_pts(p, 3, 300, dim=2) for p in polygons]
    if not polys:
        raise BadInput('no polygons')
    for p in polys:
        if len(p) <= 120 and not _is_simple(p):
            raise BadInput('a polygon is not simple')
    return _slab_area(polys, 'union')


@op('geometry.polygons_intersection_area', kind='polygons', bound='300 edges in all; simple polygons')
def polygons_intersection_area(polygons):
    """Exact area of the region common to all the given simple polygons (convex or not) by vertical slabs.
    polygons_intersection_area([[(0,0),(2,0),(2,2),(0,2)], [(1,1),(3,1),(3,3),(1,3)]]) -> 1"""
    polys = [_pts(p, 3, 300, dim=2) for p in polygons]
    if len(polys) < 2:
        raise BadInput('at least two polygons are needed')
    for p in polys:
        if len(p) <= 120 and not _is_simple(p):
            raise BadInput('a polygon is not simple')
    return _slab_area(polys, 'inter')


# ----------------------------------------------------------------------------------------------------------------------------- cyclotomic exactness

def _totient(m):
    r, x, p = m, m, 2
    while p * p <= x:
        if x % p == 0:
            while x % p == 0:
                x //= p
            r -= r // p
        p += 1
    if x > 1:
        r -= r // x
    return r


_ROOTS = {}


def _roots(m, P):
    """exp(2 pi i k / m) for k in range(m) as integer pairs scaled by 2^P (each within 1 unit)."""
    key = (m, P)
    got = _ROOTS.get(key)
    if got is not None:
        return got
    digits = int(P * 0.30103) + 30
    out = []
    with localcontext() as ctx:
        ctx.prec = digits + 10
        pi = _pi(digits + 20)
        scale = Decimal(1 << P)
        for k in range(m):
            s, c = _sincos(2 * pi * k / m)
            out.append((int((c * scale).to_integral_value()), int((s * scale).to_integral_value())))
    _ROOTS[key] = out
    return out


def _cyclo(terms, m):
    """Exact zero test and sign of v = sum c_k w^k, w = exp(2 pi i / m), c_k integers. Returns (is_zero, sign of Re v, sign of Im v).

    Proof: v is an algebraic integer of Z[w] and every conjugate of v has |.| <= L = sum |c_k|; if v != 0 its norm is a non-zero integer, so |v| >= L^-(phi(m)-1)
    =: 2^-B. v is computed to 2^-P with P = B + 80 (every root is within one unit of 2^-P, so the error is below L units); |Re|, |Im| both below 2^(P-B-2) units
    means |v| < 2^-B, hence v = 0; otherwise one of them exceeds 2^(P-B-2) units, v != 0, and the sign of a component larger than L units is certain."""
    acc = {}
    for k, c in terms.items():
        k %= m
        acc[k] = acc.get(k, 0) + c
    acc = {k: c for k, c in acc.items() if c}
    if not acc:
        return True, 0, 0
    L = sum(abs(c) for c in acc.values())
    B = (_totient(m) - 1) * (L - 1).bit_length()
    P = ((B + 80 + 63) // 64) * 64
    roots = _roots(m, P)
    re = sum(c * roots[k][0] for k, c in acc.items())
    im = sum(c * roots[k][1] for k, c in acc.items())
    thr = 1 << (P - B - 2)
    if abs(re) <= thr and abs(im) <= thr:
        return True, 0, 0
    return False, (re > L) - (re < -L), (im > L) - (im < -L)


def _pmul(a, b):
    out = {}
    for ka, ca in a.items():
        for kb, cb in b.items():
            out[ka + kb] = out.get(ka + kb, 0) + ca * cb
    return out


def _padd(a, b, sign=1):
    out = dict(a)
    for k, c in b.items():
        out[k] = out.get(k, 0) + sign * c
    return out


def _pconj(a):
    return {-k: c for k, c in a.items()}


# ----------------------------------------------------------------------------------------------------------------------------- regular n-gon

def _chord_class_counts(n):
    return {s: (n - 1) // 2 if n % 2 else (n // 2 - 1 if s % 2 == 0 else n // 2) for s in range(n)}


@op('geometry.regular_ngon_basic', kind='regular polygons', bound='3 <= n <= 10^9')
def regular_ngon_basic(n):
    """Counts of a regular n-gon: dict(sides = n, diagonals = n(n-3)/2, chords = C(n,2), interior_angle_degrees = 180(n-2)/n, central_angle_degrees = 360/n,
    diameters = n/2 or 0). regular_ngon_basic(12)['diagonals'] -> 54"""
    n = _int(n, 3, 10 ** 9)
    return {'sides': n, 'diagonals': n * (n - 3) // 2, 'chords': comb(n, 2), 'interior_angle_degrees': Fraction(180 * (n - 2), n),
            'central_angle_degrees': Fraction(360, n), 'diameters': n // 2 if n % 2 == 0 else 0}


@op('geometry.regular_ngon_chords_by_direction', kind='regular polygons', bound='3 <= n <= 100000')
def regular_ngon_chords_by_direction(n):
    """Chords (sides and diagonals) of a regular n-gon grouped by direction. The chord between vertices i and j is perpendicular to the direction of angle
    pi (i+j)/n, so chords are parallel exactly when i + j has the same residue mod n. Returns dict(classes = {residue: number of chords}, perpendicular_pairs =
    [(s, s + n/2)] for even n (none for odd n), chords). regular_ngon_chords_by_direction(12)['classes'][0] -> 5"""
    n = _int(n, 3, 100000)
    return {'classes': _chord_class_counts(n), 'perpendicular_pairs': [(s, s + n // 2) for s in range(n // 2)] if n % 2 == 0 else [], 'chords': comb(n, 2)}


@op('geometry.regular_ngon_chords_by_length', kind='regular polygons', bound='3 <= n <= 10^6')
def regular_ngon_chords_by_length(n):
    """Chords of a regular n-gon by length: {k: number} where k is the number of steps between the ends (1 = a side) and chords of equal k have equal length:
    n chords for each k < n/2 and n/2 for the diameters. regular_ngon_chords_by_length(6) -> {1: 6, 2: 6, 3: 3}"""
    n = _int(n, 3, 10 ** 6)
    return {k: (n // 2 if 2 * k == n else n) for k in range(1, n // 2 + 1)}


@op('geometry.regular_ngon_triangles', kind='regular polygons', bound='3 <= n <= 3000')
def regular_ngon_triangles(n):
    """Triangles on the vertices of a regular n-gon by shape, from the three arcs (a, b, c), a + b + c = n, between consecutive chosen vertices: an angle is
    180 arc/n degrees, so acute / right / obtuse is the largest arc against n/2, isosceles is two equal arcs. Returns dict(total, acute, right, obtuse, isosceles
    (equilateral included), equilateral, scalene, right_isosceles). regular_ngon_triangles(12) -> total 220, right 60, equilateral 4"""
    n = _int(n, 3, 3000)
    tot = acute = right = obtuse = iso = equi = right_iso = 0
    for a in range(1, n - 1):
        for b in range(1, n - a):
            c = n - a - b
            tot += 1
            m = max(a, b, c)
            if 2 * m < n:
                acute += 1
            elif 2 * m == n:
                right += 1
            else:
                obtuse += 1
            isosc = a == b or b == c or a == c
            iso += isosc
            equi += a == b == c
            right_iso += isosc and 2 * m == n
    def count(k):
        assert (n * k) % 3 == 0
        return n * k // 3
    res = {'total': count(tot), 'acute': count(acute), 'right': count(right), 'obtuse': count(obtuse), 'isosceles': count(iso), 'equilateral': count(equi),
           'right_isosceles': count(right_iso)}
    res['scalene'] = res['total'] - res['isosceles']
    return res


@op('geometry.regular_ngon_quadrilaterals', kind='regular polygons', bound='4 <= n <= 200')
def regular_ngon_quadrilaterals(n):
    """Convex quadrilaterals on the vertices of a regular n-gon by shape, from the four arcs (a, b, c, d), a+b+c+d = n, between consecutive vertices: sides
    ab and cd are parallel when b = d, sides bc and da when a = c; a cyclic parallelogram is a rectangle (a=c and b=d), a square has four equal arcs, a kite has
    a=b and c=d (or b=c and d=a). Returns dict(total, with_parallel_pair (isosceles trapezoids and rectangles), exactly_one_parallel_pair, rectangles, squares,
    kites (squares included)). regular_ngon_quadrilaterals(12) -> rectangles 15, squares 3"""
    n = _int(n, 4, 200)
    tot = par = rect = sq = kite = 0
    for a in range(1, n - 2):
        for b in range(1, n - a - 1):
            for c in range(1, n - a - b):
                d = n - a - b - c
                tot += 1
                p1, p2 = a == c, b == d
                par += p1 or p2
                rect += p1 and p2
                sq += a == b == c == d
                kite += (a == b and c == d) or (b == c and d == a)

    def count(k):
        assert (n * k) % 4 == 0
        return n * k // 4
    res = {'total': count(tot), 'with_parallel_pair': count(par), 'rectangles': count(rect), 'squares': count(sq), 'kites': count(kite)}
    res['exactly_one_parallel_pair'] = res['with_parallel_pair'] - res['rectangles']
    return res


@op('geometry.regular_ngon_vertex_rectangles', kind='regular polygons', bound='3 <= n <= 10^9')
def regular_ngon_vertex_rectangles(n):
    """Rectangles and squares whose four corners are vertices of a regular n-gon: a rectangle inscribed in the circle has two diameters as diagonals, so
    C(n/2, 2) rectangles for even n (0 for odd n) and n/4 squares when 4 divides n. regular_ngon_vertex_rectangles(12) -> {'rectangles': 15, 'squares': 3}"""
    n = _int(n, 3, 10 ** 9)
    return {'rectangles': comb(n // 2, 2) if n % 2 == 0 else 0, 'squares': n // 4 if n % 4 == 0 else 0}


@op('geometry.regular_ngon_angle', kind='regular polygons', bound='3 <= n <= 10^9')
def regular_ngon_angle(n, i, j, k):
    """The angle i-j-k at vertex j of a regular n-gon in degrees (vertices numbered 0..n-1, distinct): the inscribed angle 180 s / n where s is the number of
    sides on the arc from i to k that does not contain j. regular_ngon_angle(12, 0, 3, 6) -> 90"""
    n = _int(n, 3, 10 ** 9)
    for v in (i, j, k):
        _int(v, 0, n - 1, 'vertex')
    if len({i, j, k}) != 3:
        raise BadInput('the three vertices must be distinct')
    d1, d2 = (j - i) % n, (k - i) % n
    steps = n - d2 if d1 < d2 else d2
    return Fraction(180 * steps, n)


@op('geometry.regular_ngon_chord_angle', kind='regular polygons', bound='3 <= n <= 10^9')
def regular_ngon_chord_angle(n, chord1, chord2):
    """The acute (or right) angle in degrees between the lines of two chords (a, b) and (c, d) of a regular n-gon: the direction of chord (a, b) is 90 + 180(a+b)/n
    degrees. regular_ngon_chord_angle(12, (0,3), (0,6)) -> 45"""
    n = _int(n, 3, 10 ** 9)
    (a, b), (c, d) = chord1, chord2
    for v in (a, b, c, d):
        _int(v, 0, n - 1, 'vertex')
    if a == b or c == d:
        raise BadInput('a chord joins two different vertices')
    diff = Fraction(180 * (a + b - c - d), n) % 180
    return min(diff, 180 - diff)


@op('geometry.regular_ngon_vertices_numeric', kind='regular polygons', bound='3 <= n <= 10^4; digits <= 2000')
def regular_ngon_vertices_numeric(n, digits=20, radius=1):
    """Vertices (R cos(2 pi k/n), R sin(2 pi k/n)), k = 0..n-1, as Decimal pairs rounded to `digits` places (numeric). regular_ngon_vertices_numeric(4, 3)[1] -> (Decimal('0.000'), Decimal('1.000'))"""
    n = _int(n, 3, 10 ** 4)
    digits = _digits(digits)
    rad = _q(radius)
    out = []
    with localcontext() as ctx:
        ctx.prec = digits + 30
        pi = _pi(ctx.prec + 5)
        r = _dec(rad)
        for k in range(n):
            s, c = _sincos(2 * pi * k / n)
            out.append((_round(r * c, digits), _round(r * s, digits)))
    return out


@op('geometry.regular_ngon_area_exact', kind='regular polygons', bound='n in 3, 4, 6, 8, 12')
def regular_ngon_area_exact(n, circumradius_sq):
    """Area of the regular n-gon with the given squared circumradius R^2, exact as a Surd = (n/2) R^2 sin(2 pi / n), for n = 3, 4, 6, 8, 12 (other n have nested
    radicals: NotCovered). regular_ngon_area_exact(12, 1) -> 3"""
    n = _int(n, 3, 10 ** 9)
    r2 = _q(circumradius_sq)
    if r2 <= 0:
        raise BadInput('a squared circumradius must be positive')
    sin = {3: Surd(0, Fraction(1, 2), 3), 4: Surd(1), 6: Surd(0, Fraction(1, 2), 3), 8: Surd(0, Fraction(1, 2), 2), 12: Surd(Fraction(1, 2))}
    if n not in sin:
        raise NotCovered('the area of this n-gon needs nested radicals or a cubic irrationality')
    return sin[n] * (Fraction(n, 2) * r2)


@op('geometry.regular_ngon_side_numeric', kind='regular polygons', bound='3 <= n <= 10^9; digits <= 2000')
def regular_ngon_side_numeric(n, circumradius=1, digits=20):
    """Side length 2 R sin(pi/n) of a regular n-gon of circumradius R (rational), a Decimal (numeric). regular_ngon_side_numeric(6, 1, 3) -> Decimal('1.000')"""
    n = _int(n, 3, 10 ** 9)
    digits = _digits(digits)
    rad = _q(circumradius)
    with localcontext() as ctx:
        ctx.prec = digits + 30
        s, _c = _sincos(_pi(ctx.prec + 5) / n)
        return _round(2 * _dec(rad) * s, digits)


def _diag_points(n):
    """Group the C(n,4) crossing pairs of diagonals of a regular n-gon into the distinct meeting points, with a proof for each coincidence and each separation.

    For i < j < k < l the diagonals ik and jl cross at z = (ab(c+d) - cd(a+b)) / (ab - cd), a = w^i, b = w^k, c = w^j, d = w^l, w = exp(2 pi i/n). Two crossings
    z1 = N1/D1 and z2 = N2/D2 are the same point exactly when g = N1 D2 - N2 D1 = 0, and g is a sum of at most 32 roots of unity, tested by `_cyclo`.
    A double-precision pre-filter (error below 1e-13 because |D| >= 2 sin(pi/n)) is used only to say "different" when the values are more than 1e-9 apart."""
    zf = [cmath.exp(2j * math.pi * k / n) for k in range(n)]
    pts = []          # (z float, i, j, k, l)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                ab = zf[(i + k) % n]
                for l in range(k + 1, n):
                    cd = zf[(j + l) % n]
                    z = (ab * (zf[j] + zf[l]) - cd * (zf[i] + zf[k])) / (ab - cd)
                    pts.append((z, i, j, k, l))
    # bucket by rounded value, join neighbouring buckets
    buckets = {}
    scale = 1e6
    for idx, (z, *_r) in enumerate(pts):
        buckets.setdefault((round(z.real * scale), round(z.imag * scale)), []).append(idx)
    parent = {key: key for key in buckets}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    for (kx, ky) in buckets:
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                nb = (kx + dx, ky + dy)
                if nb in parent and nb != (kx, ky):
                    ra, rb = find((kx, ky)), find(nb)
                    if ra != rb:
                        parent[ra] = rb
    comps = {}
    for key, lst in buckets.items():
        comps.setdefault(find(key), []).extend(lst)

    def nd(i, j, k, l):
        return {i + k + j: 1, i + k + l: 1, j + l + i: -1, j + l + k: -1}, {i + k: 1, j + l: -1}   # the four exponents are distinct integers
    sizes = []
    for lst in comps.values():
        reps = []         # (index, count)
        for idx in lst:
            z, i, j, k, l = pts[idx]
            matched = None
            for r in reps:
                zr = pts[r[0]][0]
                if abs(z - zr) > 1e-9:
                    continue
                N1, D1 = nd(i, j, k, l)
                N2, D2 = nd(*pts[r[0]][1:])
                g = _padd(_pmul(N1, D2), _pmul(N2, D1), -1)
                if _cyclo(g, n)[0]:
                    matched = r
                    break
            if matched is None:
                reps.append([idx, 1])
            else:
                matched[1] += 1
        sizes.extend(r[1] for r in reps)
    return sizes


@op('geometry.regular_ngon_diagonal_intersections', kind='regular polygons', bound='3 <= n <= 60 (C(n,4) crossings, each coincidence proved in Z[zeta_n])')
def regular_ngon_diagonal_intersections(n):
    """Interior points where the diagonals of a regular n-gon meet, with multiplicity, exact. Every 4 vertices give one crossing pair; coinciding pairs are found
    by an exact test in the cyclotomic integers (a floating pre-filter only separates). Returns dict(pairs = C(n,4), points = distinct points, by_multiplicity =
    {m: number of points where exactly m diagonals meet}, regions = 1 + D + sum (m - 1), the number of parts the diagonals cut the polygon into).
    regular_ngon_diagonal_intersections(6) -> points 13, by_multiplicity {2: 12, 3: 1}, regions 24"""
    n = _int(n, 3, 60)
    sizes = _diag_points(n) if n >= 4 else []
    by_m = {}
    for s in sizes:
        m = (1 + isqrt(1 + 8 * s)) // 2
        if m * (m - 1) // 2 != s:
            raise NotCovered('inconsistent multiplicity (internal check failed)')
        by_m[m] = by_m.get(m, 0) + 1
    diagonals = n * (n - 3) // 2
    regions = 1 + diagonals + sum((m - 1) * c for m, c in by_m.items())
    return {'pairs': comb(n, 4), 'points': len(sizes), 'by_multiplicity': dict(sorted(by_m.items())), 'regions': regions}


@op('geometry.regular_ngon_rectangles_on_lines', kind='regular polygons', bound='3 <= n <= 48; exact corner test on every corner within 1e-9 of a side')
def regular_ngon_rectangles_on_lines(n):
    """Rectangles inside a regular n-gon whose four sides lie on sides or diagonals (the AIME 2024 dodecagon question: 315 for n = 12), and how many of them are
    squares. Two perpendicular classes of chord lines (i+j = s and s + n/2 mod n; none for odd n) give a rectangle for each two lines of each class whose four
    corners lie in the closed polygon (a chord is the line's whole intersection with a convex polygon). Corner positions and equal side lengths are decided exactly
    in the cyclotomic integers. Returns dict(rectangles, squares). regular_ngon_rectangles_on_lines(12) -> rectangles 315"""
    n = _int(n, 3, 48)
    if n % 2:
        return {'rectangles': 0, 'squares': 0}
    m = 2 * n                                    # w = exp(i pi/n); the vertex k is w^(2k)
    wf = [cmath.exp(1j * math.pi * k / n) for k in range(m)]
    apothem = math.cos(math.pi / n)
    classes = {}
    for i in range(n):
        for j in range(i + 1, n):
            classes.setdefault((i + j) % n, []).append((i, j))

    def corner_nd(ca, cb):
        i, k = ca
        j, l = cb
        N = {}
        for e, c in ((2 * (i + k + j), 1), (2 * (i + k + l), 1), (2 * (j + l + i), -1), (2 * (j + l + k), -1)):
            N[e] = N.get(e, 0) + c
        return N, {2 * (i + k): 1, 2 * (j + l): -1}

    cache = {}

    def inside(ca, cb):
        key = (ca, cb)
        got = cache.get(key)
        if got is not None:
            return got
        (i, k), (j, l) = ca, cb
        ab, cd = wf[(2 * (i + k)) % m], wf[(2 * (j + l)) % m]
        z = (ab * (wf[2 * j] + wf[2 * l]) - cd * (wf[2 * i] + wf[2 * k])) / (ab - cd)
        N = D = None
        ok = True
        for t in range(n):
            mu = 2 * t + 1
            f = (z * wf[(-mu) % m]).real - apothem
            if f < -1e-9:
                continue
            if f > 1e-9:
                ok = False
                break
            if N is None:
                N, D = corner_nd(ca, cb)
            g = _padd(_padd(_pmul(_pmul(N, {-mu: 1}), _pconj(D)), _pmul(_pmul(_pconj(N), {mu: 1}), D)), _pmul(_pmul({1: 1, -1: 1}, D), _pconj(D)), -1)
            zero, rs, _is = _cyclo(g, m)
            if not zero and rs > 0:
                ok = False
                break
            if not zero and rs == 0:
                raise NotCovered('precision (internal check failed)')
        cache[key] = ok
        return ok

    def offset_terms(chord):
        """2*offset of the line along the class normal, as roots of unity: +-(w^d + w^-d), d = j - i."""
        i, j = chord
        sg = -1 if i + j >= n else 1
        d = j - i
        return {d: sg, -d: sg}

    def offset_float(chord):
        i, j = chord
        return (-1 if i + j >= n else 1) * math.cos(math.pi * (j - i) / n)

    rect = squares = 0
    for s in range(n // 2):
        la, lb = classes.get(s, []), classes.get((s + n // 2) % n, [])
        valid = [frozenset(idx for idx, M in enumerate(lb) if inside(L, M)) for L in la]
        for a in range(len(la)):
            if len(valid[a]) < 2:
                continue
            for b in range(a + 1, len(la)):
                common = sorted(valid[a] & valid[b])
                c = len(common)
                if c < 2:
                    continue
                rect += c * (c - 1) // 2
                dx = abs(offset_float(la[a]) - offset_float(la[b]))
                for u in range(c):
                    for v in range(u + 1, c):
                        dy = abs(offset_float(lb[common[u]]) - offset_float(lb[common[v]]))
                        if abs(dx - dy) > 1e-9:
                            continue
                        x = _padd(offset_terms(la[a]), offset_terms(la[b]), -1)
                        y = _padd(offset_terms(lb[common[u]]), offset_terms(lb[common[v]]), -1)
                        if _cyclo(_padd(x, y, -1), m)[0] or _cyclo(_padd(x, y, 1), m)[0]:
                            squares += 1
    return {'rectangles': rect, 'squares': squares}
