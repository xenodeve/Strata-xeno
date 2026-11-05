"""tools/igc/mathbackend/geometry.py unit tests: python tools/test_mathbackend_geometry.py (standard library only, no model, no network).

Values are worked by hand or are published ones (OEIS A006561 for the diagonal intersections of a regular n-gon, AIME 2024 I Problem 6 for the rectangles in a regular
dodecagon, Gauss circle counts). The independent oracles (SymPy, mpmath) are in tools/igc/mathbackend/sim_geometry.py."""
import ast
import sys
import unittest
from decimal import Decimal
from fractions import Fraction as F
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.igc.mathbackend import BadInput, NotCovered, Surd, registry, run   # noqa: E402
from tools.igc.mathbackend import geometry as G   # noqa: E402


def r(name, *a, **k):
    return run('geometry.' + name, *a, **k)


class Registry(unittest.TestCase):
    def test_every_operation_is_documented_and_bounded(self):
        ops = {k: v for k, v in registry().items() if k.startswith('geometry.')}
        self.assertGreaterEqual(len(ops), 70)
        for name, o in ops.items():
            self.assertTrue(o['doc'], name)
            self.assertTrue(o['kind'] and o['bound'], name)

    def test_standard_library_only(self):
        tree = ast.parse(Path(G.__file__).read_text(encoding='utf-8'))
        mods = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                mods.update(a.name.split('.')[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom):
                mods.add(node.module.split('.')[0])
        self.assertEqual(mods - {'cmath', 'math', 'decimal', 'fractions', 'tools'}, set())


class Basics(unittest.TestCase):
    def test_floats_are_refused_strings_and_decimals_are_exact(self):
        with self.assertRaises(BadInput):
            r('dist2', (0.5, 0), (1, 1))
        self.assertEqual(r('dist2', ('1/2', 0), (Decimal('1.5'), 0)), 1)
        with self.assertRaises(BadInput):
            r('dist2', (0, 0), (1, 1, 1))

    def test_distance_and_area(self):
        self.assertEqual(r('dist2', (0, 0), (3, 4)), 25)
        self.assertEqual(r('dist', (0, 0), (1, 1)), Surd(0, 1, 2))
        self.assertEqual(r('triangle_area', (0, 0), (4, 0), (0, 3)), 6)
        self.assertEqual(r('triangle_area_sq', (0, 0, 0), (1, 0, 0), (0, 1, 0)), F(1, 4))
        self.assertEqual(r('tetrahedron_volume', (0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)), F(1, 6))
        self.assertEqual(r('cross', (1, 0, 0), (0, 1, 0)), (0, 0, 1))

    def test_lines(self):
        self.assertEqual(r('line_through', (0, 0), (1, 1)), (1, -1, 0))
        self.assertEqual(r('line_through', (F(1, 2), 0), (0, F(1, 3))), (2, 3, 1))     # 2x/... : 2x + 3y = 1 passes (1/2,0) and (0,1/3)
        self.assertEqual(r('line_intersection', (0, 0), (2, 2), (0, 2), (2, 0)), (1, 1))
        self.assertEqual(r('line_intersection', (0, 0), (1, 1), (0, 1), (1, 2)), 'parallel')
        self.assertEqual(r('line_intersection', (0, 0), (1, 1), (2, 2), (3, 3)), 'coincident')
        self.assertEqual(r('dist_point_line2', (0, 2), (0, 0), (1, 0)), 4)
        self.assertEqual(r('dist_point_line', (0, 1), (0, 0), (1, 1)), Surd(0, F(1, 2), 2))
        self.assertTrue(r('collinear', (0, 0), (1, 1), (5, 5)))
        self.assertFalse(r('collinear', (0, 0), (1, 1), (5, 6)))
        self.assertTrue(r('coplanar', (0, 0, 0), (1, 0, 0), (0, 1, 0), (5, 7, 0)))

    def test_segments_and_polylines(self):
        self.assertEqual(r('segment_intersection', (0, 0), (2, 2), (0, 2), (2, 0)), [(1, 1)])
        self.assertEqual(r('segment_intersection', (0, 0), (1, 0), (2, 0), (3, 0)), [])
        self.assertEqual(r('segment_intersection', (0, 0), (2, 0), (1, 0), (3, 0)), [(1, 0), (2, 0)])
        self.assertEqual(r('segment_intersection', (0, 0), (1, 1), (1, 1), (2, 0)), [(1, 1)])           # touching ends
        self.assertEqual(r('segment_intersection', (1, 1), (1, 1), (0, 0), (2, 2)), [(1, 1)])           # a point on a segment
        # y = |x - 1| against y = 2 - |x - 2| ... two V and inverted V
        P = [(0, 1), (1, 0), (2, 1), (3, 2)]
        Q = [(0, 0), (1, 1), (2, 2), (3, 1)]
        self.assertEqual(r('polyline_intersections', P, Q), [(F(1, 2), F(1, 2)), (F(5, 2), F(3, 2))])
        with self.assertRaises(NotCovered):
            r('polyline_intersections', [(0, 0), (4, 0)], [(1, 0), (2, 0)])

    def test_concyclic(self):
        self.assertTrue(r('concyclic', (1, 0), (0, 1), (-1, 0), (0, -1)))
        self.assertFalse(r('concyclic', (1, 0), (0, 1), (-1, 0), (0, -2)))


class Circles(unittest.TestCase):
    def test_circle_through_three_points(self):
        self.assertEqual(r('circle_through3', (1, 0), (0, 1), (-1, 0)), ((0, 0), 1))
        c, r2 = r('circle_through3', (0, 0), (4, 0), (0, 3))
        self.assertEqual((c, r2), ((2, F(3, 2)), F(25, 4)))
        with self.assertRaises(BadInput):
            r('circle_through3', (0, 0), (1, 1), (2, 2))

    def test_circle_line(self):
        self.assertEqual(r('circle_line_intersection', (0, 0), 2, (0, 0), (1, 0)), [(Surd(0, -1, 2), Surd(0)), (Surd(0, 1, 2), Surd(0))])
        self.assertEqual(r('circle_line_intersection', (0, 0), 1, (1, -5), (1, 5)), [(Surd(1), Surd(0))])            # tangent
        self.assertEqual(r('circle_line_intersection', (0, 0), 1, (2, -5), (2, 5)), [])
        pts = r('circle_line_intersection', (0, 0), 25, (3, 4), (5, 0))                                          # lattice chord: rational points
        self.assertEqual([(a.p, b.p) for a, b in pts], [(3, 4), (5, 0)])
        self.assertTrue(all(a.q == 0 and b.q == 0 for a, b in pts))

    def test_circle_circle(self):
        pts = r('circle_circle_intersection', (0, 0), 1, (1, 0), 1)
        self.assertEqual(pts, [(Surd(F(1, 2)), Surd(0, F(-1, 2), 3)), (Surd(F(1, 2)), Surd(0, F(1, 2), 3))])
        self.assertEqual(r('circle_circle_intersection', (0, 0), 1, (2, 0), 1), [(Surd(1), Surd(0))])
        self.assertEqual(r('circle_circle_intersection', (0, 0), 1, (5, 0), 1), [])
        with self.assertRaises(NotCovered):
            r('circle_circle_intersection', (0, 0), 1, (0, 0), 4)

    def test_relations(self):
        rel = lambda *a: r('circle_circle_relation', *a)
        self.assertEqual(rel((0, 0), 1, (2, 0), 1), 'external tangent')
        self.assertEqual(rel((0, 0), 1, (3, 0), 1), 'separate')
        self.assertEqual(rel((0, 0), 4, (1, 0), 1), 'internal tangent')
        self.assertEqual(rel((0, 0), 9, (1, 0), 1), 'contained')
        self.assertEqual(rel((0, 0), 4, (0, 0), 1), 'contained')
        self.assertEqual(rel((0, 0), 4, (1, 0), 4), 'intersecting')
        self.assertEqual(rel((0, 0), 4, (0, 0), 4), 'identical')
        self.assertEqual(rel((0, 0), 4, (3, 4), 9), 'external tangent')                                           # 2 + 3 = 5 = distance

    def test_angle_chasing_on_a_circle(self):
        self.assertEqual(r('circle_angle', 'inscribed', 100), 50)
        self.assertEqual(r('circle_angle', 'tangent_chord', 90), 45)
        self.assertEqual(r('circle_angle', 'chords_inside', 70, 50), 60)
        self.assertEqual(r('circle_angle', 'outside', 200, 40), 80)
        self.assertEqual(r('circle_angle', 'outside', 180, 0), 90)               # two tangents to a circle seen under the far arc 180
        self.assertEqual(r('circle_angle', 'inscribed', F(1, 3)), F(1, 6))
        for bad in (('outside', 40, 200), ('chords_inside', 200, 200), ('inscribed', 400), ('nothing', 10)):
            with self.assertRaises(BadInput):
                r('circle_angle', *bad)
        # the same angles from coordinates: the inscribed angle on a diameter is 90 degrees
        self.assertEqual(r('angle_degrees_exact', (-1, 0), (1, 0)), 180)
        self.assertEqual(r('angle_degrees_exact', (-1, 0), (0, 1)), 90)

    def test_power_and_tangent(self):
        self.assertEqual(r('power_of_point', (3, 0), (0, 0), 1), 8)
        self.assertEqual(r('tangent_length', (3, 0), (0, 0), 1), Surd(0, 2, 2))
        with self.assertRaises(BadInput):
            r('tangent_length', (0, 0), (0, 0), 1)


class Triangles(unittest.TestCase):
    def test_heron_13_14_15(self):
        t = r('triangle_from_sides', 13, 14, 15)
        self.assertEqual(t['area'], Surd(84))
        self.assertEqual(t['inradius'], Surd(4))
        self.assertEqual(t['circumradius'], Surd(F(65, 8)))
        self.assertEqual(t['circumradius_sq'], F(65, 8) ** 2)
        with self.assertRaises(BadInput):
            r('triangle_from_sides', 1, 2, 3)

    def test_squared_sides(self):
        with self.assertRaises(BadInput):
            r('triangle_from_sides_sq', 1, 1, 4)
        with self.assertRaises(BadInput):
            r('triangle_from_sides_sq', 2, 2, 8)                    # sides sqrt2, sqrt2, 2sqrt2: degenerate

    def test_squared_sides_values(self):
        # sides 1, 1, sqrt(2): a right isosceles triangle of area 1/2, circumradius sqrt(2)/2
        t = r('triangle_from_sides_sq', 1, 1, 2)
        # 16 area^2 = 2(1 + 2 + 2) - (1 + 1 + 4) = 4  -> area^2 = 1/4
        self.assertEqual(t['area_sq'], F(1, 4))
        self.assertEqual(t['circumradius_sq'], F(1, 2))

    def test_centres(self):
        p, q, s = (0, 0), (4, 0), (0, 3)
        self.assertEqual(r('centroid', p, q, s), (F(4, 3), 1))
        self.assertEqual(r('circumcenter', p, q, s), (2, F(3, 2)))
        self.assertEqual(r('orthocenter', p, q, s), (0, 0))
        self.assertEqual(r('incenter', p, q, s), (Surd(1), Surd(1)))
        self.assertEqual(r('inradius', p, q, s), Surd(1))
        self.assertEqual(r('circumradius_sq', p, q, s), F(25, 4))
        self.assertEqual(r('circumradius_sq', (0, 0, 0), (1, 0, 0), (0, 1, 0)), F(1, 2))
        with self.assertRaises(NotCovered):                       # sides sqrt(2), sqrt(10), 4: several radicals
            r('inradius', (0, 0), (1, 1), (4, 0))
        self.assertEqual(r('inradius_numeric', (0, 0), (1, 1), (4, 0), 6), Decimal('0.466391'))      # 2 * 2 / (sqrt2 + sqrt10 + 4)
        self.assertEqual(r('inradius_numeric', p, q, s, 3), Decimal('1.000'))
        self.assertEqual(r('incenter_numeric', p, q, s, 3), (Decimal('1.000'), Decimal('1.000')))
        self.assertEqual(r('circumradius', p, q, s), Surd(F(5, 2)))

    def test_angles(self):
        self.assertEqual(r('angle_cos', (1, 0), (1, 1)), Surd(0, F(1, 2), 2))
        self.assertEqual(r('angle_degrees_exact', (1, 0), (1, 1)), 45)
        self.assertEqual(r('angle_degrees_exact', (1, 1, 0), (1, 0, 1)), 60)
        self.assertEqual(r('angle_degrees_exact', (1, 1, 0), (-1, 0, -1)), 120)
        self.assertEqual(r('angle_degrees_exact', (1, 2), (-2, -4)), 180)
        with self.assertRaises(NotCovered):
            r('angle_degrees_exact', (1, 0), (2, 1))
        self.assertEqual(r('angle_degrees_numeric', (1, 0), (1, 1), 5), Decimal('45.00000'))
        self.assertEqual(r('angle_degrees_numeric', (1, 0), (1, 2), 8), Decimal('63.43494882'))     # atan(2) = 63.4349488229...
        self.assertEqual(r('angle_degrees_numeric', (1, 0, 0), (-1, 0, 0), 3), Decimal('180.000'))
        a = r('triangle_angles_numeric', (0, 0), (1, 0), (0, 1), 3)
        self.assertEqual(a, (Decimal('90.000'), Decimal('45.000'), Decimal('45.000')))

    def test_decimal_trigonometry_against_known_digits(self):
        self.assertEqual(str(G._pi(50))[:52], '3.14159265358979323846264338327950288419716939937510')
        s, c = None, None
        from decimal import localcontext
        with localcontext() as ctx:
            ctx.prec = 40
            s, c = G._sincos(G._pi(45) / 6)
            self.assertAlmostEqual(float(s), 0.5, places=15)
            self.assertAlmostEqual(float(c), 0.8660254037844386, places=15)


class Polygons(unittest.TestCase):
    SQUARE = [(0, 0), (2, 0), (2, 2), (0, 2)]
    ARROW = [(0, 0), (4, 0), (4, 4), (2, 1), (0, 4)]      # a 4 x 4 square with the triangle (0,4),(4,4),(2,1) cut out of its top: area 16 - 6

    def test_hull(self):
        self.assertEqual(r('convex_hull', [(0, 0), (2, 0), (1, 1), (2, 2), (0, 2), (1, 0)]), [(0, 0), (2, 0), (2, 2), (0, 2)])
        self.assertEqual(r('convex_hull', [(0, 0), (1, 1), (2, 2)]), [(0, 0), (2, 2)])
        self.assertEqual(r('convex_hull', [(1, 1), (1, 1)]), [(1, 1)])

    def test_area_perimeter_centroid(self):
        self.assertEqual(r('polygon_area', [(0, 0), (4, 0), (0, 3)]), 6)
        self.assertEqual(r('polygon_signed_area', self.SQUARE[::-1]), -4)
        self.assertEqual(r('polygon_area', self.ARROW), 10)
        self.assertEqual(r('polygon_perimeter', [(0, 0), (3, 0), (3, 4)]), Surd(12))
        self.assertEqual(r('polygon_perimeter_terms', [(0, 0), (1, 1), (2, 0), (1, -3)]), [(2, 2), (10, 2)])
        with self.assertRaises(NotCovered):
            r('polygon_perimeter', [(0, 0), (1, 1), (2, 0), (1, -3)])
        self.assertEqual(r('polygon_perimeter_numeric', [(0, 0), (1, 1), (2, 0)], 4), Decimal('4.8284'))
        self.assertEqual(r('polygon_centroid', self.SQUARE), (1, 1))

    def test_simple_convex(self):
        self.assertTrue(r('is_simple_polygon', self.ARROW))
        self.assertFalse(r('is_simple_polygon', [(0, 0), (2, 2), (2, 0), (0, 2)]))
        self.assertFalse(r('is_simple_polygon', [(0, 0), (2, 0), (1, 0)]))                  # folds back on itself
        self.assertTrue(r('is_convex', self.SQUARE))
        self.assertTrue(r('is_convex', [(0, 0), (1, 0), (2, 0), (2, 2), (0, 2)]))            # a collinear vertex
        self.assertFalse(r('is_convex', self.ARROW))
        self.assertFalse(r('is_convex', [(0, 0), (2, 0), (0, 1), (2, 1)]))                 # self-crossing, all turns not one sign
        # a pentagram: every turn has one sign, but it winds twice
        pent = [(0, 0), (2, 0), (F(1, 2), F(3, 2)), (F(1), F(-1, 2)), (F(3, 2), F(3, 2))]
        self.assertFalse(r('is_convex', pent))

    def test_point_in_polygon(self):
        for pt, want in [((1, 1), 'inside'), ((0, 1), 'boundary'), ((2, 2), 'boundary'), ((3, 1), 'outside'), ((-1, -1), 'outside')]:
            self.assertEqual(r('point_in_polygon', pt, self.SQUARE), want, pt)
        for pt, want in [((F(1, 10), 3), 'inside'), ((1, 1), 'inside'), ((2, F(1, 2)), 'inside'), ((2, 1), 'boundary'), ((2, 3), 'outside'), ((3, 3), 'outside'),
                         ((4, 2), 'boundary'), ((0, 4), 'boundary'), ((5, 1), 'outside'), ((1, 0), 'boundary')]:
            self.assertEqual(r('point_in_polygon', pt, self.ARROW), want, pt)
        # a ray through a vertex must not be counted twice
        tri = [(0, 0), (2, 2), (0, 4)]
        self.assertEqual(r('point_in_polygon', (-1, 2), tri), 'outside')
        self.assertEqual(r('point_in_polygon', (1, 2), tri), 'inside')
        self.assertEqual(r('point_in_polygon', (-1, 4), tri), 'outside')


class Lattice(unittest.TestCase):
    def test_pick(self):
        self.assertEqual(r('pick_area', 3, 8), 6)
        s = r('lattice_polygon_stats', [(0, 0), (4, 0), (0, 4)])
        self.assertEqual((s['area2'], s['boundary'], s['interior'], s['area']), (16, 12, 3, 8))
        s = r('lattice_polygon_stats', [(0, 0), (4, 0), (4, 4), (2, 1), (0, 4)])
        self.assertEqual((s['area'], s['boundary'], s['interior']), (10, 14, 4))        # edges 4 + 4 + 1 + 1 + 4 lattice steps; I = A - B/2 + 1
        with self.assertRaises(BadInput):
            r('lattice_polygon_stats', [(0, 0), (2, 2), (2, 0), (0, 2)])
        with self.assertRaises(BadInput):
            r('lattice_polygon_stats', [(0, 0), (F(1, 2), 0), (0, 1)])

    def test_triangle(self):
        self.assertEqual(r('lattice_points_in_triangle', (0, 0), (10, 0), (0, 10)), {'total': 66, 'interior': 36, 'boundary': 30})
        self.assertEqual(r('lattice_points_in_triangle', (0, 0), (4, 0), (0, 4)), {'total': 15, 'interior': 3, 'boundary': 12})
        # rational vertices: the triangle (0,0),(5/2,0),(0,5/2) holds (0,0) (1,0) (2,0) (0,1) (0,2) on its edges and (1,1) inside
        self.assertEqual(r('lattice_points_in_triangle', (0, 0), (F(5, 2), 0), (0, F(5, 2))), {'total': 6, 'interior': 1, 'boundary': 5})
        self.assertEqual(r('lattice_points_on_segment', (0, 0), (4, 6)), 3)
        with self.assertRaises(BadInput):
            r('lattice_points_in_triangle', (0, 0), (1, 1), (2, 2))

    def test_convex_polygon_rows_agree_with_pick(self):
        poly = [(0, 0), (6, 0), (7, 3), (3, 6), (-1, 4)]
        s = r('lattice_polygon_stats', poly)
        c = r('lattice_points_in_convex_polygon', poly)
        self.assertEqual((c['interior'], c['boundary']), (s['interior'], s['boundary']))
        with self.assertRaises(NotCovered):
            r('lattice_points_in_convex_polygon', [(0, 0), (4, 0), (4, 4), (2, 1), (0, 4)])

    def test_gauss_circle(self):
        self.assertEqual(r('lattice_points_in_circle', 25), 81)
        self.assertEqual(r('lattice_points_in_circle', 100), 317)
        self.assertEqual(r('lattice_points_in_circle', 10000), 31417)
        self.assertEqual(r('lattice_points_in_circle', 25, strict=True), 69)                 # 81 minus the 12 points on the circle
        self.assertEqual(r('lattice_points_in_circle', F(1, 2)), 1)
        self.assertEqual(r('lattice_points_in_circle', 0, strict=True), 0)
        with self.assertRaises(NotCovered):
            r('lattice_points_in_circle', 10 ** 14)


class Transformations(unittest.TestCase):
    def test_reflections(self):
        self.assertEqual(r('reflect_point_line', (0, 2), (0, 0), (1, 0)), (0, -2))
        self.assertEqual(r('reflect_point_line', (3, 1), (0, 0), (1, 1)), (1, 3))
        self.assertEqual(r('reflect_point_line', (2, 0), (0, 1), (1, 0)), (1, -1))
        self.assertEqual(r('reflect_point_point', (1, 2, 3), (0, 0, 0)), (-1, -2, -3))
        self.assertEqual(r('reflect_points_line', [(0, 2), (1, 3)], (0, 0), (1, 0)), [(0, -2), (1, -3)])

    def test_rotations(self):
        self.assertEqual(r('rotate_point_90', (1, 0), (0, 0), 1), (0, 1))
        self.assertEqual(r('rotate_point_90', (2, 1), (1, 1), -1), (1, 0))
        self.assertEqual(r('rotate_point_90', (2, 1), (1, 1), 6), (0, 1))
        self.assertEqual(r('rotate_points_90', [(1, 0), (2, 0)], (0, 0), 2), [(-1, 0), (-2, 0)])
        self.assertEqual(r('rotate_point_cs', (5, 0), (0, 0), F(3, 5), F(4, 5)), (3, 4))
        with self.assertRaises(BadInput):
            r('rotate_point_cs', (5, 0), (0, 0), F(1, 2), F(1, 2))
        x, y = r('rotate_point_cos', (2, 0), (0, 0), F(1, 2))
        self.assertEqual((x, y), (Surd(1), Surd(0, 1, 3)))
        x, y = r('rotate_point_cos', (2, 0), (0, 0), F(1, 2), -1)
        self.assertEqual((x, y), (Surd(1), Surd(0, -1, 3)))


class Space(unittest.TestCase):
    def test_planes_and_spheres(self):
        self.assertEqual(r('plane_through', (1, 0, 0), (0, 1, 0), (0, 0, 1)), (1, 1, 1, 1))
        self.assertEqual(r('dist_point_plane2', (0, 0, 3), (0, 0, 0), (1, 0, 0), (0, 1, 0)), 9)
        self.assertEqual(r('line_plane_intersection', (0, 0, 1), (0, 0, 2), (0, 0, 0), (1, 0, 0), (0, 1, 0)), (0, 0, 0))
        self.assertEqual(r('line_plane_intersection', (0, 0, 1), (1, 0, 1), (0, 0, 0), (1, 0, 0), (0, 1, 0)), 'parallel')
        self.assertEqual(r('line_plane_intersection', (0, 0, 0), (1, 0, 0), (0, 0, 0), (1, 0, 0), (0, 1, 0)), 'in plane')
        self.assertEqual(r('skew_lines_dist2', (0, 0, 0), (1, 0, 0), (0, 0, 1), (0, 1, 1)), 1)
        self.assertEqual(r('skew_lines_dist2', (0, 0, 0), (1, 0, 0), (0, 2, 0), (1, 2, 0)), 4)    # parallel lines
        self.assertEqual(r('sphere_through4', (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, 0, 1)), ((0, 0, 0), 1))
        with self.assertRaises(BadInput):
            r('sphere_through4', (0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0))


class UnionIntersection(unittest.TestCase):
    A = [(0, 0), (2, 0), (2, 2), (0, 2)]
    B = [(1, 1), (3, 1), (3, 3), (1, 3)]

    def test_two_squares(self):
        self.assertEqual(r('polygons_union_area', [self.A, self.B]), 7)
        self.assertEqual(r('polygons_intersection_area', [self.A, self.B]), 1)
        self.assertEqual(r('convex_polygon_intersection', self.A, self.B), [(1, 1), (2, 1), (2, 2), (1, 2)])
        self.assertEqual(r('convex_polygon_intersection', self.A, [(5, 5), (6, 5), (6, 6)]), [])
        self.assertEqual(r('convex_polygon_intersection', self.A, [(2, 0), (4, 0), (4, 2), (2, 2)]), [(2, 0), (2, 2)])      # share an edge
        self.assertEqual(r('polygons_intersection_area', [self.A, [(2, 0), (4, 0), (4, 2), (2, 2)]]), 0)

    def test_triangle_and_its_reflections(self):
        tri = [(0, 0), (2, 0), (0, 2)]
        # across the hypotenuse the union is the square 2 x 2
        refl = r('reflect_points_line', tri, (2, 0), (0, 2))
        self.assertEqual(refl, [(2, 2), (2, 0), (0, 2)])
        self.assertEqual(r('polygons_union_area', [tri, refl]), 4)
        # across the x axis the two triangles share only a vertex pair on the axis
        self.assertEqual(r('polygons_union_area', [tri, r('reflect_points_line', tri, (0, 0), (1, 0))]), 4)
        # across the line x = 1 the overlap is the triangle (1,0),(1,1),(0,... ): union 3
        self.assertEqual(r('polygons_union_area', [tri, r('reflect_points_line', tri, (1, 0), (1, 1))]), 3)

    def test_non_convex_and_crossing_edges(self):
        arrow = [(0, 0), (4, 0), (4, 4), (2, 1), (0, 4)]
        bar = [(-1, 2), (5, 2), (5, 3), (-1, 3)]
        u = r('polygons_union_area', [arrow, bar])
        i = r('polygons_intersection_area', [arrow, bar])
        self.assertEqual(u + i, 10 + 6)                           # |A u B| + |A n B| = |A| + |B|
        # in the strip 2 <= y <= 3 the arrow covers x in [0, 2 - 2(y-1)/3] and [2 + 2(y-1)/3, 4], that is 4 - 4(y-1)/3 wide: the mean over the strip is 4 - 4*1.5/3 = 2
        self.assertEqual(i, 2)
        self.assertEqual(r('polygons_intersection_area', [bar, arrow]), 2)                  # the order does not matter
        # three overlapping unit squares in an L
        sq = lambda x, y: [(x, y), (x + 1, y), (x + 1, y + 1), (x, y + 1)]
        self.assertEqual(r('polygons_union_area', [sq(0, 0), sq(1, 0), sq(0, 1)]), 3)
        self.assertEqual(r('polygons_union_area', [sq(0, 0), sq(0, 0)]), 1)
        self.assertEqual(r('polygons_intersection_area', [sq(0, 0), sq(1, 1)]), 0)

    def test_rational_vertices(self):
        a = [(0, 0), (F(5, 3), 0), (0, F(5, 3))]
        b = [(F(1, 3), F(1, 3)), (2, F(1, 3)), (F(1, 3), 2)]
        union = r('polygons_union_area', [a, b])
        inter = r('polygons_intersection_area', [a, b])
        self.assertEqual(union + inter, r('polygon_area', a) + r('polygon_area', b))


class RegularNgon(unittest.TestCase):
    OEIS_A006561 = {3: 0, 4: 1, 5: 5, 6: 13, 7: 35, 8: 49, 9: 126, 10: 161, 11: 330, 12: 301}

    def test_diagonal_points_match_oeis_and_formula(self):
        for n, v in self.OEIS_A006561.items():
            self.assertEqual(r('regular_ngon_diagonal_intersections', n)['points'], v, n)

    def test_multiplicity_and_regions(self):
        d6 = r('regular_ngon_diagonal_intersections', 6)
        self.assertEqual(d6, {'pairs': 15, 'points': 13, 'by_multiplicity': {2: 12, 3: 1}, 'regions': 24})
        d5 = r('regular_ngon_diagonal_intersections', 5)
        self.assertEqual((d5['points'], d5['regions']), (5, 11))
        d4 = r('regular_ngon_diagonal_intersections', 4)
        self.assertEqual((d4['points'], d4['regions']), (1, 4))
        d12 = r('regular_ngon_diagonal_intersections', 12)
        self.assertEqual(sum(c * m * (m - 1) // 2 for m, c in d12['by_multiplicity'].items()), 495)    # every crossing pair is accounted for
        self.assertEqual(d12['by_multiplicity'].get(6), 1)         # the centre: the 6 diameters
        self.assertEqual(r('regular_ngon_diagonal_intersections', 8)['regions'], 80)
        with self.assertRaises(NotCovered):
            r('regular_ngon_diagonal_intersections', 61)

    def test_poonen_rubinstein_formula(self):
        # the published closed form: C(n,4) + corrections by the divisors of n
        def pr(n):
            from math import comb
            d = lambda m: n % m == 0
            v = F(comb(n, 4)) + F(-5 * n ** 3 + 45 * n ** 2 - 70 * n + 24, 24) * d(2) - F(3 * n, 2) * d(4) + F(-45 * n ** 2 + 262 * n, 6) * d(6)
            return int(v + 42 * n * d(12) + 60 * n * d(18) + 35 * n * d(24) - 38 * n * d(30))
        for n in (13, 14, 15, 16, 18, 20, 24):
            self.assertEqual(r('regular_ngon_diagonal_intersections', n)['points'], pr(n), n)

    def test_dodecagon_rectangles_on_lines(self):
        res = r('regular_ngon_rectangles_on_lines', 12)
        self.assertEqual(res['rectangles'], 315)              # AIME 2024 I Problem 6
        self.assertEqual(r('regular_ngon_rectangles_on_lines', 4), {'rectangles': 1, 'squares': 1})
        self.assertEqual(r('regular_ngon_rectangles_on_lines', 7), {'rectangles': 0, 'squares': 0})
        self.assertEqual(r('regular_ngon_rectangles_on_lines', 6)['squares'], 0)
        with self.assertRaises(NotCovered):
            r('regular_ngon_rectangles_on_lines', 50)

    def test_chords_and_basic(self):
        c = r('regular_ngon_chords_by_direction', 12)
        self.assertEqual(sum(c['classes'].values()), 66)
        self.assertEqual(c['classes'][0], 5)
        self.assertEqual(c['classes'][1], 6)
        self.assertEqual(c['perpendicular_pairs'][0], (0, 6))
        self.assertEqual(r('regular_ngon_chords_by_direction', 7)['perpendicular_pairs'], [])
        self.assertEqual(r('regular_ngon_chords_by_length', 6), {1: 6, 2: 6, 3: 3})
        b = r('regular_ngon_basic', 12)
        self.assertEqual((b['diagonals'], b['interior_angle_degrees'], b['central_angle_degrees']), (54, 150, 30))

    def test_triangles_and_quadrilaterals(self):
        t = r('regular_ngon_triangles', 12)
        self.assertEqual((t['total'], t['right'], t['acute'], t['obtuse'], t['equilateral'], t['isosceles']), (220, 60, 40, 120, 4, 52))
        self.assertEqual(t['right'], 12 // 2 * (12 - 2))      # a diameter and any other vertex
        self.assertEqual(t['scalene'], 168)
        for n in (5, 7, 9, 11):
            tn = r('regular_ngon_triangles', n)
            self.assertEqual(tn['right'], 0)
            self.assertEqual(tn['isosceles'], n * ((n - 1) // 2) - (2 * n // 3 if n % 3 == 0 else 0))
        q = r('regular_ngon_quadrilaterals', 12)
        self.assertEqual((q['total'], q['rectangles'], q['squares']), (495, 15, 3))
        self.assertEqual(r('regular_ngon_vertex_rectangles', 12), {'rectangles': 15, 'squares': 3})
        self.assertEqual(r('regular_ngon_vertex_rectangles', 9), {'rectangles': 0, 'squares': 0})
        for n in (8, 10, 12, 14):
            self.assertEqual(r('regular_ngon_quadrilaterals', n)['rectangles'], r('regular_ngon_vertex_rectangles', n)['rectangles'])

    def test_angles(self):
        self.assertEqual(r('regular_ngon_angle', 12, 0, 3, 6), 90)          # angle in a semicircle
        self.assertEqual(r('regular_ngon_angle', 12, 0, 1, 2), 150)         # interior angle
        self.assertEqual(r('regular_ngon_angle', 12, 1, 0, 2), 15)
        self.assertEqual(r('regular_ngon_angle', 5, 0, 1, 2), 108)          # the interior angle of a regular pentagon
        self.assertEqual(r('regular_ngon_chord_angle', 12, (0, 3), (0, 6)), 45)
        self.assertEqual(r('regular_ngon_chord_angle', 12, (0, 5), (1, 4)), 0)      # parallel chords: same i + j
        self.assertEqual(r('regular_ngon_chord_angle', 12, (0, 6), (3, 9)), 90)
        with self.assertRaises(BadInput):
            r('regular_ngon_angle', 12, 0, 0, 6)

    def test_numeric_and_exact_forms(self):
        v = r('regular_ngon_vertices_numeric', 4, 3)
        self.assertEqual(v, [(Decimal('1.000'), Decimal('0.000')), (Decimal('0.000'), Decimal('1.000')), (Decimal('-1.000'), Decimal('0.000')), (Decimal('0.000'), Decimal('-1.000'))])
        self.assertEqual(r('regular_ngon_side_numeric', 6, 1, 3), Decimal('1.000'))
        self.assertEqual(r('regular_ngon_area_exact', 12, 1), Surd(3))
        self.assertEqual(r('regular_ngon_area_exact', 3, 1), Surd(0, F(3, 4), 3))
        self.assertEqual(r('regular_ngon_area_exact', 8, 1), Surd(0, 2, 2))
        with self.assertRaises(NotCovered):
            r('regular_ngon_area_exact', 5, 1)
        with self.assertRaises(NotCovered):
            r('regular_ngon_triangles', 5000)
        with self.assertRaises(BadInput):
            r('regular_ngon_basic', 2)

    def test_cyclotomic_zero_test(self):
        z = G._cyclo
        self.assertTrue(z({0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1}, 6)[0])              # sum of all sixth roots of unity
        self.assertTrue(z({1: 1, -1: 1, 0: -1}, 6)[0])                                # 2 cos(60 deg) = 1
        self.assertFalse(z({1: 1, -1: 1, 0: -1}, 8)[0])
        self.assertEqual(z({1: 1, -1: 1}, 8)[1], 1)                                   # 2 cos(45 deg) > 0
        self.assertEqual(z({3: 1, -3: 1}, 8)[1], -1)                                  # 2 cos(135 deg) < 0
        self.assertTrue(z({1: 1, 6: 1}, 10)[0])                                       # w + w^6 = w (1 + w^5) = 0 because w^5 = -1
        self.assertFalse(z({1: 1, 5: 1}, 10)[0])                                      # w + w^5 = w - 1
        # a 5th-root identity at larger degree: 2cos(2pi/5) = (sqrt5 - 1)/2 so w + w^-1 + w^2 + w^-2 = -1
        self.assertTrue(z({1: 1, -1: 1, 2: 1, -2: 1, 0: 1}, 5)[0])
        self.assertFalse(z({1: 1, -1: 1, 2: 1, -2: 1, 0: 1}, 7)[0])


if __name__ == '__main__':
    unittest.main()
