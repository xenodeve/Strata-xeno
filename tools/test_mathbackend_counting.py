"""tools/igc/mathbackend/counting.py unit tests. python tools/test_mathbackend_counting.py (standard library only: no model, no network, no SymPy).

The module is imported directly, not through run(), so a half-written sibling module cannot break these tests. The independent-oracle checks live in sim_counting.py."""
import ast
import sys
import unittest
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.igc.mathbackend import BadInput, NotCovered
from tools.igc.mathbackend import counting as C

F = Fraction


class Registry(unittest.TestCase):
    def test_every_operation_is_documented_and_named(self):
        ops = [f for f in vars(C).values() if hasattr(f, 'op_name')]
        self.assertGreaterEqual(len(ops), 60)
        for f in ops:
            self.assertTrue(f.op_name.startswith('counting.'), f.op_name)
            self.assertTrue((f.__doc__ or '').strip(), f.op_name)
            self.assertIn('(', f.__doc__, f.op_name)         # one example in the docstring

    def test_only_the_standard_library_is_imported(self):
        tree = ast.parse((ROOT / 'tools/igc/mathbackend/counting.py').read_text(encoding='utf-8'))
        std = {'ast', 'itertools', 'keyword', 'math', 'fractions', 'functools'}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    self.assertIn(a.name.split('.')[0], std)
            elif isinstance(node, ast.ImportFrom):
                self.assertTrue(node.module in std or node.module == 'tools.igc.mathbackend', node.module)


class Combinatorics(unittest.TestCase):
    def test_stars_and_bars(self):
        self.assertEqual(C.stars_and_bars(10, 3), 66)
        self.assertEqual(C.stars_and_bars(10, 3, upper=4), 6)
        self.assertEqual(C.stars_and_bars(0, 0), 1)
        self.assertEqual(C.stars_and_bars(3, 0), 0)
        self.assertEqual(C.stars_and_bars(5, 2, upper=[2, None], lower=[1, 0]), 2)      # x1 in 1..2, x2 free: (1,4) and (2,3)
        self.assertEqual(C.stars_and_bars(-1, 2), 0)
        with self.assertRaises(BadInput):
            C.stars_and_bars(1.5, 2)
        with self.assertRaises(BadInput):
            C.stars_and_bars(True, 2)

    def test_derangement_family(self):
        self.assertEqual([C.derangements(n) for n in range(7)], [1, 0, 1, 2, 9, 44, 265])
        self.assertEqual(C.derangements(10), 1334961)
        self.assertEqual(C.rencontres(4, 1), 8)
        self.assertEqual(sum(C.rencontres(6, k) for k in range(7)), 720)
        self.assertEqual(C.surjections(4, 2), 14)
        self.assertEqual(C.surjections(3, 4), 0)
        self.assertEqual(C.surjections(0, 0), 1)

    def test_set_partitions(self):
        self.assertEqual(C.set_partitions(5, 2), 15)
        self.assertEqual(C.set_partitions(5), 52)
        self.assertEqual(C.set_partitions(0), 1)
        self.assertEqual(C.set_partitions(6, block_sizes=[2, 3]), 25)
        self.assertEqual(C.set_partitions(6, 3, block_sizes=[2]), 15)       # pairings of 6 points
        self.assertEqual(C.set_partitions(4, 5), 0)

    def test_twelvefold_way_table(self):
        n, k = 4, 3
        got = {(b, x, r): C.twelvefold_way(n, k, b, x, r) for b in ('labelled', 'identical') for x in ('labelled', 'identical') for r in ('any', 'injective', 'surjective')}
        self.assertEqual(got[('labelled', 'labelled', 'any')], 81)
        self.assertEqual(got[('labelled', 'labelled', 'injective')], 0)
        self.assertEqual(got[('labelled', 'labelled', 'surjective')], 36)
        self.assertEqual(got[('identical', 'labelled', 'any')], 15)
        self.assertEqual(got[('identical', 'labelled', 'surjective')], 3)
        self.assertEqual(got[('labelled', 'identical', 'any')], 14)
        self.assertEqual(got[('labelled', 'identical', 'surjective')], 6)
        self.assertEqual(got[('identical', 'identical', 'any')], 4)
        self.assertEqual(got[('identical', 'identical', 'surjective')], 1)

    def test_permutation_statistics(self):
        self.assertEqual(C.permutations_with_cycle_type([2, 2, 1]), 15)
        self.assertEqual(C.permutations_with_cycle_type([1] * 5), 1)
        self.assertEqual(C.eulerian(4, 1), 11)
        self.assertEqual([C.eulerian(4, k) for k in range(5)], [1, 11, 11, 1, 0])
        self.assertEqual(C.mahonian(4, 3), 6)
        self.assertEqual(sum(C.mahonian(5, k) for k in range(11)), 120)

    def test_anagrams(self):
        self.assertEqual(C.anagram_count('MISSISSIPPI'), 34650)
        self.assertEqual(C.anagram_count('MISSISSIPPI', True), 2016)
        self.assertEqual(C.anagram_count('AAAB', True), 0)
        self.assertEqual(C.anagram_count('AAB', True), 1)
        self.assertEqual(C.anagram_count('ABA', True), 1)
        self.assertEqual(C.anagram_count(''), 1)

    def test_subsets_with_sum(self):
        self.assertEqual(C.count_subsets_with_sum([1, 2, 3, 4, 5], target=6), 3)
        self.assertEqual(C.count_subsets_with_sum(list(range(1, 11)), mod=5, rem=0), 208)
        self.assertEqual(C.count_subsets_with_sum([1, 2, 3, 4], target=5, size=2), 2)
        self.assertEqual(C.count_subsets_with_sum([], target=0), 1)
        with self.assertRaises(BadInput):
            C.count_subsets_with_sum([1], target=1, mod=2)


class Partitions(unittest.TestCase):
    def test_partition_count(self):
        self.assertEqual(C.partition_count(10), 42)
        self.assertEqual(C.partition_count(100), 190569292)
        self.assertEqual(C.partition_count(10, distinct=True), 10)
        self.assertEqual(C.partition_count(10, num_parts=3), 8)
        self.assertEqual(C.partition_count(10, max_parts=3), 14)
        self.assertEqual(C.partition_count(20, parts=[1, 5, 10]), 9)
        self.assertEqual(C.partition_count(0), 1)
        self.assertEqual(C.partition_count(0, num_parts=0), 1)
        self.assertEqual(C.partition_count(5, num_parts=0), 0)
        self.assertEqual(C.partition_count(12, distinct=True, num_parts=3), 7)
        self.assertEqual(C.partition_count(12, max_part=3, max_parts=5), 3)
        # Euler: partitions into distinct parts = partitions into odd parts
        self.assertEqual(C.partition_count(40, distinct=True), C.partition_count(40, parts=range(1, 41, 2)))
        with self.assertRaises(BadInput):
            C.partition_count(5, num_parts=2, max_parts=3)

    def test_compositions(self):
        self.assertEqual(C.compositions_count(5), 16)
        self.assertEqual(C.compositions_count(10, 3), 36)
        self.assertEqual(C.compositions_count(10, parts=[1, 2]), 89)
        self.assertEqual(C.compositions_count(10, 3, max_part=4), 6)
        self.assertEqual(C.compositions_count(0), 1)
        with self.assertRaises(BadInput):
            C.compositions_count(5, min_part=0)

    def test_tableaux(self):
        self.assertEqual(C.standard_young_tableaux([3, 2]), 5)
        self.assertEqual(C.standard_young_tableaux([4, 3, 2, 1]), 768)
        self.assertEqual(C.semistandard_young_tableaux([2, 1], 3), 8)
        self.assertEqual(C.semistandard_young_tableaux([2], 5), 15)
        with self.assertRaises(BadInput):
            C.standard_young_tableaux([1, 2])


class Symmetry(unittest.TestCase):
    def test_necklaces_and_bracelets(self):
        self.assertEqual(C.necklace_count(6, 2), 14)
        self.assertEqual(C.necklace_count(6, 2, bracelet=True), 13)
        self.assertEqual(C.necklace_count(counts=[2, 2, 2]), 16)
        self.assertEqual(C.necklace_count(2, 3, bracelet=True), 6)       # the 2-gon: reflections are the identity and the swap
        self.assertEqual(C.necklace_count(1, 5, bracelet=True), 5)
        self.assertEqual(C.necklace_count(0, 3), 1)
        self.assertEqual(C.necklace_count(5, 0), 0)
        self.assertEqual(C.necklace_count(7, 3, bracelet=True), 198)
        self.assertEqual(C.necklace_count(counts=[3, 3], bracelet=True), 3)

    def test_orbits_under_a_group(self):
        self.assertEqual(C.orbit_count_colorings(4, [[1, 2, 3, 0]], 2), 6)
        verts = [(x, y, z) for x in (0, 1) for y in (0, 1) for z in (0, 1)]
        vi = {v: i for i, v in enumerate(verts)}
        turn_z = [vi[(1 - y, x, z)] for x, y, z in verts]            # quarter turns of the cube about two axes
        turn_x = [vi[(x, 1 - z, y)] for x, y, z in verts]
        self.assertEqual(C.orbit_count_colorings(8, [turn_z, turn_x], 2), 23)                     # the classic 23 two-colourings of cube corners
        faces = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
        fi = {f: i for i, f in enumerate(faces)}
        fz = [fi[(-b, a, c)] for a, b, c in faces]
        fx = [fi[(a, -c, b)] for a, b, c in faces]
        self.assertEqual(C.orbit_count_colorings(6, [fz, fx], 2), 10)
        self.assertEqual(C.orbit_count_colorings(6, [fz, fx], 3), 57)
        self.assertEqual(C.orbit_count_colorings(6, [fz, fx], counts=[2, 4]), 2)                  # two red faces: next to each other or opposite
        with self.assertRaises(BadInput):
            C.orbit_count_colorings(3, [[0, 0, 1]], 2)


class Paths(unittest.TestCase):
    def test_lattice_paths(self):
        self.assertEqual(C.lattice_paths((4, 4)), 70)
        self.assertEqual(C.lattice_paths((4, 4), constraint='y<=x'), 14)
        self.assertEqual(C.lattice_paths((6, 0), steps=[(1, 1), (1, -1)], region=(0, 0, 6, 3)), 5)
        self.assertEqual(C.lattice_paths((2, 2), forbidden=[(1, 1)]), 2)
        self.assertEqual(C.lattice_paths((2, 2), steps=[(1, 0), (0, 1), (1, 1)]), 13)
        self.assertEqual(C.lattice_paths((0, 0)), 1)
        self.assertEqual(C.lattice_paths((3, 3), forbidden=[(0, 0)]), 0)
        with self.assertRaises(NotCovered):
            C.lattice_paths((3, 3), steps=[(1, 0), (-1, 0)])         # a cycle is possible
        with self.assertRaises(NotCovered):
            C.lattice_paths((4, 0), steps=[(1, 1), (1, -1)])          # a negative step needs a region
        with self.assertRaises(BadInput):
            C.lattice_paths((1, 1), steps=[(0, 0)])

    def test_catalan_family(self):
        self.assertEqual([C.catalan(n) for n in range(8)], [1, 1, 2, 5, 14, 42, 132, 429])
        self.assertEqual(C.dyck_paths(4), 14)
        self.assertEqual(C.dyck_paths(4, 2), 8)
        self.assertEqual(C.dyck_paths(5, 1), 1)
        self.assertEqual(C.ballot(3, 1), 2)
        self.assertEqual(C.ballot(3, 3, False), 5)
        self.assertEqual(C.ballot(2, 3), 0)
        self.assertEqual(C.ballot(4, 0), 1)
        self.assertEqual([C.motzkin(n) for n in range(8)], [1, 1, 2, 4, 9, 21, 51, 127])
        self.assertEqual([C.schroeder(n) for n in range(5)], [1, 2, 6, 22, 90])
        self.assertEqual([C.schroeder(n, True) for n in range(1, 5)], [1, 3, 11, 45])
        self.assertEqual([C.narayana(4, k) for k in range(1, 5)], [1, 6, 6, 1])
        with self.assertRaises(BadInput):
            C.ballot(0, 0)


class Strings(unittest.TestCase):
    def test_pattern_automaton(self):
        self.assertEqual(C.strings_with_patterns('ab', 5, avoid=['aa']), 13)
        self.assertEqual(C.strings_with_patterns('01', 4, contain=['11']), 8)
        self.assertEqual(C.strings_with_patterns('ab', 2, avoid=['ba'], contain=['a']), 2)
        self.assertEqual(C.strings_with_patterns('abc', 0), 1)
        self.assertEqual(C.strings_with_patterns('abc', 0, contain=['a']), 0)
        self.assertEqual(C.strings_with_patterns(3, 3, avoid=['00']), 22)          # an integer alphabet is the characters 0, 1, 2
        self.assertEqual(C.strings_with_patterns('ab', 5, avoid=['aba', 'bab']), 16)
        self.assertEqual(C.strings_with_patterns('ab', 3, contain=['ab', 'ba']), 2)
        self.assertEqual(C.strings_with_patterns('ab', 3, contain=['z']), 0)
        with self.assertRaises(BadInput):
            C.strings_with_patterns('ab', 3, avoid=[''])

    def test_transfer_matrix(self):
        self.assertEqual(C.transfer_matrix_walks([[1, 1], [1, 0]], 10, 0, 0), 89)
        self.assertEqual(C.transfer_matrix_walks([[1, 1], [1, 0]], 5), 8 + 5 + 5 + 3)
        self.assertEqual(C.transfer_matrix_walks([[F(1, 2)]], 3), F(1, 8))
        big = C.transfer_matrix_walks([[1, 1], [1, 0]], 5000, 0, 0)         # the matrix-power branch (5000 * 4 > ... uses the squaring path for large n)
        self.assertEqual(big, _walks_slow([[1, 1], [1, 0]], 5000, 0, 0))
        small_pow = C.transfer_matrix_walks([[2, 1, 0], [0, 1, 1], [1, 0, 1]], 40, 0, 2)
        self.assertEqual(small_pow, _walks_slow([[2, 1, 0], [0, 1, 1], [1, 0, 1]], 40, 0, 2))

    def test_grid_colorings(self):
        self.assertEqual(C.grid_colorings(2, 2, 3), 18)
        self.assertEqual(C.grid_colorings(3, 3, 3), 246)
        self.assertEqual(C.grid_colorings(3, 3, 2), 2)
        self.assertEqual(C.grid_colorings(1, 5, 3), 3 * 2 ** 4)
        self.assertEqual(C.grid_colorings(6, 2, 3), C.grid_colorings(2, 6, 3))                 # the profile is the shorter side
        self.assertEqual(C.grid_colorings(2, 3, 2, allowed=[[1, 1], [1, 1]]), 64)
        self.assertEqual(C.grid_colorings(1, 4, 2, allowed=[[0, 1], [0, 0]]), 0)            # a may stand left of b, but nothing may follow b
        self.assertEqual(C.grid_colorings(1, 2, 2, allowed=[[0, 1], [0, 0]]), 1)
        # vertical rule differs from horizontal: 2 rows, left-right anything, up-down must differ
        self.assertEqual(C.grid_colorings(2, 3, 2, allowed=[[1, 1], [1, 1]], allowed_v=[[0, 1], [1, 0]]), 2 ** 3)
        self.assertEqual(C.grid_colorings(0, 5, 3), 1)

    def test_dominoes(self):
        self.assertEqual(C.domino_tilings(8, 8), 12988816)
        self.assertEqual(C.domino_tilings(2, 5), 8)
        self.assertEqual(C.domino_tilings(3, 3), 0)
        self.assertEqual(C.domino_tilings(3, 4), 11)
        self.assertEqual(C.domino_tilings(8, 8, blocked=[(0, 0), (7, 7)]), 0)                   # two cells of the same colour
        self.assertEqual(C.domino_tilings(3, 3, blocked=[(1, 1)]), 2)
        self.assertEqual(C.domino_tilings(0, 3), 1)

    def test_digit_constraints(self):
        self.assertEqual(C.count_digit_constraints(1, 1000, digit_sum=10), 63)
        self.assertEqual(C.count_digit_constraints(1, 10 ** 6, forbidden_digits=[7]), 531441)
        self.assertEqual(C.count_digit_constraints(1, 99, distinct=True), 90)
        self.assertEqual(C.count_digit_constraints(1, 100, mod=7, rem=0), 14)
        self.assertEqual(C.count_digit_constraints(1, 9999, increasing=True), 9 + 36 + 84 + 126)       # strictly increasing digits, 1 to 4 of them from 1..9
        self.assertEqual(C.count_digit_constraints(5, 4), 0)
        self.assertEqual(C.count_digit_constraints(1, 7, base=2), 7)
        with self.assertRaises(BadInput):
            C.count_digit_constraints(0, 5)
        with self.assertRaises(NotCovered):
            C.count_digit_constraints(1, 10 ** 61)


def _walks_slow(A, n, s, e):
    m = len(A)
    v = [int(i == s) for i in range(m)]
    for _ in range(n):
        v = [sum(v[i] * A[i][j] for i in range(m)) for j in range(m)]
    return v[e]


K4 = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
C5 = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 4)]


class Graphs(unittest.TestCase):
    def test_spanning_trees(self):
        self.assertEqual(C.spanning_trees(4, K4), 16)
        self.assertEqual(C.spanning_trees(5, C5), 5)
        self.assertEqual(C.spanning_trees(1, []), 1)
        self.assertEqual(C.spanning_trees(3, [(0, 1)]), 0)                                    # disconnected
        self.assertEqual(C.spanning_trees(2, [(0, 1), (0, 1), (1, 0)], multi=True), 3)
        with self.assertRaises(BadInput):
            C.spanning_trees(2, [(0, 1), (0, 1)])
        with self.assertRaises(BadInput):
            C.spanning_trees(3, [(0, 0)])
        with self.assertRaises(BadInput):
            C.spanning_trees(3, [(0, 5)])
        # Cayley through the determinant, K_8
        k8 = [(u, v) for u in range(8) for v in range(u + 1, 8)]
        self.assertEqual(C.spanning_trees(8, k8), 8 ** 6)

    def test_cayley(self):
        self.assertEqual(C.labelled_trees(5), 125)
        self.assertEqual(C.labelled_trees(4, [1, 1, 2, 2]), 2)
        self.assertEqual(C.labelled_trees(4, [3, 1, 1, 1]), 1)
        self.assertEqual(C.labelled_trees(4, [1, 1, 1, 1]), 0)
        self.assertEqual(C.labelled_trees(1), 1)
        self.assertEqual(C.rooted_forests(4, 2), 8)
        self.assertEqual(C.rooted_forests(5, 1), 125)
        self.assertEqual(C.rooted_forests(3, 3), 1)

    def test_matchings(self):
        self.assertEqual(C.perfect_matchings(6), 15)
        self.assertEqual(C.perfect_matchings(5), 0)
        self.assertEqual(C.perfect_matchings(4, [(0, 1), (1, 2), (2, 3), (3, 0)]), 2)
        self.assertEqual(C.perfect_matchings(22, [(i, i + 1) for i in range(21)]), 1)
        self.assertEqual(C.perfect_matchings(0, []), 1)
        self.assertEqual(C.matchings_count(3, [(0, 1), (1, 2), (0, 2)]), 4)
        self.assertEqual(C.matchings_count(4, K4), 10)
        self.assertEqual(C.matchings_count(4, K4, 2), 3)
        self.assertEqual(C.permutations_allowed([[1, 1, 0], [1, 0, 1], [0, 1, 1]]), 2)
        self.assertEqual(C.permutations_allowed([[1] * 4] * 4), 24)
        self.assertEqual(C.permutations_allowed([[0, 1], [0, 1]]), 0)
        with self.assertRaises(NotCovered):
            C.perfect_matchings(24, [(0, 1)])

    def test_independent_sets(self):
        self.assertEqual(C.independent_sets(4, [(0, 1), (1, 2), (2, 3)]), 8)
        self.assertEqual(C.independent_sets(5, C5), 11)
        self.assertEqual(C.independent_sets(4, K4[:0]), 16)
        self.assertEqual(C.independent_sets(4, K4, 'poly'), [1, 4])
        self.assertEqual(C.independent_sets(5, C5, 2), 5)
        self.assertEqual(C.independent_sets(0, []), 1)
        self.assertEqual(C.independent_sets(45, [(i, i + 1) for i in range(44)]), 2971215073)       # Fibonacci F(47)

    def test_chromatic(self):
        self.assertEqual(C.chromatic_polynomial(3, [(0, 1), (1, 2), (0, 2)]), [0, 2, -3, 1])
        self.assertEqual(C.chromatic_polynomial(3, []), [0, 0, 0, 1])
        self.assertEqual(C.chromatic_polynomial(0, []), [1])
        self.assertEqual(C.chromatic_polynomial(4, K4), [0, -6, 11, -6, 1])
        self.assertEqual(C.colorings_count(5, C5, 3), 30)
        self.assertEqual(C.colorings_count(5, C5, 2), 0)
        self.assertEqual(C.colorings_count(6, [(0, 1), (1, 2), (3, 4)], 3), 3 * 2 * 2 * 3 * 2 * 3)
        self.assertEqual(C.colorings_count(4, K4, 4), 24)
        petersen = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0), (0, 5), (1, 6), (2, 7), (3, 8), (4, 9), (5, 7), (7, 9), (9, 6), (6, 8), (8, 5)]
        self.assertEqual(C.colorings_count(10, petersen, 3), 120)
        with self.assertRaises(NotCovered):
            C.chromatic_polynomial(14, [(0, 1)])

    def test_automorphisms(self):
        self.assertEqual(C.graph_automorphisms(4, [(0, 1), (1, 2), (2, 3), (3, 0)]), 8)
        self.assertEqual(C.graph_automorphisms(4, K4), 24)
        self.assertEqual(C.graph_automorphisms(3, [(0, 1), (1, 2)]), 2)
        self.assertEqual(C.graph_automorphisms(0, []), 1)


class InclusionExclusion(unittest.TestCase):
    def test_union(self):
        self.assertEqual(C.union_size({(0,): 10, (1,): 8, (0, 1): 3}), 15)
        with self.assertRaises(BadInput):
            C.union_size({(): 3})

    def test_divisible_and_coprime(self):
        self.assertEqual(C.count_divisible(100, [2, 3, 5]), 74)
        self.assertEqual(C.count_divisible(100, [2, 3, 5], exactly=0), 26)
        self.assertEqual(C.count_divisible(100, [2, 3, 5], exactly=3), 3)
        self.assertEqual(sum(C.count_divisible(100, [2, 3, 5], exactly=j) for j in range(4)), 100)
        self.assertEqual(C.count_divisible(100, [2, 3], lo=50), 26 + 17 - 8)
        self.assertEqual(C.count_divisible(5, [10]), 0)
        self.assertEqual(C.count_coprime(100, 30), 26)
        self.assertEqual(C.count_coprime(30, 30), 8)
        self.assertEqual(C.count_coprime(10 ** 12, 6 * 5 * 7), 10 ** 12 * 1 * 2 * 4 * 6 // 210)
        with self.assertRaises(BadInput):
            C.count_divisible(10, [0])


class BruteForce(unittest.TestCase):
    def test_counts(self):
        self.assertEqual(C.brute_count(['1..6', '1..6'], 'x0 + x1 == 7'), 6)
        self.assertEqual(C.brute_count('1..4', 'x0 < x1', 'combinations', 2), 6)
        self.assertEqual(C.brute_count('1..4', 'x0 != x1', 'permutations', 2), 12)
        self.assertEqual(C.brute_count('1..3', 'x0 <= x1', 'multisets', 2), 6)
        self.assertEqual(C.brute_count('1..5', 'sum(v) == 6', space='subsets'), 3)
        self.assertEqual(C.brute_count([[1, 2], [3, 4]], 'x0 + x1 == 5'), 2)
        self.assertEqual(C.brute_count(['1..9', '1..9'], 'a * b == 12', names=['a', 'b']), 4)
        self.assertEqual(C.brute_count(['1..3'] * 3, 'len(set(v)) == 3 and x0 < x1'), 3)
        self.assertEqual(C.predicate_histogram(['1..30'], ['x0 % 2 == 0', 'x0 % 3 == 0']), {0: 10, 1: 15, 2: 5})

    def test_the_language_is_closed(self):
        for bad in ['__import__("os").system("x")', 'x0.real', '(lambda: 1)()', '[y for y in v]', 'x0 ** x1', 'x0 / 2', 'open("f")', 'x0 ** 40', 'y0 == 1',
                    'eval("1")', 'x0 == 1.5', 'x0 == "a"', '', 'x0 +', 'v.count(1)', 'x0 << 3', 'print(1)']:
            with self.assertRaises(BadInput, msg=bad):
                C.brute_count(['1..3', '1..3'], bad)

    def test_nothing_hangs_or_explodes(self):
        with self.assertRaises(BadInput):
            C.brute_count(['1..3', '1..3'], 'len(v * 1000000000) > 0')
        with self.assertRaises(BadInput):
            C.brute_count(['1..3', '1..3'], 'comb(10**18, 5) > 0')
        with self.assertRaises(BadInput):
            C.brute_count(['1..3', '1..3'], 'x0 // 0 == 1')
        with self.assertRaises(NotCovered):
            C.brute_count(['1..10000'] * 3, 'x0 < x1')
        with self.assertRaises(NotCovered):
            C.brute_count('1..30', 'sum(v) == 3', space='subsets')

    def test_probability_and_expectation(self):
        self.assertEqual(C.prob_uniform(['1..6', '1..6'], 'x0 + x1 == 7'), F(1, 6))
        self.assertEqual(C.prob_uniform(['1..6', '1..6'], 'x0 + x1 >= 10', given='x0 == 6'), F(1, 2))
        with self.assertRaises(BadInput):
            C.prob_uniform(['1..6', '1..6'], 'x0 == 1', given='x0 > 6')
        d = C.expectation_uniform(['1..6', '1..6'], 'max(x0, x1)')
        self.assertEqual(d['mean'], F(161, 36))
        self.assertEqual(d['variance'], F(2555, 1296))
        self.assertEqual(d['count'], 36)


class Probability(unittest.TestCase):
    def test_dice(self):
        d = C.dice_sum_distribution(2)
        self.assertEqual(d, {2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 5, 9: 4, 10: 3, 11: 2, 12: 1})
        self.assertEqual(C.dice_sum_distribution(2, faces=[0, 1]), {0: 1, 1: 2, 2: 1})
        self.assertEqual(C.dice_sum_probability(2, 6, 7), F(1, 6))
        self.assertEqual(C.dice_sum_probability(3, 6, 10, 11), F(1, 4))
        self.assertEqual(C.dice_sum_probability(0, 6, 0), F(1))
        self.assertEqual(C.dice_extreme_distribution(2, 3, 'max'), {1: 1, 2: 3, 3: 5})
        self.assertEqual(C.dice_extreme_distribution(2, 3, 'min'), {1: 5, 2: 3, 3: 1})

    def test_distributions(self):
        s = C.dist_stats([(1, 1), (2, 1), (3, 1)])
        self.assertEqual((s['mean'], s['variance']), (2, F(2, 3)))
        self.assertEqual(C.dist_stats([(5, 0), (7, 3)])['mean'], 7)
        with self.assertRaises(BadInput):
            C.dist_stats([(1, 0)])
        with self.assertRaises(BadInput):
            C.dist_stats([(1, -1), (2, 3)])
        self.assertEqual(C.bayes_posterior([F(1, 2), F(1, 2)], [F(1, 4), F(1, 8)]), [F(2, 3), F(1, 3)])
        with self.assertRaises(BadInput):
            C.bayes_posterior([F(1, 2), F(1, 3)], [F(1, 4), F(1, 8)])
        with self.assertRaises(BadInput):
            C.bayes_posterior([F(1, 2), F(1, 2)], [0, 0])
        with self.assertRaises(BadInput):
            C.dist_stats([(0.5, 1)])                                                    # a float is not exact

    def test_named_distributions(self):
        self.assertEqual(C.binomial_probability(10, '1/2', 5), F(63, 256))
        self.assertEqual(C.binomial_probability(3, F(1, 3), 1, 3), 1 - F(8, 27))
        self.assertEqual(C.binomial_probability(5, 1, 5), 1)
        self.assertEqual(C.hypergeometric_probability(52, 13, 5, 2), F(9139, 33320))
        self.assertEqual(C.hypergeometric_probability(10, 3, 4, 0, 4), 1)
        self.assertEqual(C.multivariate_hypergeometric([4, 4, 4], [2, 1, 1]), F(32, 165))
        self.assertEqual(C.multivariate_hypergeometric([1, 1], [2, 0]), 0)
        g = C.geometric_distribution('1/6', 3)
        self.assertEqual((g['pmf'], g['mean'], g['variance']), (F(25, 216), 6, 30))
        self.assertEqual(g['cdf'] + g['tail'], 1)

    def test_coupons_ruin_birthday(self):
        self.assertEqual(C.coupon_collector(6)['mean'], F(147, 10))
        self.assertEqual(C.coupon_collector(probs=['1/2', '1/2'])['mean'], 3)
        self.assertEqual(C.coupon_collector(probs=['1/2', '1/2'])['variance'], 2)
        self.assertEqual(C.coupon_collector(5, 3)['mean'], F(5, 5) + F(5, 4) + F(5, 3))
        self.assertEqual(C.coupon_collector_probability(2, 3), F(3, 4))
        self.assertEqual(C.coupon_collector_probability(3, 2), 0)
        self.assertEqual(C.coupon_collector_probability(0, 0), 1)
        r = C.gamblers_ruin(3, 10, '1/2')
        self.assertEqual((r['win'], r['expected_duration']), (F(3, 10), 21))
        r = C.gamblers_ruin(1, 2, F(2, 3))
        self.assertEqual(r['win'], F(2, 3))
        self.assertEqual(r['expected_duration'], 1)
        self.assertEqual(C.gamblers_ruin(0, 5, '1/3')['ruin'], 1)
        with self.assertRaises(BadInput):
            C.gamblers_ruin(1, 5, 1)
        b = C.birthday_probability(23, 365)
        self.assertTrue(F(1, 2) < b['collision'] < F(51, 100))
        self.assertEqual(C.birthday_probability(366, 365)['collision'], 1)
        self.assertEqual(C.birthday_probability(0, 5)['distinct'], 1)

    def test_markov(self):
        P = [[1, 0, 0], [F(1, 2), 0, F(1, 2)], [0, 0, 1]]
        r = C.markov_absorption(P)
        self.assertEqual(r['probabilities'][1], {0: F(1, 2), 2: F(1, 2)})
        self.assertEqual(r['expected_steps'][1], 1)
        # a fair walk on 0..4 with walls: from 2 it takes 4 steps on average and each wall is equally likely
        W = [[1, 0, 0, 0, 0]] + [[F(1, 2) if j in (i - 1, i + 1) else 0 for j in range(5)] for i in (1, 2, 3)] + [[0, 0, 0, 0, 1]]
        r = C.markov_absorption(W)
        self.assertEqual(r['expected_steps'][2], 4)
        self.assertEqual(r['probabilities'][1], {0: F(3, 4), 4: F(1, 4)})
        with self.assertRaises(NotCovered):
            C.markov_absorption([[1, 0, 0], [0, F(1, 2), F(1, 2)], [0, F(1, 2), F(1, 2)]])         # states 1, 2 never absorb
        with self.assertRaises(BadInput):
            C.markov_absorption([[F(1, 2), F(1, 3)], [0, 1]])
        self.assertEqual(C.markov_hitting_time([[F(1, 2), F(1, 2)], [F(1, 2), F(1, 2)]], [1], 0), 2)
        h = C.markov_hitting_time([[1, 0, 0], [F(1, 2), 0, F(1, 2)], [0, 0, 1]], [2])
        self.assertEqual(h, {0: 'inf', 1: 'inf', 2: 0})
        self.assertEqual(C.markov_stationary([[F(1, 2), F(1, 2)], [F(1, 4), F(3, 4)]]), [F(1, 3), F(2, 3)])
        self.assertEqual(C.markov_stationary([[F(1, 2), F(1, 2), 0], [0, 1, 0], [0, F(1, 3), F(2, 3)]]), [0, 1, 0])
        with self.assertRaises(NotCovered):
            C.markov_stationary([[1, 0], [0, 1]])                                                  # two closed classes
        self.assertEqual(C.markov_distribution([[0, 1], [1, 0]], 0, 3), [0, 1])
        self.assertEqual(C.markov_distribution([[F(1, 2), F(1, 2)], [F(1, 2), F(1, 2)]], [F(1, 3), F(2, 3)], 4), [F(1, 2), F(1, 2)])

    def test_patterns_in_a_coin_sequence(self):
        self.assertEqual(C.expected_wait_pattern('HH'), 6)
        self.assertEqual(C.expected_wait_pattern('HT'), 4)
        self.assertEqual(C.expected_wait_pattern('HHH'), 14)
        self.assertEqual(C.expected_wait_pattern('HTH'), 10)
        r = C.pattern_race(['HH', 'TH'])
        self.assertEqual(r['probabilities'], [F(1, 4), F(3, 4)])
        self.assertEqual(r['expected_length'], 3)
        self.assertEqual(C.pattern_race(['HHT', 'THH'])['probabilities'], [F(1, 4), F(3, 4)])      # THH beats HHT 3 to 1
        self.assertEqual(sum(C.pattern_race(['HTT', 'TTH', 'HHH'])['probabilities']), 1)
        self.assertEqual(C.pattern_race(['HH'])['expected_length'], C.expected_wait_pattern('HH'))
        with self.assertRaises(NotCovered):
            C.pattern_race(['HH', 'HHT'])                                                           # one contains the other

    def test_random_walks(self):
        d = C.walk_distribution([(1, 1), (-1, 1)], 4)
        self.assertEqual(d, {-4: F(1, 16), -2: F(4, 16), 0: F(6, 16), 2: F(4, 16), 4: F(1, 16)})
        d = C.walk_distribution([(1, 1), (-1, 1)], 6, lo=-1, hi=2)
        self.assertEqual(sum(d.values()), 1)
        self.assertTrue(set(d) <= {-1, 0, 1, 2})
        self.assertGreater(d[-1], 0)
        d = C.walk_distribution([((1, 0), 1), ((0, 1), 1)], 2)
        self.assertEqual(d[(1, 1)], F(1, 2))
        self.assertEqual(C.walk_distribution([(1, 1)], 0), {0: 1})

    def test_chords(self):
        self.assertTrue(C.chords_cross(1, 3, 2, 4))
        self.assertFalse(C.chords_cross(1, 2, 3, 4))
        self.assertFalse(C.chords_cross(1, 3, 3, 4))          # a shared end point
        self.assertTrue(C.chords_cross(4, 2, 1, 3))
        self.assertEqual(C.crossing_pairs([(1, 3), (2, 4), (5, 6)]), 1)
        self.assertEqual(C.chord_crossing_probability('four_points'), F(1, 3))
        self.assertEqual(C.chord_crossing_probability('two_distinct_chords', 6), F(1, 7))
        self.assertEqual(C.chord_crossing_probability('independent_chords', 4), F(2, 36))
        self.assertEqual(C.chord_crossing_probability('independent_points', 4), F(24, 3 * 256))
        self.assertEqual(C.matching_crossings(3), {'matchings': 15, 'non_crossing': 5, 'probability_non_crossing': F(1, 3), 'expected_crossings': F(1)})
        with self.assertRaises(BadInput):
            C.chord_crossing_probability('nope', 5)


class Bounds(unittest.TestCase):
    def test_past_the_bound_is_a_refusal_not_a_hang(self):
        for fn, args, kw in [
            (C.surjections, (10 ** 6, 5), {}), (C.derangements, (10 ** 6,), {}), (C.partition_count, (10 ** 6,), {}), (C.catalan, (10 ** 7,), {}),
            (C.necklace_count, (10 ** 6, 3), {}), (C.chromatic_polynomial, (20, [(0, 1)]), {}), (C.independent_sets, (60, [(0, 1)]), {}),
            (C.spanning_trees, (200, [(0, 1)]), {}), (C.grid_colorings, (30, 30, 4), {}), (C.domino_tilings, (30, 30), {}),
            (C.anagram_count, ('ab' * 300,), {}), (C.coupon_collector, (), {'probs': [F(1, 25)] * 25}), (C.stars_and_bars, (10 ** 7, 3), {}),
            (C.markov_distribution, ([[1]], 0, 10 ** 9), {}),
        ]:
            with self.assertRaises(NotCovered, msg=fn.__name__):
                fn(*args, **kw)

    def test_results_are_exact_types(self):
        self.assertIsInstance(C.derangements(5), int)
        self.assertIsInstance(C.binomial_probability(3, '1/2', 1), F)
        self.assertIsInstance(C.gamblers_ruin(2, 5, '1/2')['win'], F)
        with self.assertRaises(BadInput):
            C.binomial_probability(3, 0.5, 1)                                                      # a float probability is not exact
        with self.assertRaises(BadInput):
            C.derangements(-1)

    def test_deterministic(self):
        a = C.markov_absorption([[1, 0, 0], [F(1, 3), F(1, 3), F(1, 3)], [0, 0, 1]])
        b = C.markov_absorption([[1, 0, 0], [F(1, 3), F(1, 3), F(1, 3)], [0, 0, 1]])
        self.assertEqual(a, b)
        self.assertEqual(list(C.dice_sum_distribution(3)), sorted(C.dice_sum_distribution(3)))


if __name__ == '__main__':
    unittest.main()
