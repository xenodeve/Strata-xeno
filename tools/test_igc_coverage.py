"""coverage.py: caught / should have been caught, from what a rollout wrote. python -m unittest tools.test_igc_coverage"""
import unittest

from tools.igc import coverage, tir4b
from tools.test_igc_tir4b import Kernel, TOK, play, rollout


def covered(r):
    rec = r.record(lambda row, text: False)
    return coverage.row_coverage(TOK, rec, dict(out_ids=r.out, spans=r.spans))


class Coverage(unittest.TestCase):
    def test_a_second_statement_on_a_line_is_checked(self):
        from tools.igc.mathexpr import wrong_relations
        self.assertEqual(wrong_relations('so 12 * 13 = 156 and 7 * 8 = 54 then'), [('7 * 8', '54', '56')])
        self.assertEqual(wrong_relations('$x = 3$, $y = 5$'), [])
        # r1-ctrl lines where the second statement is not a claim of its own
        self.assertEqual(wrong_relations('$S_b = \\frac{8}{11} S (1 - 7/18) = \\frac{8}{11} S \\frac{11}{18} = \\frac{8}{18} S = \\frac{4}{9} S$.'), [])
        self.assertEqual(wrong_relations('But $15/12 = 5/4 \\equiv 5 \\cdot 4^{-1} = 5 \\cdot 4 = 20 \\equiv 0$.'), [])
        self.assertEqual(wrong_relations('$m=33 \\implies (-66, 99)$. $12(4356) - (-66)(99) - 6(99^2) = 52272 + 6534 - 59406$.'),
                         [('12(4356) - (-66)(99) - 6(99^2)', '52272 + 6534 - 59406', '0')])   # 6 * 99^2 is 58806: a real slip

    def test_base_arm_every_point_is_by_hand(self):
        r = rollout()
        play(r, 'so 12 * 13 = 156 and 7 * 8 = 54 then</think>\n\nx<|im_end|>', Kernel([]))
        c = covered(r)
        self.assertEqual((c['points'], c['by_inline'], c['by_hand']), (2, 0, 2))
        self.assertEqual((c['wrong'], c['corrected']), (1, 0))   # 7 * 8 = 54: should have been caught, was not

    def test_inline_values_count_as_caught(self):
        r = rollout(inline_calc=True, inline_plain=True)
        play(r, 'so 12 * 13 = then 7 * 8 = and</think>\n\nx<|im_end|>', Kernel([]))
        c = covered(r)
        self.assertEqual((c['points'], c['by_inline'], c['by_hand']), (2, 2, 0))

    def test_a_correction_is_matched_to_the_wrong_line(self):
        r = rollout(check_hand=True)
        play(r, 'so 144 - 144 - 572 - 168 + 96 = -744.\nthen 2 + 2 = 4 and</think>\n\nx<|im_end|>', Kernel([]))
        c = covered(r)
        self.assertEqual((c['wrong'], c['corrected'], c['unmatched_checks']), (1, 1, 0))

    def test_a_stuck_answer_is_a_loop_with_no_program(self):
        r = rollout()
        r.done('stuck')
        c = covered(r)
        self.assertEqual((c['loops'], c['loop_programs'], c['stuck']), (1, 0, 1))


if __name__ == '__main__':
    unittest.main()
