"""contest_math: recent contest problems and their boxed exact answers. python -m unittest tools.test_igc_contest_math"""
import json
import tempfile
import unittest
from pathlib import Path

from tools.igc import contest_math as cm


def row(answer):
    return dict(suite='contest', answer=answer)


def say(text):
    return 'thinking\n</think>' + text


class Right(unittest.TestCase):
    def test_an_integer_a_fraction_and_the_forms_of_a_fraction(self):
        self.assertTrue(cm.right(row('70'), say('so \\boxed{70}')))
        self.assertTrue(cm.right(row('\\frac{13}{32}'), say('\\boxed{\\dfrac{13}{32}}')))
        self.assertTrue(cm.right(row('\\frac{13}{32}'), say('\\boxed{13/32}')))
        self.assertTrue(cm.right(row('\\frac{13}{32}'), say('\\boxed{0.40625}')))   # the same number

    def test_a_different_number_a_missing_box_and_an_unclosed_thinking(self):
        self.assertFalse(cm.right(row('70'), say('\\boxed{71}')))
        self.assertFalse(cm.right(row('70'), say('the answer is 70')))
        self.assertFalse(cm.right(row('70'), 'the thinking never closed \\boxed{70}'))

    def test_the_last_box_counts(self):
        self.assertTrue(cm.right(row('5'), say('first \\boxed{4}, on second thought \\boxed{5}')))
        self.assertFalse(cm.right(row('5'), say('first \\boxed{5}, on second thought \\boxed{4}')))

    def test_an_answer_mathexpr_cannot_read_is_compared_as_text(self):
        self.assertTrue(cm.right(row('(1, 2)'), say('\\boxed{(1,\\ 2)}'.replace('\\ ', ''))))
        self.assertFalse(cm.right(row('(1, 2)'), say('\\boxed{(2, 1)}')))


class Rows(unittest.TestCase):
    def test_a_row_per_problem_with_the_instruction(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / 'aime_2026.jsonl').write_text(json.dumps(dict(problem_idx=1, problem='What is 1+1?', answer=2)) + '\n', encoding='utf-8')
            rs = cm.rows(d)
        self.assertEqual(len(rs), 1)
        self.assertEqual((rs[0]['suite'], rs[0]['answer'], rs[0]['group']), ('contest', '2', 'aime_2026'))
        self.assertTrue(rs[0]['question'].startswith('What is 1+1?'))
        self.assertIn('\\boxed{}', rs[0]['question'])


if __name__ == '__main__':
    unittest.main()
