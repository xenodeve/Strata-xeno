"""livebench: the questions as rows and LiveBench's own scoring rules. python -m unittest tools.test_igc_livebench"""
import json
import tempfile
import unittest
from pathlib import Path

from tools.igc import livebench as lb

try:
    import sympy   # noqa: F401
    import lark    # noqa: F401
    HAVE_SYMPY = True
except ImportError:
    HAVE_SYMPY = False


def row(task, answer, subtask=None, question=''):
    return dict(group=task, subtask=subtask or task, answer=answer, question=question)


def say(text):
    """What the bench hands the scorer: the thinking, its close, the reply."""
    return 'some thinking\n</think>' + text


class Aime(unittest.TestCase):
    def test_the_solution_tag_and_the_tail(self):
        r = row('math_comp', '025', 'aime_i_2024')
        self.assertEqual(lb.score(r, say('so <solution>025</solution>')), 1)
        self.assertEqual(lb.score(r, say('The answer is 025')), 1)
        self.assertEqual(lb.score(r, say('The answer is 026')), 0)

    def test_nothing_before_the_close_counts(self):
        r = row('math_comp', '025', 'aime_i_2024')
        self.assertEqual(lb.score(r, 'the answer is 025 but the thinking never closed'), 0)


class Amc(unittest.TestCase):
    QUESTION = 'What is $3+4$? $\\textbf{(A) }5\\qquad\\textbf{(B) }6\\qquad\\textbf{(C) }7\\qquad\\textbf{(D) }8\\qquad\\textbf{(E) }9$'

    def test_the_letter_five_times_the_box_and_the_value(self):
        r = row('math_comp', 'C', 'amc_12a_2023', self.QUESTION)
        self.assertEqual(lb.score(r, say('CCCCC')), 1)
        self.assertEqual(lb.score(r, say('so \\boxed{C}')), 1)
        self.assertEqual(lb.score(r, say('it is 7')), 1)   # the choice's own text near the end
        self.assertEqual(lb.score(r, say('DDDDD')), 0)

    def test_the_updated_set_is_routed_the_same(self):
        r = row('math_comp', 'C', 'updated_amc_12a_2023', self.QUESTION)
        self.assertEqual(lb.score(r, say('CCCCC')), 1)


class Olympiad(unittest.TestCase):
    def test_the_share_of_the_order_put_back(self):
        r = row('olympiad', '1,6,7,2,3,4,5', 'usamo')
        self.assertEqual(lb.score(r, say('Answer: 1, 6, 7, 2, 3, 4, 5')), 1)
        self.assertAlmostEqual(lb.score(r, say('Answer: 1, 6, 7, 2, 3, 5, 4')), 1 - 2 / 7)   # two places differ
        self.assertEqual(lb.score(r, say('Answer: 7, 6, 1, 2, 3, 4, 5')), 1 - 2 / 7)


class Zebra(unittest.TestCase):
    def test_partial_credit_is_half_right_and_half_all_right(self):
        r = row('zebra_puzzle', 'a, b, c')
        self.assertEqual(lb.score(r, say('<solution>a, b, c</solution>')), 1)
        self.assertAlmostEqual(lb.score(r, say('<solution>a, b, x</solution>')), (0 + 2 / 3) / 2)
        self.assertEqual(lb.score(r, say('no answer here')), 0)

    def test_the_last_line_when_it_has_the_right_number_of_commas(self):
        self.assertEqual(lb.score(row('zebra_puzzle', 'a, b, c'), say('thinking\na, b, c')), 1)


class Spatial(unittest.TestCase):
    def test_bold_words_numbers_and_shapes(self):
        self.assertEqual(lb.score(row('spatial', '6'), say('so **six**')), 1)
        self.assertEqual(lb.score(row('spatial', '6'), say('so **7**')), 0)
        self.assertEqual(lb.score(row('spatial', 'triangle'), say('an **equilateral triangle**')), 1)
        self.assertEqual(lb.score(row('spatial', 'triangle'), say('**circle square triangle**')), 0)


class WebOfLies(unittest.TestCase):
    def test_bold_solution_and_plain(self):
        r = row('web_of_lies_v2', 'yes, no, unknown')
        self.assertEqual(lb.score(r, say('**yes, no, unknown**')), 1)
        self.assertEqual(lb.score(r, say('<solution>yes, no, unknown</solution>')), 1)
        self.assertEqual(lb.score(r, say('they are yes, no, unknown')), 1)
        self.assertEqual(lb.score(r, say('**yes, yes, unknown**')), 0)


class AmpsHard(unittest.TestCase):
    def test_the_same_text_needs_no_sympy(self):
        self.assertEqual(lb.score_detail(row('AMPS_Hard', '4 i \\sqrt{5}', 'amps_hard_geometric_mean'), say('\\boxed{4 i \\sqrt{5}}')), (1, 'string'))

    @unittest.skipUnless(HAVE_SYMPY, 'sympy and lark are not installed')
    def test_another_way_of_writing_it_is_decided_by_sympy(self):
        r = row('AMPS_Hard', '\\frac{1}{2}', 'amps_hard_gcd')
        self.assertEqual(lb.score_detail(r, say('\\boxed{0.5}')), (1, 'sympy'))

    @unittest.skipUnless(HAVE_SYMPY, 'sympy and lark are not installed')
    def test_a_different_value_is_not_proved_and_is_counted_as_unverified(self):
        s, note = lb.score_detail(row('AMPS_Hard', '\\frac{1}{2}', 'amps_hard_gcd'), say('\\boxed{3}'))
        self.assertEqual((s, note), (0, 'unverified'))   # the original asks an LLM here; this port asks nobody


class Rows(unittest.TestCase):
    def test_the_current_questions_are_kept_with_their_dates(self):
        old = dict(question_id='1', category='math', task='math_comp', subtask='aime_i_2024', turns=['q1'], ground_truth='025',
                   livebench_release_date='2024-06-24', livebench_removal_date='2024-08-31')
        new = dict(old, question_id='2', turns=['q2'], livebench_removal_date='')
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / 'math.jsonl').write_text(json.dumps(old) + '\n' + json.dumps(new) + '\n', encoding='utf-8')
            current = lb.rows(d)
            everything = lb.rows(d, current_only=False)
        self.assertEqual([r['question'] for r in current], ['q2'])
        self.assertEqual(len(everything), 2)
        self.assertEqual(current[0]['suite'], 'livebench')
        self.assertEqual(current[0]['answer'], '025')
        self.assertEqual(current[0]['prompt'], 'q2')

    def test_an_unknown_task_is_refused_not_scored_zero(self):
        with self.assertRaises(ValueError):
            lb.score(row('coding_completion', 'x'), say('x'))


if __name__ == '__main__':
    unittest.main()
