import unittest
from tools.igc.answer_check import prompt, triage


def record(id, answer, text):
    return dict(id=id, answer=answer, finish='stop', text=text, tokens=10, suite='gsm8k')


class Triage(unittest.TestCase):
    def rows(self, verdict, text='x</think>**5**', answer=5):
        return triage([record('a', answer, text)], {'a': dict(verdict=verdict, value=None, reason='r')})[0]

    def test_both_right_needs_no_reading(self):
        self.assertEqual(self.rows('same')['route'], 'agreed-right')

    def test_scorer_right_judge_wrong_is_read(self):
        self.assertEqual(self.rows('different')['route'], 'disagree')

    def test_scorer_wrong_judge_right_is_read(self):
        self.assertEqual(self.rows('same', answer=6)['route'], 'disagree')

    def test_both_wrong_asks_for_the_cause(self):
        self.assertEqual(self.rows('different', answer=6)['route'], 'agreed-wrong')

    def test_hedged_counts_as_wrong_for_the_judge(self):
        self.assertEqual(self.rows('hedged')['route'], 'disagree')
        self.assertEqual(self.rows('hedged', answer=6)['route'], 'agreed-wrong')

    def test_no_answer_never_goes_to_the_judge_and_needs_a_cause(self):
        row = triage([record('a', 5, 'thinking never closed')], {})[0]
        self.assertEqual(row['route'], 'agreed-wrong')

    def test_a_missing_verdict_is_read(self):
        self.assertEqual(triage([record('a', 5, 'x</think>**5**')], {})[0]['route'], 'disagree')


class Prompt(unittest.TestCase):
    def test_holds_no_thinking_and_no_arm(self):
        text = prompt(dict(question='How many?', gold=5, reply='SECRET THINKING</think>**5**'))
        self.assertNotIn('SECRET', text)
        self.assertIn('**5**', text)
        self.assertIn('How many?', text)


if __name__ == '__main__':
    unittest.main()
