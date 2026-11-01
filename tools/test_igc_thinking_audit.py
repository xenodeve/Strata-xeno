import unittest
from tools.igc.external_score import repeating
from tools.igc.thinking_audit import audit


class Repeating(unittest.TestCase):
    def test_exact_cycle_is_stuck(self):
        self.assertTrue(repeating(list(range(100)) + [7, 8, 9, 10, 11] * 40))

    def test_long_cycle_is_stuck(self):
        self.assertTrue(repeating(list(range(500)) * 20))

    def test_a_few_rounds_can_still_leave(self):
        self.assertFalse(repeating(list(range(100)) + [7, 8, 9] * 10))

    def test_a_cycle_past_512_tokens_takes_eight_rounds(self):
        cycle = [i * i % 9973 for i in range(517)]
        self.assertTrue(repeating(list(range(100)) + cycle * 8))
        self.assertFalse(repeating(list(range(100)) + cycle * 7))

    def test_varied_text_runs_on(self):
        self.assertFalse(repeating([i * i % 9973 for i in range(5000)]))


class Audit(unittest.TestCase):
    def test_first_sight_of_the_answer(self):
        row = audit('One step. So 6 + 6 = 12. Wait, check again. 12 is right.</think>**12**', 12)
        self.assertEqual(row['sentences'], 4)
        self.assertEqual(row['first_sentence'], 1)
        self.assertEqual(row['moves']['recheck'], 1)

    def test_answer_as_part_of_a_number_does_not_count(self):
        self.assertIsNone(audit('Then x = 120 and y = 312.</think>x', 12)['first_sentence'])

    def test_a_number_of_the_question_is_not_the_answer(self):
        row = audit('She has 12 pens and 12 cards. Add them. So 12 + 12 = 24.</think>x', 12)
        self.assertIsNone(row['first_sentence'])

    def test_an_outcome_word_counts(self):
        self.assertEqual(audit('Add them. Total is $24.</think>x', 24)['first_sentence'], 1)

    def test_repeat_share(self):
        row = audit('Could be 5. Could be 5. Could be 5. Done now.</think>x', 5)
        self.assertEqual(row['repeated_sentences'], 2)


if __name__ == '__main__':
    unittest.main()
