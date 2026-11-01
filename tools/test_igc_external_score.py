import unittest
from tools.igc.external_score import final_number, question_id


class FinalNumber(unittest.TestCase):
    def test_boxed_wins(self):
        self.assertEqual(final_number('x</think>**10** ways\n\[\n5 \times 45 = 225\n\]\n\[\n\boxed{225}\n\]<|im_end|>'), 225)

    def test_boxed_with_a_dollar(self):
        self.assertEqual(final_number('x</think>\\[\n\\boxed{\\$165{,}762{,}163}\n\\]\n\nNote: 6139339 is not divisible by 4.<|im_end|>'), 165762163)

    def test_bold_in_the_working_is_not_the_answer(self):
        self.assertEqual(final_number('x</think>- **2** hide: 17 − 2 = 15<|im_end|>\n'), 2)
        self.assertEqual(final_number('x</think>- **2** hide\nSo 17 − 2 = 15<|im_end|>'), 15)

    def test_closing_bold(self):
        self.assertEqual(final_number('x</think>Claire eats **7 dozen eggs** in 4 weeks.<|im_end|>'), 7)
        self.assertEqual(final_number('x</think>**12 + 11 = 23**'), 23)

    def test_another_form_of_the_answer(self):
        self.assertEqual(final_number('x</think>Total: **40 + 20 + 100 = 160 minutes**, or **2 hours 40 minutes**.'), 160)
        self.assertEqual(final_number('x</think>So she spent **70 minutes**, or **1 hour and 10 minutes**.'), 70)

    def test_closing_condition_is_not_the_answer(self):
        self.assertEqual(final_number('x</think>So there are **3 ways**.\n\nIf both boxes must be nonempty, then there would be **2 ways**.'), 3)
        self.assertEqual(final_number('x</think>The most left over is **2**.\n\nIf he had 3 or more left, he could add one.'), 2)
        self.assertEqual(final_number('x</think>\\(\\log_2 64 = 6\\), since \\(2^6 = 64\\).'), 6)
        self.assertEqual(final_number('x</think>\\[\n\\log_2 64 = 6\n\\]\n\nsince \\(2^6 = 64\\).'), 6)

    def test_closing_condition_can_be_the_last_step(self):
        self.assertEqual(final_number('x</think>Ryan planted \\(2 \\times 15 = 30\\) flowers.\n\nIf 5 did not grow, he has \\(30 - 5 = 25\\) flowers.'), 25)

    def test_closing_line_led_by_the_number(self):
        self.assertEqual(final_number('x</think>5 cars drove through in the first 15 minutes.'), 5)
        self.assertEqual(final_number('x</think>50 + 70 = 120 friends.'), 120)

    def test_unfinished(self):
        self.assertIsNone(final_number('never closed'))


class QuestionId(unittest.TestCase):
    def test_follows_the_question(self):
        self.assertEqual(question_id('math', 'What is 2+2?'), question_id('math', 'What is 2+2?'))
        self.assertNotEqual(question_id('math', 'What is 2+2?'), question_id('math', 'What is 2+3?'))
        self.assertTrue(question_id('gsm8k', 'q').startswith('gsm8k-'))


if __name__ == '__main__':
    unittest.main()
