import unittest
from tools.igc.external_score import final_number


class FinalNumber(unittest.TestCase):
    def test_boxed_wins(self):
        self.assertEqual(final_number('x</think>**10** ways\n\[\n5 \times 45 = 225\n\]\n\[\n\boxed{225}\n\]<|im_end|>'), 225)

    def test_bold_in_the_working_is_not_the_answer(self):
        self.assertEqual(final_number('x</think>- **2** hide: 17 − 2 = 15<|im_end|>\n'), 2)
        self.assertEqual(final_number('x</think>- **2** hide\nSo 17 − 2 = 15<|im_end|>'), 15)

    def test_closing_bold(self):
        self.assertEqual(final_number('x</think>Claire eats **7 dozen eggs** in 4 weeks.<|im_end|>'), 7)
        self.assertEqual(final_number('x</think>**12 + 11 = 23**'), 23)

    def test_unfinished(self):
        self.assertIsNone(final_number('never closed'))


if __name__ == '__main__':
    unittest.main()
