import unittest
from tools.igc.output_integration import final_integer


class FinalInteger(unittest.TestCase):
    def test_thinking_or_truncated_output_is_not_a_verified_final_answer(self):
        self.assertIsNone(final_integer('The calculation gives74'))
        self.assertIsNone(final_integer('74</think> answer74 or75'))
        self.assertEqual(final_integer('ignore99</think>\nThe answer is -74.<|im_end|>'),-74)
        self.assertEqual(final_integer('</think>\n0'),0)


if __name__=='__main__':unittest.main()
