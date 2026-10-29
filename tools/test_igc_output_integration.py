import unittest
from tools.igc.output_integration import final_integer,interpreted_final


class FinalInteger(unittest.TestCase):
    def test_format_is_separate_from_explicit_result_and_equation_consistency(self):
        text='</think>\n12 * 64 = 768\nThe result is 768.<|im_end|>'
        self.assertIsNone(final_integer(text))
        self.assertEqual(interpreted_final(text,'(12) * (64)'),dict(value=768,equation_consistent=True,contradictory=False))
        self.assertEqual(interpreted_final('</think>**-74**','92 - 18')['value'],-74)
        self.assertIsNone(interpreted_final('</think>74 or75','92 - 18')['value'])
        self.assertIsNone(interpreted_final('</think>The answer is74.\nThe answer is75.','92 - 18')['value'])
        self.assertFalse(interpreted_final('</think>12*65=768\nThe result is768.','12*64')['equation_consistent'])
    def test_thinking_or_truncated_output_is_not_a_verified_final_answer(self):
        self.assertIsNone(final_integer('The calculation gives74'))
        self.assertIsNone(final_integer('74</think> answer74 or75'))
        self.assertEqual(final_integer('ignore99</think>\nThe answer is -74.<|im_end|>'),-74)
        self.assertEqual(final_integer('</think>\n0'),0)


if __name__=='__main__':unittest.main()
