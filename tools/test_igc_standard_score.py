"""tools/igc/standard_score.py: the HLE rule. python tools/test_igc_standard_score.py"""
import sys, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.igc.standard_score import hle_right


def reply(line):
    return '<think>\nwork\n</think>\n\nExplanation: x\n' + line + '\nConfidence: 60%'


class Hle(unittest.TestCase):
    def test_one_value(self):
        self.assertTrue(hle_right(reply('Exact Answer: 5'), '5'))
        self.assertTrue(hle_right(reply('Exact Answer: n = 5'), '5'))
        self.assertTrue(hle_right(reply('Exact Answer: \\frac{3}{4}'), '3/4'))
        self.assertTrue(hle_right(reply('Exact Answer: 1,024 ways'), '1024'))
        self.assertTrue(hle_right(reply('Exact Answer: 3.14'), '3.15'))   # one unit in the gold's last place

    def test_hedged_line_is_no_answer(self):
        # hle-on1 6721b01e63e0 (#288): the bench rule took the first number of a hedged line and scored it right
        self.assertFalse(hle_right(reply('Exact Answer: cannot be determined; at most 5, possibly 0'), '5'))
        self.assertFalse(hle_right(reply('Exact Answer: 5 or 0'), '5'))
        self.assertFalse(hle_right(reply('Exact Answer: between 3 and 5'), '5'))

    def test_caveat_in_brackets_keeps_the_value(self):
        # hle-on1 670147f7dc19: the reader scored it right, the value before the bracket is the answer
        self.assertTrue(hle_right(reply('Exact Answer: 3 (for smooth functions; 4 if Morse/nondegenerate is required)'), '3'))

    def test_repeated_value_is_one_value(self):
        self.assertTrue(hle_right(reply('Exact Answer: 5 (that is, 5 rooms)'), '5'))

    def test_no_line_no_answer(self):
        self.assertFalse(hle_right('<think>a</think>\n\nThe answer is 5.', '5'))



class Math(unittest.TestCase):
    def test_final_answer_line_with_the_value_below(self):
        from tools.igc.standard_score import math_right
        text = '</think>\n\nThe final answer is:\n\\boxed{-8038578.5}<|im_end|>'
        self.assertTrue(math_right(text, '-16077157/2'))   # dmx_div_big-1649: was scored on `:`

    def test_final_answer_sentence_still_wins(self):
        from tools.igc.standard_score import math_right
        self.assertTrue(math_right('</think>\n\nThe final answer is 12. I hope it is correct.', '12'))

    def test_the_same_number_written_as_another_expression(self):
        from tools.igc.standard_score import math_right
        self.assertTrue(math_right('</think>\n\n\\boxed{26! 2^{25}}<|im_end|>', '2^{25}\\cdot26!'))   # r1-ctrl a6f7b142
        self.assertFalse(math_right('</think>\n\n\\boxed{26! 2^{24}}<|im_end|>', '2^{25}\\cdot26!'))
        self.assertFalse(math_right('</think>\n\n\\boxed{598.5377138413301}<|im_end|>', '1539439/2572'))   # a rounded quotient stays wrong


if __name__ == '__main__':
    unittest.main()
