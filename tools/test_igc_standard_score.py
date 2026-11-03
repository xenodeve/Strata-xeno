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


if __name__ == '__main__':
    unittest.main()
