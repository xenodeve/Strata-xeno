"""question_read: the computation a question states. python -m unittest tools.test_igc_question_read"""
import unittest
from fractions import Fraction

from tools.igc.question_read import read

FORMAT = '\n\nGive the final answer as a single number in \\boxed{} at the end.'


def value(q):
    got = read(q + FORMAT)
    return None if got is None else got[1]


class Read(unittest.TestCase):
    def test_the_english_forms_and_their_order(self):
        self.assertEqual(value('What is 3 less than 7?'), 4)                 # 7 - 3, not 3 - 7 (pc4-whole add_or_sub 5180)
        self.assertEqual(value('Subtract 3 from 7'), 4)
        self.assertEqual(value('What is 7 take away -3?'), 10)
        self.assertEqual(value('What is the distance between 20412938679033416 and -0.4?'), Fraction('20412938679033416.4'))  # 6634
        self.assertEqual(value('What is the difference between 1508489756117 and -29.392?'), Fraction('1508489756146.392'))
        self.assertEqual(value('Product of -3388.02 and 32790'), Fraction('-111093175.8'))
        self.assertEqual(value('Divide 10 by -4'), Fraction(-5, 2))
        self.assertEqual(value('Sum 55839343876742 and -8.'), 55839343876734)
        self.assertEqual(value('Put together 2 and 5'), 7)

    def test_an_expression_as_written(self):
        self.assertEqual(value('What is 20*(-8)/(-60)*329*9/((-36)/2)*7*(-6)/49?'), 376)
        self.assertEqual(value('(-1*-638*(-7)/(-7))/(((-62)/(-279))/(((-170)/3060)/(1/(-2))))'), 319)

    def test_no_single_computation_is_none(self):
        self.assertIsNone(value('Tom has 3 apples and buys 4 more. How many does he have?'))
        self.assertIsNone(value('Solve 2x + 3 = 7 for x.'))
        self.assertIsNone(value('What is 12?'))


if __name__ == '__main__':
    unittest.main()
