import unittest
from fractions import Fraction
from tools.igc.value_audit import audit, evaluate


class ValueAudit(unittest.TestCase):
    def test_precedence(self):
        self.assertEqual(evaluate('4 \\times 125 - 2'), 498)
        self.assertEqual(evaluate('3+4*2'), 11)
        self.assertEqual(evaluate('1,200/8'), 150)
        self.assertEqual(evaluate('1/3'), Fraction(1, 3))

    def test_finds_a_wrong_value(self):
        self.assertEqual(audit('Then 4 \\times 125 - 2 = 1236 pounds')[1], [('4 \\times 125 - 2', '1236', 498.0)])
        self.assertEqual(audit('so 3*4=12 and 5+5=10.')[0:2], (2, []))

    def test_rounded_decimals_pass(self):
        self.assertEqual(audit('10/3=3.33')[1], [])

    def test_a_partial_expression_is_not_checked(self):
        self.assertEqual(audit('(2+3)*4=20')[0], 0)
        self.assertEqual(audit('x^2+1=5')[0], 0)
        self.assertEqual(audit('a+b=-1+5=4')[0], 0)
        self.assertEqual(audit('8,480,500-508,830=7,971,670')[1], [])
        self.assertEqual(audit('4*125 - 2 = 500 - 2 = 498')[1], [])


if __name__ == '__main__':
    unittest.main()
