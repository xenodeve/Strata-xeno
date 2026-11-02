"""The LiveCodeBench checker on code the test wrote itself (never model code: that runs only on a throwaway VM)."""
import json
from pathlib import Path
import tempfile
import unittest
from tools.igc import lcb_score

MODPOW = 'class Solution:\n    def inv(self, a: int) -> int:\n        return pow(a, 10**9 + 5, 10**9 + 7)\n'
STDIN = 'n = int(input())\nprint(pow(n, 2, 7))\n'
BROKEN = 'class Solution:\n    def inv(self, a: int) -> int:\n        return 1 // 0\n'


class Check(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.runner = str(Path(self.dir.name) / 'runner.py')
        Path(self.runner).write_text(lcb_score.CHECK, encoding='utf-8')

    def tearDown(self):
        self.dir.cleanup()

    def one(self, source, case, kind, name=''):
        return lcb_score.one((source, case, kind, name, self.runner))

    def test_three_argument_pow_is_the_builtin(self):
        case = dict(input='2', output=json.dumps(500000004))
        self.assertEqual(self.one(MODPOW, case, 'functional', 'inv'), 'ok')
        self.assertEqual(self.one(STDIN, dict(input='5\n', output='4\n'), 'stdin'), 'ok')

    def test_the_official_header_names_and_recursion(self):
        uses_math = 'class Solution:\n    def f(self, a: int) -> int:\n        return math.isqrt(a)\n'
        self.assertEqual(self.one(uses_math, dict(input='17', output='4'), 'functional', 'f'), 'ok')
        deep = 'import sys\ndef d(n):\n    return 0 if n == 0 else 1 + d(n - 1)\nprint(d(int(input())))\n'
        self.assertEqual(self.one(deep, dict(input='5000\n', output='5000\n'), 'stdin'), 'ok')

    def test_stdin_lines_equal_as_decimals(self):
        self.assertEqual(self.one('print("1.50 2")\n', dict(input='', output='1.5 2.0\n'), 'stdin'), 'ok')
        self.assertTrue(self.one('print("1.5")\nprint(3)\n', dict(input='', output='1.5\n'), 'stdin').startswith('wrong'))

    def test_an_exception_says_which(self):
        verdict = self.one(BROKEN, dict(input='2', output='1'), 'functional', 'inv')
        self.assertTrue(verdict.startswith('error: ZeroDivisionError'), verdict)


if __name__ == '__main__':
    unittest.main()
