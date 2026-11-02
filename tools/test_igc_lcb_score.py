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

    def test_an_exception_says_which(self):
        verdict = self.one(BROKEN, dict(input='2', output='1'), 'functional', 'inv')
        self.assertTrue(verdict.startswith('error: ZeroDivisionError'), verdict)


if __name__ == '__main__':
    unittest.main()
