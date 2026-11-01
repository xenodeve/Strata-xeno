"""The #279 sandbox against CPython. Build tests/igc/sandbox_main.cpp and name it in IGC_SANDBOX_EXE; without it these skip."""
import os
from pathlib import Path
import tempfile
import unittest

from tools.igc.sandbox_oracle import compare, exception_class, expected_status, first_difference, humaneval_program, Ours, reason, Run, run_binary, run_corpus

CORPUS = Path(__file__).resolve().parents[1] / 'tests' / 'igc' / 'sandbox' / 'dev'
BINARY = os.environ.get('IGC_SANDBOX_EXE')


class Oracle(unittest.TestCase):
    def test_exception_class_is_the_last_line_at_column_zero(self):
        self.assertEqual(exception_class('Traceback (most recent call last):\n  File "x"\nValueError: bad\n'), 'ValueError')
        self.assertEqual(exception_class('Traceback (most recent call last):\n  File "x"\nKeyError\n'), 'KeyError')
        self.assertEqual(exception_class('  File "x", line 1\n    (\n    ^\nSyntaxError: invalid syntax\n'), 'SyntaxError')
        self.assertIsNone(exception_class(''))

    def test_expectation_is_read_from_the_first_line_only(self):
        self.assertEqual(expected_status('# expect: StepLimit\nwhile True:\n    pass\n'), 'StepLimit')
        self.assertIsNone(expected_status('x = 1\n# expect: StepLimit\n'))

    def test_verdicts(self):
        answer = Ours('1\n', status='Ok')
        self.assertEqual(compare('a', None, answer, Run('1\n')).kind, 'match')
        self.assertEqual(compare('a', None, answer, Run('2\n')).kind, 'MISMATCH')
        raised = Ours('', status='Raised', exception='TypeError', error='TypeError: x (line 1)')
        self.assertEqual(compare('a', None, raised, Run('', 'ValueError')).kind, 'MISMATCH')
        self.assertEqual(compare('a', None, raised, Run('', 'TypeError')).kind, 'match')

    def test_a_refusal_must_still_agree_on_what_was_printed(self):
        printed = Ours('wrong\n', status='Unsupported', error='unsupported: x (line 2)')
        self.assertEqual(compare('a', None, printed, Run('right\n')).kind, 'MISMATCH')
        self.assertEqual(compare('a', None, printed, Run('wrong\nmore\n')).kind, 'refused')

    def test_a_module_outside_the_allowlist_is_a_refusal(self):
        missing = Ours('', status='Raised', exception='ModuleNotFoundError', error="ModuleNotFoundError: No module named 'os' (line 1)")
        self.assertEqual(compare('a', None, missing, Run('anything\n')).kind, 'refused')
        self.assertEqual(compare('a', None, missing, Run('', 'ModuleNotFoundError')).kind, 'match')

    def test_faults_of_the_binary(self):
        self.assertEqual(compare('a', None, Ours(status=None, exit_code=-1073741571), Run('')).kind, 'CRASH')
        self.assertEqual(compare('a', None, Ours(timed_out=True), Run('')).kind, 'HANG')

    def test_the_expected_status_decides_a_limit_case(self):
        self.assertEqual(compare('a', 'StepLimit', Ours(status='StepLimit'), None).kind, 'expected')
        self.assertEqual(compare('a', 'StepLimit', Ours(status='Ok'), None).kind, 'MISMATCH')

    def test_first_difference_names_the_line(self):
        self.assertIn('line 2', first_difference('a\nb\n', 'a\nc\n'))

    def test_refused_feature_loses_its_line(self):
        self.assertEqual(reason('unsupported: class (line 12)'), 'unsupported: class')

    def test_humaneval_program_joins_the_parts(self):
        task = dict(prompt='def f():\n', canonical_solution='    return 1\n', test='def check(c):\n    assert c() == 1\n', entry_point='f')
        self.assertEqual(humaneval_program(task), 'def f():\n    return 1\n\ndef check(c):\n    assert c() == 1\n\ncheck(f)\n')


@unittest.skipUnless(BINARY, 'IGC_SANDBOX_EXE names no binary')
class Corpus(unittest.TestCase):
    def test_every_program_matches_or_ends_in_its_expected_status(self):
        verdicts = run_corpus(Path(BINARY), CORPUS, 30.0)
        self.assertGreaterEqual(len(verdicts), 300)
        failed = [f'{v.kind} {v.name}: {v.detail[:200]}' for v in verdicts if v.kind not in ('match', 'expected')]
        self.assertEqual(failed, [])


@unittest.skipUnless(BINARY, 'IGC_SANDBOX_EXE names no binary')
class Limits(unittest.TestCase):
    def run_source(self, source, *flags):
        with tempfile.TemporaryDirectory() as scratch:
            path = Path(scratch) / 'program.py'
            path.write_text(source, encoding='utf-8')
            return run_binary(Path(BINARY), path, 30.0, list(flags))

    def test_output_past_the_limit_is_dropped_and_the_program_keeps_running(self):
        ours = self.run_source("print('x' * 100)\nprint('y' * 100)\nraise ValueError('still ran')\n", '--output', '150')
        self.assertEqual(ours.stdout, 'x' * 100 + '\n' + 'y' * 49)
        self.assertTrue(ours.truncated)
        self.assertEqual(ours.status, 'Raised')

    def test_output_is_cut_on_a_character_boundary(self):
        ours = self.run_source("print('กกก')\n", '--output', '4')
        self.assertEqual(ours.stdout, 'ก')
        self.assertTrue(ours.truncated)

    def test_output_at_the_limit_is_not_truncated(self):
        ours = self.run_source("print('abc', end='')\n", '--output', '3')
        self.assertEqual(ours.stdout, 'abc')
        self.assertFalse(ours.truncated)

    def test_a_smaller_step_budget_stops_sooner(self):
        ours = self.run_source('i = 0\nwhile i < 1000:\n    i += 1\n', '--steps', '100')
        self.assertEqual(ours.status, 'StepLimit')
        self.assertRegex(ours.error, r'\(line \d\)$')

    def test_a_smaller_heap_stops_before_the_allocation(self):
        ours = self.run_source('a = [0] * 100000\n', '--heap', '1000000')
        self.assertEqual(ours.status, 'MemoryLimit')
        self.assertLess(ours.peak, 1000000)

    def test_cancel_ends_a_tight_loop(self):
        ours = self.run_source('while True:\n    pass\n', '--steps', '100000000000', '--cancel-ms', '20')
        self.assertEqual(ours.status, 'Cancelled')
        self.assertRegex(ours.error, r'^Cancelled \(line [12]\)$')

    def test_recursion_and_parser_limits_are_the_callers(self):
        self.assertEqual(self.run_source('def f(n):\n    return n and f(n - 1)\nf(50)\n', '--recursion', '10').status, 'RecursionLimit')
        self.assertEqual(self.run_source('def f(n):\n    return n and f(n - 1)\nf(50)\n', '--recursion', '100').status, 'Ok')
        self.assertEqual(self.run_source('x = ((((((1))))))\n', '--parser', '3').status, 'RecursionLimit')
        self.assertEqual(self.run_source('x = ((1))\n', '--parser', '10').status, 'Ok')

    def test_digit_limit_is_the_callers(self):
        self.assertEqual(self.run_source('print(10 ** 50)\n', '--digits', '10').exception, 'ValueError')

    def test_errors_carry_the_line_of_the_raise(self):
        self.assertEqual(self.run_source("x = 1\n\n\nraise ValueError('bad value')\n").error, 'ValueError: bad value (line 4)')
        self.assertEqual(self.run_source('x = 1\nclass A:\n    pass\n').error, 'unsupported: class (line 2)')

    def test_a_multi_line_message_stays_on_one_line(self):
        self.assertEqual(self.run_source("raise ValueError('a\\nb')\n").error, 'ValueError: a b (line 1)')


if __name__ == '__main__':
    unittest.main()
