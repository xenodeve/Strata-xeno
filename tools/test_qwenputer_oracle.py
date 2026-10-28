"""Independent rational math and live-call boundary specification."""
from pathlib import Path
import tempfile
import unittest

from tools.qwenputer.call import CallContext, eligible_call
from tools.qwenputer.oracle import exact_answer, reference_case, decode_prediction, held_out_cases
from tools.qwenputer.tokenizer import NativeText, NativeInputError
from tools.test_qwenputer_tokenizer import fixture


class OracleTests(unittest.TestCase):
    def test_separate_held_out_spec_is_frozen_and_not_executed(self):
        cases = held_out_cases()
        self.assertEqual(len(cases), 64)
        self.assertEqual(cases, held_out_cases())
        self.assertEqual(len({case['expression'] for case in cases}), 64)
        self.assertTrue(all(case['firmware_executed'] is False for case in cases))

    def test_exact_rational_precedence_decimal_and_negative_floor(self):
        for expression, answer in [('17*23', '391'), ('-7//3', '-3'), ('-7%3', '2'),
                                   ('7%-3', '-2'), ('2**3**2', '512'), ('-2**2', '-4'),
                                   ('1/3+1/6', '1/2'), ('0.1+0.2', '3/10'),
                                   ('gcd(48,18)', '6'), ('3^4', '81')]:
            with self.subTest(expression=expression):
                self.assertEqual(exact_answer(expression), answer)

    def test_no_eval_and_bounded_domain(self):
        for expression in ('__import__("os")', 'x+1', '[1]', 'gcd(1)', '1/0',
                           '2**100000000', '1e10000000', 'True+1', '2**-2'):
            with self.subTest(expression=expression), self.assertRaises(NativeInputError):
                exact_answer(expression)

    def test_decimal_integer_leading_zeros_do_not_acquire_python_syntax(self):
        try:
            answer = exact_answer('001+02')
        except NativeInputError as error:
            self.fail('decimal literal rejected by Python AST grammar: ' + error.code)
        self.assertEqual(answer, '3')
        self.assertEqual(exact_answer('000*9+01'), '1')


class CallTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        fixture(Path(self.temp.name))
        self.text = NativeText.from_pack(Path(self.temp.name))
        self.live = CallContext(live_assistant=True, enrolled_calc=True)
        self.valid = '<function=calc><parameter=expression>17*23</parameter></function>'

    def tearDown(self):
        self.temp.cleanup()

    def call(self, raw, context=None):
        return eligible_call(self.text, self.text.encode(raw), context or self.live)

    def test_only_enrolled_live_calc_executes(self):
        # This fixture has byte-only vocabulary, so scaffold + payload fits 80 rows.
        self.assertEqual(self.call(self.valid), b'17*23')
        for context in (CallContext(), CallContext(live_assistant=True),
                        CallContext(True, True, code_fence=True),
                        CallContext(True, True, quoted=True),
                        CallContext(True, True, history=True)):
            self.assertIsNone(self.call(self.valid, context))
        self.assertIsNone(self.call('<function=Bash><parameter=command>1+1</parameter></function>'))

    def test_malformed_or_reserved_active_capture_is_rejected(self):
        for raw in (self.valid.replace('17*23', '1<tool_call>2'),
                    self.valid.replace('17*23', '1\0+2'),
                    self.valid + self.valid, self.valid.replace('</parameter>', ''),
                    self.valid.replace('17*23', '3<4')):
            with self.subTest(raw=raw), self.assertRaises(NativeInputError):
                self.call(raw)

    def test_reference_trace_includes_text_echo_and_final_halt_echo(self):
        from tools.qwenputer.features import FAMILIES, TEXT_TOKEN
        from types import SimpleNamespace as NS
        byte_base = sum(size for _, size in FAMILIES[:6])
        halt_id = sum(size for _, size in FAMILIES[:17])
        response = [256, 10, 51, 57, 49, 10, 257]
        records = iter([[byte_base + payload] for payload in response] + [[halt_id]])
        machine = NS(text=None, halted=False, fault=None, exit_code=0, steps=0)
        def record():
            value = next(records)
            machine.steps += 1
            if value == [halt_id]:
                machine.halted = True
            return value
        machine.record = record
        result = reference_case(machine, self.text, '17*23')
        self.assertTrue(result['passed'])
        self.assertEqual(result['actual_body'], '\n391\n')
        self.assertEqual(result['stream'][-2:], [halt_id, TEXT_TOKEN + self.text.NL])
        self.assertEqual(result['stream'][-4:-2], [byte_base + 257, TEXT_TOKEN + self.text.printable[257]])

    def test_tied_output_code_decodes_kind_and_rejects_invalid_prediction(self):
        from tools.qwenputer.features import FAMILIES, TEXT_TOKEN
        def row(family, payload):
            code = family | (payload << 5)
            return [2 ** -20 * (2 * ((code >> bit) & 1) - 1) for bit in range(21)] + [1]
        self.assertEqual(decode_prediction(row(0, 17), list(range(21)), 21, self.text), 17)
        self.assertEqual(decode_prediction(row(20, 49), list(range(21)), 21, self.text), TEXT_TOKEN + 49)
        self.assertEqual(decode_prediction(row(19, 0), list(range(21)), 21, self.text),
                         sum(size for _, size in FAMILIES) - 1)
        for bad in (row(31, 0), row(20, 511), row(3, 4)):
            with self.assertRaises(NativeInputError):
                decode_prediction(bad, list(range(21)), 21, self.text)

    def test_known_capacity_refusal_keeps_independent_math_failure_visible(self):
        from tools.qwenputer.features import FAMILIES
        from types import SimpleNamespace as NS
        byte_base = sum(size for _, size in FAMILIES[:6])
        halt_id = sum(size for _, size in FAMILIES[:17])
        records = iter([[byte_base + p] for p in [256, 10, *b'error', 10, 257]] + [[halt_id]])
        machine = NS(text=None, halted=False, fault=None, exit_code=0, steps=0)
        def record():
            value = next(records)
            machine.steps += 1
            if value == [halt_id]:
                machine.halted = True
            return value
        machine.record = record
        result = reference_case(machine, self.text, '10**61')
        self.assertTrue(result['passed'])
        self.assertIn('mathematical_success', result)
        self.assertFalse(result['mathematical_success'])
        self.assertEqual(result['expected_rational'], '1' + '0' * 61)
        self.assertIn('capacity', result['refusal'])


if __name__ == '__main__':
    unittest.main()
