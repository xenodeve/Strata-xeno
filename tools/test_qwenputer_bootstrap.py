"""Preflight is data-only; these tests never execute compiler or firmware."""
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace

from tools.qwenputer.bootstrap import capture_boot, preflight
from tools.qwenputer.tokenizer import NativeInputError
from tools.test_qwenputer_tokenizer import fixture
from tools.test_qwenputer_firmware import image


VENDOR = Path(__file__).resolve().parents[1] / 'third_party/qwenputer'


class BootstrapPreflightTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.tokenizer = self.root / 'tokenizer'
        self.tokenizer.mkdir()
        fixture(self.tokenizer)
        self.elf = self.root / 'firmware.elf'
        self.elf.write_bytes(image())

    def tearDown(self):
        self.temp.cleanup()

    def test_preflight_binds_every_input_without_importing_upstream(self):
        before = {n for n in sys.modules if n == 'chipformer' or n.startswith('chipformer.')}
        report = preflight(VENDOR, self.tokenizer, self.elf)
        self.assertEqual(report['firmware_sha256'], hashlib.sha256(self.elf.read_bytes()).hexdigest())
        self.assertEqual(report['source_revision'], 'ca13310e2f9f788d1dd3feeea8213c1b0c23913c')
        self.assertEqual(report['vocab_size'], 266)
        self.assertEqual(report['max_boot_tokens'], 200000)
        self.assertFalse(report['would_execute'])
        self.assertEqual(before, {n for n in sys.modules if n == 'chipformer' or n.startswith('chipformer.')})

    def test_bad_boot_budget_refuses_before_execution(self):
        for bound in (0, -1, 1000001, True):
            with self.subTest(bound=bound):
                with self.assertRaises(NativeInputError) as caught:
                    preflight(VENDOR, self.tokenizer, self.elf, bound)
                self.assertEqual(caught.exception.code, 'boot_token_limit')

    def test_logical_state_capture_requires_a_clean_bounded_park(self):
        machine = SimpleNamespace(halted=True, fault=None, exit_code=0, out=bytearray(),
                                  anchor=0, n=3, x=[0] * 32, ram=[0] * 8192,
                                  rom=[0] * 65536, drom=b'', bn=[[0] * 128 for _ in range(8)])
        state = capture_boot(machine, [1, 2], 10)
        self.assertEqual(state.get('n'), 3)
        self.assertEqual(state.get('resume_after_park'), True)
        for field, value, code in [('halted', False, 'boot_limit'),
                                    ('fault', (1, 4), 'boot_fault'),
                                    ('out', bytearray(b'partial'), 'boot_output')]:
            previous = getattr(machine, field)
            setattr(machine, field, value)
            with self.assertRaises(NativeInputError) as caught:
                capture_boot(machine, [1, 2], 10)
            self.assertEqual(caught.exception.code, code)
            setattr(machine, field, previous)


if __name__ == '__main__':
    unittest.main()
