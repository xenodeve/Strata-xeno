"""Opt-in native output-head replay controls; no complete model load or training."""
import os
import math
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest
import numpy as np
from serve.gguf_info import read_header


@unittest.skipUnless(os.environ.get('IGC_HEAD_PROBE'), 'IGC native head probe not selected')
class HeadReplay(unittest.TestCase):
    def invoke(self, *args):
        return subprocess.run([os.environ['IGC_HEAD_PROBE'], *map(str, args)],
                              capture_output=True, text=True, timeout=90,
                              creationflags=0x08000000)

    def test_zero_residual_produces_exact_zero_logits_and_repeats(self):
        p = self.invoke('--pack', os.environ['IGC_PACK'], '--model', os.environ['IGC_MODEL'], '--zero')
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        self.assertIn('PASS zero logits=248320 repeat=bitwise', p.stdout)

    def test_invalid_geometry_is_rejected_before_weight_loading(self):
        with tempfile.TemporaryDirectory() as tmp:
            residual = Path(tmp) / 'wrong.bin'
            residual.write_bytes(struct.pack('<ii', 4, 2561))
            p = self.invoke('--pack', 'absent-pack', '--model', 'absent-model', '--residual', residual)
            self.assertEqual(p.returncode, 2, p.stdout + p.stderr)
            self.assertIn('residual_geometry', p.stderr)

    def test_nonfinite_residual_is_rejected_before_weight_loading(self):
        with tempfile.TemporaryDirectory() as tmp:
            residual = Path(tmp) / 'nonfinite.bin'
            residual.write_bytes(struct.pack('<ii', 4, 2560) + struct.pack('<f', float('nan')) * 10240)
            p = self.invoke('--pack', 'absent-pack', '--model', 'absent-model', '--residual', residual)
            self.assertEqual(p.returncode, 2, p.stdout + p.stderr)
            self.assertIn('residual_nonfinite', p.stderr)

    def test_nonzero_native_projection_matches_independent_block_oracle(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            residual, mixed_path, logits_path = (root / name for name in ('R.bin', 'mixed.bin', 'logits.bin'))
            values = [math.sin(i * .137) * (1 + (i % 7) / 8) for i in range(10240)]
            residual.write_bytes(struct.pack('<ii', 4, 2560) + struct.pack('<10240f', *values))
            command = ['--pack', os.environ['IGC_PACK'], '--model', os.environ['IGC_MODEL'],
                       '--residual', residual, '--mixed-out', mixed_path, '--logits-out', logits_path]
            p = self.invoke(*command)
            self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
            mixed = np.fromfile(mixed_path, dtype='<f4').reshape(80, 32)
            logits = np.fromfile(logits_path, dtype='<f4')
            self.assertEqual(logits.size, 248320)
            self.assertTrue(np.any(logits != 0))
            scale = np.max(np.abs(mixed), axis=1).astype(np.float32) / np.float32(127)
            quant = np.copysign(np.floor(np.abs(mixed / scale[:, None]) + np.float32(.5)), mixed)
            quant = np.clip(quant, -127, 127).astype(np.int32)
            activation_scale = scale.astype(np.float16).astype(np.float64)
            header = read_header(os.environ['IGC_MODEL'])
            tensor = next(t for t in header['tensors'] if t['name'] == 'output.weight')
            self.assertEqual((tensor['type'], tensor['dims']), (8, [2560, 248320]))
            rows = [0, 1, 31, 1000, int(logits.argmax()), 248319]
            with open(os.environ['IGC_MODEL'], 'rb') as model:
                for row in rows:
                    model.seek(header['data_start'] + tensor['offset'] + row * 80 * 34)
                    blocks = model.read(80 * 34)
                    weight_scale = np.array([struct.unpack_from('<e', blocks, i * 34)[0] for i in range(80)])
                    weight_q = np.array([struct.unpack_from('<32b', blocks, i * 34 + 2) for i in range(80)], dtype=np.int32)
                    integer_dot = (weight_q * quant).sum(axis=1, dtype=np.int64)
                    reference = np.sum(integer_dot * weight_scale * activation_scale)
                    np.testing.assert_allclose(logits[row], reference, rtol=2e-5, atol=3e-5)
            # The exact comparison must reject wrong evidence, not silently accept an argmax-only match.
            wrong = root / 'wrong.bin'
            wrong.write_bytes(bytes(248320 * 4))
            bad = self.invoke(*command, '--expected-logits', wrong)
            self.assertEqual(bad.returncode, 1, bad.stdout + bad.stderr)
            self.assertIn('bitwise_differences=', bad.stdout)

    def test_frozen_q8_weight_vjp_matches_independent_linear_surrogate(self):
        # This is d(W_dequant @ x)/dx. It deliberately declares an STE across
        # the native activation quantizer, not an exact derivative of Q8_1.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            delta_path, grad_path = root / 'delta.bin', root / 'grad.bin'
            delta = np.zeros(248320, dtype='<f4')
            rows = {0: .25, 1000: -.5, 248319: .125}
            for row, value in rows.items():
                delta[row] = value
            delta.tofile(delta_path)
            p = self.invoke('--pack', os.environ['IGC_PACK'], '--model', os.environ['IGC_MODEL'],
                            '--zero', '--vjp-input', delta_path, '--vjp-out', grad_path)
            self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
            self.assertIn('gradient=dequant_weight_ste', p.stdout)
            actual = np.fromfile(grad_path, dtype='<f4')
            reference = np.zeros(2560, dtype=np.float64)
            header = read_header(os.environ['IGC_MODEL'])
            tensor = next(t for t in header['tensors'] if t['name'] == 'output.weight')
            with open(os.environ['IGC_MODEL'], 'rb') as model:
                for row, coefficient in rows.items():
                    model.seek(header['data_start'] + tensor['offset'] + row * 80 * 34)
                    blocks = model.read(80 * 34)
                    for i in range(80):
                        scale = struct.unpack_from('<e', blocks, i * 34)[0]
                        q = np.array(struct.unpack_from('<32b', blocks, i * 34 + 2))
                        reference[i * 32:(i + 1) * 32] += coefficient * scale * q
            self.assertEqual(actual.size, 2560)
            np.testing.assert_allclose(actual, reference, rtol=1e-6, atol=2e-7)

    @unittest.skipUnless(os.environ.get('IGC_CAPTURE_DIR'), 'real model capture not selected')
    def test_direct_captured_head_input_reproduces_all_model_logits(self):
        capture = Path(os.environ['IGC_CAPTURE_DIR'])
        p = self.invoke('--pack', os.environ['IGC_PACK'], '--model', os.environ['IGC_MODEL'],
                        '--mixed-input', capture / 'legacy-mixed.bin',
                        '--expected-logits', capture / 'first-logits.bin')
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        self.assertIn('bitwise_differences=0 max_abs=0', p.stdout)


if __name__ == '__main__':
    unittest.main()
