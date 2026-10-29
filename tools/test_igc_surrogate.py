"""Bounded late-head STE feasibility on one real native-Gyro capture."""
import os
from pathlib import Path
import tempfile
import unittest
import numpy as np


@unittest.skipUnless(os.environ.get('IGC_HEAD_PROBE') and os.environ.get('IGC_CAPTURE_DIR'),
                     'native head and real activation capture not selected')
class LateHeadSurrogate(unittest.TestCase):
    def test_native_forward_and_one_bounded_gradient_step(self):
        import torch
        from tools.igc.head_surrogate import FrozenHeadSTE
        capture = Path(os.environ['IGC_CAPTURE_DIR'])
        mixed = torch.from_numpy(np.fromfile(capture / 'legacy-mixed.bin', dtype='<f4').copy())
        unchanged = mixed.clone()
        with tempfile.TemporaryDirectory() as tmp:
            head = FrozenHeadSTE(os.environ['IGC_HEAD_PROBE'], os.environ['IGC_PACK'],
                                 os.environ['IGC_MODEL'], Path(tmp))
            value = mixed.clone().requires_grad_(True)
            logits = head.project(value)
            expected = np.fromfile(capture / 'first-logits.bin', dtype='<f4')
            np.testing.assert_array_equal(logits.detach().numpy(), expected)
            target = torch.tensor([391], dtype=torch.long)  # diagnostic class, not a math task
            before = torch.nn.functional.cross_entropy(logits[None], target)
            self.assertTrue(before.requires_grad)
            before.backward()
            self.assertIsNotNone(value.grad)
            self.assertTrue(torch.isfinite(value.grad).all())
            self.assertGreater(value.grad.norm().item(), 0)
            candidate = value.detach() - .1 * value.grad
            after = torch.nn.functional.cross_entropy(head.project(candidate)[None], target)
            self.assertLess(after.item(), before.item())
            self.assertTrue(torch.equal(mixed, unchanged))
            print(f'late-head STE one-step native CE {before.item():.9g} -> {after.item():.9g}; target=391')


if __name__ == '__main__':
    unittest.main()
