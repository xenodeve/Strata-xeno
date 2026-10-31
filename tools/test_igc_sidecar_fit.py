"""Sidecar model shape, sizes and table reduction; no weights or GPU."""
import unittest
import torch
from tools.igc.embed_table import reduce
from tools.igc.sidecar_fit import Sidecar, shape_for, trainable


class SidecarFit(unittest.TestCase):
    def test_sizes_sit_near_their_targets(self):
        table = torch.zeros(100, 128)
        for target in (5, 30, 100):
            parameters = trainable(Sidecar(table, *shape_for(target))) / 1e6
            self.assertLess(abs(parameters - target) / target, .1, parameters)

    def test_pointers_cannot_reach_an_empty_entry(self):
        model = Sidecar(torch.zeros(100, 16), 32, 1, 2).eval()
        _, left, right = model(torch.randint(0, 99, (2, 9)), torch.tensor([9, 5]), torch.tensor([[1, 4, 8], [2, 4, -1]]))
        self.assertEqual(left.shape, (2, 3))
        self.assertLess(left[1, 2], -1e8)
        self.assertLess(right[1, 2], -1e8)

    def test_a_later_token_cannot_change_a_slot(self):
        torch.manual_seed(1)
        model = Sidecar(torch.randn(100, 16), 32, 2, 2).eval()
        tokens = torch.randint(0, 99, (1, 9))
        changed = tokens.clone()
        changed[0, 8] = (changed[0, 8] + 1) % 99
        ends = torch.tensor([[1, 3]])
        with torch.no_grad():
            first = model(tokens, torch.tensor([5]), ends)
            second = model(changed, torch.tensor([5]), ends)
        for a, b in zip(first, second):
            self.assertTrue(torch.allclose(a, b))

    def test_reduction_keeps_row_count_and_width(self):
        table = reduce(torch.randn(500, 64), 16)
        self.assertEqual(table.shape, (500, 16))


if __name__ == '__main__':
    unittest.main()
