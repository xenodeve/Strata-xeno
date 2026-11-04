"""tools/igc/seal.py without a GPU or a real model: the keyword labels of the paper's table 5, the thought split, and the hooks on a toy decoder (the flat vLLM layout:
tokens x hidden, layers returning (hidden_states, residual)). python tools/test_igc_seal.py"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.igc import seal

try:
    import torch
except ImportError:   # the labels are tested without it
    torch = None


class Tok:
    def decode(self, ids):
        return ''.join(chr(i) for i in ids)


class Labels(unittest.TestCase):
    def test_table_5_keywords(self):
        t = seal.thought_type
        self.assertEqual(t('Wait, that is not right'), 'R')
        self.assertEqual(t('  wait'), 'R')
        self.assertEqual(t('I should verify the sum.'), 'R')
        self.assertEqual(t("So 5 is correct, it seems right."), 'R')
        self.assertEqual(t('Alternatively, factor it.'), 'T')
        self.assertEqual(t('There is another approach: use induction.'), 'T')
        self.assertEqual(t('Compute 6 * 7 = 42.'), 'E')
        self.assertEqual(t('Wait, let me think of another way'), 'T')   # transition wins a tie
        self.assertEqual(t('Awaiting the result'), 'E')   # the prefix is the word Wait, not the letters

    def test_thoughts_split_on_the_delimiter_and_drop_the_open_tail(self):
        ids = [ord(c) for c in 'ab'] + [10] + [ord(c) for c in 'Wait'] + [10] + [ord(c) for c in 'cd']
        th = seal.thoughts(ids, Tok(), 10)
        self.assertEqual(th, [(0, 2, 'E'), (3, 7, 'R')])


@unittest.skipIf(torch is None, 'torch is not installed')
class Hooks(unittest.TestCase):
    def toy(self, n=3, h=4):
        class Layer(torch.nn.Module):
            def forward(self, hidden, residual):
                return hidden + 1.0, residual
        class Inner(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embed_tokens = torch.nn.Embedding(20, h)
                self.layers = torch.nn.ModuleList(Layer() for _ in range(n))
            def forward(self, ids):
                x, r = self.embed_tokens(ids), torch.zeros(ids.shape[0], h)
                for layer in self.layers:
                    x, r = layer(x, r)
                return x
        class Outer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = Inner()
            def forward(self, ids):
                return self.model(ids)
        return Outer()

    def test_only_delimiter_positions_are_steered(self):
        m = self.toy()
        ids = torch.tensor([3, 7, 3, 7, 5])
        base = m(ids).detach().clone()
        vec = torch.tensor([1.0, 2.0, 3.0, 4.0])
        handles, count = seal.install_steering(m, vec, layer=2, alpha=2.0, delim_id=7, n_layers=3)
        got = m(ids).detach()
        self.assertTrue(torch.allclose(got[[0, 2, 4]], base[[0, 2, 4]]))
        self.assertTrue(torch.allclose(got[[1, 3]], base[[1, 3]] + 2.0 * vec))
        self.assertEqual(count['steered'], 2)
        for h in handles:
            h.remove()
        self.assertTrue(torch.allclose(m(ids), base))

    def test_a_layer_after_the_steered_one_carries_it_on(self):
        m = self.toy()
        ids = torch.tensor([7])
        base = m(ids).detach().clone()
        handles, _ = seal.install_steering(m, torch.ones(4), layer=3, alpha=1.0, delim_id=7, n_layers=3)
        self.assertTrue(torch.allclose(m(ids), base + 1.0))   # the output of the last layer is the steered one
        for h in handles:
            h.remove()
        handles, _ = seal.install_steering(m, torch.ones(4), layer=1, alpha=1.0, delim_id=7, n_layers=3)
        self.assertTrue(torch.allclose(m(ids), base + 1.0))   # steered at layer 1, carried through layers 2 and 3
        for h in handles:
            h.remove()

    def test_no_such_layer_list_is_an_error(self):
        with self.assertRaises(RuntimeError):
            seal.find_modules(self.toy(n=3), 5)


if __name__ == '__main__':
    unittest.main()
