"""Embedding semantics are independent of the upstream callback implementation."""
from pathlib import Path
from types import SimpleNamespace as NS
import tempfile
import unittest

import numpy as np

from tools.qwenputer import features as F
from tools.qwenputer.tokenizer import NativeInputError, NativeText
from tools.test_qwenputer_tokenizer import fixture


class FeatureTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        fixture(Path(self.temp.name))
        self.text = NativeText.from_pack(Path(self.temp.name))

    def tearDown(self):
        self.temp.cleanup()

    def values(self, name, ids, kind=0):
        return F.feature_values(F.lower_feature(name), kind, np.array(ids), self.text).tolist()

    def test_family_payload_and_control_boundaries(self):
        self.assertEqual(self.values('isV', [0, 32767, 32768, F.TRACE_VOCAB - 1]), [1, 1, 0, 0])
        self.assertEqual(self.values('V.b14', [0, 16383, 16384, 32767, 32768]), [0, 0, 1, 1, 0])
        self.assertEqual(self.values('isSC', [F.TRACE_VOCAB - 2, F.TRACE_VOCAB - 1]), [0, 1])
        self.assertEqual(self.values('digit==1', [sum(n for _, n in F.FAMILIES[:9]) + 1,
                                                sum(n for _, n in F.FAMILIES[:10]) + 15]), [1, 1])

    def test_code_family_bits_and_payload_order(self):
        # V family index 0, F0 index 1, SC index 19; family bits precede payload bits.
        self.assertEqual(self.values('code0', [0, 32768, F.TRACE_VOCAB - 1]), [-128, 128, 128])
        self.assertEqual(self.values('code5', [0, 1, 2, 32768]), [-128, 128, -128, -128])
        self.assertEqual(self.values('code1', [120, 265], 1), [-128, 0])  # Q index 20
        self.assertEqual(self.values('code7', [4, 0], 1), [128, -128])

    def test_text_bytes_are_projected_and_trace_features_are_muted(self):
        self.assertEqual(self.values('tb0.b7', [255, 65, 265], 1), [1, 0, 0])
        self.assertEqual(self.values('tb15.b3', [265, 65], 1), [1, 0])
        self.assertEqual(self.values('tb0.b0', [65, 66], 0), [0, 0])
        self.assertEqual(self.values('isTCO', [256, 257], 1), [1, 0])
        self.assertEqual(self.values('isTCC', [256, 257], 1), [0, 1])
        self.assertEqual(self.values('mu', [0, 32768]), [1, 1])
        self.assertEqual(self.values('mu', [0, 265], 1), [0, 0])
        self.assertEqual(self.values('tau', [0]), [-1])
        self.assertEqual(self.values('tau', [0], 1), [2 ** -40])
        self.assertEqual(self.values('one', [0, 265], 1), [1, 1])
        self.assertEqual(self.values('mem[31]', [0, 32768]), [0, 0])

    def test_unknown_features_and_invalid_domains_fail_closed(self):
        for name in ('new_callback', 'mem[32]', 'V.b15', 'tb16.b0', 'code21', 'isQ', 'digit==8'):
            with self.subTest(name=name), self.assertRaises(NativeInputError):
                F.lower_feature(name)
        for kind, ids in ((0, [-1]), (0, [F.TRACE_VOCAB]), (1, [self.text.n]),
                          (1, [True]), (2, [0]), (0, [1.5])):
            with self.subTest(kind=kind, ids=ids), self.assertRaises(NativeInputError):
                F.feature_values(F.lower_feature('one'), kind, np.array(ids), self.text)

    def test_exhaustive_gate_detects_one_callback_changed_at_domain_end(self):
        graph = NS(dims=[NS(name='one'), NS(name='isV')], token_fns={1: None})
        graph.feature_value = lambda d, t: int(0 <= t < 32768)
        features, proof = F.check_features(graph, self.text)
        self.assertEqual(proof['comparisons'], 2 * (F.TRACE_VOCAB + self.text.n))
        self.assertEqual(len(features), 2)
        graph.feature_value = lambda d, t: int(0 <= t < 32768 or t == F.TEXT_TOKEN + self.text.n - 1)
        with self.assertRaisesRegex(NativeInputError, 'feature_mismatch:1:1:265'):
            F.check_features(graph, self.text)


if __name__ == '__main__':
    unittest.main()
