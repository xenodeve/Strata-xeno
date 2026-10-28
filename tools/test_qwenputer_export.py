"""Sparse representation/order and coefficient bits, with no upstream execution."""
from fractions import Fraction
import hashlib
from pathlib import Path
import struct
import tempfile
from unittest.mock import patch
from types import SimpleNamespace as NS
import unittest

from tools.qwenputer.export import compile_sections, export_artifact
from tools.qwenputer.format import encode_sections, decode_sections
from tools.qwenputer.tokenizer import NativeText, NativeInputError
from tools.test_qwenputer_tokenizer import fixture


class FFN:
    def __init__(self, coefficient=2 ** 82):
        self.name, self.level, self.out, self.into = 'sum', 2, 2, False
        self.terms = [(coefficient, NS(terms={0: Fraction(1, 2 ** 41)}), NS(terms={1: 1})),
                      (-1, NS(terms={1: 1, 0: -1}), NS(terms={0: 1}))]


class Head:
    def __init__(self):
        channel = NS(expr=NS(terms={0: 128}), taps=(1, -1, 0, 0))
        value = NS(v={0: channel}, a=NS(terms={0: 110}),
                   z={0: NS(terms={0: 128})}, out=[(0, 3, 128)], beta=None, into=True)
        self.name, self.level, self.n_cols, self.tiny, self.levels = 'table', 3, 128, 2 ** -90, 'byte'
        self.q, self.k, self.init = {1: channel}, {}, [{1: [2 ** 24, 2 ** 24] + [255] * 126}]
        self.vheads = lambda: [value]


class ExportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        fixture(Path(self.temp.name))
        self.text = NativeText.from_pack(Path(self.temp.name))
        self.make_graph()

    def make_graph(self):
        self.graph = NS(dims=[NS(name='one', kind='one', level=0),
                              NS(name='isV', kind='token', level=0),
                              NS(name='sum', kind='ffn', level=2),
                              NS(name='mem[0]', kind='token', level=0)],
                        token_fns={1: None, 3: None}, nodes=[FFN(), Head()])
        self.identity = {name: '0' * 64 for name in ('source_sha256', 'firmware_sha256',
                          'boot_sha256', 'model_sha256')}
        self.identity['tokenizer_sha256'] = self.text.binding_sha256

    def tearDown(self):
        self.temp.cleanup()

    def test_sparse_order_and_large_small_exact_coefficients_roundtrip(self):
        sections = compile_sections(self.graph, self.text, self.identity)
        self.assertIn('ffns', sections)
        self.assertEqual(decode_sections(encode_sections(sections)), sections)
        self.assertEqual(sections['ffns'][0], 24)
        records = list(struct.iter_unpack('<fii', sections['ffn_terms'][1]))
        self.assertEqual([r[0] for r in records], [float(2 ** 82), -1.0])
        terms = list(struct.iter_unpack('<if', sections['terms'][1]))
        self.assertIn((0, 2 ** -41), terms)
        self.assertEqual(sections['text_rows'][1][265 * 16:266 * 16], b'x' * 16)
        self.assertEqual(struct.unpack_from('<I', sections['text_lens'][1], 265 * 4)[0], 17)
        self.assertEqual(sections['init'][0], 520)
        self.assertEqual(len(sections['init'][1]), 520)
        self.assertIn('head_levels', sections)
        self.assertEqual(sections['head_levels'], (1, b'\1'))
        self.assertEqual(hashlib.sha256(encode_sections(sections)).digest(),
                         hashlib.sha256(encode_sections(compile_sections(self.graph, self.text, self.identity))).digest())

    def test_nonrepresentable_coefficient_and_unknown_feature_refused(self):
        self.graph.nodes[0] = FFN(Fraction(1, 3))
        with self.assertRaisesRegex(NativeInputError, 'coefficient_fp32'):
            compile_sections(self.graph, self.text, self.identity)
        self.graph.nodes[0] = FFN()
        self.graph.dims[1].name = 'unreviewed_callback'
        with self.assertRaisesRegex(NativeInputError, 'unknown_feature'):
            compile_sections(self.graph, self.text, self.identity)

    def test_indices_initial_rows_and_binding_are_not_trusted(self):
        for mutate in (lambda: self.graph.nodes[0].terms.append((1, NS(terms={99: 1}), NS(terms={}))),
                       lambda: self.graph.nodes[1].init[0].update({128: [0] * 128}),
                       lambda: self.identity.update(tokenizer_sha256='f' * 64)):
            with self.subTest(mutate=mutate):
                self.make_graph()
                mutate()
                with self.assertRaises(NativeInputError):
                    compile_sections(self.graph, self.text, self.identity)

    def test_export_refuses_execution_without_explicit_scope_before_import(self):
        with self.assertRaisesRegex(NativeInputError, 'execution_not_authorized'):
            export_artifact(Path('unused'), Path('unused'), Path('unused'),
                            Path('unused'), Path('unused'), '0' * 64)

    def test_retained_output_and_owner_dimensions_survive_lowering(self):
        self.graph.keep = [2, 1]
        self.graph.owner = NS(terms={3: 1})
        sections = compile_sections(self.graph, self.text, self.identity)
        self.assertIn('keep', sections)
        self.assertEqual(sections['keep'], (4, struct.pack('<2i', 2, 1)))
        self.assertEqual(struct.unpack('<16I', sections['config'][1])[-1], 3)

    def test_compiler_process_requires_frozen_hash_seed_before_import(self):
        with patch.dict('os.environ', {'PYTHONHASHSEED': '1'}):
            with self.assertRaisesRegex(NativeInputError, 'compiler_hash_seed'):
                export_artifact(Path('unused'), Path('unused'), Path('unused'),
                                Path('unused'), Path('unused'), '0' * 64, True)


if __name__ == '__main__':
    unittest.main()
