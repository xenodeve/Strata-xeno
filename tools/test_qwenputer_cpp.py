"""Native C++ validation of correctly checksummed but invalid artifact data."""
import json
import hashlib
import os
import re
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest

from tools.qwenputer.format import decode_sections, encode_sections


@unittest.skipUnless(os.environ.get('QWENPUTER_ARTIFACT_TEST') and os.environ.get('QWENPUTER_ARTIFACT'),
                     'native CPU binary/artifact integration not selected')
class CppArtifactTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.binary = os.environ['QWENPUTER_ARTIFACT_TEST']
        self.artifact = Path(os.environ['QWENPUTER_ARTIFACT'])
        self.sections = decode_sections(self.artifact.read_bytes())
        self.identity = self.root / 'identity.bin'
        manifest = json.loads((self.artifact.parent / 'manifest.json').read_bytes())
        names = ('abi_sha256', 'source_sha256', 'firmware_sha256', 'boot_sha256',
                 'model_sha256', 'tokenizer_sha256')
        self.identity.write_bytes(b''.join(bytes.fromhex(manifest['identities'][k]) for k in names))

    def run_bad(self, sections, expected, name, binding=None):
        self.run_bad_bytes(encode_sections(sections), expected, name, binding)

    def run_bad_bytes(self, data, expected, name, binding=None):
        file = self.root / (name + '.qput')
        file.write_bytes(data)
        result = subprocess.run([self.binary, '--reject', str(file), str(binding or self.identity), expected],
                                capture_output=True, text=True, timeout=60, creationflags=0x08000000)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_container_boundaries_and_all_identity_fields(self):
        original = self.artifact.read_bytes()
        mutations = (
            ('magic', 0, '<B', 0, 'artifact_magic'),
            ('version', 8, '<H', 2, 'artifact_version'),
            ('endian', 12, '<I', 0x04030201, 'artifact_endian'),
            ('count', 16, '<I', 33, 'section_count'),
            ('size', 24, '<Q', 0xffffffffffffffff, 'artifact_size'),
            ('overlap', 80, '<Q', 0, 'section_bounds'),
            ('outside', 80, '<Q', 0xffffffffffffffc0, 'section_bounds'),
            ('stride', 96, '<I', 0, 'section_stride'),
            ('flags', 100, '<I', 1, 'section_flags'),
            ('section_digest', 104, '<B', original[104] ^ 1, 'section_digest'),
            ('reserved', 136, '<B', 1, 'section_reserved'),
        )
        for name, offset, fmt, value, error in mutations:
            with self.subTest(name=name):
                data = bytearray(original)
                struct.pack_into(fmt, data, offset, value)
                data[32:64] = hashlib.sha256(data[64:]).digest()
                self.run_bad_bytes(data, error, name)
        for i in range(6):
            with self.subTest(binding=i):
                data = bytearray(self.identity.read_bytes())
                data[i * 32] ^= 1
                binding = self.root / f'wrong-binding-{i}.bin'
                binding.write_bytes(data)
                self.run_bad_bytes(original, 'artifact_binding', f'wrong-binding-{i}', binding)

    def test_correctly_resigned_semantic_mutants_are_refused(self):
        mutations = (
            ('term_dim', 'terms', 0, '<i', -1, 'term_dim'),
            ('coefficient', 'terms', 4, '<I', 0x7fc00000, 'coefficient'),
            ('expression', 'exprs', 0, '<i', -1, 'expression_range'),
            ('ffn_expr', 'ffn_terms', 4, '<i', 0x7fffffff, 'ffn_expression'),
            ('head_map', 'heads', 6 * 4, '<i', 0x7fffffff, 'head_map_range'),
            ('head_expr', 'heads', 16 * 4, '<i', 0x7fffffff, 'head_expression'),
            ('output', 'outputs', 0, '<i', 128, 'head_output'),
            ('init_row', 'init', 4, '<i', 128, 'head_init_row'),
            ('text_type', 'text_types', 0, '<B', 0, 'text_geometry'),
            ('printable', 'printable', 0, '<i', -1, 'printable_mapping'),
            ('shape', 'config', 0, '<I', 16385, 'circuit_shape'),
        )
        for name, section, offset, fmt, value, expected in mutations:
            with self.subTest(name=name):
                sections = dict(self.sections)
                stride, data = sections[section]
                data = bytearray(data)
                struct.pack_into(fmt, data, offset, value)
                sections[section] = stride, bytes(data)
                self.run_bad(sections, expected, name)
        sections = dict(self.sections, unknown=(1, b'unknown'))
        self.run_bad(sections, 'artifact_sections', 'unknown')

    def test_sparse_original_node_identity_cannot_have_gaps(self):
        sections = dict(self.sections)
        stride, data = sections['ffns']
        data = bytearray(data)
        struct.pack_into('<i', data, 0, 32767)
        sections['ffns'] = stride, bytes(data)
        self.run_bad(sections, 'node_identity', 'node_gap')

    def test_matching_caller_binding_does_not_authorize_unknown_numeric_abi(self):
        sections = dict(self.sections)
        stride, data = sections['identity']
        data = b'\x11' * 32 + data[32:]
        sections['identity'] = stride, data
        binding = self.root / 'unknown-abi.bin'
        binding.write_bytes(data)
        self.run_bad(sections, 'numeric_abi', 'unknown_abi', binding)

    @unittest.skipUnless(os.environ.get('QWENPUTER_LEXICON_FILE'), 'full token lexicon not selected')
    def test_full_cpu_lexicon_preserves_gpu_budget_and_rejects_resigned_mutants(self):
        upgraded = Path(os.environ['QWENPUTER_LEXICON_FILE'])
        original_sections = decode_sections(upgraded.read_bytes())
        values = []
        for file in (self.artifact, upgraded):
            result = subprocess.run([self.binary, str(file), str(self.identity)], capture_output=True,
                                    text=True, timeout=60, creationflags=0x08000000)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            values.append(dict((key, int(n)) for key, n in re.findall(r'(\w+)=(\d+)', result.stdout)))
        for key in ('weights', 'state_each', 'lag_each', 'scratch', 'fifo', 'device_buffers'):
            self.assertEqual(values[0][key], values[1][key], key)
        self.assertGreater(values[1]['host_peak_estimate'], values[0]['host_peak_estimate'])
        for name, offset, value, expected in (
                ('first', 0, 1, 'text_lexicon_offsets'),
                ('end', len(original_sections['text_offsets'][1]) - 4, 0, 'text_lexicon_offsets'),
                ('descending', 4, 0xffffffff, 'text_lexicon_offsets'),
                ('length', 4, 0, 'text_lexicon_offsets')):
            with self.subTest(name=name):
                sections = dict(original_sections)
                stride, data = sections['text_offsets']; data = bytearray(data)
                struct.pack_into('<I', data, offset, value)
                sections['text_offsets'] = stride, bytes(data)
                self.run_bad_bytes(encode_sections(sections, minor=1), expected, 'lexicon-' + name)
        sections = dict(original_sections)
        stride, data = sections['text_bytes']
        sections['text_bytes'] = stride, bytes([data[0] ^ 1]) + data[1:]
        self.run_bad_bytes(encode_sections(sections, minor=1), 'text_lexicon_projection', 'lexicon-projection')
        sections = dict(original_sections)
        del sections['text_offsets']
        self.run_bad_bytes(encode_sections(sections, minor=1), 'artifact_sections', 'lexicon-missing')
        self.run_bad_bytes(encode_sections(original_sections), 'artifact_sections', 'lexicon-old-version')


if __name__ == '__main__':
    unittest.main()
