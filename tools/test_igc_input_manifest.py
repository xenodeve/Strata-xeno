"""Manifest adaptation must not erase pair isolation or encode label metadata."""
import copy
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from tools.igc.matched_corpus import build
from tools.igc.input_manifest import validate_structure, validate_manifest, native_prompt_rows, cache_shapes, check_cache_header, storage_budget


def fixture():
    values = list(range(-19, 20))
    value = build([(a, b) for a in values for b in values])
    value['source_pool'] = dict(file_sha256='0aa4fa0c34e45b47f213279a638d7d11d1a04f0256877ddcf5e8be6c803452d9',
        rows_sha256='fe0e9d86210b4c0c7ef544a7b4a5982dc9c8f53c3a3eebada7a2e34b2356fa7c',
        historical_exposure=True, selection='subset of already audited pair pool; no acceptance manifest access')
    return value


class InputManifest(unittest.TestCase):
    def test_storage_prices_states_masks_manifest_and_retained_reserve(self):
        value = storage_budget(1536, 857013)
        self.assertEqual(value['state_bytes'], 5033164800)
        self.assertEqual(value['mask_bytes'], 122880)
        self.assertEqual(value['fresh_free_bytes'], 9329114037)
        with self.assertRaises(ValueError):
            storage_budget(1536, 9*1024**2)

    def test_cache_header_binds_matched_manifest_rows_protocol_and_sources(self):
        expected = dict(count=1536, rows_sha='b'*64, protocol_sha='c'*64,
                        sources_sha='d'*64, matched_sha='a'*64)
        report = dict(success=True, rows=1536, rows_sha256='b'*64, protocol_sha256='c'*64,
                      source_manifest_sha256='d'*64, input_manifest_version=1, input_manifest_sha256='a'*64)
        self.assertTrue(check_cache_header(report, **expected))
        for field in ('rows', 'rows_sha256', 'protocol_sha256', 'source_manifest_sha256', 'input_manifest_sha256'):
            bad = report.copy()
            bad[field] = 0
            with self.assertRaises(ValueError):
                check_cache_header(bad, **expected)
        legacy = dict(success=True, rows=10624, rows_sha256='b'*64,
                      protocol_sha256='c'*64, source_manifest_sha256='d'*64)
        self.assertTrue(check_cache_header(legacy, **dict(expected, count=10624, matched_sha=None)))

    def test_collector_rejects_bad_manifest_before_missing_model_or_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'bad.json'
            source.write_text('{}', encoding='utf-8')
            args = []
            for name in ('executable', 'model', 'pack', 'reference', 'sources', 'protocol'):
                args += ['--' + name, str(root / ('missing-' + name))]
            result = subprocess.run([sys.executable, '-B', '-m', 'tools.igc.context_cache',
                *args, '--output', str(root / 'new'), '--corpus-manifest', str(source)],
                capture_output=True, text=True, timeout=20,
                creationflags=0x08000000 if sys.platform == 'win32' else 0)
            self.assertIn('unrecognized input manifest', result.stderr)
            self.assertFalse((root / 'new').exists())

    def test_structured_pairing_and_native_projection_drop_supervision_metadata(self):
        value = fixture()
        self.assertEqual(validate_structure(value), 1536)
        rows = native_prompt_rows(value['rows'])
        self.assertEqual(len(rows), 1536)
        self.assertEqual(set(rows[0]), {'id', 'prompt', 'expression'})
        self.assertEqual(rows[0]['prompt'], value['rows'][0]['prompt'])
        self.assertNotIn('ready_op', rows[0])
        self.assertEqual(cache_shapes(1536), ((1536, 80, 4, 2560), (1536, 80)))

    def test_schema_refuses_group_split_intent_and_target_leakage(self):
        value = fixture()
        for field, replacement in [('split', 'test'), ('group', 'wrong'), ('ready_op', 99),
                                   ('lhs', True), ('prompt', 'a different request')]:
            bad = copy.deepcopy(value)
            bad['rows'][0][field] = replacement
            with self.assertRaises(ValueError):
                validate_structure(bad)
        bad = copy.deepcopy(value)
        bad['rows'][0]['answer'] = 123
        with self.assertRaises(ValueError):
            validate_structure(bad)
        bad = copy.deepcopy(value)
        bad['rows'][1] = bad['rows'][0].copy()
        with self.assertRaises(ValueError):
            validate_structure(bad)

    def test_valid_schema_is_not_permission_to_accept_an_unpinned_manifest(self):
        value = fixture()
        self.assertEqual(validate_structure(value), 1536)
        with self.assertRaises(ValueError):
            validate_manifest(value)


if __name__ == '__main__':
    unittest.main()
