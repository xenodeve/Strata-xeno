"""Source trust checks do not import or execute any upstream Python."""
from pathlib import Path
import os
import shutil
import subprocess
import tempfile
import unittest

from tools.qwenputer.source import REVISION, verify_source
from tools.qwenputer.tokenizer import NativeInputError


VENDOR = Path(__file__).resolve().parents[1] / 'third_party/qwenputer'


class SourceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / 'source'
        shutil.copytree(VENDOR, self.root)

    def tearDown(self):
        self.temp.cleanup()

    def error(self, code):
        with self.assertRaises(NativeInputError) as caught:
            verify_source(self.root)
        self.assertEqual(caught.exception.code, code)

    def test_exact_pin_and_all_thirteen_files(self):
        manifest = verify_source(self.root)
        self.assertEqual(manifest['revision'], REVISION)
        self.assertEqual(len(manifest['files']), 13)

    def test_manifest_cannot_authorize_its_own_new_hashes(self):
        path = self.root / 'UPSTREAM.json'
        path.write_bytes(path.read_bytes().replace(REVISION.encode(), b'0' * 40))
        self.error('compiler_manifest')

    def test_one_modified_or_missing_source_is_rejected(self):
        path = self.root / 'experimental/riscv/chipformer/rv/core.py'
        original = path.read_bytes()
        path.write_bytes(original + b'\n# changed\n')
        self.error('compiler_source')
        path.unlink()
        self.error('compiler_source')

    def test_unreviewed_extension_or_bytecode_cannot_shadow_python(self):
        for name in ('rv/core.pyd', 'rv/__pycache__/core.cpython-311.pyc'):
            with self.subTest(name=name):
                path = self.root / 'experimental/riscv/chipformer' / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b'not executed')
                self.error('compiler_unreviewed_directory' if '__pycache__' in name
                           else 'compiler_unreviewed_file')
                path.unlink()

    def test_unreviewed_import_directory_is_rejected_even_if_empty(self):
        (self.root / 'experimental/riscv/numpy').mkdir()
        self.error('compiler_unreviewed_directory')

    def test_internal_directory_symlink_cannot_shadow_a_dependency(self):
        target = self.root / 'shadow-numpy'
        target.mkdir()
        (target / '__init__.py').write_text('# marker only; never imported\n')
        link = self.root / 'experimental/riscv/numpy'
        try:
            link.symlink_to(target, target_is_directory=True)
        except OSError as error:
            self.skipTest(f'test-owned symlinks unavailable: {error}')
        self.error('compiler_reparse')

    @unittest.skipUnless(os.name == 'nt', 'Windows junction test')
    def test_internal_windows_junction_cannot_shadow_a_dependency(self):
        target = self.root / 'junction-target'
        target.mkdir()
        (target / '__init__.py').write_text('# marker only; never imported\n')
        link = self.root / 'experimental/riscv/numpy'
        self.assertTrue(target.resolve().is_relative_to(self.root.resolve()))
        self.assertTrue(link.parent.resolve().is_relative_to(self.root.resolve()))
        result = subprocess.run(['cmd.exe', '/d', '/c', 'mklink', '/J', str(link), str(target)],
                                capture_output=True, text=True, errors='replace',
                                creationflags=0x08000000, timeout=10)
        if result.returncode:
            self.skipTest('test-owned junction unavailable: ' + result.stderr)
        try:
            self.assertTrue(link.lstat().st_file_attributes & 0x400)
            self.error('compiler_reparse')
        finally:
            link.rmdir()  # removes the test link itself, never its target tree


if __name__ == '__main__':
    unittest.main()
