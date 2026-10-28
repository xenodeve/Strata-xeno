import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from tools.qwenputer.bundle import assemble_bundle
from tools.qwenputer.tokenizer import NativeInputError


class BundleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.export, self.oracle, self.isa = [self.root / n for n in ('export', 'oracle', 'isa')]
        for path in (self.export, self.oracle, self.isa):
            path.mkdir()
        data = b'fixture container; the bundle verifies bytes, not circuit semantics'
        self.digest = hashlib.sha256(data).hexdigest()
        (self.export / 'native.qput').write_bytes(data)
        for n in ('LICENSE.compiler', 'NOTICE.txt', 'UPSTREAM.json'):
            (self.export / n).write_bytes(b'fixture provenance')
        self.base = {'artifact_sha256': self.digest, 'artifact_bytes': len(data),
                     'identities': {'source_sha256': 'a' * 64, 'tokenizer_sha256': 'b' * 64},
                     'oracle_status': 'pending', 'native_execution_tested': False}
        self.write(self.export / 'manifest.json', self.base)
        for path in (self.oracle, self.isa):
            (path / 'vectors.json').write_bytes(b'[1,2,3]')
            files = {'vectors.json': {'bytes': 7, 'sha256': hashlib.sha256(b'[1,2,3]').hexdigest()}}
            self.write(path / 'manifest.json', dict(files=files, ir=True, cases=1,
                       source_sha256='a' * 64, tokenizer_sha256='b' * 64, artifact_sha256=self.digest))

    def write(self, path, value):
        path.write_text(json.dumps(value), encoding='utf-8')

    def test_data_and_provenance_are_reproducible_and_status_is_explicit(self):
        a, b = self.root / 'a', self.root / 'b'
        result = assemble_bundle(self.export, self.oracle, self.isa, a)
        self.assertIsNotNone(result)
        self.assertEqual(result['oracle_status'], 'complete')
        self.assertFalse(result['native_execution_tested'])
        self.assertEqual((a / 'oracle/vectors.json').read_bytes(), b'[1,2,3]')
        assemble_bundle(self.export, self.oracle, self.isa, b)
        self.assertEqual((a / 'manifest.json').read_bytes(), (b / 'manifest.json').read_bytes())

    def test_corruption_path_escape_and_artifact_mismatch_are_rejected(self):
        path = self.oracle / 'manifest.json'
        original = json.loads(path.read_bytes())
        for variant in ('corrupt', 'escape', 'mismatch'):
            self.write(path, original)
            (self.oracle / 'vectors.json').write_bytes(b'[1,2,3]')
            if variant == 'corrupt':
                (self.oracle / 'vectors.json').write_bytes(b'[1,2,4]')
            else:
                changed = dict(original)
                if variant == 'escape':
                    changed['files'] = {'../vectors.json': original['files']['vectors.json']}
                else:
                    changed['artifact_sha256'] = 'c' * 64
                self.write(path, changed)
            with self.subTest(variant=variant), self.assertRaises(NativeInputError):
                assemble_bundle(self.export, self.oracle, self.isa, self.root / variant)


if __name__ == '__main__':
    unittest.main()
