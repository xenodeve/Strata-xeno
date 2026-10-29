"""Independent content-binding checks; no model inference or firmware execution."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':')).encode()


@unittest.skipUnless(os.environ.get('QWENPUTER_BINDING_TEST'), 'native content binder not selected')
class NativeBinding(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.pack = self.root / 'tokenizer'; self.pack.mkdir()
        self.files = {name: name.encode() + b'\0\xff' for name in
                      ('tokenizer.json', 'vocab.json', 'merges.txt', 'token_type.json')}
        for name, data in self.files.items():(self.pack / name).write_bytes(data)
        self.model = self.root / 'model.bin'; self.model.write_bytes(b'abc')
        self.model_sha = hashlib.sha256(b'abc').hexdigest()

    def tokenizer_sha(self):
        return hashlib.sha256(canonical({name: {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
                                         for name, data in self.files.items()})).hexdigest()

    def run_binding(self, model_sha=None, tokenizer_sha=None, models=None, expected=0, code=None):
        p = subprocess.run([os.environ['QWENPUTER_BINDING_TEST'], str(self.pack),
                            model_sha or self.model_sha, tokenizer_sha or self.tokenizer_sha(),
                            *map(str, models or [self.model])], capture_output=True, text=True,
                           timeout=60, creationflags=0x08000000)
        self.assertEqual(p.returncode, expected, p.stdout+p.stderr)
        if code:self.assertIn(code, p.stdout)

    def test_single_source_exact_binding_and_optional_template(self):
        self.run_binding()
        self.files['chat_template.jinja'] = b'chat template\n'
        (self.pack / 'chat_template.jinja').write_bytes(self.files['chat_template.jinja'])
        self.run_binding()

    def test_tokenizer_mismatch_is_refused_before_model_read(self):
        self.run_binding(tokenizer_sha='00'*32, models=[self.root/'absent'], expected=1, code='tokenizer_binding')

    def test_model_content_change_is_refused(self):
        self.model.write_bytes(b'abd')
        self.run_binding(expected=1, code='model_binding')

    def test_missing_input_never_publishes_partial_proof(self):
        (self.pack/'vocab.json').unlink()
        self.run_binding(expected=1, code='binding_file')

    def test_multiple_sources_bind_ordered_contents_without_paths(self):
        other=self.root/'second.bin';other.write_bytes(b'def')
        descriptors=[{'bytes':3,'sha256':hashlib.sha256(raw).hexdigest()} for raw in (b'abc',b'def')]
        sha=hashlib.sha256(canonical({'kind':'strata-model-files-v1','shards':descriptors})).hexdigest()
        self.run_binding(model_sha=sha,models=[self.model,other])
        self.run_binding(model_sha=sha,models=[other,self.model],expected=1,code='model_binding')

    def test_chunk_boundary_data_matches_independent_hashlib(self):
        raw=bytes(range(256))*(4*1024*1024//256) + b'tail'
        self.model.write_bytes(raw)
        self.run_binding(model_sha=hashlib.sha256(raw).hexdigest())


if __name__ == '__main__': unittest.main()
