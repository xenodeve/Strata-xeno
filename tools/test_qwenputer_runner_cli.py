"""Prefix-only native execution contract; no reference trace is passed to the binary."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from tools.qwenputer.tokenizer import NativeText


@unittest.skipUnless(all(os.environ.get(k) for k in ('QWENPUTER_RUN', 'QWENPUTER_RUN_PREFIX',
                        'QWENPUTER_ARTIFACT', 'QWENPUTER_TOKENIZER')), 'prefix-only native CLI not selected')
class PrefixOnlyExecution(unittest.TestCase):
    def test_result_comes_from_autonomous_execution_and_is_bound_to_artifact(self):
        artifact = Path(os.environ['QWENPUTER_ARTIFACT'])
        manifest = json.loads((artifact.parent / 'manifest.json').read_bytes())
        names = ('abi_sha256', 'source_sha256', 'firmware_sha256', 'boot_sha256', 'model_sha256', 'tokenizer_sha256')
        with tempfile.TemporaryDirectory() as temp:
            binding = Path(temp) / 'binding.bin'
            binding.write_bytes(b''.join(bytes.fromhex(manifest['identities'][k]) for k in names))
            run = subprocess.run([os.environ['QWENPUTER_RUN'], str(artifact), str(binding),
                                  os.environ['QWENPUTER_RUN_PREFIX']], capture_output=True, text=True,
                                 timeout=120, creationflags=0x08000000)
        self.assertEqual(run.returncode, 0, run.stderr)
        result = json.loads(run.stdout)
        self.assertEqual(result.get('kind'), 'native-execution')
        self.assertEqual(result['reason'], 3)
        self.assertEqual(result['error'], 0)
        if os.name == 'nt':
            self.assertTrue(result['process_memory_available'])
            self.assertGreater(result['process_private_commit_bytes'], 0)
            self.assertGreaterEqual(result['process_private_commit_peak_bytes'], result['process_private_commit_bytes'])
            self.assertGreater(result['working_set_peak_bytes'], 0)
        self.assertEqual(result['owned_device_buffer_peak_bytes'], result['owned_gpu_bytes'])
        for key in ('cuda_free_before_executor_bytes', 'cuda_free_after_executor_bytes',
                    'cuda_free_after_runtime_bytes', 'cuda_free_after_execution_bytes'):
            self.assertLessEqual(0, result[key])
            self.assertLessEqual(result[key], result['cuda_total_bytes'])
        self.assertEqual(result['artifact_sha256'], hashlib.sha256(artifact.read_bytes()).hexdigest())
        text = NativeText.from_pack(Path(os.environ['QWENPUTER_TOKENIZER']))
        ids = result['text_ids']
        start = ids.index(text.printable[256])
        stop = ids.index(text.printable[257], start + 1)
        self.assertEqual(text.decode_bytes(ids[start + 1:stop]).strip(), b'391')


if __name__ == '__main__':
    unittest.main()
