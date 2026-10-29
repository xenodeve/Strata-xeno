import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest
import numpy as np
from tools.igc.output_fixture import pack_fixture


@unittest.skipUnless(os.environ.get('IGC_FIXTURE_PROBE'),'native fixture validator not selected')
class NativeFixture(unittest.TestCase):
    def test_prefix_matching_and_malformed_input_never_require_model_allocation(self):
        artifact=Path(os.environ['IGC_OUTPUT_ARTIFACT']);digest=hashlib.sha256(artifact.read_bytes()).digest()
        record=dict(ids=[1,2,3],invocation=7,invocation_position=2,position=2,output_offset=0,
            checked_result_hex=struct.pack('<IIQQ64s',1,0,7,2,bytes([1,0,4])+bytes(61)).hex())
        raw=pack_fixture([record],bytes.fromhex(os.environ['IGC_MODEL_SHA256']),
                         bytes.fromhex(os.environ['IGC_TOKENIZER_SHA256']),digest)
        with tempfile.TemporaryDirectory(prefix='igc-fixture-') as name:
            root=Path(name);fixture=root/'fixture.bin';prefix=root/'prefix.bin'
            prefix.write_bytes(np.array([1,2,3],dtype='<i4').tobytes())
            def run(data):
                fixture.write_bytes(data)
                return subprocess.run([os.environ['IGC_FIXTURE_PROBE'],str(artifact),str(fixture),str(prefix)],
                    capture_output=True,creationflags=0x08000000,timeout=10)
            good=run(raw);self.assertEqual(good.returncode,0,good.stderr.decode());self.assertTrue(json.loads(good.stdout)['matched'])
            prefix.write_bytes(np.array([1,2,4],dtype='<i4').tobytes())
            other=run(raw);self.assertEqual(other.returncode,0);self.assertFalse(json.loads(other.stdout)['matched'])
            prefix.write_bytes(np.array([1,2,3],dtype='<i4').tobytes())
            anchor=bytearray(raw);struct.pack_into('<Q',anchor,156,3)
            malformed=bytearray(raw);malformed[164]=63
            duplicate=bytearray(raw[:108]+raw[108:]+raw[108:]);struct.pack_into('<I',duplicate,8,2)
            wrong_artifact=bytearray(raw);wrong_artifact[76]^=1
            for bad in (raw[:-1],raw+b'x',anchor,malformed,duplicate,wrong_artifact):
                outcome=run(bad);self.assertEqual(outcome.returncode,2,outcome.stdout.decode());self.assertFalse(outcome.stdout)
            prefix.write_bytes(np.array([1,-1,3],dtype='<i4').tobytes());self.assertEqual(run(raw).returncode,2)


if __name__=='__main__':unittest.main()
