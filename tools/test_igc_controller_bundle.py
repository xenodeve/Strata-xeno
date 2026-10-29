"""Host-only trust boundary; synthetic neural weights, never an inference oracle."""
import hashlib
import os
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest
import torch
from tools.igc.adapter_artifact import pack_state
from tools.igc.attention_readout import InputReadout
from tools.igc.output_mapping import ResultMapper
from tools.igc.readiness_mapping import ReadinessMapper

MODEL='11'*32;TOKENIZER='22'*32
NAMES=('input-embedding.bin','input-attention.bin','readiness.bin','output.bin')

@unittest.skipUnless(os.environ.get('IGC_CONTROLLER_BUNDLE_PROBE'),'native controller bundle reader not selected')
class ControllerBundle(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(182)
        models=(InputReadout(False,True),InputReadout(False,True),ReadinessMapper(),ResultMapper())
        cls.raw=[pack_state(kind,m.state_dict(),MODEL,TOKENIZER,f'{i+3:02x}'*32,'55'*32) for i,(kind,m) in enumerate(zip((1,1,3,2),models,strict=True))]
    def prepare(self,root):
        for name,raw in zip(NAMES,self.raw,strict=True):(root/name).write_bytes(raw)
        return b'IGCLAT01'+struct.pack('<I',1)+bytes.fromhex(MODEL)+bytes.fromhex(TOKENIZER)+b''.join(hashlib.sha256(r).digest() for r in self.raw)
    def run_reader(self,path,production=False):
        result=subprocess.run([os.environ['IGC_CONTROLLER_BUNDLE_PROBE'],str(path)]+(['--production'] if production else []),capture_output=True,creationflags=0x08000000,timeout=10)
        return result
    def test_only_bound_neural_roles_are_loaded(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);path=root/'controller.bin';path.write_bytes(self.prepare(root))
            result=self.run_reader(path)
            self.assertEqual(result.returncode,0,result.stderr.decode(errors='replace'))
            self.assertIn(b'PASS controller_bundle kinds=1132 diagnostic_only=1',result.stdout)
    def test_production_is_refused_before_file_access(self):
        result=self.run_reader(Path('missing-controller-file'),True)
        self.assertEqual(result.returncode,2)
        self.assertIn(b'controller_bundle_diagnostic_only',result.stderr)
    def test_fixed_manifest_size_header_and_nonzero_bindings(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);path=root/'controller.bin';valid=self.prepare(root)
            zeroed=[valid[:p]+bytes(32)+valid[p+32:] for p in range(12,len(valid),32)]
            for data in (valid[:-1],valid+b'x',b'WRONG!!!'+valid[8:],valid[:8]+struct.pack('<I',2)+valid[12:],*zeroed):
                path.write_bytes(data);result=self.run_reader(path)
                self.assertEqual(result.returncode,2);self.assertIn(b'controller_bundle_manifest',result.stderr)
            result=self.run_reader(root);self.assertEqual(result.returncode,2);self.assertIn(b'controller_bundle_manifest',result.stderr)
    def test_wrong_digest_or_role_and_missing_network_are_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);path=root/'controller.bin';valid=self.prepare(root)
            path.write_bytes(valid[:76]+bytes.fromhex('99'*32)+valid[108:])
            result=self.run_reader(path);self.assertEqual(result.returncode,2);self.assertIn(b'controller_bundle_network',result.stderr)
            for offset in (12,44):
                path.write_bytes(valid[:offset]+bytes.fromhex('99'*32)+valid[offset+32:]);result=self.run_reader(path)
                self.assertEqual(result.returncode,2);self.assertIn(b'controller_bundle_network',result.stderr)
            # A correctly hashed kind2 network cannot occupy the readiness role.
            (root/NAMES[2]).write_bytes(self.raw[3]);changed=bytearray(valid);changed[140:172]=hashlib.sha256(self.raw[3]).digest();path.write_bytes(changed)
            result=self.run_reader(path);self.assertEqual(result.returncode,2);self.assertIn(b'controller_bundle_role',result.stderr)
            path.write_bytes(self.prepare(root));(root/NAMES[1]).unlink()
            result=self.run_reader(path);self.assertEqual(result.returncode,2);self.assertIn(b'controller_bundle_network',result.stderr)

    def test_failed_replacement_preserves_the_entire_old_bundle(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);path=root/'controller.bin';path.write_bytes(self.prepare(root));bad=root/'bad.bin';bad.write_bytes(b'bad')
            result=subprocess.run([os.environ['IGC_CONTROLLER_BUNDLE_PROBE'],str(path),'--transaction',str(bad)],capture_output=True,creationflags=0x08000000,timeout=10)
            self.assertEqual(result.returncode,0,result.stderr.decode(errors='replace'))
            self.assertIn(b'transaction=preserved',result.stdout)
            late=root/'late';late.mkdir();(late/'controller.bin').write_bytes(self.prepare(late));(late/NAMES[-1]).write_bytes(b'bad')
            result=subprocess.run([os.environ['IGC_CONTROLLER_BUNDLE_PROBE'],str(path),'--transaction',str(late/'controller.bin')],capture_output=True,creationflags=0x08000000,timeout=10)
            self.assertEqual(result.returncode,0,result.stderr.decode(errors='replace'));self.assertIn(b'transaction=preserved',result.stdout)

if __name__=='__main__':unittest.main()
