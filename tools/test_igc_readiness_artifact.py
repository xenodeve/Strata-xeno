import hashlib,unittest,struct,os,subprocess,tempfile,json
from pathlib import Path
import numpy as np
import torch
from tools.igc.adapter_artifact import schema,pack_state,read_artifact,HEADER,DESCRIPTOR


class ReadinessArtifact(unittest.TestCase):
    def test_kind3_is_explicit_and_separate_from_operand_and_result_schemas(self):
        try:definition=schema(3)
        except ValueError:definition=None
        self.assertIsNotNone(definition,'diagnostic readiness kind3 is not implemented')
        self.assertEqual(definition['semantics']['name'],'final-state-binary-readiness-v1')
        self.assertEqual([r['name'] for r in definition['tensors']],
                         ['mean','scale','0.weight','0.bias','2.weight','2.bias'])
        self.assertEqual(schema(1)['sha256'],'cb73a018f1b785602b1b05e3754dbd0d1687f010efa92526f2bc92ea994a3101')
        self.assertEqual(schema(2)['sha256'],'8e443014ea93de240eccc32fc7a196fa07321cdd594b2f02350853c16cde2395')

    def test_known_normalization_and_checked_positive_scale_roundtrip(self):
        from tools.igc.readiness_mapping import ReadinessMapper
        torch.manual_seed(20261107);model=ReadinessMapper();model.mean.copy_(torch.linspace(-1,1,2560));model.scale.fill_(2)
        x=torch.arange(5120,dtype=torch.float32).reshape(2,2560)/5120
        with torch.no_grad():
            expected=torch.nn.Sequential.forward(model,(x-model.mean)/model.scale)
            self.assertEqual(model(x).numpy().tobytes(),expected.numpy().tobytes())
        state=model.state_dict();raw=pack_state(3,state,'11'*32,'22'*32,'33'*32,'44'*32)
        result=read_artifact(raw,'11'*32,'22'*32,hashlib.sha256(raw).hexdigest())
        self.assertEqual(result['kind'],3);self.assertTrue(result['diagnostic_only'])
        for k,v in state.items():self.assertEqual(result['tensors'][k].tobytes(),v.numpy().tobytes())
        with self.assertRaises(ValueError):read_artifact(raw,'11'*32,'22'*32,hashlib.sha256(raw).hexdigest(),production=True)
        for invalid in (0.,-1.,0.00001):
            bad={k:v.clone() for k,v in state.items()};bad['scale'][0]=invalid
            with self.assertRaisesRegex(ValueError,'scale'):pack_state(3,bad,'11'*32,'22'*32,'33'*32,'44'*32)
            changed=bytearray(raw);count=HEADER.unpack_from(changed)[4]
            start=(HEADER.size+count*DESCRIPTOR.size+15)&~15
            scale_offset=DESCRIPTOR.unpack_from(changed,HEADER.size+DESCRIPTOR.size)[6]
            struct.pack_into('<f',changed,start+scale_offset,invalid);changed[-32:]=hashlib.sha256(changed[:-32]).digest()
            with self.assertRaisesRegex(ValueError,'scale'):
                read_artifact(bytes(changed),'11'*32,'22'*32,hashlib.sha256(changed).hexdigest())

    @unittest.skipUnless(os.environ.get('IGC_ADAPTER_PROBE'),'native host reader not selected')
    def test_native_host_roundtrip_and_resigned_scale_rejection(self):
        from tools.igc.readiness_mapping import ReadinessMapper
        raw=pack_state(3,ReadinessMapper().state_dict(),'11'*32,'22'*32,'33'*32,'44'*32)
        with tempfile.TemporaryDirectory(prefix='igc-readiness-host-') as name:
            root=Path(name);source=root/'input.bin'
            for i,invalid in enumerate((None,0.,-1.,.00001)):
                changed=bytearray(raw)
                if invalid is not None:
                    start=(HEADER.size+6*DESCRIPTOR.size+15)&~15
                    offset=DESCRIPTOR.unpack_from(changed,HEADER.size+DESCRIPTOR.size)[6]
                    struct.pack_into('<f',changed,start+offset,invalid);changed[-32:]=hashlib.sha256(changed[:-32]).digest()
                source.write_bytes(changed)
                run=subprocess.run([os.environ['IGC_ADAPTER_PROBE'],str(source),'11'*32,'22'*32,
                    hashlib.sha256(changed).hexdigest(),str(root/f'out{i}'),'--diagnostic'],
                    capture_output=True,creationflags=0x08000000,timeout=10)
                if invalid is None:
                    self.assertEqual(run.returncode,0,run.stderr);self.assertEqual(json.loads(run.stdout)['kind'],3)
                    production=subprocess.run([os.environ['IGC_ADAPTER_PROBE'],str(source),'11'*32,'22'*32,
                        hashlib.sha256(changed).hexdigest(),str(root/'production')],
                        capture_output=True,creationflags=0x08000000,timeout=10)
                    self.assertEqual(production.returncode,2);self.assertIn(b'igc_adapter_diagnostic_only',production.stderr)
                else:
                    self.assertEqual(run.returncode,2);self.assertIn(b'igc_adapter_readiness_scale',run.stderr)
            from tools.igc.attention_readout import InputReadout
            from tools.igc.output_mapping import ResultMapper
            for kind,model in ((1,InputReadout(False,True)),(2,ResultMapper())):
                compatible=pack_state(kind,model.state_dict(),'11'*32,'22'*32,'33'*32,'44'*32);source.write_bytes(compatible)
                run=subprocess.run([os.environ['IGC_ADAPTER_PROBE'],str(source),'11'*32,'22'*32,
                    hashlib.sha256(compatible).hexdigest(),str(root/f'old{kind}'),'--diagnostic'],
                    capture_output=True,creationflags=0x08000000,timeout=10)
                self.assertEqual(run.returncode,0,run.stderr);self.assertEqual(json.loads(run.stdout)['kind'],kind)


if __name__=='__main__':unittest.main()
