"""Opt-in host-only native loader boundaries; no GPU or inference claim."""
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest
from tools.test_igc_adapter_artifact import MODEL,TOKENIZER,STATE,PROTOCOL
from tools.igc.adapter_artifact import pack_state,HEADER,DESCRIPTOR,schema
from tools.igc.attention_readout import InputReadout


@unittest.skipUnless(os.environ.get('IGC_ADAPTER_PROBE'),'native adapter host probe not selected')
class NativeAdapter(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.raw=pack_state(1,InputReadout(False,True).state_dict(),MODEL,TOKENIZER,STATE,PROTOCOL)
    def run_native(self,raw,allow=True,expected=None):
        with tempfile.TemporaryDirectory() as tmp:
            source=Path(tmp)/'adapter.bin';output=Path(tmp)/'payload.bin';source.write_bytes(raw)
            command=[os.environ['IGC_ADAPTER_PROBE'],str(source),MODEL,TOKENIZER,
                     expected or hashlib.sha256(raw).hexdigest(),str(output)]
            if allow:command.append('--diagnostic')
            process=subprocess.run(command,creationflags=0x08000000,capture_output=True,timeout=10)
            return process,output.read_bytes() if output.exists() else None
    def test_native_payload_and_metadata_match_the_python_contract(self):
        from tools.igc.output_mapping import ResultMapper
        output=pack_state(2,ResultMapper().state_dict(),MODEL,TOKENIZER,STATE,PROTOCOL)
        for kind,raw in [(1,self.raw),(2,output)]:
            result,payload=self.run_native(raw)
            self.assertEqual(result.returncode,0,result.stderr)
            record=json.loads(result.stdout);self.assertEqual((record['kind'],record['tensors']),(kind,len(schema(kind)['tensors'])))
            begin=(HEADER.size+len(schema(kind)['tensors'])*DESCRIPTOR.size+15)&~15
            self.assertEqual(payload,raw[begin:-32])
    def test_refusals_publish_no_payload(self):
        changed=bytearray(self.raw);struct.pack_into('<Q',changed,HEADER.size+24,16)
        changed[-32:]=hashlib.sha256(changed[:-32]).digest()
        nonfinite=bytearray(self.raw);begin=(HEADER.size+len(schema(1)['tensors'])*DESCRIPTOR.size+15)&~15
        struct.pack_into('<f',nonfinite,begin,float('nan'));nonfinite[-32:]=hashlib.sha256(nonfinite[:-32]).digest()
        cases=[(self.raw,False,None),(self.raw,True,'00'*32),(self.raw[:-1],True,None),
               (self.raw+b'x',True,None),(bytes(changed),True,None),(bytes(nonfinite),True,None)]
        for raw,allow,expected in cases:
            result,payload=self.run_native(raw,allow,expected)
            self.assertNotEqual(result.returncode,0);self.assertIsNone(payload)


if __name__=='__main__':unittest.main()
