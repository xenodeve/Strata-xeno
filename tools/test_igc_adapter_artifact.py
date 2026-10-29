import hashlib
import struct
import unittest
import tempfile
from pathlib import Path
import torch
from tools.igc.adapter_artifact import pack_state,read_artifact,read_file,schema,HEADER,DESCRIPTOR
from tools.igc.attention_readout import InputReadout

MODEL='11'*32;TOKENIZER='22'*32;STATE='33'*32;PROTOCOL='44'*32


class AdapterArtifact(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(81)
        cls.weights=InputReadout(use_final_query=False,contextual=True).state_dict()
        cls.raw=pack_state(1,cls.weights,MODEL,TOKENIZER,STATE,PROTOCOL)
    def read(self,raw,**kwargs):
        return read_artifact(raw,kwargs.get('model',MODEL),TOKENIZER,hashlib.sha256(raw).hexdigest(),
                             production=kwargs.get('production',False))
    def changed(self,offset,value):
        raw=bytearray(self.raw);raw[offset:offset+len(value)]=value
        raw[-32:]=hashlib.sha256(raw[:-32]).digest();return bytes(raw)
    def test_exact_named_weights_and_explicit_diagnostic_scope(self):
        result=self.read(self.raw)
        self.assertEqual(result['kind'],1);self.assertEqual(result['max_digits'],2)
        self.assertTrue(result['diagnostic_only'])
        for name,value in self.weights.items():
            self.assertEqual(result['tensors'][name].tobytes(),value.numpy().tobytes())
        with self.assertRaises(ValueError):self.read(self.raw,production=True)
    def test_binding_corruption_and_trailing_bytes_are_refused(self):
        with self.assertRaises(ValueError):self.read(self.raw,model='55'*32)
        for raw in (self.raw[:-1],self.raw+b'x',self.raw[:300]+b'X'+self.raw[301:]):
            with self.assertRaises(ValueError):self.read(raw)
    def test_resigned_bad_schema_offsets_and_nonfinite_are_still_refused(self):
        # A valid checksum does not authorize malformed tensor metadata.
        for offset,value in [(12,struct.pack('<I',99)),(HEADER.size+8,struct.pack('<I',255)),
                             (HEADER.size+24,struct.pack('<Q',16))]:
            with self.assertRaises(ValueError):self.read(self.changed(offset,value))
        count=len(schema(1)['tensors']);payload=(HEADER.size+count*DESCRIPTOR.size+15)&~15
        with self.assertRaises(ValueError):self.read(self.changed(payload,struct.pack('<f',float('nan'))))
    def test_exporter_does_not_accept_a_different_network_or_dtype(self):
        bad=dict(self.weights);bad['extra']=torch.zeros(1)
        with self.assertRaises(ValueError):pack_state(1,bad,MODEL,TOKENIZER,STATE,PROTOCOL)

    def test_output_kind_and_regular_file_boundary(self):
        from tools.igc.output_mapping import ResultMapper
        state=ResultMapper().state_dict();raw=pack_state(2,state,MODEL,TOKENIZER,STATE,PROTOCOL)
        decoded=self.read(raw);self.assertEqual(decoded['max_digits'],62)
        for name,tensor in state.items():self.assertEqual(decoded['tensors'][name].tobytes(),tensor.numpy().tobytes())
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'mapper.bin';path.write_bytes(raw)
            self.assertEqual(read_file(path,MODEL,TOKENIZER,hashlib.sha256(raw).hexdigest())['kind'],2)
            with self.assertRaises(ValueError):read_file(Path(tmp),MODEL,TOKENIZER,hashlib.sha256(raw).hexdigest())

    def test_native_schema_is_derived_from_current_architectures(self):
        from tools.igc.generate_adapter_schema import render
        path=Path(__file__).resolve().parents[1]/'include/strata/core/igc_adapter_schema.hpp'
        self.assertEqual(path.read_text(encoding='utf-8'),render())
        bad=dict(self.weights);key=next(iter(bad));bad[key]=bad[key].half()
        with self.assertRaises(ValueError):pack_state(1,bad,MODEL,TOKENIZER,STATE,PROTOCOL)


if __name__=='__main__':unittest.main()
