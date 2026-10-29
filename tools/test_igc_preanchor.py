"""Cross-language F32 selected-row format and cached-prefix absence checks."""
import json
import os
from pathlib import Path
import struct
import tempfile
import unittest
import numpy as np
from tools.igc.observation import load_preanchor

HEADER=struct.Struct('<8sIIIIIQQ')
ROW_BYTES=16+10240*4


class Preanchor(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.meta=dict(version=1,kind='first_window_observation',stage='head_mixed_after_final_hc',
                       row=0,position=99,token=99,prompt_tokens=100,n_embd=2560,hc=4,n_vocab=248320,
                       preanchor_format=1,preanchor_rows=4,preanchor_capacity=80)
        (self.root/'mixed.0.meta.json').write_text(json.dumps(self.meta))
        np.arange(100,dtype='<i4').tofile(self.root/'mixed.0.tokens.bin')
        np.ones(2560,dtype='<f4').tofile(self.root/'mixed.0')
        np.ones(248320,dtype='<f4').tofile(self.root/'logits.0')
        self.raw=bytearray(HEADER.pack(b'IGCPREA1',1,4,2560,80,4,99,100))
        for pos,source,value in [(19,2,.25),(20,1,.25),(21,1,1.25),(22,1,2.25)]:
            self.raw+=struct.pack('<qII',pos,pos,source)+np.full(10240,value,dtype='<f4').tobytes()
        self.path=self.root/'mixed.0.preanchor.bin'
        self.path.write_bytes(self.raw)

    def test_sparse_positions_are_bound_and_missing_cache_is_not_filled(self):
        result=load_preanchor(self.root,0)
        self.assertEqual(result['positions'],[19,20,21,22])
        self.assertEqual(result['sources'],[2,1,1,1])
        self.assertEqual(result['missing_positions'],list(range(23,99)))
        self.assertEqual(result['residuals'].shape,(4,4,2560))
        self.assertEqual(result['residuals'][2,0,0],1.25)

    def test_activation_stage_is_explicit_and_cannot_be_cross_loaded(self):
        self.meta.update(preanchor_embedding_stage='native_embedding_broadcast',preanchor_embedding_rows=4,
                         preanchor_attention_stage='native_first_attention_output',preanchor_attention_rows=4)
        (self.root/'mixed.0.meta.json').write_text(json.dumps(self.meta))
        for stage,magic,suffix in [('embedding',b'IGCEMB01','embedding'),('attention',b'IGCATN01','attention')]:
            raw=self.raw.copy();raw[:8]=magic
            path=self.root/f'mixed.0.{suffix}.bin';path.write_bytes(raw)
            result=load_preanchor(self.root,0,stage=stage)
            self.assertEqual(result['activation_stage'],stage)
            self.assertEqual(result['positions'],[19,20,21,22])
            path.write_bytes(self.raw)
            with self.assertRaises(ValueError):load_preanchor(self.root,0,stage=stage)

    @unittest.skipUnless(os.environ.get('IGC_PREANCHOR_FIXTURE'),'native CPU contract fixture not selected')
    def test_cpp_fixture_equals_independent_wire_fixture(self):
        fixture=Path(os.environ['IGC_PREANCHOR_FIXTURE'])
        self.assertEqual(fixture.read_bytes(),self.raw)
        for suffix,magic in [('embedding',b'IGCEMB01'),('attention',b'IGCATN01')]:
            expected=self.raw.copy();expected[:8]=magic
            self.assertEqual(Path(str(fixture)+'.'+suffix).read_bytes(),expected)

    def test_invalid_headers_positions_tokens_sources_and_nonfinite_rows_refused(self):
        for offset,fmt,value in [(0,'B',0),(8,'I',2),(12,'I',1),(16,'I',2559),(20,'I',81),(24,'I',81),
                                 (28,'Q',98),(36,'Q',101),(44,'q',99),(52,'I',20),(56,'I',0),(60,'f',float('nan'))]:
            raw=self.raw.copy();struct.pack_into('<'+fmt,raw,offset,value);self.path.write_bytes(raw)
            with self.subTest(offset=offset),self.assertRaises(ValueError):load_preanchor(self.root,0)

    def test_duplicate_truncated_and_trailing_records_refused(self):
        duplicate=self.raw.copy()
        duplicate[HEADER.size+ROW_BYTES:HEADER.size+2*ROW_BYTES]=self.raw[HEADER.size:HEADER.size+ROW_BYTES]
        for raw in [duplicate,self.raw[:-1],self.raw+b'x']:
            self.path.write_bytes(raw)
            with self.assertRaises(ValueError):load_preanchor(self.root,0)


if __name__=='__main__':
    unittest.main()
