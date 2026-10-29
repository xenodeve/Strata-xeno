"""Native embedding control fidelity, not calculator or model-integration evidence."""
import os
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest
import numpy as np
from serve.gguf_info import read_header


@unittest.skipUnless(os.environ.get('IGC_EMBEDDING_PROBE'),'native embedding control not selected')
class EmbeddingControl(unittest.TestCase):
    def invoke(self,model,source,target):
        return subprocess.run([os.environ['IGC_EMBEDDING_PROBE'],str(model),str(source),str(target)],
                              capture_output=True,text=True,timeout=60,creationflags=0x08000000)

    def test_invalid_ids_refused_before_model_loading(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'ids';target=root/'out'
            for raw in [b'',b'x',struct.pack('<i',-1),struct.pack('<i',248320),bytes(1025*4)]:
                source.write_bytes(raw)
                run=self.invoke('absent-model',source,target)
                self.assertEqual(run.returncode,2,run.stdout+run.stderr)
                self.assertIn('embedding_token_',run.stderr)
                self.assertFalse(target.exists())

    def test_native_embedding_matches_independent_q8_rows_and_hc_broadcast(self):
        ids=[0,15,16,22,24,12,16]
        model=os.environ['IGC_MODEL']
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'ids';target=root/'out'
            source.write_bytes(struct.pack('<7i',*ids))
            run=self.invoke(model,source,target)
            self.assertEqual(run.returncode,0,run.stdout+run.stderr)
            self.assertIn('repeat=bitwise',run.stdout)
            actual=np.fromfile(target,dtype='<f4').reshape(len(ids),4,2560)
            header=read_header(model)
            tensor=next(t for t in header['tensors'] if t['name']=='token_embd.weight')
            self.assertEqual((tensor['type'],tensor['dims']),(8,[2560,248320]))
            with open(model,'rb') as weights:
                for i,token in enumerate(ids):
                    weights.seek(header['data_start']+tensor['offset']+token*80*34)
                    blocks=np.frombuffer(weights.read(80*34),dtype=np.uint8).reshape(80,34)
                    scales=blocks[:,:2].copy().view('<f2').reshape(80).astype(np.float32)
                    codes=blocks[:,2:].copy().view(np.int8).astype(np.float32)
                    expected=(scales[:,None]*codes).reshape(2560)
                    for stream in range(4):np.testing.assert_array_equal(actual[i,stream],expected)
            np.testing.assert_array_equal(actual[2],actual[6])


if __name__=='__main__':
    unittest.main()
