import json
from pathlib import Path
import tempfile
import unittest
import subprocess
import sys
import os
import numpy as np
from tools.igc.observation import load_observation


class Observation(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.meta = {'version':1,'kind':'first_window_observation','stage':'head_mixed_after_final_hc',
                     'row':0,'position':2,'token':271,'prompt_tokens':3,'n_embd':2560,'hc':4,'n_vocab':248320}
        self.write_meta()
        np.array([10,20,271],dtype='<i4').tofile(self.root/'mixed.0.tokens.bin')
        np.ones(2560,dtype='<f4').tofile(self.root/'mixed.0')
        np.ones(248320,dtype='<f4').tofile(self.root/'logits.0')

    def write_meta(self):
        (self.root/'mixed.0.meta.json').write_text(json.dumps(self.meta))

    def test_binds_actual_token_prefix_and_typed_tensor_shapes(self):
        record = load_observation(self.root,0)
        self.assertEqual(record.get('position'),2)
        self.assertEqual(record.get('token'),271)
        self.assertEqual(record['mixed'].shape,(2560,))
        self.assertEqual(record['logits'].shape,(248320,))
        self.assertEqual(len(record['prefix_sha256']),64)

    def test_prefix_only_reader_does_not_load_array_runtime(self):
        script = ('import sys,json;from tools.igc.observation import load_prefix;'
                  'r=load_prefix(sys.argv[1],0);'
                  'assert r["prefix_tokens"]==[10,20,271];'
                  'assert "numpy" not in sys.modules;print(r["prefix_sha256"])')
        result = subprocess.run([sys.executable,'-B','-c',script,str(self.root)],
                                capture_output=True,timeout=30,
                                creationflags=0x08000000 if os.name=='nt' else 0)
        self.assertEqual(result.returncode,0,result.stderr.decode(errors='replace'))
        self.assertEqual(result.stdout.decode().strip(),load_observation(self.root,0)['prefix_sha256'])

    def test_wrong_boundary_or_prefix_is_rejected(self):
        for key,value in [('position',1),('token',272),('row',1),('n_embd',2561),('version',2)]:
            with self.subTest(key=key):
                prior=self.meta[key];self.meta[key]=value;self.write_meta()
                with self.assertRaises(ValueError):load_observation(self.root,0)
                self.meta[key]=prior
        self.write_meta()
        np.array([10,-1,271],dtype='<i4').tofile(self.root/'mixed.0.tokens.bin')
        with self.assertRaises(ValueError):load_observation(self.root,0)

    def test_truncated_nonfinite_and_stale_zero_observations_are_rejected(self):
        for raw in [b'x',np.full(2560,np.nan,dtype='<f4').tobytes(),bytes(2560*4)]:
            with self.subTest(bytes=len(raw)):
                (self.root/'mixed.0').write_bytes(raw)
                with self.assertRaises(ValueError):load_observation(self.root,0)

    def test_index_cannot_select_an_arbitrary_file(self):
        for index in [-1,1024,'../private']:
            with self.subTest(index=index), self.assertRaises(ValueError):load_observation(self.root,index)


if __name__ == '__main__':
    unittest.main()
