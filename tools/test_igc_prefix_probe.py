"""Opt-in fidelity of an offline layer-zero extractor, never model quality."""
import os
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest
import numpy as np
from tools.igc.observation import load_preanchor


@unittest.skipUnless(os.environ.get('IGC_PREFIX_PROBE'),'native prefix probe not selected')
class PrefixProbe(unittest.TestCase):
    def run_probe(self,data,output,window=8):
        source=output.parent/'input.bin';source.write_bytes(data)
        return subprocess.run([os.environ['IGC_PREFIX_PROBE'],os.environ['IGC_MODEL'],
            os.environ['IGC_PACK'],str(source),str(output),str(window)],
            creationflags=0x08000000,capture_output=True,timeout=90)

    def test_bad_input_is_refused_before_model_allocation(self):
        for data in (b'',b'IGCPFX01'+struct.pack('<I',0),
                     b'IGCPFX01'+struct.pack('<II',1,257),
                     b'IGCPFX01'+struct.pack('<IIii',1,2,0,248320),
                     b'IGCPFX01'+struct.pack('<IIii',1,2,0,1)+b'x'):
            with tempfile.TemporaryDirectory() as tmp:
                output=Path(tmp)/'out';output.mkdir()
                result=self.run_probe(data,output)
                self.assertEqual(result.returncode,2,result.stderr)
                self.assertIn(b'prefix_input',result.stderr)
                self.assertEqual(list(output.iterdir()),[])

    def test_full_model_fidelity_window_and_reset(self):
        root=Path(os.environ['IGC_CAPTURE_ROOT'])
        references=[load_preanchor(root,i,'attention') for i in (0,1,0)]
        records=[r['anchor']['prefix_tokens'] for r in references]
        self.assertTrue(all(2<=len(tokens)<=256 for tokens in records))
        data=b'IGCPFX01'+struct.pack('<I',len(records))+b''.join(
            struct.pack('<I',len(tokens))+tokens.astype('<i4').tobytes() for tokens in records)
        with tempfile.TemporaryDirectory() as tmp:
            outputs=[]
            for window in (1,8):
                output=Path(tmp)/str(window);output.mkdir()
                result=self.run_probe(data,output,window)
                self.assertEqual(result.returncode,0,result.stderr.decode(errors='replace'))
                self.assertIn(b'PASS native prefix',result.stdout)
                runs=[]
                for i,reference in enumerate(references):
                    raw=(output/f'{i}.attention.bin').read_bytes()
                    expected=(root/f'mixed.{(0,1,0)[i]}.attention.bin').read_bytes()
                    self.assertEqual(raw,expected,f'full-model first-attention fidelity window={window} row={i}')
                    runs.append(raw)
                self.assertEqual(runs[0],runs[2])
                outputs.append(runs)
            self.assertEqual(outputs[0],outputs[1])


if __name__=='__main__':unittest.main()
