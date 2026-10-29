"""Numerical probe only: not free-running model integration."""
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest
import numpy as np
import torch
from safetensors.torch import load_file
from tools.igc.output_mapping import ResultMapper,encode_results


@unittest.skipUnless(os.environ.get('IGC_OUTPUT_PROBE'),'native output probe not selected')
class NativeOutput(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.exe=Path(os.environ['IGC_OUTPUT_PROBE'])
        if not cls.exe.is_file():raise RuntimeError('selected native output probe is missing')
        cls.root=Path(os.environ['IGC_OUTPUT_CAPTURE'])
        cls.records=json.loads((cls.root/'records.json').read_text())
        cls.artifact=Path(os.environ['IGC_OUTPUT_ARTIFACT'])
        cls.binding=[os.environ['IGC_MODEL_SHA256'],os.environ['IGC_TOKENIZER_SHA256'],hashlib.sha256(cls.artifact.read_bytes()).hexdigest()]
        torch.set_num_threads(4);torch.manual_seed(20261105)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        cls.mapper=ResultMapper().to('cuda');cls.mapper.load_state_dict(load_file(os.environ['IGC_OUTPUT_STATE'],device='cuda'));cls.mapper.eval()

    def run_probe(self,mixed,raw,ids,anchors,mode='normal'):
        with tempfile.TemporaryDirectory(prefix='igc-output-') as name:
            root=Path(name);source=root/'inputs.bin';prefix=root/'out'
            source.write_bytes(b'IGCOIN01'+struct.pack('<I',len(mixed))+np.asarray(mixed,dtype='<f4').tobytes()+raw+
                np.asarray(ids,dtype='<u8').tobytes()+np.asarray(anchors,dtype='<u8').tobytes())
            run=subprocess.run([str(self.exe),str(self.artifact),*self.binding,str(source),str(prefix),mode],
                capture_output=True,creationflags=0x08000000,timeout=30)
            self.assertEqual(run.returncode,0,run.stderr.decode(errors='replace'))
            info=json.loads(run.stdout);self.assertTrue(info['repeat_bitwise'])
            actual=np.fromfile(str(prefix)+'.mixed.bin',dtype='<f4').reshape(len(mixed),2560)
            status=np.fromfile(str(prefix)+'.status.bin',dtype='<u4')
            return actual,status,info

    def test_421_real_positions_match_torch_and_regular_graph(self):
        worst=0.;outputs=[];references=[]
        for offset in range(0,len(self.records),64):
            records=self.records[offset:offset+64]
            mixed=np.stack([np.fromfile(self.root/f'mixed.{r["index"]}',dtype='<f4') for r in records])
            raw=b''.join(bytes.fromhex(r['checked_result_hex']) for r in records)
            ids=[r['invocation'] for r in records];anchors=[r['invocation_position'] for r in records]
            with torch.no_grad():
                # Keep the same16-row reference batches as the frozen fit's
                # final scoring, while testing native1/64 and graph paths.
                expected_parts=[]
                for start in range(0,len(records),16):
                    chunk=records[start:start+16]
                    features={k:v.to('cuda') for k,v in encode_results(raw[start*88:(start+len(chunk))*88],ids[start:start+16],anchors[start:start+16]).items()}
                    expected_parts.append(self.mapper(torch.from_numpy(mixed[start:start+16]).to('cuda'),features).cpu().numpy())
                expected=np.concatenate(expected_parts)
            actual,status,info=self.run_probe(mixed,raw,ids,anchors)
            self.assertTrue(np.all(status==0));self.assertTrue(np.isfinite(actual).all())
            np.testing.assert_allclose(actual,expected,rtol=1e-4,atol=1e-3)
            worst=max(worst,float(np.max(np.abs(actual-expected))));outputs.append(actual);references.append(expected)
            if offset==0:
                graph,graph_status,_=self.run_probe(mixed,raw,ids,anchors,'graph')
                self.assertEqual(graph.tobytes(),actual.tobytes());np.testing.assert_array_equal(graph_status,status)
                one,_,_=self.run_probe(mixed[:1],raw[:88],ids[:1],anchors[:1])
                self.assertEqual(one.tobytes(),actual[:1].tobytes())
        print('native output421 max absolute difference',worst,'device weights/scratch',info['device_weights_scratch_bytes'],flush=True)
        if os.environ.get('IGC_HEAD_BRIDGE_DLL'):
            from tools.igc.cuda_head import NativeCudaHead
            library=Path(os.environ['IGC_HEAD_BRIDGE_DLL']);candidate=np.concatenate(outputs);reference=np.concatenate(references)
            saved=json.loads((Path(os.environ['IGC_OUTPUT_STATE']).parent/'predictions.json').read_text())
            largest=0.;different=0
            with NativeCudaHead(library,hashlib.sha256(library.read_bytes()).hexdigest(),os.environ['IGC_MODEL'],torch.device('cuda:0')) as head:
                with torch.no_grad():
                    for offset in range(0,len(candidate),16):
                        a=head.project(torch.from_numpy(candidate[offset:offset+16]).to('cuda'))
                        b=head.project(torch.from_numpy(reference[offset:offset+16]).to('cuda'))
                        actual_ids=a.argmax(-1).cpu().tolist();expected_ids=b.argmax(-1).cpu().tolist()
                        different+=int(torch.count_nonzero(a!=b));largest=max(largest,float((a-b).abs().max()))
                        self.assertEqual(actual_ids,expected_ids)
                        self.assertEqual(expected_ids,[r['predicted']['learned'] for r in saved[offset:offset+16]])
            print('native output421 head decisions match; non-bitwise logits',different,'max difference',largest,flush=True)
        if os.environ.get('IGC_NATIVE_OUTPUT_ARCHIVE'):
            Path(os.environ['IGC_NATIVE_OUTPUT_ARCHIVE']).write_bytes(np.concatenate(outputs).tobytes())

    def test_malformed_failure_nonfinite_and_off_preserve_exact_bytes(self):
        records=self.records[:64]
        mixed=np.stack([np.fromfile(self.root/f'mixed.{r["index"]}',dtype='<f4') for r in records])
        chunks=[bytearray.fromhex(r['checked_result_hex']) for r in records]
        ids=[r['invocation'] for r in records];anchors=[r['invocation_position'] for r in records]
        struct.pack_into('<I',chunks[1],4,3);chunks[1][24:]=bytes(64)
        ids[2]+=1000;chunks[3][26]=10;mixed[4,2]=np.nan;mixed[5,:]=np.finfo(np.float32).max
        raw=b''.join(chunks)
        actual,status,_=self.run_probe(mixed,raw,ids,anchors)
        self.assertEqual(status[:6].tolist(),[0,1,2,2,3,3]);self.assertTrue(np.all(status[6:]==0))
        self.assertEqual(actual[1:6].tobytes(),mixed[1:6].tobytes())
        graph,graph_status,_=self.run_probe(mixed,raw,ids,anchors,'graph')
        self.assertEqual(graph.tobytes(),actual.tobytes());np.testing.assert_array_equal(graph_status,status)
        off,_,info=self.run_probe(mixed,raw,ids,anchors,'off')
        self.assertEqual(off.tobytes(),mixed.tobytes());self.assertEqual(info['device_weights_scratch_bytes'],0)


if __name__=='__main__':unittest.main()
