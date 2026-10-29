"""Native numerical parity of the frozen readiness head; not a quality pass."""
import hashlib,json,os,struct,subprocess,unittest,tempfile
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import load_file
from tools.igc.readiness_mapping import ReadinessMapper
from tools.igc.adapter_artifact import pack_state


@unittest.skipUnless(os.environ.get('IGC_READINESS_PROBE'),'native readiness probe not selected')
class NativeReadiness(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.exe=Path(os.environ['IGC_READINESS_PROBE'])
        if not cls.exe.is_file():raise RuntimeError('selected native readiness probe missing')
        cls.capture=Path(os.environ['IGC_READINESS_CAPTURE']);cls.root=Path(os.environ['IGC_READINESS_EVIDENCE']);cls.root.mkdir(exist_ok=False)
        cls.state=Path(os.environ['IGC_READINESS_STATE']);cls.mapper=ReadinessMapper();cls.mapper.load_state_dict(load_file(str(cls.state)));cls.mapper.eval()
        torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
        report=json.loads((cls.capture/'report.json').read_text());raw=(cls.capture/'records.json').read_bytes()
        if hashlib.sha256(raw).hexdigest()!=report['records_sha256']:raise RuntimeError('readiness record identity')
        cls.records=json.loads(raw);cls.saved=json.loads((cls.state.parent/'predictions.json').read_text())
        cls.mixed=[]
        for record in cls.records:
            source=(cls.capture/f'mixed.{record["index"]}').read_bytes()
            if hashlib.sha256(source).hexdigest()!=record['observation']['mixed_sha256']:raise RuntimeError('readiness vector identity')
            cls.mixed.append(np.frombuffer(source,dtype='<f4'))
        cls.mixed=np.stack(cls.mixed)
        cls.artifact=cls.root/'readiness.bin';cls.artifact.write_bytes(pack_state(3,cls.mapper.state_dict(),
            os.environ['IGC_MODEL_SHA256'],os.environ['IGC_TOKENIZER_SHA256'],hashlib.sha256(cls.state.read_bytes()).hexdigest(),
            report['protocol_sha256']))
        cls.binding=[os.environ['IGC_MODEL_SHA256'],os.environ['IGC_TOKENIZER_SHA256'],hashlib.sha256(cls.artifact.read_bytes()).hexdigest()]

    def probe(self,mixed,name,mode='normal'):
        source=self.root/(name+'.bin');out=self.root/(name+'-out')
        source.write_bytes(b'IGCRIN01'+struct.pack('<I',len(mixed))+np.asarray(mixed,dtype='<f4').tobytes())
        run=subprocess.run([str(self.exe),str(self.artifact),*self.binding,str(source),str(out),mode],
            capture_output=True,creationflags=0x08000000,timeout=30)
        self.assertEqual(run.returncode,0,run.stderr.decode(errors='replace'));info=json.loads(run.stdout)
        self.assertTrue(info['valid_repeats_bitwise']);self.assertTrue(info['failure_bytes_preserved'])
        return np.fromfile(str(out)+'.logits.bin',dtype='<f4').reshape(len(mixed),2),np.fromfile(str(out)+'.status.bin',dtype='<u4'),info

    def test_all320_real_states_keep_saved_decisions_and_graph_parity(self):
        worst=0.;predictions=[]
        for offset in range(0,len(self.mixed),64):
            mixed=self.mixed[offset:offset+64]
            with torch.no_grad():expected=self.mapper(torch.from_numpy(mixed)).numpy()
            actual,status,info=self.probe(mixed,f'data-{offset}')
            self.assertTrue(np.all(status==0));np.testing.assert_allclose(actual,expected,rtol=1e-4,atol=.001)
            np.testing.assert_array_equal(actual.argmax(1),expected.argmax(1))
            np.testing.assert_array_equal(expected.argmax(1),[r['predicted'] for r in self.saved[offset:offset+64]])
            predictions.extend(actual.argmax(1).tolist());worst=max(worst,float(np.max(np.abs(actual-expected))))
            if offset==0:
                graph,gs,_=self.probe(mixed,'graph','graph');self.assertEqual(graph.tobytes(),actual.tobytes());np.testing.assert_array_equal(gs,status)
                one,_,_=self.probe(mixed[:1],'one');self.assertEqual(one.tobytes(),actual[:1].tobytes())
        (self.root/'numerical.json').write_text(json.dumps(dict(max_abs_difference=worst,predictions=predictions,device_bytes=info['device_weights_scratch_bytes']),indent=2))
        print('native readiness320 maxabs',worst,'device bytes',info['device_weights_scratch_bytes'],flush=True)

    def test_nonfinite_overflow_off_preserve_bytes(self):
        mixed=self.mixed[:8].copy();mixed[1,2]=np.nan;mixed[2,:]=np.finfo(np.float32).max
        actual,status,_=self.probe(mixed,'invalid');self.assertEqual(status.tolist(),[0,2,2,0,0,0,0,0])
        self.assertEqual(actual[1:3].tobytes(),np.full((2,2),42.125,dtype=np.float32).tobytes())
        graph,gs,_=self.probe(mixed,'invalid-graph','graph');self.assertEqual(graph.tobytes(),actual.tobytes());np.testing.assert_array_equal(gs,status)
        off,codes,info=self.probe(mixed,'off','off');self.assertEqual(off.tobytes(),np.full((8,2),42.125,dtype=np.float32).tobytes())
        self.assertTrue(np.all(codes==0x12345678));self.assertEqual(info['device_weights_scratch_bytes'],0)


if __name__=='__main__':unittest.main()
