"""Neural forward parity only. No learned-controller quality or serving claim."""
import hashlib,json,os,struct,subprocess,tempfile,unittest
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import load_file
from tools.igc.attention_readout import InputReadout
from tools.igc.adapter_artifact import pack_state
from tools.igc.readout import SIZES
from tools.igc.embedding_readout import load_cache


@unittest.skipUnless(os.environ.get('IGC_INPUT_PROBE'),'native input probe not selected')
class NativeInput(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.exe=Path(os.environ['IGC_INPUT_PROBE'])
        if not cls.exe.is_file():raise RuntimeError('selected native input probe missing')
        cls.cache=Path(os.environ['IGC_INPUT_CACHE']);cls.root=Path(os.environ['IGC_INPUT_EVIDENCE'])
        cls.context=None
        cls.root.mkdir(exist_ok=False)
        cls.state=Path(os.environ['IGC_INPUT_STATE']);cls.model=InputReadout(False,True)
        cls.model.load_state_dict(load_file(str(cls.state)));cls.model.eval()
        cls.model_binding=os.environ['IGC_MODEL_SHA256'];cls.tokenizer_binding=os.environ['IGC_TOKENIZER_SHA256']
        torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.mha.set_fastpath_enabled(False)
        cls.artifact=cls.root/'input.bin';cls.artifact.write_bytes(pack_state(1,cls.model.state_dict(),cls.model_binding,
            cls.tokenizer_binding,hashlib.sha256(cls.state.read_bytes()).hexdigest(),
            hashlib.sha256(Path(os.environ['IGC_INPUT_TRAIN_PROTOCOL']).read_bytes()).hexdigest()))
        cls.binding=[cls.model_binding,cls.tokenizer_binding,hashlib.sha256(cls.artifact.read_bytes()).hexdigest()]
        if os.environ.get('IGC_INPUT_CONTEXT')=='1':
            from tools.igc.context_cache import load_cache as load_context
            cls.context,cls.valid,cls.corpus,cls.cache_report=load_context(cls.cache,
                Path(os.environ['IGC_INPUT_SOURCES']),Path(os.environ['IGC_INPUT_CACHE_PROTOCOL']))
        else:
            cls.table,cls.indices,cls.valid,cls.corpus,cls.cache_report=load_cache(cls.cache,
                Path(os.environ['IGC_INPUT_SOURCES']),Path(os.environ['IGC_INPUT_CACHE_PROTOCOL']))
        cls.saved={r['id']:r['predicted'] for r in json.loads((cls.state.parent/'predictions.json').read_text())}

    @classmethod
    def tearDownClass(cls):
        if cls.context is not None:cls.context.close()

    def values(self,selection):
        return self.context[selection] if self.context is not None else self.table[self.indices[selection]].copy()

    def probe(self,states,valid,name,mode='normal'):
        source=self.root/(name+'.bin');prefix=self.root/(name+'-out')
        source.write_bytes(b'IGCIIN01'+struct.pack('<I',len(states))+np.asarray(states,dtype='<f4').tobytes()+np.asarray(valid,dtype=np.uint8).tobytes())
        run=subprocess.run([str(self.exe),str(self.artifact),*self.binding,str(source),str(prefix),mode],
            capture_output=True,creationflags=0x08000000,timeout=90)
        self.assertEqual(run.returncode,0,run.stderr.decode(errors='replace'))
        result=json.loads(run.stdout);self.assertTrue(result['valid_repeats_bitwise']);self.assertTrue(result['failure_bytes_preserved'])
        return np.fromfile(str(prefix)+'.logits.bin',dtype='<f4').reshape(len(states),52),np.fromfile(str(prefix)+'.status.bin',dtype='<u4'),result

    def test_native_features_and_graph_match_frozen_torch(self):
        # All known development rows, plus enough train rows to exercise64-way capacity.
        chosen=[i for i,r in enumerate(self.corpus) if r['split']!='train']
        worst=0.;decisions=[];ids=[]
        for offset in range(0,len(chosen),64):
            selection=np.array(chosen[offset:offset+64]);states=self.values(selection);valid=self.valid[selection].copy()
            args=(torch.from_numpy(states),torch.from_numpy(valid),torch.arange(80).expand(len(states),-1),torch.zeros(len(states),2560))
            with torch.no_grad():expected=self.model(*args).numpy()
            actual,status,info=self.probe(states,valid,f'dev-{offset}')
            self.assertTrue(np.all(status==0));np.testing.assert_allclose(actual,expected,rtol=1e-4,atol=1e-3)
            actual_fields=np.stack([a.argmax(1) for a in np.split(actual,np.cumsum(SIZES)[:-1],axis=1)],axis=1)
            expected_fields=np.stack([a.argmax(1) for a in np.split(expected,np.cumsum(SIZES)[:-1],axis=1)],axis=1)
            np.testing.assert_array_equal(actual_fields,expected_fields)
            np.testing.assert_array_equal(expected_fields,np.array([self.saved[self.corpus[i]['id']] for i in selection]))
            worst=max(worst,float(np.max(np.abs(actual-expected))));decisions.extend(actual_fields.tolist());ids.extend(int(i) for i in selection)
            if offset==0:
                graph,gs,_=self.probe(states,valid,'graph','graph');self.assertEqual(graph.tobytes(),actual.tobytes());np.testing.assert_array_equal(gs,status)
                one,_,_=self.probe(states[:1],valid[:1],'one');self.assertEqual(one.tobytes(),actual[:1].tobytes());self.assertTrue(info['alternating_input_changed'])
        (self.root/'numerical.json').write_text(json.dumps(dict(max_abs_difference=worst,indices=ids,predicted=decisions,device_bytes=info['device_weights_scratch_bytes']),indent=2))
        print('native input development maxabs',worst,'device bytes',info['device_weights_scratch_bytes'],flush=True)

    def test_invalid_mask_nonfinite_off_and_padding(self):
        states=self.values(np.arange(8,dtype=np.int64));valid=self.valid[:8].copy().astype(np.uint8)
        valid[1]=0;valid[2,0]=2
        first=np.flatnonzero(valid[3])[0];states[3,first,0,0]=np.nan
        valid[4,0]=0;states[4,0]=np.nan
        states[5,np.flatnonzero(valid[5])[0]]=np.finfo(np.float32).max
        actual,status,_=self.probe(states,valid,'invalid')
        self.assertEqual(status[:6].tolist(),[0,1,1,2,0,2]);self.assertTrue(np.all(status[6:]==0))
        for row in (1,2,3,5):self.assertEqual(actual[row].tobytes(),np.full(52,42.125,dtype=np.float32).tobytes())
        graph,gs,_=self.probe(states,valid,'invalid-graph','graph');self.assertEqual(graph.tobytes(),actual.tobytes());np.testing.assert_array_equal(gs,status)
        off,codes,info=self.probe(states,valid,'off','off');self.assertEqual(off.tobytes(),np.full((8,52),42.125,dtype=np.float32).tobytes())
        self.assertTrue(np.all(codes==0x12345678));self.assertEqual(info['device_weights_scratch_bytes'],0)


if __name__=='__main__':unittest.main()
