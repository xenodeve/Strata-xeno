"""Versioned observation seams, synthetic states only; no corpus builders."""
import unittest
import hashlib
import copy
import struct
from pathlib import Path
import tempfile
import threading
import numpy as np
from tools.igc.coherent_observation import encode_plan,decode_plan,decode_capture,CompactArchive,load_compact,load_input_observation,first_four_fidelity
from tools.igc.coherent_prepare import compile_plan,inspected_keys
from tools.igc.coherent_collect import admission,guarded_monitor


class CoherentObservation(unittest.TestCase):
    def plan(self):
        return dict(model='11'*32,tokenizer='22'*32,records=[
            dict(role='anchor',base=0,step=0,tokens=[1,2,3]),
            dict(role='output',base=0,step=1,tokens=[1,2,3,7])])

    def test_plan_contains_only_consumed_inputs_and_binds_content(self):
        p=self.plan();raw=encode_plan(p);r=decode_plan(raw)
        self.assertEqual(r['records'],p['records']);self.assertEqual(r['model'],p['model'])
        bad=bytearray(raw);bad[-1]^=1
        with self.assertRaises(ValueError):decode_plan(bytes(bad))
        p['records'][1]['tokens']=[1,2,8,7]
        with self.assertRaises(ValueError):encode_plan(p)
        p=self.plan();p['records'][1]['target']=9
        with self.assertRaises(ValueError):encode_plan(p)

    def test_digest_rejects_whitespace_and_non_hex_before_writing(self):
        for value in (' '*64,'11 '*21+'1','g'*64,'A'*64,'11'*31,'11'*33):
            p=self.plan();p['model']=value
            with self.assertRaises(ValueError,msg=repr(value)):encode_plan(p)

    def test_teacher_forced_target_never_enters_current_producer_prefix(self):
        class Tokenizer:
            def encode(self,s):return [ord(c) for c in s]
        rows=[dict(id='a',prompt='add',completion='42'),dict(id='b',prompt='copy')]
        manifest,raw=compile_plan(rows,['a'],Tokenizer(),'<','>','11'*32,'22'*32)
        p=decode_plan(raw)
        self.assertEqual(p['records'][0]['tokens'],[60,97,100,100,62])
        self.assertEqual(p['records'][2]['tokens'],[60,97,100,100,62,52])
        self.assertEqual(manifest['requests'][2]['target'],50)
        self.assertNotIn(50,p['records'][2]['tokens'])
        self.assertNotIn('target',p['records'][2])

    def test_inspected_ordinary_negative_has_prompt_exclusion_not_invented_operands(self):
        self.assertEqual(inspected_keys(dict(prompt='  Copy 567.  ',ready=0)),dict(prompt='copy 567.'))
        for row in [dict(prompt='x',ready=1),dict(prompt='x',ready=0,lhs=1)]:
            with self.assertRaises(ValueError):inspected_keys(row)

    def test_actual_hub_admission_and_disclosed_staging_budget_are_separate(self):
        system=dict(physical_available_bytes=24*1024**3,system_commit_available_bytes=51_128_263_976)
        self.assertTrue(admission(system,6*1024**3,[]))
        with self.assertRaisesRegex(RuntimeError,'physical'):admission({**system,'physical_available_bytes':23*1024**3},6*1024**3,[])
        with self.assertRaisesRegex(RuntimeError,'commit'):admission({**system,'system_commit_available_bytes':40*1024**3},6*1024**3,[])
        with self.assertRaisesRegex(RuntimeError,'engine'):admission(system,6*1024**3,['strata.exe'])

    def test_monitor_exception_cancels_instead_of_becoming_silent_success(self):
        cancelled=threading.Event();failures=[]
        def failed_sample():raise OSError('GetPerformanceInfo fixture failure')
        guarded_monitor(failed_sample,cancelled,failures)
        self.assertTrue(cancelled.is_set());self.assertEqual(len(failures),1)
        self.assertEqual(failures[0]['reason'],'telemetry monitor exception')
        cancelled.clear();failures.clear();stop=threading.Event()
        guarded_monitor(lambda:None,cancelled,failures,stop)
        self.assertTrue(cancelled.is_set());self.assertEqual(failures[0]['reason'],'telemetry monitor stopped unexpectedly')
        cancelled.clear();failures.clear();stop.set();guarded_monitor(lambda:None,cancelled,failures,stop)
        self.assertFalse(cancelled.is_set());self.assertEqual(failures,[])

    def capture(self):
        p=self.plan();plan=encode_plan(p);tokens=p['records'][1]['tokens']
        header=struct.pack('<8s7IQ',b'IGCCOH03',3,1,1,0,1,4,0,91)
        raw=header+bytes.fromhex(p['model'])+bytes.fromhex(p['tokenizer'])+plan[-32:]
        raw+=struct.pack('<4Q6I',0,4,90,91,0,0,1,31,0,0)
        raw+=struct.pack('<4I',*tokens)+bytes(80)
        raw+=np.ones(2560,dtype='<f4').tobytes()+np.zeros(248320,dtype='<f4').tobytes()
        return raw+hashlib.sha256(raw).digest(),p

    def test_capture_binds_same_owned_consumed_prefix_and_has_no_future_label(self):
        raw,p=self.capture();r=decode_capture(raw,encode_plan(p),1)
        self.assertEqual((r['owner'],r['tokens'],r['role']),(91,[1,2,3,7],'output'))
        self.assertEqual(r['mixed'].shape,(2560,));self.assertNotIn('target',r)
        self.assertEqual(r['provenance']['accepted'],4)
        with self.assertRaises(ValueError):decode_capture(raw,encode_plan(p),0)
        bad=bytearray(raw);bad[44]^=1
        bad[-32:]=hashlib.sha256(bad[:-32]).digest()
        with self.assertRaises(ValueError):decode_capture(bytes(bad),encode_plan(p),1)
        bad=bytearray(raw);struct.pack_into('<Q',bad,148,0);bad[-32:]=hashlib.sha256(bad[:-32]).digest()
        with self.assertRaises(ValueError):decode_capture(bytes(bad),encode_plan(p),1)

    def test_first_four_continuation_requires_native_fidelity_and_reference_bytes(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);receipts=[]
            for i in range(4):
                r=dict(ordinal=i,role='anchor',owner=i+1,provenance=dict(source=1,head_fidelity=1,flags=31,fence=i+1),files={},head={})
                for suffix,n in [('mixed',2560*4),('logits',248320*4),('attention',80*4*2560*4)]:
                    data=bytes(n);name=f'{i}.{suffix}.bin';(root/name).write_bytes(data)
                    r['files'][name]=dict(bytes=n,sha256=hashlib.sha256(data).hexdigest())
                r['head']['logits_sha256']=r['files'][f'{i}.logits.bin']['sha256'];receipts.append(r)
            self.assertTrue(first_four_fidelity(receipts,root))
            for field in ('source','head_fidelity','flags'):
                bad=copy.deepcopy(receipts);bad[3]['provenance'][field]=0
                with self.assertRaises(ValueError):first_four_fidelity(bad,root)
            (root/'3.logits.bin').write_bytes(b'incomplete')
            with self.assertRaises(ValueError):first_four_fidelity(receipts,root)

    def test_compact_archive_refuses_partial_resume_and_preserves_exact_values(self):
        raw,p=self.capture();plan=encode_plan(p)
        # A first original anchor has complete literal embedding/attention tails.
        h=struct.pack('<8s7IQ',b'IGCCOH03',3,0,0,0,0,3,2,90)
        anchor=h+bytes.fromhex(p['model'])+bytes.fromhex(p['tokenizer'])+plan[-32:]
        anchor+=struct.pack('<4Q6I',0,3,0,90,0,0,1,31,0,0)
        anchor+=struct.pack('<3I',1,2,3)+bytes(78)+b'\1\1'
        states=np.zeros((80,4,2560),dtype='<f4');states[-2]=2;states[-1]=4
        anchor+=states.tobytes()+states.tobytes()+np.ones(2560,dtype='<f4').tobytes()+np.zeros(248320,dtype='<f4').tobytes()
        anchor+=hashlib.sha256(anchor).digest()
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)/'archive';a=CompactArchive(root,plan,{'engine':'33'*32,'profile':'44'*32})
            with self.assertRaises(ValueError):load_compact(root,0)
            a.ingest(anchor,0,target=7)
            with self.assertRaises(ValueError):load_compact(root,0)
            with self.assertRaises(ValueError):CompactArchive(root,plan,{'engine':'33'*32,'profile':'44'*32})
            a.ingest(raw,1,target=8);a.finalize()
            r=load_compact(root,0)
            self.assertTrue(np.array_equal(r['attention'],states));self.assertTrue(np.array_equal(r['embedding'],states))
            self.assertEqual((r['owner'],r['head']['argmax'],r['head']['target']),(90,0,7))
            self.assertEqual(load_compact(root,1)['tokens'],[1,2,3,7])
            inputs=load_input_observation(root,0)
            self.assertNotIn('head',inputs);self.assertNotIn('target',inputs)
            with self.assertRaises(ValueError):load_input_observation(root,1)
            receipt=root/'0.json';receipt.write_bytes(receipt.read_bytes()+b' ')
            with self.assertRaises(ValueError):load_compact(root,0)


if __name__=='__main__':unittest.main()
