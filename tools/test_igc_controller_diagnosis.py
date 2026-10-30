"""Synthetic row-identity and gate-decomposition checks, no corpus builders."""
import copy
import struct
import unittest
import numpy as np
from tools.igc.native_qualification import selection,expected_records
from tools.igc.controller_diagnosis import diagnose


class ControllerDiagnosis(unittest.TestCase):
    def fixture(self):
        rows=[];refs={'retained':[],'candidate':[]};payload=b'IGCLPO01'+struct.pack('<I',5)
        for i,(ready,bad,op) in enumerate([(1,False,1),(0,False,1),(1,True,1),(0,True,1),(0,False,0)]):
            row=dict(id=f'case-{i}',lhs=2,rhs=3,ready_op=op,split='development',template='en-test',prompt='test')
            fields=[op,0,0,0,0,2,0,3,0];primary=fields.copy()
            if bad:primary[5]=4
            choice,request=selection(primary,fields,ready)
            r=dict(id=row['id'],index=i,owner=i+1,anchor=20,primary=primary,secondary=fields,
                   readiness=ready,selection=choice,request=request)
            for rr in refs.values():rr.append(copy.deepcopy(r))
            q,v=expected_records(i+1,20,choice,request)
            payload+=struct.pack('<QQ6I',i+1,20,0,1,int(choice==0),1 if choice==0 else 2 if choice==1 else 3,choice,0)
            payload+=q+v+struct.pack('<I',0 if choice==0 else 1)+bytes(2560*4)
            rows.append(row)
        scores=np.asarray([[0,2] if r['readiness'] else [2,0] for r in refs['retained']],dtype=np.float32)
        return rows,refs,payload,scores

    def test_exclusive_gate_partition_and_ceiling(self):
        a,b,c,d=self.fixture();r=diagnose(a,b,c,c,d)
        s=r['summaries']['candidate']
        self.assertEqual(s['positive_partition'],dict(selected_correct=1,readiness_only=1,embedding_only=1,both=1,attention_error=0,other=0))
        self.assertEqual(s['positive_oracle_counterfactual'],dict(readiness_only=2,embedding_only=2,both=4))
        self.assertEqual(s['negative_calls'],0);self.assertEqual(r['changed_rows']['selection'],[])
        self.assertEqual(r['rows'][0]['candidate']['native_result'],5)
        self.assertEqual(r['rows'][1]['candidate']['native_result'],None)
        self.assertGreater(r['rows'][0]['readiness']['margin'],0)

    def test_inactive_operand_drift_is_not_an_executed_change(self):
        a,b,c,d=self.fixture();b['candidate'][-1]['secondary'][5]=9
        r=diagnose(a,b,c,c,d)
        self.assertEqual(r['changed_rows']['attention_tuple'],['case-4'])
        self.assertEqual(r['changed_rows']['selection'],[]);self.assertEqual(r['changed_rows']['native_result'],[])

    def test_row_order_scores_and_native_owner_are_bound(self):
        for mutation in ('order','score','owner','count','duplicate'):
            a,b,c,d=self.fixture()
            if mutation=='order':b['candidate'][0],b['candidate'][1]=b['candidate'][1],b['candidate'][0]
            if mutation=='score':d[0]=[2,0]
            if mutation=='owner':c=bytearray(c);c[12]^=1;c=bytes(c)
            if mutation=='count':c=c[:-1]
            if mutation=='duplicate':a[-1]['id']=a[0]['id']
            with self.assertRaises(ValueError,msg=mutation):diagnose(a,b,c,c,d)

    def test_nonfinite_readiness_is_not_a_false_score(self):
        a,b,c,d=self.fixture();d[1,0]=np.nan
        with self.assertRaises(ValueError):diagnose(a,b,c,c,d)


if __name__=='__main__':unittest.main()
