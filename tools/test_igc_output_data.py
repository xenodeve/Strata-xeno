import unittest
import numpy as np
from tools.igc.output_data import validate_record


class OutputData(unittest.TestCase):
    def setUp(self):
        self.example=dict(sample='a',split='train',invocation=7,invocation_position=2,
                          position=3,output_offset=1,target=49,numeric_target=True,ids=[1,2,3,4])
        self.observed=dict(prefix_tokens=np.array([1,2,3,4]),position=3,prompt_tokens=4,
                           prefix_sha256='a',mixed_sha256='b',logits_sha256='c',metadata_sha256='d')
        self.record=dict(index=0,**self.example,observation={k:v for k,v in self.observed.items() if k!='prefix_tokens'},
                         engine_done=dict(generated=1,prompt_tokens=4),unadapted_next_token=[20])

    def test_exact_teacher_forced_boundary_not_sampled_prediction(self):
        validate_record(self.record,self.example,self.observed,0)

    def test_wrong_future_target_owner_prefix_or_incomplete_request_is_refused(self):
        for key,value in [('target',50),('invocation_position',3),('position',2),('ids',[1,2,3,49])]:
            with self.assertRaises(ValueError):validate_record({**self.record,key:value},self.example,self.observed,0)
        with self.assertRaises(ValueError):
            validate_record({**self.record,'engine_done':dict(generated=0,prompt_tokens=4)},self.example,self.observed,0)
        with self.assertRaises(ValueError):
            validate_record({**self.record,'observation':{**self.record['observation'],'mixed_sha256':'x'}},self.example,self.observed,0)


if __name__=='__main__':unittest.main()
