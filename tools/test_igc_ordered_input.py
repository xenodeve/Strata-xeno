import unittest
import numpy as np
import torch
from tools.igc.attention_readout import InputReadout
from tools.igc.ordered_input import targets,canonical_fields,SIZES,OrderedInputReadout


class OrderedInput(unittest.TestCase):
    def test_shared_initialization_is_unchanged_by_two_new_pad_logits(self):
        torch.manual_seed(20261031);baseline=InputReadout(False,True).state_dict()
        torch.manual_seed(20261031);candidate=OrderedInputReadout().state_dict()
        for name,value in baseline.items():
            observed=candidate[name]
            if name.startswith(('heads.6.','heads.8.')):observed=observed[:10]
            torch.testing.assert_close(observed,value,rtol=0,atol=0)

    def test_left_aligned_neural_view_is_distinct_from_exact_executor_abi(self):
        rows=[dict(lhs=3,rhs=-23,ready_op=3),dict(lhs=0,rhs=10,ready_op=1)]
        actual=targets(rows)
        self.assertEqual(actual.tolist(),[[3,0,1,0,1,3,10,2,3],[1,0,0,0,1,0,10,1,0]])
        self.assertEqual(canonical_fields(actual).tolist(),[[3,0,1,0,1,3,0,3,2],[1,0,0,0,1,0,0,0,1]])

    def test_padding_or_leading_zero_is_rejected_without_repair(self):
        good=targets([dict(lhs=3,rhs=-23,ready_op=3)])
        for field,value in [(6,0),(7,0),(4,0)]:
            bad=good.copy();bad[0,field]=value
            self.assertIsNone(canonical_fields(bad)[0])
        bad=good.copy();bad[0,1]=1;bad[0,5]=0
        self.assertIsNone(canonical_fields(bad)[0])
        self.assertEqual(SIZES,[4,2,2,2,2,10,11,10,11])


if __name__=='__main__':unittest.main()
