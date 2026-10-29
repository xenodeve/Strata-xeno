"""The learned readout receives frozen activations, not cache ordinals or text."""
import unittest
import numpy as np
import torch
from tools.igc.embedding_readout import batch_inputs,TensorInputs,training_lr


class EmbeddingInputs(unittest.TestCase):
    def test_cosine_decay_is_global_bounded_and_reaches_its_declared_floor(self):
        self.assertEqual(training_lr('constant',999,11200),0.001)
        self.assertEqual(training_lr('cosine',0,11200),0.001)
        self.assertAlmostEqual(training_lr('cosine',11199,11200),0.00001)
        values=[training_lr('cosine',step,11200) for step in range(11200)]
        self.assertTrue(all(.00001<=value<=.001 for value in values))
        self.assertTrue(all(a>=b for a,b in zip(values,values[1:])))
        self.assertLess(values[160],values[159])
        for step,total in [(-1,11200),(11200,11200),(0,1)]:
            with self.assertRaises(ValueError):training_lr('cosine',step,total)

    def test_tensor_cache_preserves_the_checked_cpu_batch_contract(self):
        table=np.stack([np.full((4,2560),v,dtype=np.float32) for v in (1.25,2.5)])
        indices=np.tile(np.arange(80,dtype=np.int32)%2,(2,1))
        valid=np.ones((2,80),dtype=np.bool_);valid[:,0]=False
        selection=np.array([1,0])
        expected=batch_inputs(table,indices,valid,selection)
        actual=TensorInputs(table,indices,valid,'cpu').batch(selection)
        for lhs,rhs in zip(expected,actual,strict=True):torch.testing.assert_close(lhs,rhs,rtol=0,atol=0)

    def test_cache_row_numbering_is_not_a_network_feature(self):
        table=np.stack([np.full((4,2560),v,dtype=np.float32) for v in (1.25,2.5)])
        indices=np.tile(np.arange(80,dtype=np.int32)%2,(2,1))
        valid=np.ones((2,80),dtype=np.bool_);valid[:,0]=False
        a=batch_inputs(table,indices,valid,np.array([1,0]))
        b=batch_inputs(table[::-1].copy(),1-indices,valid,np.array([1,0]))
        self.assertEqual(a[0].shape,(2,80,4,2560))
        self.assertEqual(a[0].dtype,torch.float32)
        for lhs,rhs in zip(a,b,strict=True):torch.testing.assert_close(lhs,rhs,rtol=0,atol=0)
        self.assertTrue(torch.equal(a[3],torch.zeros((2,2560))))

    def test_out_of_range_or_float_indices_are_refused_before_gather(self):
        table=np.ones((2,4,2560),dtype=np.float32)
        indices=np.zeros((1,80),dtype=np.int32);valid=np.ones((1,80),dtype=np.bool_)
        for invalid in (indices-1,indices+2,indices.astype(np.float32)):
            with self.assertRaises(ValueError):batch_inputs(table,invalid,valid,np.array([0]))


if __name__=='__main__':unittest.main()
