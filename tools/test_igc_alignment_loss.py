import unittest
import torch
from tools.igc.aligned_input_fit import pointer_loss


class AlignmentLoss(unittest.TestCase):
    def test_only_positive_requests_and_observed_positions_are_supervised(self):
        scores=torch.zeros(2,9,80,requires_grad=True)
        valid=torch.ones(2,80,dtype=torch.bool);target=torch.tensor([[1,2,3,79],[4,5,6,79]])
        selected=torch.tensor([True,False]);loss=pointer_loss(scores,valid,target,selected);loss.backward()
        self.assertEqual(float(scores.grad[1].abs().sum()),0.)
        self.assertGreater(float(scores.grad[0].abs().sum()),0.)
        valid[0,79]=False
        with self.assertRaises(ValueError):pointer_loss(scores.detach(),valid,target,selected)


if __name__=='__main__':unittest.main()
