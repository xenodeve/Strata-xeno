"""Learned-input-head mask and gradient checks, not frozen-Gyro gradients."""
import unittest
import json
from pathlib import Path
import tempfile
from unittest.mock import patch
import numpy as np
import torch
from tools.igc.attention_readout import InputReadout, contexts
from tools.igc.readout import SIZES


class AttentionContract(unittest.TestCase):
    def test_context_stage_is_bound_to_its_own_record_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            metadata=dict(sha256='embedding-sha',positions=[3],sources=[1],missing=[0,1,2])
            (root/'records.json').write_text(json.dumps([dict(observation=dict(preanchor_stages=dict(embedding=metadata)))]))
            observation=dict(preanchor_sha256='embedding-sha',positions=[3],sources=[1],missing_positions=[0,1,2],
                             anchor=dict(position=4),residuals=np.full((1,4,2560),2.5,dtype=np.float32))
            with patch('tools.igc.attention_readout.load_preanchor',return_value=observation) as read:
                values,valid,positions=contexts(root,[{}],stage='embedding')
                read.assert_called_once_with(root,0,stage='embedding')
                self.assertEqual(int(valid.sum()),1)
                self.assertTrue(valid[0,79])
                self.assertEqual(float(values[0,79,0,0]),2.5)
                observation['preanchor_sha256']='wrong-stage-sha'
                with self.assertRaises(ValueError):contexts(root,[{}],stage='embedding')

    def inputs(self):
        torch.set_num_threads(1)
        torch.manual_seed(7)
        states=torch.randn(2,4,4,2560)
        valid=torch.tensor([[True,True,False,False],[True,False,True,False]])
        positions=torch.tensor([[76,77,78,79],[76,77,78,79]])
        query=torch.randn(2,2560)
        return states,valid,positions,query

    def test_only_observed_states_affect_output(self):
        states,valid,positions,query=self.inputs()
        model=InputReadout().eval()
        a=model(states,valid,positions,query)
        changed=states.clone();changed[~valid]=100000
        b=model(changed,valid,positions,query)
        self.assertEqual(a.shape,(2,sum(SIZES)))
        torch.testing.assert_close(a,b,rtol=0,atol=0)

    def test_no_states_or_invalid_position_is_refused(self):
        states,valid,positions,query=self.inputs()
        model=InputReadout()
        with self.assertRaises(ValueError):model(states,torch.zeros_like(valid),positions,query)
        positions[0,0]=80
        with self.assertRaises(ValueError):model(states,valid,positions,query)

    def test_head_receives_gradients_without_a_frozen_model_graph(self):
        states,valid,positions,query=self.inputs()
        states.requires_grad_(True)
        model=InputReadout()
        logits=model(states,valid,positions,query)
        loss=logits.square().mean()
        loss.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        self.assertGreater(float(states.grad[valid].abs().sum()),0)
        self.assertEqual(float(states.grad[~valid].abs().sum()),0)


if __name__=='__main__':
    unittest.main()
