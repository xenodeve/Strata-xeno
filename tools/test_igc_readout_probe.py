"""Attention instrumentation must reproduce the unmodified head before controls."""
import unittest
import torch
from tools.igc.attention_readout import InputReadout
from tools.igc.readout_probe import forward_parts,repool


class ReadoutProbe(unittest.TestCase):
    def fixture(self):
        torch.set_num_threads(2);torch.manual_seed(19)
        model=InputReadout(use_final_query=False,contextual=True).eval().requires_grad_(False)
        states=torch.randn(3,80,4,2560);valid=torch.ones(3,80,dtype=torch.bool)
        valid[0,:5]=False;valid[1,:9]=False;valid[2,:3]=False
        return model,(states,valid,torch.arange(80).expand(3,-1),torch.zeros(3,2560))

    def test_decomposition_matches_real_forward_and_preserves_parameters(self):
        m,args=self.fixture();before={k:v.clone() for k,v in m.state_dict().items()}
        with torch.no_grad():
            expected=m(*args);parts=forward_parts(m,*args)
        self.assertIsInstance(parts,dict)
        self.assertTrue(torch.equal(parts['logits'],expected))
        self.assertTrue(all(torch.equal(before[k],v) for k,v in m.state_dict().items()))
        self.assertFalse(bool(parts['attention'][0,:,:5].any()))

    def test_original_attention_repool_is_bit_exact_and_empty_distribution_refused(self):
        m,args=self.fixture()
        with torch.no_grad():
            p=forward_parts(m,*args)
            self.assertIsInstance(p,dict)
            self.assertTrue(torch.equal(repool(m,p['values'],p['queries'],p['attention']),p['logits']))
            with self.assertRaises(ValueError):repool(m,p['values'],p['queries'],torch.zeros_like(p['attention']))

    def test_train_flag_only_forward_with_zero_dropout_does_not_explain_mode_gap(self):
        m,args=self.fixture();prior=torch.backends.mha.get_fastpath_enabled()
        try:
            torch.backends.mha.set_fastpath_enabled(False)
            with torch.no_grad():
                a=m.eval()(*args);b=m.train()(*args)
            self.assertTrue(torch.equal(a,b));self.assertTrue(all(p.grad is None for p in m.parameters()))
        finally:torch.backends.mha.set_fastpath_enabled(prior)


if __name__=='__main__':unittest.main()
