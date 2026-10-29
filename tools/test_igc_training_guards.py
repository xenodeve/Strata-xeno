"""Numerical refusal must occur before updates or artifact promotion."""
import unittest
import torch
import numpy as np
from pathlib import Path
from unittest.mock import patch
from tools.igc.training_guards import checked_update, checked_backward, categorical_loss, admission, progress, state_fingerprint, verify_reset
from tools.igc.readout import SIZES


class TrainingGuards(unittest.TestCase):
    def model(self):
        model=torch.nn.Linear(1,1,bias=False)
        with torch.no_grad():model.weight.fill_(1)
        return model

    def test_finite_update_matches_unguarded_optimizer_without_clipping(self):
        a=self.model();b=self.model();x=torch.ones((1,1))
        oa=torch.optim.SGD(a.parameters(),lr=.1);ob=torch.optim.SGD(b.parameters(),lr=.1)
        checked_update(a,oa,a(x).square().sum())
        b(x).square().sum().backward();ob.step()
        self.assertTrue(torch.equal(a.weight,b.weight))

    def test_nonfinite_loss_never_steps(self):
        m=self.model();o=torch.optim.AdamW(m.parameters());before=m.weight.detach().clone()
        with self.assertRaisesRegex(RuntimeError,'loss'):
            checked_update(m,o,m(torch.ones((1,1))).sum()*float('nan'))
        self.assertTrue(torch.equal(m.weight,before));self.assertEqual(len(o.state),0)

    def test_finite_loss_but_nonfinite_gradient_never_steps(self):
        m=self.model();o=torch.optim.AdamW(m.parameters());before=m.weight.detach().clone()
        hook=m.weight.register_hook(lambda g:torch.full_like(g,float('nan')))
        with self.assertRaisesRegex(RuntimeError,'gradient'):
            checked_update(m,o,m(torch.ones((1,1))).square().sum())
        hook.remove();self.assertTrue(torch.equal(m.weight,before));self.assertEqual(len(o.state),0)

    def test_post_update_nonfinite_parameters_are_rejected(self):
        m=self.model();o=torch.optim.SGD(m.parameters(),lr=float('inf'))
        with self.assertRaisesRegex(RuntimeError,'parameter'):
            checked_update(m,o,m(torch.ones((1,1))).square().sum())

    def test_pricing_backward_does_not_change_weights(self):
        m=self.model();before=m.weight.detach().clone()
        checked_backward(m,m(torch.ones((1,1))).square().sum())
        self.assertIsNotNone(m.weight.grad);self.assertTrue(torch.equal(m.weight,before))

    def test_categorical_loss_keeps_positive_only_operand_mask_and_legacy_sum(self):
        torch.manual_seed(4);logits=torch.randn((2,sum(SIZES)),requires_grad=True)
        y=torch.tensor([[1,0,0,0,0,3,0,8,0],[0,1,1,1,1,9,9,9,9]])
        actual=categorical_loss(logits,y)
        self.assertIsInstance(actual,torch.Tensor)
        heads=logits.split(SIZES,dim=1);expected=torch.nn.functional.cross_entropy(heads[0],y[:,0])
        for i in range(1,9):expected=expected+torch.nn.functional.cross_entropy(heads[i][:1],y[:1,i])
        self.assertTrue(torch.equal(actual,expected))
        actual.backward();self.assertFalse(bool(logits.grad[1,4:].any()))

    def test_optimizer_moment_overflow_is_not_hidden_by_finite_parameters(self):
        m=self.model();o=torch.optim.AdamW(m.parameters(),lr=.001)
        with self.assertRaisesRegex(RuntimeError,'optimizer'):
            checked_update(m,o,m(torch.ones((1,1))).sum()*1e30)

    def test_admission_and_post_step_progress_fail_without_a_private_memory_cap(self):
        good=dict(physical_available_bytes=8*1024**3,system_commit_available_bytes=12*1024**3)
        admission(good,4*1024**3+16*1024**2)
        with self.assertRaises(RuntimeError):admission({**good,'physical_available_bytes':0},8*1024**3)
        with self.assertRaises(RuntimeError):progress(61,60,good)
        with self.assertRaises(RuntimeError):progress(1,60,{**good,'system_commit_available_bytes':0})

    def test_initial_fingerprint_can_prove_reset_across_pricing_and_fresh_fit(self):
        a=self.model();b=self.model()
        self.assertIsInstance(state_fingerprint(a),str)
        self.assertEqual(state_fingerprint(a),state_fingerprint(b))
        with torch.no_grad():b.weight.add_(1)
        self.assertNotEqual(state_fingerprint(a),state_fingerprint(b))

    def test_final_matched_metrics_keep_raw_arrays_and_padding_variant_separate(self):
        from tools.igc.embedding_readout import matched_analysis
        from tools.test_igc_matched_evaluate import row
        rows=[row('a',3,8),row('b',3,8,intent='quote-copy'),
              row('c',3,8,family='new'),row('d',3,8,intent='quote-copy',family='new')]
        raw=np.array([[1,0,0,0,0,3,3,8,0],[0,0,0,0,0,0,0,0,0],
                      [1,0,0,0,0,4,0,8,0],[1,0,0,0,0,3,0,8,0]])
        before=raw.copy();result=matched_analysis(rows,raw)
        self.assertEqual(result.get('positive_recovered'),1)
        self.assertEqual(result['raw']['development']['counts']['positive_exact_tuple'],0)
        self.assertEqual(result['padding_variant']['development']['counts']['positive_exact_tuple'],1)
        self.assertTrue(np.array_equal(raw,before))

    def test_price_cli_forwards_no_optimizer_mode_and_reset_receipt(self):
        from tools.igc.embedding_readout import main
        calls=[]
        # Isolate only the expensive train boundary; exercise the real CLI parser.
        with patch('tools.igc.embedding_readout.train',side_effect=lambda *a,**k:calls.append((a,k))):
            main(['--dataset','cache','--sources','sources','--protocol','protocol','--output','new',
                  '--price-only','--pricing-receipt','priced.json'])
        self.assertEqual(calls[0][1].get('price_only'),True)
        self.assertEqual(calls[0][1].get('pricing_receipt'),Path('priced.json'))

    def test_reset_binding_includes_mask_even_when_states_are_identical(self):
        cache=dict(state='same',mask='one',report='frozen',manifest='paired')
        priced=dict(success=True,optimizer_steps=0,backward_batches=1,initial_state_sha256='init',
            initial_batch_ids=['train'],source_sha256={'code':'one'},cache_identity=cache,protocol_sha256='frozen',recipe={'seed':7})
        verify_reset(priced,'init',['train'],{'code':'one'},cache,'frozen',{'seed':7})
        with self.assertRaises(ValueError):
            verify_reset(priced,'init',['train'],{'code':'one'},{**cache,'mask':'changed'},'frozen',{'seed':7})
        with self.assertRaises(ValueError):
            verify_reset(priced,'different',['train'],{'code':'one'},cache,'frozen',{'seed':7})
        with self.assertRaises(ValueError):
            verify_reset(priced,'init',['train'],{'code':'one'},cache,'frozen',{'seed':8})


if __name__=='__main__':unittest.main()
