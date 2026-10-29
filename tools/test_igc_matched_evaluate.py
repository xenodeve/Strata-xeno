"""Frozen readout evaluation must keep routing, encoding and cluster units distinct."""
import tempfile
import subprocess
import sys
import os
import json
from unittest.mock import patch
from pathlib import Path
import unittest
import numpy as np
import torch
from tools.igc.matched_evaluate import score, cluster_interval, matched_contrasts, infer, evaluate, validate_cuda_environment, frozen_cache_report, CACHE_SHA


def row(name, a, b, op=1, intent='compute', pair='p', family='seen'):
    return dict(id=name, lhs=a, rhs=b, ready_op=op if intent=='compute' else 0,
                operation=op, intent=intent, pair_group=pair, group=pair+str(op),
                split='development', family=family, language='en')


class MatchedEvaluate(unittest.TestCase):
    def test_raw_scoring_keeps_padding_active_errors_and_no_calls_separate(self):
        rows=[row('a',3,8),row('b',12,-4),row('c',3,8,intent='quote-copy')]
        raw=np.array([[1,0,0,0,0,3,3,8,0], [0,0,1,1,0,2,1,5,0],
                      [2,0,0,0,0,3,0,8,0]])
        result=score(rows,raw)
        self.assertEqual(result.get('counts'),dict(rows=3,positives=2,negatives=1,
            positive_exact_tuple=0,positive_operands_exact=0,positive_no_call=1,
            positive_wrong_nonzero_operator=0,negative_false_call=1,negative_no_call=0,
            positive_operator_correct=1,signs_correct=4,signs_total=4,
            lengths_correct=4,lengths_total=4,active_digits_correct=4,active_digits_total=5,
            padding_correct=2,padding_total=3,padding_only_operands=1,
            active_only_operands=1,both_digit_error_operands=0,other_error_operands=0))
        self.assertEqual(raw[0,6],3)

    def test_cluster_interval_averages_pairs_not_rows_or_nested_operation_groups(self):
        rows=[row('a',1,2,pair='one'),row('b',1,2,op=2,pair='one'),row('c',4,5,pair='two')]
        values=[1.,1.,0.]
        result=cluster_interval(rows,values)
        self.assertEqual(result.get('clusters'),2)
        self.assertEqual(result.get('estimate'),.5)
        self.assertEqual(result,cluster_interval(rows+rows,values+values))

    def test_matched_family_and_intent_comparisons_hold_expression_and_language_fixed(self):
        rows=[row('a',3,8),row('b',3,8,intent='quote-copy'),
              row('c',3,8,family='new'),row('d',3,8,intent='quote-copy',family='new')]
        raw=np.array([[1,0,0,0,0,3,0,8,0],[0,0,0,0,0,0,0,0,0],
                      [1,0,0,0,0,4,0,8,0],[1,0,0,0,0,3,0,8,0]])
        result=matched_contrasts(rows,raw)
        self.assertEqual(result.get('family_positive_transitions'),dict(both_correct=0,
            seen_only_correct=1,new_only_correct=0,both_wrong=0,pairs=1))
        self.assertEqual(result.get('intent_joint'),dict(both_correct=1,
            compute_only_correct=0,quote_only_correct=0,both_wrong=1,pairs=2))

    def test_invalid_prediction_classes_and_duplicate_ids_fail_closed(self):
        rows=[row('a',3,8)];bad=np.array([[1,0,0,2,0,3,0,8,0]])
        with self.assertRaises(ValueError):score(rows,bad)
        with self.assertRaises(ValueError):score(rows+rows,np.zeros((2,9),dtype=np.int64))

    def test_infer_disables_gradients_without_updating_parameters_and_bounds_batches(self):
        from tools.igc.attention_readout import InputReadout
        from tools.igc.context_cache import ContextInputs
        torch.set_num_threads(2);torch.manual_seed(7)
        model=InputReadout(use_final_query=False,contextual=True)
        cache=ContextInputs(np.ones((3,80,4,2560),dtype=np.float32),np.ones((3,80),dtype=np.bool_),'cpu')
        before={k:v.detach().clone() for k,v in model.state_dict().items()};modes=[]
        hook=model.register_forward_pre_hook(lambda *_:modes.append(torch.is_grad_enabled()))
        result=infer(model,cache,3,batch_size=2,deadline_seconds=10)
        hook.remove()
        self.assertEqual(getattr(result,'shape',None),(3,9))
        self.assertEqual(modes,[False,False]);self.assertFalse(model.training)
        self.assertTrue(all(torch.equal(before[k],v) for k,v in model.state_dict().items()))
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        with self.assertRaises(ValueError):infer(model,cache,3,batch_size=65,deadline_seconds=10)

    def test_wrong_retained_state_refused_before_cache_or_output_access(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);state=root/'bad.safetensors';state.write_bytes(b'wrong')
            with self.assertRaisesRegex(ValueError,'retained state identity'):
                evaluate(root/'missing-cache',state,root/'missing-sources',root/'missing-protocol',root/'out')
            self.assertFalse((root/'out').exists())

    def test_last_batch_or_observer_crossing_deadline_cannot_report_success(self):
        from tools.igc.attention_readout import InputReadout
        from tools.igc.context_cache import ContextInputs
        model=InputReadout(use_final_query=False,contextual=True)
        cache=ContextInputs(np.ones((1,80,4,2560),dtype=np.float32),np.ones((1,80),dtype=np.bool_),'cpu')
        now=[0.]
        with patch('tools.igc.matched_evaluate.time.monotonic',lambda:now[0]):
            with self.assertRaises(TimeoutError):
                infer(model,cache,1,deadline_seconds=10,observer=lambda _:now.__setitem__(0,11.))

    def test_frozen_gpu_environment_is_checked_without_cuda_access(self):
        valid={'CUDA_VISIBLE_DEVICES':'1,0','CUBLAS_WORKSPACE_CONFIG':':4096:8'}
        for extra in [{'CUDA_VISIBLE_DEVICES':'0,1'},{'CUDA_DEVICE_ORDER':''},{'CUBLAS_WORKSPACE_CONFIG':':16:8'}]:
            with patch.dict(os.environ,{**valid,**extra},clear=True):
                with self.assertRaises(ValueError):validate_cuda_environment()
        with patch.dict(os.environ,valid,clear=True):validate_cuda_environment()

    def test_changed_mask_with_matching_untrusted_report_is_not_frozen_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'report.json'
            path.write_text(json.dumps(dict(success=True,states_sha256=CACHE_SHA,valid_sha256='0'*64)))
            with self.assertRaisesRegex(ValueError,'frozen cache report identity'):frozen_cache_report(path)

    def test_cli_rejects_wrong_state_without_creating_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);state=root/'bad';state.write_bytes(b'wrong')
            args=[sys.executable,'-B','-m','tools.igc.matched_evaluate','--cache',str(root/'missing'),
                  '--state',str(state),'--sources',str(root/'sources'),'--protocol',str(root/'protocol'),
                  '--cache-protocol',str(root/'cache-protocol'),'--output',str(root/'out')]
            result=subprocess.run(args,capture_output=True,text=True,timeout=30,creationflags=0x08000000)
            self.assertNotEqual(result.returncode,0)
            self.assertIn('retained state identity',result.stderr)
            self.assertFalse((root/'out').exists())


if __name__=='__main__':unittest.main()
