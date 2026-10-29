"""Typed ownership and hard gating before any learned numeric result-use claim."""
import struct
import unittest
import os
import hashlib
from pathlib import Path
import numpy as np
import torch
from tools.igc.output_mapping import encode_results,ResultMapper


def result(value=b'\x01\x00'+bytes(62),status=0,invocation=7,position=123):
    return struct.pack('<IIQQ64s',1,status,invocation,position,value)


class OutputMapping(unittest.TestCase):
    def test_left_aligned_view_preserves_little_endian_abi_without_text(self):
        raw=result(bytes([3,1,3,2,1])+bytes(59)) # canonical -123
        batch=encode_results(raw,[7],[123])
        self.assertEqual(batch['features'].shape,(1,62,12))
        self.assertEqual(batch['features'][0,:3,:10].argmax(-1).tolist(),[1,2,3])
        self.assertTrue(bool((batch['features'][0,:3,11]==1).all()))
        self.assertEqual(int(batch['valid'].sum()),3);self.assertTrue(bool(batch['active'][0]))

    def test_malformed_and_unowned_results_are_not_repaired(self):
        bad=[result(bytes([1,1])+bytes(62)), # negative zero
             result(bytes([2,0,1,0])+bytes(60)), # leading zero
             result(bytes([1,0,10])+bytes(61)),
             result(bytes([1,0,1,2])+bytes(60)), # unused nonzero
             result(status=1),result(status=9),result(invocation=0),
             result(invocation=8),result(position=124),result()+b'x',result()[:-1]]
        for raw in bad:
            with self.assertRaises(ValueError):encode_results(raw,[7],[123])

    def test_failure_has_no_numeric_value_and_never_changes_activation(self):
        batch=encode_results(result(bytes(64),status=3),[7],[123])
        self.assertFalse(bool(batch['active'].any()));self.assertFalse(bool(batch['valid'].any()))
        self.assertEqual(batch['status'].tolist(),[3])
        mapper=ResultMapper();mixed=torch.randn(1,2560);mixed[0,0]=-0.
        actual=mapper(mixed,batch)
        self.assertEqual(actual.numpy().tobytes(),mixed.numpy().tobytes())

    def test_hard_off_does_not_run_modules_or_require_result_features(self):
        mapper=ResultMapper();mixed=torch.randn(1,2560);calls=[]
        hooks=[module.register_forward_hook(lambda *_:calls.append(1)) for module in mapper.children()]
        actual=mapper(mixed,None,enabled=False)
        self.assertIs(actual,mixed);self.assertEqual(calls,[])
        for hook in hooks:hook.remove()

    def test_active_residual_is_trainable_and_padding_has_no_effect(self):
        torch.manual_seed(71);mapper=ResultMapper()
        # Nonzero proposal prevents a zero-initialized branch hiding bad masks.
        torch.nn.init.normal_(mapper.proposal.weight,std=.01)
        batch=encode_results(result(bytes([2,0,2,4])+bytes(60)),[7],[123])
        mixed=torch.randn(1,2560);original=mixed.clone()
        first=mapper(mixed,batch);first.square().mean().backward()
        self.assertTrue(any(p.grad is not None and bool(p.grad.abs().sum()>0) for p in mapper.parameters()))
        self.assertGreater(float(mapper.query.weight.grad.norm()),0)
        altered={key:value.clone() for key,value in batch.items()}
        altered['features'][~altered['valid']]=999
        torch.testing.assert_close(mapper(mixed,altered),first,rtol=0,atol=0)
        torch.testing.assert_close(mixed,original,rtol=0,atol=0)

    def test_mixed_status_batch_hard_bypasses_only_failed_rows(self):
        torch.manual_seed(72);mapper=ResultMapper()
        torch.nn.init.normal_(mapper.proposal.weight,std=.01)
        batch=encode_results(result()+result(bytes(64),status=2,invocation=8),[7,8],[123,123])
        mixed=torch.randn(2,2560);mixed[1,0]=-0.
        actual=mapper(mixed,batch)
        self.assertNotEqual(actual[0].detach().numpy().tobytes(),mixed[0].numpy().tobytes())
        self.assertEqual(actual[1].detach().numpy().tobytes(),mixed[1].numpy().tobytes())
        batch['active'][1]=True;batch['valid'][1,0]=True
        with self.assertRaises(ValueError):mapper(mixed,batch)


@unittest.skipUnless(os.environ.get('IGC_HEAD_BRIDGE_DLL'),'native output training seam not selected')
class NativeOutputGradient(unittest.TestCase):
    def test_learned_result_residual_receives_native_head_lm_loss_gradient(self):
        from tools.igc.cuda_head import NativeCudaHead
        from tools.qwenputer.tokenizer import NativeText
        torch.manual_seed(73);torch.backends.cuda.matmul.allow_tf32=False
        device=torch.device('cuda:0')
        library=Path(os.environ['IGC_HEAD_BRIDGE_DLL'])
        mapper=ResultMapper().to(device)
        # Oracle result features qualify only this output training seam. This
        # test is not input selection, real model result use or an arithmetic run.
        batch={key:value.to(device) for key,value in encode_results(
            result(bytes([2,0,2,4])+bytes(60)),[7],[123]).items()}
        source=np.fromfile(os.environ['IGC_MIXED_INPUT'],dtype='<f4').copy()
        mixed=torch.from_numpy(source).reshape(1,2560).to(device);original=mixed.clone()
        tokenizer=NativeText.from_pack(Path(os.environ['IGC_PACK'])/'tokenizer')
        ids=tokenizer.encode('4');self.assertEqual(len(ids),1)
        target=torch.tensor(ids,dtype=torch.long,device=device)
        with NativeCudaHead(library,hashlib.sha256(library.read_bytes()).hexdigest(),os.environ['IGC_MODEL'],device) as head:
            optimizer=torch.optim.SGD(mapper.parameters(),lr=.1)
            before=torch.nn.functional.cross_entropy(head.project(mapper(mixed,batch)),target)
            before.backward()
            self.assertTrue(bool(torch.isfinite(mapper.proposal.weight.grad).all()))
            self.assertGreater(float(mapper.proposal.weight.grad.norm()),0)
            optimizer.step()
            after=torch.nn.functional.cross_entropy(head.project(mapper(mixed,batch)),target)
            self.assertLess(float(after.detach()),float(before.detach()))
            self.assertIsNone(mixed.grad)
            torch.testing.assert_close(mixed,original,rtol=0,atol=0)
            print('typed-result learned residual native CE',float(before.detach()),'->',float(after.detach()),
                  'target is one numeric symbol; not full result-use evidence')


if __name__=='__main__':unittest.main()
