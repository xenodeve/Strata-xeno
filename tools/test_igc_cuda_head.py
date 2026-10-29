"""Reusable native-head training seam, not arithmetic or model result-use tests."""
import hashlib
import ctypes
import os
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from tools.igc.cuda_head import NativeCudaHead,validate_tensor,verify_library


class HeadBoundary(unittest.TestCase):
    def test_host_or_malformed_tensors_never_reach_native_code(self):
        for value in (None,torch.zeros(2560),torch.zeros((1,2560)),torch.zeros((65,2560))):
            with self.assertRaises(ValueError):validate_tensor(value,2560,torch.device('cuda:0'))

    def test_library_identity_is_checked_before_loading(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'not-code.dll';path.write_bytes(b'not executable code')
            with self.assertRaises(ValueError):verify_library(path,'0'*64)
            with self.assertRaises(ValueError):verify_library(path,'bad-digest')
            with self.assertRaises(ValueError):verify_library(Path(tmp),'0'*64)


@unittest.skipUnless(os.environ.get('IGC_HEAD_BRIDGE_DLL'),'native CUDA training bridge not selected')
class NativeBridge(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dll=Path(os.environ['IGC_HEAD_BRIDGE_DLL'])
        cls.sha=hashlib.sha256(cls.dll.read_bytes()).hexdigest()
        cls.model=os.environ['IGC_MODEL']
        cls.device=torch.device('cuda:0')
        cls.head=NativeCudaHead(cls.dll,cls.sha,cls.model,cls.device)
    @classmethod
    def tearDownClass(cls):cls.head.close()

    def test_real_capture_forward_and_repeat_are_bitwise(self):
        source=np.fromfile(os.environ['IGC_MIXED_INPUT'],dtype='<f4')
        expected=np.fromfile(os.environ['IGC_EXPECTED_LOGITS'],dtype='<f4')
        self.assertEqual(source.shape,(2560,))
        self.assertEqual(expected.shape,(248320,))
        self.assertTrue(np.isfinite(source).all() and np.any(source))
        self.assertTrue(np.isfinite(expected).all() and np.any(expected))
        value=torch.from_numpy(source.copy()).reshape(1,2560).to(self.device)
        first=self.head.project(value);second=self.head.project(value)
        self.assertEqual(first.shape,(1,248320))
        self.assertEqual(first.detach().cpu().numpy().tobytes(),expected.tobytes())
        self.assertEqual(second.detach().cpu().numpy().tobytes(),expected.tobytes())

    def test_batched_sparse_vjp_matches_independent_q8_rows(self):
        from serve.gguf_info import read_header
        x=torch.zeros((2,2560),dtype=torch.float32,device=self.device,requires_grad=True)
        logits=self.head.project(x)
        (logits[0,391]-2*logits[1,23]).backward()
        actual=x.grad.cpu().numpy()
        header=read_header(self.model)
        tensor=next(t for t in header['tensors'] if t['name']=='output.weight')
        self.assertEqual((tensor['type'],tensor['dims']),(8,[2560,248320]))
        with open(self.model,'rb') as weights:
            for i,(token,coefficient) in enumerate(((391,1),(23,-2))):
                weights.seek(header['data_start']+tensor['offset']+token*80*34)
                blocks=np.frombuffer(weights.read(80*34),dtype=np.uint8).reshape(80,34)
                scale=blocks[:,:2].copy().view('<f2').reshape(80).astype(np.float32)
                codes=blocks[:,2:].copy().view(np.int8).astype(np.float32)
                expected=(scale[:,None]*codes).reshape(2560)*coefficient
                np.testing.assert_array_equal(actual[i],expected)

    def test_bad_shapes_dtype_alignment_and_nonfinite_are_refused(self):
        invalid=[torch.zeros((0,2560),device=self.device),torch.zeros((65,2560),device=self.device),
                 torch.zeros((1,2560),device=self.device,dtype=torch.float16),
                 torch.zeros((1,5120),device=self.device)[:,::2],
                 torch.zeros(2561,device=self.device)[1:].reshape(1,2560),
                 torch.full((1,2560),float('nan'),device=self.device)]
        for value in invalid:
            with self.assertRaises(ValueError):self.head.project(value)

    def test_raw_api_refusals_do_not_enqueue_or_poison_later_work(self):
        value=torch.zeros((1,2560),device=self.device)
        output=torch.full((1,248320),17.,device=self.device)
        host=ctypes.create_string_buffer(2560*4)
        function=self.head._lib.igc_head_project
        cases=[(0,value.data_ptr(),output.data_ptr(),1,self.head.stream.cuda_stream),
               (self.head._handle,value.data_ptr(),output.data_ptr(),1,0),
               (self.head._handle,value.data_ptr(),output.data_ptr(),65,self.head.stream.cuda_stream),
               (self.head._handle,ctypes.addressof(host),output.data_ptr(),1,self.head.stream.cuda_stream),
               (self.head._handle,value.data_ptr(),value.data_ptr(),1,self.head.stream.cuda_stream)]
        for arguments in cases:
            self.assertEqual(function(*arguments),1)
            self.assertTrue(bool(torch.all(output==17)))
        self.assertTrue(bool(torch.all(self.head.project(value)==0)))

    def test_one_native_forward_loss_step_uses_the_declared_ste(self):
        source=np.fromfile(os.environ['IGC_MIXED_INPUT'],dtype='<f4')
        value=torch.from_numpy(source.copy()).reshape(1,2560).to(self.device).requires_grad_(True)
        original=value.detach().clone()
        target=torch.tensor([391],device=self.device) # diagnostic class, not arithmetic answer 391
        before=torch.nn.functional.cross_entropy(self.head.project(value),target)
        before.backward()
        self.assertTrue(bool(torch.isfinite(value.grad).all()))
        self.assertGreater(float(value.grad.norm()),0)
        candidate=value.detach()-.1*value.grad
        after=torch.nn.functional.cross_entropy(self.head.project(candidate),target)
        self.assertLess(float(after.detach()),float(before.detach()))
        torch.testing.assert_close(value.detach(),original,rtol=0,atol=0)
        print('native cached-head diagnostic CE',float(before.detach()),'->',float(after.detach()))

    def test_close_fences_its_work_and_prevents_later_backward(self):
        value=torch.zeros((1,2560),device=self.device,requires_grad=True)
        head=NativeCudaHead(self.dll,self.sha,self.model,self.device)
        result=head.project(value);head.close();head.close()
        self.assertTrue(bool(torch.isfinite(result).all()))
        with self.assertRaises(RuntimeError):result.sum().backward()
        with self.assertRaises(RuntimeError):head.project(value)


if __name__=='__main__':unittest.main()
