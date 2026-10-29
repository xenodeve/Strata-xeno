"""Reusable CUDA-native forward with a declared late-head STE backward.

No base-model update, decoder, arithmetic or early-layer backward. The caller
pins its first-party library digest and model inputs. Pointer arguments are
trusted in-process tensor interfaces, never model-supplied ABI records.
"""
import ctypes
import hashlib
from pathlib import Path
import re
import stat
import threading
import torch
from torch.autograd.function import once_differentiable


def verify_library(path,digest):
    path=Path(path)
    if not isinstance(digest,str) or not re.fullmatch('[0-9a-f]{64}',digest):
        raise ValueError('cuda_head_library_digest')
    info=path.lstat()
    if (not stat.S_ISREG(info.st_mode) or getattr(info,'st_file_attributes',0)&0x400
            or info.st_size>256*1024**2 or hashlib.sha256(path.read_bytes()).hexdigest()!=digest):
        raise ValueError('cuda_head_library_identity')
    return path.resolve()


def validate_tensor(value,width,device):
    if (not isinstance(value,torch.Tensor) or value.dtype!=torch.float32 or not value.is_cuda
            or value.device!=device or value.ndim!=2 or value.shape[1]!=width
            or not 1<=value.shape[0]<=64 or not value.is_contiguous() or value.data_ptr()%16
            or not bool(torch.isfinite(value).all())):
        raise ValueError('cuda_head_tensor')


class _Projection(torch.autograd.Function):
    @staticmethod
    def forward(ctx,value,head):
        ctx.head=head
        return head._run(value,False)
    @staticmethod
    @once_differentiable
    def backward(ctx,delta):
        return ctx.head._run(delta.contiguous(),True),None


class NativeCudaHead:
    def __init__(self,library,digest,model,device):
        path=verify_library(library,digest)
        self._lock=threading.Lock();self._handle=0;self._closed=True
        if not torch.cuda.is_available():raise RuntimeError('cuda_head_backend_unavailable')
        self.device=torch.device(device)
        if self.device.type!='cuda' or self.device.index is None:raise ValueError('explicit CUDA device required')
        model=str(model)
        if not model or '\0' in model or len(model.encode('utf-8'))>32768:raise ValueError('cuda_head_model_path')
        self._lib=ctypes.CDLL(str(path))
        self._lib.igc_head_bridge_abi.restype=ctypes.c_uint32
        self._lib.igc_head_error.restype=ctypes.c_char_p
        self._lib.igc_head_open.argtypes=[ctypes.c_char_p]
        self._lib.igc_head_open.restype=ctypes.c_uint64
        for name in ('igc_head_project','igc_head_vjp'):
            function=getattr(self._lib,name)
            function.argtypes=[ctypes.c_uint64,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_uint32,ctypes.c_void_p]
            function.restype=ctypes.c_int
        self._lib.igc_head_close.argtypes=[ctypes.c_uint64]
        self._lib.igc_head_close.restype=ctypes.c_int
        if self._lib.igc_head_bridge_abi()!=1:raise RuntimeError('cuda_head_bridge_abi')
        with torch.cuda.device(self.device):
            self.stream=torch.cuda.Stream(device=self.device)
            self._handle=self._lib.igc_head_open(model.encode('utf-8'))
        if not self._handle:raise RuntimeError(self._error())
        self._closed=False
    def _error(self):
        return (self._lib.igc_head_error() or b'cuda_head_unknown').decode('utf-8',errors='replace')[:1024]
    def _run(self,value,backward):
        with self._lock:
            if self._closed:raise RuntimeError('cuda_head_closed')
            width,output_width=(248320,2560) if backward else (2560,248320)
            validate_tensor(value,width,self.device)
            with torch.cuda.device(self.device):
                caller=torch.cuda.current_stream(self.device)
                with torch.cuda.stream(self.stream):
                    self.stream.wait_stream(caller)
                    output=torch.empty((len(value),output_width),dtype=torch.float32,device=self.device)
                    value.record_stream(self.stream)
                    function=self._lib.igc_head_vjp if backward else self._lib.igc_head_project
                    status=function(self._handle,value.data_ptr(),output.data_ptr(),len(value),self.stream.cuda_stream)
                    if status:raise RuntimeError(self._error())
                caller.wait_stream(self.stream)
                output.record_stream(caller)
            return output
    def project(self,value):return _Projection.apply(value,self)
    def close(self):
        with self._lock:
            if self._closed:return
            self._closed=True
            with torch.cuda.device(self.device):status=self._lib.igc_head_close(self._handle)
            self._handle=0
            if status:raise RuntimeError(self._error())
    def __enter__(self):return self
    def __exit__(self,*args):self.close()
    def __del__(self):
        try:
            if not getattr(self,'_closed',True):self.close()
        except Exception:pass
