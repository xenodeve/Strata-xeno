"""Explicit offline warm-start: bounded copied F32 tensors, never optimizer state."""
import hashlib
import re
from pathlib import Path
import torch
from safetensors.torch import load
from tools.igc.observation import _read


def initialization_options(path=None,digest=None):
    if (path is None)!=(digest is None):
        raise ValueError('initial state and explicit SHA256 must be supplied together')
    if digest is not None and (not isinstance(digest,str) or re.fullmatch('[0-9a-f]{64}',digest) is None):
        raise ValueError('initial state requires lowercase SHA256')


def initialize(model,path=None,digest=None):
    initialization_options(path,digest)
    if path is None:return {'kind':'fresh_seeded'}
    raw=_read(Path(path),limit=2*1024**2)
    if hashlib.sha256(raw).hexdigest()!=digest:raise ValueError('initial state content identity')
    try:values=load(raw)
    except Exception as error:raise ValueError('initial state must be bounded safetensors') from error
    expected=model.state_dict()
    if set(values)!=set(expected):raise ValueError('initial state exact tensor names; no optimizer payload')
    for name,tensor in values.items():
        if (tensor.dtype!=torch.float32 or expected[name].dtype!=torch.float32
                or tensor.shape!=expected[name].shape or not bool(torch.isfinite(tensor).all())):
            raise ValueError('initial state F32 shape/finiteness: '+name)
    # All tensors validate before the first copy; assign=False avoids shared parameters.
    model.load_state_dict(values,strict=True,assign=False)
    return {'kind':'copied_safetensors','artifact_sha256':digest,'optimizer_state_transferred':False}


def fresh_optimizer(model):
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01,foreach=False,fused=False)
    if len(optimizer.state):raise RuntimeError('optimizer must begin without moment state')
    return optimizer
