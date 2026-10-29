"""Fail closed on nonfinite training; never clip, repair or retry updates."""
import torch
import hashlib
import json
from tools.igc.readout import SIZES


def admission(system,free_disk):
    if (system['physical_available_bytes']<8*1024**3 or system['system_commit_available_bytes']<12*1024**3
            or free_disk<4*1024**3+16*1024**2):
        raise RuntimeError('offline training admission; no private-memory cap')


def progress(elapsed,timeout,system):
    if not 0<=elapsed<=timeout:
        raise RuntimeError('bounded training/pricing deadline')
    if system['physical_available_bytes']<1024**3 or system['system_commit_available_bytes']<2*1024**3:
        raise RuntimeError('offline training emergency floor; no private-memory cap')


def state_fingerprint(model):
    digest=hashlib.sha256()
    for name,value in sorted(model.state_dict().items()):
        tensor=value.detach().cpu().contiguous()
        digest.update(json.dumps([name,str(tensor.dtype),list(tensor.shape)],separators=(',',':')).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def verify_reset(priced,initial,ids,sources,cache_identity,protocol,recipe):
    if (not priced.get('success') or priced.get('optimizer_steps')!=0 or priced.get('backward_batches')!=1
            or priced.get('initial_state_sha256')!=initial or priced.get('initial_batch_ids')!=ids
            or priced.get('source_sha256')!=sources or priced.get('cache_identity')!=cache_identity
            or priced.get('protocol_sha256')!=protocol or priced.get('recipe')!=recipe):
        raise ValueError('fresh fit/pricing reset identity mismatch')


def finite_parameters(model):
    for name, value in model.named_parameters():
        if not bool(torch.isfinite(value).all()):
            raise RuntimeError('nonfinite training parameter: '+name)


def checked_backward(model,loss):
    finite_parameters(model)
    if loss.ndim != 0 or not bool(torch.isfinite(loss)):
        raise RuntimeError('nonfinite or nonscalar training loss')
    loss.backward()
    gradients = 0
    for name, parameter in model.named_parameters():
        if parameter.grad is not None:
            gradients += 1
            if not bool(torch.isfinite(parameter.grad).all()):
                raise RuntimeError('nonfinite training gradient: '+name)
    if not gradients:
        raise RuntimeError('missing training gradients')


def checked_update(model,optimizer,loss):
    checked_backward(model,loss)
    optimizer.step()
    finite_parameters(model)
    for state in optimizer.state.values():
        if any(torch.is_tensor(v) and not bool(torch.isfinite(v).all()) for v in state.values()):
            raise RuntimeError('nonfinite optimizer state')


def categorical_loss(logits,labels):
    if (logits.ndim != 2 or logits.shape[1] != sum(SIZES) or labels.shape != (len(logits),9)
            or labels.dtype != torch.long or not bool(torch.isfinite(logits).all())
            or any(bool(((labels[:,i]<0)|(labels[:,i]>=size)).any()) for i,size in enumerate(SIZES))):
        raise ValueError('bounded finite categorical training inputs')
    heads = logits.split(SIZES,dim=1);positive = labels[:,0]>0
    loss = torch.nn.functional.cross_entropy(heads[0],labels[:,0])
    if bool(positive.any()):
        for i in range(1,len(SIZES)):
            loss = loss+torch.nn.functional.cross_entropy(heads[i][positive],labels[positive,i])
    return loss
