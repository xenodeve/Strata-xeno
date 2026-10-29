"""Offline gated numeric-result adapter; no trained result-use/integration claim.

The ABI stays canonical little-endian decimal. Only the neural view is
left-aligned. No token IDs, formatted answer, pointers or host arithmetic enter
this boundary. The deployed native tensor encoder is a later integration step.
"""
import struct
import torch

_RESULT=struct.Struct('<IIQQ64s')


def encode_results(raw,invocations,positions):
    if (type(raw) is not bytes or not 1<=len(invocations)<=64 or len(positions)!=len(invocations)
            or len(raw)!=len(invocations)*_RESULT.size
            or any(type(i) is not int or not 1<=i<2**64 for i in invocations)
            or any(type(p) is not int or not 0<=p<2**64 for p in positions)):
        raise ValueError('output result batch/ownership bounds')
    features=torch.zeros((len(invocations),62,12),dtype=torch.float32)
    valid=torch.zeros((len(invocations),62),dtype=torch.bool)
    active=torch.zeros(len(invocations),dtype=torch.bool);statuses=[]
    for i,(expected_id,expected_position) in enumerate(zip(invocations,positions,strict=True)):
        version,status,invocation,position,value=_RESULT.unpack_from(raw,i*_RESULT.size)
        if (version!=1 or status not in range(5) or invocation!=expected_id or position!=expected_position):
            raise ValueError('output result version/status/ownership')
        statuses.append(status)
        if status:
            if any(value):raise ValueError('failed result carries a numeric value')
            continue
        length,negative=value[:2];digits=value[2:]
        if (not 1<=length<=62 or negative not in (0,1) or any(d>9 for d in digits)
                or any(digits[length:]) or (length>1 and digits[length-1]==0)
                or (length==1 and digits[0]==0 and negative)):
            raise ValueError('output result noncanonical integer')
        active[i]=True;valid[i,:length]=True
        for j,digit in enumerate(reversed(digits[:length])):features[i,j,digit]=1
        features[i,:length,11]=negative
    return dict(features=features,valid=valid,active=active,status=torch.tensor(statuses,dtype=torch.long))


class ResultMapper(torch.nn.Module):
    """Learned per-position attention and residual gate at the final head input.

This component never chooses a digit or token directly. Training requires
real teacher-forced output-position states and the qualified native head loss.
"""
    def __init__(self):
        super().__init__()
        self.norm=torch.nn.LayerNorm(2560)
        self.query=torch.nn.Linear(2560,128)
        self.result=torch.nn.Linear(12,128)
        self.position=torch.nn.Embedding(62,128)
        self.key=torch.nn.Linear(128,128)
        self.value=torch.nn.Linear(128,128)
        self.proposal=torch.nn.Linear(128,2560)
        self.gate=torch.nn.Linear(2560+128,1)
        torch.nn.init.zeros_(self.proposal.weight);torch.nn.init.zeros_(self.proposal.bias)

    def forward(self,mixed,batch,enabled=True):
        if not enabled:return mixed # Hard bypass: no feature validation, module call or allocation.
        if (not isinstance(mixed,torch.Tensor) or mixed.dtype!=torch.float32 or mixed.ndim!=2
                or mixed.shape[1]!=2560 or not 1<=len(mixed)<=64 or not bool(torch.isfinite(mixed).all())
                or not isinstance(batch,dict)):
            raise ValueError('bounded output-mapper activation')
        features,valid,active,status=(batch.get(k) for k in ('features','valid','active','status'))
        if (not all(isinstance(x,torch.Tensor) and x.device==mixed.device for x in (features,valid,active,status))
                or features.dtype!=torch.float32 or features.shape!=(len(mixed),62,12)
                or valid.dtype!=torch.bool or valid.shape!=(len(mixed),62)
                or active.dtype!=torch.bool or active.shape!=(len(mixed),)
                or status.dtype!=torch.long or status.shape!=(len(mixed),)
                or bool(((status<0)|(status>4)).any()) or not torch.equal(status==0,active)
                or not torch.equal(valid.any(-1),active)):
            raise ValueError('bounded output-mapper result features')
        if not bool(active.any()):return mixed # Failure is never a successful zero.
        safe=features.masked_fill(~valid[:,:,None],0)
        if not bool(torch.isfinite(safe).all()):raise ValueError('nonfinite output result features')
        normalized=self.norm(mixed)
        tokens=self.result(safe)+self.position.weight[None,:,:]
        scores=torch.bmm(self.query(normalized)[:,None,:],self.key(tokens).transpose(1,2))/128**.5
        mask=valid.clone();mask[:,0]|=~active # inactive rows need a finite softmax, then hard bypass.
        weights=torch.softmax(scores.masked_fill(~mask[:,None,:],float('-inf')),dim=-1)
        result=torch.bmm(weights,self.value(tokens)).squeeze(1)
        gate=torch.sigmoid(self.gate(torch.cat([normalized,result],dim=-1)))
        delta=gate*self.proposal(result)
        return torch.where(active[:,None],mixed+delta,mixed)
