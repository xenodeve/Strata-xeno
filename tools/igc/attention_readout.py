"""Bounded activation-attention input-head experiment; no model/result injection."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch

from tools.igc.observation import load_preanchor
from tools.igc.readout import SIZES, load_dataset, metrics, standardize, targets


class InputReadout(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.norm=torch.nn.LayerNorm(2560)
        self.stream=torch.nn.Linear(2560,16)
        self.position=torch.nn.Embedding(80,64)
        self.key=torch.nn.Linear(64,64)
        self.value=torch.nn.Linear(64,64)
        self.context=torch.nn.Linear(2560,64)
        self.queries=torch.nn.Parameter(torch.empty(9,64))
        self.heads=torch.nn.ModuleList(torch.nn.Linear(64,size) for size in SIZES)
        torch.nn.init.normal_(self.position.weight,std=0.02)
        torch.nn.init.normal_(self.queries,std=0.02)

    def forward(self,states,valid,positions,query):
        if (states.ndim!=4 or states.shape[2:]!=(4,2560) or states.shape[1]>80
                or valid.dtype!=torch.bool or valid.shape!=states.shape[:2]
                or positions.dtype!=torch.long or positions.shape!=valid.shape
                or query.shape!=(states.shape[0],2560)
                or not bool(valid.any(dim=1).all()) or bool(((positions<0)|(positions>=80)).any())):
            raise ValueError('bounded activation attention input')
        # Padding and missing cache rows carry no evidence or gradient.
        safe=states.masked_fill(~valid[:,:,None,None],0)
        tokens=self.stream(self.norm(safe)).flatten(2)+self.position(positions)
        q=self.queries[None,:,:]+self.context(query)[:,None,:]
        scores=torch.bmm(q,self.key(tokens).transpose(1,2))/8
        weights=torch.softmax(scores.masked_fill(~valid[:,None,:],float('-inf')),dim=-1)
        result=torch.bmm(weights,self.value(tokens))+q
        return torch.cat([head(result[:,i]) for i,head in enumerate(self.heads)],dim=1)


def contexts(root,rows,stage='final'):
    if stage not in ('final','embedding','attention'):
        raise ValueError('unknown input activation stage')
    values=np.zeros((len(rows),80,4,2560),dtype=np.float32)
    valid=np.zeros((len(rows),80),dtype=np.bool_)
    positions=np.broadcast_to(np.arange(80,dtype=np.int64),(len(rows),80)).copy()
    records=json.loads((root/'records.json').read_text(encoding='utf-8'))
    for i in range(len(rows)):
        observed=load_preanchor(root,i,stage=stage)
        meta=records[i]['observation']
        identity=(dict(sha256=meta['preanchor_sha256'],positions=meta['preanchor_positions'],
                       sources=meta['preanchor_sources'],missing=meta['preanchor_missing'])
                  if stage=='final' else meta['preanchor_stages'][stage])
        if (observed['preanchor_sha256']!=identity['sha256']
                or observed['positions']!=identity['positions']
                or observed['sources']!=identity['sources']
                or observed['missing_positions']!=identity['missing'] or not observed['positions']):
            raise ValueError('preanchor collection identity or empty observation')
        anchor=observed['anchor']['position']
        for p,value in zip(observed['positions'],observed['residuals'],strict=True):
            index=p-anchor+80
            values[i,index]=value
            valid[i,index]=True
    return values,valid,positions


def train(root,sources,output,protocol,stage='final'):
    from safetensors.torch import save_file
    output.mkdir(parents=True,exist_ok=False)
    query,rows,collection=load_dataset(root,sources)
    train_mask=np.array([r['split']=='train' for r in rows])
    normalized,mean,scale=standardize(query,train_mask)
    values,valid,positions=contexts(root,rows,stage=stage)
    labels=targets(rows)
    torch.set_num_threads(4)
    torch.manual_seed(20261029)
    torch.use_deterministic_algorithms(True)
    model=InputReadout()
    optimizer=torch.optim.AdamW(model.parameters(),lr=0.001,weight_decay=0.01)
    inputs=[torch.from_numpy(x[train_mask]) for x in [values,valid,positions,normalized]]
    y=torch.from_numpy(labels[train_mask])
    positive=y[:,0]>0
    started=time.monotonic();losses=[]
    for epoch in range(300):
        if time.monotonic()-started>180:
            raise RuntimeError('bounded attention fit timeout; no completed-model claim')
        optimizer.zero_grad(set_to_none=True)
        heads=model(*inputs).split(SIZES,dim=1)
        loss=torch.nn.functional.cross_entropy(heads[0],y[:,0])
        for i in range(1,len(SIZES)):
            loss=loss+torch.nn.functional.cross_entropy(heads[i][positive],y[positive,i])
        if not torch.isfinite(loss):
            raise RuntimeError('nonfinite attention fit')
        loss.backward();optimizer.step();losses.append(float(loss.detach()))
    with torch.no_grad():
        prediction=model(*[torch.from_numpy(x) for x in [values,valid,positions,normalized]])
        predicted=np.stack([head.argmax(dim=1).numpy() for head in prediction.split(SIZES,dim=1)],axis=1)
    tensors={key:value.detach().contiguous() for key,value in model.state_dict().items()}
    tensors.update(mean=torch.from_numpy(mean),scale=torch.from_numpy(scale))
    save_file(tensors,str(output/'attention.safetensors'))
    train_report=metrics(predicted[train_mask],[r for r in rows if r['split']=='train'])
    diagnostic=metrics(predicted[~train_mask],[r for r in rows if r['split']=='heldout'])
    eligible=(train_report['positive_exact_routed']==train_report['positive_count']
              and diagnostic['positive_exact_routed']>0)
    report=dict(kind='activation-attention-input-readout; no deployed mapping',
                activation_stage=stage,query_stage='head_mixed_after_final_hc',
                training_seconds=time.monotonic()-started,epochs=300,seed=20261029,
                torch_version=torch.__version__,device='cpu',threads=4,
                parameters=sum(p.numel() for p in model.parameters()),
                train=train_report,known_diagnostic=diagnostic,eligible_for_fresh_evaluation=eligible,
                initial_loss=losses[0],final_loss=losses[-1],
                trained_state_sha256=hashlib.sha256((output/'attention.safetensors').read_bytes()).hexdigest(),
                protocol_sha256=hashlib.sha256(protocol.read_bytes()).hexdigest(),
                source_manifest_sha256=collection['source_manifest_sha256'],
                collection_exe_sha256=collection['exe_sha256'],
                gradient_scope='input readout only; no gradient through frozen Gyro or output mapper')
    (output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    (output/'predictions.json').write_text(json.dumps([dict(id=r['id'],split=r['split'],target=t.tolist(),predicted=p.tolist())
        for r,t,p in zip(rows,labels,predicted,strict=True)],indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--sources',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--protocol',type=Path,required=True)
    parser.add_argument('--activation-stage',choices=('final','embedding','attention'),default='final')
    args=parser.parse_args()
    train(args.dataset,args.sources,args.output,args.protocol,args.activation_stage)
