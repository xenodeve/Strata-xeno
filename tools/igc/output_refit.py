"""One warm output-mapper fit through the frozen native head; worded rows held out by id hash. No routing or serving."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct
import numpy as np
import torch
from safetensors.torch import save_file
from tools.igc.adapter_artifact import read_artifact
from tools.igc.coherent_observation import load_compact
from tools.igc.cuda_head import NativeCudaHead
from tools.igc.native_qualification import expected_records
from tools.igc.observation import _read
from tools.igc.output_mapping import ResultMapper,encode_results
from tools.igc.readiness_refit import held

SEED=20261104


def request(prompt):
    a,word,b=prompt.removeprefix('What is ').removesuffix('?').split()
    return [{'plus':1,'minus':2,'times':3}[word],int(a),int(b)]


def load(archive,manifest,extra):
    data=json.loads(_read(manifest,limit=2*1024**2));rows=data['rows'];mixed=[];targets=[];results=[];groups=[];train=[]
    for q in data['requests']:
        if q.get('target') is None:continue
        row=rows[q['base']];mixed.append(np.array(load_compact(archive,q['ordinal'])['mixed'],dtype='<f4'));targets.append(q['target'])
        results.append(expected_records(1,0,0,[row['ready_op'],row['lhs'],row['rhs']])[1]);groups.append(row['id']);train.append(True)
    for root in extra:
        cases=json.loads((root/'cases.json').read_text(encoding='utf-8'));plan=json.loads((root/'targets.json').read_text())
        states=np.load(root/'mixed.npy')
        for state,target,base in zip(states,plan['targets'],plan['bases'],strict=True):
            c=cases[base];mixed.append(state);targets.append(target)
            results.append(expected_records(1,0,0,request(c['prompt']))[1]);groups.append(c['id']);train.append(not held(c['id']))
    return np.stack(mixed).astype('<f4'),np.array(targets),b''.join(results),groups,np.array(train)


def fit(artifact,archive,manifest,extra,library,model,output,epochs):
    torch.manual_seed(SEED);device=torch.device('cuda:0')
    raw=artifact.read_bytes();fields=struct.unpack_from('<8s12IQ32s32s',raw)
    weights=read_artifact(raw,fields[14].hex(),fields[15].hex(),hashlib.sha256(raw).hexdigest())['tensors']
    mapper=ResultMapper();mapper.load_state_dict({k:torch.from_numpy(np.array(v)) for k,v in weights.items()},strict=True);mapper.to(device)
    x,y,results,groups,train=load(archive,manifest,extra)
    pieces=[encode_results(results[s*88:(s+64)*88],[1]*len(x[s:s+64]),[0]*len(x[s:s+64])) for s in range(0,len(x),64)]
    batch={k:torch.cat([p[k] for p in pieces]).to(device) for k in pieces[0]}
    mixed=torch.from_numpy(x).to(device);labels=torch.from_numpy(y).long().to(device)
    pick=lambda s:{k:v[s] for k,v in batch.items()}
    index=np.flatnonzero(train);rng=np.random.default_rng(SEED);steps=(len(index)+15)//16;total=steps*epochs
    optimizer=torch.optim.AdamW(mapper.parameters(),lr=.001,weight_decay=.01,foreach=False,fused=False)
    with NativeCudaHead(library,hashlib.sha256(library.read_bytes()).hexdigest(),model,device) as head:
        def predict():
            out=[]
            with torch.no_grad():
                for s in range(0,len(x),16):
                    e=torch.arange(s,min(s+16,len(x)),device=device);out.append(head.project(mapper(mixed[e],pick(e)).contiguous()).argmax(-1).cpu().numpy())
            return np.concatenate(out)
        def scores(p):
            ok={}
            for g,good,t in zip(groups,p==y,train,strict=True):ok.setdefault((g,bool(t)),[]).append(bool(good))
            part=lambda t:dict(positions=int((p==y)[train==t].sum()),of=int((train==t).sum()),
                               sequences=sum(all(v) for (g,k),v in ok.items() if k==t),samples=sum(k==t for (g,k) in ok))
            return dict(fit=part(True),held=part(False))
        before=scores(predict());print('before',before,flush=True)
        for epoch in range(epochs):
            rng.shuffle(index);loss_sum=0.
            for s in range(0,len(index),16):
                step=epoch*steps+s//16;rate=.001*(.01+.99*.5*(1+math.cos(math.pi*step/max(1,total-1))))
                for group in optimizer.param_groups:group['lr']=rate
                e=torch.as_tensor(index[s:s+16],device=device,dtype=torch.long);optimizer.zero_grad(set_to_none=True)
                loss=torch.nn.functional.cross_entropy(head.project(mapper(mixed[e],pick(e)).contiguous()),labels[e])
                if not bool(torch.isfinite(loss)):raise RuntimeError('nonfinite output loss')
                loss.backward();optimizer.step();loss_sum+=float(loss.detach())*len(e)
            print('epoch',epoch,round(loss_sum/len(index),5),flush=True)
        after=scores(predict())
    output.mkdir(exist_ok=False)
    save_file({k:v.detach().cpu().contiguous() for k,v in mapper.state_dict().items()},str(output/'output.safetensors'))
    report=dict(kind='warm output fit, inspected development; not acceptance',seed=SEED,epochs=epochs,before=before,after=after,promoted=False)
    (output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('artifact','archive','manifest','library','model','output'):parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--extra',type=Path,nargs='+',required=True)
    parser.add_argument('--epochs',type=int,default=40)
    fit(**vars(parser.parse_args()))
