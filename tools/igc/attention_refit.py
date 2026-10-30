"""One warm attention fit adding worded arithmetic to coherent anchors; external rows are held out by id hash. No routing or serving."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import time
import numpy as np
import torch
from safetensors.torch import save_file
from tools.igc.adapter_artifact import read_artifact
from tools.igc.attention_readout import InputReadout
from tools.igc.coherent_evaluate import _expression
from tools.igc.coherent_observation import load_input_observation,decode_plan
from tools.igc.native_qualification import decisions
from tools.igc.observation import _read
from tools.igc.readiness_refit import held
from tools.igc.training_guards import categorical_loss

SEED=20261103


def word_tuple(prompt):
    m=re.fullmatch(r'What is (\d+) (plus|minus|times) (\d+)\?',prompt)
    a,b=int(m[1]),int(m[3])
    return [{'plus':1,'minus':2,'times':3}[m[2]],0,0,int(a>9),int(b>9),a%10,a//10,b%10,b//10]


def load(archive,manifest,extra):
    rows=json.loads(_read(manifest,limit=2*1024**2))['rows'];states=[];masks=[];labels=[];fit=[];ids=[]
    for i,row in enumerate(rows):
        obs=load_input_observation(archive,i);states.append(np.array(obs['attention'],dtype='<f4'));masks.append(np.array(obs['mask'],dtype=bool))
        labels.append(_expression(row) if row['ready_op'] else [0]*9);fit.append(True);ids.append(row['id'])
    cases=[c for c in json.loads(_read(extra/'cases.json',limit=2*1024**2)) if c['ready']]
    worded=np.load(extra/'attention.npy');records=decode_plan((extra/'plan.bin').read_bytes())['records']
    for k,c in enumerate(cases):
        n=min(len(records[k]['tokens'])-1,80);states.append(worded[k]);masks.append(np.array([False]*(80-n)+[True]*n))
        labels.append(word_tuple(c['prompt']));fit.append(not held(c['id']));ids.append(c['id'])
    return np.stack(states),np.stack(masks),np.array(labels,dtype=np.int64),np.array(fit),ids,len(rows)


def predict(model,x,mask):
    out=[]
    with torch.no_grad():
        for s in range(0,len(x),32):
            v=torch.from_numpy(x[s:s+32]);m=torch.from_numpy(mask[s:s+32]);p=torch.arange(80).expand(len(v),80)
            out.append(model(v,m,p,torch.zeros(len(v),2560)).numpy())
    return np.concatenate([decisions(o) for o in out])


def fit(artifact,archive,manifest,extra,output,epochs):
    torch.set_num_threads(6);torch.manual_seed(SEED);rng=np.random.default_rng(SEED)
    raw=artifact.read_bytes()
    import struct
    fields=struct.unpack_from('<8s12IQ32s32s',raw)
    weights=read_artifact(raw,fields[14].hex(),fields[15].hex(),hashlib.sha256(raw).hexdigest())['tensors']
    model=InputReadout(use_final_query=False,contextual=True)
    model.load_state_dict({k:torch.from_numpy(np.array(v)) for k,v in weights.items()},strict=True)
    x,mask,labels,train,ids,coherent=load(archive,manifest,extra)
    before=predict(model.eval(),x,mask)
    optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=.01)
    index=np.flatnonzero(train);start=time.monotonic();losses=[]
    model.train()
    for epoch in range(epochs):
        rng.shuffle(index)
        for s in range(0,len(index),32):
            b=index[s:s+32];v=torch.from_numpy(x[b]);m=torch.from_numpy(mask[b]);p=torch.arange(80).expand(len(b),80)
            optimizer.zero_grad(set_to_none=True)
            loss=categorical_loss(model(v,m,p,torch.zeros(len(b),2560)),torch.from_numpy(labels[b]))
            loss.backward();optimizer.step();losses.append(float(loss.detach()))
        print('epoch',epoch,round(losses[-1],5),round(time.monotonic()-start),flush=True)
    after=predict(model.eval(),x,mask)
    output.mkdir(exist_ok=False)
    save_file({k:v.detach().contiguous() for k,v in model.state_dict().items()},str(output/'attention.safetensors'))
    positive=labels[:,0]>0;tail=np.arange(len(x))>=coherent
    exact=lambda p,m:[int((p[m]==labels[m]).all(1).sum()),int(m.sum())]
    report=dict(kind='warm attention fit, inspected development; not acceptance',seed=SEED,epochs=epochs,final_loss=losses[-1],seconds=time.monotonic()-start,
        coherent_positive_before=exact(before,~tail&positive),coherent_positive_after=exact(after,~tail&positive),
        worded_heldout_before=exact(before,tail&~train),worded_heldout_after=exact(after,tail&~train),
        worded_fit_after=exact(after,tail&train),promoted=False)
    (output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('artifact','archive','manifest','extra','output'):parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--epochs',type=int,default=40)
    fit(**vars(parser.parse_args()))
