"""One readiness fit on old anchors plus coherent anchors; structural clusters are held out. No routing or serving."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from tools.igc.coherent_observation import load_input_observation
from tools.igc.observation import load_observation,_read
from tools.igc.readout import standardize

SEED=20261102


def held(cluster):
    """Fixed before fitting: a quarter of the structural clusters."""
    return hashlib.sha256(f'{SEED}:{cluster}'.encode()).digest()[0]%4==0


def old_rows(root):
    rows=json.loads(_read(root/'corpus.json',limit=1024**2))['rows']
    return np.stack([load_observation(root,i)['mixed'] for i in range(len(rows))]),rows


def coherent_rows(archive,manifest):
    rows=manifest['rows'];clusters=manifest['clusters'];vectors=[];out=[]
    for i,row in enumerate(rows):
        vectors.append(load_input_observation(archive,i)['mixed'])
        key=clusters[row['group']] if row.get('group') is not None else f"ordinary:{row['id']}"
        out.append(dict(id=row['id'],ready=int(row['ready_op']!=0),language=row['language'],intent=row['intent'],held=held(key)))
    return np.stack(vectors),out


def counts(predicted,rows):
    result={}
    for name,mask in [('all',[True]*len(rows)),*[(f"{r['language']}/{r['intent']}",[q['language']==r['language'] and q['intent']==r['intent'] for q in rows]) for r in rows]]:
        m=np.array(mask);truth=np.array([r['ready'] for r in rows])
        result[name]=dict(n=int(m.sum()),correct=int(((predicted==truth)&m).sum()),
                          false_calls=int(((predicted==1)&(truth==0)&m).sum()),missed=int(((predicted==0)&(truth==1)&m).sum()))
    return result


def external_rows(root):
    rows=json.loads(_read(root/'cases.json',limit=2*1024**2))
    return np.load(root/'mixed.npy'),[dict(id=r['id'],ready=r['ready'],language='en',intent=r['suite'],held=held(r['id'])) for r in rows]


def fit(archive,manifest,old,output,extra=None):
    import torch
    from safetensors.torch import save_file
    x_new,new=coherent_rows(archive,json.loads(_read(manifest,limit=2*1024**2)))
    x=[x_new];labels=[r['ready'] for r in new];train=[not r['held'] for r in new]
    for root in old:
        vectors,rows=old_rows(root);x.append(vectors)
        labels+=[int(r['ready']) for r in rows];train+=[r['split']=='train' for r in rows]
    outside=[]
    if extra is not None:
        vectors,outside=external_rows(extra);x.append(vectors)
        labels+=[r['ready'] for r in outside];train+=[not r['held'] for r in outside]
    x=np.concatenate(x);labels=np.array(labels);train=np.array(train);x,mean,scale=standardize(x,train)
    torch.set_num_threads(4);torch.manual_seed(SEED);torch.use_deterministic_algorithms(True)
    model=torch.nn.Sequential(torch.nn.Linear(2560,128),torch.nn.GELU(),torch.nn.Linear(128,2))
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
    inputs=torch.from_numpy(x[train]);truth=torch.from_numpy(labels[train])
    for _ in range(300):
        optimizer.zero_grad(set_to_none=True)
        loss=torch.nn.functional.cross_entropy(model(inputs),truth)
        if not torch.isfinite(loss):raise RuntimeError('nonfinite readiness loss')
        loss.backward();optimizer.step()
    with torch.no_grad():predicted=model(torch.from_numpy(x)).argmax(1).numpy()
    output.mkdir(exist_ok=False)
    weights={k:v.detach().contiguous() for k,v in model.state_dict().items()}
    weights.update(mean=torch.from_numpy(mean),scale=torch.from_numpy(scale))
    save_file(weights,str(output/'readiness.safetensors'))
    n=len(new);mask=np.array([r['held'] for r in new]);tail=predicted[len(predicted)-len(outside):] if outside else predicted[:0]
    held_outside=np.array([r['held'] for r in outside],dtype=bool)
    report=dict(kind='readiness fit, inspected development; not acceptance',seed=SEED,train_rows=int(train.sum()),final_loss=float(loss.detach()),
        external_held=counts(tail[held_outside],[r for r,m in zip(outside,held_outside) if m]) if outside else None,
        external_fit=counts(tail[~held_outside],[r for r,m in zip(outside,held_outside) if not m]) if outside else None,
        coherent_held=counts(predicted[:n][mask],[r for r,m in zip(new,mask) if m]),
        coherent_fit=counts(predicted[:n][~mask],[r for r,m in zip(new,mask) if not m]),
        old_development=dict(n=int((~train[n:len(train)-len(outside)]).sum()),correct=int((predicted[n:len(train)-len(outside)]==labels[n:len(train)-len(outside)])[~train[n:len(train)-len(outside)]].sum())),promoted=False)
    (output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive',type=Path,required=True);parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--old',type=Path,nargs='+',required=True);parser.add_argument('--extra',type=Path);parser.add_argument('--output',type=Path,required=True)
    fit(**vars(parser.parse_args()))
