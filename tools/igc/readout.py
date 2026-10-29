"""Bounded CPU readout of frozen real final-head activations, not latent integration."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np

from tools.igc.mapper_corpus import manifest
from tools.igc.observation import load_observation

SIZES=[4,2,2,2,2,10,10,10,10]


def targets(rows):
    result=[]
    for row in rows:
        a,b=row['lhs'],row['rhs']
        if not (-99<=a<=99 and -99<=b<=99 and 0<=row['ready_op']<=3):
            raise ValueError('unsupported readout label')
        result.append([row['ready_op'],int(a<0),int(b<0),int(abs(a)>=10),int(abs(b)>=10),
                       abs(a)%10,abs(a)//10,abs(b)%10,abs(b)//10])
    return np.array(result,dtype=np.int64)


def standardize(x,train):
    if x.ndim!=2 or train.shape!=(len(x),) or train.dtype!=np.bool_ or not train.any() or not np.isfinite(x).all():
        raise ValueError('training normalization inputs')
    mean=x[train].mean(axis=0)
    scale=np.maximum(x[train].std(axis=0),np.float32(1e-4))
    return ((x-mean)/scale).astype(np.float32),mean,scale


def metrics(predicted,rows):
    truth=targets(rows)
    if predicted.shape!=truth.shape or any(np.any((predicted[:,i]<0)|(predicted[:,i]>=size)) for i,size in enumerate(SIZES)):
        raise ValueError('prediction shape or class')
    positive=truth[:,0]>0
    op=predicted[:,0]==truth[:,0]
    # Every sign/length/digit label must match; noncanonical representations fail.
    operands=np.all(predicted[:,1:]==truth[:,1:],axis=1)
    confusion=np.zeros((4,4),dtype=np.int64)
    for actual,guess in zip(truth[:,0],predicted[:,0],strict=True):
        confusion[actual,guess]+=1
    return dict(n=len(rows),positive_count=int(positive.sum()),
                negative_count=int((~positive).sum()),ready_operator_correct=int(op.sum()),
                positive_operand_tuple_correct=int((positive&operands).sum()),
                positive_exact_routed=int((positive&operands&op).sum()),
                negative_false_calls=int(((~positive)&(predicted[:,0]!=0)).sum()),
                positive_missed_calls=int((positive&(predicted[:,0]==0)).sum()),
                confusion=confusion.tolist())


def load_dataset(root,source_manifest):
    root=Path(root)
    report=json.loads((root/'result.json').read_text(encoding='utf-8'))
    if not report.get('success') or not report.get('owned_job_drain_verified'):
        raise ValueError('collection not completed successfully')
    if hashlib.sha256(source_manifest.read_bytes()).hexdigest()!=report['source_manifest_sha256']:
        raise ValueError('source manifest identity')
    identities=json.loads(source_manifest.read_text(encoding='utf-8'))
    if not identities.get('verified'):
        raise ValueError('source manifest not verified')
    for entry in identities['files']:
        st=Path(entry['path']).stat()
        if (st.st_size,st.st_mtime_ns)!=(entry['bytes'],entry['mtime_ns']):
            raise ValueError('enumerated source changed since collection hash')
    expected=manifest()
    observed=json.loads((root/'corpus.json').read_text(encoding='utf-8'))
    if expected!=observed:
        raise ValueError('frozen corpus identity')
    records=json.loads((root/'records.json').read_text(encoding='utf-8'))
    if len(records)!=64:
        raise ValueError('incomplete observations')
    vectors=[]
    for i,(row,record) in enumerate(zip(expected['rows'],records,strict=True)):
        if record['index']!=i or any(record.get(key)!=value for key,value in row.items()):
            raise ValueError('label or request identity')
        if not record['response'].get('choices'):
            raise ValueError('request not completed')
        observation=load_observation(root,i)
        for key in ['prefix_sha256','mixed_sha256','logits_sha256','metadata_sha256','position','token']:
            if observation[key]!=record['observation'][key]:
                raise ValueError('observation identity')
        vectors.append(observation['mixed'])
    return np.stack(vectors),expected['rows'],report


def train(root,source_manifest,output):
    import torch
    from safetensors.torch import save_file
    output.mkdir(parents=True,exist_ok=False)
    x,rows,collection=load_dataset(root,source_manifest)
    train_mask=np.array([r['split']=='train' for r in rows])
    normalized,mean,scale=standardize(x,train_mask)
    y=targets(rows)
    torch.set_num_threads(4)
    torch.manual_seed(20261029)
    torch.use_deterministic_algorithms(True)
    model=torch.nn.Sequential(torch.nn.Linear(2560,128),torch.nn.GELU(),torch.nn.Linear(128,sum(SIZES)))
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=0.01)
    inputs=torch.from_numpy(normalized[train_mask])
    labels=torch.from_numpy(y[train_mask])
    positives=labels[:,0]>0
    start=time.monotonic()
    losses=[]
    for epoch in range(300):
        if time.monotonic()-start>120:
            raise RuntimeError('bounded training deadline; no completed model claim')
        optimizer.zero_grad(set_to_none=True)
        heads=model(inputs).split(SIZES,dim=1)
        loss=torch.nn.functional.cross_entropy(heads[0],labels[:,0])
        for i in range(1,len(SIZES)):
            loss=loss+torch.nn.functional.cross_entropy(heads[i][positives],labels[positives,i])
        if not torch.isfinite(loss):
            raise RuntimeError('nonfinite training')
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    with torch.no_grad():
        logits=model(torch.from_numpy(normalized))
        predicted=np.stack([h.argmax(dim=1).numpy() for h in logits.split(SIZES,dim=1)],axis=1)
    tensors={key:value.detach().contiguous() for key,value in model.state_dict().items()}
    tensors.update(mean=torch.from_numpy(mean),scale=torch.from_numpy(scale))
    save_file(tensors,str(output/'readout.safetensors'))
    report=dict(kind='fixed-answer-start-final-head-readout; not production IGC',
                seed=20261029,epochs=300,optimizer='AdamW lr=0.001 weight_decay=0.01',
                torch_version=torch.__version__,device='cpu',threads=4,parameters=sum(p.numel() for p in model.parameters()),
                training_seconds=time.monotonic()-start,train=metrics(predicted[train_mask],[r for r in rows if r['split']=='train']),
                heldout=metrics(predicted[~train_mask],[r for r in rows if r['split']=='heldout']),
                initial_loss=losses[0],final_loss=losses[-1],source_manifest_sha256=collection['source_manifest_sha256'],
                collection_exe_sha256=collection['exe_sha256'],corpus_rows_sha256=manifest()['rows_sha256'],
                trained_state_sha256=hashlib.sha256((output/'readout.safetensors').read_bytes()).hexdigest(),
                gradient_scope='readout head only; no gradients or updates to deployed Gyro',
                limitations=manifest()['exclusions'])
    (output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    (output/'predictions.json').write_text(json.dumps([dict(id=r['id'],split=r['split'],target=t.tolist(),predicted=p.tolist())
        for r,t,p in zip(rows,y,predicted,strict=True)],indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--sources',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    train(args.dataset,args.sources,args.output)
