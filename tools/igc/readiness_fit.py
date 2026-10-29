"""Diagnostic final-Gyro-activation routing, never operand extraction or serving."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from tools.igc.observation import _read,load_observation
from tools.igc.output_data import validate_record
from tools.igc.output_examples import wrapped_prefix
from tools.igc.provenance import verify_sources
from tools.igc.readiness_corpus import dataset,anchor_examples
from tools.igc.readout import standardize
from tools.qwenputer.tokenizer import NativeText


def metrics(predicted,rows):
    truth=np.array([r['ready'] for r in rows])
    if (predicted.shape!=truth.shape or not np.issubdtype(predicted.dtype,np.integer)
            or not np.isin(predicted,[0,1]).all()):raise ValueError('readiness prediction class/shape')
    confusion=np.zeros((2,2),dtype=np.int64)
    for y,p in zip(truth,predicted,strict=True):confusion[y,p]+=1
    result=dict(n=len(rows),correct=int((truth==predicted).sum()),positive_count=int((truth==1).sum()),
        positive_correct=int(((truth==1)&(predicted==1)).sum()),negative_count=int((truth==0).sum()),
        negative_false_calls=int(((truth==0)&(predicted==1)).sum()),confusion=confusion.tolist())
    pairs={}
    for row,guess in zip(rows,predicted,strict=True):
        if row['pair'] is not None:pairs.setdefault(row['pair'],[]).append(row['ready']==guess)
    if any(len(v)!=2 for v in pairs.values()):raise ValueError('incomplete readiness pair')
    result.update(matched_pairs=len(pairs),matched_pairs_correct=sum(all(v) for v in pairs.values()))
    for name,key in [('by_template','template'),('by_language','language')]:
        result[name]={}
        for value in sorted({r[key] for r in rows}):
            mask=np.array([r[key]==value for r in rows]);result[name][value]=dict(
                n=int(mask.sum()),correct=int(((truth==predicted)&mask).sum()),
                false_calls=int(((truth==0)&(predicted==1)&mask).sum()))
    return result


def load_dataset(root,reference,pack,sources,protocol):
    raw=_read(root/'report.json',limit=256*1024);report=json.loads(raw);corpus=dataset()
    if (report.get('success') is not True or report.get('readiness_anchors') is not True
            or report.get('positions')!=320 or report.get('samples')!=320
            or report.get('rows_sha256')!=corpus['rows_sha256']
            or report.get('source_manifest_sha256')!=hashlib.sha256(_read(sources,limit=256*1024)).hexdigest()
            or report.get('protocol_sha256')!=hashlib.sha256(_read(protocol,limit=256*1024)).hexdigest()
            or report.get('hard_private_commit_cap') is not None
            or report.get('actual_hub_vision_initialized') is not True
            or report.get('launcher_gpu_environment')!={'CUDA_VISIBLE_DEVICES':'1,0','CUDA_DEVICE_ORDER':''}):
        raise ValueError('readiness collection identity/completion')
    resolution=verify_sources(sources)
    if json.loads(_read(root/'corpus.json',limit=1024**2))!=corpus:raise ValueError('readiness corpus changed')
    records_raw=_read(root/'records.json',limit=8*1024**2)
    if hashlib.sha256(records_raw).hexdigest()!=report['records_sha256']:raise ValueError('readiness records changed')
    records=json.loads(records_raw);tokenizer=NativeText.from_pack(pack/'tokenizer')
    if tokenizer.binding_sha256!=report['tokenizer_binding_sha256']:raise ValueError('readiness tokenizer identity')
    prefix,suffix=wrapped_prefix(reference,tokenizer)
    examples=anchor_examples(corpus['rows'],tokenizer,prefix,suffix)
    if len(records)!=len(examples):raise ValueError('readiness incomplete records')
    vectors=[]
    for i,(record,example) in enumerate(zip(records,examples,strict=True)):
        observed=load_observation(root,i);validate_record(record,example,observed,i)
        if record.get('checked_result_hex') is not None or record.get('target') is not None:
            raise ValueError('readiness must not receive result/target')
        vectors.append(observed['mixed'])
    return np.stack(vectors),corpus['rows'],report,hashlib.sha256(raw).hexdigest(),resolution


def train(root,reference,pack,sources,protocol,output,linear=False,fit_protocol=None):
    import torch
    from safetensors.torch import save_file
    source_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    x,rows,collection,collection_hash,resolution=load_dataset(root,reference,pack,sources,protocol)
    train_mask=np.array([r['split']=='train' for r in rows]);x,mean,scale=standardize(x,train_mask)
    truth=np.array([r['ready'] for r in rows],dtype=np.int64)
    torch.set_num_threads(4);torch.manual_seed(20261107);torch.use_deterministic_algorithms(True)
    if linear and fit_protocol is None:raise ValueError('linear control requires its own frozen fit protocol')
    model=(torch.nn.Sequential(torch.nn.Linear(2560,2)) if linear else
           torch.nn.Sequential(torch.nn.Linear(2560,128),torch.nn.GELU(),torch.nn.Linear(128,2)))
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
    inputs=torch.from_numpy(x[train_mask]);labels=torch.from_numpy(truth[train_mask])
    output.mkdir(exist_ok=False);losses=[];start=time.monotonic()
    for epoch in range(300):
        if time.monotonic()-start>120:raise RuntimeError('readiness fit deadline; no completed checkpoint')
        optimizer.zero_grad(set_to_none=True)
        loss=torch.nn.functional.cross_entropy(model(inputs),labels)
        if not torch.isfinite(loss):raise RuntimeError('nonfinite readiness loss')
        loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise RuntimeError('nonfinite readiness gradient')
        optimizer.step();losses.append(float(loss.detach()))
    with torch.no_grad():logits=model(torch.from_numpy(x)).numpy()
    if not np.isfinite(logits).all():raise RuntimeError('nonfinite readiness predictions')
    predictions=logits.argmax(axis=1)
    weights={k:v.detach().contiguous() for k,v in model.state_dict().items()}
    weights.update(mean=torch.from_numpy(mean),scale=torch.from_numpy(scale))
    state=output/'readiness.safetensors';save_file(weights,str(state))
    report=dict(kind='diagnostic fixed-answer-start readiness; no learned operands or serving',seed=20261107,
        epochs=300,optimizer='AdamW lr0.001 weight_decay0.01 full-batch no clipping',torch_version=torch.__version__,
        device='cpu',threads=4,features='only actual2560 final mixed activation; train-only standardization',
        gradient_scope='readiness head only; no gradients/updates through Gyro',
        architecture='linear2560x2' if linear else 'mlp2560x128x2',
        fit_protocol_sha256=hashlib.sha256((fit_protocol or protocol).read_bytes()).hexdigest(),
        parameters=sum(p.numel() for p in model.parameters()),initial_loss=losses[0],final_loss=losses[-1],
        training_seconds=time.monotonic()-start,train=metrics(predictions[train_mask],[r for r in rows if r['split']=='train']),
        development=metrics(predictions[~train_mask],[r for r in rows if r['split']=='development']),
        source_sha256=source_hash,collection_sha256=collection_hash,source_resolution=resolution,
        rows_sha256=dataset()['rows_sha256'],trained_state_sha256=hashlib.sha256(state.read_bytes()).hexdigest(),
        collection_source_sha256=collection['source_sha256'],protocol_sha256=collection['protocol_sha256'],
        acceptance64_consumed=False,promoted=False)
    if source_hash!=hashlib.sha256(Path(__file__).read_bytes()).hexdigest():raise RuntimeError('fit source changed')
    (output/'predictions.json').write_text(json.dumps([dict(id=r['id'],pair=r['pair'],split=r['split'],
        target=r['ready'],predicted=int(p),logits=list(map(float,l)))
        for r,p,l in zip(rows,predictions,logits,strict=True)],indent=2),encoding='utf-8')
    (output/'losses.json').write_text(json.dumps(losses),encoding='utf-8')
    (output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('root','reference','pack','sources','protocol','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--linear',action='store_true')
    parser.add_argument('--fit-protocol',type=Path)
    args=parser.parse_args();train(**vars(args))
