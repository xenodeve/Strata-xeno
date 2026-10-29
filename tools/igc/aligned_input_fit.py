"""Unchanged input NN plus one supervised nullable digit-attention objective."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
from tools.igc.alignment import digit_positions
from tools.igc.attention_readout import InputReadout
from tools.igc.embedding_readout import load_cache,TensorInputs,training_lr
from tools.igc.output_examples import wrapped_prefix
from tools.igc.readout import SIZES,targets,metrics
from tools.igc.diagnostics import analyze
from tools.qwenputer.tokenizer import NativeText


def pointer_loss(scores,valid,target,positive):
    n=len(scores)
    if (scores.shape!=(n,9,80) or valid.dtype!=torch.bool or valid.shape!=(n,80)
            or target.dtype!=torch.long or target.shape!=(n,4) or positive.dtype!=torch.bool or positive.shape!=(n,)
            or bool(((target<0)|(target>=80)).any())):raise ValueError('alignment loss bounds')
    if not bool(positive.any()):return scores.new_zeros(())
    selected=target[positive];mask=valid[positive]
    if not bool(mask.gather(1,selected).all()):raise ValueError('alignment targets a missing row')
    selected_scores=scores[positive][:,[5,6,7,8],:].masked_fill(~mask[:,None,:],float('-inf'))
    return torch.nn.functional.cross_entropy(selected_scores.reshape(-1,80),selected.reshape(-1))


def train(dataset,sources,cache_protocol,protocol,reference,pack,output):
    from safetensors.torch import save_file
    from tools.qwenputer.memory import process_snapshot,system_snapshot
    output.mkdir(exist_ok=False)
    files=[Path(__file__),Path(__file__).with_name('alignment.py'),Path(__file__).with_name('attention_readout.py'),
           Path(__file__).with_name('embedding_readout.py'),Path(__file__).with_name('readout.py'),protocol,cache_protocol]
    before={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    table,indices,valid,rows,cache=load_cache(dataset,sources,cache_protocol)
    tokenizer=NativeText.from_pack(pack/'tokenizer');prefix,suffix=wrapped_prefix(reference,tokenizer)
    encoded=[tokenizer.encode((prefix+r['prompt'].encode('utf-8')+suffix).decode('utf-8')) for r in rows]
    expected_tokens=np.array(sorted({t for ids in encoded for t in ids[-81:-1]}),dtype='<i4')
    actual_tokens=np.fromfile(dataset/'native-token-ids.bin',dtype='<i4')
    if not np.array_equal(expected_tokens,actual_tokens):raise ValueError('alignment token table reconstruction')
    mapping={int(t):i for i,t in enumerate(expected_tokens)};positions=[]
    for i,(row,ids) in enumerate(zip(rows,encoded,strict=True)):
        observed=ids[-81:-1];expected=np.zeros(80,dtype=np.int32);mask=np.zeros(80,dtype=np.bool_)
        expected[-len(observed):]=[mapping[t] for t in observed];mask[-len(observed):]=True
        if not np.array_equal(expected,indices[i]) or not np.array_equal(mask,valid[i]):raise ValueError('alignment cache index/window')
        if tokenizer.decode_bytes([ids[-2]])!=b'<think>' or not valid[i,79]:raise ValueError('alignment null marker missing')
        positions.append([p if p>=0 else 79 for p in digit_positions(row,ids,tokenizer)])
    positions=np.asarray(positions,dtype=np.int64);labels=targets(rows)
    torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.manual_seed(20261031);torch.use_deterministic_algorithms(True)
    if os.environ.get('CUBLAS_WORKSPACE_CONFIG')!=':4096:8':raise ValueError('deterministic workspace required')
    if torch.cuda.mem_get_info()[0]<2*1024**3:raise RuntimeError('offline alignment fit needs 2GiB free')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True);torch.backends.mha.set_fastpath_enabled(False)
    inputs=TensorInputs(table,indices,valid,'cuda');model=InputReadout(False,True).to('cuda')
    keys=[];hook=model.key.register_forward_hook(lambda _m,_i,value:keys.append(value))
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01,foreach=False,fused=False)
    random=torch.Generator().manual_seed(20261031);fit=np.array([i for i,r in enumerate(rows) if r['split']=='train'],dtype=np.int64)
    steps=(len(fit)+63)//64;started=time.monotonic();history=[];resources=[]
    for epoch in range(70):
        order=fit[torch.randperm(len(fit),generator=random).numpy()];total=main_total=align_total=0.
        for offset in range(0,len(order),64):
            if time.monotonic()-started>600:raise TimeoutError('bounded alignment fit; no final model')
            for group in optimizer.param_groups:group['lr']=training_lr('cosine',epoch*steps+offset//64,70*steps)
            selection=order[offset:offset+64];y=torch.from_numpy(labels[selection]).to('cuda');positive=y[:,0]>0
            target=torch.from_numpy(positions[selection]).to('cuda');batch=inputs.batch(selection)
            keys.clear();optimizer.zero_grad(set_to_none=True)
            heads=model(*batch).split(SIZES,dim=1)
            main=torch.nn.functional.cross_entropy(heads[0],y[:,0])
            if bool(positive.any()):
                for j in range(1,9):main=main+torch.nn.functional.cross_entropy(heads[j][positive],y[positive,j])
            scores=torch.bmm(model.queries[None,:,:].expand(len(selection),-1,-1),keys[0].transpose(1,2))/8
            alignment=pointer_loss(scores,batch[1],target,positive);loss=main+alignment
            if not bool(torch.isfinite(loss)):raise RuntimeError('nonfinite alignment fit loss')
            loss.backward();optimizer.step();total+=float(loss.detach())*len(selection)
            main_total+=float(main.detach())*len(selection);align_total+=float(alignment.detach())*len(selection)
        history.append(dict(total=total/len(fit),main=main_total/len(fit),alignment=align_total/len(fit)))
        if (epoch+1)%5==0:
            resources.append(dict(epoch=epoch+1,process=process_snapshot(os.getpid()),system=system_snapshot()))
            print('alignment epoch',epoch+1,history[-1],'seconds',round(time.monotonic()-started,3),flush=True)
    model.eval();predicted=[];pointer=[]
    with torch.no_grad():
        for offset in range(0,len(rows),64):
            selection=np.arange(offset,min(offset+64,len(rows)),dtype=np.int64);keys.clear();batch=inputs.batch(selection)
            heads=model(*batch).split(SIZES,dim=1)
            predicted.extend(np.stack([h.argmax(1).cpu().numpy() for h in heads],axis=1))
            scores=torch.bmm(model.queries[None,:,:].expand(len(selection),-1,-1),keys[0].transpose(1,2))/8
            pointer.extend(scores.masked_fill(~batch[1][:,None,:],float('-inf'))[:,[5,6,7,8],:].argmax(-1).cpu().numpy())
    hook.remove();keys.clear();predicted=np.stack(predicted);pointer=np.stack(pointer);summaries={}
    for split in dict.fromkeys(r['split'] for r in rows):
        chosen=np.array([r['split']==split for r in rows]);positive=chosen&(labels[:,0]>0);selected=[r for r in rows if r['split']==split]
        null=positions[positive]==79;correct=pointer[positive]==positions[positive]
        summaries[split]=dict(metrics=metrics(predicted[chosen],selected),fields=analyze(selected,predicted[chosen]),
            pointer=dict(active_correct=int(correct[~null].sum()),active_n=int((~null).sum()),null_correct=int(correct[null].sum()),null_n=int(null.sum())))
    save_file({k:v.detach().cpu().contiguous() for k,v in model.state_dict().items()},str(output/'input.safetensors'))
    if before!={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}:raise RuntimeError('alignment training source changed')
    report=dict(kind='unchanged input NN with nullable digit alignment objective; not deployed IGC',seed=20261031,
        epochs=70,batch=64,alignment_weight=1,parameters=sum(p.numel() for p in model.parameters()),
        training_seconds=time.monotonic()-started,history=history,summaries=summaries,source_sha256=before,
        rows_sha256=cache['rows_sha256'],cache_sha256=cache['cache_sha256'],labels_sha256=hashlib.sha256(positions.tobytes()).hexdigest(),
        state_sha256=hashlib.sha256((output/'input.safetensors').read_bytes()).hexdigest(),resources=resources,
        eligible_for_fresh_evaluation=all(s['metrics']['positive_exact_routed']==s['metrics']['positive_count'] and s['metrics']['negative_false_calls']==0 for s in summaries.values()),
        fresh_acceptance_consumed=False,torch_version=torch.__version__,cuda_build=torch.version.cuda,
        gradient_scope='input head only; frozen native embeddings; positions/IDs only supervised labels')
    (output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    (output/'predictions.json').write_text(json.dumps([dict(id=r['id'],split=r['split'],target=t.tolist(),predicted=p.tolist())
        for r,t,p in zip(rows,labels,predicted,strict=True)],indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ('source_sha256','resources','history')},indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('dataset','sources','cache-protocol','protocol','reference','pack','output'):parser.add_argument('--'+name,type=Path,required=True)
    train(**vars(parser.parse_args()))
