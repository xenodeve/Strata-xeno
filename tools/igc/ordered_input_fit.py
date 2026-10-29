"""One frozen digit-view experiment on the existing native embedding cache."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
from tools.igc.embedding_readout import load_cache,TensorInputs,training_lr
from tools.igc.ordered_input import OrderedInputReadout,SIZES,targets,canonical_fields
from tools.igc.readout import targets as abi_targets


def summarize(rows,predicted):
    truth=targets(rows);positive=truth[:,0]>0;operator=predicted[:,0]==truth[:,0]
    valid=canonical_fields(predicted);expected=abi_targets(rows)
    exact=np.array([p is not None and p==t.tolist() for p,t in zip(valid,expected,strict=True)])
    operands=np.all(predicted[:,1:]==truth[:,1:],axis=1)
    return dict(n=len(rows),positive_count=int(positive.sum()),
        positive_exact_routed=int((positive&exact).sum()),positive_operand_tuple_correct=int((positive&operands).sum()),
        positive_operator_correct=int((positive&operator).sum()),
        negative_false_calls=int(((~positive)&(predicted[:,0]!=0)).sum()),
        positive_noncanonical=int(sum(p is None for p,yes in zip(valid,positive,strict=True) if yes)),
        field_accuracy=[dict(correct=int((predicted[positive,i]==truth[positive,i]).sum()),n=int(positive.sum())) for i in range(9)])


def train(dataset,sources,cache_protocol,protocol,output):
    from safetensors.torch import save_file
    from tools.qwenputer.memory import process_snapshot,system_snapshot
    output.mkdir(exist_ok=False)
    files=[Path(__file__),Path(__file__).with_name('ordered_input.py'),Path(__file__).with_name('embedding_readout.py'),
           Path(__file__).with_name('attention_readout.py'),Path(__file__).with_name('readout.py'),protocol,cache_protocol]
    before={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    table,indices,valid,rows,cache=load_cache(dataset,sources,cache_protocol)
    torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.manual_seed(20261031)
    torch.use_deterministic_algorithms(True)
    if os.environ.get('CUBLAS_WORKSPACE_CONFIG')!=':4096:8':raise ValueError('deterministic workspace required')
    if torch.cuda.mem_get_info()[0]<2*1024**3:raise RuntimeError('offline input fit needs 2GiB free')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True);torch.backends.mha.set_fastpath_enabled(False)
    inputs=TensorInputs(table,indices,valid,'cuda');model=OrderedInputReadout().to('cuda')
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01,foreach=False,fused=False)
    labels=targets(rows);fit=np.array([i for i,r in enumerate(rows) if r['split']=='train'],dtype=np.int64)
    random=torch.Generator().manual_seed(20261031);started=time.monotonic();history=[];resources=[]
    steps=(len(fit)+63)//64;total_steps=70*steps
    for epoch in range(70):
        order=fit[torch.randperm(len(fit),generator=random).numpy()];total=0.
        for offset in range(0,len(order),64):
            if time.monotonic()-started>600:raise TimeoutError('bounded ordered fit; no final model')
            rate=training_lr('cosine',epoch*steps+offset//64,total_steps)
            for group in optimizer.param_groups:group['lr']=rate
            selection=order[offset:offset+64];y=torch.from_numpy(labels[selection]).to('cuda');positive=y[:,0]>0
            optimizer.zero_grad(set_to_none=True)
            heads=model(*inputs.batch(selection)).split(SIZES,dim=1)
            loss=torch.nn.functional.cross_entropy(heads[0],y[:,0])
            if bool(positive.any()):
                for i in range(1,len(SIZES)):loss=loss+torch.nn.functional.cross_entropy(heads[i][positive],y[positive,i])
            if not bool(torch.isfinite(loss)):raise RuntimeError('nonfinite ordered loss')
            loss.backward();optimizer.step();total+=float(loss.detach())*len(selection)
        history.append(total/len(fit))
        if (epoch+1)%5==0:
            resources.append(dict(epoch=epoch+1,process=process_snapshot(os.getpid()),system=system_snapshot()))
            print('ordered epoch',epoch+1,'loss',history[-1],'seconds',round(time.monotonic()-started,3),flush=True)
    predicted=[];model.eval()
    with torch.no_grad():
        for offset in range(0,len(rows),64):
            selection=np.arange(offset,min(offset+64,len(rows)),dtype=np.int64)
            heads=model(*inputs.batch(selection)).split(SIZES,dim=1)
            predicted.extend(np.stack([h.argmax(1).cpu().numpy() for h in heads],axis=1))
    predicted=np.stack(predicted);summaries={}
    for split in dict.fromkeys(r['split'] for r in rows):
        selected=np.array([r['split']==split for r in rows])
        summaries[split]=summarize([r for r in rows if r['split']==split],predicted[selected])
    save_file({k:v.detach().cpu().contiguous() for k,v in model.state_dict().items()},str(output/'input-left-pad.safetensors'))
    if before!={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}:raise RuntimeError('ordered source changed')
    report=dict(kind='left-aligned neural digit/PAD diagnostic; exact executor ABI unchanged',
        seed=20261031,epochs=70,batch=64,parameters=sum(p.numel() for p in model.parameters()),
        final_checkpoint_only=True,training_seconds=time.monotonic()-started,loss_history=history,
        summaries=summaries,source_sha256=before,cache_sha256=cache['cache_sha256'],rows_sha256=cache['rows_sha256'],
        state_sha256=hashlib.sha256((output/'input-left-pad.safetensors').read_bytes()).hexdigest(),resources=resources,
        eligible_for_fresh_evaluation=all(s['positive_exact_routed']==s['positive_count'] and s['negative_false_calls']==0 for s in summaries.values()),
        fresh_evaluation_consumed=False,gradient_scope='input readout only; frozen native embedding cache',
        torch_version=torch.__version__,cuda_build=torch.version.cuda,device=torch.cuda.get_device_name(0))
    (output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    converted=canonical_fields(predicted)
    (output/'predictions.json').write_text(json.dumps([dict(id=r['id'],split=r['split'],target=t.tolist(),
        neural_prediction=p.tolist(),canonical_prediction=c) for r,t,p,c in zip(rows,labels,predicted,converted,strict=True)],indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ('source_sha256','resources','loss_history')},indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('dataset','sources','cache-protocol','protocol','output'):parser.add_argument('--'+name,type=Path,required=True)
    train(**vars(parser.parse_args()))
