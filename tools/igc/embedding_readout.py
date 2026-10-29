"""Bounded training on verified native embedding activations; no model injection."""
import argparse
import hashlib
import io
import json
import math
from pathlib import Path
import sys
import time
import numpy as np
import torch

from tools.igc.attention_readout import InputReadout
from tools.igc.balanced_corpus import dataset
from tools.igc.input_manifest import MATCHED, EXPANDED, validate_manifest, binding, check_cache_header
from tools.igc.diagnostics import analyze
from tools.igc.observation import _read
from tools.igc.provenance import verify_sources
from tools.igc.readout import SIZES,metrics,targets


def training_lr(schedule,step,total_steps):
    if type(step) is not int or type(total_steps) is not int or total_steps<2 or not 0<=step<total_steps:
        raise ValueError('global training step bounds')
    if schedule=='constant':return 0.001
    if schedule!='cosine':raise ValueError('unknown learning-rate schedule')
    return 0.001*(0.01+0.99*0.5*(1+math.cos(math.pi*step/(total_steps-1))))


def fit_budget(kind,seconds):
    if kind not in ('embedding','context'):raise ValueError('unknown activation cache kind')
    if seconds is None:return 1200 if kind=='context' else 600
    if kind!='context' or type(seconds) is not int or not 600<=seconds<=3600:
        raise ValueError('bounded explicit contextual fit budget')
    return seconds


def batch_inputs(table,indices,valid,selection):
    if (table.dtype!=np.float32 or table.ndim!=3 or table.shape[1:]!=(4,2560)
            or indices.dtype!=np.int32 or indices.ndim!=2 or indices.shape[1]!=80
            or valid.dtype!=np.bool_ or valid.shape!=indices.shape
            or np.any(indices<0) or np.any(indices>=len(table))):
        raise ValueError('bounded frozen activation cache indices')
    values=table[indices[selection]]
    mask=valid[selection]
    positions=np.broadcast_to(np.arange(80,dtype=np.int64),mask.shape).copy()
    return (torch.from_numpy(values),torch.from_numpy(mask.copy()),torch.from_numpy(positions),
            torch.zeros((len(selection),2560),dtype=torch.float32))


class TensorInputs:
    """Frozen activation rows and bounded gather indices, never ordinal features."""
    def __init__(self,table,indices,valid,device):
        batch_inputs(table,indices,valid,np.array([0]))
        self.device=torch.device(device)
        self.table=torch.from_numpy(table).to(self.device)
        self.indices=torch.from_numpy(indices.astype(np.int64)).to(self.device)
        self.valid=torch.from_numpy(valid).to(self.device)
        self.positions=torch.arange(80,dtype=torch.long,device=self.device)
    def batch(self,selection):
        if (selection.ndim!=1 or len(selection)>64 or selection.dtype not in (np.int32,np.int64)
                or np.any(selection<0) or np.any(selection>=len(self.indices))):
            raise ValueError('bounded activation minibatch selection')
        selected=torch.as_tensor(selection,dtype=torch.long,device=self.device)
        mask=self.valid[selected]
        return (self.table[self.indices[selected]],mask,self.positions.expand(len(selection),-1),
                torch.zeros((len(selection),2560),dtype=torch.float32,device=self.device))


def load_cache(root,sources,protocol):
    report=json.loads(_read(root/'report.json',limit=256*1024))
    if (report.get('success') is not True or report.get('native_cache_max_abs_difference')!=0
            or report.get('protocol_sha256')!=hashlib.sha256(protocol.read_bytes()).hexdigest()
            or report.get('source_manifest_sha256')!=hashlib.sha256(sources.read_bytes()).hexdigest()):
        raise ValueError('cache preparation/protocol identity')
    corpus=json.loads(_read(root/'corpus.json',limit=16*1024**2))
    if corpus.get('kind') in (MATCHED,EXPANDED):
        count=validate_manifest(corpus);manifest_binding=binding(corpus)
        if manifest_binding:
            check_cache_header(report,count=count,rows_sha=corpus['rows_sha256'],
                protocol_sha=hashlib.sha256(protocol.read_bytes()).hexdigest(),
                sources_sha=hashlib.sha256(sources.read_bytes()).hexdigest(),
                matched_sha=manifest_binding['input_manifest_sha256'])
    elif corpus.get('kind')!='balanced-native-embedding-input-development' or corpus!=dataset():
        raise ValueError('frozen balanced corpus identity')
    if corpus['rows_sha256']!=report['rows_sha256']:
        raise ValueError('frozen corpus rows identity')
    verify_sources(sources)
    count=report.get('unique_tokens')
    if type(count) is not int or not 1<=count<=1024:raise ValueError('bounded native row count')
    raw=_read(root/'native-embeddings.bin',size=count*4*2560*4,limit=40*1024**2)
    if hashlib.sha256(raw).hexdigest()!=report['cache_sha256']:raise ValueError('native cache identity')
    table=np.frombuffer(raw,dtype='<f4').reshape(count,4,2560).copy()
    arrays=[]
    for name in ('indices','valid'):
        data=_read(root/f'{name}.npy',limit=8*1024**2)
        if hashlib.sha256(data).hexdigest()!=report[name+'_sha256']:raise ValueError('cache index identity')
        arrays.append(np.load(io.BytesIO(data),allow_pickle=False))
    indices,valid=arrays
    if (indices.shape!=(len(corpus['rows']),80) or valid.shape!=indices.shape or not np.isfinite(table).all()
            or not valid.any(axis=1).all()):raise ValueError('cache geometry or missing input')
    batch_inputs(table,indices,valid,np.array([0]))
    return table,indices,valid,corpus['rows'],report


def train(root,sources,protocol,output,cache_protocol=None,contextual=False,device='cpu',epochs=30,seed=20261030,schedule='constant',activation_cache='embedding',fit_timeout=None):
    from safetensors.torch import save_file
    from tools.qwenputer.memory import process_snapshot,system_snapshot
    import os
    timeout=fit_budget(activation_cache,fit_timeout)
    output.mkdir(parents=True,exist_ok=False)
    source_paths=[Path(__file__),Path(__file__).with_name('attention_readout.py'),Path(__file__).with_name('balanced_corpus.py'),
                  Path(__file__).with_name('input_manifest.py'),Path(__file__).with_name('matched_corpus.py')]
    source_hashes={path.name:hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths}
    if activation_cache=='embedding':
        table,indices,valid,rows,cache=load_cache(root,sources,cache_protocol or protocol)
    elif activation_cache=='context':
        from tools.igc.context_cache import load_cache as load_context,ContextInputs
        table,valid,rows,cache=load_context(root,sources,cache_protocol or protocol)
        source_paths.append(Path(__file__).with_name('context_cache.py'))
        source_hashes={path.name:hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths}
    else:raise ValueError('unknown activation cache kind')
    if device not in ('cpu','cuda') or not 1<=epochs<=100:raise ValueError('bounded training recipe')
    labels=targets(rows)
    fit=np.array([i for i,r in enumerate(rows) if r['split']=='train'],dtype=np.int64)
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    torch.manual_seed(seed);torch.use_deterministic_algorithms(True)
    if device=='cuda':
        if os.environ.get('CUBLAS_WORKSPACE_CONFIG')!=':4096:8':raise ValueError('deterministic CUDA workspace required')
        if not torch.cuda.is_available():raise ValueError('CUDA training backend unavailable')
        if torch.cuda.mem_get_info()[0]<2*1024**3:raise ValueError('training-only GPU preflight needs 2GiB free')
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True);torch.backends.mha.set_fastpath_enabled(False)
        tensor_inputs=(TensorInputs(table,indices,valid,'cuda') if activation_cache=='embedding'
                       else ContextInputs(table,valid,'cuda'))
        inputs=tensor_inputs.batch
    else:
        inputs=(lambda selection:batch_inputs(table,indices,valid,selection)) if activation_cache=='embedding' else ContextInputs(table,valid,'cpu').batch
    random=torch.Generator().manual_seed(seed)
    model=InputReadout(use_final_query=False,contextual=contextual).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=0.001,weight_decay=0.01,foreach=False,fused=False)
    started=time.monotonic();history=[];resources=[]
    steps_per_epoch=(len(fit)+63)//64;total_steps=steps_per_epoch*epochs
    for epoch in range(epochs):
        order=fit[torch.randperm(len(fit),generator=random).numpy()]
        total=0.
        for offset in range(0,len(order),64):
            if time.monotonic()-started>timeout:raise RuntimeError('bounded fit deadline; no completed model')
            rate=training_lr(schedule,epoch*steps_per_epoch+offset//64,total_steps)
            for group in optimizer.param_groups:group['lr']=rate
            selection=order[offset:offset+64]
            y=torch.from_numpy(labels[selection]).to(device);positive=y[:,0]>0
            optimizer.zero_grad(set_to_none=True)
            heads=model(*inputs(selection)).split(SIZES,dim=1)
            loss=torch.nn.functional.cross_entropy(heads[0],y[:,0])
            if bool(positive.any()):
                for i in range(1,len(SIZES)):
                    loss=loss+torch.nn.functional.cross_entropy(heads[i][positive],y[positive,i])
            if not torch.isfinite(loss):raise RuntimeError('nonfinite balanced input loss')
            loss.backward();optimizer.step();total+=float(loss.detach())*len(selection)
        history.append(total/len(fit))
        if (epoch+1)%5==0:
            resources.append(dict(epoch=epoch+1,process=process_snapshot(os.getpid()),system=system_snapshot()))
            print('epoch',epoch+1,'loss',history[-1],'seconds',time.monotonic()-started,flush=True)
    predictions=[]
    with torch.no_grad():
        for offset in range(0,len(rows),64):
            selection=np.arange(offset,min(offset+64,len(rows)),dtype=np.int64)
            heads=model(*inputs(selection)).split(SIZES,dim=1)
            predictions.extend(np.stack([h.argmax(1).cpu().numpy() for h in heads],axis=1))
    predicted=np.stack(predictions)
    summaries={}
    for split in dict.fromkeys(r['split'] for r in rows):
        choose=np.array([r['split']==split for r in rows]);selected=[r for r in rows if r['split']==split]
        summaries[split]=dict(metrics=metrics(predicted[choose],selected),fields=analyze(selected,predicted[choose]))
    eligible=all(item['metrics']['positive_exact_routed']==item['metrics']['positive_count']
                 and item['metrics']['negative_false_calls']==0 for item in summaries.values())
    save_file({k:v.detach().cpu().contiguous() for k,v in model.state_dict().items()},str(output/'input.safetensors'))
    report=dict(kind='balanced frozen-activation neural input readout; not deployed IGC',
                seed=seed,epochs=epochs,batch=64,threads=4,device=device,torch_version=torch.__version__,
                parameters=sum(p.numel() for p in model.parameters()),training_seconds=time.monotonic()-started,
                query_mode='learned_queries_only',activation_stage='native_embedding_broadcast' if activation_cache=='embedding' else 'native_first_attention_output',
                activation_cache=activation_cache,
                fit_timeout_seconds=timeout,
                learning_rate_schedule=schedule,final_learning_rate=rate,global_steps=total_steps,
                contextual_block=contextual,
                gradient_scope='input readout only; frozen activation cache, no Gyro or output-mapper backward',
                initial_epoch_loss=history[0],final_epoch_loss=history[-1],loss_history=history,
                summaries=summaries,eligible_for_fresh_evaluation=eligible,fresh_evaluation_consumed=False,
                resources=resources,cache_sha256=cache['cache_sha256'],rows_sha256=cache['rows_sha256'],
                protocol_sha256=hashlib.sha256(protocol.read_bytes()).hexdigest(),
                cache_preparation_protocol_sha256=cache['protocol_sha256'],source_manifest_sha256=cache['source_manifest_sha256'],
                trained_state_sha256=hashlib.sha256((output/'input.safetensors').read_bytes()).hexdigest())
    if source_hashes!={path.name:hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths}:
        raise RuntimeError('training source changed during run')
    report.update(training_source_sha256=source_hashes,python_version=sys.version,
                  python_exe_sha256=hashlib.sha256(Path(sys.executable).read_bytes()).hexdigest())
    if device=='cuda':
        report['cuda']=dict(build=torch.version.cuda,device=torch.cuda.get_device_name(0),tf32=False,
                           sdp='math',peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                           peak_reserved_bytes=torch.cuda.max_memory_reserved())
    (output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    (output/'predictions.json').write_text(json.dumps([dict(id=r['id'],split=r['split'],target=t.tolist(),predicted=p.tolist())
        for r,t,p in zip(rows,labels,predicted,strict=True)],indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ('resources','loss_history')},indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('dataset','sources','protocol','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--cache-protocol',type=Path)
    parser.add_argument('--contextual',action='store_true')
    parser.add_argument('--device',choices=('cpu','cuda'),default='cpu')
    parser.add_argument('--epochs',type=int,default=30)
    parser.add_argument('--seed',type=int,default=20261030)
    parser.add_argument('--schedule',choices=('constant','cosine'),default='constant')
    parser.add_argument('--activation-cache',choices=('embedding','context'),default='embedding')
    parser.add_argument('--fit-timeout',type=int)
    args=parser.parse_args();train(args.dataset,args.sources,args.protocol,args.output,args.cache_protocol,args.contextual,
                                 args.device,args.epochs,args.seed,args.schedule,args.activation_cache,args.fit_timeout)
