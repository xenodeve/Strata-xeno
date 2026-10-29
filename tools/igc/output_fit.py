"""Offline result-mapper pilot through the real quantized native head.

Only a late residual trains. Frozen teacher-forced states and the declared
dequantized-head STE do not establish full-body gradients or free generation.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time
import numpy as np
import torch
from tools.igc.cuda_head import NativeCudaHead
from tools.igc.output_data import load_dataset
from tools.igc.output_mapping import encode_results,ResultMapper
from tools.igc.observation import _read


def digest(path):
    with Path(path).open('rb') as source:return hashlib.file_digest(source,'sha256').hexdigest()


def score_predictions(records,predictions,losses):
    if len(records)!=len(predictions) or len(records)!=len(losses):
        raise ValueError('output score lengths')
    result={}
    for split in dict.fromkeys(r['split'] for r in records):
        selected=[(r,int(p),float(loss)) for r,p,loss in zip(records,predictions,losses,strict=True) if r['split']==split]
        numeric=[(r,p) for r,p,_ in selected if r['numeric_target']]
        ordinary=[(r,p) for r,p,_ in selected if not r['numeric_target']]
        samples=list(dict.fromkeys(r['sample'] for r,_,_ in selected))
        result[split]=dict(positions=len(selected),correct=sum(p==r['target'] for r,p,_ in selected),
            numeric_positions=len(numeric),numeric_correct=sum(p==r['target'] for r,p in numeric),
            other_positions=len(ordinary),other_correct=sum(p==r['target'] for r,p in ordinary),
            samples=len(samples),complete_sequences_correct=sum(all(p==r['target'] for r,p,_ in selected if r['sample']==s) for s in samples),
            numeric_sequences_correct=sum(all(p==r['target'] for r,p in numeric if r['sample']==s) for s in samples),
            mean_cross_entropy=sum(loss for _,_,loss in selected)/len(selected))
    return result


def train(dataset,reference,pack,sources,capture_protocol,protocol,library,library_sha256,model,output,qualify_only=False):
    from safetensors.torch import save_file
    from tools.qwenputer.memory import process_snapshot,system_snapshot
    if os.environ.get('CUBLAS_WORKSPACE_CONFIG')!=':4096:8':raise ValueError('deterministic workspace required')
    torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.manual_seed(20261105)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    device=torch.device('cuda:0')
    if torch.cuda.mem_get_info()[0]<2*1024**3:raise RuntimeError('offline mapper fit needs 2GiB free GPU')
    if system_snapshot()['system_commit_available_bytes']<4*1024**3:raise RuntimeError('offline mapper system headroom')
    output.mkdir(exist_ok=False)
    files=[Path(__file__),Path(__file__).with_name('output_data.py'),Path(__file__).with_name('output_examples.py'),
           Path(__file__).with_name('output_corpus.py'),Path(__file__).with_name('output_mapping.py'),
           Path(__file__).with_name('cuda_head.py'),protocol,capture_protocol,library]
    before={str(p):digest(p) for p in files}
    if before[str(library)]!=library_sha256:raise ValueError('native library pin')
    data=load_dataset(dataset,reference,pack,sources,capture_protocol)
    # The full model digest is inherited only from the recorded unchanged-stat
    # source manifest, avoiding an undocumented fresh 91GB hash in every trial.
    resolved=next((r for r in data['current_source_resolution'] if Path(r['resolved']).resolve()==model.resolve()),None)
    if resolved is None:raise ValueError('native head model not bound to source manifest')
    records=data['records'];mixed=torch.from_numpy(data['mixed']).to(device);original=mixed.clone()
    raw=b''.join(bytes.fromhex(r['checked_result_hex']) for r in records)
    pieces=[]
    for offset in range(0,len(records),64):
        chunk=records[offset:offset+64]
        pieces.append(encode_results(raw[offset*88:(offset+len(chunk))*88],
            [r['invocation'] for r in chunk],[r['invocation_position'] for r in chunk]))
    batch={k:torch.cat([b[k] for b in pieces]).to(device) for k in pieces[0]}
    labels=torch.tensor([r['target'] for r in records],dtype=torch.long,device=device)
    fit=np.array([i for i,r in enumerate(records) if r['split']=='train'],dtype=np.int64)
    mapper=ResultMapper().to(device)
    baseline_predictions=[];baseline_losses=[];qualification_started=time.monotonic()
    with NativeCudaHead(library,library_sha256,model,device) as head:
        with torch.no_grad():
            for offset in range(0,len(records),16):
                selection=slice(offset,min(offset+16,len(records)))
                actual=head.project(mixed[selection].contiguous())
                expected=np.stack([np.frombuffer(_read(dataset/f'logits.{i}',size=248320*4),dtype='<f4')
                    for i in range(selection.start,selection.stop)])
                if actual.cpu().numpy().tobytes()!=expected.tobytes():
                    raise ValueError('output native head/captured full logits are not bitwise equal')
                baseline_predictions.extend(actual.argmax(-1).cpu().tolist())
                baseline_losses.extend(torch.nn.functional.cross_entropy(actual,labels[selection],reduction='none').cpu().tolist())
        qualification=dict(positions=len(records),logits_per_position=248320,bitwise=True,
                           seconds=time.monotonic()-qualification_started)
        (output/'qualification.json').write_text(json.dumps(qualification,indent=2),encoding='utf-8')
        print('output native full-logit qualification',qualification,flush=True)
        if qualify_only:return
        optimizer=torch.optim.AdamW(mapper.parameters(),lr=.001,weight_decay=.01,foreach=False,fused=False)
        random=torch.Generator().manual_seed(20261105);history=[];resources=[]
        started=time.monotonic();steps=(len(fit)+15)//16;total_steps=70*steps
        def features(selection):return {k:v[selection] for k,v in batch.items()}
        for epoch in range(70):
            order=fit[torch.randperm(len(fit),generator=random).numpy()];total=0.
            for offset in range(0,len(order),16):
                if time.monotonic()-started>1800:raise TimeoutError('bounded output fit; final checkpoint not produced')
                step=epoch*steps+offset//16
                rate=.001*(.01+.99*.5*(1+math.cos(math.pi*step/(total_steps-1))))
                for group in optimizer.param_groups:group['lr']=rate
                selection=torch.as_tensor(order[offset:offset+16],device=device,dtype=torch.long)
                optimizer.zero_grad(set_to_none=True)
                logits=head.project(mapper(mixed[selection],features(selection)).contiguous())
                loss=torch.nn.functional.cross_entropy(logits,labels[selection])
                if not bool(torch.isfinite(loss)):raise RuntimeError('nonfinite output loss')
                loss.backward()
                if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in mapper.parameters()):
                    raise RuntimeError('nonfinite output gradient')
                optimizer.step();total+=float(loss.detach())*len(selection)
            history.append(total/len(fit))
            if (epoch+1)%5==0:
                resources.append(dict(epoch=epoch+1,process=process_snapshot(os.getpid()),system=system_snapshot()))
                print('output epoch',epoch+1,'loss',history[-1],'seconds',round(time.monotonic()-started,3),flush=True)
        # Evaluate only final weights. Corrupted/swapped feature views below are
        # named counterfactual diagnostics, never accepted production ABI inputs.
        controls={};predictions={};mapper.eval()
        with torch.no_grad():
            for name in ('learned','off','failed_result','zero_content','swapped_content'):
                guessed=[];losses=[]
                for offset in range(0,len(records),16):
                    selection=torch.arange(offset,min(offset+16,len(records)),device=device)
                    b=features(selection)
                    if name=='zero_content':b={**b,'features':torch.zeros_like(b['features'])}
                    elif name=='swapped_content':
                        # Shift by a whole sample, not merely a repeated position
                        # of the same numeric result. Ownership stays unchanged.
                        alternatives=[]
                        for i in selection.cpu().tolist():
                            j=(i+1)%len(records)
                            while records[j]['sample']==records[i]['sample']:j=(j+1)%len(records)
                            alternatives.append(j)
                        alternative=torch.tensor(alternatives,device=device)
                        b={**b,'features':batch['features'][alternative],'valid':batch['valid'][alternative]}
                    elif name=='failed_result':
                        b={'features':torch.zeros_like(b['features']),'valid':torch.zeros_like(b['valid']),
                           'active':torch.zeros_like(b['active']),'status':torch.full_like(b['status'],3)}
                    adapted=mapper(mixed[selection],b,enabled=name!='off')
                    if name in ('off','failed_result') and adapted.cpu().numpy().tobytes()!=mixed[selection].cpu().numpy().tobytes():
                        raise ValueError('output hard bypass changed activation')
                    logits=head.project(adapted.contiguous())
                    guessed.extend(logits.argmax(-1).cpu().tolist())
                    losses.extend(torch.nn.functional.cross_entropy(logits,labels[selection],reduction='none').cpu().tolist())
                controls[name]=score_predictions(records,guessed,losses);predictions[name]=guessed
            if not torch.equal(original,mixed):raise RuntimeError('output fit modified frozen activations')
            if predictions['off']!=baseline_predictions or predictions['failed_result']!=baseline_predictions:
                raise ValueError('output baseline prediction parity failed')
        save_file({k:v.detach().cpu().contiguous() for k,v in mapper.state_dict().items()},str(output/'output.safetensors'))
    if before!={str(p):digest(p) for p in files}:raise RuntimeError('output fit source/protocol changed during execution')
    report=dict(kind='offline teacher-forced result-mapper fit; not free generation or serving integration',
        seed=20261105,epochs=70,batch=16,optimizer='AdamW0.001/0.01 cosine to0.00001; no clipping',
        final_checkpoint_only=True,fit_seconds=time.monotonic()-started,qualification=qualification,
        loss_history=history,controls=controls,baseline=score_predictions(records,baseline_predictions,baseline_losses),
        parameters=sum(p.numel() for p in mapper.parameters()),
        state_sha256=digest(output/'output.safetensors'),training_source_sha256=before,
        collection_report_sha256=data['collection_report_sha256'],corpus_sha256=data['collection']['rows_sha256'],
        records_sha256=data['collection']['records_sha256'],source_manifest_sha256=data['collection']['source_manifest_sha256'],
        model_binding=resolved,resources=resources,torch_version=torch.__version__,cuda_build=torch.version.cuda,
        cuda_device=torch.cuda.get_device_name(0),peak_cuda_allocated=torch.cuda.max_memory_allocated(),
        peak_cuda_reserved=torch.cuda.max_memory_reserved(),
        gradient_scope='late result mapper and native dequantized-head STE only; no frozen Gyro body backward',
        limitations=['oracle operands isolate I3; I2 fails','teacher-forced previous correct answer tokens',
                     '30-question development pilot; no fresh acceptance','not full62-digit range',
                     'no native NN adapter injection or autonomous model result-use claim'])
    (output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    (output/'predictions.json').write_text(json.dumps([dict(index=i,sample=r['sample'],split=r['split'],
        target=r['target'],numeric=r['numeric_target'],predicted={k:v[i] for k,v in predictions.items()})
        for i,r in enumerate(records)],indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ('resources','training_source_sha256','loss_history','model_binding')},indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('dataset','reference','pack','sources','capture-protocol','protocol','library','model','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--library-sha256',required=True)
    parser.add_argument('--qualify-only',action='store_true')
    train(**vars(parser.parse_args()))
