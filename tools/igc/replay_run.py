"""One bounded, identity-bound replay price or fit; no inference integration."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
MODULE_STARTED=time.monotonic()
import numpy as np
import torch
from tools.igc.attention_readout import InputReadout
from tools.igc.context_cache import load_cache,digest,ContextInputs
from tools.igc.mixed_context import MixedContextInputs
from tools.igc.replay_protocol import verify_prepared,INITIAL_SHA
from tools.igc.initial_state import initialize,fresh_optimizer
from tools.igc.embedding_readout import training_lr
from tools.igc.readout import targets,SIZES
from tools.igc.matched_evaluate import validate_cuda_environment
from tools.igc.training_guards import admission,progress,state_fingerprint,checked_backward,checked_update,categorical_loss,finite_parameters
from tools.igc.observation import _read
from tools.qwenputer.memory import system_snapshot,process_snapshot

WARM_FINGERPRINT='c30c061edcd438bf8fbf15c18b9f6dcf6cd18e2210a37c0da65f43ffcbb8f943'
HISTORICAL_FIT_SECONDS=132.99816550000105
PRICE_BACKWARDS={0,119,239,359,479,599,719,839}


def estimate_budget(io_seconds,backward_seconds,validation_seconds,gather_batches):
    numbers=[io_seconds,validation_seconds,*backward_seconds]
    if (gather_batches!=840 or len(backward_seconds)!=8 or any(not np.isfinite(x) or x<0 for x in numbers)):
        raise ValueError('complete finite schedule pricing required')
    # Planning allowance, not a latency/speed claim. Historical same-shape fit
    # includes optimizer/guards absent from disposable backward-only pricing.
    compute=max(4*HISTORICAL_FIT_SECONDS,2*(backward_seconds[0]+839*max(backward_seconds[1:])))
    fit=2*io_seconds+compute+2*validation_seconds+60
    outer=fit+validation_seconds+15
    return dict(io_allowance_seconds=2*io_seconds,compute_allowance_seconds=compute,
        validation_allowance_seconds=2*validation_seconds,save_allowance_seconds=60,
        fit_estimate_seconds=fit,outer_estimate_seconds=outer,
        fit_bound_seconds=1800,outer_bound_seconds=1920,admitted=fit<=1800 and outer<=1920,
        scope='conservative admission estimate, not a guaranteed upper bound or serving benchmark')


def verify_price(receipt,binding):
    if (receipt.get('success') is not True or type(receipt.get('optimizer_steps')) is not int
            or receipt['optimizer_steps']!=0 or receipt.get('backward_batches')!=8
            or receipt.get('optimizer_constructed') is not False
            or receipt.get('gather_batches')!=840 or receipt.get('estimate',{}).get('admitted') is not True
            or any(receipt.get(k)!=v for k,v in binding.items())):
        raise ValueError('replay pricing/reset/recipe identity mismatch or budget refusal')
    expected=estimate_budget(receipt['io_total_seconds'],receipt['backward_seconds'],receipt['initial_validation_seconds'],receipt['gather_batches'])
    if receipt['estimate']!=expected or not expected['admitted']:
        raise ValueError('replay pricing estimate does not reproduce measured evidence')


def read_receipt(path):
    raw=_read(Path(path),limit=256*1024)
    return json.loads(raw),hashlib.sha256(raw).hexdigest()


def current_admission(receipt,validation_seconds,outer_prefix_seconds):
    result=estimate_budget(receipt['io_total_seconds'],receipt['backward_seconds'],
        max(validation_seconds,receipt['initial_validation_seconds']),receipt['gather_batches'])
    result['outer_estimate_seconds']=max(result['outer_estimate_seconds'],result['fit_estimate_seconds']+outer_prefix_seconds)
    result['admitted']=result['fit_estimate_seconds']<=1800 and result['outer_estimate_seconds']<=1920
    result['current_outer_prefix_seconds']=outer_prefix_seconds
    return result


def qualify_mixed(inputs,batches):
    selectors=[batches[0],batches[-1],[batches[0][63],batches[0][0],batches[0][63],batches[0][33],batches[0][0],batches[0][17]]]
    checked=0
    for refs in selectors:
        values,mask,positions,query=inputs.cpu_batch(np.array(refs,dtype=np.int64))
        for i,member in enumerate(refs):
            r=inputs.members[member];states,valid,_=inputs.sources[r['source']]
            expected=states[np.array([r['index']],dtype=np.int64)][0]
            if not np.array_equal(values[i],expected) or not np.array_equal(mask[i],valid[r['index']]):
                raise RuntimeError('mixed row/mask replay differs from direct source')
            checked+=1
        if not np.array_equal(positions,np.tile(np.arange(80),(len(refs),1))) or query.any():raise RuntimeError('mixed position/query contract')
    return dict(rows=checked,direct_source_exact=True,duplicate_selector=selectors[-1])


def execute(args):
    from safetensors.torch import save
    import os
    if args.mode=='fit' and args.pricing_receipt is None:raise ValueError('reviewed price required')
    if args.output.exists():raise FileExistsError('new output directory required')
    pre=system_snapshot();free=shutil.disk_usage(args.output.parent).free;admission(pre,free)
    if free<4*1024**3+32*1024**2:raise RuntimeError('32MiB output plus reserve required')
    start_validation=time.monotonic()
    priced,receipt_sha=(read_receipt(args.pricing_receipt) if args.mode=='fit' else (None,None))
    replay=verify_prepared(args.replay,args.old/'corpus.json',args.matched/'corpus.json',args.selection,args.replay_sha256)
    if args.initial_sha256!=INITIAL_SHA or replay['initial_artifact_sha256']!=INITIAL_SHA:raise ValueError('original retained initialization only')
    validate_cuda_environment()
    if torch.__version__!='2.11.0+cu130' or digest(sys.executable)!='060435b7c202f88ad5404eadbea481c06c0bb9cee00ebac9084840dcbf03ff92':
        raise ValueError('qualified runtime identity')
    torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.manual_seed(20261031)
    torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True);torch.backends.mha.set_fastpath_enabled(False)
    if torch.cuda.mem_get_info()[0]<2*1024**3:raise RuntimeError('offline GPU admission2GiB')
    if '5060 Ti' not in torch.cuda.get_device_name(0):raise ValueError('qualified primary GPU mapping')
    source_names=['replay_run.py','mixed_context.py','context_cache.py','replay_protocol.py','initial_state.py',
        'training_guards.py','attention_readout.py','embedding_readout.py','readout.py','matched_evaluate.py',
        'input_manifest.py','matched_corpus.py','observation.py','provenance.py','development_audit.py','diagnostics.py','balanced_corpus.py']
    paths=[Path(__file__).with_name(n) for n in source_names]+[Path(__file__).parents[1]/'qwenputer/memory.py',args.protocol]
    sources_hash={str(p):digest(p) for p in paths};protocol_sha=digest(args.protocol)
    small=[args.replay,args.selection,args.sources,args.old_protocol,args.matched_protocol,args.initial]
    for root in (args.old,args.matched):small += [root/'report.json',root/'corpus.json',root/'valid.npy',root/'records.json']
    small_hash={str(p):digest(p) for p in small}
    sources={};cache_identity={};completed=0;gathered=0;backwards=[];resources=[];history=[];io_times=[]
    try:
        for name,root,protocol in [('old',args.old,args.old_protocol),('matched',args.matched,args.matched_protocol)]:
            states,valid,rows,header=load_cache(root,args.sources,protocol)
            sources[name]=(states,valid,rows)
            expected=replay['expected_cache_identity'][name]
            observed=dict(states=header['states_sha256'],mask=header['valid_sha256'],report=digest(root/'report.json'),protocol=digest(protocol))
            if observed!=expected:raise ValueError('replay cache expected identity')
            cache_identity[name]={**observed,'corpus':digest(root/'corpus.json'),'source_manifest':digest(args.sources)}
        inputs=MixedContextInputs(sources,replay['members'],'cuda');labels=targets(inputs.rows)
        qualification=qualify_mixed(inputs,replay['batches'])
        model=InputReadout(use_final_query=False,contextual=True).to('cuda');initial=initialize(model,args.initial,args.initial_sha256)
        fingerprint=state_fingerprint(model)
        if fingerprint!=WARM_FINGERPRINT:raise ValueError('loaded initial fingerprint')
        recipe=dict(model_seed=20261031,updates=840,batch=64,parameters=sum(p.numel() for p in model.parameters()),
            lr=.001,weight_decay=.01,lr_schedule='cosine',final_lr=.00001,tf32=False,attention='math',threads=4,
            torch_version=str(torch.__version__),python_exe_sha256=digest(sys.executable),initialization=initial,
            schedule_sha256=replay['schedule_sha256'],member_sha256=replay['member_sha256'],fit_bound=1800,outer_bound=1920)
        if recipe['parameters']!=96964:raise ValueError('readout architecture changed')
        first_ids=[m['source']+':'+m['id'] for m in [replay['members'][i] for i in replay['batches'][0]]]
        binding=dict(initial_state_sha256=fingerprint,first_batch_ids=first_ids,source_sha256=sources_hash,
            cache_identity=cache_identity,protocol_sha256=protocol_sha,recipe=recipe,replay_sha256=args.replay_sha256)
        if args.mode=='fit':
            verify_price(priced,binding)
        validation_seconds=time.monotonic()-start_validation
        args.output.mkdir()
        admission(system_snapshot(),shutil.disk_usage(args.output.parent).free)
        if torch.cuda.mem_get_info()[0]<2*1024**3:raise RuntimeError('GPU admission changed during validation')
        admission_estimate=None
        if args.mode=='fit':
            admission_estimate=current_admission(priced,validation_seconds,time.monotonic()-MODULE_STARTED)
            (args.output/'admission.json').write_text(json.dumps(admission_estimate,indent=2),encoding='utf-8')
            if not admission_estimate['admitted']:raise RuntimeError('current validation no longer fits1800/1920 budget; no updates')
        def recheck():
            for p,sha in sources_hash.items():
                if digest(p)!=sha:raise RuntimeError('execution source/protocol changed')
            for p,sha in small_hash.items():
                if digest(p)!=sha:raise RuntimeError('execution input changed')
            for name,root in [('old',args.old),('matched',args.matched)]:
                if digest(root/'states.npy')!=cache_identity[name]['states']:raise RuntimeError('source states changed')
            if receipt_sha is not None and digest(args.pricing_receipt)!=receipt_sha:raise RuntimeError('price receipt changed')
        torch.cuda.reset_peak_memory_stats();started=time.monotonic()
        destination=np.empty((64,80,4,2560),dtype=np.float32)
        optimizer=None
        if args.mode=='fit':
            optimizer=fresh_optimizer(model)
            if optimizer.state:raise RuntimeError('nonempty optimizer state')
        for step,batch_refs in enumerate(replay['batches']):
            progress(time.monotonic()-started,120 if args.mode=='price' else 1800,system_snapshot())
            selection=np.array(batch_refs,dtype=np.int64);t=time.monotonic();cpu=inputs.cpu_batch(selection,destination)
            io_times.append(time.monotonic()-t);gathered+=1
            if args.mode=='fit' or step in PRICE_BACKWARDS:
                model.zero_grad(set_to_none=True);torch.cuda.synchronize();t=time.monotonic()
                gpu=tuple(torch.from_numpy(a).to('cuda') for a in cpu);y=torch.from_numpy(labels[selection]).to('cuda')
                loss=categorical_loss(model(*gpu),y)
                if args.mode=='price':checked_backward(model,loss)
                else:
                    for group in optimizer.param_groups:group['lr']=training_lr('cosine',step,840)
                    checked_update(model,optimizer,loss);completed+=1;history.append(float(loss.detach()))
                torch.cuda.synchronize();seconds=time.monotonic()-t
                if args.mode=='price':backwards.append(seconds)
                del gpu,y,loss
            if (step+1)%42==0:
                resources.append(dict(step=step+1,process=process_snapshot(os.getpid()),system=system_snapshot()))
                print(args.mode,'step',step+1,'seconds',time.monotonic()-started,flush=True)
            progress(time.monotonic()-started,120 if args.mode=='price' else 1800,system_snapshot())
        loop_seconds=time.monotonic()-started
        if gathered!=840:raise RuntimeError('incomplete fixed replay')
        final_predictions=None
        if args.mode=='fit':
            model.zero_grad(set_to_none=True);model.eval().requires_grad_(False)
            states,valid,rows=sources['matched'];evaluation=ContextInputs(states,valid,'cuda');final_predictions=[]
            with torch.no_grad():
                for offset in range(0,len(rows),64):
                    selection=np.arange(offset,min(offset+64,len(rows)),dtype=np.int64)
                    logits=model(*evaluation.batch(selection))
                    if logits.shape!=(len(selection),sum(SIZES)) or not bool(torch.isfinite(logits).all()):raise RuntimeError('final logits')
                    predicted=torch.stack([v.argmax(1) for v in logits.split(SIZES,dim=1)],dim=1).cpu().numpy()
                    final_predictions.extend(dict(id=rows[i]['id'],split=rows[i]['split'],target=targets([rows[i]])[0].tolist(),predicted=p.tolist()) for i,p in zip(selection,predicted,strict=True))
                    progress(time.monotonic()-started,1800,system_snapshot())
        phase_seconds=time.monotonic()-started
        finite_parameters(model)
        if args.mode=='price' and state_fingerprint(model)!=fingerprint:raise RuntimeError('pricing changed parameters')
        model.zero_grad(set_to_none=True);recheck()
        progress(time.monotonic()-started,120 if args.mode=='price' else 1800,system_snapshot())
        report={**binding,'success':True,'mode':args.mode,'optimizer_steps':completed,'backward_batches':len(backwards),
            'gather_batches':gathered,'initialization':initial,'optimizer_initial_state_entries':0 if optimizer is not None else None,
            'optimizer_state_transferred':False,'initial_validation_seconds':validation_seconds,'phase_seconds':phase_seconds,
            'loop_seconds':loop_seconds,'phase_including_recheck_seconds':time.monotonic()-started,
            'optimizer_constructed':optimizer is not None,'pricing_reset_verified':args.mode=='fit','mixed_qualification':qualification,
            'admission_estimate':admission_estimate,
            'io_total_seconds':sum(io_times),'io_batch_max_seconds':max(io_times),'backward_seconds':backwards,
            'resources':resources,'preflight':pre,'process':process_snapshot(os.getpid()),'system':system_snapshot(),
            'cuda':dict(device=torch.cuda.get_device_name(0),peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved()),
            'acceptance64_access':False,'artifact_promoted':False}
        if args.mode=='price':report['estimate']=estimate_budget(sum(io_times),backwards,validation_seconds,gathered)
        else:
            if completed!=840:raise RuntimeError('incomplete optimizer budget')
            data=save({k:v.detach().cpu().contiguous() for k,v in model.state_dict().items()})
            (args.output/'input.safetensors').write_bytes(data)
            (args.output/'predictions.json').write_text(json.dumps(final_predictions,indent=2),encoding='utf-8')
            report.update(trained_state_sha256=hashlib.sha256(data).hexdigest(),pricing_receipt_sha256=receipt_sha,loss_history=history)
        (args.output/('price.json' if args.mode=='price' else 'report.json')).write_text(json.dumps(report,indent=2),encoding='utf-8')
        print('TERMINAL',args.mode,json.dumps({k:v for k,v in report.items() if k in ('optimizer_steps','gather_batches','phase_seconds','estimate','trained_state_sha256')}),flush=True)
    except Exception as error:
        if args.output.exists():
            (args.output/'failure.json').write_text(json.dumps(dict(success=False,error=repr(error),completed_steps=completed,gather_batches=gathered,
                backward_batches=len(backwards),system=system_snapshot()),indent=2),encoding='utf-8')
        raise
    finally:
        for states,_,_ in sources.values():states.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=('price','fit'),required=True)
    for name in ('old','matched','old-protocol','matched-protocol','sources','replay','selection','protocol','initial','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--replay-sha256',required=True);p.add_argument('--initial-sha256',required=True)
    p.add_argument('--pricing-receipt',type=Path)
    execute(p.parse_args())


if __name__=='__main__':main()
