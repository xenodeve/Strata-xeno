"""Bounded opt-in actual-Hub observations; never train, alter routing or open corpora."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
from tools.igc.coherent_observation import decode_plan,CompactArchive,first_four_fidelity,_json_bytes,_exclusive
from tools.igc.observation import _read
from tools.igc.hub_contract import validate_launcher_env,current_job

PROFILE_SHA='06ef847e063bb9643c7bf3f486e2ffdadc29737493268238c21f59188d5f6c5b'
SERVER_SHA='10be781197be5ffe7e41e9307c53d7444631f697cc1c81d456a2b75ee950ccbc'


def digest(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def admission(system,free_bytes,active_engines):
    if system['physical_available_bytes']<24*1024**3:raise RuntimeError('actual Hub physical24GiB admission; no model started')
    if system['system_commit_available_bytes']<51_128_263_976:raise RuntimeError('diagnostic commit47.6169GiB admission estimate; no model started')
    if free_bytes<5.5*1024**3:raise RuntimeError('diagnostic disk5.5GiB admission; no model started')
    if active_engines:raise RuntimeError('another engine is active; no duplicate model started')
    return True


def guarded_monitor(operation,cancel,failure,stop=None):
    try:
        operation()
        if stop is not None and not stop.is_set() and not cancel.is_set():
            failure.append(dict(reason='telemetry monitor stopped unexpectedly'));cancel.set()
    except Exception as error:
        failure.append(dict(reason='telemetry monitor exception',error=repr(error)))
        cancel.set()


def require_clean_finish(output,remaining,failure,samples,summary):
    if remaining or failure or not samples or summary is None:
        _exclusive(output/'FAILURE.json',_json_bytes(dict(remaining=remaining,failure=failure,samples=len(samples),summary=summary is not None)))
        raise RuntimeError('coherent cleanup/telemetry incomplete; no successful run report')


def collect(prepared,profile,server,executable,bundle,pack,sources,protocol,output):
    # Account predictable compaction-runtime commitment before admission rather
    # than cold-importing it after the model fills most system commitment.
    import numpy
    import psutil
    from tools.qwenputer.memory import system_snapshot,process_snapshot
    from tools.igc.provenance import verify_sources
    validate_launcher_env(os.environ)
    active=[p.info['name'] for p in psutil.process_iter(['name']) if (p.info['name'] or '').lower() in ('strata.exe','strata-vision.exe')]
    admission(system_snapshot(),shutil.disk_usage(output.parent).free,active)
    if output.exists():raise ValueError('coherent output exists/partial; resume is unsupported')
    if digest(profile)!=PROFILE_SHA or digest(server)!=SERVER_SHA:
        raise ValueError('coherent selected actual Hub profile/server changed; revise protocol before collection')
    resolution=verify_sources(sources)
    if any(Path(r['original']).resolve()!=Path(r['resolved']).resolve() for r in resolution):
        raise ValueError('current collector cannot substitute historical frozen source files')
    pinned={Path(r['original']).resolve():r['sha256'] for r in resolution}
    required=[profile,server,executable,bundle,pack/'index.txt',prepared/'plan.bin',prepared/'manifest.json',protocol,
              Path(sys.executable),Path(__file__),Path(__file__).with_name('coherent_observation.py'),Path(__file__).with_name('coherent_prepare.py')]
    if any(p.resolve() not in pinned or digest(p)!=pinned[p.resolve()] for p in required):
        raise ValueError('coherent actual sources not pinned')
    plan_raw=_read(prepared/'plan.bin',limit=3*1024**2);plan=decode_plan(plan_raw)
    manifest=json.loads(_read(prepared/'manifest.json',limit=2*1024**2))
    if (manifest.get('version')!=2 or manifest.get('anchors')!=400 or manifest.get('output_samples')!=48
            or manifest.get('positive')!=192 or manifest.get('negative')!=208 or manifest.get('acceptance64_access') is not False
            or manifest.get('plan_sha256')!=hashlib.sha256(plan_raw).hexdigest()
            or len(manifest.get('requests',[]))!=len(plan['records'])):raise ValueError('coherent frozen manifest')
    for i,(r,q) in enumerate(zip(plan['records'],manifest['requests'],strict=True)):
        if q.get('ordinal')!=i or any(q.get(k)!=r[k] for k in ('role','base','step')):raise ValueError('coherent request roles')
        if (i<400)!=(r['role']=='anchor'):raise ValueError('coherent anchor-first pass')
        if q.get('target') is not None and (type(q['target']) is not int or not 0<=q['target']<248320):raise ValueError('coherent offline target')
    cfg=json.loads(profile.read_text(encoding='utf-8-sig'))
    output.mkdir(exist_ok=False)
    # Source identities are evidence, not activation features. Only the binary
    # plan with consumed tokens/roles is supplied to the native producer.
    before={str(p):digest(p) for p in required}
    archive_sources=dict(engine=digest(executable),profile=digest(profile),server=digest(server),
                         bundle=digest(bundle),pack_index=digest(pack/'index.txt'),protocol=digest(protocol))
    compact=CompactArchive(output/'compact',plan_raw,archive_sources)
    original_popen=subprocess.Popen;engine=None;vision=None;vision_log=None;summary=None
    samples=[];failure=[];completed=[];stop=threading.Event();cancel=threading.Event();request_start=[None]
    started=time.monotonic();capture_start=[None];owner=psutil.Process(os.getpid())
    def headless(*args,**kwargs):
        kwargs['creationflags']=kwargs.get('creationflags',0)|0x08000000
        return original_popen(*args,**kwargs)
    def monitor():
        while not stop.wait(.2):
            system=system_snapshot();children=[]
            for p in owner.children(recursive=True):
                try:children.append(process_snapshot(p.pid))
                except (psutil.Error,OSError):pass
            now=time.monotonic();samples.append(dict(seconds=now-started,system=system,processes=children))
            if (now-started>2100 or (capture_start[0] and now-capture_start[0]>1800)
                    or (request_start[0] and now-request_start[0]>45)
                    or system['system_commit_available_bytes']<2*1024**3 or system['physical_available_bytes']<1024**3):
                failure.append(dict(seconds=now-started,reason='diagnostic time/system safety bound',system=system));cancel.set();return
    thread=threading.Thread(target=lambda:guarded_monitor(monitor,cancel,failure,stop),name='igc-coherent-telemetry',daemon=True)
    try:
        sys.path[:0]=[str(server.parent),str(server.parent.parent)]
        spec=importlib.util.spec_from_file_location('igc_coherent_pinned_server',server)
        helper=importlib.util.module_from_spec(spec);sys.modules[spec.name]=helper;spec.loader.exec_module(helper)
        if cfg.get('cjk_guard') or helper.layer_split_of(cfg) or cfg.get('replicas'):
            raise ValueError('coherent collector requires selected non-replica D2x profile')
        from tools.qwenputer.tokenizer import NativeText
        tokenizer=NativeText.from_pack(pack/'tokenizer')
        if tokenizer.binding_sha256!=plan['tokenizer']:raise ValueError('coherent runtime tokenizer binding')
        admission(system_snapshot(),shutil.disk_usage(output.parent).free,
            [p.info['name'] for p in psutil.process_iter(['name']) if (p.info['name'] or '').lower() in ('strata.exe','strata-vision.exe')])
        env=helper.child_env(cfg);env.pop('STRATA_TIMELINE',None);validate_launcher_env(env)
        args=list(cfg['args'])
        if any(str(x).startswith('--igc-') or x in ('--dump-mixed','--dump-residual') for x in args):raise ValueError('coherent profile already has an intervention/observation')
        learned=Path(args[args.index('--expert-profile')+1]);learned_hash=digest(learned)
        shutil.copyfile(learned,output/'initial-expert-profile.bin')
        args[args.index('--expert-profile')+1]=str(output/'initial-expert-profile.bin')
        args[args.index('--expert-profile-save')+1]=str(output/'learned-profile.bin')
        args+=['--igc-learned-diagnostic-bundle',str(bundle),'--igc-observation-plan',str(prepared/'plan.bin'),
               '--igc-observation-root',str(output/'raw')]
        cfg.update(exe=str(executable),args=args,log=str(output/'engine.log'))
        _exclusive(output/'config.json',_json_bytes(cfg))
        subprocess.Popen=headless;thread.start();job=current_job()
        vision_cfg=cfg.get('vision')
        if vision_cfg:
            vision_cfg={k:(str(Path(cfg.get('cwd') or '.')/v) if k in ('exe','mmproj','model') and isinstance(v,str) and not os.path.isabs(v) else v) for k,v in vision_cfg.items()}
            vision_log=(output/'vision.log').open('w',encoding='utf-8')
            vision=helper.Vision(vision_cfg,log=vision_log,env=helper.vision_env(cfg,env),lazy=False)
        effort=helper.effort_end_args(cfg,str(executable),tokenizer) or []
        engine=helper.StrataEngine(str(executable),helper.engine_args(cfg)+effort,cwd=cfg.get('cwd'),log=cfg['log'],env=env)
        engine.silence_s=helper.engine_silence_s(cfg);capture_start[0]=time.monotonic()
        durations={0:[],1:[]};first_receipts=[]
        for i,(r,q) in enumerate(zip(plan['records'],manifest['requests'],strict=True)):
            if cancel.is_set():raise TimeoutError('coherent capture cancelled')
            if i==400:_exclusive(output/'ANCHORS.json',_json_bytes(dict(count=400,receipts=list(compact.receipts))))
            request_start[0]=time.monotonic()
            cycle_start=request_start[0]
            try:generated=list(engine.generate(r['tokens'],1,{'temperature':0},cancel))
            finally:
                duration=time.monotonic()-request_start[0];request_start[0]=None
            if failure or not engine.alive() or len([t for t in generated if t is not None])!=1:
                raise RuntimeError('coherent owned request did not complete')
            staged=output/'raw'/f'{i}.bin';raw=_read(staged,limit=8*1024**2)
            receipt=compact.ingest(raw,i,target=q.get('target'))
            if receipt['provenance']['source']!=1:raise ValueError('synthetic source cannot qualify actual Hub observations')
            if i<4:first_receipts.append(receipt)
            completed.append(dict(ordinal=i,owner=receipt['owner'],source_capture_sha256=receipt['source_capture_sha256'],
                                  emitted=[int(t) for t in generated if t is not None],engine_done=engine.last,inference_seconds=duration))
            _exclusive(output/f'request-{i}.json',_json_bytes(completed[-1]))
            # Only this task's just-consumed staging row is removed, after its
            # committed compact receipt. Existing evidence and user files stay.
            if staged.resolve().parent!=(output/'raw').resolve():raise ValueError('coherent staging path escape')
            staged.unlink()
            if i==3:
                first_four_fidelity(first_receipts,output/'compact')
                _exclusive(output/'FIDELITY.json',_json_bytes(dict(passed=True,source='native bound-head bitwise replay plus prehook snapshot',
                    receipts=list(compact.receipts),before_next_anchor=4)))
            role=int(r['role']=='output');cycle_seconds=time.monotonic()-cycle_start
            durations[role].append(cycle_seconds);completed[-1]['cycle_seconds']=cycle_seconds
            if len(durations[role])==4:
                remaining=sum(int(v['role']=='output')==role for v in plan['records'][i+1:])
                if max(durations[role])*remaining>1800-(time.monotonic()-capture_start[0]):
                    raise TimeoutError('coherent fixed-first-four pricing exceeds remaining bound')
            if (i+1)%16==0:print('coherent observations',i+1,'/',len(plan['records']),flush=True)
        if before!={str(p):digest(p) for p in required} or digest(learned)!=learned_hash:
            raise RuntimeError('coherent source/profile changed')
        if any(Path(r['original']).resolve()!=Path(r['resolved']).resolve() for r in verify_sources(sources)):
            raise RuntimeError('coherent current source changed; historical fallback is not live provenance')
        if failure or cancel.is_set():raise RuntimeError('coherent monitor failed; no complete collection')
        compact.finalize()
        summary=dict(success=True,requests=len(completed),seconds=time.monotonic()-started,
            python=sys.version,numpy=numpy.__version__,psutil=psutil.__version__,
            sources=before,source_resolution=resolution,job_initial=job,model=plan['model'],tokenizer=plan['tokenizer'],
            row_timings=completed,first_four_producer_head_fidelity=True,
            full_model_observations=True,result_injection=False,training=False,acceptance64_access=False,
            differences=['task-owned engine/logs/expert-profile outputs','diagnostic observation and teacher-forced prefixes',
                         'direct canonical token requests, not HTTP latency','external time/system-safety bounds; no private cap'])
    finally:
        try:
            if engine is not None:engine.close()
        finally:
            if vision is not None:vision.close()
            if vision_log is not None:vision_log.close()
            stop.set()
            if thread.ident is not None:thread.join(timeout=2)
            if thread.is_alive():failure.append(dict(reason='telemetry monitor did not drain'))
            subprocess.Popen=original_popen
            _exclusive(output/'memory.json',_json_bytes(samples))
            if not compact.closed:_exclusive(output/'INCOMPLETE.json',_json_bytes(dict(completed=len(completed),failure=failure,resume='refused; preserve this archive')))
    remaining=[]
    for child in owner.children(recursive=True):
        try:remaining.append(dict(pid=child.pid,name=child.name()))
        except psutil.Error:pass
    final_job=current_job();_exclusive(output/'job-final.json',_json_bytes(final_job))
    require_clean_finish(output,remaining,failure,samples,summary)
    summary.update(job_final=final_job,owned_children_remaining=remaining)
    _exclusive(output/'report.json',_json_bytes(summary))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('prepared','profile','server','executable','bundle','pack','sources','protocol','output'):p.add_argument('--'+key,type=Path,required=True)
    collect(**vars(p.parse_args()))


if __name__=='__main__':main()
