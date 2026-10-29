"""Controlled native I3 generation; oracle fixture, not learned I2 or a benchmark."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import threading
import time
import numpy as np
import psutil
from tools.igc.output_data import load_dataset
from tools.igc.output_examples import wrapped_prefix
from tools.igc.output_observe import validate_launcher_env,current_job
from tools.qwenputer.memory import process_snapshot,system_snapshot


def digest(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def final_integer(text):
    # Frozen pilot rubric: reasoning must close, and the final section contains
    # exactly one integer. A truncated thinking prefix never counts as an answer.
    if '</think>' not in text:return None
    final=text.rsplit('</think>',1)[1].replace('<|im_end|>','').strip()
    numbers=re.findall(r'(?<![\w.])-?\d+(?!\w|\.\d)',final)
    return int(numbers[0]) if len(numbers)==1 else None


def run(profile,server,candidate,baseline,sources,capture,reference,pack,capture_protocol,protocol,artifact,fixture,output,smoke=False,parity_only=False):
    validate_launcher_env(os.environ)
    if any((p.info['name'] or '').lower().startswith('strata') for p in psutil.process_iter(['name'])):
        raise RuntimeError('another Strata process is active; no model launched')
    preflight=system_snapshot()
    if (preflight['physical_available_bytes']<24*1024**3 or preflight['system_commit_available_bytes']<55_000_000_000+4*1024**3):
        raise RuntimeError('diagnostic system-headroom admission; not a Hub memory cap')
    data=load_dataset(capture,reference,pack,sources,capture_protocol)
    tokenizer=data['tokenizer'];rows={r['id']:r for r in data['rows']}
    cases=[dict(id=r['sample'],ids=r['ids'],expected=rows[r['sample']]['answer'],split=r['split'],matched=True)
           for r in data['records'] if r['output_offset']==0]
    if smoke:cases=[r for r in cases if r['id'] in ('output-development-op2-00','output-development-op3-01')]
    prefix,suffix=wrapped_prefix(reference,tokenizer)
    for i,prompt in enumerate(('Reply with only the word READY.','ตอบเพียงคำว่า พร้อม')):
        cases.append(dict(id=f'unmatched-{i}',ids=tokenizer.encode((prefix+prompt.encode('utf-8')+suffix).decode('utf-8')),
                          expected=None,split='unmatched',matched=False))
    cfg=json.loads(profile.read_text(encoding='utf-8-sig'))
    if cfg.get('replicas') or cfg.get('cjk_guard'):raise ValueError('unsupported diagnostic profile combination')
    vision_cfg=({key:(os.path.abspath(os.path.join(cfg.get('cwd') or '.',value))
        if key in ('exe','model','mmproj') and isinstance(value,str) and not os.path.isabs(value) else value)
        for key,value in cfg['vision'].items()} if cfg.get('vision') else None)
    source_paths=[profile,server,candidate,baseline,sources,protocol,capture_protocol,artifact,fixture,Path(__file__)]
    if vision_cfg:source_paths.extend(Path(vision_cfg[k]) for k in ('exe','model','mmproj'))
    before={str(path):digest(path) for path in source_paths}
    output.mkdir(exist_ok=False)
    (output/'inputs.json').write_text(json.dumps(dict(preflight=preflight,job=current_job(),cases=cases,source_sha256=before,
        original_collection=data['collection_report_sha256'],smoke=smoke,parity_only=parity_only),ensure_ascii=False,indent=2),encoding='utf-8')
    sys.path[:0]=[str(server.parent),str(server.parent.parent)]
    spec=importlib.util.spec_from_file_location('igc_output_integration_hub',server)
    helper=importlib.util.module_from_spec(spec);sys.modules[spec.name]=helper;spec.loader.exec_module(helper)
    if helper.layer_split_of(cfg):raise ValueError('layer split is not the selected D2x expert offload')
    root=psutil.Process(os.getpid());owned={candidate.resolve(),baseline.resolve()}
    if vision_cfg:owned.add(Path(vision_cfg['exe']).resolve())
    started=time.monotonic();stop=threading.Event();cancel=threading.Event();failure=[];memory=[];active=[None];current_arm=['startup']
    def monitor():
        while not stop.wait(.2):
            system=system_snapshot();children=[]
            for child in root.children(recursive=True):
                try:
                    if Path(child.exe()).resolve() in owned:children.append(process_snapshot(child.pid))
                except (OSError,psutil.Error):pass
            memory.append(dict(seconds=time.monotonic()-started,arm=current_arm[0],system=system,processes=children))
            if (time.monotonic()-started>900 or (active[0] is not None and time.monotonic()-active[0]>30)
                    or system['system_commit_available_bytes']<2*1024**3 or system['physical_available_bytes']<1024**3):
                failure.append(dict(reason='diagnostic deadline/system emergency floor',system=system,arm=current_arm[0]));cancel.set()
                for child in root.children(recursive=True):
                    try:
                        if Path(child.exe()).resolve() in owned:child.terminate()
                    except (OSError,psutil.Error):pass
                return
    thread=threading.Thread(target=monitor,daemon=True);thread.start()
    popen=subprocess.Popen
    def headless(*args,**kwargs):kwargs['creationflags']=kwargs.get('creationflags',0)|0x08000000;return popen(*args,**kwargs)
    subprocess.Popen=headless;all_records={}
    try:
        arms=([('prehook',baseline,False),('candidate_off',candidate,False),
               ('candidate_off_repeat',candidate,False),('prehook_repeat',baseline,False)] if parity_only else
              [('prehook',baseline,False),('candidate_off',candidate,False),('candidate_on',candidate,True)])
        for arm,exe,enabled in arms:
            current_arm[0]=arm;folder=output/arm;folder.mkdir()
            local=json.loads(json.dumps(cfg));local['exe']=str(exe);local['log']=str(folder/'engine.log')
            args=list(local['args']);original=Path(args[args.index('--expert-profile')+1]);initial_hash=digest(original)
            shutil.copyfile(original,folder/'initial-profile.bin')
            args[args.index('--expert-profile')+1]=str(folder/'initial-profile.bin')
            args[args.index('--expert-profile-save')+1]=str(folder/'learned-profile.bin')
            if enabled:args+=['--igc-output-diagnostic-artifact',str(artifact),'--igc-output-diagnostic-fixture',str(fixture)]
            local['args']=args
            env=helper.child_env(local);env.pop('STRATA_TIMELINE',None);env.pop('STRATA_DUMP_FIRST_LOGITS',None)
            validate_launcher_env(env)
            (folder/'config.json').write_text(json.dumps(local,indent=2),encoding='utf-8')
            engine=vision=vision_log=None;records=[]
            try:
                if vision_cfg:
                    vision_log=(folder/'vision.log').open('w',encoding='utf-8')
                    vision=helper.Vision(vision_cfg,log=vision_log,env=helper.vision_env(local,env),lazy=False)
                arguments=helper.engine_args(local)+(helper.effort_end_args(local,str(exe),tokenizer) or [])
                engine=helper.StrataEngine(str(exe),arguments,cwd=local.get('cwd'),log=local['log'],env=env)
                engine.silence_s=helper.engine_silence_s(local)
                for case in cases:
                    if cancel.is_set():raise RuntimeError('controlled generation cancelled')
                    active[0]=time.monotonic()
                    try:generated=[int(t) for t in engine.generate(case['ids'],96,{'temperature':0},cancel) if t is not None]
                    finally:active[0]=None
                    if failure or not engine.alive():raise RuntimeError('controlled engine failed')
                    text=tokenizer.decode_bytes(generated).decode('utf-8',errors='replace');actual=final_integer(text)
                    records.append(dict(id=case['id'],split=case['split'],matched=case['matched'],expected=case['expected'],
                        actual=actual,numeric_match=actual==case['expected'] if case['expected'] is not None else None,
                        tokens=generated,text=text,done=engine.last))
                    (folder/'records.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')
                    print('I3 generation',arm,case['id'],'expected',case['expected'],'actual',actual,'finish',engine.last.get('finish'),flush=True)
            finally:
                if engine is not None:engine.close()
                if vision is not None:vision.close()
                if vision_log is not None:vision_log.close()
            if digest(original)!=initial_hash:raise RuntimeError('served expert profile changed')
            all_records[arm]=records
    finally:
        stop.set();thread.join(timeout=2);subprocess.Popen=popen
        (output/'memory.json').write_text(json.dumps(memory,indent=2),encoding='utf-8')
    if before!={str(path):digest(path) for path in source_paths}:raise RuntimeError('controlled source changed during run')
    by_id={arm:{r['id']:r for r in rows} for arm,rows in all_records.items()}
    off_equal=all(by_id['prehook'][c['id']]['tokens']==by_id['candidate_off'][c['id']]['tokens'] for c in cases)
    unmatched_equal=(all(by_id['candidate_off'][c['id']]['tokens']==by_id['candidate_on'][c['id']]['tokens'] for c in cases if not c['matched']) if not parity_only else None)
    comparisons={}
    if parity_only:
        for name,a,b in [('prehook_self','prehook','prehook_repeat'),('candidate_self','candidate_off','candidate_off_repeat'),
                         ('cross_first','prehook','candidate_off'),('cross_second','prehook_repeat','candidate_off_repeat')]:
            comparisons[name]={c['id']:by_id[a][c['id']]['tokens']==by_id[b][c['id']]['tokens'] for c in cases}
    report=dict(run_completed=True,kind='controlled native output generation; no learned I2 or performance claim',
        smoke=smoke,cases=len(cases),off_raw_token_parity=off_equal,unmatched_raw_token_parity=unmatched_equal,
        parity_only=parity_only,repeat_comparisons=comparisons,final_answer_rubric='single-integer final format, not mathematical correctness',
        same_executable_control=before[str(candidate)]==before[str(baseline)],
        arms={arm:{split:dict(n=len(selected),numeric_correct=sum(r['numeric_match'] is True for r in selected),
            limit=sum(r['done'].get('finish')=='length' for r in selected))
            for split in dict.fromkeys(r['split'] for r in rows)
            for selected in [[r for r in rows if r['split']==split]]} for arm,rows in all_records.items()},
        source_sha256=before,elapsed_seconds=time.monotonic()-started,hard_private_commit_cap=None,
        differences=['oracle fixture only on candidate_on','fresh content hash before enabled boot',
                     'diagnostic per-window status fence','direct native protocol, not HTTP benchmark',
                     'isolated paths and bounded task cleanup; no served settings changed'])
    (output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('profile','server','candidate','baseline','sources','capture','reference','pack','capture-protocol','protocol','artifact','fixture','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--parity-only',action='store_true')
    run(**vars(parser.parse_args()))
