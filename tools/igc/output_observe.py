"""Offline native teacher-forced output positions, not a trained adapter or benchmark."""
import argparse
import ctypes
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import threading
import time
import numpy as np
import psutil
from tools.igc.output_corpus import dataset
from tools.igc.observation import load_observation,_read
from tools.igc.provenance import verify_sources
from tools.qwenputer.memory import system_snapshot,process_snapshot,ExtendedLimit,job_fields
from tools.qwenputer.tokenizer import NativeText


def digest(path):
    with Path(path).open('rb') as source:return hashlib.file_digest(source,'sha256').hexdigest()


def position_examples(rows,tokenizer,prefix,suffix):
    examples=[];seen=set()
    for invocation,row in enumerate(rows,1):
        if row['id'] in seen:raise ValueError('duplicate output sample')
        seen.add(row['id'])
        base=tokenizer.encode((prefix+row['prompt'].encode('utf-8')+suffix).decode('utf-8'))
        completion=tokenizer.encode(row['completion'])
        if not 1<=len(completion)<=64 or not 2<=len(base)<=256:raise ValueError('output sequence bounds')
        anchor=len(base)-1
        for j,target in enumerate(completion):
            tokens=base+completion[:j]
            if len(tokens)>256 or len(examples)>=1024:raise ValueError('output observation bound')
            piece=tokenizer.decode_bytes([target])
            numeric=bool(piece) and all(c in b'-0123456789' for c in piece)
            examples.append(dict(sample=row['id'],split=row['split'],invocation=invocation,
                                 invocation_position=anchor,position=len(tokens)-1,output_offset=j,
                                 target=target,numeric_target=numeric,ids=tokens))
    return examples


def wrapped_prefix(reference,tokenizer):
    original=json.loads(_read(reference/'corpus.json',limit=1024**2))['rows']
    prefix=suffix=None
    for i,row in enumerate(original):
        observed=load_observation(reference,i);tokens=observed['prefix_tokens'].tolist()
        raw=tokenizer.decode_bytes(tokens);prompt=row['prompt'].encode('utf-8')
        if raw.count(prompt)!=1:raise ValueError('output wrapper occurrence')
        at=raw.index(prompt)
        if prefix is None:prefix,suffix=raw[:at],raw[at+len(prompt):]
        if raw!=prefix+prompt+suffix or tokenizer.encode(raw.decode('utf-8'))!=tokens:
            raise ValueError('output native wrapper fidelity')
    return prefix,suffix


def operand(value):
    # Offline typed request/independent label preparation only, never inference.
    digits=list(map(int,reversed(str(abs(value)))))
    if not 1<=len(digits)<=62:raise ValueError('output pilot operand range')
    return bytes([len(digits),int(value<0),*digits])+bytes(62-len(digits))


def checked_results(rows,examples,executable,output,env):
    anchors={r['sample']:(r['invocation'],r['invocation_position']) for r in examples}
    request=struct.Struct('<IIIIQQ64s64s');result=struct.Struct('<IIQQ64s')
    source=output/'exact-requests.bin';target=output/'exact-results.bin'
    source.write_bytes(b''.join(request.pack(1,1,row['operation'],0,*anchors[row['id']],operand(row['lhs']),operand(row['rhs'])) for row in rows))
    with (output/'exact.stdout').open('w') as so,(output/'exact.stderr').open('w') as se:
        process=subprocess.run([str(executable),str(source),str(target)],env=env,creationflags=0x08000000,
                               stdout=so,stderr=se,timeout=30)
    if process.returncode:raise RuntimeError('output pilot native exact result run failed')
    raw=_read(target,size=len(rows)*result.size,limit=64*result.size)
    values={}
    for i,row in enumerate(rows):
        version,status,invocation,position,value=result.unpack_from(raw,i*result.size)
        if (version,status,invocation,position,value)!=(1,0,*anchors[row['id']],operand(row['answer'])):
            raise ValueError('output native result disagrees with independent oracle')
        values[row['id']]=raw[i*result.size:(i+1)*result.size].hex()
    return values


def current_job():
    info=ExtendedLimit();query=ctypes.WinDLL('kernel32',use_last_error=True).QueryInformationJobObject
    query.argtypes=[ctypes.c_void_p,ctypes.c_int,ctypes.c_void_p,ctypes.c_uint32,ctypes.c_void_p]
    if not query(None,9,ctypes.byref(info),ctypes.sizeof(info),None):raise ctypes.WinError(ctypes.get_last_error())
    value=job_fields(info)
    if value['limit_flags']&0x300:raise RuntimeError('output collection must not have a private/job commit cap')
    return value


def collect(profile,server,executable,exact_executable,pack,reference,sources,protocol,output):
    started=time.monotonic();resolution=verify_sources(sources)
    pinned={Path(r['path']).resolve() for r in json.loads(_read(sources,limit=256*1024))['files']}
    if any(path.resolve() not in pinned for path in (profile,server,executable,pack/'index.txt')):
        raise ValueError('output collector source not bound to verified identity')
    before={str(p):digest(p) for p in (profile,server,executable,exact_executable,protocol)}
    tokenizer=NativeText.from_pack(pack/'tokenizer');prefix,suffix=wrapped_prefix(reference,tokenizer)
    corpus=dataset();examples=position_examples(corpus['rows'],tokenizer,prefix,suffix)
    system=system_snapshot()
    if (system['physical_available_bytes']<24*1024**3 or system['system_commit_available_bytes']<55_000_000_000+4*1024**3
            or shutil.disk_usage(output.parent).free<4*1024**3):
        raise RuntimeError('output collection diagnostic/Hub admission; no model started')
    for process in psutil.process_iter(['name']):
        if (process.info['name'] or '').lower() in ('strata.exe','strata-vision.exe'):
            raise RuntimeError('another engine is active; no duplicate output collection')
    output.mkdir(exist_ok=False)
    (output/'corpus.json').write_text(json.dumps(corpus,ensure_ascii=False,indent=2),encoding='utf-8')
    (output/'planned-positions.json').write_text(json.dumps(examples,indent=2),encoding='utf-8')
    cfg=json.loads(profile.read_text(encoding='utf-8-sig'));cfg['exe']=str(executable)
    args=list(cfg['args']);learned=Path(args[args.index('--expert-profile')+1])
    learned_hash=digest(learned);shutil.copyfile(learned,output/'initial-expert-profile.bin')
    args[args.index('--expert-profile')+1]=str(output/'initial-expert-profile.bin')
    args[args.index('--expert-profile-save')+1]=str(output/'learned-profile.bin')
    args+=['--dump-mixed',str(output/'mixed')]
    cfg['args']=args;cfg['log']=str(output/'engine.log')
    cfg.setdefault('env',{})['STRATA_DUMP_FIRST_LOGITS']=str(output/'logits')
    (output/'config.json').write_text(json.dumps(cfg,indent=2),encoding='utf-8')
    # Load the already deployed project helper, without running its HTTP main.
    sys.path[:0]=[str(server.parent),str(server.parent.parent)]
    spec=importlib.util.spec_from_file_location('igc_pinned_hub_server',server)
    helper=importlib.util.module_from_spec(spec);sys.modules[spec.name]=helper;spec.loader.exec_module(helper)
    env=helper.child_env(cfg);env.pop('STRATA_TIMELINE',None) # This is a dataset, not a timing run.
    if cfg.get('cjk_guard') or helper.layer_split_of(cfg) or cfg.get('vision') or cfg.get('replicas'):
        raise ValueError('output pilot only qualified for the selected non-replica D2x profile')
    effort=helper.effort_end_args(cfg,str(executable),tokenizer) or []
    engine_arguments=helper.engine_args(cfg)+effort
    results=checked_results(corpus['rows'],examples,exact_executable,output,env)
    record=[];samples=[];stop=threading.Event();cancel=threading.Event();failure=[];active_request=[None]
    engine=None;root=psutil.Process(os.getpid());original_popen=subprocess.Popen
    def headless(*args,**kwargs):
        kwargs['creationflags']=kwargs.get('creationflags',0)|0x08000000
        return original_popen(*args,**kwargs)
    def monitor():
        while not stop.wait(.2):
            system=system_snapshot();owned=[]
            for child in root.children(recursive=True):
                try:
                    if Path(child.exe()).resolve()==executable.resolve():owned.append(process_snapshot(child.pid))
                except (psutil.Error,OSError):pass
            samples.append(dict(seconds=time.monotonic()-started,system=system,processes=owned))
            if (time.monotonic()-started>900 or (active_request[0] is not None and time.monotonic()-active_request[0]>30)
                    or system['system_commit_available_bytes']<2*1024**3
                    or system['physical_available_bytes']<1024**3):
                failure.append('owned collection deadline/system emergency floor');cancel.set()
                for child in root.children(recursive=True):
                    try:
                        if Path(child.exe()).resolve()==executable.resolve():child.terminate()
                    except (psutil.Error,OSError):pass
                return
    thread=threading.Thread(target=monitor,daemon=True);thread.start()
    subprocess.Popen=headless
    try:
        limits=current_job()
        engine=helper.StrataEngine(str(executable),engine_arguments,cwd=cfg.get('cwd'),log=cfg['log'],env=env)
        engine.silence_s=helper.engine_silence_s(cfg)
        for index,example in enumerate(examples):
            if cancel.is_set() or time.monotonic()-started>900:raise TimeoutError('output collection cancelled/deadline')
            active_request[0]=time.monotonic()
            try:generated=list(engine.generate(example['ids'],1,{'temperature':0},cancel))
            finally:active_request[0]=None
            if failure or not engine.alive():raise RuntimeError('output collection owned engine failed')
            observed=load_observation(output,index)
            if observed['prefix_tokens'].tolist()!=example['ids'] or observed['position']!=example['position']:
                raise ValueError('output teacher-forced native boundary mismatch')
            metadata={key:value for key,value in observed.items() if key not in ('mixed','logits','prefix_tokens')}
            record.append(dict(index=index,**example,observation=metadata,checked_result_hex=results[example['sample']],
                               unadapted_next_token=[int(t) for t in generated if t is not None],engine_done=engine.last))
            (output/'records.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
            if (index+1)%16==0:print('output positions',index+1,'/',len(examples),'seconds',round(time.monotonic()-started,2),flush=True)
    finally:
        if engine is not None:engine.close()
        stop.set();thread.join(timeout=2);subprocess.Popen=original_popen
        (output/'memory.json').write_text(json.dumps(samples,indent=2),encoding='utf-8')
        (output/'incomplete.json').write_text(json.dumps(dict(completed=len(record),planned=len(examples),failure=failure),indent=2),encoding='utf-8')
    if (failure or before!={str(p):digest(p) for p in (profile,server,executable,exact_executable,protocol)}
            or digest(learned)!=learned_hash or len(record)!=len(examples)):
        raise RuntimeError('output source/owned completion validation failed')
    verify_sources(sources)
    report=dict(success=True,kind='native teacher-forced output positions; no trained result-use or speed claim',
                positions=len(record),samples=len(corpus['rows']),rows_sha256=corpus['rows_sha256'],job=limits,
                records_sha256=digest(output/'records.json'),source_manifest_sha256=digest(sources),
                source_resolution=resolution,protocol_sha256=before[str(protocol)],source_sha256=before,
                tokenizer_binding_sha256=tokenizer.binding_sha256,seconds=time.monotonic()-started,
                original_learned_sha256=learned_hash,hard_private_commit_cap=None,
                differences=['direct canonical token requests/teacher forcing, not HTTP latency',
                             'isolated main-based candidate, task-owned logs/expert-profile output',
                             'opt-in first-window mixed/logit capture','bounded headless owned tree/emergency cleanup'])
    (output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('output positions complete',len(record),'seconds',report['seconds'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('profile','server','executable','exact-executable','pack','reference','sources','protocol','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();collect(**vars(args))
