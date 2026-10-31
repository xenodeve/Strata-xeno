"""External benchmark, Gyro-M with and without the learned bundle. Fixed first-N cases, greedy, direct engine protocol."""
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
from tools.igc.hub_contract import validate_launcher_env
from tools.igc.learned_integration import parse_audit
from tools.igc.output_examples import wrapped_prefix
from tools.qwenputer.memory import system_snapshot
from tools.qwenputer.tokenizer import NativeText

ARITHMETIC=('two_digit_addition','two_digit_subtraction','two_digit_multiplication')
BIGBENCH=('3d_addition','3d_subtraction','3d_multiplication','5d_addition','5d_subtraction','5d_multiplication')


def cases(data,arithmetic,gsm,bigbench=0,math=0,gsm_file='gsm8k-test.jsonl',math_file='math500.jsonl'):
    rows=[]
    if math:
        problems=[json.loads(line) for line in (data/math_file).read_text(encoding='utf-8').splitlines()]
        problems=[row for row in problems if re.fullmatch(r'-?\d+',str(row['answer']))]
        for i,row in enumerate(problems[:math]):
            rows.append(dict(id=f'math-{i}',suite='math500',prompt=row['problem'],answer=int(row['answer']),limit=8192))
    for name in BIGBENCH if bigbench else ():
        for i,row in enumerate(json.loads((data/'bigbench'/f'{name}.json').read_text(encoding='utf-8'))['examples'][:bigbench]):
            rows.append(dict(id=f'{name}-{i}',suite=name,prompt=row['input'],answer=int(row['target']),limit=2048))
    for name in ARITHMETIC:
        lines=(data/f'{name}.jsonl').read_text(encoding='utf-8').splitlines()[:arithmetic]
        for i,line in enumerate(lines):
            row=json.loads(line)
            rows.append(dict(id=f'{name}-{i}',suite=name,prompt=row['context'].strip().removeprefix('Q:').removesuffix('A:').strip(),
                             answer=int(row['completion'].strip()),limit=512))
    for i,line in enumerate((data/gsm_file).read_text(encoding='utf-8').splitlines()[:gsm]):
        row=json.loads(line)
        rows.append(dict(id=f'gsm8k-{i}',suite='gsm8k',prompt=row['question'],
                         answer=int(row['answer'].split('####')[1].strip().replace(',','')),limit=1536))
    return rows


def final_number(text):
    """Last integer in the answer after the thinking block; None when the answer never started."""
    tail=text.rsplit('</think>',1)
    if len(tail)<2:return None
    numbers=re.findall(r'-?\d+(?:\.\d+)?',tail[1].replace(',',''))
    return float(numbers[-1]) if numbers else None


def score(records):
    suites={}
    for r in records:
        s=suites.setdefault(r['suite'],dict(n=0,correct=0,unfinished=0,calls=0,completion_tokens=0,seconds=0.))
        s['n']+=1;s['correct']+=r['correct'];s['unfinished']+=r['final'] is None;s['calls']+=r.get('evaluations') or 0
        s['completion_tokens']+=r['tokens'];s['seconds']+=r['seconds']
    return suites


def run(profile,server,executable,bundle,data,reference,pack,output,arithmetic,gsm,bigbench=0,bridge=False,inline=False,math=0,no_adapt=False,timeline=None,gsm_file='gsm8k-test.jsonl',math_file='math500.jsonl',sidecar=None):
    validate_launcher_env(os.environ)
    output.mkdir(exist_ok=False)
    cfg=json.loads(profile.read_text(encoding='utf-8-sig'))
    tokenizer=NativeText.from_pack(pack/'tokenizer');prefix,suffix=wrapped_prefix(reference,tokenizer)
    rows=cases(data,arithmetic,gsm,bigbench,math,gsm_file,math_file)
    (output/'cases.json').write_text(json.dumps(rows,ensure_ascii=False),encoding='utf-8')
    sys.path[:0]=[str(server.parent),str(server.parent.parent)]
    spec=importlib.util.spec_from_file_location('igc_bench_hub',server)
    helper=importlib.util.module_from_spec(spec);sys.modules[spec.name]=helper;spec.loader.exec_module(helper)
    args=list(cfg['args']);shutil.copyfile(args[args.index('--expert-profile')+1],output/'initial-profile.bin')
    args[args.index('--expert-profile')+1]=str(output/'initial-profile.bin')
    args[args.index('--expert-profile-save')+1]=str(output/'learned-profile.bin')
    if bundle:args+=['--igc-learned-diagnostic-bundle',str(bundle)]+(['--igc-inline-exec'] if inline else ['--igc-result-bridge'] if bridge else [])
    if sidecar:args+=['--igc-sidecar',str(sidecar)]
    if no_adapt:args[args.index('--adapt-swaps')+1]='0';args[args.index('--adapt-secondary')+1]='0'
    cfg.update(exe=str(executable),args=args,log=str(output/'engine.log'))
    env=helper.child_env(cfg);env.pop('STRATA_TIMELINE',None);validate_launcher_env(env)
    if timeline:env['STRATA_TIMELINE']=str(timeline)
    original=subprocess.Popen
    subprocess.Popen=lambda *a,**k:original(*a,**{**k,'creationflags':k.get('creationflags',0)|0x08000000})
    cancel=threading.Event();stop=threading.Event();active=[None];failure=[]
    def monitor():
        while not stop.wait(1):
            state=system_snapshot()
            if state['physical_available_bytes']<1024**3 or (active[0] and time.monotonic()-active[0]>600):
                failure.append(dict(reason='time or physical bound',system=state));cancel.set();return
    thread=threading.Thread(target=monitor,daemon=True);thread.start()
    engine=None;records=[]
    try:
        engine=helper.StrataEngine(str(executable),helper.engine_args(cfg)+(helper.effort_end_args(cfg,str(executable),tokenizer) or []),
                                   cwd=cfg.get('cwd'),log=cfg['log'],env=env)
        engine.silence_s=helper.engine_silence_s(cfg)
        for row in rows:
            if cancel.is_set():raise RuntimeError('bench stopped before request')
            ids=tokenizer.encode((prefix+row['prompt'].encode('utf-8')+suffix).decode('utf-8'))
            active[0]=start=time.monotonic()
            tokens=[int(t) for t in engine.generate(ids,row['limit'],{'temperature':0},cancel) if t is not None]
            seconds=time.monotonic()-start;active[0]=None
            text=tokenizer.decode_bytes(tokens).decode('utf-8',errors='replace');final=final_number(text)
            records.append(dict(id=row['id'],suite=row['suite'],answer=row['answer'],final=final,correct=final==row['answer'],
                                tokens=len(tokens),seconds=seconds,finish=engine.last.get('finish'),prompt_tokens=len(ids),text=text))
            (output/'records.json').write_text(json.dumps(records,ensure_ascii=False),encoding='utf-8')
            print(row['id'],row['answer'],final,len(tokens),round(seconds,1),flush=True)
    finally:
        if engine is not None:engine.close()
        stop.set();thread.join(timeout=3);subprocess.Popen=original
    if bundle:
        lines=(output/'engine.log').read_text(encoding='utf-8',errors='replace')
        audits=[parse_audit(bytes.fromhex(h)) for h in re.findall(r'strata IGC learned audit: head_status=\d+ record=([0-9a-f]{576})',lines)]
        if len(audits)!=len(records):raise ValueError('incomplete audit pairing')
        for record,audit in zip(records,audits,strict=True):
            record.update(evaluations=audit['evaluations'],request=audit['request'],result=audit['result'],status=audit['status'])
    report=dict(kind='external comparison, fixed first-N subset, greedy; not published-protocol scores',arm='on' if bundle else 'off',
                executable_sha256=hashlib.sha256(executable.read_bytes()).hexdigest(),suites=score(records),failure=failure)
    (output/'records.json').write_text(json.dumps(records,ensure_ascii=False),encoding='utf-8')
    (output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('profile','server','executable','data','reference','pack','output'):parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--bundle',type=Path);parser.add_argument('--timeline',type=Path);parser.add_argument('--sidecar',type=Path);parser.add_argument('--gsm-file',default='gsm8k-test.jsonl');parser.add_argument('--math-file',default='math500.jsonl');parser.add_argument('--arithmetic',type=int,default=40);parser.add_argument('--gsm',type=int,default=40);parser.add_argument('--bigbench',type=int,default=0);parser.add_argument('--bridge',action='store_true');parser.add_argument('--inline',action='store_true');parser.add_argument('--math',type=int,default=0);parser.add_argument('--no-adapt',action='store_true')
    run(**vars(parser.parse_args()))
