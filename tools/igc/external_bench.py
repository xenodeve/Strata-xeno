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
from tools.igc.external_score import final_number, question_id, repeating
from tools.igc.hub_contract import validate_launcher_env
from tools.igc.learned_integration import parse_audit
from tools.igc.output_examples import wrapped_prefix
from tools.igc.few_shot import few_shot
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
            rows.append(dict(id=question_id('math',row['problem']),suite='math500',prompt=row['problem'],answer=int(row['answer']),limit=8192))
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
        rows.append(dict(id=question_id('gsm8k',row['question']),suite='gsm8k',prompt=row['question'],
                         answer=int(row['answer'].split('####')[1].strip().replace(',','')),limit=1536))
    return rows


def score(records):
    suites={}
    for r in records:
        s=suites.setdefault(r['suite'],dict(n=0,correct=0,unfinished=0,calls=0,completion_tokens=0,seconds=0.))
        s['n']+=1;s['correct']+=r['correct'];s['unfinished']+=r['final'] is None;s['calls']+=r.get('evaluations') or 0
        s['completion_tokens']+=r['tokens'];s['seconds']+=r['seconds']
    return suites


def run(profile,server,executable,bundle,data,reference,pack,output,arithmetic,gsm,bigbench=0,bridge=False,inline=False,math=0,no_adapt=False,timeline=None,gsm_file='gsm8k-test.jsonl',math_file='math500.jsonl',sidecar=None,whole_context=False,hint=None,no_note=False,effort_medium=False,close_repeat=None,sum_note=None,no_spec=False,no_reply=False,sandbox=False,think_hint=None,sandbox_after=None,sandbox_close=False,sandbox_close_line=False,repeat_note=None,no_think=False,shots=0,sandbox_call=False,stuck_rounds=20):
    validate_launcher_env(os.environ)
    output.mkdir(exist_ok=False)
    cfg=json.loads(profile.read_text(encoding='utf-8-sig'))
    tokenizer=NativeText.from_pack(pack/'tokenizer');prefix,suffix=wrapped_prefix(reference,tokenizer)
    if think_hint:suffix+=think_hint.encode('utf-8')
    if no_think:suffix+=b'\n</think>\n\n'
    if effort_medium:prefix=prefix[prefix.index(b'<|im_start|>user'):]   # the template writes no system turn at medium
    if hint:prefix=prefix.replace(b'<|im_end|>',b' '+hint.encode('utf-8')+b'<|im_end|>',1)
    rows=cases(data,arithmetic,gsm,bigbench,math,gsm_file,math_file)
    if shots:rows=[{**row,'prompt':few_shot(row['prompt'],shots)} if row['suite']=='gsm8k' else row for row in rows]
    (output/'cases.json').write_text(json.dumps(rows,ensure_ascii=False),encoding='utf-8')
    sys.path[:0]=[str(server.parent),str(server.parent.parent)]
    spec=importlib.util.spec_from_file_location('igc_bench_hub',server)
    helper=importlib.util.module_from_spec(spec);sys.modules[spec.name]=helper;spec.loader.exec_module(helper)
    args=list(cfg['args']);shutil.copyfile(args[args.index('--expert-profile')+1],output/'initial-profile.bin')
    args[args.index('--expert-profile')+1]=str(output/'initial-profile.bin')
    args[args.index('--expert-profile-save')+1]=str(output/'learned-profile.bin')
    if bundle:args+=['--igc-learned-diagnostic-bundle',str(bundle)]+(['--igc-inline-exec'] if inline else ['--igc-result-bridge'] if bridge else [])
    if sidecar:args+=['--igc-sidecar',str(sidecar)]
    if no_note:args+=['--igc-no-note']
    if close_repeat is not None:args+=['--igc-close-repeat',str(close_repeat)]
    if sum_note is not None:args+=['--igc-sum-note',str(sum_note)]
    if repeat_note is not None:args+=['--igc-repeat-note',str(repeat_note)]
    if no_reply:args.append('--igc-no-reply')
    if sandbox:args.append('--igc-sandbox')
    if sandbox_after is not None:args+=['--igc-sandbox-after',str(sandbox_after)]
    if sandbox_close:args.append('--igc-sandbox-close')
    if sandbox_close_line:args.append('--igc-sandbox-close-line')
    if sandbox_call:args.append('--igc-sandbox-call')
    if no_adapt:args[args.index('--adapt-swaps')+1]='0';args[args.index('--adapt-secondary')+1]='0'
    if no_spec:
        args[args.index('--spec')+1]='2'
        mtp=args.index('--mtp');del args[mtp:mtp+2]
    cfg.update(exe=str(executable),args=args,log=str(output/'engine.log'))
    env=helper.child_env(cfg);env.pop('STRATA_TIMELINE',None);validate_launcher_env(env)
    if timeline:env['STRATA_TIMELINE']=str(timeline)
    original=subprocess.Popen
    subprocess.Popen=lambda *a,**k:original(*a,**{**k,'creationflags':k.get('creationflags',0)|0x08000000})
    cancel=threading.Event();stop=threading.Event();active=[None];failure=[]
    def monitor():
        while not stop.wait(1):
            state=system_snapshot()
            if state['physical_available_bytes']<1024**3 or (active[0] and time.monotonic()-active[0]>(7200 if whole_context else 600)):
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
            limit=262144-len(ids)-64 if whole_context else row['limit'];tokens=[];stuck=False
            for t in engine.generate(ids,limit,{'temperature':0},cancel):
                if t is None:continue
                tokens.append(int(t))
                if len(tokens)%64==0 and repeating(tokens,rounds=stuck_rounds):stuck=True;break
            seconds=time.monotonic()-start;active[0]=None
            text=tokenizer.decode_bytes(tokens).decode('utf-8',errors='replace')
            if no_think:text='</think>\n\n'+text   # the close is in the prompt; the readers look for it here
            final=final_number(text)
            records.append(dict(id=row['id'],suite=row['suite'],answer=row['answer'],final=final,correct=final==row['answer'],
                                tokens=len(tokens),seconds=seconds,finish='stuck' if stuck else engine.last.get('finish'),prompt_tokens=len(ids),text=text))
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
    parser.add_argument('--whole-context',action='store_true');parser.add_argument('--hint',help='added to the system message');parser.add_argument('--no-note',action='store_true');parser.add_argument('--effort-medium',action='store_true');parser.add_argument('--bundle',type=Path);parser.add_argument('--timeline',type=Path);parser.add_argument('--sidecar',type=Path);parser.add_argument('--gsm-file',default='gsm8k-test.jsonl');parser.add_argument('--math-file',default='math500.jsonl');parser.add_argument('--arithmetic',type=int,default=40);parser.add_argument('--gsm',type=int,default=40);parser.add_argument('--bigbench',type=int,default=0);parser.add_argument('--bridge',action='store_true');parser.add_argument('--inline',action='store_true');parser.add_argument('--math',type=int,default=0);parser.add_argument('--no-adapt',action='store_true')
    parser.add_argument('--close-repeat',type=int,help='close thinking at this repeat, 0 never')
    parser.add_argument('--sum-note',type=int,help='note on a top-level sum: 0 open, 1 stop, 2 none')
    parser.add_argument('--repeat-note',type=int,help='note on a repeat: 0 stop checking, 1 no instruction, 2 next step')
    parser.add_argument('--no-spec',action='store_true',help='spec 2 and no MTP drafts, for a repeatable run')
    parser.add_argument('--no-reply',action='store_true',help='no inline value after the thinking closes')
    parser.add_argument('--sandbox',action='store_true',help='run a python block in the reasoning')
    parser.add_argument('--think-hint',help='written after the opening of the thinking')
    parser.add_argument('--sandbox-after',type=int,help='open a block after this many tokens of reasoning')
    parser.add_argument('--sandbox-close',action='store_true',help='close the thinking once a program has printed')
    parser.add_argument('--sandbox-close-line',action='store_true',help='the model\'s own line before the close')
    parser.add_argument('--sandbox-call',action='store_true',help='open a print( after a block that printed nothing')
    parser.add_argument('--stuck-rounds',type=int,default=20,help='rounds of a cycle before the bench gives up; 40 leaves a loop of up to 3 tokens to the engine, which closes at 64')
    parser.add_argument('--no-think',action='store_true',help='close the thinking in the prompt')
    parser.add_argument('--shots',type=int,default=0,help='few-shot exemplars before each GSM8K question')
    run(**vars(parser.parse_args()))
