"""Final and attention observations for readiness scope: GSM8K train negatives, worded arithmetic positives past the tested rows."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import numpy as np
from tools.igc.coherent_observation import encode_plan,decode_capture
from tools.igc.external_bench import ARITHMETIC
from tools.igc.hub_contract import validate_launcher_env
from tools.igc.observation import _read
from tools.igc.output_examples import wrapped_prefix
from tools.qwenputer.tokenizer import NativeText

MODEL='7aa36d15131d255179ab6c87fa9c9f81a0b541fc9e81e959d4c3e29eb20feb5f'


def answer(prompt):
    a,word,b=prompt.removeprefix('What is ').removesuffix('?').split()
    return int(a)+int(b) if word=='plus' else int(a)-int(b) if word=='minus' else int(a)*int(b)


def rows(data,tokenizer,prefix,suffix,count,offset=1000):
    out=[]
    for name in ARITHMETIC:
        for i,line in enumerate((data/f'{name}.jsonl').read_text(encoding='utf-8').splitlines()[offset:offset+count]):
            row=json.loads(line);prompt=row['context'].strip().removeprefix('Q:').removesuffix('A:').strip()
            out.append(dict(id=f'{name}-{offset+i}',suite=name,ready=1,prompt=prompt))
    negatives=0
    for i,line in enumerate((data/'gsm8k-train.jsonl').read_text(encoding='utf-8').splitlines()):
        if negatives==3*count:break
        prompt=json.loads(line)['question']
        if len(tokenizer.encode((prefix+prompt.encode('utf-8')+suffix).decode('utf-8')))<=240:
            out.append(dict(id=f'gsm8k-train-{i}',suite='gsm8k-train',ready=0,prompt=prompt));negatives+=1
    return out


def run(profile,server,executable,bundle,data,reference,pack,output,count,outputs=False,offset=1000):
    validate_launcher_env(os.environ)
    output.mkdir(exist_ok=False)
    cfg=json.loads(profile.read_text(encoding='utf-8-sig'))
    tokenizer=NativeText.from_pack(pack/'tokenizer');prefix,suffix=wrapped_prefix(reference,tokenizer)
    cases=rows(data,tokenizer,prefix,suffix,count,offset)
    ids=[tokenizer.encode((prefix+c['prompt'].encode('utf-8')+suffix).decode('utf-8')) for c in cases]
    records=[dict(role='anchor',base=i,step=0,tokens=t) for i,t in enumerate(ids)];targets=[None]*len(ids)
    if outputs:
        keep=[i for i,c in enumerate(cases) if c['ready']];cases=[cases[i] for i in keep];ids=[ids[i] for i in keep]
        records=[dict(role='anchor',base=i,step=0,tokens=t) for i,t in enumerate(ids)];targets=[None]*len(ids)
        for base,c in enumerate(cases):
            c['answer']=answer(c['prompt']);completion=tokenizer.encode(f"The result is {c['answer']}.\n</think>\n\n{c['answer']}<|im_end|>")
            targets[base]=completion[0]
            for j,target in enumerate(completion[1:],1):
                records.append(dict(role='output',base=base,step=j,tokens=ids[base]+completion[:j]));targets.append(target)
    ids=[r['tokens'] for r in records]
    plan=encode_plan(dict(model=MODEL,tokenizer=tokenizer.binding_sha256,records=records))
    (output/'targets.json').write_text(json.dumps(dict(targets=targets,bases=[r['base'] for r in records])),encoding='utf-8')
    (output/'plan.bin').write_bytes(plan);(output/'cases.json').write_text(json.dumps(cases,ensure_ascii=False),encoding='utf-8')
    sys.path[:0]=[str(server.parent),str(server.parent.parent)]
    spec=importlib.util.spec_from_file_location('igc_collect_hub',server)
    helper=importlib.util.module_from_spec(spec);sys.modules[spec.name]=helper;spec.loader.exec_module(helper)
    args=list(cfg['args']);shutil.copyfile(args[args.index('--expert-profile')+1],output/'initial-profile.bin')
    args[args.index('--expert-profile')+1]=str(output/'initial-profile.bin')
    args[args.index('--expert-profile-save')+1]=str(output/'learned-profile.bin')
    args+=['--igc-learned-diagnostic-bundle',str(bundle),'--igc-observation-plan',str(output/'plan.bin'),'--igc-observation-root',str(output/'raw')]
    cfg.update(exe=str(executable),args=args,log=str(output/'engine.log'))
    env=helper.child_env(cfg);env.pop('STRATA_TIMELINE',None);validate_launcher_env(env)
    original=subprocess.Popen
    subprocess.Popen=lambda *a,**k:original(*a,**{**k,'creationflags':k.get('creationflags',0)|0x08000000})
    engine=None;mixed=[];attention=[]
    try:
        engine=helper.StrataEngine(str(executable),helper.engine_args(cfg)+(helper.effort_end_args(cfg,str(executable),tokenizer) or []),
                                   cwd=cfg.get('cwd'),log=cfg['log'],env=env)
        engine.silence_s=helper.engine_silence_s(cfg)
        for i,token_ids in enumerate(ids):
            list(engine.generate(token_ids,1,{'temperature':0},threading.Event()))
            raw=output/'raw'/f'{i}.bin';row=decode_capture(_read(raw,limit=8*1024**2),plan,i)
            mixed.append(np.array(row['mixed']))
            if not outputs and cases[i]['ready']:attention.append(np.array(row['attention'],dtype='<f4'))
            raw.unlink()
            if (i+1)%20==0:print('collected',i+1,len(ids),flush=True)
    finally:
        if engine is not None:engine.close()
        subprocess.Popen=original
    np.save(output/'mixed.npy',np.stack(mixed))
    if attention:np.save(output/'attention.npy',np.stack(attention))
    print('COLLECTED',len(mixed),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('profile','server','executable','bundle','data','reference','pack','output'):parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--count',type=int,default=60);parser.add_argument('--outputs',action='store_true');parser.add_argument('--offset',type=int,default=1000)
    run(**vars(parser.parse_args()))
