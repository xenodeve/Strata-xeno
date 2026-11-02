"""External benchmark, Gyro-M with and without the learned bundle. Fixed first-N cases, greedy, direct engine protocol."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
from tools.igc.external_score import final_number, question_id, repeating
from tools.igc.hypothesis_read import last_check
from tools.igc.arm_plan import ArmPlan
from tools.igc.hub_contract import validate_launcher_env
from tools.igc.learned_integration import parse_audit
from tools.igc.output_examples import wrapped_prefix
from tools.igc.few_shot import few_shot
from tools.igc.standard_rows import plain_prompt
from tools.qwenputer.memory import system_snapshot
from tools.qwenputer.tokenizer import NativeText

# The model's own tool format, serve/chat_template.jinja: think, call, read the result, think on. It was trained on this loop, and on nothing we force into its reasoning.
TOOL_HINT=('\n\n# Tools\n\nYou have access to the following functions:\n\n<tools>\n'
           '{"type":"function","function":{"name":"run_python","description":"Run Python and read what it prints. Names stay between calls. Test an idea on small cases, compare with a brute force, or run your program on the examples, as often as it helps, before you answer.",'
           '"parameters":{"type":"object","properties":{"code":{"type":"string"}},"required":["code"]}}}\n</tools>\n\n'
           'If you choose to call a function ONLY reply in the following format with NO suffix:\n\n'
           '<tool_call>\n<function=run_python>\n<parameter=code>\nprint(1)\n</parameter>\n</function>\n</tool_call>')
TOOL_CALL=re.compile(r'<tool_call>\s*<function=run_python>\s*<parameter=code>\n?(.*?)\n?</parameter>',re.S)
TOOL_JSON=re.compile(r'<tool_call>\s*(\{.*\})\s*</tool_call>',re.S)
TOOL_ROUNDS=24
# Past the runs, a call is answered with this and nothing runs. Cut off at the 24th, 3478 was still testing its program against a grid it walked as
# an oracle, and ended with no code (#288).
TOOL_SPENT='[no more runs: finish the thinking and write the answer]'
TOOL_AFTER=4


def repeated(ran,code,times=3):
    """The same program a third time running ends the runs. mpw's 0fd92db9 sent one 4116-token program nine times running, 37k tokens of the
    row's 150k (#288), and the repeat check reads one turn so it never saw it. Once again is left alone: after the second line hint m5h3 and
    mph2 wrote the program they had just run, then went on to a new one (de922fcd)."""
    last=[c['code'] for c in ran if c['ok'] is not None][-(times-1):]
    return len(last)==times-1 and all(c==code[:20000] for c in last)
# --hint-after N: every call of 205 in five arms came after the whole thinking, so the program checked an answer already reached by hand. START's
# Hint-infer (2503.04625) puts its own words in the reasoning and the model goes on into a call; here they open the call in the model's format.
# The hint closes the thinking first: with the response inside the thinking, 4 of 22 rows of m5h2 ended the turn at once (`</tool_call>` ends
# an assistant turn in the model's training), with no answer. As a reply call the response comes as the tool's turn and a new thinking reads it.
HINT='\n\nWait, maybe using Python here is a good idea.\n</think>\n\n<tool_call>\n<function=run_python>\n<parameter=code>\n'
HINT_MAX=2
# --hint-at wait: at a fixed line end the hint cut the weighing of a question's reading on MMLU-Pro (mph2: 11 of 19 against 16, tokens 0.37x,
# 3 losses committed to a formula or a convention at the call). START puts its hints after the model's own "Wait" / "Alternatively" (2503.04625,
# Intro and A.4), where it is already reconsidering, and words them as a check of the approach and of the reading. The rest of its first hint follows
# the model's word, so the sentence stays the model's.
HINT_WAIT=', I can use Python to check if my approach is correct and refine it, if necessary.\n</think>\n\n<tool_call>\n<function=run_python>\n<parameter=code>\n'
# --hint-at stop: START's own comparison on QwQ put hints only before the stop token, to leave the reasoning whole; GPQA did not drop (58.5 -> 59.6).
HINT_STOP='\n\nWait, I can use Python to check if my approach is correct and refine it, if necessary.'+HINT_WAIT[HINT_WAIT.index('\n</think>'):]
RECONSIDER=re.compile(r'(?:^|[\n.!?]\s*)(?:Wait|Alternatively)$')


def hint_point(mode,tail):
    """Whether the thinking so far ends where a hint goes in: `line`, any line end; `wait`, the model's own Wait or Alternatively opening a sentence."""
    if mode=='wait':return bool(RECONSIDER.search(tail))
    return tail.endswith('\n')
TOOL_END={248059}   # `</tool_call>` is one token in the Gyro-M pack
TOOL_OUTPUT=4096


def program_of(call):
    """The code of one call, in the template's form or as JSON: on 3674 the model wrote {"name":"run_python","arguments":{"code":...}}."""
    found=TOOL_CALL.findall(call)
    if found:return found[-1]
    for body in TOOL_JSON.findall(call):
        try:data=json.loads(body)
        except ValueError:continue
        if data.get('name')=='run_python' and isinstance(data.get('arguments'),dict) and isinstance(data['arguments'].get('code'),str):return data['arguments']['code']
    return None


def tool_call(text):
    """The program of a reply that ends in a call, or None for an answer."""
    return program_of(text.rsplit('</think>',1)[-1])


def called_in_thinking(text):
    """A call the model closed while still thinking. It was trained to call there too: on 3478 and 3674 it did, got nothing back, wrote a tool's answer of its own and ended with no code."""
    if '</think>' in text or not text.rstrip().endswith('</tool_call>'):return None
    return program_of(text[text.rfind('<tool_call>'):])


ARITHMETIC=('two_digit_addition','two_digit_subtraction','two_digit_multiplication')
BIGBENCH=('3d_addition','3d_subtraction','3d_multiplication','5d_addition','5d_subtraction','5d_multiplication')


def cases(data,arithmetic,gsm,bigbench=0,math=0,gsm_file='gsm8k-test.jsonl',math_file='math500.jsonl',rows_file=None):
    rows=[]
    if rows_file:   # whole prompts: a range of the file, one shard of a long suite
        name,_,span=rows_file.partition(':')
        lines=(data/name).read_text(encoding='utf-8').splitlines()
        if span:
            first,last=span.split('-');lines=lines[int(first):int(last)]
        return [json.loads(line) for line in lines]
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


def log_size(path):
    try:return path.stat().st_size
    except OSError:return 0


def read_events(path,start,end=None):
    """What the engine logged while one row ran, cut by byte offset: the log has no row ids, and a tool round is a request of its own, so a
    count of requests cannot pair them (m5t: 92 requests for 60 rows, and its loop closes landed on the wrong rows). The MIP rule lines
    (`strata igc <kind>: ...`), the requests, and every learned head that fired."""
    try:
        with open(path,'rb') as f:f.seek(start);text=f.read(-1 if end is None else max(end-start,0)).decode('utf-8','replace')
    except OSError:return None
    heads=[h for h in re.findall(r'IGC learned audit: head_status=(\d+)',text) if h!='4294967295']
    return dict(requests=text.count('request metrics'),rules=re.findall(r'strata igc ([a-z]+: [^\r\n]*)',text),heads=heads)


def provenance(bundle,pack):
    """What a pair's two arms must share but the flag under test: the command, the code, the bundle and the tokenizer bytes it binds (#293)."""
    def sha(path):
        try:return hashlib.sha256(Path(path).read_bytes()).hexdigest()
        except OSError:return None
    try:git=subprocess.run(['git','rev-parse','HEAD'],capture_output=True,text=True,cwd=Path(__file__).resolve().parent).stdout.strip() or None
    except OSError:git=None
    return dict(argv=sys.argv[1:],git=git,bundle_sha256=sha(bundle) if bundle else None,
                tokenizer_sha256={n:sha(Path(pack)/'tokenizer'/n) for n in ('merges.txt','tokenizer.json','vocab.json')})


def injected_text(rules,tokenizer):
    """The engine's `inject: <why> at=<n> ids=<a,b,...>` lines with the text they put in the model's stream, read back by the tokenizer."""
    out=[]
    for line in rules:
        m=re.match(r'inject: (.*?) at=(\d+) ids=([\d,]*)',line)
        if m:out.append(dict(why=m.group(1),at=int(m.group(2)),text=tokenizer.decode_bytes([int(x) for x in m.group(3).split(',') if x]).decode('utf-8','replace')))
    return out


def run_slots(engine,rows,tokenizer,prefix,suffix,slots,pair,no_think,whole_context,stuck_rounds,output,cancel,sampling):
    """The questions through batch slots, each as `BGEN`; with `pair` each also runs with `igc=0`, written to records-off.json."""
    jobs=[(row,True) for row in rows]+[(row,False) for row in rows if pair]
    if pair:jobs=[job for two in zip(jobs[:len(rows)],jobs[len(rows):]) for job in two]
    jobs.reverse();free=list(range(slots));live={};admitting=None;done=[]
    def finish(b,why):
        state=live.pop(b);free.append(b)
        text=tokenizer.decode_bytes(state['tokens']).decode('utf-8',errors='replace')
        if no_think:text='</think>\n\n'+text
        final=final_number(text);row=state['row']
        done.append(dict(id=row['id'],suite=row['suite'],answer=row['answer'],final=final,correct=final==row['answer'],tokens=len(state['tokens']),
                         seconds=time.monotonic()-state['start'],finish=why,prompt_tokens=state['ids'],text=text,mip=state['on']))
        (output/'records.json').write_text(json.dumps([r for r in done if r['mip']],ensure_ascii=False),encoding='utf-8')
        if pair:(output/'records-off.json').write_text(json.dumps([r for r in done if not r['mip']],ensure_ascii=False),encoding='utf-8')
        print(row['id'],row['answer'],final,len(state['tokens']),'mip' if state['on'] else 'off',round(done[-1]['seconds'],1),flush=True)
    def take(state,token):
        state['tokens'].append(token)
        if len(state['tokens'])%64==0 and repeating(state['tokens'],rounds=stuck_rounds) and not state['stuck']:
            state['stuck']=True;engine.proc.stdin.write(f"BSTOP {state['slot']}\n");engine.proc.stdin.flush()
    while jobs or live:
        if cancel.is_set():raise RuntimeError('bench stopped')
        if jobs and free and admitting is None:
            row,on=jobs.pop();b=free.pop()
            ids=tokenizer.encode((prefix+row['prompt'].encode('utf-8')+suffix).decode('utf-8'))
            limit=262144-len(ids)-64 if whole_context else row['limit']
            head=f"BGEN {b} {limit}{engine.sampling_keys(sampling)}"+('' if on else ' igc=0')
            live[b]=dict(row=row,on=on,slot=b,tokens=[],stuck=False,start=time.monotonic(),ids=len(ids));admitting=b
            engine.proc.stdin.write(f"{head} {','.join(str(int(t)) for t in ids)}\n");engine.proc.stdin.flush()
        try:line=engine.lines.get(timeout=0.02)
        except queue.Empty:line=''
        if line is None:raise RuntimeError('the engine ended')
        if line.startswith('T ') and admitting is not None:take(live[admitting],int(line.split()[1]))
        elif line.startswith('BADM '):
            part=line.split()
            if part[2]=='0':finish(admitting,'stop')
            admitting=None
        elif line.startswith('ERR'):raise RuntimeError(line)
        for b in list(live):   # the server's pump routes a slot's own lines to its queue; a slot still admitting has its T lines first
            while b!=admitting and b in live and not engine.slot_q[b].empty():
                line=engine.slot_q[b].get_nowait()
                if line is None:raise RuntimeError('the engine ended')
                part=line.split()
                if line.startswith('BT '):take(live[b],int(part[2]))
                else:finish(b,'stuck' if live[b]['stuck'] else part[3])
    return [r for r in done if r['mip']]


def run(profile,server,executable,bundle,data,reference,pack,output,arithmetic,gsm,bigbench=0,bridge=False,inline=False,math=0,no_adapt=False,timeline=None,gsm_file='gsm8k-test.jsonl',math_file='math500.jsonl',sidecar=None,whole_context=False,hint=None,no_note=False,effort_medium=False,close_repeat=None,sum_note=None,no_spec=False,no_reply=False,sandbox=False,think_hint=None,sandbox_after=None,sandbox_close=False,sandbox_close_line=False,sandbox_read=0,sandbox_expect=False,sandbox_why=False,sandbox_gate=False,sandbox_draft=False,sandbox_falsify=False,sandbox_pass_close=False,sandbox_brute=False,sandbox_verify=False,sandbox_tool=False,sandbox_exec=False,repeat_note=None,no_think=False,shots=0,sandbox_call=False,stuck_rounds=20,slots=0,pair=False,keep_prefix=False,rows_file=None,plain=False,retries=0,retry_temperature=0.6,tool_loop=False,reuse_from=None,reuse_rules=None,short_first=False,stop_on_loss=None,lcb_tests=None,hint_after=0,hint_at='line'):
    validate_launcher_env(os.environ)
    output.mkdir(exist_ok=False)
    cfg=json.loads(profile.read_text(encoding='utf-8-sig'))
    tokenizer=NativeText.from_pack(pack/'tokenizer');prefix,suffix=wrapped_prefix(reference,tokenizer)
    if think_hint:suffix+=think_hint.encode('utf-8')
    if no_think:suffix+=b'\n</think>\n\n'
    if effort_medium:prefix=prefix[prefix.index(b'<|im_start|>user'):]   # the template writes no system turn at medium
    if tool_loop:hint=(hint or '')+TOOL_HINT
    if hint:prefix=prefix.replace(b'<|im_end|>',b' '+hint.encode('utf-8')+b'<|im_end|>',1)
    rows=cases(data,arithmetic,gsm,bigbench,math,gsm_file,math_file,rows_file)
    plan=ArmPlan(reuse_from,[r for r in (reuse_rules or '').split(',') if r],short_first,stop_on_loss,hashlib.sha256(executable.read_bytes()).hexdigest())
    if plan.why_not:print(plan.why_not,flush=True)
    rows=plan.order(rows)
    if shots:rows=[{**row,'prompt':few_shot(row['prompt'],shots)} if row['suite']=='gsm8k' else row for row in rows]
    if plain:rows=[{**row,'prompt':plain_prompt(row)} for row in rows]
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
    if sandbox_read:args+=['--igc-sandbox-read',str(sandbox_read)]
    if sandbox_expect:args.append('--igc-sandbox-expect')
    if sandbox_why:args.append('--igc-sandbox-why')
    if sandbox_gate:args.append('--igc-sandbox-gate')
    if sandbox_draft:args.append('--igc-sandbox-draft')
    if sandbox_falsify:args.append('--igc-sandbox-falsify')
    if sandbox_pass_close:args.append('--igc-sandbox-pass-close')
    if sandbox_brute:args.append('--igc-sandbox-brute')
    if sandbox_verify:args.append('--igc-sandbox-verify')
    if sandbox_tool:args.append('--igc-sandbox-tool')
    if sandbox_exec:args.append('--igc-sandbox-exec')
    if sandbox_call:args.append('--igc-sandbox-call')
    if slots:args+=['--batch',str(slots)]
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
        if sandbox_exec:
            from serve.sandbox.exec_bridge import ExecBridge
            engine.sandbox_exec=ExecBridge()
        sampling={'temperature':0}
        if keep_prefix:   # the turn every question shares is read once: the engine keeps a checkpoint at its end
            all_ids=[tokenizer.encode((prefix+row['prompt'].encode('utf-8')+suffix).decode('utf-8')) for row in rows]
            sampling['_ckpt_at']=len(os.path.commonprefix(all_ids))
        judge=None
        if stop_on_loss and lcb_tests:   # each row is scored as it ends, so a loss stops the arm at once
            import multiprocessing
            from tools.igc import lcb_score
            tests=json.loads(Path(lcb_tests).read_text(encoding='utf-8'));runner=output/'lcb.runner.py';runner.write_text(lcb_score.CHECK,encoding='utf-8')
            pool=multiprocessing.Pool(8);judge=lambda r:lcb_score.judge(r,tests,str(runner),pool)
        elif stop_on_loss:   # a written answer: the suite's own rule, not the last number in the reply (a letter, an expression)
            from tools.igc.select_rows import right
            judge=lambda r:'ok' if right(r,r['text']) else 'wrong'
        if slots:records=run_slots(engine,rows,tokenizer,prefix,suffix,slots,pair,no_think,whole_context,stuck_rounds,output,cancel,sampling)
        for row in rows if not slots else ():
            if cancel.is_set():raise RuntimeError('bench stopped before request')
            old=plan.reusable(row['id'])
            if old:   # this machine repeats itself, and no rule of this arm could fire on the base row
                records.append({**{k:v for k,v in old.items() if k!='log_from'},'reused':True})   # its events are the base run's; its offsets were in that log
                (output/'records.json').write_text(json.dumps(records,ensure_ascii=False),encoding='utf-8')
                print(row['id'],'reused',old['tokens'],flush=True)
                continue
            ids=tokenizer.encode((prefix+row['prompt'].encode('utf-8')+suffix).decode('utf-8'))
            active[0]=start=time.monotonic();mark=log_size(output/'engine.log')
            limit=262144-len(ids)-64 if whole_context else row['limit']
            # Greedy repeats itself, so a retry samples. It starts from the prompt: a wrong draft left in the context costs far more than a blank one (2602.04288).
            attempts=[]
            for attempt in range(retries+1):
                tokens=[];stuck=False
                keys=sampling if not attempt else {**sampling,'temperature':retry_temperature,'seed':attempt}
                said=[];kernel=None;asked=list(ids);ran=[];hints=[];quiet=0;lead='';spent=False
                for turn in range(TOOL_ROUNDS+TOOL_AFTER if tool_loop else 1):
                    tokens=[]
                    thinking='</think>' not in lead;inner=None;hinted=False   # after the hint the model writes a reply call, not thinking
                    for t in engine.generate(asked,262144-len(asked)-64 if whole_context else limit-sum(len(s) for s in said),keys,cancel):
                        if t is None:continue
                        tokens.append(int(t))
                        if len(tokens)%64==0 and repeating(tokens,rounds=stuck_rounds):stuck=True;break
                        if (hint_at=='stop' and hint_after and tool_loop and not sandbox_tool and int(t)==248069 and thinking and not ran and not hints
                                and quiet>=hint_after):   # before the stop token, as START's QwQ comparison: the reasoning is whole
                            tokens.pop();hinted=True;break
                        thinking=thinking and int(t)!=248069
                        quiet+=thinking
                        if (hint_after and tool_loop and not sandbox_tool and thinking and not lead and quiet>=hint_after and len(hints)<HINT_MAX
                                and hint_at!='stop' and (int(t) in (198,271) if hint_at=='line' else hint_point(hint_at,tokenizer.decode_bytes(tokens[-16:]).decode('utf-8','replace')))):
                            hinted=True;break   # N thinking tokens with no call, at a hint point: the hint goes in
                        if tool_loop and not sandbox_tool and thinking and int(t) in TOOL_END:   # the close of a call while thinking: run it before the model writes an answer of its own
                            inner=called_in_thinking(lead+tokenizer.decode_bytes(tokens).decode('utf-8',errors='replace'))
                            if inner is not None:break
                        if tool_loop and not thinking and int(t) in TOOL_END:break   # a reply call ends here: ddfff815 wrote a user turn of its own after it
                    said.append(tokens)
                    if hinted:   # the model writes the program on from the hint's open call
                        words=HINT if hint_at=='line' else HINT_WAIT if hint_at=='wait' else HINT_STOP
                        # the hint is not the model's, so it stays out of the text and its count; where it went in and what it said stay here
                        hints.append(dict(turn=turn,after=quiet,at=hint_at,token=sum(len(s) for s in said),char=len(tokenizer.decode_bytes([x for s in said for x in s]).decode('utf-8','replace')),words=words))
                        asked=asked+tokens+tokenizer.encode(words);lead=words;quiet=0
                        continue
                    code=inner if inner is not None else None if stuck or not tool_loop else tool_call(lead+tokenizer.decode_bytes(tokens).decode('utf-8',errors='replace'))
                    if code is None:break
                    quiet=0;lead=''
                    spent=spent or repeated(ran,code)
                    if spent and sum(c['ok'] is None for c in ran)>=TOOL_AFTER:break   # as many answered calls as past the 24th runs, no more
                    if turn<TOOL_ROUNDS and not spent:
                        from serve.sandbox.exec_bridge import Kernel
                        if kernel and not kernel.alive():kernel.close();kernel=None   # a call that ran past its time took the kernel with it
                        restarted=kernel is None and bool(ran);kernel=kernel or Kernel()
                        began=time.monotonic();ok,shown=kernel.run(code);took=time.monotonic()-began
                        full=len(shown or '');shown=(shown or '[it ran and printed nothing]')[:TOOL_OUTPUT]
                    else:shown=TOOL_SPENT;ok=None;took=0.;full=0;restarted=False
                    # each call whole enough to audit: what ran, what came back and how much was cut, whether it failed, its seconds
                    ran.append(dict(turn=turn,thinking=inner is not None,code=code[:20000],code_chars=len(code),shown=shown[:20000],shown_chars=full,
                                    ok=ok,seconds=round(took,3),restarted=restarted))
                    if inner is not None:   # the answer goes into the thinking, which goes on
                        asked=asked+tokens+tokenizer.encode('\n<tool_response>\n'+shown+'\n</tool_response>\n')
                        continue
                    spoken=tokens[:-1] if tokens[-1]==248046 else tokens   # the end of turn goes back in below, with the rest of the template's shape
                    asked=asked+spoken+tokenizer.encode((b'<|im_end|>\n<|im_start|>user\n<tool_response>\n'+shown.encode('utf-8')+b'\n</tool_response>'+suffix).decode('utf-8'))
                if kernel:kernel.close()
                tokens=[t for s in said for t in s]
                text=tokenizer.decode_bytes(tokens).decode('utf-8',errors='replace')
                if no_think:text='</think>\n\n'+text   # the close is in the prompt; the readers look for it here
                attempts.append(dict(tokens=len(tokens),text=text,finish='stuck' if stuck else engine.last.get('finish'),check=last_check(text),turns=len(said),ran=ran,
                                     turn_tokens=[len(x) for x in said],hints=hints))
                if attempts[-1]['check'] not in ('mismatch','stub'):break
            seconds=time.monotonic()-start;active[0]=None
            # The first whose tests did not fail, else the first: a sampled run has no claim over the greedy one.
            kept=next((a for a in attempts if a['check'] not in ('mismatch','stub')),attempts[0])
            text=kept['text'];final=final_number(text)
            records.append(dict(id=row['id'],suite=row['suite'],answer=row['answer'],final=final,correct=final==row['answer'],
                                tokens=sum(a['tokens'] for a in attempts),seconds=seconds,finish=kept['finish'],prompt_tokens=len(ids),text=text,
                                attempts=[dict(tokens=a['tokens'],check=a['check'],turns=a['turns'],turn_tokens=a['turn_tokens'],ran=a['ran'],hints=a['hints']) for a in attempts],
                                tool_seconds=round(sum(c['seconds'] for a in attempts for c in a['ran']),3),log_from=mark))
            ev=read_events(output/'engine.log',mark)
            if ev is not None:records[-1]['events']={**ev,'injected':injected_text(ev['rules'],tokenizer)}
            if plan.verdicts is not None:
                records[-1]['verdict']=judge(records[-1]) if judge else 'ok' if records[-1]['correct'] else 'wrong'
            (output/'records.json').write_text(json.dumps(records,ensure_ascii=False),encoding='utf-8')
            print(row['id'],row['answer'],final,records[-1]['tokens'],round(seconds,1),records[-1].get('verdict',''),flush=True)
            if plan.verdicts is not None and plan.lost(row['id'],records[-1]['verdict']):
                print('LOST',row['id'],'right in the base arm: stopped, read it first',flush=True)
                break
    finally:
        if engine is not None:engine.close()
        stop.set();thread.join(timeout=3);subprocess.Popen=original
    # A row's last lines can land after its record was written: read each row again, from its offset to the next row's.
    ends=[r.get('log_from') for r in records][1:]+[None]
    for r,end in zip(records,ends):
        if r.get('log_from') is None:continue
        ev=read_events(output/'engine.log',r['log_from'],end)
        if ev is not None:r['events']={**ev,'injected':injected_text(ev['rules'],tokenizer)}
    if bundle and not slots and not retries and not tool_loop and not plan.rules:   # one audit a request: a retry or a tool round is a request, a reused row none
        lines=(output/'engine.log').read_text(encoding='utf-8',errors='replace')
        audits=[parse_audit(bytes.fromhex(h)) for h in re.findall(r'strata IGC learned audit: head_status=\d+ record=([0-9a-f]{576})',lines)]
        if len(audits)!=len(records):raise ValueError('incomplete audit pairing')
        for record,audit in zip(records,audits,strict=True):
            record.update(evaluations=audit['evaluations'],request=audit['request'],result=audit['result'],status=audit['status'])
    report=dict(kind='external comparison, fixed first-N subset, greedy; not published-protocol scores',arm='on' if bundle else 'off',
                executable_sha256=hashlib.sha256(executable.read_bytes()).hexdigest(),suites=score(records),failure=failure,
                **provenance(bundle,pack))
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
    parser.add_argument('--sandbox-read',type=int,default=0,help='tokens the model reads a printed result for before the close')
    parser.add_argument('--sandbox-expect',action='store_true',help='open the call as expect(result, expected)')
    parser.add_argument('--sandbox-why',action='store_true',help='the model\'s own line after a mismatch')
    parser.add_argument('--sandbox-gate',action='store_true',help='hold the end of the thinking for a check')
    parser.add_argument('--sandbox-draft',action='store_true',help='a bare fence holding a whole draft runs too')
    parser.add_argument('--sandbox-falsify',action='store_true',help='after a passing check, ask what input would break it')
    parser.add_argument('--sandbox-pass-close',action='store_true',help='end the thinking once every check of a run has passed')
    parser.add_argument('--sandbox-tool',action='store_true',help='the engine runs a run_python call made while thinking (with --sandbox; --tool-loop answers calls in the reply)')
    parser.add_argument('--sandbox-verify',action='store_true',help="finish the model's `Let's verify` as a check by program")
    parser.add_argument('--sandbox-brute',action='store_true',help='the block opened by --sandbox-after asks for a brute force from the statement')
    parser.add_argument('--sandbox-exec',action='store_true',help='what the interpreter refuses runs in real Python: throwaway VM only')
    parser.add_argument('--sandbox-call',action='store_true',help='open a print( after a block that printed nothing')
    parser.add_argument('--plain',action='store_true',help='with --rows-file: the question alone, no exemplars')
    parser.add_argument('--rows-file',help='file of whole prompts in the data dir, optionally name:first-last')
    parser.add_argument('--keep-prefix',action='store_true',help='read the prompt part every question shares once')
    parser.add_argument('--slots',type=int,default=0,help='decode in this many batch slots, questions at once')
    parser.add_argument('--pair',action='store_true',help='with --slots: run each question with MIP and with igc=0, in one engine')
    parser.add_argument('--stuck-rounds',type=int,default=20,help='rounds of a cycle before the bench gives up; 40 leaves a loop of up to 3 tokens to the engine, which closes at 64')
    parser.add_argument('--no-think',action='store_true',help='close the thinking in the prompt')
    parser.add_argument('--reuse-from',type=Path,help='a base arm on this machine; its rows where none of --reuse-rules could fire are taken as they are')
    parser.add_argument('--reuse-rules',help='comma list of the rules this arm adds: draft, gate, verify, pass-close, falsify, brute, after')
    parser.add_argument('--short-first',action='store_true',help='rows in the order of the base arm\'s tokens, fewest first')
    parser.add_argument('--stop-on-loss',type=Path,help='verdicts of the base arm: stop at the first row it got right and this arm did not')
    parser.add_argument('--hint-after',type=int,default=0,help='with --tool-loop and not --sandbox-tool: after N thinking tokens with no call, at a line end, '
                        'START\'s hint closes the thinking and opens a run_python call (at most 2 a row; 0 never)')
    parser.add_argument('--hint-at',choices=('line','wait','stop'),default='line',help='with --hint-after: at any line end, after the model\'s own Wait / Alternatively, or in place of its </think> in a row with no call (once)')
    parser.add_argument('--lcb-tests',type=Path,help='LiveCodeBench tests, to score each row as it ends (throwaway VM only)')
    parser.add_argument('--tool-loop',action='store_true',help='declare run_python in the model\'s own tool format and answer its calls: throwaway VM only')
    parser.add_argument('--retries',type=int,default=0,help='sampled runs from the prompt after one that ended with its tests failing')
    parser.add_argument('--retry-temperature',type=float,default=0.6)
    parser.add_argument('--shots',type=int,default=0,help='few-shot exemplars before each GSM8K question')
    run(**vars(parser.parse_args()))
