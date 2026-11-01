"""LiveCodeBench verdicts for bench records: the benchmark's public and private tests, under CPython, on a throwaway machine only.
Model code runs here, so this belongs on a Colab VM, never on the developer's PC. lcb_score.py records.json tests.json output.json"""
import json, multiprocessing, re, sys

CHECK = r'''
import ast, json, sys, time, io
source, case, kind, name = json.load(open(sys.argv[1]))
header = "from typing import *\nfrom collections import *\nfrom functools import *\nfrom itertools import *\nfrom math import *\nfrom heapq import *\nfrom bisect import *\nimport sys, re, string, random, operator\n"
got = None
if kind == 'functional':
    space = {}
    exec(header + source, space)
    args = [json.loads(line) for line in case['input'].split('\n')]
    fn = getattr(space['Solution'](), name)
    got = fn(*args)
    want = json.loads(case['output'])
    if isinstance(got, tuple):
        got = list(got)
    if isinstance(want, float) or isinstance(got, float):
        print('ok' if abs(float(got) - float(want)) < 1e-6 else 'wrong')
    else:
        print('ok' if got == want else 'wrong')
else:
    sys.stdin = io.StringIO(case['input'])
    sys.stdout, real = io.StringIO(), sys.stdout
    space = {'__name__': '__main__'}
    exec(header + source, space)
    text = sys.stdout.getvalue()
    sys.stdout = real
    a = [l.strip() for l in text.strip().split('\n')]
    b = [l.strip() for l in case['output'].strip().split('\n')]
    print('ok' if a == b else 'wrong')
'''
FENCE = chr(96) * 3


def code_of(text):
    reply = text.rsplit('</think>', 1)[-1]
    found = re.findall(FENCE + r'(?:python|py)?\n(.*?)' + FENCE, reply, re.S)
    return found[-1] if found else None


def one(job):
    import os, subprocess, tempfile
    source, case, kind, name, runner = job
    with tempfile.NamedTemporaryFile('w', suffix='.json', delete=False, encoding='utf-8') as f:
        json.dump([source, case, kind, name], f)
    try:
        r = subprocess.run([sys.executable, '-I', '-B', runner, f.name], capture_output=True, text=True, timeout=8)
        return r.stdout.strip().split('\n')[-1] if r.stdout.strip() else 'error'
    except subprocess.TimeoutExpired:
        return 'timeout'
    finally:
        os.unlink(f.name)


def judge(record, tests, runner, pool):
    source = code_of(record['text'])
    if source is None:
        return 'no code'
    t = tests[record['id']]
    cases = t['public'] + t['private']
    name = t['meta'].get('func_name', '')
    for start in range(0, len(cases), 32):
        for verdict in pool.map(one, [(source, c, c['testtype'], name, runner) for c in cases[start:start + 32]]):
            if verdict != 'ok':
                return verdict
    return 'ok'


if __name__ == '__main__':
    records = json.load(open(sys.argv[1], encoding='utf-8'))
    tests = json.load(open(sys.argv[2], encoding='utf-8'))
    runner = sys.argv[3] + '.runner.py'
    open(runner, 'w', encoding='utf-8').write(CHECK)
    out = []
    with multiprocessing.Pool(8) as pool:
        for r in records:
            verdict = judge(r, tests, runner, pool)
            out.append(dict(id=r['id'], verdict=verdict, tokens=r['tokens'], finish=r['finish']))
    json.dump(out, open(sys.argv[3], 'w', encoding='utf-8'))
    print(len(out), 'ok', sum(o['verdict'] == 'ok' for o in out), 'tokens', sum(o['tokens'] for o in out))
