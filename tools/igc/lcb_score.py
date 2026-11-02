"""LiveCodeBench verdicts for bench records: the benchmark's public and private tests, under CPython, on a throwaway machine only.
Model code runs here, so this belongs on a Colab VM, never on the developer's PC. lcb_score.py records.json tests.json output.json"""
import json, multiprocessing, re, sys

CHECK = r'''
import ast, json, sys, time, io
source, case, kind, name = json.load(open(sys.argv[1]))
from decimal import Decimal
# LiveCodeBench's own import_string (lcb_runner/evaluation/testing_util.py), word for word: builtins after math (pow was math.pow,
# #294), `import math`, and the recursion limit
header = "from string import *\nfrom re import *\nfrom datetime import *\nfrom collections import *\nfrom heapq import *\nfrom bisect import *\nfrom copy import *\nfrom math import *\nfrom random import *\nfrom statistics import *\nfrom itertools import *\nfrom functools import *\nfrom operator import *\nfrom io import *\nfrom sys import *\nfrom json import *\nfrom builtins import *\nfrom typing import *\nimport string\nimport re\nimport datetime\nimport collections\nimport heapq\nimport bisect\nimport copy\nimport math\nimport random\nimport statistics\nimport itertools\nimport functools\nimport operator\nimport io\nimport sys\nimport json\nsys.setrecursionlimit(50000)\n"
def decimals(line):
    try:
        return [Decimal(x) for x in line.split()]
    except Exception:
        return None
got = None
if kind == 'functional':   # its grade_call_based: exact, a tuple read as a list
    space = {}
    exec(header + source, space)
    args = [json.loads(line) for line in case['input'].split('\n')]
    fn = getattr(space['Solution'](), name)
    got = fn(*args)
    want = json.loads(case['output'])
    if isinstance(got, tuple):
        got = list(got)
    print('ok' if got == want else 'wrong')
else:   # its grade_stdio: stripped lines, the same count, each equal or equal as decimals
    sys.stdin = io.StringIO(case['input'])
    sys.stdout, real = io.StringIO(), sys.stdout
    space = {'__name__': '__main__'}
    exec(header + source, space)
    text = sys.stdout.getvalue()
    sys.stdout = real
    a = [l.strip() for l in text.strip().split('\n')]
    b = [l.strip() for l in case['output'].strip().split('\n')]
    same = len(a) == len(b) and all(x == y or (decimals(x) is not None and decimals(x) == decimals(y)) for x, y in zip(a, b))
    print('ok' if same else 'wrong')
'''
FENCE = chr(96) * 3


def code_of(text):
    if '</think>' not in text:
        return None
    reply = text.rsplit('</think>', 1)[-1]
    found = re.findall(FENCE + r'(?:python|py)?\n(.*?)' + FENCE, reply, re.S)
    return found[-1] if found else None


def one(job):
    import os, subprocess, tempfile
    source, case, kind, name, runner = job
    with tempfile.NamedTemporaryFile('w', suffix='.json', delete=False, encoding='utf-8') as f:
        json.dump([source, case, kind, name], f)
    try:
        r = subprocess.run([sys.executable, '-I', '-B', runner, f.name], capture_output=True, text=True, timeout=6)
        if r.stdout.strip():
            return r.stdout.strip().split('\n')[-1]
        last = r.stderr.strip().split('\n')[-1] if r.stderr.strip() else ''
        return ('error: ' + last[:160]) if last else 'error'   # the exception, so a scorer fault shows as one
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
        jobs = [(source, c, c['testtype'], name, runner) for c in cases[start:start + 32]]
        for job, verdict in zip(jobs, pool.map(one, jobs)):
            if verdict == 'timeout':   # 8 at once on a VM that also runs the engine: a case near 6 s gets one more try, alone
                verdict = one(job)
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
