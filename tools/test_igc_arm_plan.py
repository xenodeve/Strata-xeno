"""arm_plan decides what an arm need not run again: python tools/test_igc_arm_plan.py"""
import json
import sys
import tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tools.igc.arm_plan import ArmPlan, events_by_row, runs_a_block, says_check, silent

F = chr(96) * 3
draft = f'thinking\n{F}\nclass Solution:\n    def f(self, a):\n        b = a\n        return b\n{F}\nmore'
sketch = f'thinking\n{F}\nclass Solution:\n    def f(...):\n        ...\n    pass\n{F}\nmore'
assert runs_a_block(draft) and not runs_a_block(sketch)
assert runs_a_block(f'x\n{F}python\nprint(1)\n{F}\n'), 'a python block always runs'
assert not runs_a_block(f'x\n{F}text\ndef f():\n  a\n  b\n  c\n  d\n{F}\n'), 'another language never'
assert not runs_a_block(f'Ensure delimiters: {F}python ...{F}. Done.'), 'a fence in a sentence opens nothing'
assert not runs_a_block('no fences at all')
assert runs_a_block(f'a\n{F}\nF=(1,2)\n{F}\nthen\n{F}python\nx=1\n{F}\n'), 'the second block of a pair-scan'
assert says_check("So x = 3.\n\nLet's verify") and says_check('x. Check: 3') and not says_check('we can check it')

rec = lambda text, tokens=1000: dict(id='a', text=text, tokens=tokens)
plain = 'long thinking with no fence</think>\n\nanswer'
assert silent(rec(plain), [], ['draft']), 'no block: the draft rule cannot fire'
assert not silent(rec(draft + '</think>x'), [], ['draft'])
assert not silent(rec(plain), [], ['gate']), 'the model closed its own thinking: the gate fires there'
assert silent(rec(plain), ['closed the thinking'], ['gate']), 'a forced close never reaches the gate'
assert not silent(rec(plain), ['closed the thinking'], ['gate', 'unknown']), 'an unknown rule always runs'
assert silent(rec(plain), [], ['pass-close']) and not silent(rec('x\nmatch: 3\n</think>y'), [], ['pass-close'])
assert silent(rec('short</think>a', tokens=300), [], ['brute']) and not silent(rec('long</think>a', tokens=3000), [], ['brute'])

log = 'strata igc loop: closed the thinking\nstrata serve: request metrics: x\nstrata igc sandbox: run 1\nstrata serve: request metrics: y\n'
assert events_by_row(log, 2) == [['closed the thinking'], ['run 1']]

with tempfile.TemporaryDirectory() as d:
    d = Path(d)
    rows = [dict(id='long', text=plain, tokens=900), dict(id='short', text=draft + '</think>x', tokens=100)]
    (d / 'records.json').write_text(json.dumps(rows), encoding='utf-8')
    (d / 'engine.log').write_text('strata igc loop: closed the thinking\nstrata serve: request metrics: a\nstrata serve: request metrics: b\n', encoding='utf-8')
    (d / 'v.json').write_text(json.dumps([dict(id='long', verdict='ok'), dict(id='short', verdict='wrong')]), encoding='utf-8')
    p = ArmPlan(d, ['draft', 'gate'], short_first=True, verdicts=d / 'v.json')
    assert [r['id'] for r in p.order([dict(id='long'), dict(id='short'), dict(id='new')])] == ['short', 'long', 'new']
    assert p.reusable('long')['tokens'] == 900, 'no block and a forced close: the base row stands'
    assert p.reusable('short') is None and p.reusable('new') is None
    assert p.lost('long', 'wrong') and not p.lost('short', 'wrong') and not p.lost('long', 'ok')
    (d / 'report.json').write_text(json.dumps(dict(executable_sha256='aa' * 32)), encoding='utf-8')
    assert ArmPlan(d, ['draft', 'gate'], engine_sha='aa' * 32).reusable('long'), 'the same engine'
    other = ArmPlan(d, ['draft', 'gate'], engine_sha='bb' * 32)
    assert other.reusable('long') is None and 'nothing is reused' in other.why_not, 'another engine writes other tokens'
    (d / 'engine.log').write_text('strata serve: request metrics: a\n', encoding='utf-8')
    assert ArmPlan(d, ['draft']).reusable('long') is None, 'a log that does not pair reuses nothing'
print('ok')
