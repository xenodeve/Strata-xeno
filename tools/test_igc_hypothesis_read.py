"""hypothesis_read reads the runs out of a stored reasoning: python tools/test_igc_hypothesis_read.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tools.igc.hypothesis_read import corrected, last_check, result_of, row, runs_of

F = chr(96) * 3


def block(code, output):
    return f'{F}python\n{code}\n{F}\n{F}output\n{output}\n{F}\n\n'


def text(*blocks, reply='answer'):
    return 'thinking\n' + ''.join(blocks) + '</think>\n\n' + reply


assert result_of('MISMATCH: expected 3, got 1') == 'mismatch'
assert result_of('match: 3\nMISMATCH: expected 4, got 1') == 'mismatch', 'one failure among passes'
assert result_of('match: 3') == 'match'
assert result_of('[it ran and printed nothing: call your function and print the result]') == 'nothing'
assert result_of('TypeError: expect expected 2 arguments, got 1 (line 2)\n    expect(') == 'error'
assert result_of('None') == 'none'
assert result_of('12') == 'value'
assert result_of('[only math, fractions] ') == 'no module'
assert result_of('[stdin was empty: put feed("...")]') == 'no stdin'

fixed = text(block('expect(f(1), 2)', 'MISMATCH: expected 2, got 1'), block('def f(a):\n    return 2', '[it ran and printed nothing: call your function and print the result]'), block('expect(f(1), 2)', 'match: 2'))
assert [r['result'] for r in runs_of(fixed)] == ['mismatch', 'nothing', 'match']
assert corrected(runs_of(fixed)), 'a mismatch, then a match'
assert not corrected(runs_of(text(block('expect(f(1), 2)', 'match: 2'), block('expect(f(1), 3)', 'MISMATCH: expected 3, got 2')))), 'a match first is not a correction'
assert not corrected(runs_of(text(block('expect(f(1), 2)', 'MISMATCH: expected 2, got 1')))), 'a mismatch left alone'
stub = text(block('expect(f(1), 2)', 'MISMATCH: expected 2, got None'), block('expect(f(1), 2)', 'match: 2'))
assert [r['result'] for r in runs_of(stub)] == ['stub', 'match']
assert not corrected(runs_of(stub)), 'a stub filled in is no corrected hypothesis'
assert result_of('match: 1\nMISMATCH: expected 2, got None\nMISMATCH: expected 3, got 4') == 'mismatch', 'one real result among stubs'
assert runs_of('no blocks at all') == []
assert runs_of(text(reply=block('print(1)', '1'))) == [], 'a block in the reply was not run'

assert last_check(fixed) == 'match'
assert last_check(stub) == 'match'
assert last_check(text(block('expect(f(1), 2)', 'match: 2'), block('expect(f(1), 3)', 'MISMATCH: expected 3, got 2'))) == 'mismatch', 'the last test decides'
assert last_check(text(block('print(1)', '1'))) is None, 'a print is no test'
r = row(dict(id='a', tokens=9, text=fixed, finish='stop'), 'ok')
assert r['ok'] and r['corrected'] and r['tested'] and r['runs'] == 3, r
assert row(dict(id='a', tokens=9, text=fixed, finish='stop'), None)['ok'] is None
print('ok')
