"""Did the sandbox change a hypothesis? For each row of an arm: what the checks said, whether a mismatch was followed by a run that matched, and how the row ended against
a baseline arm. hypothesis_read.py arm_dir [--base arm_dir] [--verdicts file.json] [--base-verdicts file.json]

A verdict file maps a row id to `ok`, or to anything else for a row that failed; LiveCodeBench verdicts come from `lcb_score.py`, written answers from `answer_check.py`."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re

FENCE = chr(96) * 3
BLOCK = re.compile(FENCE + r'python\n(.*?)\n?' + FENCE + r'\s*\n?' + FENCE + r'output\n(.*?)\n?' + FENCE, re.S)


def result_of(output):
    """What one run told the model."""
    if 'MISMATCH' in output:
        # `got None` is a stub that was called, not a hypothesis that failed
        return 'mismatch' if re.search(r'MISMATCH: expected .*, got (?!None$)', output, re.M) else 'stub'
    if output.startswith('match:'):
        return 'match'
    if output.startswith('[it ran and printed nothing'):
        return 'nothing'
    if output.startswith('[too long'):
        return 'too long'
    if output.startswith('[only') or 'ModuleNotFoundError' in output:
        return 'no module'
    if output.startswith('[stdin'):
        return 'no stdin'
    if re.search(r'^\w*Error\b|^Traceback', output, re.M):
        return 'error'
    return 'none' if output.strip() == 'None' else 'value'


def runs_of(text):
    """Each run in order, with its offset in characters, read from the reasoning."""
    head, _, _ = text.partition('</think>')
    return [dict(at=match.start(), code=match.group(1), output=match.group(2), result=result_of(match.group(2))) for match in BLOCK.finditer(head)]


def corrected(runs):
    """The loop the plan is after: a run whose result was wrong, then a run that passed."""
    missed = None
    for run in runs:
        if run['result'] == 'mismatch' and missed is None:
            missed = run
        elif run['result'] == 'match' and missed is not None:
            return True
    return False


def last_check(text):
    """How the last test of a reasoning came out, or None when it ran none."""
    checks = [run['result'] for run in runs_of(text) if run['result'] in ('match', 'mismatch', 'stub')]
    return checks[-1] if checks else None


def row(record, verdict):
    runs = runs_of(record['text'])
    results = Counter(run['result'] for run in runs)
    return dict(id=record['id'], tokens=record['tokens'], ok=verdict == 'ok' if verdict is not None else None, verdict=verdict, runs=len(runs), results=dict(results),
                first_run=runs[0]['at'] if runs else None, corrected=corrected(runs), tested=results['match'] + results['mismatch'] > 0, finish=record['finish'])


def read_verdicts(path, records):
    if path is None:
        return {r['id']: ('ok' if r.get('correct') else 'wrong') for r in records}
    data = json.loads(path.read_text(encoding='utf-8'))
    if isinstance(data, list):
        return {r['id']: r['verdict'] for r in data}
    return {key: (value if isinstance(value, str) else value.get('verdict')) for key, value in data.items()}


def load(arm, verdicts):
    records = json.loads((arm / 'records.json').read_text(encoding='utf-8'))
    table = read_verdicts(verdicts, records)
    return {r['id']: row(r, table.get(r['id'])) for r in records}


def summary(rows):
    known = [r for r in rows.values() if r['ok'] is not None]
    return dict(rows=len(rows), right=sum(r['ok'] for r in known), tokens=sum(r['tokens'] for r in rows.values()), tested=sum(r['tested'] for r in rows.values()),
                saw_mismatch=sum('mismatch' in r['results'] for r in rows.values()), corrected=sum(r['corrected'] for r in rows.values()),
                corrected_right=sum(r['corrected'] and bool(r['ok']) for r in rows.values()),
                results=dict(sum((Counter(r['results']) for r in rows.values()), Counter())))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('arm', type=Path)
    parser.add_argument('--base', type=Path)
    parser.add_argument('--verdicts', type=Path)
    parser.add_argument('--base-verdicts', type=Path)
    args = parser.parse_args()
    rows = load(args.arm, args.verdicts)
    base = load(args.base, args.base_verdicts) if args.base else {}
    print(args.arm.name, json.dumps(summary(rows)))
    if base:
        same = {key: value for key, value in base.items() if key in rows}
        print(args.base.name, 'on the same rows', json.dumps(summary(same)))
    print(f'{"id":14}{"verdict":10}{"tokens":>8}{"base":>10}{"base tok":>9}{"runs":>5}  results')
    for key, r in rows.items():
        b = base.get(key, {})
        change = '' if not b or r['ok'] == b.get('ok') else ' GAINED' if r['ok'] else ' LOST'
        print(f'{key:14}{str(r["verdict"]):10}{r["tokens"]:8}{str(b.get("verdict", "")):>10}{b.get("tokens", 0):9}{r["runs"]:5}  {r["results"]}{" corrected" if r["corrected"] else ""}{change}')


if __name__ == '__main__':
    main()
