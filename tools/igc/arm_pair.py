"""arm_pair.py BASE_OUT FULL_OUT [--rows rows.jsonl]: two arms of one question set, row by row (#298).

    right in both / only the base / only the full arm / neither, per group of the rows; how each arm ended (stop, length, stuck) and the tokens it used;
    and what the software did in the full arm: events by kind, programs run (who opened them, how many ended with output), values put in.

Arms on this PC are not paired in the strict sense (a sampled model parts paths before the rule acts, 1 of 49 identical here), so a difference of a few
rows is noise: the table shows the counts and the rows that flipped, to be read, not a verdict."""
import collections
import json
import statistics
import sys
from pathlib import Path


def load(out):
    return {r['id']: r for r in json.loads((Path(out) / 'records.json').read_text(encoding='utf-8'))}


def main():
    base, full = load(sys.argv[1]), load(sys.argv[2])
    rows = {}
    if '--rows' in sys.argv:
        rows = {json.loads(l)['id']: json.loads(l) for l in Path(sys.argv[sys.argv.index('--rows') + 1]).read_text(encoding='utf-8').splitlines()}
    ids = [i for i in base if i in full]
    group = lambda i: (rows.get(i, {}).get('group', '?') + ':' + (rows.get(i, {}).get('subtask', '') or '').split('_')[0]).rstrip(':')
    print(f'questions in both arms: {len(ids)} (base {len(base)}, full {len(full)})\n')
    print(f'{"group":24s} {"n":>3s} {"base":>5s} {"full":>5s} | {"both":>4s} {"only base":>9s} {"only full":>9s} {"neither":>7s}')
    for g in sorted({group(i) for i in ids}) + ['all']:
        sel = [i for i in ids if g == 'all' or group(i) == g]
        b = [bool(base[i]['right']) for i in sel]
        f = [bool(full[i]['right']) for i in sel]
        print(f'{g:24s} {len(sel):3d} {sum(b):5d} {sum(f):5d} | {sum(x and y for x, y in zip(b, f)):4d} {sum(x and not y for x, y in zip(b, f)):9d} '
              f'{sum(y and not x for x, y in zip(b, f)):9d} {sum(not x and not y for x, y in zip(b, f)):7d}')
    flipped = [(i, base[i]['right'], full[i]['right']) for i in ids if bool(base[i]['right']) != bool(full[i]['right'])]
    print('\nrows that flipped (id, base right, full right):', flipped)
    for name, arm in (('base', base), ('full', full)):
        sel = [arm[i] for i in ids]
        fin = collections.Counter(r['finish'] for r in sel)
        toks = [int(r['tokens']) + int(r.get('injected', 0)) for r in sel]
        print(f'\n{name}: ended {dict(fin)}; tokens median {int(statistics.median(toks))}, total {sum(toks)}; '
              f'seconds median {statistics.median(float(r["seconds"]) for r in sel):.0f}')
    kinds = collections.Counter(e['kind'] for i in ids for e in full[i]['events'])
    print('\nwhat the software did in the full arm (events):', dict(kinds))
    sel = [full[i] for i in ids]
    print('programs run:', sum(int(r['runs']) for r in sel), '| opened by the model', sum(int(r['runs_by_model']) for r in sel),
          '| forced', sum(int(r['runs_forced']) for r in sel), '| by the tool', sum(int(r['runs_by_tool']) for r in sel),
          '| rows with a program', sum(int(r['runs']) > 0 for r in sel), 'of', len(sel), '| tool seconds', round(sum(float(r['tool_seconds']) for r in sel)))
    print('values put in (inline):', sum(int(r['inline_values']) for r in sel), 'in', sum(int(r['inline_values']) > 0 for r in sel), 'of', len(sel), 'rows')


if __name__ == '__main__':
    main()
