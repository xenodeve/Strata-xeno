"""coverage.py OUT [OUT ...] --model M: how much of what each mechanism should have caught it caught, per arm, from a run's ids.jsonl.

A count of how often a mechanism fired says nothing alone (developer, 2026-11-04): ten hand calculations that got past the other layers and
eight corrections is 8/10, and the 10 has to be shown. So each line here is caught / should have been caught:

- `=` with a computable expression before it: the value put in by software (inline value), the line taken back for a program (hint open), or
  left to the model's hand. The base arm has no mechanism, so all of its points are hand.
- hand values the parser calls wrong (mathexpr.wrong_relations, run again over the final text): corrected by --check-hand, or not.
- loops: every loop the runner saw (a loop hint, a loop close, or the stuck stop), and how many got a program.
- programs: blocks run, by who opened them, and how many finished with output.

Not counted, because nothing here can see it: arithmetic done in the head with no expression written (only a hidden-state probe could), and
an expression the parser cannot read (its `=` is not a point at all). Both make the true denominator larger than the one shown.
"""
import argparse
import collections
import json
import re
import sys
from pathlib import Path

INJECTED_CAUGHT = ('inline value',)


def segments(tok, out_ids, spans):
    """[(text, kind)] in order: kind None is the model's own text, else the span's kind."""
    cuts, at = [], 0
    for s, e, kind in sorted(spans):
        if s > at:
            cuts.append((out_ids[at:s], None))
        cuts.append((out_ids[s:e], kind))
        at = e
    if at < len(out_ids):
        cuts.append((out_ids[at:], None))
    return [(tok.decode(ids), kind) for ids, kind in cuts if ids]


def equals_points(segs):
    """(computable `=` the model wrote, of them followed by a software value). The text before each `=` is all text so far, the injected too."""
    from tools.igc.tir4b import inline_found
    done = ''
    points = caught = 0
    for i, (text, kind) in enumerate(segs):
        if kind is None:
            for m in re.finditer(r'(?<![<>!=])=(?!=)', text):
                before = done + text[:m.start()]
                try:
                    found = inline_found(before)
                except Exception:  # noqa: BLE001 - a text the parser fails on is no point, as at run time
                    found = None
                if found is None:
                    continue
                points += 1
                if not text[m.end():].strip() and i + 1 < len(segs) and segs[i + 1][1] in INJECTED_CAUGHT:
                    caught += 1
        done += text
    return points, caught


def wrong_hand(segs):
    """The hand relations the parser calls wrong, over each line the model wrote (a correction line is the runner's)."""
    from tools.igc.mathexpr import moduli_in, wrong_relations
    text = ''.join(t for t, _ in segs)
    model = ''.join(t if k is None else '\0' * len(t) for t, k in segs)   # the model's characters, the rest masked
    found, moduli, at = [], set(), 0
    for line in text.split('\n'):
        mine = model[at:at + len(line)]
        at += len(line) + 1
        if '\0' in mine and not mine.replace('\0', '').strip():
            continue
        line_model = mine.replace('\0', ' ')
        try:
            moduli |= moduli_in(line)
            found += [(expr, wrote) for expr, wrote, _ in wrong_relations(line_model, moduli)]
        except Exception:  # noqa: BLE001
            pass
    return found


def row_coverage(tok, rec, ids_row):
    segs = segments(tok, ids_row['out_ids'], ids_row['spans'])
    points, by_inline = equals_points(segs)
    kinds = collections.Counter(e['kind'] for e in rec['events'])
    hints = kinds['hint open']
    wrong = wrong_hand(segs)
    checks = [(e.get('expr'), e.get('wrote')) for e in rec['events'] if e['kind'] == 'hand check']
    corrected = sum(1 for w in wrong if w in checks)   # the wrong line stays in the text after its correction, so the audit sees both
    loops = kinds['loop hint'] + kinds['loop close'] + (rec['finish'] == 'stuck')
    runs = [e for e in rec['events'] if e['kind'] == 'output']
    return dict(points=points + hints, by_inline=by_inline, by_hint=hints, by_hand=points - by_inline,
                wrong=len(wrong), corrected=corrected, unmatched_checks=sum(1 for c in checks if c not in wrong),
                loops=loops, loop_programs=kinds['loop hint'], loop_closes=kinds['loop close'], stuck=int(rec['finish'] == 'stuck'),
                runs=len(runs), runs_fit=sum(bool(e.get('fits')) for e in runs), runs_by=collections.Counter(e.get('by') for e in runs),
                repeat_closes=kinds['repeated program close'])


def arm_coverage(tok, out):
    out = Path(out)
    recs = {(r['id'], r['sample']): r for r in json.loads((out / 'records.json').read_text(encoding='utf-8'))}
    rows = {}
    for line in (out / 'ids.jsonl').read_text(encoding='utf-8').splitlines():
        if line.strip():
            d = json.loads(line)
            key = (d['id'], d['sample'])
            if key in recs:
                rows[key] = row_coverage(tok, recs[key], d)
    return recs, rows


def share(a, b):
    return f'{a}/{b}' + (f' ({100 * a / b:.0f}%)' if b else '')


def report(name, recs, rows):
    t = collections.Counter()
    by = collections.Counter()
    for r in rows.values():
        for k, v in r.items():
            if k == 'runs_by':
                by.update(v)
            else:
                t[k] += v
    right = sum(bool(recs[k]['right']) for k in rows)
    print(f'== {name}: {len(rows)} answers, {right} right, {sum(recs[k]["tokens"] for k in rows)} model tokens')
    print(f'   computed at `=` (readable): {t["points"]}  software value {share(t["by_inline"], t["points"])}  program hint {share(t["by_hint"], t["points"])}'
          f'  by hand {share(t["by_hand"], t["points"])}')
    print(f'   hand values the parser calls wrong: corrected {share(t["corrected"], t["wrong"])}'
          f'  (corrections the audit does not match: {t["unmatched_checks"]})')
    print(f'   loops seen: {t["loops"]}  given a program {share(t["loop_programs"], t["loops"])}  closed {t["loop_closes"]}  ended stuck {t["stuck"]}')
    print(f'   programs run: {t["runs"]} ({dict(by)}), finished with output {share(t["runs_fit"], t["runs"])}, repeat closes {t["repeat_closes"]}')
    print('   not counted: arithmetic in the head with no expression written, and expressions the parser cannot read')


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('outs', nargs='+')
    p.add_argument('--model', required=True)
    p.add_argument('--rows', action='store_true', help='one line per answer too')
    a = p.parse_args(argv)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    for out in a.outs:
        recs, rows = arm_coverage(tok, out)
        report(Path(out).name, recs, rows)
        if a.rows:
            for k, r in sorted(rows.items()):
                print('   ', k[0], 'right' if recs[k]['right'] else 'wrong', recs[k]['finish'], {x: y for x, y in r.items() if y and x != 'runs_by'})


if __name__ == '__main__':
    sys.exit(main())
