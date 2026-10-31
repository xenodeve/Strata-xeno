"""Sidecar rows from the model's own reasoning: computed numbers as positives, copied ones as `none`.

Only traces of train questions enter here; test questions stay out of training. Ambiguous numbers are dropped,
since neither label is safe.
"""
import argparse
from fractions import Fraction
import json
from pathlib import Path
from tools.igc.auto_label import after_expression, label_text
from tools.igc.sidecar_data import pointer, table


def rows_of(prompt, thinking):
    context = prompt.strip() + '\n'
    text = context + thinking
    for row in label_text(context.strip(), thinking):
        if row['label'] == 'ambiguous' or after_expression(thinking[:row['start']]):
            continue
        slot = len(context) + row['start']
        entries = table(text, slot)
        if not entries:
            continue
        done = dict(text=text, slot=slot, table=[[e['start'], e['end']] for e in entries], group=prompt, op='none', left=0, right=0,
                    level=row.get('level', -1))
        if row['label'] == 'computed':
            a, b = (Fraction(v.replace(',', '')) for v in row['operands'])
            left, right = pointer(entries, a), pointer(entries, b)
            if left is None or right is None:
                continue
            done.update(op=row['op'], left=left, right=right)
        yield done


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('arms', type=Path, nargs='+')
    parser.add_argument('--annotated', type=Path, help='rows of sidecar_data.py to add')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rows = json.loads(args.annotated.read_text(encoding='utf-8')) if args.annotated else []
    for arm in args.arms:
        prompts = {c['id']: c['prompt'] for c in json.loads((arm / 'cases.json').read_text(encoding='utf-8'))}
        for record in json.loads((arm / 'records.json').read_text(encoding='utf-8')):
            rows += rows_of(prompts[record['id']], record['text'].split('</think>')[0])
    args.output.write_text(json.dumps(rows, ensure_ascii=False), encoding='utf-8')
    counts = {}
    for row in rows:
        counts[row['op']] = counts.get(row['op'], 0) + 1
    print(len(rows), counts)


if __name__ == '__main__':
    main()
