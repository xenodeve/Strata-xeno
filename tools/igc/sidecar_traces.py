"""Sidecar rows from the model's own reasoning: computed numbers as positives, copied ones as `none`.

Only traces of train questions enter here; test questions stay out of training. Ambiguous numbers are dropped,
since neither label is safe.
"""
import argparse
from fractions import Fraction
import json
from pathlib import Path
import re
from tools.igc.auto_label import after_expression, label_text
from tools.igc.sidecar_data import clean, pointer, table

WRITTEN = re.compile(r'(?<![\w.])\$?\d[\d.,]*%?(?:\s*[-+*/×÷]\s*\$?\d[\d.,]*%?)+\s*=\s*\$?')


def without_expressions(text):
    """The same reasoning with each `a op b =` removed, so a computed number stands in plain words."""
    return WRITTEN.sub('', text)


def rows_of(prompt, thinking, source='trace'):
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
                    level=row.get('level', -1), source=source)
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
    parser.add_argument('--solutions', type=Path, help='GSM8K train file; its solutions go through the same labeller')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rows = []

    def both(prompt, text, source):
        rows.extend(rows_of(prompt, text, source))
        rows.extend(r for r in rows_of(prompt, without_expressions(text), source + '-cut') if r['op'] != 'none')

    for line in args.solutions.read_text(encoding='utf-8').splitlines() if args.solutions else ():
        row = json.loads(line)
        both(row['question'], clean(row['answer'].split('####')[0])[0], 'gsm')
    for arm in args.arms:
        prompts = {c['id']: c['prompt'] for c in json.loads((arm / 'cases.json').read_text(encoding='utf-8'))}
        for record in json.loads((arm / 'records.json').read_text(encoding='utf-8')):
            both(prompts[record['id']], record['text'].split('</think>')[0], 'trace')
    args.output.write_text(json.dumps(rows, ensure_ascii=False), encoding='utf-8')
    counts = {}
    for row in rows:
        key = row['source'], row['op'] != 'none'
        counts[key] = counts.get(key, 0) + 1
    print(len(rows), counts)


if __name__ == '__main__':
    main()
