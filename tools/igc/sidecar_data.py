"""Sidecar examples from GSM8K train annotations: which earlier numbers and operation give the number at a slot.

Values stay out of the label: the target is an operation and pointers into the quantity table, so a model
built on these rows can only choose among numbers that were really written. Test splits never enter here.
"""
import argparse
from collections import Counter
from fractions import Fraction
import json
from pathlib import Path
import re
from tools.igc.auto_label import NUMBER

CALL = re.compile(r'<<([^<>=]+)=([^<>]+)>>')
SIMPLE = re.compile(r'^\s*(-?[\d.,]+%?)\s*([-+*/])\s*(-?[\d.,]+%?)\s*$')
WRITTEN = re.compile(r'[-\d.,%$\s]*\$?-?[\d.,]+%?\s*[-+*/]\s*\$?-?[\d.,]+%?\s*=\s*\$?$')
NAMES = {'+': 'add', '-': 'sub', '*': 'mul', '/': 'div'}


def clean(solution):
    """Solution text without the annotations, and where each annotation's result then sits."""
    out = []
    calls = []
    at = 0
    for m in CALL.finditer(solution):
        out.append(solution[at:m.start()])
        calls.append((sum(len(p) for p in out), m[1], m[2]))
        at = m.end()
    out.append(solution[at:])
    return ''.join(out), calls


def fraction(text):
    text = text.replace(',', '').strip()
    return Fraction(text[:-1]) / 100 if text.endswith('%') else Fraction(text)


def table(text, end):
    entries = []
    for m in NUMBER.finditer(text, 0, end):
        entries.append(dict(start=m.start(), end=m.end(), value=Fraction(m[0].replace(',', ''))))
    return entries


def pointer(entries, value):
    """Latest entry with this value, or None; only written numbers can be pointed at."""
    for i in range(len(entries) - 1, -1, -1):
        if entries[i]['value'] == value:
            return i
    return None


def examples(row):
    question = row['question'].strip() + '\n'
    solution, calls = clean(row['answer'].split('####')[0])
    text = question + solution
    for offset, expression, result in calls:
        m = SIMPLE.match(expression)
        if not m:
            yield dict(kept=False, reason='not one operation')
            continue
        slot = len(question) + offset
        entries = table(text, slot)
        try:
            a, b = fraction(m[1]), fraction(m[3])
        except (ValueError, ZeroDivisionError):
            yield dict(kept=False, reason='operand')
            continue
        left, right = pointer(entries, a), pointer(entries, b)
        if left is None or right is None:
            yield dict(kept=False, reason='operand not written')
            continue
        yield dict(kept=True, text=text, slot=slot, op=NAMES[m[2]], left=left, right=right, entries=len(entries), result=result,
                   table=[[e['start'], e['end']] for e in entries], group=row['question'], written=True)
        before = WRITTEN.search(text[:slot])
        if before is None:
            continue
        kept_text = text[:before.start()].rstrip() + ' '
        text_without = kept_text + text[slot:]
        entries = table(text_without, len(kept_text))
        left, right = pointer(entries, a), pointer(entries, b)
        if left is not None and right is not None:
            yield dict(kept=True, text=text_without, slot=len(kept_text), op=NAMES[m[2]], left=left, right=right, entries=len(entries),
                       result=result, table=[[e['start'], e['end']] for e in entries], group=row['question'], written=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    reasons = Counter()
    operations = Counter()
    kept = []
    for line in args.train.read_text(encoding='utf-8').splitlines():
        for example in examples(json.loads(line)):
            if example.pop('kept'):
                kept.append(example)
                operations[example['op']] += 1
            else:
                reasons[example['reason']] += 1
    args.output.write_text(json.dumps(kept, ensure_ascii=False), encoding='utf-8')
    print(len(kept), dict(operations), dict(reasons), 'median entries', sorted(k['entries'] for k in kept)[len(kept) // 2])


if __name__ == '__main__':
    main()
