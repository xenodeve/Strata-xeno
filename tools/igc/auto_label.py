"""Label numbers in reasoning traces by value, not by wording, for sidecar data and level counts.

A new number is `computed` when exactly one short program over earlier numbers gives it. Level 0 has
the expression right before `=`, level 1 takes its operands from the same sentence, level 2 reaches back.
"""
import argparse
from collections import Counter
from fractions import Fraction
import itertools
import json
from pathlib import Path
import re

NUMBER = re.compile(r'(?<![\w.,])\d{1,3}(?:,\d{3})+(?:\.\d+)?(?![\w])|(?<![\w.,])\d+(?:\.\d+)?(?![\w])')
SENTENCE_END = re.compile(r'[.!?]\s|\n')
SMALL = 12          # results at or below this happen by chance too often to label
RECENT = 16         # distinct earlier values a program may use
OPERATIONS = dict(add=lambda a, b: a + b, sub=lambda a, b: a - b, mul=lambda a, b: a * b,
                  div=lambda a, b: a / b if b else None)


def numbers(text):
    return [(m.start(), m.end(), m[0], Fraction(m[0].replace(',', ''))) for m in NUMBER.finditer(text)]


def programs(value, earlier):
    """Distinct one-step programs; a commutative pair counts once."""
    found = {}
    for a, b in itertools.product(earlier, earlier):
        for name, operation in OPERATIONS.items():
            if name in ('add', 'mul') and a > b:
                continue
            if operation(a, b) == value:
                found[(name, a, b)] = True
    return list(found)


def level_of(thinking, start, operand_positions):
    if '=' in thinking[max(0, start - 4):start]:
        before = thinking[:start].rstrip()[:-1]
        if re.search(r'\d\s*[-+*×/÷]\s*\S*\d\s*$', before):
            return 0
    sentence = max((m.end() for m in SENTENCE_END.finditer(thinking, 0, start)), default=0)
    return 1 if all(p >= sentence for p in operand_positions) else 2


def label_text(prompt, thinking):
    seen = {}       # value -> last position, negative inside the prompt
    for _, end, _, value in numbers(prompt):
        seen[value] = end - len(prompt) - 1
    rows = []
    for start, end, text, value in numbers(thinking):
        row = dict(value=text, start=start, end=end)
        if value in seen:
            row['label'] = 'copied'
        elif _measure(thinking, start, end):
            row['label'] = 'not_computed'
        else:
            recent = sorted(seen, key=seen.get)[-RECENT:]
            found = [] if value.denominator == 1 and value <= SMALL else programs(value, recent)
            if not found:
                row['label'] = 'not_computed'
            elif len(found) > 1:
                row['label'] = 'ambiguous'
            else:
                name, a, b = found[0]
                row.update(label='computed', op=name, operands=[_show(a), _show(b)],
                           level=level_of(thinking, start, [seen[a], seen[b]]))
        rows.append(row)
        seen[value] = start
    return rows


def _measure(thinking, start, end):
    """Angles, coordinates and approximations match short programs by chance."""
    after = thinking[end:end + 1]
    before = thinking[max(0, start - 2):start]
    return after == chr(176) or before.endswith(('(', ',', '~')) or thinking[max(0, start - 1):start] == '-'


def _show(value):
    return str(value.numerator) if value.denominator == 1 else str(float(value))


def summarize(traces):
    counts = Counter()
    for rows in traces:
        for row in rows:
            counts[row['label']] += 1
            if row['label'] == 'computed':
                counts[f"computed_level_{row['level']}"] += 1
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('arms', type=Path, nargs='+')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    labelled = []
    by_suite = {}
    for arm in args.arms:
        cases = {c['id']: c for c in json.loads((arm / 'cases.json').read_text(encoding='utf-8'))}
        for record in json.loads((arm / 'records.json').read_text(encoding='utf-8')):
            rows = label_text(cases[record['id']]['prompt'], record['text'].split('</think>')[0])
            by_suite.setdefault(record['suite'], []).append(rows)
            labelled.append(dict(arm=arm.name, id=record['id'], suite=record['suite'], labels=rows))
    for suite, traces in sorted(by_suite.items()):
        print(suite, len(traces), dict(sorted(summarize(traces).items())))
    if args.output:
        args.output.write_text(json.dumps(labelled, ensure_ascii=False), encoding='utf-8')


if __name__ == '__main__':
    main()
