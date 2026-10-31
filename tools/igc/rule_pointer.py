"""Offline rule baseline for natural-language firing: word cues choose the operation, recent numbers the operands.

A sidecar has to beat this. The number the model actually wrote is the answer key, so a firing is right
only if the computed value equals it. Interval: exact binomial (Clopper-Pearson), never a single value.
"""
import argparse
import json
from pathlib import Path
import re
from fractions import Fraction
from tools.igc.auto_label import label_text, numbers, OPERATIONS

CUES = (
    (r'\b(total|altogether|combined|in all|sum|together|plus)\b', 'add'),
    (r'\b(left|remaining|remain|difference|fewer|less than|minus|after spending)\b', 'sub'),
    (r'\b(each|per|times|every|product|multipl\w*|twice|double|triple)\b', 'mul'),
    (r'\b(half|divid\w*|split|average|ratio|quotient|shared)\b', 'div'),
)
WINDOW = 80


def proposal(prefix):
    """Operation and the two most recent distinct numbers in the same clause, or nothing."""
    clause = prefix[-WINDOW:]
    clause = re.split(r'[.!?\n]\s', clause)[-1]
    found = [name for pattern, name in CUES if re.search(pattern, clause, re.IGNORECASE)]
    if len(set(found)) != 1:
        return None
    values = list(dict.fromkeys(v for *_, v in numbers(clause)))
    if len(values) < 2:
        return None
    return found[0], values[-2], values[-1]


def evaluate(records, prompts):
    """Every computed number written outside syntax is a slot; copied numbers are chances to misfire."""
    fired = right = slots = caught = 0
    for record in records:
        thinking = record['text'].split('</think>')[0]
        for row in label_text(prompts[record['id']], thinking):
            if row['label'] == 'computed' and row['level'] == 0:
                continue
            if row['label'] == 'computed':
                slots += 1
            guess = proposal(thinking[:row['start']])
            if guess is None:
                continue
            name, a, b = guess
            value = OPERATIONS[name](a, b)
            if name in ('sub', 'div') and value is not None and value <= 0:
                value = OPERATIONS[name](b, a)
            fired += 1
            good = value == Fraction(row['value'].replace(',', ''))
            right += good
            caught += good and row['label'] == 'computed'
    return dict(fired=fired, right=right, slots=slots, caught=caught)


def interval(hits, n, z=0.05):
    """Clopper-Pearson by bisection on the binomial tail, using only the standard library."""
    from math import comb

    def tail(p, low):
        return sum(comb(n, k) * p ** k * (1 - p) ** (n - k) for k in (range(hits, n + 1) if low else range(0, hits + 1)))

    def solve(target, low):
        a, b = 0.0, 1.0
        for _ in range(60):
            m = (a + b) / 2
            if (tail(m, low) < target) == low:
                a = m
            else:
                b = m
        return (a + b) / 2
    return (0.0 if hits == 0 else solve(z / 2, True), 1.0 if hits == n else solve(z / 2, False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('arms', type=Path, nargs='+')
    args = parser.parse_args()
    records = []
    prompts = {}
    for arm in args.arms:
        prompts.update({c['id']: c['prompt'] for c in json.loads((arm / 'cases.json').read_text(encoding='utf-8'))})
        records += json.loads((arm / 'records.json').read_text(encoding='utf-8'))
    for suite in sorted({r['suite'] for r in records}):
        c = evaluate([r for r in records if r['suite'] == suite], prompts)
        low, high = interval(c['right'], c['fired']) if c['fired'] else (0.0, 1.0)
        recall = c['caught'] / c['slots'] if c['slots'] else 0.0
        print(f"{suite}: fired {c['fired']} right {c['right']} precision 95% [{low:.3f}, {high:.3f}] recall {c['caught']}/{c['slots']} = {recall:.3f}")


if __name__ == '__main__':
    main()
