"""One scorer for every external arm, and the census of what the model computes in its thinking."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re

MINUS = chr(0x2212)
FORMS = dict(
    times=r'\\times', cdot=r'\\cdot', frac=r'\\frac', sqrt=r'\\sqrt', power=r'\^', sum=r'\\sum', binom=r'\\binom',
    factorial=r'\d!', modulo=r'\bmod\b|\\pmod|\\equiv', logarithm=r'\\log|\\ln', trig=r'\\sin|\\cos|\\tan', fraction=r'\d/\d',
    equation=r'[a-z]\s*=\s*-?\d', variable_term=r'\d[a-z]\b')


def final_number(text):
    """The bold number if the answer has one, else the last number; None when thinking never closed."""
    parts = text.rsplit('</think>', 1)
    if len(parts) < 2:
        return None
    answer = parts[1].replace(',', '').replace(MINUS, '-')
    bold = [b for b in re.findall(r'\*\*([^*]*)\*\*', answer) if re.search(r'\d', b)]
    numbers = re.findall(r'-?\d+(?:\.\d+)?', bold[-1] if bold else answer)
    if not numbers:
        return None
    # `**7 minutes**` keeps its first number, `**12 + 11 = 23**` its result.
    equation = bold and '=' in bold[-1]
    return float(numbers[-1] if equation or not bold else numbers[0])


def score(records):
    suites = {}
    for r in records:
        s = suites.setdefault(r['suite'], Counter())
        s.update(n=1, correct=final_number(r['text']) == r['answer'], tokens=r['tokens'], limit=r['finish'] == 'length',
                 calls=r['text'].count('(exact)'))
        s['seconds'] += r['seconds']
    return suites


def census(records):
    """Forms in the thinking only, per suite: what a trigger would have to meet."""
    out = {}
    for r in records:
        thinking = r['text'].split('</think>')[0]
        c = out.setdefault(r['suite'], Counter())
        c['cases'] += 1
        c['equals'] += thinking.count('=')
        for name, pattern in FORMS.items():
            c[name] += len(re.findall(pattern, thinking))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('arms', type=Path, nargs='+')
    parser.add_argument('--census', action='store_true')
    args = parser.parse_args()
    for arm in args.arms:
        records = json.loads((arm / 'records.json').read_text(encoding='utf-8'))
        print(arm.name, len(records))
        if args.census:
            for suite, c in census(records).items():
                print(' ', suite, dict(c.most_common()))
            continue
        for suite, s in score(records).items():
            print(f"  {suite}: {s['correct']}/{s['n']} tokens {s['tokens'] / s['n']:.1f} seconds {s['seconds'] / s['n']:.2f} limit {s['limit']} calls {s['calls']}")


if __name__ == '__main__':
    main()
