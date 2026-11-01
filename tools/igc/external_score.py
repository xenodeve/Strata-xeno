"""One scorer for every external arm, and the census of what the model computes in its thinking."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

MINUS = chr(0x2212)
FORMS = dict(
    times=r'\\times', cdot=r'\\cdot', frac=r'\\frac', sqrt=r'\\sqrt', power=r'\^', sum=r'\\sum', binom=r'\\binom',
    factorial=r'\d!', modulo=r'\bmod\b|\\pmod|\\equiv', logarithm=r'\\log|\\ln', trig=r'\\sin|\\cos|\\tan', fraction=r'\d/\d',
    equation=r'[a-z]\s*=\s*-?\d', variable_term=r'\d[a-z]\b')


def question_id(prefix, prompt):
    """From the question, not its place in a case file: `math-3` was a different question in four selected sets."""
    return f"{prefix}-{hashlib.sha256(prompt.encode('utf-8')).hexdigest()[:8]}"


def final_number(text):
    """The last boxed number, else a bold number on the closing line, else the last number; None when thinking never closed."""
    parts = text.rsplit('</think>', 1)
    if len(parts) < 2:
        return None
    answer = parts[1].replace('<|im_end|>', '').replace('{,}', '').replace(',', '').replace(MINUS, '-').strip()
    boxed = re.findall(r'\\boxed\{\s*(?:\\\$)?\s*(-?\d+(?:\.\d+)?)\s*\}', answer)
    if boxed:
        return float(boxed[-1])
    # A closing `If both boxes must be nonempty...` or `since 2^6 = 64` comes after the answer.
    lines = [line for line in answer.splitlines() if line.strip()]
    while len(lines) > 1 and (lines[-1].startswith('since') or (lines[-1].startswith('If ') and re.search(r'\*\*[^*]*\d', lines[-2]))):
        lines.pop()
    closing = re.sub(r'\s+since\b.*', '', lines[-1]) if lines else ''
    answer = '\n'.join(lines[:-1] + [closing])
    # `**160 minutes**, or **2 hours 40 minutes**`: what follows `or` is the same answer in another form.
    closing = re.sub(r'\s+or\s+\*\*.*', '', closing)
    bold = [b for b in re.findall(r'\*\*([^*]*)\*\*', closing) if re.search(r'\d', b)]
    numbers = re.findall(r'-?\d+(?:\.\d+)?', bold[-1] if bold else answer)
    if not numbers:
        return None
    if not bold and '=' not in closing and re.match(r'\d', closing):    # `5 cars drove through in the first 15 minutes.`
        return float(re.findall(r'-?\d+(?:\.\d+)?', closing)[0])
    # `**7 minutes**` keeps its first number, `**12 + 11 = 23**` its result.
    equation = bold and '=' in bold[-1]
    return float(numbers[-1] if equation or not bold else numbers[0])


def repeating(tokens, longest=4096, rounds=20):
    """Greedy text whose tail is one exact cycle, many rounds long, cannot leave it. Past 512 tokens eight rounds are enough, as in the
    engine: a longer cycle once ran to the end of the context, 31 minutes, because this stopped looking at 512."""
    for period in range(1, longest + 1):
        count = rounds if period <= 512 else 8
        if len(tokens) < period * count or tokens[-1] != tokens[-1 - period]:
            continue
        tail = tokens[-period * count:]
        if tail == tail[:period] * count:
            return True
    return False


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
